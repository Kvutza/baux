#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaTensorLayout {
    pub batch: usize,
    pub sequence_length: usize,
    pub heads: usize,
    pub key_width: usize,
    pub value_width: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaPackedLinear {
    pub byte_offset: usize,
    pub scale_offset: usize,
    pub bias_offset: usize,
    pub input_width: usize,
    pub output_width: usize,
    pub bits: u8,
    pub group_size: usize,
    pub element_offset: usize,
    pub perturb_whole: u32,
    pub perturb_threshold: u32,
}

impl KdaPackedLinear {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_width == 0 || self.output_width == 0 || self.group_size == 0 {
            return Err("packed KDA linear dimensions must be positive".to_string());
        }
        if self.bits != 4 && self.bits != 8 {
            return Err(format!(
                "packed KDA linear bits must be 4 or 8, got {}",
                self.bits
            ));
        }
        Ok(())
    }

    pub fn packed_bytes(&self) -> usize {
        let bytes_per_row = (self.input_width * self.bits as usize).div_ceil(8);
        bytes_per_row * self.output_width
    }

    pub fn row_groups(&self) -> usize {
        self.input_width.div_ceil(self.group_size)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaForwardRequest {
    pub tensor: KdaTensorLayout,
    pub qkv: KdaPackedLinear,
    pub control: KdaPackedLinear,
    pub output: KdaPackedLinear,
    pub seed: u64,
}

/// Remaining packed projections and vectors needed to turn a KDA recurrence
/// into a complete attention update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdaControlRequest {
    pub qkv_conv: KdaPackedLinear,
    pub control: KdaPackedLinear,
    pub forget: KdaPackedLinear,
    pub output_gate: KdaPackedLinear,
    pub decay: KdaPackedLinear,
    pub time_bias: KdaPackedLinear,
    pub output_norm: KdaPackedLinear,
    pub output: KdaPackedLinear,
    pub gate_rank: usize,
}

impl KdaControlRequest {
    pub fn validate(&self, tensor: KdaTensorLayout, hidden_width: usize) -> Result<(), String> {
        for linear in [
            self.qkv_conv,
            self.control,
            self.forget,
            self.output_gate,
            self.decay,
            self.time_bias,
            self.output_norm,
            self.output,
        ] {
            linear.validate()?;
        }
        let width = tensor.heads * tensor.value_width;
        if self.qkv_conv.input_width != 4
            || self.qkv_conv.output_width != 3 * width
            || self.qkv_conv.bits != 8
            || self.qkv_conv.group_size != 4
        {
            return Err("KDA QKV convolution must contain four INT8 taps per channel".to_string());
        }
        if self.control.input_width != hidden_width
            || self.control.output_width != 2 * self.gate_rank + tensor.heads
        {
            return Err("KDA control projection shape is invalid".to_string());
        }
        if self.forget.input_width != self.gate_rank || self.forget.output_width != width {
            return Err("KDA forget projection shape is invalid".to_string());
        }
        if self.output_gate.input_width != self.gate_rank || self.output_gate.output_width != width
        {
            return Err("KDA output-gate projection shape is invalid".to_string());
        }
        if self.output.input_width != width || self.output.output_width != hidden_width {
            return Err("KDA output projection shape is invalid".to_string());
        }
        if self.decay.input_width != tensor.heads
            || self.decay.output_width != 1
            || self.time_bias.input_width != width
            || self.time_bias.output_width != 1
            || self.output_norm.input_width != tensor.value_width
            || self.output_norm.output_width != 1
        {
            return Err("KDA recurrent vector shapes are invalid".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaDispatch {
    ProjectQkv { threads: usize },
    SplitQkv { threads: usize },
    Recur { threads: usize },
}

/// Packed descriptors for one complete KDA-MoE decoder layer.
///
/// This is deliberately a data-only ABI: the Rust Metal device owns buffers
/// and command encoding, while Python only supplies the model layout once.
#[derive(Debug, Clone, PartialEq)]
pub struct KdaMoeLayerRequest {
    pub kda: KdaForwardRequest,
    pub attention_norm: KdaPackedLinear,
    pub moe_norm: KdaPackedLinear,
    pub router: KdaPackedLinear,
    pub expert_gate: Vec<KdaPackedLinear>,
    pub expert_up: Vec<KdaPackedLinear>,
    pub expert_down: Vec<KdaPackedLinear>,
    pub top_k: usize,
    pub residual_scale: f32,
    pub rms_epsilon: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaMoeDispatch {
    AttentionNorm { threads: usize },
    Kda(KdaDispatch),
    AttentionResidual { threads: usize },
    MoeNorm { threads: usize },
    RouteExperts { threads: usize },
    ExpertGateUp { threads: usize },
    ExpertDown { threads: usize },
    MoeResidual { threads: usize },
}

impl KdaMoeLayerRequest {
    fn check_layers(&self, hidden: usize, experts: usize) -> Result<(), String> {
        if self.attention_norm.input_width != hidden || self.attention_norm.output_width != 1 {
            return Err("attention norm must be a hidden-width vector".to_string());
        }
        if self.moe_norm.input_width != hidden || self.moe_norm.output_width != 1 {
            return Err("MoE norm must be a hidden-width vector".to_string());
        }
        if experts == 0 || self.expert_up.len() != experts || self.expert_down.len() != experts {
            return Err(
                "KDA-MoE expert descriptor lists must be non-empty and equal in length".to_string(),
            );
        }
        if self.top_k == 0 || self.top_k > experts || self.top_k > 8 {
            return Err("KDA-MoE top_k must be in 1..=min(experts, 8)".to_string());
        }
        if !self.residual_scale.is_finite() || !self.rms_epsilon.is_finite() {
            return Err("KDA-MoE numeric parameters must be finite".to_string());
        }
        if self.rms_epsilon <= 0.0 {
            return Err("KDA-MoE RMS epsilon must be positive".to_string());
        }
        if self.router.input_width != hidden || self.router.output_width != experts {
            return Err("router dimensions must be hidden_width by num_experts".to_string());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.kda.validate()?;
        for linear in [self.attention_norm, self.moe_norm, self.router] {
            linear.validate()?;
        }
        let hidden = self.kda.qkv.input_width;
        let experts = self.expert_gate.len();
        self.check_layers(hidden, experts)?;
        for ((gate, up), down) in self
            .expert_gate
            .iter()
            .zip(&self.expert_up)
            .zip(&self.expert_down)
        {
            gate.validate()?;
            up.validate()?;
            down.validate()?;
            if gate.input_width != hidden
                || up.input_width != hidden
                || gate.output_width != up.output_width
            {
                return Err(
                    "expert gate and up projections must share hidden input and expert width"
                        .to_string(),
                );
            }
            if down.input_width != gate.output_width || down.output_width != hidden {
                return Err(
                    "expert down projection must map expert width to hidden width".to_string(),
                );
            }
        }
        Ok(())
    }

    pub fn hidden_elements(&self) -> usize {
        self.kda.tensor.batch * self.kda.tensor.sequence_length * self.kda.qkv.input_width
    }

    pub fn expert_elems(&self) -> usize {
        self.kda.tensor.batch
            * self.kda.tensor.sequence_length
            * self.top_k
            * self.expert_gate[0].output_width
    }

    pub fn encode(&self) -> Result<Vec<KdaMoeDispatch>, String> {
        self.validate()?;
        let kda = KdaEncoder::new(self.kda)?;
        let tokens = self.kda.tensor.batch * self.kda.tensor.sequence_length;
        let hidden = self.hidden_elements();
        Ok(vec![
            KdaMoeDispatch::AttentionNorm { threads: tokens },
            KdaMoeDispatch::Kda(kda.encode()[0]),
            KdaMoeDispatch::Kda(kda.encode()[1]),
            KdaMoeDispatch::Kda(kda.encode()[2]),
            KdaMoeDispatch::AttentionResidual { threads: hidden },
            KdaMoeDispatch::MoeNorm { threads: tokens },
            KdaMoeDispatch::RouteExperts { threads: tokens },
            KdaMoeDispatch::ExpertGateUp {
                threads: self.expert_elems(),
            },
            KdaMoeDispatch::ExpertDown { threads: hidden },
            KdaMoeDispatch::MoeResidual { threads: hidden },
        ])
    }
}

pub struct KdaEncoder {
    request: KdaForwardRequest,
}

impl KdaEncoder {
    pub fn new(request: KdaForwardRequest) -> Result<Self, String> {
        request.validate()?;
        Ok(Self { request })
    }

    pub fn request(&self) -> KdaForwardRequest {
        self.request
    }

    pub fn encode(&self) -> [KdaDispatch; 3] {
        [
            KdaDispatch::ProjectQkv {
                threads: self.request.projection_threads(),
            },
            KdaDispatch::SplitQkv {
                threads: self.request.tensor.qkv_elements(),
            },
            KdaDispatch::Recur {
                threads: self.request.recurrence_threads(),
            },
        ]
    }
}

impl KdaForwardRequest {
    pub fn qkv_width(&self) -> usize {
        3 * self.tensor.heads * self.tensor.key_width
    }

    pub fn projection_threads(&self) -> usize {
        self.tensor.batch * self.tensor.sequence_length * self.qkv_width()
    }

    pub fn recurrence_threads(&self) -> usize {
        self.tensor.batch * self.tensor.heads * self.tensor.value_width
    }

    pub fn validate(&self) -> Result<(), String> {
        for linear in [self.qkv, self.control, self.output] {
            linear.validate()?;
        }
        let expected_qkv = self.qkv_width();
        if self.qkv.input_width != self.output.input_width {
            return Err("KDA projection input widths do not match".to_string());
        }
        if self.qkv.output_width != expected_qkv {
            return Err(format!(
                "KDA QKV output width must be {expected_qkv}, got {}",
                self.qkv.output_width
            ));
        }
        Ok(())
    }
}

impl KdaTensorLayout {
    pub fn new(
        batch: usize,
        sequence_length: usize,
        heads: usize,
        key_width: usize,
        value_width: usize,
    ) -> Result<Self, String> {
        if batch == 0 || sequence_length == 0 || heads == 0 || key_width == 0 || value_width == 0 {
            return Err("KDA tensor dimensions must be positive".to_string());
        }
        Ok(Self {
            batch,
            sequence_length,
            heads,
            key_width,
            value_width,
        })
    }

    pub fn qkv_elements(&self) -> usize {
        self.batch * self.sequence_length * self.heads * self.key_width
    }

    pub fn value_elements(&self) -> usize {
        self.batch * self.sequence_length * self.heads * self.value_width
    }

    pub fn state_elements(&self) -> usize {
        self.batch * self.heads * self.key_width * self.value_width
    }
}
