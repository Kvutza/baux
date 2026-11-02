use super::VOCAB;
pub(super) use super::proposal::WeightBackup;
use super::proposal::{GaussianRandom, ProposalSpec};
use crate::capture::{CudaResult, cuda_error, read_prefix};
use baux::semantics::mhc4::Initialization as Mhc4Scales;
use baux::semantics::random::splitmix64 as splitmix;
use baux::semantics::tensor::key as tensor_key;
use baux_cuda_device::fbt_model;
use cuda_core::{CudaStream, DeviceBuffer};
use half::f16;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::collections::BTreeMap;

pub(super) struct Site {
    pub predictor: DeviceBuffer<f32>,
    pub bias: DeviceBuffer<f32>,
    pub control: DeviceBuffer<f32>,
    pub norm: DeviceBuffer<f32>,
}

pub(super) struct Layer {
    pub qkv: DeviceBuffer<f32>,
    pub output: DeviceBuffer<f32>,
    pub router: DeviceBuffer<f32>,
    pub gate: DeviceBuffer<f32>,
    pub down: DeviceBuffer<f32>,
    pub attention: Site,
    pub moe: Site,
}

pub(crate) struct DiffusionWeights {
    pub mask: DeviceBuffer<f32>,
    pub index: DeviceBuffer<f32>,
}

pub(crate) struct Mhc4Weights {
    pub mask: DeviceBuffer<f32>,
    pub positions: DeviceBuffer<f32>,
}

pub(crate) struct Weights {
    pub format: &'static str,
    pub diffusion: Option<DiffusionWeights>,
    pub mhc4: Option<Mhc4Weights>,
    pub embedding: DeviceBuffer<f32>,
    pub readout: DeviceBuffer<f32>,
    pub final_norm: DeviceBuffer<f32>,
    pub layers: Vec<Layer>,
    pub(super) proposal: Vec<ProposalSpec>,
    pub(super) gaussian_random: Option<GaussianRandom>,
}

/// Storage-only fixture through the real full-row materialization adapter.
/// Three differently sized tensors exercise partial CTAs, status-buffer reuse,
/// invalid tail coordinates, unchanged rows, rollback and subsequent recovery.
/// This is not a model forward pass or an optimizer learning experiment.
pub fn proposal_check() -> CudaResult<()> {
    super::proposal::check()
}

fn spherical(
    random: &mut StdRng,
    matrices: usize,
    input: usize,
    output: usize,
    column_norm: f32,
) -> Vec<f32> {
    let elements = matrices
        .checked_mul(input)
        .and_then(|v| v.checked_mul(output))
        .expect("fixed model tensor dimensions must fit usize");
    let mut values = vec![0.0_f32; elements];
    baux::semantics::mhc4::columns(
        &mut values,
        input,
        output,
        column_norm,
        || StandardNormal.sample(random),
        |v| f16::from_f32(v).to_f32(),
    )
    .expect("fixed mHC-4 Gaussian column initialization");
    values
}

fn mhc_bias(layers: usize, coefficients: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; layers * coefficients];
    let input_bias = f16::from_f32(-3.0_f32.ln()).to_f32();
    for layer in 0..layers {
        values[layer * coefficients..layer * coefficients + 4].fill(input_bias);
    }
    values
}

fn mhc4_specs(
    weights: &Weights,
    stream: &CudaStream,
    policy: Mhc4Scales,
    shape: baux::semantics::mhc4::ScaleShape,
) -> CudaResult<Vec<ProposalSpec>> {
    // Order matches buffers_mut. The tied embedding is a physical transpose,
    // not another logical parameter family.
    let mut inventory = vec![
        (
            "embedding_readout",
            weights.embedding.len(),
            512 * VOCAB,
            0,
            512,
        ),
        (
            "embedding_readout",
            weights.readout.len(),
            512 * VOCAB,
            0,
            0,
        ),
        ("final_norm", weights.final_norm.len(), 512, 0, 0),
    ];
    for (index, layer) in weights.layers.iter().enumerate() {
        for (name, elements, block) in [
            ("qkv", layer.qkv.len(), 512 * 640),
            ("attention_output", layer.output.len(), 512 * 512),
            ("router", layer.router.len(), 512 * 64),
            ("expert_gate_up", layer.gate.len(), 512 * 512),
            ("expert_down", layer.down.len(), 256 * 512),
            (
                "mhc_attention_predictor",
                layer.attention.predictor.len(),
                2048 * 24,
            ),
            ("mhc_attention_bias", layer.attention.bias.len(), 24),
            ("mhc_attention_control", layer.attention.control.len(), 4),
            ("attention_norm", layer.attention.norm.len(), 512),
            ("mhc_moe_predictor", layer.moe.predictor.len(), 2048 * 24),
            ("mhc_moe_bias", layer.moe.bias.len(), 24),
            ("mhc_moe_control", layer.moe.control.len(), 4),
            ("ffn_norm", layer.moe.norm.len(), 512),
        ] {
            inventory.push((name, elements, block, index * elements, 0));
        }
    }
    inventory
        .into_iter()
        .map(|(name, elements, block_len, start, transpose)| {
            if block_len == 0 || elements % block_len != 0 {
                return Err("invalid procedural mHC-4 proposal geometry".into());
            }
            let host_scales =
                vec![policy.proposal_scale(name)? * shape.multiplier(name); elements / block_len];
            Ok(ProposalSpec {
                elements,
                block_len: block_len as u32,
                scales: DeviceBuffer::from_host(stream, &host_scales).map_err(cuda_error)?,
                host_scales,
                tensor_id: tensor_key(name),
                logical_start: start as u64,
                transpose_columns: transpose as u32,
            })
        })
        .collect()
}

fn init_layers(seed: u64, policy: Mhc4Scales, stream: &CudaStream) -> CudaResult<Vec<Layer>> {
    use baux::semantics::shape as dimensions;
    const LAYERS: usize = dimensions::MHC4_LAYERS as usize;
    const WIDTH: usize = dimensions::MHC4_WIDTH as usize;
    const ROUTED: usize = dimensions::ROUTED_COUNT as usize;
    const EXPERTS: usize = dimensions::EXPERT_COUNT as usize;
    const GATE: usize = dimensions::GATE_WIDTH as usize;
    const DOWN: usize = dimensions::EXPERT_WIDTH as usize;
    const MHC_INPUT: usize = dimensions::MHC_INPUT as usize;
    const MHC_COEFFICIENTS: usize = dimensions::MHC_COEFS as usize;
    let random = |domain| StdRng::seed_from_u64(splitmix(seed ^ domain));
    let mut router_rng = random(0x6a09_e667);
    let mut qkv_rng = random(0xbb67_ae85);
    let mut output_rng = random(0x3c6e_f372);
    let mut gate_rng = random(0xa54f_f53a);
    let mut down_rng = random(0x510e_527f);
    let mut attention_rng = random(0x428a_2f98);
    let mut moe_rng = random(0x7137_4491);
    let upload = |values: &[f32]| DeviceBuffer::from_host(stream, values).map_err(cuda_error);
    let site = |rng: &mut StdRng| -> CudaResult<Site> {
        Ok(Site {
            predictor: upload(&spherical(rng, 1, MHC_INPUT, MHC_COEFFICIENTS, 1.0))?,
            bias: upload(&mhc_bias(1, MHC_COEFFICIENTS))?,
            control: upload(&[
                f16::from_f32(policy.predictor_control).to_f32(),
                f16::from_f32(policy.predictor_control).to_f32(),
                f16::from_f32(policy.predictor_control).to_f32(),
                f16::from_f32(policy.transport_lambda).to_f32(),
            ])?,
            norm: upload(&vec![1.0_f32; WIDTH])?,
        })
    };
    let mut layers = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        layers.push(Layer {
            qkv: upload(&spherical(&mut qkv_rng, 1, WIDTH, 640, 1.0))?,
            output: upload(&spherical(
                &mut output_rng,
                1,
                WIDTH,
                WIDTH,
                policy.residual_gain,
            ))?,
            router: upload(&spherical(
                &mut router_rng,
                1,
                WIDTH,
                ROUTED,
                policy.router_column_norm,
            ))?,
            gate: upload(&spherical(&mut gate_rng, EXPERTS, WIDTH, GATE, 1.0))?,
            down: upload(&spherical(
                &mut down_rng,
                EXPERTS,
                DOWN,
                WIDTH,
                policy.down_norm(),
            ))?,
            attention: site(&mut attention_rng)?,
            moe: site(&mut moe_rng)?,
        });
    }
    Ok(layers)
}

impl Weights {
    /// Hash every logical parameter in packed-family order, independently of
    /// this backend's per-layer allocation order and tied physical transpose.
    pub(super) fn logical_hashes(
        &self,
        stream: &CudaStream,
    ) -> CudaResult<Vec<baux::artifact::forward::WeightHash>> {
        use baux::artifact::forward::WordHash;
        use baux::semantics::mhc4::probe::initialized::FAMILIES;
        let mut hashes = FAMILIES
            .iter()
            .map(|family| (tensor_key(family.name), WordHash::new(family.name)))
            .collect::<BTreeMap<_, _>>();
        let buffers = self.buffers();
        if buffers.len() != self.proposal.len() {
            return Err("CUDA digest omitted a physical parameter buffer".into());
        }
        for (index, buffer) in buffers.into_iter().enumerate() {
            let spec = self
                .proposal
                .get(index)
                .ok_or("CUDA digest inventory is incomplete")?;
            if spec.transpose_columns != 0 {
                continue;
            }
            let hash = hashes
                .get_mut(&spec.tensor_id)
                .ok_or("CUDA digest found an unknown parameter family")?;
            if spec.logical_start != hash.elements() || spec.elements != buffer.len() {
                return Err(
                    "CUDA logical parameter order or size differs from its descriptor".into(),
                );
            }
            let words = read_prefix(buffer, stream, buffer.len())?
                .into_iter()
                .map(|v| f16::from_f32(v).to_bits())
                .collect::<Vec<_>>();
            hash.update(&words)?;
        }
        let mut output = hashes
            .into_values()
            .map(WordHash::finish)
            .collect::<Vec<_>>();
        output.sort_by(|a, b| a.name.cmp(&b.name));
        if output.iter().map(|value| value.elements).sum::<u64>()
            != baux::semantics::shape::MHC4_PARAMS
        {
            return Err("CUDA logical parameter count differs from the mHC-4 contract".into());
        }
        // Verify the actual tied alias independently of the packing helper.
        let readout = read_prefix(&self.readout, stream, self.readout.len())?;
        let embedding = read_prefix(&self.embedding, stream, self.embedding.len())?;
        for token in 0..VOCAB {
            for dimension in 0..512 {
                if embedding[token * 512 + dimension].to_bits()
                    != readout[dimension * VOCAB + token].to_bits()
                {
                    return Err(format!(
                        "CUDA tied embedding differs at token {token}, dimension {dimension}"
                    ));
                }
            }
        }
        Ok(output)
    }

    pub fn initialize_mhc4(seed: u64, stream: &CudaStream) -> CudaResult<Self> {
        const WIDTH: usize = baux::semantics::shape::MHC4_WIDTH as usize;
        let policy = Mhc4Scales::new();
        let random = |domain| StdRng::seed_from_u64(splitmix(seed ^ domain));
        let upload = |values: &[f32]| DeviceBuffer::from_host(stream, values).map_err(cuda_error);
        let layers = init_layers(seed, policy, stream)?;
        let mut readout_rng = random(0x9b05_688c);
        let readout_values = spherical(&mut readout_rng, 1, WIDTH, VOCAB, 1.0);
        let embedding = DeviceBuffer::from_host(
            stream,
            &crate::prefill::pack_embedding(&readout_values, WIDTH, VOCAB),
        )
        .map_err(cuda_error)?;
        let readout = upload(&readout_values)?;
        let mut mask_rng = random(0x4d41_534b);
        let mut position_rng = random(0x494e_4458);
        let mhc4 = Some(Mhc4Weights {
            mask: upload(&spherical(&mut mask_rng, 1, WIDTH, 1, 1.0))?,
            positions: upload(&spherical(&mut position_rng, 1, 64, WIDTH, 1.0))?,
        });
        let mut weights = Self {
            format: "mhc4",
            diffusion: None,
            mhc4,
            embedding,
            readout,
            final_norm: upload(&vec![1.0_f32; WIDTH])?,
            layers,
            proposal: Vec::new(),
            gaussian_random: Some(GaussianRandom::new(stream)?),
        };
        weights.proposal = mhc4_specs(
            &weights,
            stream,
            policy,
            baux::semantics::mhc4::ScaleShape::Scalar,
        )?;
        Ok(weights)
    }

    /// Resolve scales before creating the initial observation or weight backup.
    pub(super) fn set_shape(
        &mut self,
        shape: baux::semantics::mhc4::ScaleShape,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        self.proposal = mhc4_specs(self, stream, Mhc4Scales::new(), shape)?;
        Ok(())
    }

    fn buffers(&self) -> Vec<&DeviceBuffer<f32>> {
        let mut list = vec![&self.embedding, &self.readout, &self.final_norm];
        for layer in &self.layers {
            list.extend([
                &layer.qkv,
                &layer.output,
                &layer.router,
                &layer.gate,
                &layer.down,
            ]);
            for site in [&layer.attention, &layer.moe] {
                list.extend([&site.predictor, &site.bias, &site.control, &site.norm]);
            }
        }
        if let Some(diffusion) = &self.diffusion {
            list.extend([&diffusion.mask, &diffusion.index]);
        }
        list
    }

    fn buffers_mut(&mut self) -> Vec<&mut DeviceBuffer<f32>> {
        let mut list = vec![&mut self.embedding, &mut self.readout, &mut self.final_norm];
        for layer in &mut self.layers {
            list.extend([
                &mut layer.qkv,
                &mut layer.output,
                &mut layer.router,
                &mut layer.gate,
                &mut layer.down,
            ]);
            for site in [&mut layer.attention, &mut layer.moe] {
                list.extend([
                    &mut site.predictor,
                    &mut site.bias,
                    &mut site.control,
                    &mut site.norm,
                ]);
            }
        }
        if let Some(diffusion) = &mut self.diffusion {
            list.extend([&mut diffusion.mask, &mut diffusion.index]);
        }
        list
    }

    pub(super) fn backup(&self, stream: &CudaStream) -> CudaResult<WeightBackup> {
        WeightBackup::new(&self.buffers(), self.proposal.len(), stream)
    }

    pub(super) fn restore(
        &mut self,
        backup: &WeightBackup,
        stream: &CudaStream,
    ) -> CudaResult<f32> {
        backup.restore(&mut self.buffers_mut(), stream)
    }

    pub(super) fn commit(&self, backup: &mut WeightBackup, stream: &CudaStream) -> CudaResult<f32> {
        backup.commit(&self.buffers(), stream)
    }

    pub(super) fn materialize_gaussian(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
        seed: u64,
    ) -> CudaResult<f32> {
        let mut random = self
            .gaussian_random
            .take()
            .ok_or("CUDA Gaussian random state is missing")?;
        let specs = std::mem::take(&mut self.proposal);
        let result = random.materialize(
            &specs,
            backup,
            &mut self.buffers_mut(),
            module,
            stream,
            radius,
            seed,
        );
        // A rejected proposal must not strand the host inventory/RNG or leave
        // invalid candidate weights available to a later evaluation.
        self.proposal = specs;
        self.gaussian_random = Some(random);
        result
    }

    pub(super) fn materialize_basis0(
        &mut self,
        backup: &WeightBackup,
        module: &fbt_model::LoadedModule,
        stream: &CudaStream,
        radius: f32,
    ) -> CudaResult<f32> {
        let specs = std::mem::take(&mut self.proposal);
        let result = backup.step_basis0(&specs, &mut self.buffers_mut(), module, stream, radius);
        self.proposal = specs;
        result
    }

    pub(super) fn physical_elements(&self) -> u64 {
        self.buffers().into_iter().map(|b| b.len() as u64).sum()
    }
}
