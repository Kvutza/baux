#![allow(clippy::too_many_arguments)]

use crate::py_parameter::{PyParameter, parameters};
use baux::experimental::{
    self, AcquisitionKind, AcquisitionParams, ComputeDevice, ForwardProgram, PackedModel,
    ProposalParams, ResidentBoState, ResidentRound, RoundConfig, SurrogateParams,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use baux::experimental::{
    KdaControlRequest, KdaForwardRequest, KdaMoeLayerRequest, KdaMoeMetalExecutor,
    KdaMoeMetalKdaVectors, KdaMoeMetalModel, KdaPackedLinear, KdaTensorLayout,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(error: String) -> PyErr {
    PyValueError::new_err(error)
}

/// An BAUX-native quantized model package.
///
/// Opening this object parses and validates the package in Rust. It does not
/// import JAX, NumPy, or model tensors into Python.
#[pyclass(name = "ModelPackage", unsendable)]
pub struct PyModelPackage {
    inner: PackedModel,
}

#[pymethods]
impl PyModelPackage {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PackedModel::open(path).map_err(err)?,
        })
    }

    /// Return the packed descriptor required by a native forward program.
    fn linear(
        &self,
        name: &str,
    ) -> PyResult<(
        usize,
        usize,
        usize,
        usize,
        usize,
        u8,
        usize,
        usize,
        u32,
        u32,
    )> {
        let linear = self.inner.linear(name).map_err(err)?;
        Ok((
            linear.byte_offset,
            linear.scale_offset,
            linear.bias_offset,
            linear.input_width,
            linear.output_width,
            linear.bits,
            linear.group_size,
            linear.element_offset,
            linear.perturb_whole,
            linear.perturb_threshold,
        ))
    }

    #[getter]
    fn weight_bytes(&self) -> usize {
        self.inner.packed().len()
    }

    #[getter]
    fn scale_count(&self) -> usize {
        self.inner.scales().len()
    }

    #[getter]
    fn bias_count(&self) -> usize {
        self.inner.biases().len()
    }

    #[pyo3(signature=(base_value,capacity,device="auto",scale=1.0,weight=1.0,radius=1.0))]
    fn resident_session(
        &self,
        base_value: f32,
        capacity: usize,
        device: &str,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> PyResult<PyResidentBoSession> {
        Ok(PyResidentBoSession {
            inner: ResidentBoState::new(
                self.inner.packed(),
                base_value,
                self.inner
                    .trial_leaves(scale, weight, radius)
                    .map_err(err)?,
                capacity,
                ComputeDevice::parse(device).map_err(err)?,
                ForwardProgram::kda().map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }
}

#[pyclass(name = "ResidentBoSession", unsendable)]
pub struct PyResidentBoSession {
    inner: ResidentBoState,
    pending: Option<ResidentRound>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
type NativeLinear = (
    String,
    usize,
    usize,
    usize,
    usize,
    usize,
    u8,
    usize,
    usize,
    u32,
    u32,
);

#[cfg(all(target_os = "macos", feature = "metal"))]
fn native_linear(raw: NativeLinear) -> (String, KdaPackedLinear) {
    let (name, byte, scale, bias, input, output, bits, group, element, whole, threshold) = raw;
    (
        name,
        KdaPackedLinear {
            byte_offset: byte,
            scale_offset: scale,
            bias_offset: bias,
            input_width: input,
            output_width: output,
            bits,
            group_size: group,
            element_offset: element,
            perturb_whole: whole,
            perturb_threshold: threshold,
        },
    )
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn named_linear(
    linears: &std::collections::BTreeMap<String, KdaPackedLinear>,
    name: &str,
) -> PyResult<KdaPackedLinear> {
    linears
        .get(name)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("native model has no linear {name:?}")))
}

/// Complete resident KDA-MoE greedy decoder for Apple Metal.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[pyclass(name = "NativeKdaModel", unsendable)]
pub struct PyNativeKdaModel {
    pub(crate) inner: KdaMoeMetalModel,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn req_arg<'py, T>(
    args: &Bound<'py, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
    idx: usize,
    name: &str,
) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py>,
    for<'a> PyErr: From<<T as FromPyObject<'a, 'py>>::Error>,
{
    if idx < args.len() {
        return Ok(args.get_item(idx)?.extract()?);
    }
    if let Some(kw) = kwargs
        && let Some(val) = kw.get_item(name)?
    {
        return Ok(val.extract()?);
    }
    Err(PyValueError::new_err(format!("missing argument {name:?}")))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn build_layer(
    linears: &std::collections::BTreeMap<String, KdaPackedLinear>,
    index: usize,
    heads: usize,
    head_width: usize,
    gate_rank: usize,
    num_experts: usize,
    top_k: usize,
    residual_scale: f32,
    rms_epsilon: f32,
    vectors: KdaMoeMetalKdaVectors<'_>,
    device: ComputeDevice,
    arena: &experimental::KdaMoeMetalArena,
) -> PyResult<KdaMoeMetalExecutor> {
    let prefix = format!("layers.{index}");
    let attention = format!("{prefix}.attention");
    let moe = format!("{prefix}.moe");
    let qkv = named_linear(linears, &format!("{attention}.qkv"))?;
    let control_projection = named_linear(linears, &format!("{attention}.control"))?;
    let output = named_linear(linears, &format!("{attention}.output"))?;
    let expert_gate = (0..num_experts)
        .map(|expert| named_linear(linears, &format!("{moe}.experts.{expert}.gate")))
        .collect::<PyResult<Vec<_>>>()?;
    let expert_up = (0..num_experts)
        .map(|expert| named_linear(linears, &format!("{moe}.experts.{expert}.up")))
        .collect::<PyResult<Vec<_>>>()?;
    let expert_down = (0..num_experts)
        .map(|expert| named_linear(linears, &format!("{moe}.experts.{expert}.down")))
        .collect::<PyResult<Vec<_>>>()?;
    let request = KdaMoeLayerRequest {
        kda: KdaForwardRequest {
            tensor: KdaTensorLayout::new(1, 1, heads, head_width, head_width).map_err(err)?,
            qkv,
            control: control_projection,
            output,
            seed: 0,
        },
        attention_norm: named_linear(linears, &format!("{prefix}.attention_norm"))?,
        moe_norm: named_linear(linears, &format!("{prefix}.moe_norm"))?,
        router: named_linear(linears, &format!("{moe}.router"))?,
        expert_gate,
        expert_up,
        expert_down,
        top_k,
        residual_scale,
        rms_epsilon,
    };
    let control = KdaControlRequest {
        qkv_conv: named_linear(linears, &format!("{attention}.qkv_conv"))?,
        control: control_projection,
        forget: named_linear(linears, &format!("{attention}.forget_b"))?,
        output_gate: named_linear(linears, &format!("{attention}.gate_b"))?,
        decay: named_linear(linears, &format!("{attention}.decay"))?,
        time_bias: named_linear(linears, &format!("{attention}.time_bias"))?,
        output_norm: named_linear(linears, &format!("{attention}.output_norm"))?,
        output,
        gate_rank,
    };
    KdaMoeMetalExecutor::new_arena(&request, control, vectors, device, arena).map_err(err)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[pymethods]
impl PyNativeKdaModel {
    #[new]
    #[pyo3(signature=(*args, **kwargs))]
    fn new(
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let packed: PyReadonlyArray1<'_, u8> = req_arg(args, kwargs, 0, "packed")?;
        let scales: PyReadonlyArray1<'_, f32> = req_arg(args, kwargs, 1, "scales")?;
        let biases: PyReadonlyArray1<'_, f32> = req_arg(args, kwargs, 2, "biases")?;
        let linears: Vec<NativeLinear> = req_arg(args, kwargs, 3, "linears")?;
        let num_layers: usize = req_arg(args, kwargs, 4, "num_layers")?;
        let _ = req_arg::<usize>(args, kwargs, 5, "hidden_size")?;
        let heads: usize = req_arg(args, kwargs, 6, "heads")?;
        let head_width: usize = req_arg(args, kwargs, 7, "head_width")?;
        let gate_rank: usize = req_arg(args, kwargs, 8, "gate_rank")?;
        let num_experts: usize = req_arg(args, kwargs, 9, "num_experts")?;
        let top_k: usize = req_arg(args, kwargs, 10, "top_k")?;
        let _ = req_arg::<usize>(args, kwargs, 11, "expert_width")?;
        let residual_scale: f32 = req_arg(args, kwargs, 12, "residual_scale")?;
        let rms_epsilon: f32 = req_arg(args, kwargs, 13, "rms_epsilon")?;
        let embedding_scale: f32 = req_arg(args, kwargs, 14, "embedding_scale")?;
        let device: String =
            req_arg(args, kwargs, 15, "device").unwrap_or_else(|_| "metal".to_owned());
        let linears = linears
            .into_iter()
            .map(native_linear)
            .collect::<std::collections::BTreeMap<_, _>>();
        let arena = experimental::KdaMoeMetalArena::new(experimental::KdaMoeMetalWeights {
            packed: packed
                .as_slice()
                .map_err(|_| PyValueError::new_err("packed model arena must be contiguous"))?,
            scales: scales
                .as_slice()
                .map_err(|_| PyValueError::new_err("scale arena must be contiguous"))?,
            biases: biases
                .as_slice()
                .map_err(|_| PyValueError::new_err("bias arena must be contiguous"))?,
        })
        .map_err(err)?;
        let device = ComputeDevice::parse(&device).map_err(err)?;
        let decay = vec![0.0_f32; heads];
        let time_bias = vec![0.0_f32; heads * head_width];
        let output_norm = vec![1.0_f32; head_width];
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            layers.push(build_layer(
                &linears,
                index,
                heads,
                head_width,
                gate_rank,
                num_experts,
                top_k,
                residual_scale,
                rms_epsilon,
                KdaMoeMetalKdaVectors {
                    decay: &decay,
                    time_bias: &time_bias,
                    output_norm: &output_norm,
                },
                device,
                &arena,
            )?);
        }
        let mut inner = KdaMoeMetalModel::new(layers).map_err(err)?;
        inner
            .attach_head(
                named_linear(&linears, "embedding")?,
                named_linear(&linears, "final_norm")?,
                embedding_scale,
            )
            .map_err(err)?;
        inner.prepare_candidate(0).map_err(err)?;
        Ok(Self { inner })
    }

    fn reset(&mut self) {
        self.inner.reset_state();
    }

    #[pyo3(signature=(token,seed=0))]
    fn decode_token(&mut self, token: u32, seed: u64) -> PyResult<u32> {
        self.inner.decode_token(token, seed).map_err(err)
    }

    fn logit_bits(&self) -> PyResult<Vec<u16>> {
        self.inner.logit_bits().map_err(err)
    }

    #[pyo3(signature=(prompt,max_new_tokens,seed=0))]
    fn generate(
        &mut self,
        prompt: Vec<u32>,
        max_new_tokens: usize,
        seed: u64,
    ) -> PyResult<Vec<u32>> {
        self.inner
            .generate(&prompt, max_new_tokens, seed)
            .map_err(err)
    }
}

#[pymethods]
impl PyResidentBoSession {
    #[new]
    #[pyo3(signature=(base,base_value,leaves,capacity,device="auto"))]
    fn new(
        base: Vec<u8>,
        base_value: f32,
        leaves: Vec<PyRef<'_, PyParameter>>,
        capacity: usize,
        device: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ResidentBoState::new(
                &base,
                base_value,
                parameters(leaves),
                capacity,
                ComputeDevice::parse(device).map_err(err)?,
                ForwardProgram::kda().map_err(err)?,
            )
            .map_err(err)?,
            pending: None,
        })
    }

    #[pyo3(signature=(seeds,length,neighbors,epistemic_scale=0.7,aleatoric_scale=0.05,y_scale=1.0,beta=1.0,acquisition="ucb",seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn ask(
        &mut self,
        seeds: Vec<u64>,
        length: f32,
        neighbors: usize,
        epistemic_scale: f32,
        aleatoric_scale: f32,
        y_scale: f32,
        beta: f32,
        acquisition: &str,
        seed: u64,
    ) -> PyResult<(usize, u64, f32, u32)> {
        let round = self
            .inner
            .ask(
                &seeds,
                ask_config(
                    length,
                    neighbors,
                    epistemic_scale,
                    aleatoric_scale,
                    y_scale,
                    beta,
                    acquisition,
                    seed,
                )?,
            )
            .map_err(err)?;
        let out = (
            round.trial.index,
            round.trial.seed,
            round.trial.score,
            round.program_version,
        );
        self.pending = Some(round);
        Ok(out)
    }

    fn tell(&mut self, reward: f32, accept: bool) -> PyResult<()> {
        let round = self
            .pending
            .take()
            .ok_or_else(|| PyValueError::new_err("there is no pending resident BO round"))?;
        self.inner.tell(round, reward, accept).map_err(err)
    }

    #[getter]
    fn rewards(&self) -> Vec<f32> {
        self.inner.rewards().to_vec()
    }
}

fn ask_config(
    length: f32,
    neighbors: usize,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    acquisition: &str,
    seed: u64,
) -> PyResult<RoundConfig> {
    Ok(RoundConfig {
        proposal: ProposalParams { radius: length },
        surrogate: SurrogateParams {
            neighbors,
            epistemic_scale,
            aleatoric_scale,
            output_scale: y_scale,
        },
        acquisition: AcquisitionParams {
            kind: AcquisitionKind::parse(acquisition).map_err(err)?,
            beta,
            seed,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{PyModelPackage, PyResidentBoSession};

    #[test]
    fn package_linked() {
        let _ = std::mem::size_of::<PyModelPackage>();
    }

    #[test]
    fn resident_linked() {
        let _ = std::mem::size_of::<PyResidentBoSession>();
    }
}
