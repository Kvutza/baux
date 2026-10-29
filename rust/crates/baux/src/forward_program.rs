//! Experimental contract for resident, fused model evaluation.
//!
//! Academic References & Citations:
//! - Epistemic Neural Networks (ENN): Osband et al. (2021) "Epistemic Neural
//!   Networks", NeurIPS.
//! - Linear Attention & Recurrence: Gu & Dao (2023) "Mamba", De et al. (2024)
//!   "Griffin".
//! - Sparsely Gated MoE: Shazeer et al. (2017) "Outrageously Large Neural
//!   Networks".
//! - Hardware-Resident BO: Bafna et al. (2026) "Hardware-Resident Bayesian
//!   Optimization in BAUX".

use crate::trials::{Parameter, RoundConfig, Trial, TrialEngine};
use crate::weights::ComputeDevice;

pub use baux::semantics::graph::*;
pub use baux::semantics::model::*;
pub use baux::semantics::recurrence::*;
pub use baux::semantics::schedule::*;
pub use baux::semantics::tensor::*;
#[path = "forward_program/contract.rs"]
mod contract;
#[path = "forward_program/kda.rs"]
pub(crate) mod kda;
pub use baux::semantics::mhc4;
pub use contract::*;
pub use kda::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardOp {
    EmbeddingLookup,
    PerturbPackedWeights,
    Dequantize,
    Matmul,
    KdaStateUpdate,
    RouteExperts,
    Activation,
    Normalize,
    Residual,
    ReduceObjective,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardProgram {
    ops: Vec<ForwardOp>,
    version: u32,
}

impl ForwardProgram {
    pub const VERSION: u32 = 1;

    pub fn new(ops: impl Into<Vec<ForwardOp>>) -> Result<Self, String> {
        let ops = ops.into();
        if !ops.contains(&ForwardOp::PerturbPackedWeights) {
            return Err("forward program must perturb packed weights".to_string());
        }
        if !ops.contains(&ForwardOp::Matmul) {
            return Err("forward program must contain a matrix multiply".to_string());
        }
        if !ops.contains(&ForwardOp::ReduceObjective) {
            return Err("forward program must reduce an objective".to_string());
        }
        Ok(Self {
            ops,
            version: Self::VERSION,
        })
    }

    pub fn kda() -> Result<Self, String> {
        Self::new([
            ForwardOp::EmbeddingLookup,
            ForwardOp::PerturbPackedWeights,
            ForwardOp::Dequantize,
            ForwardOp::Matmul,
            ForwardOp::KdaStateUpdate,
            ForwardOp::Activation,
            ForwardOp::Residual,
            ForwardOp::ReduceObjective,
        ])
    }

    pub fn ops(&self) -> &[ForwardOp] {
        &self.ops
    }

    pub const fn version(&self) -> u32 {
        self.version
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidentRound {
    pub trial: Trial,
    pub program_version: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkAxis {
    Token,
    Output,
    Input,
    Head,
    Expert,
    State,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkTile {
    pub axis: WorkAxis,
    pub items: usize,
}

impl WorkTile {
    pub fn new(axis: WorkAxis, items: usize) -> Result<Self, String> {
        if items == 0 {
            return Err("kernel tile size must be positive".to_string());
        }
        Ok(Self { axis, items })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkGrid {
    pub items: usize,
    pub threads_per_group: usize,
}

impl WorkGrid {
    pub fn new(items: usize, threads_per_group: usize) -> Result<Self, String> {
        if items == 0 || threads_per_group == 0 {
            return Err("kernel work grid dimensions must be positive".to_string());
        }
        Ok(Self {
            items,
            threads_per_group,
        })
    }

    pub fn groups(&self) -> usize {
        self.items.div_ceil(self.threads_per_group)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub kernel: &'static str,
    pub grid: WorkGrid,
    pub tiles: Vec<WorkTile>,
    pub scratch_bytes: usize,
}

impl KernelPlan {
    pub fn new(
        kernel: &'static str,
        items: usize,
        threads_per_group: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            kernel,
            grid: WorkGrid::new(items, threads_per_group)?,
            tiles: Vec::new(),
            scratch_bytes: 0,
        })
    }

    pub fn with_tile(mut self, axis: WorkAxis, items: usize) -> Result<Self, String> {
        self.tiles.push(WorkTile::new(axis, items)?);
        Ok(self)
    }

    pub fn scratch_bytes(mut self, bytes: usize) -> Self {
        self.scratch_bytes = bytes;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedAffinePlan {
    pub token_tile: usize,
    pub output_tile: usize,
    pub input_tile: usize,
    pub threads_per_group: usize,
}

impl PackedAffinePlan {
    pub fn apple_simd() -> Self {
        Self {
            token_tile: 1,
            output_tile: 16,
            input_tile: 64,
            threads_per_group: 256,
        }
    }

    pub fn validate(&self, linear: KdaPackedLinear) -> Result<(), String> {
        linear.validate()?;
        if self.token_tile == 0
            || self.output_tile == 0
            || self.input_tile == 0
            || self.threads_per_group == 0
        {
            return Err("packed affine schedule dimensions must be positive".to_string());
        }
        if !linear.input_width.is_multiple_of(self.input_tile) {
            return Err("packed affine input width must be divisible by input tile".to_string());
        }
        Ok(())
    }

    pub fn kernel_plan(
        &self,
        kernel: &'static str,
        tokens: usize,
        linear: KdaPackedLinear,
    ) -> Result<KernelPlan, String> {
        self.validate(linear)?;
        let items = tokens
            .checked_mul(linear.output_width)
            .ok_or("packed affine dispatch size overflow")?;
        KernelPlan::new(kernel, items, self.threads_per_group)?
            .with_tile(WorkAxis::Token, self.token_tile)?
            .with_tile(WorkAxis::Output, self.output_tile)?
            .with_tile(WorkAxis::Input, self.input_tile)
    }
}

impl KdaDispatch {
    pub fn kernel_plan(&self, affine: PackedAffinePlan) -> Result<KernelPlan, String> {
        match *self {
            KdaDispatch::ProjectQkv { threads } => {
                KernelPlan::new("kda_project_packed_16k", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Output, affine.output_tile)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
            KdaDispatch::SplitQkv { threads } => {
                KernelPlan::new("kda_split_qkv_16k", threads, 256)?
                    .with_tile(WorkAxis::Head, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaDispatch::Recur { threads } => KernelPlan::new("kda_recurrence_16k", threads, 256)?
                .with_tile(WorkAxis::State, 1)?
                .with_tile(WorkAxis::Token, 16_384)
                .map(|plan| plan.scratch_bytes(0)),
        }
    }
}

impl KdaMoeDispatch {
    pub fn kernel_plan(&self, affine: PackedAffinePlan) -> Result<KernelPlan, String> {
        match *self {
            KdaMoeDispatch::AttentionNorm { threads } => {
                KernelPlan::new("decoder_rms_norm", threads, 256)?
                    .with_tile(WorkAxis::Token, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::Kda(dispatch) => dispatch.kernel_plan(affine),
            KdaMoeDispatch::AttentionResidual { threads }
            | KdaMoeDispatch::MoeResidual { threads } => {
                KernelPlan::new("decoder_residual", threads, 256)?.with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::MoeNorm { threads } => {
                KernelPlan::new("decoder_rms_norm", threads, 256)?
                    .with_tile(WorkAxis::Token, 1)?
                    .with_tile(WorkAxis::Input, 256)
            }
            KdaMoeDispatch::RouteExperts { threads } => {
                KernelPlan::new("moe_router_topk", threads, 1)?.with_tile(WorkAxis::Token, 1)
            }
            KdaMoeDispatch::ExpertGateUp { threads } => {
                KernelPlan::new("moe_gate_up", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Expert, 1)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
            KdaMoeDispatch::ExpertDown { threads } => {
                KernelPlan::new("moe_down", threads, affine.threads_per_group)?
                    .with_tile(WorkAxis::Token, affine.token_tile)?
                    .with_tile(WorkAxis::Output, affine.output_tile)?
                    .with_tile(WorkAxis::Input, affine.input_tile)
            }
        }
    }
}

impl KdaMoeLayerRequest {
    pub fn kernel_plans(&self, affine: PackedAffinePlan) -> Result<Vec<KernelPlan>, String> {
        self.validate()?;
        affine.validate(self.kda.qkv)?;
        affine.validate(self.router)?;
        for linear in self
            .expert_gate
            .iter()
            .chain(&self.expert_up)
            .chain(&self.expert_down)
        {
            affine.validate(*linear)?;
        }
        self.encode()?
            .into_iter()
            .map(|dispatch| dispatch.kernel_plan(affine))
            .collect()
    }
}

pub trait ForwardEvaluator {
    fn evaluate(&mut self, round: &ResidentRound, program: &ForwardProgram) -> Result<f32, String>;
}

/// Rust-owned BO session for a resident forward evaluator.
///
/// `TrialEngine` owns the device model slots and history. The forward device
/// binds the selected slot directly, so candidate weights are materialized once
/// and are not regenerated or perturbed a second time during evaluation.
pub struct ResidentBoState {
    search: TrialEngine,
    program: ForwardProgram,
    pending: Option<ResidentRound>,
    rewards: Vec<f32>,
}

impl ResidentBoState {
    pub fn new(
        base: &[u8],
        base_value: f32,
        leaves: Vec<Parameter>,
        history_capacity: usize,
        device: ComputeDevice,
        program: ForwardProgram,
    ) -> Result<Self, String> {
        Ok(Self {
            search: TrialEngine::new(base, base_value, leaves, history_capacity, device)?,
            program,
            pending: None,
            rewards: Vec::with_capacity(history_capacity),
        })
    }

    pub fn ask(&mut self, seeds: &[u64], config: RoundConfig) -> Result<ResidentRound, String> {
        if self.pending.is_some() {
            return Err("tell must finish the resident forward round before ask".to_string());
        }
        let trial = self.search.ask(seeds, config)?;
        let round = ResidentRound {
            trial,
            program_version: self.program.version(),
        };
        self.pending = Some(round.clone());
        Ok(round)
    }

    pub fn tell(&mut self, round: ResidentRound, reward: f32, accept: bool) -> Result<(), String> {
        let pending = self
            .pending
            .take()
            .ok_or("no resident forward round is pending")?;
        if pending != round {
            self.pending = Some(pending);
            return Err("resident forward round does not match pending trial".to_string());
        }
        self.search.tell(round.trial, reward, accept)?;
        self.rewards.push(reward);
        Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn metal_row(
        &self,
        round: &ResidentRound,
    ) -> Result<(metal::Buffer, usize, usize), String> {
        let pending = self
            .pending
            .as_ref()
            .ok_or("no resident forward round is pending")?;
        if pending != round {
            return Err("resident forward round does not match pending trial".to_string());
        }
        let (rows, offset) = self.search.metal_row(round.trial)?;
        Ok((rows, offset, self.search.row_bytes()))
    }

    /// Borrow the pending packed row for an OpenCL evaluator.
    ///
    /// The returned binding keeps the OpenCL context, queue, and allocation
    /// alive through the evaluator handoff. The caller can bind the buffer and
    /// use the byte offset without copying the row.
    #[cfg(feature = "opencl")]
    #[allow(dead_code)]
    pub(crate) fn bind_opencl(
        &self,
        round: &ResidentRound,
    ) -> Result<crate::trials::OpenClResidentRow<'_>, String> {
        let pending = self
            .pending
            .as_ref()
            .ok_or("no resident forward round is pending")?;
        if pending != round {
            return Err("resident forward round does not match pending trial".to_string());
        }
        self.search.opencl_row(round.trial)
    }

    pub fn ask_evaluate<E: ForwardEvaluator>(
        &mut self,
        seeds: &[u64],
        config: RoundConfig,
        evaluator: &mut E,
        accept: bool,
    ) -> Result<(ResidentRound, f32), String> {
        let round = self.ask(seeds, config)?;
        let reward = match evaluator.evaluate(&round, &self.program) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        self.tell(round.clone(), reward, accept)?;
        Ok((round, reward))
    }

    pub fn program(&self) -> &ForwardProgram {
        &self.program
    }

    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ForwardEvaluator, ForwardOp, ForwardProgram, KdaDispatch, KdaEncoder, KdaForwardRequest,
        KdaMoeDispatch, KdaMoeLayerRequest, KdaPackedLinear, KdaTensorLayout, PackedAffinePlan,
        ResidentBoState, ResidentRound, WorkAxis,
    };
    use crate::trials::{Parameter, RoundConfig};
    use crate::weights::ComputeDevice;
    #[cfg(feature = "opencl")]
    use opencl3::{
        kernel::{ExecuteKernel, Kernel},
        memory::{Buffer, CL_MEM_WRITE_ONLY, ClMem},
        program::Program,
        types::CL_BLOCKING,
    };
    #[cfg(feature = "opencl")]
    use std::ptr;

    #[cfg(feature = "opencl")]
    const OPENCL_SUM: &str = r#"
__kernel void sum_row(__global const uchar* rows, uint offset, uint row_bytes, __global uint* out) {
    uint sum = 0;
    for (uint i = 0; i < row_bytes; ++i) {
        sum += rows[offset + i];
    }
    out[0] = sum;
}
"#;

    fn leaves() -> Vec<Parameter> {
        vec![Parameter::new(0, 1, 8, 1.0, 1.0, 1.0).unwrap()]
    }

    #[cfg(feature = "opencl")]
    fn opencl_unavailable(error: &str) -> bool {
        error.contains("no OpenCL GPU or CPU device")
            || error.contains("CL_PLATFORM_NOT_FOUND_KHR")
            || error.contains("failed to enumerate OpenCL GPU devices")
            || error.contains("failed to enumerate OpenCL")
    }

    #[test]
    fn kda_program() {
        let program = ForwardProgram::kda().unwrap();
        assert_eq!(program.version(), ForwardProgram::VERSION);
        assert!(program.ops().contains(&ForwardOp::KdaStateUpdate));
    }

    #[test]
    fn fbtprogram() {
        let program = super::ModelProgram::fbt_pisa();
        program.validate().unwrap();
        assert!(
            program
                .nodes
                .iter()
                .any(|node| node.op == super::SemanticOp::PisaAttention)
        );
        assert!(
            program
                .tensors
                .iter()
                .any(|tensor| tensor.storage == super::TensorStorage::State)
        );
        assert!(
            program
                .weight_arenas
                .iter()
                .any(|arena| arena.overlay == "candidate")
        );
        // KernelPlan remains an independent execution schedule.
        assert_eq!(program.version, super::ModelProgram::VERSION);
        let core = program.recurrent_core.unwrap();
        assert_eq!(core.first_layer, 1);
        assert_eq!(core.layer_count, 2);
        assert_eq!(
            program
                .layer_visits(core.training_visits)
                .unwrap()
                .iter()
                .map(|visit| (visit.layer, visit.visit))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 0), (4, 0)]
        );
    }

    #[test]
    fn kda_layout() {
        let layout = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        assert_eq!(layout.qkv_elements(), 16_777_216);
        assert_eq!(layout.value_elements(), 16_777_216);
        assert_eq!(layout.state_elements(), 131_072);
    }

    #[test]
    fn kda_request() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let linear = KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width: 1024,
            output_width: 1024,
            bits: 8,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let request = KdaForwardRequest {
            tensor,
            qkv: KdaPackedLinear {
                output_width: 3 * 8 * 128,
                ..linear
            },
            control: linear,
            output: linear,
            seed: 7,
        };
        request.validate().unwrap();
        assert_eq!(request.qkv.packed_bytes(), 3 * 8 * 128 * 1024);
        let dispatches = KdaEncoder::new(request).unwrap().encode();
        assert_eq!(
            dispatches[0],
            KdaDispatch::ProjectQkv {
                threads: 50_331_648
            }
        );
        assert_eq!(
            dispatches[1],
            KdaDispatch::SplitQkv {
                threads: 16_777_216
            }
        );
        assert_eq!(dispatches[2], KdaDispatch::Recur { threads: 1_024 });
    }

    #[test]
    fn kda_schedule() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let hidden = 1_024;
        let expert = 576;
        let linear = |input_width, output_width| KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width,
            output_width,
            bits: 4,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let kda = KdaForwardRequest {
            tensor,
            qkv: linear(hidden, hidden * 3),
            control: linear(hidden, 264),
            output: linear(hidden, hidden),
            seed: 7,
        };
        let layer = KdaMoeLayerRequest {
            kda,
            attention_norm: linear(hidden, 1),
            moe_norm: linear(hidden, 1),
            router: linear(hidden, 32),
            expert_gate: vec![linear(hidden, expert); 32],
            expert_up: vec![linear(hidden, expert); 32],
            expert_down: vec![linear(expert, hidden); 32],
            top_k: 4,
            residual_scale: 0.22,
            rms_epsilon: 1.0e-6,
        };
        let dispatches = layer.encode().unwrap();
        assert_eq!(dispatches.len(), 10);
        assert!(matches!(
            dispatches[6],
            KdaMoeDispatch::RouteExperts { threads: 16_384 }
        ));
        assert!(matches!(dispatches[7], KdaMoeDispatch::ExpertGateUp { .. }));
    }

    #[test]
    fn kda_kernels() {
        let tensor = KdaTensorLayout::new(1, 16_384, 8, 128, 128).unwrap();
        let hidden = 1_024;
        let expert = 576;
        let linear = |input_width, output_width| KdaPackedLinear {
            byte_offset: 0,
            scale_offset: 0,
            bias_offset: 0,
            input_width,
            output_width,
            bits: 4,
            group_size: 64,
            element_offset: 0,
            perturb_whole: 0,
            perturb_threshold: 0,
        };
        let layer = KdaMoeLayerRequest {
            kda: KdaForwardRequest {
                tensor,
                qkv: linear(hidden, hidden * 3),
                control: linear(hidden, 264),
                output: linear(hidden, hidden),
                seed: 7,
            },
            attention_norm: linear(hidden, 1),
            moe_norm: linear(hidden, 1),
            router: linear(hidden, 32),
            expert_gate: vec![linear(hidden, expert); 32],
            expert_up: vec![linear(hidden, expert); 32],
            expert_down: vec![linear(expert, hidden); 32],
            top_k: 4,
            residual_scale: 0.22,
            rms_epsilon: 1.0e-6,
        };
        let plans = layer.kernel_plans(PackedAffinePlan::apple_simd()).unwrap();
        assert_eq!(plans.len(), 10);
        assert_eq!(plans[1].kernel, "kda_project_packed_16k");
        assert_eq!(plans[1].grid.threads_per_group, 256);
        assert!(
            plans[1]
                .tiles
                .iter()
                .any(|tile| tile.axis == WorkAxis::Input && tile.items == 64)
        );
        assert_eq!(plans[3].kernel, "kda_recurrence_16k");
        assert!(
            plans[3]
                .tiles
                .iter()
                .any(|tile| tile.axis == WorkAxis::Token && tile.items == 16_384)
        );
    }

    #[test]
    fn resident_contract() {
        let mut state = ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::Cpu,
            ForwardProgram::kda().unwrap(),
        )
        .unwrap();
        let round = state
            .ask(&[7], {
                let mut config = RoundConfig::default();
                config.surrogate.neighbors = 1;
                config
            })
            .unwrap();
        state.tell(round, 1.25, true).unwrap();
        assert_eq!(state.rewards(), &[1.25]);
    }

    struct ConstantEvaluator;

    impl ForwardEvaluator for ConstantEvaluator {
        fn evaluate(
            &mut self,
            _round: &ResidentRound,
            _program: &ForwardProgram,
        ) -> Result<f32, String> {
            Ok(2.5)
        }
    }

    #[test]
    fn resident_round() {
        let mut state = ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::Cpu,
            ForwardProgram::kda().unwrap(),
        )
        .unwrap();
        let mut evaluator = ConstantEvaluator;
        let (_, reward) = state
            .ask_evaluate(
                &[11],
                {
                    let mut config = RoundConfig::default();
                    config.surrogate.neighbors = 1;
                    config
                },
                &mut evaluator,
                true,
            )
            .unwrap();
        assert_eq!(reward, 2.5);
        assert_eq!(state.rewards(), &[2.5]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_binding() {
        let mut state = match ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::OpenCl,
            ForwardProgram::kda().unwrap(),
        ) {
            Ok(state) => state,
            Err(error) if opencl_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let round = state
            .ask(&[11], {
                let mut config = RoundConfig::default();
                config.surrogate.neighbors = 1;
                config
            })
            .unwrap();
        let row = state.bind_opencl(&round).unwrap();
        assert_eq!(row.row_bytes(), state.search.row_bytes());
        assert_eq!(row.end(), row.offset() + row.row_bytes());
        assert!(row.end() <= row.buffer().size().unwrap() as usize);
        drop(row);
        state.tell(round, 1.0, true).unwrap();
        assert_eq!(state.rewards(), &[1.0]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_evaluator() {
        let mut state = match ResidentBoState::new(
            &[0],
            0.0,
            leaves(),
            4,
            ComputeDevice::OpenCl,
            ForwardProgram::kda().unwrap(),
        ) {
            Ok(state) => state,
            Err(error) if opencl_unavailable(&error) => return,
            Err(error) => panic!("{error}"),
        };
        let round = state
            .ask(&[11], {
                let mut config = RoundConfig::default();
                config.surrogate.neighbors = 1;
                config
            })
            .unwrap();
        let row = state.bind_opencl(&round).unwrap();
        assert_eq!(row.row_bytes(), state.search.row_bytes());
        assert_eq!(row.end(), row.offset() + row.row_bytes());
        assert!(row.end() <= row.buffer().size().unwrap() as usize);

        let program = match Program::create_and_build_from_source(row.context(), OPENCL_SUM, "") {
            Ok(program) => program,
            Err(error) => panic!("failed to build OpenCL row-sum kernel: {error}"),
        };
        let kernel = Kernel::create(&program, "sum_row")
            .unwrap_or_else(|error| panic!("missing OpenCL row-sum kernel: {error}"));
        let out = unsafe {
            Buffer::<u32>::create(row.context(), CL_MEM_WRITE_ONLY, 1, ptr::null_mut())
                .unwrap_or_else(|error| panic!("failed to allocate OpenCL output buffer: {error}"))
        };
        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(row.buffer())
                .set_arg(&(row.offset() as u32))
                .set_arg(&(row.row_bytes() as u32))
                .set_arg(&out)
                .set_global_work_size(1)
                .set_local_work_size(1)
                .enqueue_nd_range(row.queue())
                .unwrap_or_else(|error| panic!("failed to launch OpenCL row-sum kernel: {error}"));
        }
        let mut sum = [0u32];
        unsafe {
            row.queue()
                .enqueue_read_buffer(&out, CL_BLOCKING, 0, &mut sum, &[])
                .unwrap_or_else(|error| panic!("failed to read OpenCL row-sum output: {error}"));
        }
        let expected: u32 = state
            .search
            .row(round.trial)
            .unwrap()
            .iter()
            .map(|&byte| u32::from(byte))
            .sum();
        assert_eq!(sum[0], expected);
        drop(row);
        state.tell(round, sum[0] as f32, true).unwrap();
        assert_eq!(state.rewards(), &[sum[0] as f32]);
    }
}

/// Backend-independent corruption and denoising execution semantics.
pub mod diffusion;
