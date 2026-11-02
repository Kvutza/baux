//! Exact-shape grouped-MoE feasibility gate for the proposed subsecond scorer.

use crate::apple_gpu::Runtime;
use crate::fbt_pisa1::Pisa1;
pub(super) use metal::{
    Buffer, BufferRef, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState,
    MTLCommandBufferStatus,
};
use std::time::Instant;

mod resources;
use resources::*;

mod weight_stats;
use weight_stats::{checkpoint_layout, load_checkpoint, save_checkpoint};

mod initialization;
use initialization::Mhc4Initialization;

mod hnet2;
use hnet2::*;

mod buffers;
use buffers::*;

mod dispatch;
use dispatch::*;

mod envelope;
use envelope::*;

mod candidate;
use candidate::*;

#[path = "heldout/causal.rs"]
mod causal;
#[cfg(test)]
#[path = "heldout/forward_probe.rs"]
mod forward_probe;
#[path = "heldout/long.rs"]
mod long;
#[path = "heldout/matrix.rs"]
mod matrix;

mod objective;
use objective::*;

#[path = "heldout/optimization.rs"]
mod optimization;
use optimization::*;
#[path = "heldout/round.rs"]
mod round;
use round::*;

mod architecture;
#[path = "heldout/benchmarks.rs"]
mod benchmarks;
use benchmarks::*;

mod validation;

use architecture::ResidualArchitecture;
use validation::*;

#[path = "heldout/experiment.rs"]
pub mod experiment;

#[path = "updates.rs"]
mod updates;
use updates::UpdateLog;

#[cfg(test)]
#[path = "ablation.rs"]
mod ablation;

#[cfg(test)]
#[path = "loss_tests.rs"]
mod loss_tests;

#[cfg(test)]
mod mhc_reference;

#[cfg(test)]
#[path = "mhc/adjoint.rs"]
mod mhc_adjoint;

#[path = "routing.rs"]
pub(crate) mod routing;
pub(crate) use routing::grad as routing_grad;

#[cfg(test)]
#[path = "routing/adjoint.rs"]
mod routing_adjoint;

#[path = "mhc/grad.rs"]
mod mhc_grad;

#[path = "mhc/train_grad.rs"]
mod train_grad;

#[path = "train.rs"]
mod train;

#[path = "generation/alignment.rs"]
mod gen_alignment;

#[path = "block.rs"]
mod block_decode;
#[path = "contrast.rs"]
mod code_contrast;
#[path = "code_environment.rs"]
mod code_environment;
#[path = "decode.rs"]
pub(crate) mod decode;
#[path = "block/diffusion.rs"]
mod diffusion;
#[path = "generation/checkpoint.rs"]
mod gen_checkpoint;
#[path = "generation/finish.rs"]
mod gen_finish;
#[path = "generation/journal.rs"]
mod gen_journal;
#[path = "generation/metrics.rs"]
mod gen_metrics;
#[path = "generation/oracle.rs"]
mod gen_oracle;
#[path = "generation/pool.rs"]
mod gen_pool;
#[path = "generation/protocol.rs"]
mod gen_protocol;
#[path = "generation/record.rs"]
mod gen_record;
#[path = "generation/reward.rs"]
mod gen_reward;
#[path = "generation/rounds.rs"]
mod gen_rounds;
#[path = "generation/setup.rs"]
mod gen_setup;
#[path = "generation.rs"]
pub mod generation;
#[path = "reward.rs"]
mod generation_reward;
#[path = "heldout.rs"]
mod heldout;
#[path = "generation/muon.rs"]
mod muon_train;
#[path = "scorer/readout.rs"]
mod readout;
#[path = "reconstruction.rs"]
mod reconstruction;
#[path = "scorer.rs"]
mod scorer;
#[path = "scorer/replay.rs"]
mod scorer_replay;
#[path = "scorer/trace.rs"]
mod scorer_trace;
#[path = "block/stats.rs"]
mod stats;
#[path = "block/target.rs"]
mod target;
#[path = "block/window.rs"]
mod window;

const ROWS: u32 = 8192;
const BATCH: u32 = 2;
const CONTEXT: u32 = ROWS / BATCH;
const WIDTH: u32 = baux::semantics::mhc4::WIDTH as u32;
const EXPERTS: u32 = 32;
const EXPERT_ROWS: u32 = ROWS / EXPERTS;
const EXPERT_WIDTH: u32 = 1024;
const GATE_UP: u32 = 2 * EXPERT_WIDTH;
const QUERY_HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = WIDTH / QUERY_HEADS;
const QKV_WIDTH: u32 = baux::semantics::shape::QKV_WIDTH as u32;
const ROPE_PAIRS: u32 = HEAD_DIM / 2;
const ROPE_HEADS: u32 = QUERY_HEADS + KV_HEADS;
const ROPE_BASE: f64 = 10_000.0;
const MAX_CONTEXT: u32 = 8192;
const PISA_BLOCK: u32 = 64;
const PISA_LEAVES: u32 = CONTEXT / PISA_BLOCK;
const PISA_NODES: u32 = 2 * PISA_LEAVES - 1;
const PISA_SELECTED: u32 = 8;
const MODEL_LAYERS: u32 = baux::semantics::shape::MHC4_LAYERS as u32;
const FEEDBACK_PASSES: u32 = 2;
const VOCAB: u32 = baux::semantics::mhc4::VOCABULARY as u32;
const LAYER_PARAMETERS: usize = (WIDTH * routing::ROUTED_EXPERTS) as usize
    + (WIDTH * QKV_WIDTH) as usize
    + (WIDTH * WIDTH) as usize
    + routing::FFN_PARAMETERS
    + (2 * WIDTH) as usize;
const GLOBAL_PARAMETERS: usize =
    (WIDTH * VOCAB) as usize + (2 * WIDTH * WIDTH) as usize + WIDTH as usize;
const FULL_PARAMETERS: usize = MODEL_LAYERS as usize * LAYER_PARAMETERS + GLOBAL_PARAMETERS;
const _: () = assert!(FULL_PARAMETERS == 265_655_808);
const MHC_STREAMS: u32 = baux::semantics::shape::MHC4_SHAPE.streams as u32;
const MHC_COEFFICIENTS: u32 = baux::semantics::shape::MHC_COEFS as u32;
const MHC_INPUT: u32 = baux::semantics::shape::MHC_INPUT as u32;
const MHC_SITEPARAMS: usize = (MHC_INPUT * MHC_COEFFICIENTS + MHC_COEFFICIENTS + 4) as usize;
const MHC_PARAMETERS: usize = MODEL_LAYERS as usize * 2 * MHC_SITEPARAMS;
const MHC_FULLPARAMS: usize = FULL_PARAMETERS - 2 * (WIDTH * WIDTH) as usize + MHC_PARAMETERS;
const _: () = assert!(MHC_SITEPARAMS == 49_180);
const _: () = assert!(MHC_FULLPARAMS == baux::semantics::shape::MHC4_PARAMS as usize);
// Exact seeded replay needs one recyclable model-sized scratch row.  The second
// historical row was only required by the former approximate-distance path;
// its allocation is now the immutable replay-window origin instead.
const HISTORY_CAPACITY: usize = 1;

pub(crate) use crate::tensor::MoeShape;

#[derive(Debug, Clone)]
pub struct GroupedMoeProbe {
    updates: UpdateLog,
    pub parameters: usize,
    pub routing_gpu_seconds: f64,
    pub activation_gpu_seconds: f64,
    pub residual_gpu_seconds: f64,
    pub materialize_gate_up_gpu_seconds: f64,
    pub materialize_down_gpu_seconds: f64,
    pub projections_gpu_seconds: f64,
    pub projections_wall_seconds: f64,
    pub pisa1_pyramid_gpu_seconds: f64,
    pub pisa1_selection_gpu_seconds: f64,
    pub pisa1_attention_gpu_seconds: f64,
    pub pisa1_layer_gpu_seconds: f64,
    pub pisa1_layer_wall_seconds: f64,
    pub layer_gpu_seconds: f64,
    pub layer_wall_seconds: f64,
    pub projected_ffn_seconds: f64,
    pub projected_ffn_and_projections_seconds: f64,
    pub projected_measured_model_seconds: f64,
    pub sustained_model_gpu_seconds: f64,
    pub sustained_model_wall_seconds: f64,
    pub sustained_model_min_wall_seconds: f64,
    pub sustained_model_max_wall_seconds: f64,
    pub sustained_materialization_gpu_seconds: f64,
    pub sustained_projections_gpu_seconds: f64,
    pub sustained_pisa1_gpu_seconds: f64,
    pub sustained_ffn_gpu_seconds: f64,
    pub sustained_mps_projections_gpu_seconds: f64,
    pub sustained_mps_projections_wall_seconds: f64,
    pub sustained_mps_ffn_gpu_seconds: f64,
    pub sustained_mps_ffn_wall_seconds: f64,
    pub tail_gpu_seconds: f64,
    pub tail_wall_seconds: f64,
    pub complete_envelope_gpu_seconds: f64,
    pub complete_envelope_wall_seconds: f64,
    pub complete_envelope_min_wall_seconds: f64,
    pub complete_envelope_max_wall_seconds: f64,
    pub optimizer_median_wall_seconds: f64,
    pub optimizer_min_wall_seconds: f64,
    pub optimizer_max_wall_seconds: f64,
    pub actual_bo_median_wall_seconds: f64,
    pub actual_bo_min_wall_seconds: f64,
    pub actual_bo_max_wall_seconds: f64,
    pub actual_bo_median_gpu_seconds: f64,
    pub actual_bo_accepted: u32,
    pub target_seconds: f64,
    pub objective_flops: u64,
    pub tail_flops: u64,
    pub complete_objective_flops: u64,
    pub projection_objective_flops: u64,
    pub pisa1_objective_flops: u64,
    pub effective_tflops: f64,
    pub gate_up_max_abs_error: f64,
    pub down_max_abs_error: f64,
    pub qkv_max_abs_error: f64,
    pub output_projection_max_abs_error: f64,
    pub pisa1_max_abs_error: f64,
    pub tail_max_abs_error: f64,
    pub meets_target: bool,
}

impl GroupedMoeProbe {
    pub fn write_updates(&self, path: &std::path::Path) -> Result<(), String> {
        self.updates.write(path)
    }
}

struct Pipelines {
    fine_grained: routing::FineGrainedMoePipelines,
    fine_grained_grad: std::cell::OnceCell<Result<routing_grad::FineGrainedMoeGrad, String>>,
    mhc_grad: std::cell::OnceCell<Result<mhc_grad::Kernels, String>>,
    train_grad: std::cell::OnceCell<Result<train_grad::Kernels, String>>,
    quantize_gate_up_int8: ComputePipelineState,
    gate: ComputePipelineState,
    group: ComputePipelineState,
    swiglu: ComputePipelineState,
    ungroup: ComputePipelineState,
    feedback_fuse: ComputePipelineState,
    cross_entropy: ComputePipelineState,
    readout_loss_reduce: ComputePipelineState,
    readout_proposal_reduce: ComputePipelineState,
    sequence_loss: ComputePipelineState,
    embed: ComputePipelineState,
    denoise_embed: ComputePipelineState,
    denoise_reduce: ComputePipelineState,
    rms: ComputePipelineState,
    residual: ComputePipelineState,
    residual_rms: ComputePipelineState,
    feedback_fuse_rms: ComputePipelineState,
    mhc_replicate: ComputePipelineState,
    mhc_predict: ComputePipelineState,
    mhc_predict_rows: ComputePipelineState,
    mhc_mix_rms: ComputePipelineState,
    mhc_update: ComputePipelineState,
    combine_mhc: ComputePipelineState,
    mhc_mean_rms: ComputePipelineState,
    rope: ComputePipelineState,
}

struct TensorOpsPipelines {
    gate_up: ComputePipelineState,
    gate_activation: ComputePipelineState,
    down: ComputePipelineState,
    materialize_gate_up: ComputePipelineState,
    materialize_down: ComputePipelineState,
    qkv: ComputePipelineState,
    output_projection: ComputePipelineState,
    qkv_wide: ComputePipelineState,
    qkv_rope: ComputePipelineState,
    output_projection_wide: ComputePipelineState,
    readout: ComputePipelineState,
    readout_loss_tiles: ComputePipelineState,
    readout_proposal_tiles: ComputePipelineState,
    denoise_tiles: ComputePipelineState,
}

pub struct ActualBoResult {
    pub parameters: usize,
    pub loop_seconds: f64,
    pub median_wall_seconds: f64,
    pub min_wall_seconds: f64,
    pub max_wall_seconds: f64,
    pub median_gpu_seconds: f64,
    pub accepted: u32,
    pub learning_seconds: f64,
    pub validation: Vec<baux::artifact::json::Value>,
    optimizer_seconds: Vec<f64>,
    optimization_records: Vec<baux::artifact::json::Value>,
    updates: UpdateLog,
}

impl ActualBoResult {
    pub fn write_validation(&self, path: &std::path::Path) -> Result<(), String> {
        baux::artifact::json::pretty_writer(
            std::fs::File::create(path).map_err(|error| error.to_string())?,
            &baux::artifact::json::json!({
                "schema":"baux.pretrain_validation.v1", "learning_seconds":self.learning_seconds,
                "used_for_acceptance":false,
                "measurements":self.validation,
            }),
        )
        .map_err(|error| error.to_string())
    }

    pub fn write_updates(&self, path: &std::path::Path) -> Result<(), String> {
        self.updates.write(path)
    }

    pub fn write_optimization(&self, path: &std::path::Path) -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| error.to_string())?;
        for record in &self.optimization_records {
            baux::artifact::json::write_line(&mut file, record)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Gate,
    Group,
    Swiglu,
    Ungroup,
}

impl Stage {
    const ALL: [(Self, &'static str); 4] = [
        (Self::Gate, "router_gate"),
        (Self::Group, "group"),
        (Self::Swiglu, "swiglu"),
        (Self::Ungroup, "ungroup_residual"),
    ];
}

fn family_group(family: &str) -> usize {
    match family {
        "expert_gate_up" | "expert_down" => 0,
        "qkv" | "attention_output" | "attention_norm" | "ffn_norm" | "final_norm" => 1,
        "hnet"
        | "index_query"
        | "router"
        | "feedback_state"
        | "feedback_gate"
        | "mhc_attention_predictor"
        | "mhc_moe_predictor"
        | "mhc_attention_bias"
        | "mhc_attention_control"
        | "mhc_moe_bias"
        | "mhc_moe_control" => 2,
        "embedding_readout" | "mask_embed" => 3,
        _ => unreachable!("unmapped model tensor family"),
    }
}

#[derive(Clone, Copy)]
struct BoControl<'a> {
    length: crate::trust_region::TRLengthConfig,
    enn: crate::config::ResidentEnnConfig,
    proposal_seed: u64,
    acquisition_seed: u64,
    perturbation: crate::ProposalKernel,
    shape: crate::config::TrustRegionShape,
    paired_objective: bool,
    objective_reference: crate::config::ObjectiveReference,
    kernel_trial: Option<&'a crate::config::KernelTrial>,
    validation: Option<&'a crate::pretrain_data::PretrainDataset>,
    validation_interval: u32,
    random_selection: bool,
}

impl BoControl<'_> {
    fn diagnostic(perturbation: crate::ProposalKernel) -> Result<Self, String> {
        let acquisition_seed = 0xbb67_ae85_84ca_a73b;
        Ok(Self {
            length: crate::trust_region::TRLengthConfig::new(0.01, 0.0001, 0.1),
            enn: crate::config::RunConfig {
                acquisition: Some(crate::config::AcquisitionConfig::Thompson),
                ..Default::default()
            }
            .resident_enn(acquisition_seed)?,
            proposal_seed: 0x6a09_e667_f3bc_c909,
            acquisition_seed,
            perturbation,
            shape: crate::config::TrustRegionShape::TensorFamilyStatic,
            paired_objective: false,
            objective_reference: crate::config::ObjectiveReference::MovingIncumbent,
            kernel_trial: None,
            validation: None,
            validation_interval: 1,
            random_selection: false,
        })
    }
}

pub(super) fn complete_committed(command: &CommandBufferRef) -> Result<f64, String> {
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "candidate objective command failed: {:?}",
            command.status()
        ));
    }
    gpu_seconds(command).ok_or("Metal did not report candidate objective GPU timing".into())
}
