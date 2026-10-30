//! Execution schedules remain distinct while referring to the same operations.
use super::graph::{SemanticOp, TargetCapability};
use super::model::{ModelProgram, ProgramKind};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Metal,
    CudaOxide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compiler {
    Metal,
    CudaOxide,
    Tvm,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledKernel {
    pub nodes: Vec<&'static str>,
    pub kernel: &'static str,
    pub tile: [usize; 3],
    pub threads: [usize; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionSchedule {
    pub target: ExecutionTarget,
    pub compiler: Compiler,
    pub capabilities: Vec<TargetCapability>,
    pub kernels: Vec<ScheduledKernel>,
}

impl ExecutionSchedule {
    pub fn existing_metal() -> Self {
        Self {
            target: ExecutionTarget::Metal,
            compiler: Compiler::Metal,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["embed"], "fbt_moe_embed", [1, 512, 1], [256, 1, 1]),
                kernel(
                    &["attention-norm", "moe-norm"],
                    "fbt_mhc_predict_rows",
                    [1, 24, 2048],
                    [256, 1, 1],
                ),
                kernel(
                    &["attention-norm", "moe-norm"],
                    "fbt_mhc_mix_rms",
                    [1, 512, 4],
                    [128, 1, 1],
                ),
                kernel(&["qkv"], "qkv_rope", [128, 128, 512], [256, 1, 1]),
                kernel(
                    &["pisa-summaries"],
                    "fbt_pisa1_leaf_means",
                    [64, 64, 1],
                    [64, 1, 1],
                ),
                kernel(
                    &["pisa-select", "pisa-attention"],
                    "fbt_pisa1_select_attention",
                    [1, 512, 64],
                    [128, 1, 1],
                ),
                kernel(
                    &["attention-project"],
                    "fbt_model_tensorops_output_projection_wide",
                    [128, 128, 512],
                    [256, 1, 1],
                ),
                kernel(
                    &["attention-residual", "moe-residual"],
                    "fbt_mhc_update_rows",
                    [1, 512, 4],
                    [256, 1, 1],
                ),
                kernel(&["route"], "fbt_moe_select_top3", [1, 128, 1], [128, 1, 1]),
                kernel(
                    &["pack-routes"],
                    "fbt_moe_route_pack",
                    [64, 512, 1],
                    [128, 1, 1],
                ),
                kernel(
                    &["gate-up"],
                    "fbt_moe_routed_gate_fused_columns",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(
                    &["down"],
                    "fbt_moe_routed_down_fused_columns",
                    [128, 64, 216],
                    [128, 1, 1],
                ),
                kernel(&["combine-routes"], "combine_mhc", [1, 512, 3], [128, 1, 1]),
                kernel(
                    &["feedback"],
                    "fbt_moe_feedback_fuse",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["readout", "score-generation"],
                    "fbt_readout_loss_tiles",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(
                    &["readout", "score-generation"],
                    "fbt_readout_loss_reduce",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["draft"],
                    "denoise_embed+fbt_readout_proposal_tiles+denoise_reduce",
                    [128, 64, 512],
                    [128, 1, 1],
                ),
                kernel(&["verify"], "fbt_block_verify", [128, 4, 1], [128, 1, 1]),
                kernel(
                    &["objective"],
                    "fbt_mhc4_objective",
                    [1, 1048576, 1],
                    [256, 1, 1],
                ),
            ],
        }
    }

    pub fn mhc4_metal() -> Self {
        let mut schedule = Self::existing_metal();
        for entry in &mut schedule.kernels {
            match entry.kernel {
                "fbt_moe_embed" => entry.nodes.push("patch-encode"),
                "fbt_readout_loss_tiles" | "fbt_readout_loss_reduce" => {
                    entry.nodes.push("patch-expand");
                }
                _ => {}
            }
        }
        schedule.kernels.extend([
            kernel(
                &["patch-encode"],
                "fbt_mhc_replicate",
                [1, 512, 4],
                [256, 1, 1],
            ),
            kernel(
                &["final-norm"],
                "fbt_mhc_mean_rms",
                [1, 512, 4],
                [128, 1, 1],
            ),
        ]);
        schedule
    }

    pub fn cuda_oxide() -> Self {
        Self {
            target: ExecutionTarget::CudaOxide,
            compiler: Compiler::CudaOxide,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["embed"], "embed", [1, 512, 1], [256, 1, 1]),
                kernel(
                    &["attention-norm", "moe-norm"],
                    "rms",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(
                    &["qkv", "attention-project"],
                    "matmul",
                    [16, 16, 16],
                    [16, 16, 1],
                ),
                kernel(
                    &["readout"],
                    "cuda-graph(mhc4_readout_logits+mhc4_readout_loss)",
                    [64, 8, 6],
                    [64, 1, 1],
                ),
                kernel(
                    &["pisa-summaries"],
                    "pisaleaves+pisaupper",
                    [64, 64, 1],
                    [64, 1, 1],
                ),
                kernel(&["pisa-select"], "pisa_select", [1, 8, 64], [64, 1, 1]),
                kernel(
                    &["pisa-attention"],
                    "pisa_attention",
                    [1, 8, 512],
                    [256, 1, 1],
                ),
                kernel(&["route"], "routetopk", [1, 625, 1], [256, 1, 1]),
                kernel(
                    &["pack-routes"],
                    "route_counts+route_layout+route_pack",
                    [64, 512, 3],
                    [256, 1, 1],
                ),
                kernel(
                    &["gate-up"],
                    "routed_project+routed_activate",
                    [64, 16, 8],
                    [256, 1, 1],
                ),
                kernel(&["down"], "routed_project", [64, 16, 8], [256, 1, 1]),
                kernel(
                    &["combine-routes"],
                    "combine_routes",
                    [1, 512, 3],
                    [256, 1, 1],
                ),
                kernel(
                    &["attention-residual", "moe-residual"],
                    "residual",
                    [1, 512, 1],
                    [256, 1, 1],
                ),
                kernel(&["feedback"], "feedback_shift", [1, 512, 1], [256, 1, 1]),
                kernel(&["feedback"], "feedback_norm", [1, 512, 1], [256, 1, 1]),
                kernel(
                    &["feedback"],
                    "matmul(state,gate)",
                    [16, 16, 16],
                    [16, 16, 1],
                ),
                kernel(&["feedback"], "feedback_combine", [1, 512, 1], [256, 1, 1]),
                kernel(&["sample"], "sample", [1, 8192, 1], [256, 1, 1]),
                kernel(&["draft"], "diffusion_sample", [1, 8192, 1], [256, 1, 1]),
                kernel(&["verify"], "verify_prefix", [1, 256, 1], [256, 1, 1]),
                kernel(&["score-generation"], "score", [1, 8192, 1], [256, 1, 1]),
                kernel(&["objective"], "objective", [256, 1, 1], [256, 1, 1]),
            ],
        }
    }

    pub fn mhc4_cuda() -> Self {
        let mut schedule = Self::cuda_oxide();
        for entry in &mut schedule.kernels {
            if entry.kernel == "embed" {
                entry.nodes.push("patch-encode");
            }
        }
        schedule.kernels.extend([
            kernel(
                &["patch-encode"],
                "fbt_mhc_replicate",
                [1, 512, 4],
                [256, 1, 1],
            ),
            kernel(
                &["attention-norm", "moe-norm"],
                "fbt_mhc_predict_rows",
                [1, 24, 2048],
                [256, 1, 1],
            ),
            kernel(
                &["attention-norm", "moe-norm"],
                "fbt_mhc_mix_rms",
                [1, 512, 4],
                [256, 1, 1],
            ),
            kernel(&["qkv"], "qkv_rope", [128, 128, 512], [256, 1, 1]),
            kernel(
                &["attention-project"],
                "fbt_model_tensorops_output_projection_wide",
                [128, 128, 512],
                [256, 1, 1],
            ),
            kernel(
                &["attention-residual", "moe-residual"],
                "fbt_mhc_update_rows",
                [1, 512, 4],
                [256, 1, 1],
            ),
            kernel(
                &["gate-up"],
                "fbt_moe_routed_gate_fused_columns",
                [128, 64, 512],
                [256, 1, 1],
            ),
            kernel(
                &["down"],
                "fbt_moe_routed_down_fused_columns",
                [128, 64, 216],
                [256, 1, 1],
            ),
            kernel(&["combine-routes"], "combine_mhc", [1, 512, 3], [256, 1, 1]),
            kernel(
                &["final-norm"],
                "fbt_mhc_mean_rms",
                [1, 512, 4],
                [256, 1, 1],
            ),
            kernel(
                &["patch-expand"],
                "fbt_readout_loss_tiles",
                [128, 64, 512],
                [256, 1, 1],
            ),
            kernel(
                &["patch-expand"],
                "fbt_readout_loss_reduce",
                [1, 512, 1],
                [256, 1, 1],
            ),
        ]);
        schedule
    }

    /// TVM lowering boundary for stateless matrix operations on Metal.
    /// Stateful PISA, routing, recurrence, verification, and objectives remain
    /// visible as missing coverage until a parity-qualified lowering exists.
    pub fn tvm_metal() -> Self {
        Self::tvm(ExecutionTarget::Metal, [32, 8, 16], [32, 1, 1])
    }

    /// TVM lowering boundary for stateless matrix operations on CUDA.
    pub fn tvm_cuda() -> Self {
        Self::tvm(ExecutionTarget::CudaOxide, [16, 16, 16], [256, 1, 1])
    }

    fn tvm(target: ExecutionTarget, tile: [usize; 3], threads: [usize; 3]) -> Self {
        Self {
            target,
            compiler: Compiler::Tvm,
            capabilities: ModelProgram::fbt_pisa().requirements,
            kernels: vec![
                kernel(&["qkv"], "tvm.project-qkv", tile, threads),
                kernel(
                    &["attention-project"],
                    "tvm.project-attention",
                    tile,
                    threads,
                ),
                kernel(&["gate-up"], "tvm.routed-gate-up", tile, threads),
                kernel(&["down"], "tvm.routed-down", tile, threads),
                kernel(&["readout"], "tvm.readout", tile, threads),
            ],
        }
    }

    pub fn validate(&self, program: &ModelProgram) -> Result<(), String> {
        program.validate()?;
        let capabilities = self.capabilities.iter().copied().collect::<HashSet<_>>();
        if program
            .requirements
            .iter()
            .any(|required| !capabilities.contains(required))
        {
            return Err("execution schedule does not satisfy the model program".into());
        }
        for kernel in &self.kernels {
            if kernel.kernel.is_empty()
                || kernel.nodes.is_empty()
                || kernel.tile.contains(&0)
                || kernel.threads.contains(&0)
                || kernel.nodes.iter().any(|name| program.node(name).is_none())
            {
                return Err(format!("kernel {} is invalid", kernel.kernel));
            }
        }
        Ok(())
    }

    pub fn coverage(
        &self,
        program: &ModelProgram,
        kind: ProgramKind,
    ) -> Result<Vec<SemanticOp>, String> {
        self.validate(program)?;
        let scheduled = self
            .kernels
            .iter()
            .flat_map(|kernel| kernel.nodes.iter().copied())
            .collect::<HashSet<_>>();
        let function = program
            .function(kind)
            .ok_or("model program function is missing")?;
        Ok(function
            .nodes
            .iter()
            .filter(|name| !scheduled.contains(*name))
            .filter_map(|name| program.node(name).map(|node| node.op))
            .collect())
    }

    pub fn kernels_for(
        &self,
        program: &ModelProgram,
        kind: ProgramKind,
    ) -> Result<Vec<&ScheduledKernel>, String> {
        self.validate(program)?;
        let function = program
            .function(kind)
            .ok_or("model program function is missing")?;
        let active = function.nodes.iter().copied().collect::<HashSet<_>>();
        Ok(self
            .kernels
            .iter()
            .filter(|kernel| kernel.nodes.iter().any(|name| active.contains(name)))
            .collect())
    }

    pub fn partitions(&self) -> HashMap<&'static str, Vec<&'static str>> {
        self.kernels
            .iter()
            .map(|kernel| (kernel.kernel, kernel.nodes.clone()))
            .collect()
    }
}

fn kernel(
    nodes: &[&'static str],
    name: &'static str,
    tile: [usize; 3],
    threads: [usize; 3],
) -> ScheduledKernel {
    ScheduledKernel {
        nodes: nodes.to_vec(),
        kernel: name,
        tile,
        threads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules() {
        let program = ModelProgram::fbt_pisa();
        ExecutionSchedule::existing_metal()
            .validate(&program)
            .unwrap();
        ExecutionSchedule::cuda_oxide().validate(&program).unwrap();
        ExecutionSchedule::tvm_metal().validate(&program).unwrap();
        ExecutionSchedule::tvm_cuda().validate(&program).unwrap();
        let missing = ExecutionSchedule::cuda_oxide()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(!missing.contains(&SemanticOp::Feedback));
        assert!(!missing.contains(&SemanticOp::PackRoutes));
        assert!(!missing.contains(&SemanticOp::CombineRoutes));
        assert!(!missing.contains(&SemanticOp::PisaSelect));
        assert!(!missing.contains(&SemanticOp::RouteTopK));
        let metal_missing = ExecutionSchedule::existing_metal()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(metal_missing.is_empty());
        let round_missing = ExecutionSchedule::existing_metal()
            .coverage(&program, ProgramKind::ExperimentRound)
            .unwrap();
        assert!(round_missing.is_empty());
        let tvm_missing = ExecutionSchedule::tvm_cuda()
            .coverage(&program, ProgramKind::Prefill)
            .unwrap();
        assert!(tvm_missing.contains(&SemanticOp::PisaAttention));
        assert!(!tvm_missing.contains(&SemanticOp::ProjectQkv));

        let mhc4 = ModelProgram::mhc4_program();
        mhc4.validate().unwrap();
        let metal = ExecutionSchedule::mhc4_metal();
        let cuda = ExecutionSchedule::mhc4_cuda();
        assert!(
            metal
                .coverage(&mhc4, ProgramKind::ExperimentRound)
                .unwrap()
                .is_empty()
        );
        assert!(
            cuda.coverage(&mhc4, ProgramKind::ExperimentRound)
                .unwrap()
                .is_empty()
        );
        for schedule in [&metal, &cuda] {
            let names = schedule
                .kernels_for(&mhc4, ProgramKind::ExperimentRound)
                .unwrap()
                .into_iter()
                .map(|entry| entry.kernel)
                .collect::<HashSet<_>>();
            for required in [
                "fbt_mhc_replicate",
                "qkv_rope",
                "fbt_model_tensorops_output_projection_wide",
                "fbt_moe_routed_gate_fused_columns",
                "fbt_moe_routed_down_fused_columns",
                "fbt_mhc_predict_rows",
                "fbt_mhc_mix_rms",
                "fbt_mhc_update_rows",
                "combine_mhc",
                "fbt_mhc_mean_rms",
                "fbt_readout_loss_tiles",
                "fbt_readout_loss_reduce",
            ] {
                assert!(names.contains(required), "missing kernel {required}");
            }
            for stale in [
                "fbt_patch_transpose",
                "fbt_patch_encode",
                "fbt_patch_capture",
                "fbt_patch_norm",
                "fbt_readout_tiles",
                "mhc4_patch_mean",
                "patch_mean_nll",
            ] {
                assert!(!names.contains(stale), "stale kernel {stale}");
            }
        }
        let metal_names = metal
            .kernels_for(&mhc4, ProgramKind::ExperimentRound)
            .unwrap()
            .into_iter()
            .map(|entry| entry.kernel)
            .collect::<HashSet<_>>();
        assert!(metal_names.contains("fbt_moe_embed"));
    }
}
