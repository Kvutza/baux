use super::FbtModel;
use super::bo_run::{BoLoop, configure_task};
use super::bo_task::{BoConfig, Mhc4Task};
use super::readout::Mhc4Evaluation;
use crate::capture::{CudaResult, cuda_error, target_name};
use baux::artifact::json::{Value, json};
use baux::semantics::optimizer::MAX_HISTORY;
use baux_cuda_device::ScoreStats;
use std::fs;
use std::io::{BufWriter, Write};

pub(super) fn validate_tasks(
    model: &mut FbtModel,
    tasks: &[Mhc4Task],
    cfg: &BoConfig,
    round: usize,
    phase: &str,
    progress: &mut BufWriter<fs::File>,
) -> CudaResult<Value> {
    let started = std::time::Instant::now();
    let mut rewards = Vec::with_capacity(tasks.len());
    let mut nlls = Vec::with_capacity(tasks.len());
    let mut sources = Vec::with_capacity(tasks.len());
    let mut weighted = [0.0_f64; 2];
    let mut total_weight = 0_u64;
    let mut evaluated_tokens = 0_usize;
    let mut device_ms = 0.0_f32;
    for task in tasks {
        configure_task(task, model)?;
        let evaluation = model.evaluate_mhc4(
            cfg.visits,
            cfg.temperature,
            task.sample_seed(cfg.evaluation_seed),
            false,
        )?;
        let stats = evaluation
            .stats
            .ok_or("CUDA mHC-4 held-out task was not scored")?;
        rewards.push(stats.reward);
        nlls.push(stats.mean_nll);
        let weight = task.weight();
        total_weight += u64::from(weight);
        weighted[0] += f64::from(stats.mean_nll) * f64::from(weight);
        let estimate = evaluation
            .estimate
            .ok_or("CUDA held-out likelihood omitted its estimate")?;
        let target_len = task.targets().len();
        weighted[1] += estimate.bits / target_len as f64 * f64::from(weight);
        evaluated_tokens += target_len;
        sources.push(json!({
            "source": task.corpus.as_ref().map_or("task", |c| c.source.as_str()),
            "weight": weight, "nll": stats.mean_nll,
            "bits_per_token": estimate.bits / target_len as f64,
            "corpus_offset": task.corpus.as_ref().map(|c| c.batch.offset),
            "scored_tokens": target_len,
        }));
        device_ms += evaluation.device_ms;
    }
    if total_weight == 0 {
        return Err("CUDA held-out likelihood has no positive source weight".into());
    }
    let mean_nll = weighted[0] / total_weight as f64;
    let record = json!({
        "schema": "baux.cuda.mhc4-validation.v1",
        "phase": phase, "round": round, "tasks": tasks.len(),
        "reward": -mean_nll, "mean_nll": mean_nll,
        "bits_per_token": weighted[1] / total_weight as f64,
        "sources": sources, "evaluated_tokens": evaluated_tokens,
        "aggregation": "weighted-mean",
        "task_rewards": rewards, "task_nlls": nlls,
        "device_ms": device_ms,
        "wall_ms": started.elapsed().as_secs_f64() * 1000.0,
        "used_for_acceptance": false,
        "excluded_from_round_timing": true
    });
    writeln!(
        progress,
        "{}",
        baux::artifact::json::to_string(&record).map_err(|e| e.to_string())?
    )
    .map_err(|e| format!("write CUDA mHC-4 validation log: {e}"))?;
    progress
        .flush()
        .map_err(|e| format!("flush CUDA mHC-4 validation log: {e}"))?;
    Ok(record)
}

pub(super) fn record_stage(
    cfg: &BoConfig,
    model: &mut FbtModel,
    round: usize,
    round_seed: u64,
    accepted: bool,
    stage_profiles: &mut Vec<Value>,
) -> CudaResult<()> {
    let Some(sample) = super::bo_task::stage_sample(round, cfg.bo_rounds, cfg.stage_samples) else {
        return Ok(());
    };
    let profiled = model.evaluate_mhc4(cfg.visits, cfg.temperature, round_seed, true)?;
    let core = profiled
        .core_profile
        .as_ref()
        .ok_or("CUDA mHC-4 stage sample omitted its core profile")?;
    let generation = profiled
        .generation_profile
        .as_ref()
        .ok_or("CUDA mHC-4 stage sample omitted its generation profile")?;
    stage_profiles.push(json!({
        "sample": sample,
        "round": round,
        "state": if accepted { "accepted_candidate" } else { "restored_incumbent" },
        "core": {
            "encode_ms": core.embed_ms, "mhc_ms": core.mhc_ms, "qkv_ms": core.qkv_ms,
            "index_ms": core.index_ms, "pisa_ms": core.pisa_ms, "output_ms": core.output_ms,
            "attention_ms": core.attention_ms, "moe_router_ms": core.moe_router_ms,
            "moe_route_ms": core.moe_route_ms, "moe_experts_ms": core.moe_experts_ms,
            "moe_combine_ms": core.moe_combine_ms, "moe_ms": core.moe_ms,
            "total_ms": core.total_ms
        },
        "generation": {
            "init_ms": generation.init_ms, "gemm_ms": generation.gemm_ms,
            "project_ms": generation.project_ms, "reduce_ms": generation.reduce_ms,
            "total_ms": generation.total_ms
        },
        "device_ms": profiled.device_ms,
        "wall_ms": profiled.wall_ms
    }));
    Ok(())
}

fn audit_index(
    cfg: &BoConfig,
    model: &mut FbtModel,
    shadow_reward: f32,
) -> CudaResult<Option<Value>> {
    if std::env::var_os("BAUX_CUDA_PISA_COMPARE").is_none() {
        return Ok(None);
    }
    model.force_index()?;
    let exact = model.evaluate_mhc4(cfg.visits, cfg.temperature, cfg.evaluation_seed, false)?;
    let exact_stats = exact
        .stats
        .ok_or("CUDA mHC-4 exact index audit was not scored")?;
    let (support_recall, exact_rows, compared_rows) = model.mhc4_overlap(exact.visits)?;
    Ok(Some(json!({
        "shadow_support_recall": support_recall,
        "exact_support_rows": exact_rows,
        "compared_rows": compared_rows,
        "shadow_reward": shadow_reward,
        "full_reward": exact_stats.reward,
        "absolute_reward_error": (shadow_reward - exact_stats.reward).abs(),
        "full_core_device_ms": exact.core_ms,
        "full_generation_device_ms": exact.generation_ms,
        "full_total_device_ms": exact.device_ms,
        "excluded_from_timed_transaction": true
    })))
}

fn build_summary(
    cfg: &BoConfig,
    model: &FbtModel,
    state: &BoLoop,
    base: &Mhc4Evaluation,
    base_stats: &ScoreStats,
    final_inc: &Mhc4Evaluation,
    final_stats: &ScoreStats,
    validation_ms: f64,
    tell_ms: f64,
    index_audit: Option<Value>,
    params: (u64, u64),
    retain_fixed: bool,
) -> CudaResult<Value> {
    let candidate = state
        .candidate
        .as_ref()
        .ok_or("CUDA mHC-4 BO produced no candidate")?;
    let stats = state
        .stats
        .as_ref()
        .ok_or("CUDA mHC-4 BO produced no candidate score")?;
    let device_ms = state.proposal_ms + state.incumbent_ms + state.eval_ms + state.restore_ms;
    let evaluated_total = state
        .candidate_tokens
        .checked_add(state.incumbent_tokens)
        .ok_or("CUDA mHC-4 evaluated-token count overflow")?;
    let shape_name = match cfg.trust_shape {
        baux::semantics::mhc4::ScaleShape::Scalar => "scalar",
        baux::semantics::mhc4::ScaleShape::TensorFamilyStatic => "tensor-family-static",
    };
    Ok(json!({
        "schema": "baux.cuda.mhc4-bo-round.v7",
        "target": target_name(&model.context)?,
        "hardware": model.context.device_name().map_err(cuda_error)?,
        "model_seed": cfg.model_seed,
        "checkpoint_format": model.weights.format,
        "execution_architecture": "mhc4",
        "architecture_weights_match": model.weights.format == "mhc4",
        "phase": "optimization",
        "bo": {
            "rounds": cfg.bo_rounds,
            "observations": state.observations,
            "accepted_rounds": state.accepted_rounds,
            "optimizer_incumbent": state.history.base_value(),
            "optimizer_variance": state.history.base_variance(),
            "acquisition": cfg.acquisition,
            "ucb_beta": cfg.beta,
            "surrogate": "neighbor-value-v1",
            "initial_observations": cfg.surrogate.neighbors,
            "history": state.history.len(),
            "total_observations": state.total_observations,
            "history_capacity": MAX_HISTORY
        },
        "training_conditions": {
            "tasks": cfg.training.count(cfg.bo_rounds),
            "schedule": if cfg.training.fixed() { "fixed" } else { "metal-corpus-window" },
            "paired_candidate_incumbent": true, "incumbent_remeasured_each_round": !cfg.training.fixed(), "fixed_mhc4_state_reused": retain_fixed
        },
        "validation": {
            "configured": !cfg.validation.is_empty(), "tasks": cfg.validation.len(), "interval": cfg.validation_interval,
            "learning_checkpoints": cfg.window_tokens.is_none(), "records": state.validation_records,
            "used_for_acceptance": false, "excluded_from_round_timing": true
        },
        "proposal": {
            "trust_region_shape": shape_name, "method": "independent", "distribution": "gaussian", "full_coordinate_innovation": true,
            "arms": cfg.arms, "candidates": cfg.candidates, "candidates_per_arm": cfg.candidates / cfg.arms,
            "radius_initial": cfg.radius, "radius_final": state.controller.radius(), "radius_min": cfg.radius_min, "radius_max": cfg.radius_max,
            "device_ms": state.proposal_ms, "logical_parameter_coordinates": params.0, "materialized_physical_elements": params.1
        },
        "objective": if cfg.window_tokens.is_some() { "causal-corpus-cross-entropy" } else { "conditional-target-cross-entropy" },
        "context_window_tokens": cfg.primary.capacity()?,
        "prompt_tokens": if cfg.window_tokens.is_some() { 0 } else { cfg.primary.input().len() },
        "drafted_tokens": 0,
        "core_context_positions": cfg.core_context,
        "evaluated_core_positions": cfg.core_rows,
        "core_active_positions": cfg.core_positions,
        "evaluated_tokens": evaluated_total,
        "candidate_evaluated_tokens": state.candidate_tokens,
        "incumbent_evaluated_tokens": state.incumbent_tokens,
        "repaired_tokens": 0,
        "generated_tokens": if cfg.window_tokens.is_some() { 0 } else { evaluated_total },
        "prediction_tokens": if cfg.window_tokens.is_some() { evaluated_total } else { 0 },
        "committed_tokens": state.committed_tokens,
        "scored_tokens": evaluated_total,
        "forward_passes": cfg.bo_rounds * if cfg.training.fixed() { 1 } else { 2 },
        "temperature": cfg.temperature,
        "seeds": {
            "proposal": cfg.proposal_seed,
            "acquisition": cfg.acquisition_seed,
            "evaluation": cfg.evaluation_seed,
            "corpus": cfg.reference_seed
        },
        "readout_schedule": "flat-linear",
        "incumbent": {
            "reward": base_stats.reward,
            "mean_nll": base_stats.mean_nll,
            "core_device_ms": base.core_ms,
            "generation_device_ms": base.generation_ms,
            "objective_device_ms": base.objective_ms,
            "device_ms": base.device_ms,
            "evaluation_wall_ms": base.wall_ms,
            "timed_in_round": false
        },
        "candidate": {
            "reward": stats.reward,
            "mean_nll": stats.mean_nll,
            "maximum_nll": stats.maximum_nll,
            "positional_accuracy": stats.positional_accuracy,
            "mismatches": stats.mismatches,
            "patch_nll_octiles": stats.patch_nll_octiles,
            "core_device_ms": candidate.core_ms,
            "generation_device_ms": candidate.generation_ms,
            "objective_reduce_device_ms": candidate.objective_ms,
            "evaluation_wall_ms": candidate.wall_ms,
            "total_device_ms": candidate.device_ms
        },
        "final_incumbent_validation": {
            "reward": final_stats.reward,
            "mean_nll": final_stats.mean_nll,
            "device_ms": final_inc.device_ms,
            "wall_ms": validation_ms,
            "tokens_provenance": "recomputed-from-final-accepted-incumbent",
            "excluded_from_round_timing": true
        },
        "profile": (!state.stage_profiles.is_empty()).then(|| json!({
            "requested_samples": cfg.stage_samples,
            "recorded_samples": state.stage_profiles.len(),
            "samples": state.stage_profiles,
            "excluded_from_round_timing": true
        })),
        "index_audit": index_audit,
        "tell": {
            "accepted": state.accepted_rounds > 0,
            "device_restore_ms": state.restore_ms,
            "wall_ms": tell_ms,
            "history_after": state.history.len(),
            "total_observations": state.total_observations,
            "trust_region_length": state.controller.radius()
        },
        "incumbent_evaluation_device_ms": state.incumbent_ms,
        "candidate_evaluation_device_ms": state.eval_ms,
        "device_ms": device_ms,
        "wall_ms": state.round_wall,
        "rounds_per_device_second": cfg.bo_rounds as f64 * 1000.0 / f64::from(device_ms),
        "full_generation_loop": cfg.window_tokens.is_none(),
        "full_scoring_loop": true,
        "full_bo_loop": true,
        "complete_candidate_transaction": true
    }))
}

pub(super) fn finish_bo(
    cfg: &BoConfig,
    model: &mut FbtModel,
    state: &BoLoop,
    base: &Mhc4Evaluation,
    base_stats: &ScoreStats,
    params: (u64, u64),
    retain_fixed: bool,
    output: &std::path::Path,
) -> CudaResult<()> {
    let stats = state
        .stats
        .ok_or("CUDA mHC-4 BO produced no candidate score")?;
    let val_wall = std::time::Instant::now();
    configure_task(&cfg.primary, model)?;
    let final_inc = model.evaluate_mhc4(
        cfg.visits,
        cfg.temperature,
        cfg.primary.sample_seed(cfg.evaluation_seed),
        false,
    )?;
    let final_stats = final_inc
        .stats
        .ok_or("CUDA mHC-4 final incumbent was not scored")?;
    if cfg.training.fixed() && (final_stats.reward - state.best_reward).abs() > 1.0e-5 {
        return Err(format!(
            "CUDA mHC-4 final incumbent reward {} differs from optimizer best {}",
            final_stats.reward, state.best_reward
        ));
    }
    let tokens = model.mhc4_tokens()?;
    let val_ms = val_wall.elapsed().as_secs_f64() * 1000.0;
    let index_audit = audit_index(cfg, model, stats.reward)?;
    let tell_wall = std::time::Instant::now();
    let tell_ms = tell_wall.elapsed().as_secs_f64() * 1000.0;
    let record = build_summary(
        cfg,
        model,
        state,
        base,
        base_stats,
        &final_inc,
        &final_stats,
        val_ms,
        tell_ms,
        index_audit,
        params,
        retain_fixed,
    )?;
    fs::write(
        output.join("tokens.json"),
        baux::artifact::json::to_string(&tokens).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let json = baux::artifact::json::pretty_string(&record).map_err(|e| e.to_string())?;
    fs::write(output.join("result.json"), &json).map_err(|e| e.to_string())?;
    println!("{json}");
    Ok(())
}
