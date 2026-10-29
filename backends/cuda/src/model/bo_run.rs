use super::FbtModel;
use super::bo_task::{BoConfig, Mhc4Task, condition_seed, learning_checkpoint, validate_task};
use super::readout::Mhc4Evaluation;
use super::weights::WeightBackup;
use crate::{CudaResult, cuda_error, target_name};
use baux::artifact::json::{Value, json};
use baux::semantics::optimizer::{
    MAX_HISTORY, NeighborValueEstimator, RadiusController, ReliabilityEvidence, response_rms,
    splitmix64,
};
use baux_cuda_device::ScoreStats;
use std::fs;
use std::io::{BufWriter, Write};

fn configure_task(task: &Mhc4Task, model: &mut FbtModel) -> CudaResult<()> {
    if let Some(corpus) = &task.corpus {
        model.configure_causal(&corpus.batch)
    } else {
        model.configure_mhc4(&task.prompt, task.targets.len(), Some(&task.targets))
    }
}

fn validate_tasks(
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

struct RoundProposal {
    round_radius: f32,
    root: u64,
    proposal_seed: u64,
    selected: usize,
    geometry_row: Vec<f32>,
    acquisition_score: f32,
    candidate_mean: f32,
    candidate_se: f32,
    incumbent_mean: f32,
    incumbent_se: f32,
    initializing: bool,
    proposal_ms: f32,
}

struct BoLoop {
    controller: RadiusController,
    history: NeighborValueEstimator,
    current_stats: ScoreStats,
    current_nlls: Vec<f32>,
    best_reward: f32,
    accepted_rounds: usize,
    round_wall: f64,
    proposal_ms: f32,
    eval_ms: f32,
    incumbent_ms: f32,
    restore_ms: f32,
    candidate_tokens: usize,
    incumbent_tokens: usize,
    committed_tokens: usize,
    total_observations: usize,
    observations: Vec<Value>,
    stage_profiles: Vec<Value>,
    validation_records: Vec<Value>,
    candidate: Option<Mhc4Evaluation>,
    stats: Option<ScoreStats>,
}

fn pair_response(
    cfg: &BoConfig,
    incumbent_nlls: &[f32],
    candidate_nlls: &[f32],
    evaluated: &mut Mhc4Evaluation,
) -> CudaResult<(Vec<f32>, f32, f32)> {
    if candidate_nlls.len() != incumbent_nlls.len() || candidate_nlls.is_empty() {
        return Err("CUDA mHC-4 paired patch response shape changed".into());
    }
    if cfg.window_tokens.is_some() {
        let estimate = evaluated
            .estimate
            .ok_or("CUDA causal objective omitted its paired estimate")?;
        return Ok((
            evaluated
                .patch_gains
                .take()
                .ok_or("CUDA causal objective omitted patch responses")?,
            estimate
                .improvement
                .ok_or("CUDA causal objective omitted paired gain")?,
            estimate
                .paired_variance
                .ok_or("CUDA causal objective omitted paired variance")?,
        ));
    }
    let responses = incumbent_nlls
        .iter()
        .zip(candidate_nlls)
        .map(|(inc, cand)| inc - cand)
        .collect::<Vec<_>>();
    let mean = responses.iter().sum::<f32>() / responses.len() as f32;
    let variance = if responses.len() > 1 {
        responses.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / ((responses.len() - 1) * responses.len()) as f32
    } else {
        0.0
    };
    Ok((responses, mean, variance))
}

impl BoLoop {
    fn new(cfg: &BoConfig, base_stats: ScoreStats, base_nlls: Vec<f32>) -> CudaResult<Self> {
        Ok(Self {
            controller: RadiusController::configured(
                cfg.radius,
                cfg.radius_min,
                cfg.radius_max,
                cfg.reliability,
            )?,
            // Paired Metal search is relative: the measured initial incumbent is the
            // zero origin, and accepted response means/variances accumulate from it.
            history: NeighborValueEstimator::with_spec(
                0.0,
                0.0,
                cfg.acquisition_seed,
                cfg.surrogate,
            )?,
            current_stats: base_stats,
            current_nlls: base_nlls,
            best_reward: base_stats.reward,
            accepted_rounds: 0,
            round_wall: 0.0,
            proposal_ms: 0.0,
            eval_ms: 0.0,
            incumbent_ms: 0.0,
            restore_ms: 0.0,
            candidate_tokens: 0,
            incumbent_tokens: 0,
            committed_tokens: 0,
            total_observations: 1,
            observations: Vec::with_capacity(cfg.bo_rounds),
            stage_profiles: Vec::with_capacity(cfg.stage_samples.min(cfg.bo_rounds)),
            validation_records: Vec::new(),
            candidate: None,
            stats: None,
        })
    }

    fn measure_incumbent(
        &mut self,
        cfg: &BoConfig,
        model: &mut FbtModel,
        task: &Mhc4Task,
        round_seed: u64,
        retain_fixed: bool,
    ) -> CudaResult<(ScoreStats, Vec<f32>, f32)> {
        if !retain_fixed {
            configure_task(task, model)?;
        }
        let out = if cfg.training.fixed() {
            (self.current_stats, self.current_nlls.clone(), 0.0_f32)
        } else {
            let incumbent = model.evaluate_mhc4(cfg.visits, cfg.temperature, round_seed, false)?;
            let stats = incumbent
                .stats
                .ok_or("CUDA mHC-4 round incumbent was not scored")?;
            let nlls = incumbent
                .patch_nlls
                .ok_or("CUDA mHC-4 round incumbent patch losses were not returned")?;
            self.incumbent_tokens = self
                .incumbent_tokens
                .checked_add(task.targets().len())
                .ok_or("CUDA mHC-4 incumbent token count overflow")?;
            (stats, nlls, incumbent.device_ms)
        };
        self.incumbent_ms += out.2;
        if cfg.window_tokens.is_some() {
            model.capture_pair()?;
        }
        Ok(out)
    }

    fn propose(
        &mut self,
        cfg: &BoConfig,
        model: &mut FbtModel,
        backup: &WeightBackup,
        round: usize,
    ) -> CudaResult<RoundProposal> {
        let round_radius = self.controller.radius();
        let root = cfg
            .proposal_seed
            .checked_add(round as u64)
            .ok_or("CUDA mHC-4 proposal seed overflow")?;
        let seeds = (0..cfg.pool_shape.count())
            .map(|c| cfg.pool_shape.seed(root, c))
            .collect::<Vec<_>>();
        let geometry = model.weights.pool_distances(
            backup,
            &model.module,
            &model.stream,
            round_radius,
            &seeds,
            self.history.len(),
        )?;
        let incumbent_pred = self.history.incumbent_prediction()?;
        let round_acquire = cfg
            .acquisition_seed
            .checked_add(round as u64)
            .ok_or("CUDA mHC-4 acquisition seed overflow")?;
        let draws = self
            .history
            .identities()
            .iter()
            .map(|&id| baux::semantics::random::normal_metric(round_acquire, id, 0) as f32)
            .collect::<Vec<_>>();
        let pool = geometry
            .iter()
            .map(|row| {
                self.history.acquire(
                    row,
                    cfg.beta,
                    (cfg.acquisition == "thompson").then_some(draws.as_slice()),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let modeled = pool
            .iter()
            .enumerate()
            .max_by(|(li, l), (ri, r)| l.score.total_cmp(&r.score).then_with(|| ri.cmp(li)))
            .map(|(i, _)| i)
            .ok_or("CUDA mHC-4 candidate pool is empty")?;
        let initializing = self.history.len() < cfg.surrogate.neighbors;
        let selected = if initializing {
            round % pool.len()
        } else {
            modeled
        };
        let cand_pred = pool[selected];
        let proposal_ms = model.weights.materialize_gaussian(
            backup,
            &model.module,
            &model.stream,
            round_radius,
            seeds[selected],
        )?;
        self.proposal_ms += proposal_ms;
        Ok(RoundProposal {
            round_radius,
            root,
            proposal_seed: seeds[selected],
            selected,
            geometry_row: geometry[selected].clone(),
            acquisition_score: if initializing { 0.0 } else { cand_pred.score },
            candidate_mean: cand_pred.mean,
            candidate_se: cand_pred.standard_error,
            incumbent_mean: incumbent_pred.mean,
            incumbent_se: incumbent_pred.standard_error,
            initializing,
            proposal_ms,
        })
    }

    fn apply_decision(
        &mut self,
        cfg: &BoConfig,
        model: &mut FbtModel,
        backup: &mut WeightBackup,
        task: &Mhc4Task,
        prop: &RoundProposal,
        evaluated: &mut Mhc4Evaluation,
        incumbent_nlls: &[f32],
        rewarming: bool,
    ) -> CudaResult<(
        baux::semantics::optimizer::ReliabilityDecision,
        f32,
        f32,
        f32,
        f32,
        usize,
        f64,
        f32,
    )> {
        let evaluated_stats = evaluated
            .stats
            .ok_or("CUDA mHC-4 candidate was not scored")?;
        let candidate_nlls = evaluated
            .patch_nlls
            .as_ref()
            .ok_or("CUDA mHC-4 candidate patch losses were not returned")?
            .clone();
        let (responses, paired_mean, paired_var) =
            pair_response(cfg, incumbent_nlls, &candidate_nlls, evaluated)?;
        let resp_radius = response_rms(Some(&responses), f64::from(prop.round_radius));
        let opt_cand = self.history.base_value() + paired_mean;
        let opt_var = self.history.base_variance() + paired_var;
        let pred_gain = f64::from(prop.candidate_mean) - f64::from(prop.incumbent_mean);
        let pred_var = f64::from(prop.candidate_se).powi(2) + f64::from(prop.incumbent_se).powi(2);
        let decision = if prop.initializing {
            let threshold = if rewarming {
                2.0 * (f64::from(opt_var) + f64::from(self.history.base_variance())).sqrt()
            } else {
                0.0
            };
            self.controller.initial_evidence(
                f64::from(opt_cand) - f64::from(self.history.base_value()),
                pred_gain,
                threshold,
            )?
        } else {
            self.controller.observe_evidence(
                ReliabilityEvidence {
                    observed_improvement: f64::from(paired_mean),
                    observation_variance: f64::from(paired_var),
                    predicted_improvement: pred_gain,
                    prediction_variance: pred_var,
                    response_radius: resp_radius,
                    nominal_radius: f64::from(prop.round_radius),
                    accepted: false,
                },
                true,
            )?
        };
        model
            .weights
            .archive_neighbor(prop.selected, self.history.next_slot()?, &model.stream)?;
        self.history
            .append(&prop.geometry_row, opt_cand, opt_var, decision.accepted)?;
        let restore_round = if decision.accepted {
            self.accepted_rounds += 1;
            self.best_reward = evaluated_stats.reward;
            self.committed_tokens = self
                .committed_tokens
                .checked_add(task.targets().len())
                .ok_or("CUDA mHC-4 committed token count overflow")?;
            let elapsed = model.weights.commit(backup, &model.stream)?;
            model.commit_index(true);
            if cfg.training.fixed() {
                self.current_stats = evaluated_stats;
                self.current_nlls = candidate_nlls;
            }
            elapsed
        } else {
            let elapsed = model.weights.restore(backup, &model.stream)?;
            model.commit_index(false);
            elapsed
        };
        self.restore_ms += restore_round;
        self.total_observations += 1;
        Ok((
            decision,
            opt_cand,
            opt_var,
            paired_mean,
            paired_var,
            responses.len(),
            resp_radius,
            restore_round,
        ))
    }

    fn step_round(
        &mut self,
        cfg: &BoConfig,
        model: &mut FbtModel,
        backup: &mut WeightBackup,
        round: usize,
        retain_fixed: bool,
        progress: &mut BufWriter<fs::File>,
        val_progress: &mut Option<BufWriter<fs::File>>,
    ) -> CudaResult<()> {
        let iter_wall = std::time::Instant::now();
        let task = cfg.training.task(round)?;
        validate_task(&task, cfg.generated)?;
        let seed_cond = condition_seed(cfg.evaluation_seed, round)?;
        let round_seed = task.sample_seed(if cfg.training.fixed() {
            cfg.evaluation_seed
        } else {
            splitmix64(seed_cond ^ 0x7361_6d70_6c65_2d78)
        });
        let (inc_stats, inc_nlls, round_inc) =
            self.measure_incumbent(cfg, model, &task, round_seed, retain_fixed)?;
        let rewarming = self.total_observations > self.history.len();
        let prop = self.propose(cfg, model, backup, round)?;
        let mut evaluated = model.evaluate_mhc4(cfg.visits, cfg.temperature, round_seed, false)?;
        self.eval_ms += evaluated.device_ms;
        self.candidate_tokens = self
            .candidate_tokens
            .checked_add(task.targets().len())
            .ok_or("CUDA mHC-4 candidate token count overflow")?;
        let evaluated_stats = evaluated
            .stats
            .ok_or("CUDA mHC-4 candidate was not scored")?;
        let outcome = self.apply_decision(
            cfg,
            model,
            backup,
            &task,
            &prop,
            &mut evaluated,
            &inc_nlls,
            rewarming,
        )?;
        let timed_wall = iter_wall.elapsed().as_secs_f64() * 1000.0;
        record_stage(
            cfg,
            model,
            round,
            round_seed,
            outcome.0.accepted,
            &mut self.stage_profiles,
        )?;
        let observation = self.format_round(
            cfg,
            round,
            seed_cond,
            round_seed,
            &task,
            &prop,
            &inc_stats,
            &evaluated,
            &evaluated_stats,
            round_inc,
            timed_wall,
            &outcome,
        );
        writeln!(
            progress,
            "{}",
            baux::artifact::json::to_string(&observation).map_err(|e| e.to_string())?
        )
        .map_err(|e| format!("write round log: {e}"))?;
        progress
            .flush()
            .map_err(|e| format!("flush round log: {e}"))?;
        self.observations.push(observation);
        self.candidate = Some(evaluated);
        self.stats = Some(evaluated_stats);
        self.round_wall += timed_wall;
        let completed = round + 1;
        let scheduled = cfg
            .validation_interval
            .is_some_and(|interval| completed % interval == 0)
            || (cfg.window_tokens.is_none() && learning_checkpoint(completed));
        if completed < cfg.bo_rounds
            && scheduled
            && let Some(vp) = val_progress.as_mut()
        {
            self.validation_records.push(validate_tasks(
                model,
                &cfg.validation,
                cfg,
                completed,
                "checkpoint",
                vp,
            )?);
        }
        Ok(())
    }

    fn format_round(
        &self,
        cfg: &BoConfig,
        round: usize,
        seed_cond: u64,
        round_seed: u64,
        task: &Mhc4Task,
        prop: &RoundProposal,
        inc_stats: &ScoreStats,
        evaluated: &Mhc4Evaluation,
        eval_stats: &ScoreStats,
        round_inc: f32,
        timed_wall: f64,
        outcome: &(
            baux::semantics::optimizer::ReliabilityDecision,
            f32,
            f32,
            f32,
            f32,
            usize,
            f64,
            f32,
        ),
    ) -> Value {
        let (
            decision,
            opt_cand,
            opt_var,
            paired_mean,
            paired_var,
            channels,
            resp_radius,
            restore_round,
        ) = outcome;
        let round_ms = prop.proposal_ms + round_inc + evaluated.device_ms + restore_round;
        json!({
            "round": round,
            "condition": {
                "task": if cfg.training.fixed() { 0 } else { round },
                "seed": seed_cond,
                "evaluation_seed": round_seed,
                "input_tokens": task.input().len(),
                "target_tokens": task.targets().len(),
                "corpus_offset": task.corpus.as_ref().map(|c| c.batch.offset),
                "source": task.corpus.as_ref().map(|c| c.source.as_str()),
                "causal": cfg.window_tokens.is_some()
            },
            "phase": if prop.initializing { "initialization" } else { "operator_bo" },
            "radius": prop.round_radius,
            "next_radius": decision.next_radius,
            "proposal_root": prop.root,
            "proposal_seed": prop.proposal_seed,
            "selected": prop.selected,
            "acquisition_score": prop.acquisition_score,
            "predicted_mean": prop.candidate_mean,
            "predicted_standard_error": prop.candidate_se,
            "incumbent_mean": prop.incumbent_mean,
            "incumbent_standard_error": prop.incumbent_se,
            "reward": eval_stats.reward,
            "incumbent_condition_reward": inc_stats.reward,
            "incumbent_condition_nll": inc_stats.mean_nll,
            "optimizer_reward": opt_cand,
            "optimizer_variance": opt_var,
            "paired_patch_improvement": paired_mean,
            "paired_patch_variance": paired_var,
            "paired_patch_channels": channels,
            "mean_nll": eval_stats.mean_nll,
            "last_accepted_condition_reward": self.best_reward,
            "optimizer_incumbent": self.history.base_value(),
            "accepted": decision.accepted,
            "observed_improvement": decision.improvement,
            "acceptance_threshold": decision.threshold,
            "predicted_improvement": decision.predicted_improvement,
            "reliability_action": decision.action.name(),
            "standardized_error": decision.standardized_error,
            "sharpness_ratio": decision.sharpness_ratio,
            "agreement_ratio": decision.agreement_ratio,
            "response_radius": resp_radius,
            "accepted_rounds": self.accepted_rounds,
            "history": self.history.len(),
            "acceptance_rate": self.accepted_rounds as f64 / (round + 1) as f64,
            "proposal_device_ms": prop.proposal_ms,
            "incumbent_evaluation_device_ms": round_inc,
            "candidate_evaluation_device_ms": evaluated.device_ms,
            "evaluation_device_ms": round_inc + evaluated.device_ms,
            "restore_device_ms": restore_round,
            "device_ms": round_ms,
            "evaluation_wall_ms": evaluated.wall_ms,
            "wall_ms": timed_wall,
            "candidate_evaluated_tokens": self.candidate_tokens,
            "incumbent_evaluated_tokens": self.incumbent_tokens,
            "evaluated_tokens": self.candidate_tokens + self.incumbent_tokens,
            "committed_tokens": self.committed_tokens
        })
    }
}

fn record_stage(
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
        "readout_schedule": "hierarchical-8x6",
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

/// Run the resident Gaussian ENN experiment with the shared host controller.
/// Optional reliability and surrogate JSON carry the resolved TOML settings.
pub fn mhc4_bo(args: &[String]) -> CudaResult<()> {
    let cfg = BoConfig::parse(args)?;
    let output = std::path::Path::new(&cfg.output_dir);
    fs::create_dir(output).map_err(|e| format!("create fresh output directory: {e}"))?;
    let mut progress = BufWriter::new(
        fs::File::create(output.join("rounds.jsonl"))
            .map_err(|e| format!("create round log: {e}"))?,
    );
    let mut val_progress = if cfg.validation.is_empty() {
        None
    } else {
        Some(BufWriter::new(
            fs::File::create(output.join("validation.jsonl"))
                .map_err(|e| format!("create validation log: {e}"))?,
        ))
    };
    let mut model = FbtModel::initialize_mhc4(cfg.model_seed, cfg.core_rows, cfg.core_context)?;
    model.weights.set_shape(cfg.trust_shape, &model.stream)?;
    let mut backup = model.weights.backup(&model.stream)?;
    let physical = model.weights.physical_elements();
    let logical = physical - model.weights.embedding.len() as u64;
    configure_task(&cfg.primary, &mut model)?;
    let base = model.evaluate_mhc4(
        cfg.visits,
        cfg.temperature,
        cfg.primary.sample_seed(cfg.evaluation_seed),
        false,
    )?;
    let base_stats = base.stats.ok_or("CUDA mHC-4 baseline was not scored")?;
    let base_nlls = base
        .patch_nlls
        .as_ref()
        .ok_or("CUDA mHC-4 baseline patch losses were not returned")?
        .clone();
    model
        .weights
        .reset_history(&backup, &model.module, &model.stream)?;
    let mut state = BoLoop::new(&cfg, base_stats, base_nlls)?;
    if let Some(vp) = val_progress.as_mut() {
        state.validation_records.push(validate_tasks(
            &mut model,
            &cfg.validation,
            &cfg,
            0,
            "initial",
            vp,
        )?);
    }
    let retain_fixed = cfg.training.fixed() && cfg.validation.is_empty();
    for round in 0..cfg.bo_rounds {
        state.step_round(
            &cfg,
            &mut model,
            &mut backup,
            round,
            retain_fixed,
            &mut progress,
            &mut val_progress,
        )?;
    }
    if let Some(vp) = val_progress.as_mut() {
        state.validation_records.push(validate_tasks(
            &mut model,
            &cfg.validation,
            &cfg,
            cfg.bo_rounds,
            "final",
            vp,
        )?);
    }
    finish_bo(
        &cfg,
        &mut model,
        &state,
        &base,
        &base_stats,
        (logical, physical),
        retain_fixed,
        output,
    )
}

fn finish_bo(
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
