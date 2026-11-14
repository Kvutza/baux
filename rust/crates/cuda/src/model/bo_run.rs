use super::FbtModel;
use super::bo_report::{finish_bo, record_stage, validate_tasks};
use super::bo_task::{BoConfig, Mhc4Task, condition_seed, learning_checkpoint, validate_task};
use super::readout::Mhc4Evaluation;
use super::weights::WeightBackup;
use crate::capture::CudaResult;
use baux::artifact::json::{Value, json};
use baux::semantics::optimizer::surrogate::{AcquisitionRule, PoolSelection};
use baux::semantics::optimizer::{NeighborValueEstimator, RadiusController, splitmix64};
use baux_cuda_device::ScoreStats;
use std::fs;
use std::io::{BufWriter, Write};

pub(super) fn configure_task(task: &Mhc4Task, model: &mut FbtModel) -> CudaResult<()> {
    if let Some(corpus) = &task.corpus {
        model.configure_causal(&corpus.batch)
    } else {
        model.configure_mhc4(&task.prompt, task.targets.len(), Some(&task.targets))
    }
}

struct RoundProposal {
    round_radius: f32,
    root: u64,
    proposal_seed: u64,
    selected: usize,
    geometry_row: Vec<f64>,
    selection: PoolSelection,
    acquisition_score: f32,
    candidate_mean: f32,
    candidate_se: f32,
    incumbent_mean: f32,
    incumbent_se: f32,
    initializing: bool,
    proposal_ms: f32,
}

pub(super) struct BoLoop {
    pub(super) controller: RadiusController,
    pub(super) history: NeighborValueEstimator,
    current_stats: ScoreStats,
    current_nlls: Vec<f32>,
    pub(super) best_reward: f32,
    pub(super) accepted_rounds: usize,
    pub(super) round_wall: f64,
    pub(super) proposal_ms: f32,
    pub(super) eval_ms: f32,
    pub(super) incumbent_ms: f32,
    pub(super) restore_ms: f32,
    pub(super) candidate_tokens: usize,
    pub(super) incumbent_tokens: usize,
    pub(super) committed_tokens: usize,
    pub(super) total_observations: usize,
    pub(super) observations: Vec<Value>,
    pub(super) stage_profiles: Vec<Value>,
    pub(super) validation_records: Vec<Value>,
    pub(super) candidate: Option<Mhc4Evaluation>,
    pub(super) stats: Option<ScoreStats>,
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
    let n = responses.len() as f64;
    let mean = responses.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let variance = if responses.len() > 1 {
        responses
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean;
                d * d
            })
            .sum::<f64>()
            / (n * (n - 1.0))
    } else {
        0.0
    };
    Ok((responses, mean as f32, variance as f32))
}

impl BoLoop {
    fn new(cfg: &BoConfig, base_stats: ScoreStats, base_nlls: Vec<f32>) -> CudaResult<Self> {
        Ok(Self {
            controller: RadiusController::new(cfg.radius, cfg.radius_min, cfg.radius_max)?,
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
        let r2 = f64::from(round_radius).powi(2);
        let base_slot = self.history.base_slot();
        let base_row = self.history.distance_row(base_slot).unwrap_or(&[]);
        let geometry_f64 = (0..seeds.len())
            .map(|_| {
                (0..self.history.len())
                    .map(|j| {
                        let d_base = base_row.get(j).copied().unwrap_or(0.0);
                        if j == base_slot { r2 } else { d_base + r2 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let incumbent_pred = self.history.incumbent_prediction()?;
        let selection = if geometry_f64.len() == 1 {
            self.history.evaluate_singleton(&geometry_f64[0])?
        } else {
            let round_acquire = cfg
                .acquisition_seed
                .checked_add(round as u64)
                .ok_or("CUDA mHC-4 acquisition seed overflow")?;
            let rule = if cfg.acquisition == "thompson" {
                AcquisitionRule::Thompson {
                    seed: round_acquire,
                }
            } else {
                AcquisitionRule::Ucb { beta: cfg.beta }
            };
            self.history
                .select_pool(&geometry_f64, round as u32, rule, None)?
        };
        let selected = selection.selected;
        let initializing = selection.initializing;
        let cand_pred = selection.candidate;
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
            geometry_row: geometry_f64[selected].clone(),
            selection,
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
        baux::semantics::optimizer::StepDecision,
        f32,
        f32,
        f32,
        f32,
        usize,
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
        let opt_cand = self.history.base_value() + paired_mean;
        let opt_var = paired_var;
        let pred_gain = f64::from(prop.candidate_mean) - f64::from(prop.incumbent_mean);
        let _ = rewarming;
        let decision = if prop.initializing {
            self.controller.initial_evidence(
                f64::from(opt_cand) - f64::from(self.history.base_value()),
                pred_gain,
                0.0,
            )?
        } else {
            self.controller.evaluate_selection(
                opt_cand,
                paired_var,
                prop.round_radius,
                &prop.selection,
            )?
        };
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
            baux::semantics::optimizer::StepDecision,
            f32,
            f32,
            f32,
            f32,
            usize,
            f32,
        ),
    ) -> Value {
        let (decision, opt_cand, opt_var, paired_mean, paired_var, channels, restore_round) =
            outcome;
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
            "phase": if prop.initializing { "initialization" } else { "single_state_bo" },
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
            "agreement_ratio": decision.agreement_ratio,
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
