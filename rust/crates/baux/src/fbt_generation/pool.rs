use super::gen_rounds::{
    ConditionBaseline, LoopContext, LoopState, RoundCondition, attach_pair, transient,
};
use super::generation::*;
use crate::traits::Oracle;

impl LoopContext<'_> {
    pub(super) fn score_candidate(
        &mut self,
        state: &LoopState,
        row: &metal::Buffer,
        condition: &RoundCondition,
        incumbent: Option<&(ConditionBaseline, f64, bool)>,
        region: usize,
        path: &Path,
        step: u32,
    ) -> Result<Evaluation, String> {
        let incumbent_buffer = state.search.base_buffer();
        let causal = match (self.causal.as_ref(), state.causal_value) {
            (Some(scorer), Some(baseline)) => Some(CausalObjective {
                scorer,
                incumbent: self.weights.row(&incumbent_buffer)?,
                batch: step % scorer.batches(),
                baseline,
            }),
            (None, None) => None,
            _ => return Err("causal pretraining state is incomplete".into()),
        };
        let candidate = crate::search::DeviceView::metal(row.clone(), 0, row.length() as usize);
        let drafts = incumbent
            .map(|(baseline, _, _)| baseline.rollouts.as_slice())
            .unwrap_or_else(|| state.regions.rollouts[region].as_slice());
        let persist = !transient(&condition.config.reward) || self.checkpoint_step(step);
        let mut oracle = FbtOracle {
            runtime: self.runtime,
            decoder: self.decoder,
            verifier: self
                .verifier
                .as_ref()
                .map(|verifier| (verifier, Some(drafts))),
            weights: self.weights,
            config: &condition.config,
            seed: condition.evaluation_seed,
            path,
            evaluator: self.evaluator.as_mut(),
            tokenizer: self.tokenizer,
            audit: condition.config.verify.audit_candidate,
            causal,
            vector_control: self.run.objective_acquisition.is_some(),
            persist,
        };
        let (_, mut evaluation) = oracle.observe(candidate)?;
        if let Some((baseline, gpu_seconds, reused)) = incumbent {
            attach_pair(
                &mut evaluation,
                baseline,
                state.search.best()?,
                state.search.best_variance()?,
                condition.condition_seed,
                condition.evaluation_seed,
                condition.corpus_token_offset,
                *gpu_seconds,
                *reused,
            )?;
        }
        Ok(evaluation)
    }

    pub(super) fn audit_pool(
        &mut self,
        state: &mut LoopState,
        proposal: &crate::resident_metal::Proposals,
        condition: &RoundCondition,
        incumbent: Option<&(ConditionBaseline, f64, bool)>,
        region: usize,
        path: &Path,
        step: u32,
    ) -> Result<Evaluation, String> {
        if incumbent.is_none() {
            return Err("pool rank audit requires a frozen scalar incumbent".into());
        }
        let mut ask = self.policy.ask;
        ask.acquisition.seed = self.acquisition_seed + u64::from(step);
        let root = self.proposal_seed + u64::from(step);
        let selected = proposal.index;
        let order = (0..4)
            .filter(|&candidate| candidate != selected)
            .chain(std::iter::once(selected));
        let mut evaluations: [Option<Evaluation>; 4] = std::array::from_fn(|_| None);
        for candidate in order {
            let row = state
                .search
                .diagnostic_row(proposal, root, ask, candidate)?;
            evaluations[candidate] = Some(self.score_candidate(
                state,
                &row,
                condition,
                incumbent,
                region,
                &path.join(format!("pool-candidate-{candidate}")),
                step,
            )?);
        }
        let improvements = improvements(&evaluations)?;
        let best = improvements
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(candidate, _)| candidate)
            .ok_or("pool rank audit produced no candidates")?;
        let rank = 1 + improvements
            .iter()
            .filter(|&&improvement| improvement > improvements[selected])
            .count();
        let mut report = report(
            proposal,
            condition,
            &evaluations,
            &improvements,
            selected,
            best,
            rank,
        );
        report["point"] = baux::artifact::json::json!(state.pool_rank_audited + 1);
        let selected_evaluation = evaluations[selected]
            .take()
            .ok_or("pool rank audit lost the selected evaluation")?;
        state.pool_rank_audited += 1;
        state.pool_rank_report = Some(report);
        Ok(selected_evaluation)
    }
}

fn improvements(evaluations: &[Option<Evaluation>; 4]) -> Result<Vec<f32>, String> {
    evaluations
        .iter()
        .map(|evaluation| {
            evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.paired)
                .map(|paired| paired.improvement)
                .ok_or_else(|| "pool rank audit lost a paired response".to_string())
        })
        .collect()
}

fn report(
    proposal: &crate::resident_metal::Proposals,
    condition: &RoundCondition,
    evaluations: &[Option<Evaluation>; 4],
    improvements: &[f32],
    selected: usize,
    best: usize,
    rank: usize,
) -> baux::artifact::json::Value {
    let candidates = evaluations
        .iter()
        .enumerate()
        .map(|(candidate, evaluation)| {
            let evaluation = evaluation.as_ref().unwrap();
            let paired = evaluation.paired.unwrap();
            let predicted = proposal.pool_predictions[candidate];
            let response_mean = predicted.0 - proposal.incumbent_mean;
            let response_standard_error = predicted
                .1
                .mul_add(predicted.1, -proposal.incumbent_standard_error.powi(2))
                .max(0.0)
                .sqrt();
            baux::artifact::json::json!({
                "candidate": candidate,
                "selected": candidate == selected,
                "predicted_mean": predicted.0,
                "predicted_standard_error": predicted.1,
                "response_mean": response_mean,
                "response_standard_error": response_standard_error,
                "acquisition_score": predicted.2,
                "conditional_target_nll": evaluation.rollouts.first().and_then(|rollout| rollout.conditional_target_nll),
                "target_quality": evaluation.rollouts.first().and_then(|rollout| rollout.target_quality),
                "improvement": paired.improvement,
                "improvement_variance": paired.improvement_variance,
                "optimizer_value": paired.optimizer_value,
                "generated_tokens": evaluation.rollouts.iter().map(|rollout| rollout.tokens.len()).sum::<usize>(),
                "gpu_seconds": evaluation.rollouts.iter().map(|rollout| rollout.gpu_seconds).sum::<f64>(),
            })
        })
        .collect::<Vec<_>>();
    baux::artifact::json::json!({
        "schema": "baux.pool_rank_audit.v2",
        "objective": "paired_conditional_cross_entropy_improvement",
        "ranking_model": "enn-value-v1",
        "incumbent_mean": proposal.incumbent_mean,
        "incumbent_standard_error": proposal.incumbent_standard_error,
        "condition_seed": condition.condition_seed,
        "sampling_seed": condition.evaluation_seed,
        "corpus_token_offset": condition.corpus_token_offset,
        "selected": selected,
        "best": best,
        "selected_rank": rank,
        "selected_improvement": improvements[selected],
        "best_improvement": improvements[best],
        "selection_regret": improvements[best] - improvements[selected],
        "same_incumbent": true,
        "same_condition": true,
        "same_sampling_stream": true,
        "updates_between_candidates": 0,
        "used_for_selection": false,
        "candidates": candidates,
    })
}

pub(super) fn tell_paired(
    search: &mut crate::resident_metal::SearchState,
    proposal: &crate::resident_metal::Proposals,
    evaluation: &Evaluation,
) -> Result<crate::resident_metal::NoisyDecision, String> {
    let paired = evaluation
        .paired
        .ok_or("paired generation lost its response")?;
    let best = search.best()?;
    let best_variance = search.best_variance()?;
    search.paired_modeled(
        proposal,
        paired.optimizer_value,
        paired.optimizer_variance,
        best,
        best_variance,
        paired.improvement,
        paired.improvement_variance,
    )
}
