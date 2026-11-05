//! Persistent native rewards across BO candidates.

use super::decode::Rollout;
use crate::config::{GenerationReward, GenerationTask};
use crate::text::ByteDecoder;
use baux::artifact::json::{Value, json};
use std::path::Path;

pub(super) enum Evaluator {
    Reconstruction(super::reconstruction::Reconstruction),
    Contrast(super::code_contrast::Scorer),
    Execution(super::code_environment::Executor),
}

impl Evaluator {
    pub(super) fn new(
        reward: &GenerationReward,
        tasks: &[GenerationTask],
        tokenizer: Option<&ByteDecoder>,
    ) -> Result<Option<Self>, String> {
        if let GenerationReward::CodeExecution {
            environment,
            interpreter,
            timeout_ms,
        } = reward
        {
            return Ok(Some(Self::Execution(
                super::code_environment::Executor::new(environment, interpreter, *timeout_ms)?,
            )));
        }
        if let GenerationReward::CodeContrastive { max_ngram, .. }
        | GenerationReward::DraftedCode { max_ngram, .. } = reward
        {
            return Ok(Some(Self::Contrast(super::code_contrast::Scorer::new(
                *max_ngram,
            ))));
        }
        if matches!(
            reward,
            GenerationReward::CodeReconstruction | GenerationReward::CodeObjectives { .. }
        ) {
            return Ok(Some(Self::Reconstruction(
                super::reconstruction::Reconstruction::new(
                    tasks,
                    tokenizer.ok_or("code_reconstruction requires the corpus tokenizer")?,
                )?,
            )));
        }
        if matches!(reward, GenerationReward::FrozenQwen { .. }) {
            return Err("frozen_qwen evaluator has been retired; use native Pisa-1 rewards".into());
        }
        Ok(None)
    }

    pub(super) fn score(
        &mut self,
        tokenizer: Option<&ByteDecoder>,
        tasks: &[GenerationTask],
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Execution(_) => {
                Err("execution requires independent objective observations".into())
            }
            Self::Contrast(evaluator) => evaluator.score(
                tokenizer.ok_or("code contrast requires the corpus tokenizer")?,
                tasks,
                rollouts,
                path,
            ),
            Self::Reconstruction(evaluator) => evaluator.score(
                tokenizer.ok_or("code_reconstruction requires the corpus tokenizer")?,
                tasks,
                rollouts,
                path,
            ),
        }
    }

    pub(super) fn audit(
        &mut self,
        tasks: &[GenerationTask],
        tokenizer: Option<&ByteDecoder>,
    ) -> Result<Option<Value>, String> {
        match self {
            Self::Execution(executor) => executor.audit().map(Some),
            Self::Contrast(scorer) => scorer
                .audit(
                    tasks,
                    tokenizer.ok_or("code contrast requires a tokenizer")?,
                )
                .map(Some),
            Self::Reconstruction(_) => Ok(None),
        }
    }

    pub(super) fn execute(
        &mut self,
        tokenizer: &ByteDecoder,
        rollouts: &[Rollout],
        path: &Path,
    ) -> Result<super::code_environment::RewardResult, String> {
        let Self::Execution(executor) = self else {
            return Err("missing execution evaluator".into());
        };
        executor.score(tokenizer, rollouts, path)
    }
}

/// One negative control, not a certificate that the reward measures learning.
pub(super) fn reference_audit(
    generated: &[f32],
    reference: &[f32],
    diagnostics: &[Value],
) -> Result<Value, String> {
    if generated.is_empty()
        || generated.len() != reference.len()
        || generated.len() != diagnostics.len()
        || generated
            .iter()
            .chain(reference)
            .any(|score| !score.is_finite())
    {
        return Err("reward audit requires aligned finite scores and diagnostics".into());
    }
    let comparisons = generated
        .iter()
        .zip(reference)
        .zip(diagnostics)
        .map(|((generated, reference), diagnostics)| {
            json!({
                "generated_reward":generated,
                "corpus_reward":reference,
                "generated_preferred":generated > reference,
                "generated_minus_corpus":f64::from(*generated) - f64::from(*reference),
                "generated_diagnostics":diagnostics,
            })
        })
        .collect::<Vec<_>>();
    let prefers_generated = generated.iter().zip(reference).any(|(g, r)| g > r);
    Ok(json!({
        "schema":"baux.reward_audit.v1",
        "status":if prefers_generated {"generated_preferred_to_corpus"} else {"corpus_not_outscored"},
        "higher_is_better":true,
        "learning_quality_established":false,
        "interpretation":"A generated completion is not automatically a negative control. Inspect its diagnostics and text; this comparison alone cannot certify the objective.",
        "used_for_acceptance":false,
        "comparisons":comparisons,
        "unmeasured_controls":["shuffled_continuation","unrelated_continuation","controlled_repetition"],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_direction() {
        let diagnostics = [json!({"generated_tokens":4096})];
        let audit = reference_audit(&[-0.428], &[-1.614], &diagnostics).unwrap();
        assert_eq!(audit["status"], "generated_preferred_to_corpus");
        assert_eq!(audit["comparisons"][0]["generated_preferred"], true);
        assert_eq!(audit["learning_quality_established"], false);
        let audit = reference_audit(&[-2.0], &[-1.0], &diagnostics).unwrap();
        assert_eq!(audit["status"], "corpus_not_outscored");
        assert_eq!(audit["learning_quality_established"], false);
    }

    #[test]
    fn audit_invalid() {
        assert!(reference_audit(&[], &[], &[]).is_err());
        assert!(reference_audit(&[0.0], &[], &[Value::null()]).is_err());
        assert!(reference_audit(&[f32::NAN], &[0.0], &[Value::null()]).is_err());
    }
}
