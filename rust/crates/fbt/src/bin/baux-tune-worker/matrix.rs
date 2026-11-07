//! Matrix experiments report optimizer updates, without BO acceptance metadata.
use super::PretrainRep;
use baux::config::{RunConfig, TrainingOptimizer};
use baux_fbt::ActualBoResult;
use std::path::Path;

pub(super) fn selected(run: &RunConfig) -> bool {
    run.window_tokens.is_some()
        && matches!(
            run.optimizer(),
            TrainingOptimizer::MuonV1 | TrainingOptimizer::AdamwV1
        )
}

pub(super) fn write(
    run: &RunConfig,
    output: &Path,
    rep: u32,
    result: &ActualBoResult,
) -> Result<PretrainRep, String> {
    let optimizer = if run.optimizer() == TrainingOptimizer::MuonV1 {
        "muon"
    } else {
        "adamw"
    };
    let goal_met = result.max_wall_seconds * 1000.0 <= f64::from(run.target_ms());
    result.write_optimization(&output.join("optimization.jsonl"))?;
    result.write_validation(&output.join("validation.cbor"))?;
    std::fs::write(
        output.join("result.toml"),
        format!(
            "status = \"completed\"\nstage = \"fixed_corpus_matrix\"\noptimizer = \"{optimizer}\"\n\
         rounds = {}\nparameters = {}\nmodel_seed = {}\nsampling_seed = {}\n\
         training_schedule = \"repeated-initial-window\"\npersistent_optimizer_state = true\n\
         loop_seconds = {:.9}\nmedian_wall_ms = {:.6}\nmedian_gpu_ms = {:.6}\n\
         max_wall_ms = {:.6}\ntarget_round_ms = {}\nlatency_target_met = {goal_met}\n",
            run.rounds(),
            result.parameters,
            run.model_seeded(rep),
            run.reference_seed(),
            result.loop_seconds,
            result.median_wall_seconds * 1000.0,
            result.median_gpu_seconds * 1000.0,
            result.max_wall_seconds * 1000.0,
            run.target_ms(),
        ),
    )
    .map_err(|error| error.to_string())?;
    Ok(PretrainRep {
        index: rep + 1,
        proposal_seed: run.proposal_seeded(rep),
        acquisition_seed: run.acquisition_seeded(rep),
        median_wall_ms: result.median_wall_seconds * 1000.0,
        median_gpu_ms: result.median_gpu_seconds * 1000.0,
        max_wall_ms: result.max_wall_seconds * 1000.0,
        accepted: result.accepted,
        goal_met,
    })
}
