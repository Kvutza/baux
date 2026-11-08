//! Observe actual TuRBO retirement and configured replacement at tell time.
//! TuRBO-m specification: Eriksson et al. (NeurIPS 2019, arXiv:1910.01739,
//! Section 2). The controller thresholds in this BAUX diagnostic are configured
//! separately.

use super::{ArmResult, Config, make_optimizer, score};
use baux::artifact::json::json;
use baux::optimizer::{SharingPolicy, TrustRegionState};
use baux::prng;
use ndarray::Array2;

pub(super) fn run_arm(
    config: &Config,
    seed: u64,
    centers: &Array2<f64>,
    sharing: SharingPolicy,
    memory: &str,
    roles: &str,
    center_policy: &str,
    objective: &str,
) -> Result<ArmResult, String> {
    let mut rng = prng(seed);
    let mut optimizer = make_optimizer(
        config,
        centers,
        sharing,
        memory,
        roles,
        center_policy,
        &mut rng,
    )?;
    let mut values = Vec::with_capacity(config.evaluations);
    let mut events = Vec::new();
    let mut first_values = None;
    let mut first_counts = None;
    for start in (0..config.evaluations).step_by(config.regions) {
        let points = optimizer
            .ask(config.regions, &mut rng)
            .map_err(|error| error.to_string())?;
        let before = generations(&optimizer)?;
        let counts = optimizer.model_counts().ok_or("missing local models")?;
        let rewards = score(&points, config.dimensions, objective);
        values.extend(rewards.column(0).iter().copied());
        optimizer
            .tell_variance(&points.view(), &rewards.view(), None, &mut rng)
            .map_err(|error| error.to_string())?;
        let after = generations(&optimizer)?;
        for (slot, (&prior, &current)) in before.iter().zip(&after).enumerate() {
            if current == prior {
                continue;
            }
            if current != prior + 1 {
                return Err("automatic generation jumped by more than one".into());
            }
            if first_values.is_none() {
                first_values = Some(values.clone());
                first_counts = Some(counts.clone());
            }
            let TrustRegionState::Multi { state, .. } = optimizer.trust_region() else {
                unreachable!("host regional strategy");
            };
            events.push(json!({
                "evaluation_count":start + config.regions,
                "slot":slot, "generation_before":prior, "generation_after":current,
                "center_after":state.centers.row(slot).to_vec(),
                "model_counts_before_tell":counts,
                "model_counts_after_tell":optimizer.model_counts()
            }));
        }
    }
    let final_counts = optimizer.model_counts().ok_or("missing local models")?;
    let prefix_values = first_values.unwrap_or_else(|| values.clone());
    let prefix_counts = first_counts.unwrap_or_else(|| final_counts.clone());
    let length = config.length();
    let report = json!({
        "restart_memory":memory, "allocation":config.allocation,
        "roles":roles, "restart_center_policy":center_policy,
        "restart_candidates":config.candidates(),
        "region_acquisition_beta":if roles == "split_beta" { vec![0.0,4.0,2.0,2.0] } else { vec![2.0;4] },
        "controller":"turbo",
        "length_init":length.length_init, "length_min":length.length_min,
        "length_max":length.length_max,
        "success_tolerance":config.succeed(), "failure_tolerance":config.fail(),
        "acquisition":"turbo_enn (RAASP, ENN, per-region UCB beta as reported)",
        "evaluation_count_including_startup":config.evaluations,
        "observation_value_budget_indices":[1, config.evaluations],
        "observation_values":values,
        "automatic_restarts":events,
        "local_model_counts_final":final_counts
    });
    Ok(ArmResult {
        report,
        prefix_values,
        prefix_counts,
    })
}

fn generations(optimizer: &baux::BayesOptimizer) -> Result<Vec<u64>, String> {
    let TrustRegionState::Multi { state, .. } = optimizer.trust_region() else {
        return Err("not a host multi-region optimizer".into());
    };
    (0..state.num_regions())
        .map(|slot| {
            state
                .generation(slot)
                .ok_or("missing regional generation".into())
        })
        .collect()
}
