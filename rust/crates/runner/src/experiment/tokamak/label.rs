//! Optimizer provenance labels and deterministic per-round PRNG helpers.

use baux::config::{AcquisitionSpec, DistanceMetric, ObservationSharing, OptimizerSpec};
use baux::{Prng, prng};

fn sharing_tag(sharing: ObservationSharing) -> &'static str {
    match sharing {
        ObservationSharing::OriginOnly => "origin-only",
        ObservationSharing::BoxContain => "box-contain",
        ObservationSharing::EllipsoidBall => "ellipsoid-ball",
        ObservationSharing::NearestCenter => "nearest-center",
    }
}

fn metric_tag(metric: DistanceMetric) -> &'static str {
    match metric {
        DistanceMetric::Euclidean => "euclidean",
        DistanceMetric::Conformal => "conformal",
        DistanceMetric::Hyperbolic => "hyperbolic",
        DistanceMetric::Wasserstein => "wasserstein",
    }
}

pub fn opt_label(spec: &OptimizerSpec) -> String {
    match spec {
        OptimizerSpec::Ppo => "candle-ppo-metal".into(),
        OptimizerSpec::Muon => "muon".into(),
        OptimizerSpec::Adamw => "adamw".into(),
        OptimizerSpec::OperatorBo {
            surrogate,
            regional_search,
            ..
        } => {
            let count = regional_search.as_ref().map_or(1, |r| r.count);
            format!("operator-bo-r{count}-{}", metric_tag(surrogate.metric))
        }
        OptimizerSpec::BayesianSearch {
            acquisition,
            surrogate,
            regional_search,
            ..
        } => {
            let count = regional_search.as_ref().map_or(1, |r| r.count);
            if matches!(acquisition, AcquisitionSpec::Random) {
                let sh = regional_search
                    .as_ref()
                    .map_or("single", |r| sharing_tag(r.sharing));
                format!("turbo-zero-r{count}-{sh}")
            } else {
                format!("turbo-enn-r{count}-{}", metric_tag(surrogate.metric))
            }
        }
    }
}

pub fn round_rng(seed: u64, round: usize) -> Prng {
    let tag = (round as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    prng(baux::semantics::random::splitmix64(seed ^ tag))
}
