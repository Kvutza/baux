//! Portfolio sharing, allocation, controller, and restart policies.

use deser::{Deserialize, Serialize};

/// Policy for admitting observations into regional models and incumbents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum SharingPolicy {
    /// Admit observations inside the region's L_inf hyper-rectangle bounds.
    BoxContain,
    /// Admit observations inside the region's weighted L2 RMS ellipsoid ball.
    EllipsoidBall,
    /// Admit observations whose nearest active region center is this region.
    NearestCenter,
    /// Admit only observations proposed by this region.
    OriginOnly,
}

impl Default for SharingPolicy {
    fn default() -> Self {
        Self::BoxContain
    }
}

impl SharingPolicy {
    /// Route an observation across slots using squared distances to each slot's
    /// center and each slot's squared trust radius `(sq_distance, sq_radius)`.
    pub fn route_distances(
        self,
        source: Option<usize>,
        distances_and_radii2: &[(f64, f64)],
    ) -> Vec<usize> {
        match self {
            Self::OriginOnly => Vec::new(),
            Self::NearestCenter => distances_and_radii2
                .iter()
                .enumerate()
                .min_by(|(a, (x, _)), (b, (y, _))| x.total_cmp(y).then_with(|| a.cmp(b)))
                .map(|(slot, _)| slot)
                .filter(|&nearest| Some(nearest) != source)
                .into_iter()
                .collect(),
            Self::BoxContain | Self::EllipsoidBall => distances_and_radii2
                .iter()
                .enumerate()
                .filter(|&(slot, &(distance, radius2))| Some(slot) != source && distance <= radius2)
                .map(|(slot, _)| slot)
                .collect(),
        }
    }
}

/// Check whether `x[..dim]` lies within axis-aligned box `[lb, ub]`.
pub fn contains_box(dim: usize, x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    (0..dim).all(|d| x[d] >= lb[d] && x[d] <= ub[d])
}

/// Check whether `x[..dim]` lies within the RMS ellipsoid ball inscribed in
/// `[lb, ub]`.
pub fn contains_ball(dim: usize, x: &[f64], lb: &[f64], ub: &[f64]) -> bool {
    let mut sum_sq = 0.0;
    for d in 0..dim {
        let mid = 0.5 * (lb[d] + ub[d]);
        let half = 0.5 * (ub[d] - lb[d]).max(1e-12);
        let z = (x[d] - mid) / half;
        sum_sq += z * z;
    }
    sum_sq <= dim as f64
}

/// How candidates for a new region center are ranked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartCenterSelection {
    /// Choose the first unit-cube draw. Further draws permit matched random
    /// consumption when comparing this rule with another ranking.
    First,
    /// Choose the candidate farthest from its nearest active region center.
    Maximin,
}

/// Bounded candidate search for an automatically retired region slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartCenterPolicy {
    pub candidates: usize,
    pub selection: RestartCenterSelection,
}

impl Default for RestartCenterPolicy {
    fn default() -> Self {
        Self {
            candidates: 1,
            selection: RestartCenterSelection::First,
        }
    }
}

/// Policy used to divide a candidate budget among active region slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationPolicy {
    /// Auer UCB1 bandit over min-max normalized region incumbent rewards.
    IncumbentUcb1,
    /// Favor the least-evaluated active slots, with stable slot-order ties.
    LeastTrials,
    /// Evaluate candidates in all active regions and select top acquisition
    /// winners globally.
    JointAcquisition,
}

impl Default for AllocationPolicy {
    fn default() -> Self {
        Self::IncumbentUcb1
    }
}

/// Policy for regional observations when a slot starts a new generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartMemoryPolicy {
    /// Start with an empty local model, as a new search trajectory.
    Fresh,
    /// Refit the new generation on observations previously relevant to the
    /// slot.
    RetainRelevant,
}

impl Default for RestartMemoryPolicy {
    fn default() -> Self {
        Self::Fresh
    }
}

/// Numerical method used to update a region's trust-region controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionController {
    /// TuRBO-style success and failure counters adapt region length.
    Turbo,
    /// Keep the configured length fixed while retaining relevance and credit.
    FixedRadius,
}

impl Default for RegionController {
    fn default() -> Self {
        Self::Turbo
    }
}
