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
    /// Allocation-free `#![no_std]` routing across slots using
    /// `(sq_distance, sq_radius)` pairs, writing matched slots to `out` and
    /// returning the count written.
    pub fn route_slots(
        self,
        source: Option<usize>,
        distances_and_radii2: &[(f64, f64)],
        out: &mut [usize],
    ) -> usize {
        if out.is_empty() {
            return 0;
        }
        match self {
            Self::OriginOnly => 0,
            Self::NearestCenter => {
                let Some((slot, _)) = distances_and_radii2
                    .iter()
                    .enumerate()
                    .min_by(|(a, (x, _)), (b, (y, _))| x.total_cmp(y).then_with(|| a.cmp(b)))
                else {
                    return 0;
                };
                if Some(slot) != source {
                    out[0] = slot;
                    1
                } else {
                    0
                }
            }
            Self::BoxContain | Self::EllipsoidBall => {
                let mut len = 0usize;
                for (slot, &(distance, radius2)) in distances_and_radii2.iter().enumerate() {
                    if Some(slot) != source && distance <= radius2 {
                        out[len] = slot;
                        len += 1;
                        if len == out.len() {
                            break;
                        }
                    }
                }
                len
            }
        }
    }

    /// Route an observation across slots using squared distances to each slot's
    /// center and each slot's squared trust radius `(sq_distance, sq_radius)`.
    pub fn route_distances(
        self,
        source: Option<usize>,
        distances_and_radii2: &[(f64, f64)],
    ) -> Vec<usize> {
        let mut out = vec![0usize; distances_and_radii2.len()];
        let len = self.route_slots(source, distances_and_radii2, &mut out);
        out.truncate(len);
        out
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

/// `#![no_std]` `RestartCenterSelection::Maximin` selector: returns the
/// candidate index in `0..num_cands` maximizing minimum squared L2 distance to
/// active region centers.
pub fn maximin_idx(
    candidates: &[f64],
    centers: &[f64],
    active: &[bool],
    num_cands: usize,
    num_regions: usize,
    dim: usize,
) -> usize {
    if num_cands == 0 || dim == 0 || candidates.len() < num_cands * dim {
        return 0;
    }
    let regions = num_regions.min(active.len()).min(centers.len() / dim);
    let has_active = (0..regions).any(|r| active[r]);
    if !has_active {
        return 0;
    }
    let mut best_cand = 0usize;
    let mut best_min = f64::NEG_INFINITY;
    for c in 0..num_cands {
        let c_row = &candidates[c * dim..(c + 1) * dim];
        let mut min_d2 = f64::INFINITY;
        for r in 0..regions {
            if !active[r] {
                continue;
            }
            let r_row = &centers[r * dim..(r + 1) * dim];
            let d2 = baux::semantics::neighbor::simd_l2(c_row, r_row);
            if d2 < min_d2 {
                min_d2 = d2;
            }
        }
        if min_d2 > best_min {
            best_min = min_d2;
            best_cand = c;
        }
    }
    best_cand
}

/// `#![no_std]` `RestartMemoryPolicy::RetainRelevant` filter marking which
/// rows of `points` (`num_obs * dim`) fall inside `[lb, ub]` under `sharing`.
pub fn retain_mask(
    points: &[f64],
    lb: &[f64],
    ub: &[f64],
    sharing: SharingPolicy,
    num_obs: usize,
    dim: usize,
    out: &mut [bool],
) -> usize {
    if dim == 0 || lb.len() < dim || ub.len() < dim {
        return 0;
    }
    let n = num_obs.min(out.len()).min(points.len() / dim);
    let mut kept = 0usize;
    for i in 0..n {
        let row = &points[i * dim..(i + 1) * dim];
        let keep = match sharing {
            SharingPolicy::OriginOnly => false,
            SharingPolicy::EllipsoidBall => contains_ball(dim, row, lb, ub),
            SharingPolicy::BoxContain | SharingPolicy::NearestCenter => {
                contains_box(dim, row, lb, ub)
            }
        };
        out[i] = keep;
        if keep {
            kept += 1;
        }
    }
    kept
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
