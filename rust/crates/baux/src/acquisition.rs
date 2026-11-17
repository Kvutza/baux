//! Acquisition function optimizers for TuRBO.

use ndarray::{ArrayView1, ArrayView2};
use rand::Rng;
use thiserror::Error;

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use baux::semantics::optimizer::surrogate::BetterScore;

use crate::hash::{AffineKey, Key};

#[derive(Clone, Copy, PartialEq, Eq)]
struct MinHeapItem {
    score: BetterScore<f64>,
    tie: u64,
    index: usize,
}

impl MinHeapItem {
    #[inline(always)]
    fn from_score(score: f64, tie: u64, index: usize) -> Self {
        Self {
            score: BetterScore::new(score),
            tie,
            index,
        }
    }

    #[inline(always)]
    fn better_than(&self, other: &Self) -> bool {
        self.score
            .cmp(&other.score)
            .then_with(|| self.tie.cmp(&other.tie))
            .then_with(|| other.index.cmp(&self.index))
            .is_gt()
    }
}

impl PartialOrd for MinHeapItem {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapItem {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order so BinaryHeap acts as a Min-Heap (worst of top-k on
        // top).
        other
            .score
            .cmp(&self.score)
            .then_with(|| other.tie.cmp(&self.tie))
            .then_with(|| self.index.cmp(&other.index))
    }
}

/// Errors in acquisition optimization.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AcquisitionError {
    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Empty candidate set.
    #[error("No candidates available")]
    NoCandidates,
}

/// Result of acquisition optimization.
pub type AcquisitionResult = Result<Vec<usize>, AcquisitionError>;

/// UCB (Upper Confidence Bound) acquisition optimizer parameterized by
/// [` crate::sense::ObjectiveSense `].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UCBAcquisition<O: crate::sense::ObjectiveSense = crate::sense::Sense> {
    /// Beta parameter (exploration weight).
    beta: f64,
    sense: O,
}

impl UCBAcquisition<crate::sense::Sense> {
    /// Canonical default exploration weight (`beta = 1.0`).
    pub const DEFAULT_BETA: f64 = 1.0;

    /// Create new UCB acquisition optimizer (defaults to maximization).
    pub const fn new(beta: f64) -> Self {
        Self {
            beta,
            sense: crate::sense::Sense::Maximize,
        }
    }
}

impl Default for UCBAcquisition<crate::sense::Sense> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(Self::DEFAULT_BETA)
    }
}

impl<O: crate::sense::ObjectiveSense> UCBAcquisition<O> {
    /// Create a UCB acquisition optimizer with an explicit
    /// [`crate::sense::ObjectiveSense`].
    pub const fn with_sense(beta: f64, sense: O) -> Self {
        Self { beta, sense }
    }

    #[inline(always)]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    #[inline(always)]
    pub const fn sense(&self) -> O
    where
        O: Copy,
    {
        self.sense
    }

    /// Evaluate the UCB score at a single `(mu, sigma)` candidate.
    #[inline(always)]
    pub fn score_at(&self, mu: f64, sigma: f64) -> f64 {
        self.sense.ucb(mu, sigma, self.beta).raw()
    }

    /// Select arms using UCB acquisition into a reusable 64-byte-aligned
    /// [`crate::scratch::CandidateSoa`] workspace.
    pub fn select_soa(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        soa: &mut crate::scratch::CandidateSoa,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }
        if mu.len() != sigma.len() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu length {} does not match sigma length {}",
                mu.len(),
                sigma.len()
            )));
        }
        if num_arms == 0 {
            return Ok(Vec::new());
        }
        Ok(soa.select_topk(mu, sigma, self.beta, num_arms, self.sense))
    }

    /// Pure deterministic UCB selection with canonical index-order
    /// tie-breaking.
    pub fn select_pure(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }
        if mu.len() != sigma.len() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu length {} does not match sigma length {}",
                mu.len(),
                sigma.len()
            )));
        }
        if num_arms == 0 {
            return Ok(Vec::new());
        }

        let k = num_arms.min(mu.len());
        if k == 1 {
            let mut best = MinHeapItem::from_score(self.score_at(mu[0], sigma[0]), 0, 0);
            for i in 1..mu.len() {
                let item = MinHeapItem::from_score(self.score_at(mu[i], sigma[i]), 0, i);
                if item.better_than(&best) {
                    best = item;
                }
            }
            return Ok(vec![best.index]);
        }

        let mut heap = BinaryHeap::with_capacity(k);
        for i in 0..mu.len() {
            let item = MinHeapItem::from_score(self.score_at(mu[i], sigma[i]), 0, i);
            if heap.len() < k {
                heap.push(item);
            } else if let Some(min_item) = heap.peek()
                && item.better_than(min_item)
            {
                heap.pop();
                heap.push(item);
            }
        }

        let mut result: Vec<_> = heap.into_iter().collect();
        result.sort_unstable();
        Ok(result.into_iter().map(|item| item.index).collect())
    }

    /// Pure functional UCB selection breaking exact score ties via `Key::fold`.
    pub fn select_key(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        key: Key,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }
        if mu.len() != sigma.len() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu length {} does not match sigma length {}",
                mu.len(),
                sigma.len()
            )));
        }
        if num_arms == 0 {
            return Ok(Vec::new());
        }

        let k = num_arms.min(mu.len());
        let mut heap = BinaryHeap::with_capacity(k);
        for i in 0..mu.len() {
            let tie = key.fold(i as u64).seed();
            let item = MinHeapItem::from_score(self.score_at(mu[i], sigma[i]), tie, i);
            if heap.len() < k {
                heap.push(item);
            } else if let Some(min_item) = heap.peek()
                && item.better_than(min_item)
            {
                heap.pop();
                heap.push(item);
            }
        }

        let mut result: Vec<_> = heap.into_iter().collect();
        result.sort_unstable();
        Ok(result.into_iter().map(|item| item.index).collect())
    }

    /// Affine-key UCB selection advancing the single-use `AffineKey`.
    pub fn select_affine(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(Vec<usize>, AffineKey), AcquisitionError> {
        let (draw, next) = key.step();
        let arms = self.select_key(mu, sigma, num_arms, draw)?;
        Ok((arms, next))
    }

    /// Select arms using UCB acquisition.
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        _rng: &mut R,
    ) -> AcquisitionResult {
        self.select_pure(mu, sigma, num_arms)
    }
}

/// Independent Gaussian marginal sampling from supplied means and standard
/// errors.
///
/// This helper has no neighbor information and cannot preserve ENN posterior
/// correlations. The optimizer uses `ENN::posterior_draw` for that purpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThompsonAcquisition;

impl ThompsonAcquisition {
    /// Create new Thompson sampling acquisition optimizer.
    pub const fn new() -> Self {
        Self
    }

    /// Stateless marginal Gaussian Thompson score at candidate `index` under
    /// `key` via shared `#![no_std]`
    /// [`baux::semantics::optimizer::surrogate::score_thompson`].
    #[inline(always)]
    pub fn score_at(mu: f64, sigma: f64, key: Key, index: usize) -> f64 {
        baux::semantics::optimizer::surrogate::score_thompson(key, index, mu, sigma)
    }

    /// Pure functional Thompson selection driven by an immutable `Key`.
    pub fn select_key(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        key: Key,
    ) -> AcquisitionResult {
        if mu.is_empty() {
            return Err(AcquisitionError::NoCandidates);
        }
        if mu.len() != sigma.len() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu length {} does not match sigma length {}",
                mu.len(),
                sigma.len()
            )));
        }
        if num_arms == 0 {
            return Ok(Vec::new());
        }

        let k = num_arms.min(mu.len());
        let mut heap = BinaryHeap::with_capacity(k);
        for i in 0..mu.len() {
            let score = Self::score_at(mu[i], sigma[i], key, i);
            let tie = key.fold(i as u64).seed();
            let item = MinHeapItem::from_score(score, tie, i);
            if heap.len() < k {
                heap.push(item);
            } else if let Some(min_item) = heap.peek()
                && item.better_than(min_item)
            {
                heap.pop();
                heap.push(item);
            }
        }

        let mut result: Vec<_> = heap.into_iter().collect();
        result.sort_unstable();
        Ok(result.into_iter().map(|item| item.index).collect())
    }

    /// Affine-key Thompson selection advancing the single-use `AffineKey`.
    pub fn select_affine(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(Vec<usize>, AffineKey), AcquisitionError> {
        let (draw, next) = key.step();
        let arms = self.select_key(mu, sigma, num_arms, draw)?;
        Ok((arms, next))
    }

    /// Select arms using Thompson sampling (thin `&mut Rng` adapter over
    /// `select_key`).
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        let key = Key::from_pair(rng.next_u64(), rng.next_u64());
        self.select_key(mu, sigma, num_arms, key)
    }
}

impl Default for ThompsonAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

/// Random acquisition optimizer (baseline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomAcquisition;

impl RandomAcquisition {
    /// Create new random acquisition optimizer.
    pub const fn new() -> Self {
        Self
    }

    /// Pure `O(num_arms)` uniform random subset selection without materializing
    /// `0..n_candidates` (Floyd subset + Fisher-Yates order over `Key::fold`).
    pub fn select_key(&self, n_candidates: usize, num_arms: usize, key: Key) -> AcquisitionResult {
        if n_candidates == 0 {
            return Err(AcquisitionError::NoCandidates);
        }
        if num_arms == 0 {
            return Ok(Vec::new());
        }

        let k = num_arms.min(n_candidates);
        let mut out = Vec::with_capacity(k);
        let mut seen = HashSet::with_capacity(k);
        for j in (n_candidates - k)..n_candidates {
            let word = key.fold(j as u64).seed();
            let pick = ((u128::from(word) * (j as u128 + 1)) >> 64) as usize;
            let chosen = if seen.insert(pick) {
                pick
            } else {
                seen.insert(j);
                j
            };
            out.push(chosen);
        }
        let perm_key = key.fold(0x517C_C1B7_2722_0A95);
        for i in (1..out.len()).rev() {
            let word = perm_key.fold(i as u64).seed();
            let swap = ((u128::from(word) * (i as u128 + 1)) >> 64) as usize;
            out.swap(i, swap);
        }
        Ok(out)
    }

    /// Affine-key uniform random arm selection.
    pub fn select_affine(
        &self,
        n_candidates: usize,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(Vec<usize>, AffineKey), AcquisitionError> {
        let (draw, next) = key.step();
        let arms = self.select_key(n_candidates, num_arms, draw)?;
        Ok((arms, next))
    }

    /// Select random arms (thin `&mut Rng` adapter over `select_key`).
    pub fn select<R: Rng + ?Sized>(
        &self,
        n_candidates: usize,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        let key = Key::from_pair(rng.next_u64(), rng.next_u64());
        self.select_key(n_candidates, num_arms, key)
    }
}

impl Default for RandomAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

/// Pareto front acquisition optimizer for multi-objective problems.
///
/// Uses non-dominated sorting across predictive objectives, breaking ties
/// within an overflowing front by exact exclusive hypervolume contribution (for
/// up to 4 objectives and moderate front sizes), NSGA-II crowding distance,
/// predictive uncertainty `se`, and deterministic `Key::fold` tie-breaking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParetoAcquisition;

impl ParetoAcquisition {
    /// Create new Pareto acquisition optimizer.
    pub const fn new() -> Self {
        Self
    }

    /// Pure functional Pareto front selection driven by an immutable `Key`.
    pub fn select_key(
        &self,
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        num_arms: usize,
        key: Key,
    ) -> AcquisitionResult {
        self.select_inner(mu, se, num_arms, key, key.seed())
    }

    /// Affine-key Pareto front selection advancing the single-use `AffineKey`.
    pub fn select_affine(
        &self,
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(Vec<usize>, AffineKey), AcquisitionError> {
        let (draw, next) = key.step();
        let arms = self.select_key(mu, se, num_arms, draw)?;
        Ok((arms, next))
    }

    /// Select arms from Pareto front (thin `&mut Rng` adapter over pure core).
    pub fn select<R: Rng + ?Sized>(
        &self,
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        num_arms: usize,
        rng: &mut R,
    ) -> AcquisitionResult {
        let seed = rng.next_u64();
        let key = Key::new(seed);
        self.select_inner(mu, se, num_arms, key, seed)
    }

    fn select_inner(
        &self,
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        num_arms: usize,
        key: Key,
        seed_1d: u64,
    ) -> AcquisitionResult {
        let n_candidates = mu.nrows();
        let n_objectives = mu.ncols();

        if n_candidates == 0 {
            return Err(AcquisitionError::NoCandidates);
        }

        if num_arms == 0 {
            return Ok(Vec::new());
        }

        if mu.shape() != se.shape() {
            return Err(AcquisitionError::InvalidParameter(format!(
                "mu shape {:?} does not match se shape {:?}",
                mu.shape(),
                se.shape()
            )));
        }

        if n_objectives == 1 {
            return Self::pareto_seed(&mu.column(0), &se.column(0), num_arms, seed_1d);
        }

        let pareto_fronts = self.nondomin_sort(mu);
        let mut selected = Vec::with_capacity(num_arms.min(n_candidates));
        for (front_idx, front) in pareto_fronts.into_iter().enumerate() {
            if selected.len() >= num_arms {
                break;
            }

            let remaining = num_arms - selected.len();
            if front.len() <= remaining {
                selected.extend(front);
            } else {
                let chosen =
                    Self::truncate_front(&front, mu, se, remaining, key.fold(front_idx as u64));
                selected.extend(chosen);
            }
        }

        Ok(selected)
    }

    /// Compute NSGA-II crowding distances for candidates in `front`.
    pub fn crowding_dist(front: &[usize], objectives: &ArrayView2<f64>) -> Vec<f64> {
        let k = front.len();
        let m = objectives.ncols();
        if k == 0 {
            return Vec::new();
        }
        if k <= 2 || m == 0 {
            return vec![f64::INFINITY; k];
        }

        let mut distances = vec![0.0_f64; k];
        let mut order: Vec<usize> = (0..k).collect();

        for obj in 0..m {
            order.sort_by(|&a, &b| {
                BetterScore::new(objectives[[front[a], obj]])
                    .cmp(&BetterScore::new(objectives[[front[b], obj]]))
                    .then_with(|| front[a].cmp(&front[b]))
            });
            let low = objectives[[front[order[0]], obj]];
            let high = objectives[[front[order[k - 1]], obj]];
            let span = high - low;
            if !span.is_finite() || span <= 0.0 {
                continue;
            }
            distances[order[0]] = f64::INFINITY;
            distances[order[k - 1]] = f64::INFINITY;
            for pos in 1..(k - 1) {
                if distances[order[pos]].is_finite() {
                    let prev = objectives[[front[order[pos - 1]], obj]];
                    let next = objectives[[front[order[pos + 1]], obj]];
                    distances[order[pos]] += (next - prev) / span;
                }
            }
        }

        distances
    }

    fn truncate_front(
        front: &[usize],
        mu: &ArrayView2<f64>,
        se: &ArrayView2<f64>,
        remaining: usize,
        key: Key,
    ) -> Vec<usize> {
        let k = front.len();
        if remaining >= k {
            return front.to_vec();
        }
        let m = mu.ncols();
        let crowd = Self::crowding_dist(front, mu);

        let hv = if m <= 4 && k <= 64 {
            let mut flat = Vec::with_capacity(k * m);
            let mut mins = vec![f64::INFINITY; m];
            let mut maxs = vec![f64::NEG_INFINITY; m];
            for &idx in front {
                for obj in 0..m {
                    let ucb = mu[[idx, obj]] + se[[idx, obj]].max(0.0);
                    let val = if ucb.is_finite() { ucb } else { 0.0 };
                    flat.push(val);
                    mins[obj] = mins[obj].min(val);
                    maxs[obj] = maxs[obj].max(val);
                }
            }
            let ref_pt: Vec<f64> = (0..m)
                .map(|obj| {
                    let span = (maxs[obj] - mins[obj]).max(1.0);
                    mins[obj] - 0.1 * span
                })
                .collect();
            let y_view = ArrayView2::from_shape((k, m), &flat).expect("k x m shape");
            let ref_view = ArrayView1::from(&ref_pt);
            crate::hypervolume::hv_contributions(&y_view, &ref_view)
                .unwrap_or_else(|_| vec![0.0; k])
        } else {
            vec![0.0; k]
        };

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            let idx_a = front[a];
            let idx_b = front[b];
            let clean_a = (0..m).all(|obj| !mu[[idx_a, obj]].is_nan());
            let clean_b = (0..m).all(|obj| !mu[[idx_b, obj]].is_nan());
            let se_a: f64 = (0..m).map(|obj| se[[idx_a, obj]].max(0.0)).sum();
            let se_b: f64 = (0..m).map(|obj| se[[idx_b, obj]].max(0.0)).sum();
            let tie_a = key.fold(idx_a as u64).seed();
            let tie_b = key.fold(idx_b as u64).seed();
            clean_b
                .cmp(&clean_a)
                .then_with(|| BetterScore::new(hv[b]).cmp(&BetterScore::new(hv[a])))
                .then_with(|| BetterScore::new(crowd[b]).cmp(&BetterScore::new(crowd[a])))
                .then_with(|| BetterScore::new(se_b).cmp(&BetterScore::new(se_a)))
                .then_with(|| tie_b.cmp(&tie_a))
                .then_with(|| idx_a.cmp(&idx_b))
        });

        order
            .into_iter()
            .take(remaining)
            .map(|pos| front[pos])
            .collect()
    }

    /// Non-dominated sorting across candidate rows (maximization in all
    /// columns).
    pub fn nondomin_sort(&self, objectives: &ArrayView2<f64>) -> Vec<Vec<usize>> {
        if objectives.ncols() == 2 && !objectives.iter().any(|v| v.is_nan()) {
            return self.nondomin_2d(objectives);
        }

        let n = objectives.nrows();
        if n == 0 {
            return Vec::new();
        }

        // Domination counts and dominated set
        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        // Compute domination
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                if dominates(objectives, i, j) {
                    dominated[i].push(j);
                } else if dominates(objectives, j, i) {
                    domination_count[i] += 1;
                }
            }
        }

        // First front: points with domination count 0
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front.is_empty() {
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }

        fronts
    }

    /// Non-dominated sorting specialized for 2 objectives.
    ///
    /// Uses NaN-safe skyline peeling with objective-0 sort + objective-1 sweep
    /// via [`BetterScore`], guaranteeing finite termination (`<= n` passes)
    /// even when `-Infinity` or `NaN` entries are present.
    fn nondomin_2d(&self, objectives: &ArrayView2<f64>) -> Vec<Vec<usize>> {
        let n = objectives.nrows();
        if n == 0 {
            return Vec::new();
        }

        let mut remaining: Vec<usize> = (0..n).collect();
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        while !remaining.is_empty() {
            remaining.sort_by(|&i, &j| {
                BetterScore::new(objectives[[j, 0]])
                    .cmp(&BetterScore::new(objectives[[i, 0]]))
                    .then_with(|| {
                        BetterScore::new(objectives[[j, 1]])
                            .cmp(&BetterScore::new(objectives[[i, 1]]))
                    })
                    .then_with(|| i.cmp(&j))
            });

            let mut front: Vec<usize> = Vec::new();
            let mut best_obj1: Option<BetterScore<f64>> = None;
            let mut last_pair: Option<(BetterScore<f64>, BetterScore<f64>)> = None;

            for &idx in &remaining {
                let obj0 = BetterScore::new(objectives[[idx, 0]]);
                let obj1 = BetterScore::new(objectives[[idx, 1]]);
                if best_obj1.is_none_or(|best| obj1 > best) {
                    front.push(idx);
                    best_obj1 = Some(obj1);
                    last_pair = Some((obj0, obj1));
                } else if Some((obj0, obj1)) == last_pair {
                    front.push(idx);
                }
            }

            let mut in_front = vec![false; n];
            for &idx in &front {
                in_front[idx] = true;
            }
            remaining.retain(|&idx| !in_front[idx]);
            fronts.push(front);
        }

        fronts
    }

    /// Select arms from 2D `(mu, se)` Pareto fronts for single-objective inputs
    /// without materializing intermediate arrays.
    fn pareto_seed(
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        num_arms: usize,
        seed: u64,
    ) -> AcquisitionResult {
        let n = mu.len();
        if n == 0 {
            return Err(AcquisitionError::NoCandidates);
        }
        let k = num_arms.min(n);
        if mu.iter().all(|v| v.is_finite()) && sigma.iter().all(|v| v.is_finite()) {
            let empty = ArrayView2::from_shape((0, 0), &[]).expect("0x0 view is valid");
            return Ok(crate::util::pareto_arms(&empty, mu, sigma, k, seed));
        }
        let pair =
            ndarray::Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { mu[i] } else { sigma[i] });
        let zero_se = ndarray::Array2::zeros((n, 2));
        ParetoAcquisition.select_key(&pair.view(), &zero_se.view(), k, Key::new(seed))
    }
}

fn dominates(objectives: &ArrayView2<f64>, left: usize, right: usize) -> bool {
    let l_row = objectives.row(left);
    let r_row = objectives.row(right);
    if let (Some(l_slice), Some(r_slice)) = (l_row.as_slice(), r_row.as_slice()) {
        return baux::semantics::morbo::dominates(l_slice, r_slice);
    }
    let cols = objectives.ncols();
    if cols == 0 {
        return false;
    }
    let mut strictly_better = false;
    for objective in 0..cols {
        let a = objectives[[left, objective]];
        let b = objectives[[right, objective]];
        if a.is_nan() {
            return false;
        }
        if b.is_nan() {
            strictly_better = true;
            continue;
        }
        let ordering = a.total_cmp(&b);
        if ordering.is_lt() {
            return false;
        }
        strictly_better |= ordering.is_gt();
    }
    strictly_better
}

impl Default for ParetoAcquisition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_001() {
        let ucb = UCBAcquisition::new(1.0);
        let mu = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = ucb.select(&mu.view(), &sigma.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        // Highest UCB should be selected (with random tie-breaking)
        assert!(selected.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_002() {
        let thompson = ThompsonAcquisition::new();
        let mu = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigma = array![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = thompson
            .select(&mu.view(), &sigma.view(), 3, &mut rng)
            .unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_003() {
        let random = RandomAcquisition::new();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = random.select(10, 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.iter().all(|&i| i < 10));
    }

    #[test]
    fn test_004() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let se = array![[0.1], [0.2], [0.3], [0.4], [0.5]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_005() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0], [1.0], [2.0]];
        let se = array![[0.01], [10.0], [1.5]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 1, &mut rng).unwrap();

        assert_eq!(selected, vec![2]);
    }

    #[test]
    fn test_006() {
        let pareto = ParetoAcquisition::new();
        let mu = array![[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [1.5, 1.5], [3.0, 3.0]];
        let se = array![[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = pareto.select(&mu.view(), &se.view(), 3, &mut rng).unwrap();

        assert_eq!(selected.len(), 3);
        assert!(selected.contains(&4));
    }

    #[test]
    fn test_007() {
        let pareto = ParetoAcquisition::new();
        let objectives = array![[3.0, 1.0], [2.0, 2.0], [1.0, 3.0], [0.5, 0.5]];
        let fronts = pareto.nondomin_sort(&objectives.view());
        assert_eq!(fronts.len(), 2);
        assert_eq!(fronts[0], vec![0, 1, 2]);
        assert_eq!(fronts[1], vec![3]);
    }

    #[test]
    fn test_008() {
        let pareto = ParetoAcquisition::new();
        // Points 0 and 1 are identical and both should be on first front.
        let objectives = array![[2.0, 2.0], [2.0, 2.0], [1.0, 1.0], [0.0, 3.0]];
        let fronts = pareto.nondomin_sort(&objectives.view());
        assert!(!fronts.is_empty());
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        assert!(fronts[0].contains(&3));
    }

    #[test]
    fn test_009() {
        let ucb = UCBAcquisition::new(1.0);
        let mu = array![];
        let sigma = array![];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = ucb.select(&mu.view(), &sigma.view(), 3, &mut rng);

        assert!(matches!(result, Err(AcquisitionError::NoCandidates)));
    }

    #[test]
    fn test_010() {
        let ucb = UCBAcquisition::new(1.0);
        let mu = array![1.0, 2.0, 3.0];
        let sigma = array![0.1, 0.2, 0.3];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let selected = ucb.select(&mu.view(), &sigma.view(), 0, &mut rng).unwrap();

        assert!(selected.is_empty());
    }

    #[test]
    fn pure_keys() {
        let ucb = UCBAcquisition::default();
        assert_eq!(ucb.beta(), 1.0);

        let mu = array![1.0, 2.0, 2.0, 5.0];
        let sigma = array![0.5, 0.5, 0.5, 0.1];
        let key = Key::new(123);
        let pure_sel = ucb.select_pure(&mu.view(), &sigma.view(), 2).unwrap();
        assert_eq!(pure_sel[0], 3);
        let (aff_sel, next_key) = ucb
            .select_affine(&mu.view(), &sigma.view(), 2, AffineKey::from_key(key))
            .unwrap();
        assert_eq!(aff_sel[0], 3);

        let ts = ThompsonAcquisition::new();
        let ts_a = ts.select_key(&mu.view(), &sigma.view(), 2, key).unwrap();
        let ts_b = ts.select_key(&mu.view(), &sigma.view(), 2, key).unwrap();
        assert_eq!(ts_a, ts_b);

        let rnd = RandomAcquisition::new();
        let (rnd_sel, _after) = rnd.select_affine(100_000, 4, next_key).unwrap();
        assert_eq!(rnd_sel.len(), 4);
        let unique: HashSet<usize> = rnd_sel.iter().copied().collect();
        assert_eq!(unique.len(), 4);

        let pareto = ParetoAcquisition::new();
        let mo_mu = array![[4.0, 0.0], [3.5, 0.5], [2.0, 2.0], [0.0, 4.0], [1.0, 1.0]];
        let mo_se = array![[0.1, 0.1], [0.1, 0.1], [0.2, 0.2], [0.1, 0.1], [0.1, 0.1]];
        let p_a = pareto
            .select_key(&mo_mu.view(), &mo_se.view(), 3, key)
            .unwrap();
        let p_b = pareto
            .select_key(&mo_mu.view(), &mo_se.view(), 3, key)
            .unwrap();
        assert_eq!(p_a, p_b);
        assert_eq!(p_a.len(), 3);
        assert!(!p_a.contains(&4));
        let crowd = ParetoAcquisition::crowding_dist(&[0, 1, 2, 3], &mo_mu.view());
        assert!(crowd[0].is_infinite() && crowd[3].is_infinite());
        assert!(crowd[2] > crowd[1]);
    }

    #[test]
    fn nonfinite_edges() {
        let pos_nan = f64::from_bits(0x7ff8_0000_0000_0000);
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        let key = Key::new(77);

        let ucb = UCBAcquisition::default();
        let mu = array![pos_nan, f64::NEG_INFINITY, -10.0, neg_nan];
        let sigma = array![1.0, 0.1, 0.2, 1.0];
        assert_eq!(
            ucb.select_pure(&mu.view(), &sigma.view(), 1).unwrap(),
            vec![2]
        );
        assert_eq!(
            ucb.select_pure(&mu.view(), &sigma.view(), 2).unwrap(),
            vec![2, 1]
        );
        assert_eq!(
            ucb.select_key(&mu.view(), &sigma.view(), 2, key).unwrap(),
            vec![2, 1]
        );
        let only_inf = array![pos_nan, f64::NEG_INFINITY];
        let only_sig = array![1.0, 0.0];
        assert_eq!(
            ucb.select_pure(&only_inf.view(), &only_sig.view(), 1)
                .unwrap(),
            vec![1]
        );

        let ts = ThompsonAcquisition::new();
        assert_eq!(
            ts.select_key(&mu.view(), &sigma.view(), 2, key).unwrap(),
            vec![2, 1]
        );

        let pareto = ParetoAcquisition::new();
        let obj2 = array![
            [pos_nan, 2.0],
            [1.0, f64::NEG_INFINITY],
            [f64::NEG_INFINITY, f64::NEG_INFINITY],
            [2.0, 2.0],
            [neg_nan, pos_nan]
        ];
        let se2 = array![[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]];
        let fronts2 = pareto.nondomin_sort(&obj2.view());
        assert_eq!(fronts2[0], vec![3]);
        assert_eq!(fronts2.iter().map(Vec::len).sum::<usize>(), obj2.nrows());
        let sel2 = pareto
            .select_key(&obj2.view(), &se2.view(), 2, key)
            .unwrap();
        assert_eq!(sel2[0], 3);

        // Verify -Infinity skyline termination in 2D without NaN and verify NaN
        // never dominates finite rows even when its other coordinate is large.
        let inf2 = array![
            [f64::NEG_INFINITY, f64::NEG_INFINITY],
            [1.0, f64::NEG_INFINITY],
            [f64::NEG_INFINITY, 1.0]
        ];
        let fronts_inf = pareto.nondomin_sort(&inf2.view());
        assert_eq!(fronts_inf, vec![vec![1, 2], vec![0]]);

        let nan_vs_fin = array![[pos_nan, 100.0], [0.0, 0.0]];
        let nan_se = array![[0.1, 0.1], [0.1, 0.1]];
        let fronts_nf = pareto.nondomin_sort(&nan_vs_fin.view());
        assert!(fronts_nf[0].contains(&1));
        assert_eq!(
            pareto
                .select_key(&nan_vs_fin.view(), &nan_se.view(), 1, key)
                .unwrap(),
            vec![1]
        );

        let obj3 = array![
            [pos_nan, 2.0, 2.0],
            [1.0, 1.0, f64::NEG_INFINITY],
            [2.0, 2.0, 2.0],
            [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]
        ];
        let se3 = array![
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1]
        ];
        let fronts3 = pareto.nondomin_sort(&obj3.view());
        assert_eq!(fronts3[0], vec![2]);
        assert_eq!(fronts3.iter().map(Vec::len).sum::<usize>(), obj3.nrows());
        let sel3 = pareto
            .select_key(&obj3.view(), &se3.view(), 1, key)
            .unwrap();
        assert_eq!(sel3, vec![2]);
    }
}
