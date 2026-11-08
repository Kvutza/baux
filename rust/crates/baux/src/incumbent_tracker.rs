//! Incremental incumbent candidate tracker parameterized over
//! [`ObjectiveSense`].
//!
//! Replaces boolean state flags with an algebraic sum type (`TrackerMode`) and
//! maintains top-$m$ candidates in a bounded min-heap
//! (`BinaryHeap<WorstFirst>`) with niche-packed [`ObsId`] indices.

use core::cmp::Ordering;
use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::{BTreeSet, BinaryHeap};

use crate::sense::{BetterScore, Maximize, Minimize, ObjectiveSense, ObsId, Sense, Target};

const ALL_THRESHOLD: usize = 1_000_000_000;

/// Heap entry ordered so the *worst* retained candidate sits at the top of a
/// [`BinaryHeap`] (smallest [`BetterScore`], breaking ties by largest
/// [`ObsId`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WorstFirst {
    score: BetterScore,
    id: ObsId,
}

impl WorstFirst {
    #[inline(always)]
    fn is_better(id: ObsId, score: BetterScore, worst: Self) -> bool {
        match score.cmp(&worst.score) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => id < worst.id,
        }
    }
}

impl PartialOrd for WorstFirst {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorstFirst {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Algebraic state machine for incumbent candidate tracking.
#[derive(Clone, Debug)]
enum TrackerMode {
    /// Retain all observation indices ($m \ge 10^9$).
    AllCandidates { indices: Vec<ObsId> },
    /// Retain exact ties at the best noiseless scalar observation.
    NoiselessBest {
        best: Option<(BetterScore, Vec<ObsId>)>,
    },
    /// Retain the top-$m$ noisy scalar candidates in $O(\log m)$ per update.
    ScalarTopM {
        m: usize,
        heap: BinaryHeap<WorstFirst>,
    },
    /// Retain the union of top-$m$ candidates across $K$ objective columns.
    MultiTopM {
        m: usize,
        heaps: Vec<BinaryHeap<WorstFirst>>,
    },
}

impl TrackerMode {
    fn new(m: usize, noise_aware: bool, num_metrics: usize) -> Self {
        if m >= ALL_THRESHOLD {
            Self::AllCandidates {
                indices: Vec::new(),
            }
        } else if num_metrics == 1 && !noise_aware {
            Self::NoiselessBest { best: None }
        } else if num_metrics == 1 {
            Self::ScalarTopM {
                m,
                heap: BinaryHeap::with_capacity(m),
            }
        } else {
            Self::MultiTopM {
                m,
                heaps: (0..num_metrics)
                    .map(|_| BinaryHeap::with_capacity(m))
                    .collect(),
            }
        }
    }

    fn reset(&mut self) {
        match self {
            Self::AllCandidates { indices } => indices.clear(),
            Self::NoiselessBest { best } => *best = None,
            Self::ScalarTopM { heap, .. } => heap.clear(),
            Self::MultiTopM { heaps, .. } => {
                for heap in heaps {
                    heap.clear();
                }
            }
        }
    }
}

/// Sense-parametric incremental incumbent tracker.
#[derive(Clone, Debug)]
pub struct GenericTracker<O: ObjectiveSense = Sense> {
    observation_count: usize,
    mode: TrackerMode,
    sense: O,
}

/// Default runtime-configurable incumbent tracker.
pub type IncumbentTracker = GenericTracker<Sense>;
/// Compile-time maximizing incumbent tracker.
pub type MaxTracker = GenericTracker<Maximize>;
/// Compile-time minimizing incumbent tracker.
pub type MinTracker = GenericTracker<Minimize>;
/// Compile-time target-matching incumbent tracker.
pub type TargetTracker = GenericTracker<Target>;

impl IncumbentTracker {
    pub fn new(m: usize, noise_aware: bool, num_metrics: usize) -> Self {
        Self::with_sense(m, noise_aware, num_metrics, Sense::Maximize)
    }

    pub fn with_sense(m: usize, noise_aware: bool, num_metrics: usize, sense: Sense) -> Self {
        Self::with_objective(m, noise_aware, num_metrics, sense)
    }
}

impl<O: ObjectiveSense> GenericTracker<O> {
    pub fn with_objective(m: usize, noise_aware: bool, num_metrics: usize, sense: O) -> Self {
        assert!(m >= 1);
        assert!(num_metrics >= 1);
        Self {
            observation_count: 0,
            mode: TrackerMode::new(m, noise_aware, num_metrics),
            sense,
        }
    }

    #[inline(always)]
    pub fn sense(&self) -> O {
        self.sense
    }

    pub fn tell(&mut self, index: usize, y: &Array1<f64>) {
        self.tell_view(index, y.view());
    }

    pub fn tell_slice(&mut self, index: usize, y: &[f64]) {
        self.tell_view(index, ArrayView1::from(y));
    }

    pub fn tell_view(&mut self, index: usize, y: ArrayView1<'_, f64>) {
        let id = ObsId::new(index);
        self.observation_count += 1;
        let sense = self.sense;
        match &mut self.mode {
            TrackerMode::AllCandidates { indices } => {
                indices.push(id);
            }
            TrackerMode::NoiselessBest { best } => {
                let score = sense.score(y[0]);
                match best {
                    None => *best = Some((score, vec![id])),
                    Some((best_score, ids)) => match score.cmp(best_score) {
                        Ordering::Greater => {
                            *best_score = score;
                            ids.clear();
                            ids.push(id);
                        }
                        Ordering::Equal => ids.push(id),
                        Ordering::Less => {}
                    },
                }
            }
            TrackerMode::ScalarTopM { m, heap } => {
                Self::push_heap(heap, id, sense.score(y[0]), *m);
            }
            TrackerMode::MultiTopM { m, heaps } => {
                for (col, heap) in heaps.iter_mut().enumerate() {
                    Self::push_heap(heap, id, sense.score(y[col]), *m);
                }
            }
        }
    }

    pub fn ask(&self) -> Vec<usize> {
        if self.observation_count == 0 {
            return Vec::new();
        }
        match &self.mode {
            TrackerMode::AllCandidates { indices } => Self::ids_sorted(indices.iter().copied()),
            TrackerMode::NoiselessBest { best } => match best {
                Some((_, ids)) => Self::ids_sorted(ids.iter().copied()),
                None => Vec::new(),
            },
            TrackerMode::ScalarTopM { heap, .. } => {
                Self::ids_sorted(heap.iter().map(|entry| entry.id))
            }
            TrackerMode::MultiTopM { heaps, .. } => {
                let mut union = BTreeSet::new();
                for heap in heaps {
                    for entry in heap {
                        union.insert(entry.id.index());
                    }
                }
                union.into_iter().collect()
            }
        }
    }

    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    pub fn reset(&mut self) {
        self.observation_count = 0;
        self.mode.reset();
    }

    /// Zero-copy rebuild over an observation matrix view without per-row heap
    /// allocation.
    pub fn rebuild(&mut self, y_obs: &ArrayView2<f64>) {
        self.reset();
        for i in 0..y_obs.nrows() {
            self.tell_view(i, y_obs.row(i));
        }
    }

    #[inline]
    fn push_heap(heap: &mut BinaryHeap<WorstFirst>, id: ObsId, score: BetterScore, m: usize) {
        if heap.len() < m {
            heap.push(WorstFirst { score, id });
        } else if let Some(&worst) = heap.peek() {
            if WorstFirst::is_better(id, score, worst) {
                heap.pop();
                heap.push(WorstFirst { score, id });
            }
        }
    }

    fn ids_sorted(iter: impl Iterator<Item = ObsId>) -> Vec<usize> {
        let mut out: Vec<usize> = iter.map(ObsId::index).collect();
        out.sort_unstable();
        out
    }

    #[cfg(test)]
    fn push_m(entries: &mut Vec<(usize, f64)>, index: usize, value: f64, m: usize) {
        entries.push((index, value));
        entries.sort_by(|&(ia, va), &(ib, vb)| vb.total_cmp(&va).then_with(|| ia.cmp(&ib)));
        entries.truncate(m);
    }

    #[cfg(test)]
    fn sorted_indices(entries: &[(usize, f64)]) -> Vec<usize> {
        let mut out: Vec<usize> = entries.iter().map(|(i, _)| *i).collect();
        out.sort_unstable();
        out
    }
}

pub fn enn_k(k: i32) -> usize {
    k.max(1) as usize
}

pub fn tracker_surrogate() -> usize {
    ALL_THRESHOLD + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_noiselessmaxties() {
        let mut t = IncumbentTracker::new(3, false, 1);
        t.tell(0, &array![1.0]);
        assert_eq!(t.ask(), vec![0]);
        t.tell(1, &array![3.0]);
        assert_eq!(t.ask(), vec![1]);
        t.tell(3, &array![3.0]);
        assert_eq!(t.ask(), vec![1, 3]);
    }

    #[test]
    fn test_noiseawaretopm() {
        let mut t = IncumbentTracker::new(3, true, 1);
        for (i, v) in [(0, 0.1), (1, 0.9), (2, 0.3), (3, 0.8), (4, 0.2)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![1, 2, 3]);
    }

    #[test]
    fn test_multimetricunion() {
        let mut t = IncumbentTracker::new(2, false, 2);
        let rows = [
            array![10.0, 0.0],
            array![9.0, 1.0],
            array![0.0, 10.0],
            array![1.0, 9.0],
            array![5.0, 5.0],
        ];
        for (i, row) in rows.into_iter().enumerate() {
            t.tell(i, &row);
        }
        assert_eq!(t.ask(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_allcandidatesmode() {
        let mut t = IncumbentTracker::new(tracker_surrogate(), false, 1);
        for (i, v) in [(0, 0.1), (1, 0.3), (2, -0.2)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![0, 1, 2]);
    }

    #[test]
    fn test_005() {
        assert_eq!(enn_k(4), 4);
        let mut t = IncumbentTracker::new(2, true, 1);
        t.rebuild(&array![[1.0], [3.0], [2.0]].view());
        assert_eq!(t.ask(), vec![1, 2]);
        t.reset();
        assert!(t.ask().is_empty());
    }

    #[test]
    fn test_006() {
        let mut t = IncumbentTracker::new(2, true, 1);
        for (i, v) in [(0, 5.0), (1, 5.0), (2, 5.0)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![0, 1]);
    }

    #[test]
    fn sense_tracker() {
        let mut min_t = MinTracker::with_objective(2, true, 1, Minimize);
        for (i, v) in [(0, 0.9), (1, -2.5), (2, 0.1), (3, -5.0)] {
            min_t.tell_slice(i, &[v]);
        }
        assert_eq!(min_t.ask(), vec![1, 3]);

        let mut tgt = TargetTracker::with_objective(1, false, 1, Target(10.0));
        tgt.tell_slice(0, &[12.0]);
        tgt.tell_slice(1, &[9.8]);
        tgt.tell_slice(2, &[10.5]);
        assert_eq!(tgt.ask(), vec![1]);
    }

    #[test]
    fn helpers_order() {
        let mut entries = vec![(0usize, 1.0), (1, 3.0), (2, 2.0)];
        IncumbentTracker::push_m(&mut entries, 3, 4.0, 2);
        let idx = IncumbentTracker::sorted_indices(&entries);
        assert!(!idx.is_empty());
    }
}
