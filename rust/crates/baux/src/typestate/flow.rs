//! Alternating [`Ready`] / [`Pending`] affine typestates and functional
//! `scan_step` / `pmap_branches` combinators.

use core::marker::PhantomData;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::error::ENNError;
use crate::hash::AffineKey;
use crate::sense::ObjectiveSense;
use crate::ticket::ProposalTicket;

use super::{Active, TypedOptimizer};

/// Typestate marker: active optimizer ready to issue a proposal (`ask_step`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ready;

/// Typestate marker: optimizer awaiting resolution (`tell_step` or
/// `cancel_step`) of an issued `ProposalTicket`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pending;

/// Per-step provenance and incumbent trace produced by [`scan_step`].
#[derive(Clone, Debug, PartialEq)]
pub struct StepTrace {
    pub step: usize,
    pub digest: [u8; 32],
    pub best_y: Option<f64>,
}

impl<O: ObjectiveSense> TypedOptimizer<O, Active> {
    /// Enter the strict alternating [`Ready`] / [`Pending`] affine typestate
    /// protocol.
    pub fn into_ready(self) -> TypedOptimizer<O, Ready> {
        TypedOptimizer {
            inner: self.inner,
            sense: self.sense,
            phase: PhantomData,
        }
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Ready> {
    /// Issue an affine [`ProposalTicket`], advance [`AffineKey`], and
    /// transition to [`TypedOptimizer<O, Pending>`].
    pub fn ask_step(
        self,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Pending>, ProposalTicket, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut next_inner = self.inner.fork()?;
        let ticket = next_inner.ask_ticket(num_arms, &mut draw_key.prng())?;
        let pending = TypedOptimizer {
            inner: next_inner,
            sense: self.sense,
            phase: PhantomData,
        };
        Ok((pending, ticket, next_key))
    }
}

impl<O: ObjectiveSense> TypedOptimizer<O, Pending> {
    /// Consume an affine [`ProposalTicket`] and [`AffineKey`], record
    /// observations, and return to [`TypedOptimizer<O, Ready>`].
    pub fn tell_step(
        self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Ready>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let y_canon = self.canonicalize_y(y);
        let mut next_inner = self.inner.fork()?;
        next_inner.tell_ticket(ticket, &y_canon.view(), None, &mut draw_key.prng())?;
        let ready = TypedOptimizer {
            inner: next_inner,
            sense: self.sense,
            phase: PhantomData,
        };
        Ok((ready, next_key))
    }

    /// Retire an affine [`ProposalTicket`] without controller penalty and
    /// return to [`TypedOptimizer<O, Ready>`].
    pub fn cancel_step(
        self,
        ticket: ProposalTicket,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Ready>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut next_inner = self.inner.fork()?;
        next_inner.cancel_ticket(ticket, &mut draw_key.prng())?;
        let ready = TypedOptimizer {
            inner: next_inner,
            sense: self.sense,
            phase: PhantomData,
        };
        Ok((ready, next_key))
    }
}

/// Fold `steps` pure `(ask_step -> eval_fn -> tell_step)` rounds over `opt`
/// and `key`.
pub fn scan_step<O: ObjectiveSense, F>(
    mut opt: TypedOptimizer<O, Ready>,
    mut key: AffineKey,
    steps: usize,
    num_arms: usize,
    mut eval_fn: F,
) -> Result<(TypedOptimizer<O, Ready>, AffineKey, Vec<StepTrace>), ENNError>
where
    F: FnMut(usize, &Array2<f64>) -> Result<Array2<f64>, ENNError>,
{
    let mut traces = Vec::with_capacity(steps);
    for step in 0..steps {
        let (pending, ticket, k_mid) = opt.ask_step(num_arms, key)?;
        let y = eval_fn(step, ticket.as_array())?;
        let (ready, k_next) = pending.tell_step(ticket, &y.view(), k_mid)?;
        traces.push(StepTrace {
            step,
            digest: ready.digest(),
            best_y: ready.best_y(),
        });
        opt = ready;
        key = k_next;
    }
    Ok((opt, key, traces))
}

/// Fork `opt` and split `key` across `branches` independent trajectories,
/// executing `branch_fn` in parallel via Rayon.
pub fn pmap_branches<O: ObjectiveSense, F, R: Send>(
    opt: &TypedOptimizer<O, Ready>,
    key: AffineKey,
    branches: usize,
    branch_fn: F,
) -> Result<Vec<R>, ENNError>
where
    F: Fn(usize, TypedOptimizer<O, Ready>, AffineKey) -> Result<R, ENNError> + Sync,
{
    let keys = key.split_n(branches);
    let mut items = Vec::with_capacity(branches);
    for (idx, branch_key) in keys.into_iter().enumerate() {
        items.push((idx, opt.fork()?, branch_key));
    }
    items
        .into_par_iter()
        .map(|(idx, branch_opt, branch_key)| branch_fn(idx, branch_opt, branch_key))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{AcqStage, InitStage, ModelStage, PerturbStage};
    use crate::sense::Minimize;
    use ndarray::{Axis, array};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_ready(seed: u64) -> TypedOptimizer<Minimize, Ready> {
        let mut rng = StdRng::seed_from_u64(seed);
        let bounds = array![[-4.0, 4.0], [-4.0, 4.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.5, Minimize);
        let mut warmup = pipe.build(bounds, &mut rng).unwrap();
        let x0 = warmup.warmup_design(4, &mut rng).unwrap();
        let y0 = x0
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));
        warmup
            .activate(&x0.view(), &y0.view(), &mut rng)
            .unwrap()
            .into_ready()
    }

    fn sphere_eval(_step: usize, x: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
        Ok(x.map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1)))
    }

    #[test]
    fn affine_protocol() {
        let ready = make_ready(17);
        let key = AffineKey::new(900);
        let (pending, ticket, key) = ready.ask_step(2, key).unwrap();
        let (ready_cancelled, key) = pending.cancel_step(ticket, key).unwrap();
        let (pending2, ticket2, key) = ready_cancelled.ask_step(2, key).unwrap();
        let y = sphere_eval(0, ticket2.as_array()).unwrap();
        let (ready_done, key) = pending2.tell_step(ticket2, &y.view(), key).unwrap();
        let _ = key.into_key();
        assert!(ready_done.best_y().unwrap().is_finite());
    }

    #[test]
    fn scan_pmap() {
        let root = make_ready(73);
        let run_branches = || {
            pmap_branches(&root, AffineKey::new(2026), 4, |idx, opt, bkey| {
                let shift = (idx as f64) * 0.25;
                let (final_opt, next_key, traces) = scan_step(opt, bkey, 3, 2, |_, x| {
                    Ok(
                        x.map_axis(Axis(1), |r| (r[0] - shift).powi(2) + r[1].powi(2))
                            .insert_axis(Axis(1)),
                    )
                })?;
                let raw_seed = next_key.into_key().seed();
                Ok((final_opt.digest(), raw_seed, traces))
            })
            .unwrap()
        };
        let res_a = run_branches();
        let res_b = run_branches();
        assert_eq!(res_a, res_b);
        assert_ne!(res_a[0].0, res_a[1].0);
    }
}
