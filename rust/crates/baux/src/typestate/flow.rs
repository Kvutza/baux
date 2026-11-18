//! Alternating [`Ready`] / [`Pending`] affine typestates and functional
//! `scan_step` / `pmap_branches` combinators.

use core::marker::PhantomData;

use ndarray::{Array2, ArrayView2};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::error::ENNError;
use crate::hash::AffineKey;
use crate::sense::ObjectiveSense;
use crate::ticket::ProposalTicket;
use crate::unified_opt::OptBackend;

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

impl<O: ObjectiveSense, B: OptBackend> TypedOptimizer<O, Active, B> {
    /// Enter the strict alternating [`Ready`] / [`Pending`] affine typestate
    /// protocol.
    pub fn into_ready(self) -> TypedOptimizer<O, Ready, B> {
        TypedOptimizer {
            engine: self.engine,
            sense: self.sense,
            backend: self.backend,
            phase: PhantomData,
        }
    }
}

impl<O: ObjectiveSense, B: OptBackend> TypedOptimizer<O, Ready, B> {
    /// Issue an affine [`ProposalTicket`], advance [`AffineKey`], and
    /// transition to [`TypedOptimizer<O, Pending, B>`]. Consumes `self` and
    /// `key` by value.
    pub fn ask_step(
        mut self,
        num_arms: usize,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Pending, B>, ProposalTicket, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut prng = draw_key.prng();
        let (ticket, next) = self.engine.ask_ticket(num_arms, &mut prng)?;
        self.engine = next;
        let pending = TypedOptimizer {
            engine: self.engine,
            sense: self.sense,
            backend: self.backend,
            phase: PhantomData,
        };
        Ok((pending, ticket, next_key))
    }
}

impl<O: ObjectiveSense, B: OptBackend> TypedOptimizer<O, Pending, B> {
    /// Consume an affine [`ProposalTicket`] and [`AffineKey`], record
    /// observations, and return to [`TypedOptimizer<O, Ready, B>`]. Consumes
    /// `self` by value.
    pub fn tell_step(
        mut self,
        ticket: ProposalTicket,
        y: &ArrayView2<f64>,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Ready, B>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let y_canon = self.canonicalize_y(y);
        let mut prng = draw_key.prng();
        self.engine = self
            .engine
            .tell_ticket(ticket, &y_canon.view(), None, &mut prng)?;
        let ready = TypedOptimizer {
            engine: self.engine,
            sense: self.sense,
            backend: self.backend,
            phase: PhantomData,
        };
        Ok((ready, next_key))
    }

    /// Retire an affine [`ProposalTicket`] without controller penalty and
    /// return to [`TypedOptimizer<O, Ready, B>`]. Consumes `self` by value.
    pub fn cancel_step(
        mut self,
        ticket: ProposalTicket,
        key: AffineKey,
    ) -> Result<(TypedOptimizer<O, Ready, B>, AffineKey), ENNError> {
        let (draw_key, next_key) = key.step();
        let mut prng = draw_key.prng();
        self.engine = self.engine.cancel_ticket(ticket, &mut prng)?;
        let ready = TypedOptimizer {
            engine: self.engine,
            sense: self.sense,
            backend: self.backend,
            phase: PhantomData,
        };
        Ok((ready, next_key))
    }
}

/// Fold `steps` pure `(ask_step -> eval_fn -> tell_step)` rounds over `opt`
/// and `key`.
pub fn scan_step<O: ObjectiveSense, B: OptBackend, F>(
    mut opt: TypedOptimizer<O, Ready, B>,
    mut key: AffineKey,
    steps: usize,
    num_arms: usize,
    mut eval_fn: F,
) -> Result<(TypedOptimizer<O, Ready, B>, AffineKey, Vec<StepTrace>), ENNError>
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
/// executing `branch_fn` in parallel via Rayon. Consumes `opt` as pure value
/// semantics.
pub fn pmap_branches<O: ObjectiveSense, B: OptBackend, F, R: Send>(
    opt: TypedOptimizer<O, Ready, B>,
    key: AffineKey,
    branches: usize,
    branch_fn: F,
) -> Result<Vec<R>, ENNError>
where
    F: Fn(usize, TypedOptimizer<O, Ready, B>, AffineKey) -> Result<R, ENNError> + Sync,
{
    if branches == 0 {
        return Ok(Vec::new());
    }
    let keys = key.split_n(branches);
    let mut items = Vec::with_capacity(branches);
    let mut opt_slot = Some(opt);
    for (idx, branch_key) in keys.into_iter().enumerate() {
        let branch_opt = if idx + 1 == branches {
            opt_slot.take().expect("last branch takes owned optimizer")
        } else {
            opt_slot.as_ref().expect("branch optimizer").clone()
        };
        items.push((idx, branch_opt, branch_key));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        items
            .into_par_iter()
            .map(|(idx, branch_opt, branch_key)| branch_fn(idx, branch_opt, branch_key))
            .collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items
            .into_iter()
            .map(|(idx, branch_opt, branch_key)| branch_fn(idx, branch_opt, branch_key))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{AcqStage, InitStage, ModelStage, PerturbStage};
    use crate::sense::Minimize;
    use ndarray::{Axis, array};

    fn make_ready(seed: u64) -> (TypedOptimizer<Minimize, Ready>, AffineKey) {
        let key = AffineKey::new(seed);
        let bounds = array![[-4.0, 4.0], [-4.0, 4.0]];
        let pipe = InitStage::Lhd(4)
            | PerturbStage::Raasp(16)
            | ModelStage::Enn(3)
            | AcqStage::ucb_sense(1.0, Minimize);
        let (warmup, key) = pipe.build_key(bounds, key).unwrap();
        let (x_init, warmup, key) = warmup.design_step(4, key).unwrap();
        let y_init = x_init
            .map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1));
        warmup
            .activate_ready(&x_init.view(), &y_init.view(), key)
            .unwrap()
    }

    fn sphere_eval(_step: usize, x: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
        Ok(x.map_axis(Axis(1), |r| r[0].powi(2) + r[1].powi(2))
            .insert_axis(Axis(1)))
    }

    #[test]
    fn affine_protocol() {
        let (ready, key) = make_ready(17);
        let (pending, ticket, key) = ready.ask_step(2, key).unwrap();
        let (ready_cancelled, key) = pending.cancel_step(ticket, key).unwrap();
        let (pending_next, ticket_next, key) = ready_cancelled.ask_step(2, key).unwrap();
        let y = sphere_eval(0, ticket_next.as_array()).unwrap();
        let (ready_done, key) = pending_next.tell_step(ticket_next, &y.view(), key).unwrap();
        let _ = key.into_key();
        assert!(ready_done.best_y().unwrap().is_finite());
    }

    #[test]
    fn scan_pmap() {
        let (root, key) = make_ready(73);
        let seed = key.into_key().seed();
        let run_branches = || {
            pmap_branches(root.clone(), AffineKey::new(seed), 4, |idx, opt, bkey| {
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
