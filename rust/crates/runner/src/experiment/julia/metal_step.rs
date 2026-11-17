//! Unified step-by-step and multi-arm batch execution for resident Apple
//! Silicon Metal GPU search.

use std::collections::VecDeque;

use baux::artifact::inc::{EvalOutcome, StepRecord};
use baux::config::{ObjectivePolicy, OptimizerSpec, RegionalSearchSpec, ResidentEnnConfig};
use baux::proposal_distribution::ProposalDistribution;

use super::metal::{
    explicit_dist, is_single, lhd_batch, make_enn, regional_spec, resolve_kernel, restart_seed,
    round_seed,
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::metal::{ask_retry, make_region, obj_obs};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use baux::resident_metal::{DeviceRow, RegionalProposal, RegionalSearch};

#[derive(Clone)]
pub struct MetalJobSpec {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub n_obj: usize,
    pub rounds: usize,
    pub single_state: bool,
    pub explicit_dist: bool,
    pub has_surrogate: bool,
    pub tr_cfg: baux::trust_region::TRLengthConfig,
    pub kernel: ProposalDistribution,
    pub cfg_opt: baux::optimizer_config::OptimizerConfig,
    pub reg_spec: RegionalSearchSpec,
    pub enn_res: ResidentEnnConfig,
    pub policy: Option<ObjectivePolicy>,
}

impl MetalJobSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        rep: u32,
        seed: u64,
        num_dim: usize,
        n_obj: usize,
        rounds: usize,
        spec: &OptimizerSpec,
        cfg_opt: baux::optimizer_config::OptimizerConfig,
        policy: Option<ObjectivePolicy>,
    ) -> Result<Self, String> {
        let single_state = is_single(spec);
        let explicit = explicit_dist(spec);
        if !single_state
            && !explicit
            && cfg_opt.candidates.candidate_rv == baux::candidates::CandidateSampler::Sobol
            && (num_dim == 0 || num_dim > 21201)
        {
            return Err(format!(
                "Invalid parameter: Sobol dimension must be in 1..=21201, got {num_dim}"
            ));
        }
        let enn_res = make_enn(spec, &cfg_opt, num_dim, seed)?;
        let tr_cfg = match &cfg_opt.trust_region {
            baux::trust_region::TrustRegionConfig::Turbo(l) => *l,
            baux::trust_region::TrustRegionConfig::Morbo(m) => m.length,
        };
        let has_surrogate = matches!(
            cfg_opt.surrogate,
            baux::optimizer_config::SurrogateConfig::ENN(_)
        );
        let arms = enn_res.pool.arms().max(1) as usize;
        Ok(Self {
            rep,
            seed,
            num_dim,
            n_obj: n_obj.max(1),
            rounds,
            single_state,
            explicit_dist: explicit,
            has_surrogate,
            tr_cfg,
            kernel: resolve_kernel(spec),
            reg_spec: regional_spec(spec, num_dim, arms),
            cfg_opt,
            enn_res,
            policy,
        })
    }

    #[inline]
    pub fn arms(&self) -> usize {
        self.enn_res.pool.arms().max(1) as usize
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
enum PendingBatch {
    Warmup(Vec<Vec<u16>>),
    Replace(Vec<(usize, u64)>, Vec<Vec<u16>>),
    Active(Vec<(RegionalProposal, Vec<DeviceRow>, Vec<Vec<u16>>)>),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalFiber {
    pub(crate) spec: MetalJobSpec,
    pub(crate) search: Option<RegionalSearch>,
    pub(crate) warm_rows: Vec<(Vec<u16>, Vec<f64>)>,
    pub(crate) restarts: VecDeque<(usize, u64)>,
    pub(crate) slot_gen: Vec<u64>,
    pub(crate) best_row: Vec<u16>,
    pub(crate) round: usize,
    pub(crate) best_score: f64,
    pub(crate) best_viol: f64,
    pub(crate) initial_error: f64,
    pub(crate) total_accepts: u32,
    pub(crate) worker_s: f64,
    pending: Option<PendingBatch>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalFiber {
    pub fn new(spec: MetalJobSpec) -> Self {
        let n_reg = spec.reg_spec.count.max(1);
        let init_row = lhd_batch(round_seed(spec.seed, 1), 1, spec.num_dim)
            .into_iter()
            .next()
            .unwrap_or_default();
        Self {
            spec,
            search: None,
            warm_rows: Vec::new(),
            restarts: VecDeque::new(),
            slot_gen: vec![0; n_reg],
            best_row: init_row,
            round: 0,
            best_score: f64::NEG_INFINITY,
            best_viol: f64::INFINITY,
            initial_error: 0.0,
            total_accepts: 0,
            worker_s: 0.0,
            pending: None,
        }
    }

    fn warmup_goal(&self) -> usize {
        let n_reg = self.spec.reg_spec.count.max(1);
        if self.spec.has_surrogate && !self.spec.single_state && !self.spec.explicit_dist {
            self.spec.enn_res.ask.surrogate.neighbors.max(n_reg)
        } else {
            n_reg
        }
    }

    fn prepare_n(&mut self, max_arms: usize) -> Result<Vec<Vec<u16>>, String> {
        if self.pending.is_some() {
            return Err("proposal batch already pending".into());
        }
        let want = max_arms.max(1);
        let step_seed = round_seed(self.spec.seed, self.round + 1);
        if self.search.is_none() {
            let rem = self.warmup_goal().saturating_sub(self.warm_rows.len());
            let count = rem.min(want).max(1);
            let rows = lhd_batch(step_seed, count, self.spec.num_dim);
            self.pending = Some(PendingBatch::Warmup(rows.clone()));
            return Ok(rows);
        }
        if !self.restarts.is_empty() {
            let s = self.search.as_ref().ok_or("no search")?;
            let count = self.restarts.len().min(want).max(1);
            let mut targets = Vec::with_capacity(count);
            let mut rows = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some((slot, gseq)) = self.restarts.pop_front() {
                    let c_seed = restart_seed(self.spec.seed, slot, gseq);
                    rows.push(s.sample_center(slot, c_seed)?);
                    targets.push((slot, gseq));
                }
            }
            self.pending = Some(PendingBatch::Replace(targets, rows.clone()));
            return Ok(rows);
        }
        let mut s = self.search.take().ok_or("no search")?;
        let n_reg = self.spec.reg_spec.count.max(1);
        let issue_n = n_reg.min(want).max(1);
        let mut issued = Vec::with_capacity(issue_n);
        let mut flat_rows = Vec::with_capacity(want);
        for q in 0..issue_n {
            let q_seed = baux::semantics::random::splitmix64(
                step_seed ^ ((q as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
            );
            let (next_s, prop) = ask_retry(s, q_seed, self.spec.enn_res.ask)?;
            s = next_s;
            let quota = (want / issue_n) + usize::from(q < (want % issue_n));
            let (dev_rows, host_rows) = if quota <= 1 {
                let buf = s.buffer(&prop)?;
                let ptr = buf.contents().cast::<u16>();
                let row = unsafe { std::slice::from_raw_parts(ptr, self.spec.num_dim) }.to_vec();
                drop(buf);
                (Vec::new(), vec![row])
            } else {
                let batch = s.materialize_batch(&prop)?;
                let take_n = quota.min(batch.len()).max(1);
                let mut hrs = Vec::with_capacity(take_n);
                let mut drs = Vec::with_capacity(take_n);
                for dev in batch.into_iter().take(take_n) {
                    hrs.push(dev.read_u16()?);
                    drs.push(dev);
                }
                (drs, hrs)
            };
            flat_rows.extend(host_rows.iter().cloned());
            issued.push((prop, dev_rows, host_rows));
        }
        self.search = Some(s);
        self.pending = Some(PendingBatch::Active(issued));
        Ok(flat_rows)
    }

    pub fn prepare_batch(&mut self) -> Result<Vec<Vec<u16>>, String> {
        self.prepare_n(self.spec.arms())
    }

    pub fn prepare_proposal(&mut self) -> Result<Vec<u16>, String> {
        let mut rows = self.prepare_n(1)?;
        rows.pop().ok_or_else(|| "empty proposal batch".into())
    }

    fn finish_warmup(&mut self) -> Result<(), String> {
        let n_reg = self.spec.reg_spec.count.max(1);
        let n_obj = self.spec.n_obj;
        if n_reg == 1 {
            let mut best_idx = 0usize;
            let mut best_sc = f64::NEG_INFINITY;
            for (idx, (_, scs)) in self.warm_rows.iter().enumerate() {
                let sc = crate::experiment::julia::driver::scalar_score(scs, 0, n_obj);
                if sc > best_sc {
                    best_sc = sc;
                    best_idx = idx;
                }
            }
            let st0 = make_region(
                &self.spec,
                &self.warm_rows[best_idx].0,
                0,
                0,
                &self.warm_rows[best_idx].1,
            )?;
            let mut s = RegionalSearch::from_spec(vec![st0], &self.spec.reg_spec)?;
            if self.spec.has_surrogate {
                for (idx, (row, scs)) in self.warm_rows.iter().enumerate() {
                    if idx != best_idx {
                        let buf = baux::resident_metal::upload(row)?;
                        s = if n_obj >= 2 {
                            s.import_vec(0, &buf, obj_obs(scs, n_obj)?)?
                        } else {
                            s.import_region(0, &buf, scs[0] as f32, 0.0)?
                        };
                    }
                }
            }
            self.warm_rows.clear();
            self.search = Some(s);
            return Ok(());
        }
        let mut states = Vec::with_capacity(n_reg);
        for slot in 0..n_reg {
            let (ref row, ref scs) = self.warm_rows[slot];
            states.push(make_region(&self.spec, row, slot, 0, scs)?);
        }
        let mut s = RegionalSearch::from_spec(states, &self.spec.reg_spec)?;
        if self.spec.has_surrogate && self.warm_rows.len() > n_reg {
            for (idx, (row, scs)) in self.warm_rows.iter().enumerate().skip(n_reg) {
                let buf = baux::resident_metal::upload(row)?;
                let slot = idx % n_reg;
                s = if n_obj >= 2 {
                    s.import_vec(slot, &buf, obj_obs(scs, n_obj)?)?
                } else {
                    s.import_region(slot, &buf, scs[0] as f32, 0.0)?
                };
            }
        }
        self.warm_rows.clear();
        self.search = Some(s);
        Ok(())
    }

    fn commit_replace(
        &mut self,
        targets: Vec<(usize, u64)>,
        rows: &[Vec<u16>],
        scores: &[f64],
    ) -> Result<(), String> {
        let mut s = self.search.take().ok_or("no search")?;
        let n_obj = self.spec.n_obj;
        for (i, (slot, gseq)) in targets.into_iter().enumerate() {
            let sl = &scores[i * n_obj..(i + 1) * n_obj];
            let st = make_region(&self.spec, &rows[i], slot, gseq, sl)?;
            s = s.restart(slot, st)?;
        }
        self.search = Some(s);
        Ok(())
    }

    fn commit_active(
        &mut self,
        issued: Vec<(RegionalProposal, Vec<DeviceRow>, Vec<Vec<u16>>)>,
        scores: &[f64],
    ) -> Result<(), String> {
        let mut s = self.search.take().ok_or("no search")?;
        let n_obj = self.spec.n_obj;
        let mut cursor = 0usize;
        for (prop, dev_rows, host_rows) in issued {
            let slot = prop.ticket().region.slot;
            if dev_rows.len() > 1 {
                let mut obs_vec = Vec::with_capacity(dev_rows.len());
                for a in 0..dev_rows.len() {
                    let sl = &scores[(cursor + a) * n_obj..(cursor + a + 1) * n_obj];
                    let obs = if n_obj >= 2 {
                        obj_obs(sl, n_obj)?
                    } else {
                        baux::objective_observation::ObjectiveObservation::scalar_adapter(
                            sl[0] as f32,
                            0.0,
                        )
                    };
                    obs_vec.push(obs);
                }
                let (next_s, _dec) = s.tell_arms(&prop, &dev_rows, &obs_vec)?;
                s = next_s;
            } else {
                let sc0 = &scores[cursor * n_obj..(cursor + 1) * n_obj];
                let (next_s, _dec) = if n_obj >= 2 {
                    s.tell_objectives(&prop, obj_obs(sc0, n_obj)?)?
                } else {
                    s.tell(&prop, sc0[0] as f32, 0.0)?
                };
                s = next_s;
            }
            if baux::resident_metal::regional_restarts(&s, slot)? > 0 {
                self.slot_gen[slot] += 1;
                self.restarts.push_back((slot, self.slot_gen[slot]));
            }
            cursor += host_rows.len();
        }
        self.search = Some(s);
        Ok(())
    }

    pub fn commit_batch(
        &mut self,
        scores: &[f64],
        viols: &[f64],
        cost_us: u64,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        let pending = self.pending.take().ok_or("no proposal to commit")?;
        let n_obj = self.spec.n_obj;
        let batch_rows: Vec<Vec<u16>> = match pending {
            PendingBatch::Warmup(rows) => {
                for (i, r) in rows.iter().enumerate() {
                    let sl = scores[i * n_obj..(i + 1) * n_obj].to_vec();
                    self.warm_rows.push((r.clone(), sl));
                }
                if self.warm_rows.len() >= self.warmup_goal() {
                    self.finish_warmup()?;
                }
                rows
            }
            PendingBatch::Replace(targets, rows) => {
                self.commit_replace(targets, &rows, scores)?;
                rows
            }
            PendingBatch::Active(issued) => {
                let rows: Vec<Vec<u16>> = issued
                    .iter()
                    .flat_map(|(_, _, hrs)| hrs.iter().cloned())
                    .collect();
                self.commit_active(issued, scores)?;
                rows
            }
        };

        self.round += 1;
        let mut accepted = false;
        let mut round_viol = viols.first().copied().unwrap_or(0.0);
        for (i, row) in batch_rows.iter().enumerate() {
            let sc = crate::experiment::julia::driver::scalar_score(scores, i, n_obj);
            let vl = viols.get(i).copied().unwrap_or(round_viol);
            if sc > self.best_score {
                self.best_score = sc;
                self.best_viol = vl;
                self.best_row.copy_from_slice(row);
                round_viol = vl;
                accepted = true;
            }
        }
        if accepted {
            self.total_accepts += 1;
        }
        if self.round == 1 {
            self.initial_error = loss_fn(self.best_score);
        }
        self.worker_s += (cost_us as f64) * 1.0e-6;
        Ok(StepRecord {
            eval_scores: scores.to_vec(),
            eval_viol: round_viol,
            best_score: self.best_score,
            best_viol: self.best_viol,
            accepted,
            cost_us,
        })
    }

    pub fn commit_eval(
        &mut self,
        outcome: &EvalOutcome,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        self.commit_batch(&outcome.scores, &[outcome.viol], outcome.cost_us, loss_fn)
    }

    pub fn replay_prefix(
        &mut self,
        steps: &[StepRecord],
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<(), String> {
        for step in steps {
            let _ = self.prepare_proposal()?;
            let outcome = EvalOutcome {
                scores: step.eval_scores.clone(),
                viol: step.eval_viol,
                cost_us: step.cost_us,
            };
            let _ = self.commit_eval(&outcome, loss_fn)?;
        }
        Ok(())
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) struct MetalFiber;
