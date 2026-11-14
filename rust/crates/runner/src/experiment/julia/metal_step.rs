//! Step-by-step fiber execution for resident Apple Silicon Metal GPU search.

use baux::artifact::inc::{EvalOutcome, StepRecord};
use baux::config::{ObjectivePolicy, RegionalSearchSpec, ResidentEnnConfig};
use baux::proposal_distribution::ProposalDistribution;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::metal::{
    MetalEngine, MetalJob, ask_retry, copy_best, obj_obs, observe_init, unscored_state,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use baux::resident_metal::{RegionalProposal, RegionalSearch, SearchState};

#[derive(Clone)]
pub struct MetalJobSpec {
    pub rep: u32,
    pub seed: u64,
    pub num_dim: usize,
    pub n_obj: usize,
    pub rounds: usize,
    pub has_surrogate: bool,
    pub tr_cfg: baux::trust_region::TRLengthConfig,
    pub kernel: ProposalDistribution,
    pub reg_spec: RegionalSearchSpec,
    pub enn_res: ResidentEnnConfig,
    pub policy: Option<ObjectivePolicy>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalJobSpec {
    pub fn as_job(&self) -> MetalJob<'_> {
        MetalJob {
            rep: self.rep,
            seed: self.seed,
            num_dim: self.num_dim,
            n_obj: self.n_obj,
            rounds: self.rounds,
            has_surrogate: self.has_surrogate,
            tr_cfg: self.tr_cfg,
            kernel: self.kernel,
            reg_spec: self.reg_spec.clone(),
            enn_res: self.enn_res.clone(),
            policy: self.policy.clone(),
            ckpt_dir: None,
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
enum PendingProposal {
    InitSlot(SearchState, Vec<u16>),
    RestartSlot(usize, SearchState, Vec<u16>),
    AskProp(RegionalProposal, Vec<u16>),
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) struct MetalFiber {
    pub(crate) spec: MetalJobSpec,
    pub(crate) eng: MetalEngine,
    pub(crate) round: usize,
    pub(crate) best_score: f64,
    pub(crate) best_viol: f64,
    pub(crate) initial_error: f64,
    pub(crate) total_accepts: u32,
    pub(crate) worker_s: f64,
    pending: Option<PendingProposal>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl MetalFiber {
    pub fn new(spec: MetalJobSpec) -> Self {
        let eng = MetalEngine::new(&spec.as_job());
        Self {
            spec,
            eng,
            round: 0,
            best_score: f64::NEG_INFINITY,
            best_viol: f64::INFINITY,
            initial_error: 0.0,
            total_accepts: 0,
            worker_s: 0.0,
            pending: None,
        }
    }

    pub fn prepare_proposal(&mut self) -> Result<Vec<u16>, String> {
        if self.pending.is_some() {
            return Err("prepare_proposal called while proposal pending".into());
        }
        if self.eng.search.is_none() {
            let slot = self.eng.init_states.len();
            let job = self.spec.as_job();
            let st = unscored_state(&job, slot, 0)?;
            let buf = st.base_buffer();
            let ptr = buf.contents().cast::<u16>();
            let row = unsafe { std::slice::from_raw_parts(ptr, self.spec.num_dim) }.to_vec();
            drop(buf);
            self.pending = Some(PendingProposal::InitSlot(st, row.clone()));
            return Ok(row);
        }
        if let Some((slot, gseq)) = self.eng.pending_restarts.pop_front() {
            let job = self.spec.as_job();
            let st = unscored_state(&job, slot, gseq)?;
            let buf = st.base_buffer();
            let ptr = buf.contents().cast::<u16>();
            let row = unsafe { std::slice::from_raw_parts(ptr, self.spec.num_dim) }.to_vec();
            drop(buf);
            self.pending = Some(PendingProposal::RestartSlot(slot, st, row.clone()));
            return Ok(row);
        }
        let s = self.eng.search.as_mut().ok_or("no search")?;
        let r_seed = self.spec.seed.wrapping_add((self.round + 1) as u64);
        let prop = ask_retry(s, r_seed, self.spec.enn_res.ask)?;
        let buf = s.buffer(&prop)?;
        let ptr = buf.contents().cast::<u16>();
        let row = unsafe { std::slice::from_raw_parts(ptr, self.spec.num_dim) }.to_vec();
        drop(buf);
        self.pending = Some(PendingProposal::AskProp(prop, row.clone()));
        Ok(row)
    }

    fn commit_init(
        &mut self,
        mut st: SearchState,
        row: Vec<u16>,
        sc: f64,
        outcome: &EvalOutcome,
    ) -> Result<(Vec<u16>, f64), String> {
        let job = self.spec.as_job();
        copy_best(row.as_ptr(), sc, self.best_score, &mut self.eng.best_row);
        observe_init(&job, &mut st, &outcome.scores)?;
        self.eng.init_states.push(st);
        if self.eng.init_states.len() == self.spec.reg_spec.count.max(1) {
            let states = std::mem::take(&mut self.eng.init_states);
            self.eng.search = Some(RegionalSearch::from_spec(states, &self.spec.reg_spec)?);
        }
        Ok((row, sc))
    }

    fn commit_restart(
        &mut self,
        slot: usize,
        mut st: SearchState,
        row: Vec<u16>,
        sc: f64,
        outcome: &EvalOutcome,
    ) -> Result<(Vec<u16>, f64), String> {
        let job = self.spec.as_job();
        copy_best(row.as_ptr(), sc, self.best_score, &mut self.eng.best_row);
        observe_init(&job, &mut st, &outcome.scores)?;
        self.eng
            .search
            .as_mut()
            .ok_or("no search")?
            .restart(slot, st)?;
        Ok((row, sc))
    }

    fn commit_ask(
        &mut self,
        prop: RegionalProposal,
        row: Vec<u16>,
        sc: f64,
        outcome: &EvalOutcome,
    ) -> Result<(Vec<u16>, f64), String> {
        let s = self.eng.search.as_mut().ok_or("no search")?;
        copy_best(row.as_ptr(), sc, self.best_score, &mut self.eng.best_row);
        if self.spec.n_obj >= 2 {
            let _ = s.tell_objectives(&prop, obj_obs(&outcome.scores, self.spec.n_obj)?)?;
        } else {
            let _ = s.tell(&prop, outcome.scores[0] as f32, 0.0)?;
        }
        let slot = prop.ticket().region.slot;
        if baux::resident_metal::regional_restarts(s, slot)? > 0 {
            self.eng.slot_gen[slot] += 1;
            self.eng
                .pending_restarts
                .push_back((slot, self.eng.slot_gen[slot]));
        }
        Ok((row, sc))
    }

    pub fn commit_eval(
        &mut self,
        outcome: &EvalOutcome,
        loss_fn: &dyn Fn(f64) -> f64,
    ) -> Result<StepRecord, String> {
        let pending = self.pending.take().ok_or("no proposal to commit")?;
        let sc =
            crate::experiment::julia::driver::scalar_score(&outcome.scores, 0, self.spec.n_obj);
        let (row, _) = match pending {
            PendingProposal::InitSlot(st, r) => self.commit_init(st, r, sc, outcome)?,
            PendingProposal::RestartSlot(slot, st, r) => {
                self.commit_restart(slot, st, r, sc, outcome)?
            }
            PendingProposal::AskProp(prop, r) => self.commit_ask(prop, r, sc, outcome)?,
        };
        self.round += 1;
        let accepted = sc > self.best_score;
        if accepted {
            self.best_score = sc;
            self.best_viol = outcome.viol;
            self.total_accepts += 1;
        }
        let cur_loss = loss_fn(self.best_score);
        if self.round == 1 {
            self.initial_error = cur_loss;
        }
        self.worker_s += (outcome.cost_us as f64) * 1.0e-6;
        let _ = row;
        Ok(StepRecord {
            eval_scores: outcome.scores.clone(),
            eval_viol: outcome.viol,
            best_score: self.best_score,
            best_viol: self.best_viol,
            accepted,
            cost_us: outcome.cost_us,
        })
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
