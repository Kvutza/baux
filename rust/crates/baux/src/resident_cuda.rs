use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::trials::RoundConfig;
use crate::trust_region::TRLengthConfig;

use baux::semantics::neighbor::MAX_HISTORY;
const MAX_PENDING: usize = 32;
static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

#[path = "resident_cuda/inspect.rs"]
mod inspect;
#[path = "resident_cuda/validation.rs"]
mod validation;
use validation::*;

fn cuda_config(config: RoundConfig) -> baux_cuda::RoundConfig {
    let crate::trials::SurrogateConfig {
        neighbors,
        epistemic_scale,
        aleatoric_scale,
        output_scale: y_scale,
    } = config.surrogate;
    baux_cuda::RoundConfig {
        neighbors,
        acquisition: crate::weights::acquisition_code(config.acquisition.kind),
        epistemic_scale,
        aleatoric_scale,
        y_scale,
        beta: config.acquisition.beta,
        adaptive_neighbors: false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TellMode {
    Legacy,
    Paired,
    PairedRelative,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamBlock {
    pub key: u64,
    pub offset: usize,
    pub len: usize,
    pub scale: f32,
    pub weight: f32,
}

impl ParamBlock {
    pub fn new(
        key: u64,
        offset: usize,
        len: usize,
        scale: f32,
        weight: f32,
    ) -> Result<Self, String> {
        if len == 0 {
            return Err("BF16 block length must be positive".to_string());
        }
        if !scale.is_finite() || scale < 0.0 || !weight.is_finite() || weight <= 0.0 {
            return Err("BF16 block weight must be positive and scale non-negative".to_string());
        }
        Ok(Self {
            key,
            offset,
            len,
            scale,
            weight,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proposal {
    owner: u64,
    id: u64,
    slot: usize,
    pub index: usize,
    pub seed: u64,
    pub score: f32,
    pub length: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Proposals {
    trials: Vec<Proposal>,
}

impl Proposals {
    pub fn arms(&self) -> usize {
        self.trials.len()
    }
}

/// Stateful BF16 weight search on CUDA, with sign or Gaussian proposals.
pub struct SearchState {
    owner: u64,
    tell_mode: Option<TellMode>,
    engine: baux_cuda::Bf16SearchEngine,
    dimensions: usize,
    capacity: usize,
    pending_capacity: usize,
    history: VecDeque<usize>,
    pending: Vec<Proposal>,
    next_id: u64,
    length: f64,
    length_config: TRLengthConfig,
    best: f32,
    best_variance: f32,
    restarts: usize,
    queued: Option<usize>,
    failure_limit: Option<usize>,
    correlated: bool,
    gaussian: bool,
    started: bool,
}

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        check_length(length)?;
        let len = check_search(
            base.len(),
            base_value,
            base_variance,
            &blocks,
            capacity,
            pending_capacity,
        )?;
        let slots = slot_count(capacity, pending_capacity)?;
        let leaves = cuda_blocks(&blocks);
        let engine = baux_cuda::Bf16SearchEngine::new(base, &leaves, slots)?;
        Self::create(
            engine,
            len,
            base_value,
            base_variance,
            capacity,
            pending_capacity,
            length,
        )
    }

    /// Copy a contiguous device-0 BF16 allocation into resident search state.
    ///
    /// # Safety
    /// `pointer` must address at least `len * 2` readable bytes on CUDA device
    /// 0.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_device(
        pointer: u64,
        len: usize,
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        check_length(length)?;
        let dimensions = check_search(
            len,
            base_value,
            base_variance,
            &blocks,
            capacity,
            pending_capacity,
        )?;
        let slots = slot_count(capacity, pending_capacity)?;
        let leaves = cuda_blocks(&blocks);
        let engine =
            unsafe { baux_cuda::Bf16SearchEngine::from_device(pointer, len, &leaves, slots)? };
        Self::create(
            engine,
            dimensions,
            base_value,
            base_variance,
            capacity,
            pending_capacity,
            length,
        )
    }

    fn create(
        mut engine: baux_cuda::Bf16SearchEngine,
        dimensions: usize,
        base_value: f32,
        base_variance: f32,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        engine.copy_row(0, 1)?;
        engine.init_search(
            base_value,
            base_variance,
            capacity,
            length.length_init,
            length.length_min,
            length.length_max,
        )?;
        Ok(Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            tell_mode: None,
            engine,
            dimensions,
            capacity,
            pending_capacity,
            history: VecDeque::from([1]),
            pending: Vec::with_capacity(pending_capacity),
            next_id: 0,
            length: length.length_init,
            length_config: length,
            best: base_value,
            best_variance: base_variance,
            restarts: 0,
            queued: None,
            failure_limit: None,
            correlated: false,
            gaussian: false,
            started: false,
        })
    }

    /// Enable correlated Gaussian proposals and the GPU accepted-radius
    /// controller once, before any asks. Requires one pending arm and no
    /// failure-tolerance override.
    pub fn enable_correlated(&mut self, reference_seed: u64) -> Result<(), String> {
        if self.gaussian || self.started || !self.pending.is_empty() || self.queued.is_some() {
            return Err("Enable correlated BF16 sampling once, before any rounds".into());
        }
        if self.pending_capacity != 1 || self.failure_limit.is_some() {
            return Err(
                "Correlated BF16 sampling requires max_pending=1 and no failure tolerance".into(),
            );
        }
        let length = correlated_length(self.length_config)?;
        self.engine.init_search(
            self.best,
            self.best_variance,
            self.capacity,
            length.length_init,
            length.length_min,
            length.length_max,
        )?;
        self.engine.enable_correlated(reference_seed)?;
        self.length_config = length;
        self.length = length.length_init;
        self.correlated = true;
        self.gaussian = true;
        Ok(())
    }

    /// Enable independent Gaussian proposals once, before any asks.
    /// This mode retains TuRBO and does not allocate a reference direction.
    pub fn enable_gaussian(&mut self) -> Result<(), String> {
        if self.gaussian || self.started || !self.pending.is_empty() || self.queued.is_some() {
            return Err("Enable Gaussian BF16 sampling once, before any rounds".into());
        }
        self.engine.enable_gaussian()?;
        self.gaussian = true;
        Ok(())
    }

    /// Pin a zero-valued incumbent anchor and use paired-relative observations.
    /// Enable once on a fresh correlated search, before any asks or tells.
    pub fn enable_relative(&mut self, failure_tolerance: usize) -> Result<(), String> {
        check_relativeinit(
            self.correlated,
            self.started || !self.pending.is_empty() || self.queued.is_some(),
            self.tell_mode,
            self.capacity,
            failure_tolerance,
        )?;
        self.engine.enable_relative(failure_tolerance)?;
        self.tell_mode = Some(TellMode::PairedRelative);
        Ok(())
    }

    pub fn ask(&mut self, seeds: &[u64], config: RoundConfig) -> Result<Proposal, String> {
        Ok(self.ask_batch(seeds, 1, config)?[0])
    }

    pub fn ask_batch(
        &mut self,
        seeds: &[u64],
        arms: usize,
        config: RoundConfig,
    ) -> Result<Vec<Proposal>, String> {
        if self.correlated {
            return Err("Correlated BF16 sampling requires ask_round, not explicit seeds".into());
        }
        self.sync()?;
        if !self.pending.is_empty() {
            return Err("tell must finish outstanding BF16 trials before ask".to_string());
        }
        if arms == 0 || arms > self.pending_capacity || seeds.is_empty() || seeds.len() % arms != 0
        {
            return Err("BF16 batch shape exceeds pending capacity".to_string());
        }
        let candidates = seeds.len() / arms;
        let slots = self.free_slots(arms)?;
        let length = self.length as f32;
        self.started = true;
        let selections = self.engine.ask(
            0,
            self.history.len(),
            &slots,
            seeds,
            candidates,
            length,
            config.acquisition.seed,
            cuda_config(config),
        )?;
        let mut trials = Vec::with_capacity(arms);
        for (&slot, selection) in slots.iter().zip(selections) {
            let index = selection.index as usize;
            let trial = Proposal {
                owner: self.owner,
                id: self.next_id,
                slot: slot as usize,
                index,
                seed: seeds[index],
                score: selection.score,
                length,
            };
            self.next_id = self.next_id.wrapping_add(1);
            self.pending.push(trial);
            trials.push(trial);
        }
        Ok(trials)
    }

    pub fn ask_round(
        &mut self,
        arms: usize,
        candidates: usize,
        seed_root: u64,
        config: RoundConfig,
    ) -> Result<Proposals, String> {
        if self.correlated && (arms != 1 || candidates != 4 || self.pending_capacity != 1) {
            return Err(
                "Correlated BF16 rounds require arms=1, candidates=4, max_pending=1".into(),
            );
        }
        if !self.pending.is_empty() {
            return Err("tell must finish the outstanding BF16 round".to_string());
        }
        if arms == 0
            || arms > self.pending_capacity
            || candidates == 0
            || arms.checked_mul(candidates).is_none()
        {
            return Err("BF16 round shape exceeds pending capacity".to_string());
        }
        let slots = self.free_slots(arms)?;
        self.started = true;
        self.engine.ask_seeded(
            0,
            self.capacity,
            &slots,
            candidates,
            seed_root,
            1.0,
            config.acquisition.seed,
            cuda_config(config),
        )?;
        let mut trials = Vec::with_capacity(arms);
        for &slot in &slots {
            let trial = Proposal {
                owner: self.owner,
                id: self.next_id,
                slot: slot as usize,
                index: 0,
                seed: 0,
                score: 0.0,
                length: 0.0,
            };
            self.next_id = self.next_id.wrapping_add(1);
            self.pending.push(trial);
            trials.push(trial);
        }
        Ok(Proposals { trials })
    }

    pub fn tell(&mut self, trial: Proposal, value: f32, variance: f32) -> Result<bool, String> {
        Ok(self.tell_batch(&[trial], &[value], &[variance])?[0])
    }

    pub fn tell_batch(
        &mut self,
        trials: &[Proposal],
        values: &[f32],
        variances: &[f32],
    ) -> Result<Vec<bool>, String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        check_tell(trials, values, variances)?;
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, trials.len()));
        let output = self
            .engine
            .tell(&slots, values, variances, self.capacity, tolerance)?;
        self.tell_mode = Some(TellMode::Legacy);
        Ok(self.finish_tell(trials, output))
    }

    /// Consume contiguous device-0 FP32 rewards and variances.
    ///
    /// # Safety
    /// The pointers must address `trials.len() * 4` readable bytes on CUDA
    /// device 0.
    pub unsafe fn tell_device(
        &mut self,
        trials: &[Proposal],
        values: u64,
        variances: Option<u64>,
    ) -> Result<Vec<bool>, String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        if trials.is_empty() {
            return Err("BF16 tell batch cannot be empty".to_string());
        }
        self.check_trials(trials)?;
        let slots = trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, trials.len()));
        let output = unsafe {
            self.engine.tell_device(
                &slots,
                values,
                variances,
                trials.len(),
                self.capacity,
                tolerance,
            )?
        };
        self.tell_mode = Some(TellMode::Legacy);
        Ok(self.finish_tell(trials, output))
    }

    pub fn tell_round(
        &mut self,
        round: &Proposals,
        values: &[f32],
        variances: &[f32],
    ) -> Result<Vec<bool>, String> {
        self.tell_batch(&round.trials, values, variances)
    }

    pub fn queue_round(
        &mut self,
        round: &Proposals,
        values: &[f32],
        variances: &[f32],
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        check_tell(&round.trials, values, variances)?;
        self.check_trials(&round.trials)?;
        let slots = round
            .trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, round.arms()));
        self.engine
            .queue_values(&slots, values, variances, self.capacity, tolerance)?;
        self.tell_mode = Some(TellMode::Legacy);
        self.pending.clear();
        self.queued = Some(round.arms());
        Ok(())
    }

    /// Queue a paired comparison. The explicit decision is independent of
    /// cached rewards. Rejection refreshes only the incumbent measurement;
    /// FIFO stores the candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn tell_paired(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        accept: bool,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Paired)?;
        check_paired(
            self.correlated,
            round.arms(),
            value,
            variance,
            incumbent_value,
            incumbent_variance,
        )?;
        self.check_trials(&round.trials)?;
        self.engine.queue_paired(
            round.trials[0].slot as u32,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            accept,
            self.capacity,
        )?;
        self.tell_mode = Some(TellMode::Paired);
        self.pending.clear();
        self.queued = Some(1);
        Ok(())
    }

    /// Queue an explicit decision with improvement measured against this
    /// incumbent. Absolute measurements update reporting only; accepted
    /// steps reset relative history. Rejections count toward radius
    /// contraction only when `reject_is_failure` is true.
    #[allow(clippy::too_many_arguments)]
    pub fn tell_relative(
        &mut self,
        round: &Proposals,
        value: f32,
        variance: f32,
        incumbent_value: f32,
        incumbent_variance: f32,
        improvement: f32,
        improvement_variance: f32,
        accept: bool,
        reject_is_failure: bool,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::PairedRelative)?;
        check_paired(
            self.correlated,
            round.arms(),
            value,
            variance,
            incumbent_value,
            incumbent_variance,
        )?;
        check_improvement(improvement, improvement_variance)?;
        self.check_trials(&round.trials)?;
        self.engine.queue_relative(
            round.trials[0].slot as u32,
            value,
            variance,
            incumbent_value,
            incumbent_variance,
            improvement,
            improvement_variance,
            accept,
            self.capacity,
            reject_is_failure,
        )?;
        self.pending.clear();
        self.queued = Some(1);
        Ok(())
    }

    /// Consume device rewards for an opaque resident round.
    ///
    /// # Safety
    /// The pointers must address `round.arms() * 4` readable CUDA bytes.
    pub unsafe fn finish_round(
        &mut self,
        round: &Proposals,
        values: u64,
        variances: Option<u64>,
    ) -> Result<(), String> {
        check_tellmode(self.tell_mode, TellMode::Legacy)?;
        self.check_trials(&round.trials)?;
        let slots = round
            .trials
            .iter()
            .map(|trial| trial.slot as u32)
            .collect::<Vec<_>>();
        let tolerance = self
            .failure_limit
            .unwrap_or_else(|| failure_tolerance(self.dimensions, round.arms()));
        unsafe {
            self.engine.queue_tell(
                &slots,
                values,
                variances,
                round.arms(),
                self.capacity,
                tolerance,
            )?;
        }
        self.tell_mode = Some(TellMode::Legacy);
        self.pending.clear();
        self.queued = Some(round.arms());
        Ok(())
    }

    pub fn sync(&mut self) -> Result<Vec<bool>, String> {
        let Some(count) = self.queued else {
            return Ok(Vec::new());
        };
        let output = self.engine.collect_tell(count)?;
        self.queued = None;
        let accepted = output.accepted.clone();
        self.update_state(output);
        Ok(accepted)
    }

    fn finish_tell(&mut self, trials: &[Proposal], output: baux_cuda::TellOutput) -> Vec<bool> {
        self.pending
            .retain(|candidate| !trials.iter().any(|trial| trial.id == candidate.id));
        let accepted = output.accepted.clone();
        self.update_state(output);
        accepted
    }

    fn update_state(&mut self, output: baux_cuda::TellOutput) {
        self.history = (1..=output.history).collect();
        self.length = output.length;
        self.best = output.best;
        self.best_variance = output.best_variance;
        self.restarts = output.restarts;
    }
}
