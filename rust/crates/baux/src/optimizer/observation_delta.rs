use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use super::BayesOptimizer;
use super::incumbent::IncumbentPoint;
use super::obs_access::{PosteriorBackend, PosteriorStore};
use crate::candidates::SobolEngine;
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::scratch::ImLog;
use crate::snapshot_io::{
    MAX_DIM, MAX_METRICS, MAX_OBS, MAX_PENDING, MAX_TEXT, WireReader, bad_state, seal_frame,
    write_arr, write_bool, write_byte, write_header, write_len, write_mat, write_real, write_slice,
    write_word,
};
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};

/// Host-language pure state cursor metadata embedded in portable checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCursor {
    pub seed: u64,
    pub epoch: u64,
    pub identified: bool,
    pub id_map: Vec<(u64, u64)>,
}

/// Closed phase lifecycle of the optimizer state machine (eliminating boolean
/// `in_init` flags).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Space-filling design only (e.g. LHD-only).
    DesignOnly,
    /// Initial warm-up phase before entering active trust-region search.
    Warmup,
    /// Active trust-region / surrogate search phase.
    Active,
}

/// Alias preserving source compatibility for callers naming `WarmupMode`.
pub type WarmupMode = Lifecycle;

/// Unified lifecycle state for initial warmup, post-restart local warmup, and
/// generation tracking.
#[derive(Debug, Clone)]
pub struct WarmupSchedule {
    pub kind: InitStrategy,
    pub budget: usize,
    pub completed: usize,
    pub mode: Lifecycle,
    pub reinit_left: Option<usize>,
    pub generation: usize,
}

impl WarmupSchedule {
    pub fn design_only(kind: InitStrategy, budget: usize, completed: usize) -> Self {
        Self {
            kind,
            budget,
            completed,
            mode: Lifecycle::DesignOnly,
            reinit_left: None,
            generation: 0,
        }
    }

    pub fn active(kind: InitStrategy, budget: usize, completed: usize, in_init: bool) -> Self {
        Self {
            kind,
            budget,
            completed,
            mode: if in_init {
                Lifecycle::Warmup
            } else {
                Lifecycle::Active
            },
            reinit_left: None,
            generation: 0,
        }
    }

    pub fn progress(&self) -> Option<(usize, usize)> {
        match self.mode {
            Lifecycle::DesignOnly | Lifecycle::Warmup => Some((self.completed, self.budget)),
            Lifecycle::Active => None,
        }
    }

    pub fn in_init(&self) -> bool {
        matches!(self.mode, Lifecycle::DesignOnly | Lifecycle::Warmup)
    }

    pub fn advance_init(&mut self, rows: usize) {
        self.completed += rows;
    }

    pub fn check_transition(&mut self) {
        if self.mode == Lifecycle::Warmup && self.completed >= self.budget {
            self.mode = Lifecycle::Active;
        }
    }

    pub fn restart_local(&mut self) {
        self.generation += 1;
        self.reinit_left = (self.budget > 0).then_some(self.budget);
    }

    pub fn consume_reinit(&mut self, count: usize) {
        if let Some(left) = self.reinit_left {
            let remaining = left.saturating_sub(count);
            self.reinit_left = (remaining > 0).then_some(remaining);
        }
    }

    pub(crate) fn mode_tag(&self) -> u8 {
        match self.mode {
            Lifecycle::DesignOnly => 0,
            Lifecycle::Active => 1,
            Lifecycle::Warmup => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Result<Lifecycle, ENNError> {
        match tag {
            0 => Ok(Lifecycle::DesignOnly),
            1 => Ok(Lifecycle::Active),
            2 => Ok(Lifecycle::Warmup),
            _ => Err(ENNError::InvalidParameter(
                "invalid frozen warmup mode".into(),
            )),
        }
    }

    pub(crate) fn hash_state(&self, buf: &mut Vec<u8>) {
        buf.push(match self.kind {
            InitStrategy::LHD => 0,
            InitStrategy::Random => 1,
        });
        buf.extend_from_slice(&self.budget.to_le_bytes());
        buf.extend_from_slice(&self.completed.to_le_bytes());
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.push(self.mode_tag());
        buf.extend_from_slice(&self.reinit_left.unwrap_or(usize::MAX).to_le_bytes());
    }
}

/// Result of an append-only observation batch.
#[derive(Debug, Clone)]
pub struct ObservationDelta {
    pub old_n: usize,
    pub new_n: usize,
    pub x_new: Array2<f64>,
    pub y_new: Array2<f64>,
}

impl ObservationDelta {
    pub fn x_view(&self) -> ArrayView2<'_, f64> {
        self.x_new.view()
    }

    pub fn y_view(&self) -> ArrayView2<'_, f64> {
        self.y_new.view()
    }

    /// Replay this observation delta by appending to a boxed surrogate.
    pub fn replay_append(
        &self,
        surrogate: BoxedSurrogate,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        surrogate.with_append(&self.x_view(), &self.y_view(), yvar, rng)
    }

    /// Replay this observation delta by fitting from scratch onto a cleared
    /// boxed surrogate.
    pub fn replay_fit(
        &self,
        surrogate: BoxedSurrogate,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<BoxedSurrogate, ENNError> {
        let cleared = surrogate.cleared()?;
        cleared.with_fit(&self.x_view(), &self.y_view(), yvar, rng)
    }

    /// Replay this observation delta onto a posterior store.
    pub fn replay_posterior(
        &self,
        posterior: PosteriorStore,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(PosteriorStore, Option<f64>), ENNError> {
        posterior.fit_delta(self, yvar, rng)
    }

    /// Restore a boxed surrogate from a checkpoint buffer.
    pub fn replay_thawed(
        surrogate: BoxedSurrogate,
        bytes: &[u8],
        pos: &mut usize,
    ) -> Result<BoxedSurrogate, ENNError> {
        surrogate.with_thawed(bytes, pos)
    }
}

pub(crate) fn observation_batch(
    old_n: usize,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) -> Result<ObservationDelta, ENNError> {
    let new_n = old_n.saturating_add(x.nrows());
    if new_n <= old_n {
        return Err(ENNError::InvalidParameter(
            "observation delta requires appended rows".to_string(),
        ));
    }
    Ok(ObservationDelta {
        old_n,
        new_n,
        x_new: x.to_owned(),
        y_new: y.to_owned(),
    })
}

pub(crate) fn freeze_state(opt: &BayesOptimizer) -> Result<Vec<u8>, ENNError> {
    freeze_full(opt, None)
}

pub(crate) fn freeze_full(
    opt: &BayesOptimizer,
    host: Option<&HostCursor>,
) -> Result<Vec<u8>, ENNError> {
    let mut buf = Vec::with_capacity(512);
    write_header(&mut buf);

    // 1. DomainFrame
    write_len(&mut buf, opt.domain.num_dim());
    write_bool(&mut buf, opt.domain.clamped());
    write_mat(&mut buf, opt.domain.bounds());

    // 2. WarmupSchedule
    write_byte(
        &mut buf,
        match opt.warmup.kind {
            InitStrategy::LHD => 0,
            InitStrategy::Random => 1,
        },
    );
    write_len(&mut buf, opt.warmup.budget);
    write_len(&mut buf, opt.warmup.completed);
    write_len(&mut buf, opt.warmup.generation);
    write_byte(&mut buf, opt.warmup.mode_tag());
    match opt.warmup.reinit_left {
        Some(left) => {
            write_bool(&mut buf, true);
            write_len(&mut buf, left);
        }
        None => write_bool(&mut buf, false),
    }

    // 3. ProposalEngine & SobolEngine
    write_word(&mut buf, opt.proposals.sobol_base);
    match opt.proposals.ticket_tag {
        Some(tag) => {
            write_bool(&mut buf, true);
            buf.extend_from_slice(&tag);
        }
        None => write_bool(&mut buf, false),
    }
    match opt.proposals.sobol_ref() {
        Some(eng) => {
            write_bool(&mut buf, true);
            write_word(&mut buf, eng.consumed());
            match eng.shift() {
                Some(shift) => {
                    write_bool(&mut buf, true);
                    write_slice(&mut buf, shift);
                }
                None => write_bool(&mut buf, false),
            }
        }
        None => write_bool(&mut buf, false),
    }

    // 4. PosteriorStore
    write_byte(
        &mut buf,
        match opt.posterior.yvar_mode {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
    );
    match &opt.posterior.backend {
        PosteriorBackend::Empirical { x_log, y_log } => {
            write_byte(&mut buf, 0);
            let n = x_log.nrows();
            let m = if n > 0 { y_log.ncols() } else { 0 };
            write_len(&mut buf, n);
            write_len(&mut buf, m);
            for i in 0..n {
                for &v in x_log.row_slice(i) {
                    write_real(&mut buf, v);
                }
            }
            for i in 0..n {
                for &v in y_log.row_slice(i) {
                    write_real(&mut buf, v);
                }
            }
        }
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        PosteriorBackend::Disk { .. } => {
            write_byte(&mut buf, 1);
            let n = opt.posterior.obs_count();
            let m = if n > 0 {
                opt.posterior.y_slice(0)?.len()
            } else {
                0
            };
            write_len(&mut buf, n);
            write_len(&mut buf, m);
            for i in 0..n {
                for &v in opt.posterior.x_slice(i)?.iter() {
                    write_real(&mut buf, v);
                }
            }
            for i in 0..n {
                for &v in opt.posterior.y_slice(i)?.iter() {
                    write_real(&mut buf, v);
                }
            }
        }
        PosteriorBackend::Enn(enn) => {
            write_byte(&mut buf, 2);
            <ENNSurrogate as Surrogate>::freeze_surrogate(enn.as_ref(), &mut buf)?;
        }
        PosteriorBackend::External(surrogate) => {
            write_byte(&mut buf, 2);
            surrogate.freeze_surrogate(&mut buf)?;
        }
    }

    // 5. IncumbentLedger
    match opt.incumbent.current_point() {
        Some(pt) => {
            write_bool(&mut buf, true);
            write_len(&mut buf, pt.index);
            write_arr(&mut buf, &pt.unit_x);
            write_arr(&mut buf, &pt.y_scalar);
        }
        None => write_bool(&mut buf, false),
    }

    // 6. TrustRegionState
    opt.regions.freeze_regions(&mut buf);

    // 7. RegionalModels
    match &opt.local_models {
        Some(models) => {
            write_bool(&mut buf, true);
            models.freeze_models(&mut buf)?;
        }
        None => write_bool(&mut buf, false),
    }

    // 8. Legacy ingestion-error wire flag (always false under pure FBIP state)
    write_bool(&mut buf, false);

    // 9. Optional HostCursor
    match host {
        Some(h) => {
            write_bool(&mut buf, true);
            write_word(&mut buf, h.seed);
            write_word(&mut buf, h.epoch);
            write_bool(&mut buf, h.identified);
            let mut pairs = h.id_map.clone();
            pairs.sort_unstable_by_key(|&(orig, _)| orig);
            write_len(&mut buf, pairs.len());
            for &(orig, cur) in &pairs {
                write_word(&mut buf, orig);
                write_word(&mut buf, cur);
            }
        }
        None => write_bool(&mut buf, false),
    }

    seal_frame(&mut buf);
    Ok(buf)
}

pub(crate) fn thaw_state(opt: &BayesOptimizer, bytes: &[u8]) -> Result<BayesOptimizer, ENNError> {
    thaw_full(opt, bytes).map(|(next, _)| next)
}

pub(crate) fn thaw_full(
    opt: &BayesOptimizer,
    bytes: &[u8],
) -> Result<(BayesOptimizer, Option<HostCursor>), ENNError> {
    thaw_inner(opt.fork()?, bytes)
}

pub(crate) fn thaw_consumed(
    opt: BayesOptimizer,
    bytes: &[u8],
) -> Result<(BayesOptimizer, Option<HostCursor>), ENNError> {
    thaw_inner(opt, bytes)
}

fn decode_reals(raw: &[u8], label: &str) -> Result<Vec<f64>, ENNError> {
    let count = raw.len() / 8;
    let mut vals = Vec::new();
    vals.try_reserve_exact(count)
        .map_err(|_| bad_state("allocation failed"))?;
    for chunk in raw.chunks_exact(8) {
        let mut arr = [0u8; 8];
        arr.copy_from_slice(chunk);
        let v = f64::from_bits(u64::from_le_bytes(arr));
        if !v.is_finite() {
            return Err(bad_state(format!("non-finite {label} in frozen state")));
        }
        vals.push(v);
    }
    Ok(vals)
}

fn thaw_inner(
    mut next: BayesOptimizer,
    bytes: &[u8],
) -> Result<(BayesOptimizer, Option<HostCursor>), ENNError> {
    let mut reader = WireReader::open_frame(bytes)?;
    let payload = &bytes[..bytes.len() - 32];

    // 1. DomainFrame compatibility
    let num_dim = reader.read_len(MAX_DIM)?;
    if num_dim != next.domain.num_dim() {
        return Err(bad_state("frozen state dimension mismatch"));
    }
    let clamped = reader.read_bool()?;
    if clamped != next.domain.clamped() {
        return Err(bad_state("frozen state domain clamp mismatch"));
    }
    let bounds = reader.read_mat(num_dim, 2, true)?;
    if bounds.shape() != [num_dim, 2]
        || !bounds
            .iter()
            .zip(next.domain.bounds().iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    {
        return Err(bad_state("frozen state domain bounds mismatch"));
    }

    // 2. WarmupSchedule
    let kind = match reader.read_byte()? {
        0 => InitStrategy::LHD,
        1 => InitStrategy::Random,
        _ => return Err(bad_state("invalid frozen InitStrategy tag")),
    };
    if kind != next.warmup.kind {
        return Err(bad_state("frozen state warmup strategy mismatch"));
    }
    let budget = reader.read_len(MAX_OBS)?;
    let completed = reader.read_len(MAX_OBS)?;
    let generation = reader.read_len(MAX_OBS)?;
    let mode = WarmupSchedule::from_tag(reader.read_byte()?)?;
    let reinit_left = if reader.read_bool()? {
        Some(reader.read_len(MAX_OBS)?)
    } else {
        None
    };

    // 3. ProposalEngine & SobolEngine
    let sobol_base = reader.read_word()?;
    let ticket_tag = if reader.read_bool()? {
        let mut tag = [0u8; 32];
        tag.copy_from_slice(reader.read_bytes(32)?);
        Some(tag)
    } else {
        None
    };
    let has_sobol = reader.read_bool()?;
    if has_sobol != next.proposals.sobol_ref().is_some() {
        return Err(bad_state("frozen state candidate sampler mismatch"));
    }
    let sobol_eng = if has_sobol {
        let consumed = reader.read_word()?;
        let shift = if reader.read_bool()? {
            Some(reader.read_vec(num_dim, true)?)
        } else {
            None
        };
        Some(SobolEngine::restore(num_dim, consumed, shift)?)
    } else {
        None
    };

    // 4. PosteriorStore
    let yvar_mode = match reader.read_byte()? {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => return Err(bad_state("invalid frozen yvar_mode tag")),
    };
    let backend_tag = reader.read_byte()?;
    match backend_tag {
        0 => {
            if let PosteriorBackend::Empirical { .. } = next.posterior.backend {
                let n = reader.read_len(MAX_OBS)?;
                let m = reader.read_len(MAX_METRICS)?;
                if (n == 0 && m != 0) || (n > 0 && m == 0) {
                    return Err(bad_state("invalid frozen empirical observation shape"));
                }
                let x_total = n
                    .checked_mul(num_dim)
                    .ok_or_else(|| bad_state("empirical x overflow"))?;
                let y_total = n
                    .checked_mul(m)
                    .ok_or_else(|| bad_state("empirical y overflow"))?;
                let x_bytes = x_total
                    .checked_mul(8)
                    .ok_or_else(|| bad_state("empirical x byte overflow"))?;
                let y_bytes = y_total
                    .checked_mul(8)
                    .ok_or_else(|| bad_state("empirical y byte overflow"))?;
                let raw_x = reader.read_bytes(x_bytes)?;
                let raw_y = reader.read_bytes(y_bytes)?;
                let x_vals = decode_reals(raw_x, "empirical x")?;
                let y_vals = decode_reals(raw_y, "empirical y")?;
                let x_log = if n > 0 {
                    ImLog::from_array(
                        Array2::from_shape_vec((n, num_dim), x_vals)
                            .map_err(|e| bad_state(e.to_string()))?,
                    )
                } else {
                    ImLog::with_cols(num_dim)
                };
                let y_log = if n > 0 {
                    ImLog::from_array(
                        Array2::from_shape_vec((n, m), y_vals)
                            .map_err(|e| bad_state(e.to_string()))?,
                    )
                } else {
                    ImLog::empty()
                };
                next.posterior.backend = PosteriorBackend::Empirical { x_log, y_log };
            } else {
                return Err(bad_state(
                    "frozen posterior backend does not match template optimizer",
                ));
            }
        }
        #[cfg(all(feature = "disk-bpann", not(target_arch = "wasm32")))]
        1 => {
            if let PosteriorBackend::Disk { .. } = next.posterior.backend {
                let n = reader.read_len(MAX_OBS)?;
                let m = reader.read_len(MAX_METRICS)?;
                if (n == 0 && m != 0) || (n > 0 && m == 0) {
                    return Err(bad_state("invalid frozen disk observation shape"));
                }
                next.posterior = next.posterior.clear()?;
                if n > 0 {
                    let x_total = n
                        .checked_mul(num_dim)
                        .ok_or_else(|| bad_state("disk x overflow"))?;
                    let y_total = n
                        .checked_mul(m)
                        .ok_or_else(|| bad_state("disk y overflow"))?;
                    let raw_x = reader.read_bytes(
                        x_total
                            .checked_mul(8)
                            .ok_or_else(|| bad_state("disk x byte overflow"))?,
                    )?;
                    let raw_y = reader.read_bytes(
                        y_total
                            .checked_mul(8)
                            .ok_or_else(|| bad_state("disk y byte overflow"))?,
                    )?;
                    let x_vals = decode_reals(raw_x, "disk x")?;
                    let y_vals = decode_reals(raw_y, "disk y")?;
                    let x_mat = Array2::from_shape_vec((n, num_dim), x_vals)
                        .map_err(|e| bad_state(e.to_string()))?;
                    let y_mat = Array2::from_shape_vec((n, m), y_vals)
                        .map_err(|e| bad_state(e.to_string()))?;
                    next.posterior.append_disk(&x_mat.view(), &y_mat.view())?;
                }
            } else {
                return Err(bad_state(
                    "frozen posterior backend does not match template optimizer",
                ));
            }
        }
        2 => match next.posterior.backend {
            PosteriorBackend::Enn(enn) => {
                let mut pos = reader.position();
                let inner = Arc::try_unwrap(enn).unwrap_or_else(|arc| (*arc).clone());
                let thawed = inner.with_thawed(payload, &mut pos)?;
                reader = WireReader::from_slice(payload, pos);
                next.posterior.backend = PosteriorBackend::Enn(Arc::new(thawed));
            }
            PosteriorBackend::External(surrogate) => {
                let mut pos = reader.position();
                let boxed = match Arc::try_unwrap(surrogate) {
                    Ok(b) => b,
                    Err(arc) => arc.clone_box()?,
                };
                let thawed = boxed.with_thawed(payload, &mut pos)?;
                reader = WireReader::from_slice(payload, pos);
                next.posterior.backend = PosteriorBackend::External(Arc::new(thawed));
            }
            _ => {
                return Err(bad_state(
                    "frozen posterior backend does not match template optimizer",
                ));
            }
        },
        _ => {
            return Err(bad_state(
                "frozen posterior backend does not match template optimizer",
            ));
        }
    }
    next.posterior.yvar_mode = yvar_mode;

    // 5. IncumbentLedger
    next.incumbent = next.incumbent.reset();
    if let Some(y_obs) = next.posterior.y_obs() {
        next.incumbent = next.incumbent.rebuild(&y_obs.view());
    }
    let obs_count = next.posterior.obs_count();
    let current = if reader.read_bool()? {
        let index = reader.read_len(MAX_OBS)?;
        if index >= obs_count {
            return Err(bad_state("frozen incumbent index out of range"));
        }
        let unit_x = reader.read_arr(num_dim, true)?;
        let y_vec = reader.read_vec(MAX_METRICS, true)?;
        if y_vec.is_empty() {
            return Err(bad_state("empty frozen incumbent y_scalar"));
        }
        Some(IncumbentPoint {
            index,
            unit_x,
            y_scalar: Array1::from_vec(y_vec),
        })
    } else {
        None
    };
    next.incumbent = next.incumbent.set_current(current);

    // 6. TrustRegionState
    let regions = Arc::try_unwrap(next.regions).unwrap_or_else(|arc| (*arc).clone());
    next.regions = Arc::new(regions.thaw_regions(&mut reader)?);

    // 7. RegionalModels
    let has_local = reader.read_bool()?;
    match (has_local, next.local_models.take()) {
        (true, Some(models)) => {
            let mut pos = reader.position();
            let thawed_models = models.thaw_models(num_dim, payload, &mut pos)?;
            reader = WireReader::from_slice(payload, pos);
            next.local_models = Some(thawed_models);
        }
        (false, None) => {
            next.local_models = None;
        }
        _ => {
            return Err(bad_state(
                "frozen regional-model configuration does not match template optimizer",
            ));
        }
    }

    // 8. Legacy ingestion-error wire flag
    if reader.read_bool()? {
        let err = reader.read_str(MAX_TEXT)?;
        return Err(bad_state(format!(
            "frozen state carries ingestion failure: {err}"
        )));
    }

    // 9. Optional HostCursor
    let host = if reader.read_bool()? {
        let seed = reader.read_word()?;
        if seed >= (1u64 << 63) {
            return Err(bad_state("frozen host seed exceeds 63-bit range"));
        }
        let epoch = reader.read_word()?;
        let identified = reader.read_bool()?;
        let map_len = reader.read_len(MAX_PENDING)?;
        let pair_bytes = reader.read_bytes(
            map_len
                .checked_mul(16)
                .ok_or_else(|| bad_state("host id_map byte overflow"))?,
        )?;
        let mut id_map = Vec::new();
        id_map
            .try_reserve_exact(map_len)
            .map_err(|_| bad_state("allocation failed"))?;
        for chunk in pair_bytes.chunks_exact(16) {
            let mut a = [0u8; 8];
            let mut b = [0u8; 8];
            a.copy_from_slice(&chunk[..8]);
            b.copy_from_slice(&chunk[8..]);
            id_map.push((u64::from_le_bytes(a), u64::from_le_bytes(b)));
        }
        Some(HostCursor {
            seed,
            epoch,
            identified,
            id_map,
        })
    } else {
        None
    };

    reader.finish()?;

    next.warmup.budget = budget;
    next.warmup.completed = completed;
    next.warmup.generation = generation;
    next.warmup.mode = mode;
    next.warmup.reinit_left = reinit_left;
    next.proposals.sobol_base = sobol_base;
    next.proposals.ticket_tag = ticket_tag;
    next.proposals.set_sobol(sobol_eng);

    Ok((next, host))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn observation_appends() {
        let x = array![[1.0, 0.0]];
        let y = array![[2.0]];
        let delta = observation_batch(1, &x.view(), &y.view()).unwrap();
        assert_eq!(delta.old_n, 1);
        assert_eq!(delta.new_n, 2);
        assert_eq!(delta.x_new.nrows(), 1);
        assert_eq!(delta.y_new.nrows(), 1);
        let _ = delta.x_view();
        let _ = delta.y_view();
        assert!(std::mem::size_of::<ObservationDelta>() > 0);
    }
}
