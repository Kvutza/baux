use ndarray::{Array1, Array2, ArrayView2};

use super::BayesOptimizer;
use super::incumbent::IncumbentPoint;
use super::obs_access::PosteriorBackend;
use crate::candidates::SobolEngine;
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::snapshot_io::{
    MAX_DIM, MAX_METRICS, MAX_OBS, MAX_PENDING, MAX_TEXT, WireReader, bad_state, seal_frame,
    write_arr, write_bool, write_byte, write_header, write_len, write_mat, write_real, write_slice,
    write_str, write_word,
};

/// Host-language pure state cursor metadata embedded in portable checkpoints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostCursor {
    pub seed: u64,
    pub epoch: u64,
    pub identified: bool,
    pub id_map: Vec<(u64, u64)>,
}

/// Phase policy governing whether the optimizer stays in space-filling design
/// or enters active search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmupMode {
    /// Space-filling design only (e.g. LHD-only).
    DesignOnly,
    /// Warmup followed by active trust-region search.
    ActiveSearch { in_init: bool },
}

/// Unified lifecycle state for initial warmup, post-restart local warmup, and
/// generation tracking.
#[derive(Debug, Clone)]
pub struct WarmupSchedule {
    pub kind: InitStrategy,
    pub budget: usize,
    pub completed: usize,
    pub mode: WarmupMode,
    pub reinit_left: Option<usize>,
    pub generation: usize,
}

impl WarmupSchedule {
    pub fn design_only(kind: InitStrategy, budget: usize, completed: usize) -> Self {
        Self {
            kind,
            budget,
            completed,
            mode: WarmupMode::DesignOnly,
            reinit_left: None,
            generation: 0,
        }
    }

    pub fn active(kind: InitStrategy, budget: usize, completed: usize, in_init: bool) -> Self {
        Self {
            kind,
            budget,
            completed,
            mode: WarmupMode::ActiveSearch { in_init },
            reinit_left: None,
            generation: 0,
        }
    }

    pub fn progress(&self) -> Option<(usize, usize)> {
        match self.mode {
            WarmupMode::DesignOnly => Some((self.completed, self.budget)),
            WarmupMode::ActiveSearch { in_init: true } => Some((self.completed, self.budget)),
            WarmupMode::ActiveSearch { in_init: false } => None,
        }
    }

    pub fn in_init(&self) -> bool {
        matches!(
            self.mode,
            WarmupMode::DesignOnly | WarmupMode::ActiveSearch { in_init: true }
        )
    }

    pub fn advance_init(&mut self, rows: usize) {
        self.completed += rows;
    }

    pub fn check_transition(&mut self) {
        if let WarmupMode::ActiveSearch { in_init } = &mut self.mode {
            if *in_init && self.completed >= self.budget {
                *in_init = false;
            }
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
            WarmupMode::DesignOnly => 0,
            WarmupMode::ActiveSearch { in_init: false } => 1,
            WarmupMode::ActiveSearch { in_init: true } => 2,
        }
    }

    pub(crate) fn from_tag(tag: u8) -> Result<WarmupMode, ENNError> {
        match tag {
            0 => Ok(WarmupMode::DesignOnly),
            1 => Ok(WarmupMode::ActiveSearch { in_init: false }),
            2 => Ok(WarmupMode::ActiveSearch { in_init: true }),
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
        PosteriorBackend::Empirical { x_rows, y_rows } => {
            write_byte(&mut buf, 0);
            let n = x_rows.len();
            let m = y_rows.first().map_or(0, Array1::len);
            write_len(&mut buf, n);
            write_len(&mut buf, m);
            for row in x_rows {
                for &v in row.iter() {
                    write_real(&mut buf, v);
                }
            }
            for row in y_rows {
                for &v in row.iter() {
                    write_real(&mut buf, v);
                }
            }
        }
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
        PosteriorBackend::Model(surrogate) => {
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

    // 8. Ingestion error
    match &opt.ingestion_error {
        Some(err) => {
            write_bool(&mut buf, true);
            write_str(&mut buf, err);
        }
        None => write_bool(&mut buf, false),
    }

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
    let mut reader = WireReader::open_frame(bytes)?;
    let payload = &bytes[..bytes.len() - 32];

    // 1. DomainFrame compatibility
    let num_dim = reader.read_len(MAX_DIM)?;
    if num_dim != opt.domain.num_dim() {
        return Err(bad_state("frozen state dimension mismatch"));
    }
    let clamped = reader.read_bool()?;
    if clamped != opt.domain.clamped() {
        return Err(bad_state("frozen state domain clamp mismatch"));
    }
    let bounds = reader.read_mat(num_dim, 2, true)?;
    if bounds.shape() != [num_dim, 2]
        || !bounds
            .iter()
            .zip(opt.domain.bounds().iter())
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
    if kind != opt.warmup.kind {
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
    if has_sobol != opt.proposals.sobol_ref().is_some() {
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

    let mut next = opt.fork()?;

    // 4. PosteriorStore
    let yvar_mode = match reader.read_byte()? {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => return Err(bad_state("invalid frozen yvar_mode tag")),
    };
    let backend_tag = reader.read_byte()?;
    match (backend_tag, &mut next.posterior.backend) {
        (0, PosteriorBackend::Empirical { x_rows, y_rows }) => {
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
            let mut new_x = Vec::new();
            new_x
                .try_reserve_exact(n)
                .map_err(|_| bad_state("allocation failed"))?;
            let mut new_y = Vec::new();
            new_y
                .try_reserve_exact(n)
                .map_err(|_| bad_state("allocation failed"))?;
            for row_bytes in raw_x.chunks_exact(num_dim * 8) {
                let mut row = Vec::new();
                row.try_reserve_exact(num_dim)
                    .map_err(|_| bad_state("allocation failed"))?;
                for chunk in row_bytes.chunks_exact(8) {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(chunk);
                    let v = f64::from_bits(u64::from_le_bytes(arr));
                    if !v.is_finite() {
                        return Err(bad_state("non-finite empirical x in frozen state"));
                    }
                    row.push(v);
                }
                new_x.push(Array1::from_vec(row));
            }
            if n > 0 {
                for row_bytes in raw_y.chunks_exact(m * 8) {
                    let mut row = Vec::new();
                    row.try_reserve_exact(m)
                        .map_err(|_| bad_state("allocation failed"))?;
                    for chunk in row_bytes.chunks_exact(8) {
                        let mut arr = [0u8; 8];
                        arr.copy_from_slice(chunk);
                        let v = f64::from_bits(u64::from_le_bytes(arr));
                        if !v.is_finite() {
                            return Err(bad_state("non-finite empirical y in frozen state"));
                        }
                        row.push(v);
                    }
                    new_y.push(Array1::from_vec(row));
                }
            }
            *x_rows = new_x;
            *y_rows = new_y;
        }
        (1, PosteriorBackend::Disk { .. }) => {
            let n = reader.read_len(MAX_OBS)?;
            let m = reader.read_len(MAX_METRICS)?;
            if (n == 0 && m != 0) || (n > 0 && m == 0) {
                return Err(bad_state("invalid frozen disk observation shape"));
            }
            next.posterior.clear()?;
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
                let mut x_vals = Vec::new();
                x_vals
                    .try_reserve_exact(x_total)
                    .map_err(|_| bad_state("allocation failed"))?;
                for chunk in raw_x.chunks_exact(8) {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(chunk);
                    let v = f64::from_bits(u64::from_le_bytes(arr));
                    if !v.is_finite() {
                        return Err(bad_state("non-finite disk x in frozen state"));
                    }
                    x_vals.push(v);
                }
                let mut y_vals = Vec::new();
                y_vals
                    .try_reserve_exact(y_total)
                    .map_err(|_| bad_state("allocation failed"))?;
                for chunk in raw_y.chunks_exact(8) {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(chunk);
                    let v = f64::from_bits(u64::from_le_bytes(arr));
                    if !v.is_finite() {
                        return Err(bad_state("non-finite disk y in frozen state"));
                    }
                    y_vals.push(v);
                }
                let x_mat = Array2::from_shape_vec((n, num_dim), x_vals)
                    .map_err(|e| bad_state(e.to_string()))?;
                let y_mat =
                    Array2::from_shape_vec((n, m), y_vals).map_err(|e| bad_state(e.to_string()))?;
                next.posterior.append_disk(&x_mat.view(), &y_mat.view())?;
            }
        }
        (2, PosteriorBackend::Model(surrogate)) => {
            let mut pos = reader.position();
            surrogate.thaw_surrogate(payload, &mut pos)?;
            reader = WireReader::from_slice(payload, pos);
        }
        _ => {
            return Err(bad_state(
                "frozen posterior backend does not match template optimizer",
            ));
        }
    }
    next.posterior.yvar_mode = yvar_mode;

    // 5. IncumbentLedger
    next.incumbent.reset();
    if let Some(y_obs) = next.posterior.y_obs() {
        next.incumbent.tracker.rebuild(&y_obs.view());
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
    next.incumbent.set_current(current);

    // 6. TrustRegionState
    next.regions.thaw_regions(&mut reader)?;

    // 7. RegionalModels
    let has_local = reader.read_bool()?;
    match (has_local, next.local_models.as_mut()) {
        (true, Some(models)) => {
            let mut pos = reader.position();
            models.thaw_models(num_dim, payload, &mut pos)?;
            reader = WireReader::from_slice(payload, pos);
        }
        (false, None) => {}
        _ => {
            return Err(bad_state(
                "frozen regional-model configuration does not match template optimizer",
            ));
        }
    }

    // 8. Ingestion error
    let ingestion_error = if reader.read_bool()? {
        Some(reader.read_str(MAX_TEXT)?)
    } else {
        None
    };

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
    next.ingestion_error = ingestion_error;

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
