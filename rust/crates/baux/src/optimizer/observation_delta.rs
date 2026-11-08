use ndarray::{Array2, ArrayView2};

use super::BayesOptimizer;
use crate::config::InitStrategy;
use crate::error::ENNError;

const FREEZE_MAGIC: &[u8; 8] = b"BAUXFRZ1";

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
        buf.extend_from_slice(&self.budget.to_le_bytes());
        buf.extend_from_slice(&self.completed.to_le_bytes());
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.push(u8::from(self.in_init()));
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

fn write_u64(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, ENNError> {
    let end = offset.saturating_add(8);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| ENNError::InvalidParameter("truncated frozen state".into()))?;
    *offset = end;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(slice);
    Ok(u64::from_le_bytes(arr))
}

fn read_matrix(
    bytes: &[u8],
    offset: &mut usize,
    rows: usize,
    cols: usize,
) -> Result<Array2<f64>, ENNError> {
    let total = rows.saturating_mul(cols);
    let mut vals = Vec::with_capacity(total);
    for _ in 0..total {
        vals.push(f64::from_bits(read_u64(bytes, offset)?));
    }
    Array2::from_shape_vec((rows, cols), vals)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub(crate) fn freeze_state(opt: &BayesOptimizer) -> Result<Vec<u8>, ENNError> {
    let n = opt.posterior.obs_count();
    let y_cols = if n > 0 {
        opt.posterior.y_slice(0)?.len()
    } else {
        0
    };
    let mut buf = Vec::with_capacity(113 + n * (opt.domain.num_dim() + y_cols) * 8);
    buf.extend_from_slice(FREEZE_MAGIC);
    write_u64(&mut buf, opt.domain.num_dim() as u64);
    write_u64(&mut buf, opt.warmup.budget as u64);
    write_u64(&mut buf, opt.warmup.completed as u64);
    write_u64(&mut buf, opt.warmup.generation as u64);
    buf.push(opt.warmup.mode_tag());
    write_u64(
        &mut buf,
        opt.warmup.reinit_left.map_or(u64::MAX, |v| v as u64),
    );
    write_u64(&mut buf, opt.proposals.sobol_base);
    write_u64(&mut buf, opt.regions.turbo_obs() as u64);
    write_u64(&mut buf, n as u64);
    write_u64(&mut buf, y_cols as u64);
    for i in 0..n {
        for &v in opt.posterior.x_slice(i)?.iter() {
            write_u64(&mut buf, v.to_bits());
        }
    }
    for i in 0..n {
        for &v in opt.posterior.y_slice(i)?.iter() {
            write_u64(&mut buf, v.to_bits());
        }
    }
    let tag = crate::artifact::blake3::hash(&buf);
    buf.extend_from_slice(&tag);
    Ok(buf)
}

fn verify_tag(bytes: &[u8]) -> Result<&[u8], ENNError> {
    if bytes.len() < 113 {
        return Err(ENNError::InvalidParameter(
            "frozen state buffer too short".into(),
        ));
    }
    let (payload, tag) = bytes.split_at(bytes.len() - 32);
    if crate::artifact::blake3::hash(payload).as_slice() != tag {
        return Err(ENNError::InvalidParameter(
            "frozen optimizer BLAKE3 integrity check failed".into(),
        ));
    }
    if &payload[..8] != FREEZE_MAGIC {
        return Err(ENNError::InvalidParameter(
            "invalid frozen state magic".into(),
        ));
    }
    Ok(payload)
}

fn restore_obs(
    next: &mut BayesOptimizer,
    x_unit: &Array2<f64>,
    y_mat: &Array2<f64>,
    turbo_obs: usize,
) -> Result<(), ENNError> {
    next.posterior.clear()?;
    next.incumbent.reset();
    next.regions.restart_local();
    next.proposals.ticket_tag = None;
    if x_unit.nrows() > 0 {
        let mut rng = baux::semantics::random::prng(next.proposals.sobol_base());
        let delta = next.add_observations(&x_unit.view(), &y_mat.view())?;
        next.posterior.fit_delta(&delta, None, &mut rng)?;
        let dim = next.domain.num_dim();
        let noisy = next.config.noise_aware;
        next.incumbent
            .update(&next.posterior, &next.regions, dim, noisy, &mut rng)?;
        if let Some(best) = next.incumbent.y_scalar().and_then(|y| y.first().copied()) {
            next.regions.seed_baseline(best, turbo_obs);
        } else {
            next.regions.set_obs(turbo_obs);
        }
    }
    Ok(())
}

pub(crate) fn thaw_state(opt: &BayesOptimizer, bytes: &[u8]) -> Result<BayesOptimizer, ENNError> {
    let payload = verify_tag(bytes)?;
    let mut offset = 8usize;
    let num_dim = read_u64(payload, &mut offset)? as usize;
    if num_dim != opt.domain.num_dim() {
        return Err(ENNError::InvalidParameter(
            "frozen state dimension mismatch".into(),
        ));
    }
    let budget = read_u64(payload, &mut offset)? as usize;
    let completed = read_u64(payload, &mut offset)? as usize;
    let generation = read_u64(payload, &mut offset)? as usize;
    let mode = WarmupSchedule::from_tag(
        *payload
            .get(offset)
            .ok_or_else(|| ENNError::InvalidParameter("truncated frozen mode".into()))?,
    )?;
    offset += 1;
    let reinit_raw = read_u64(payload, &mut offset)?;
    let sobol_base = read_u64(payload, &mut offset)?;
    let turbo_obs = read_u64(payload, &mut offset)? as usize;
    let n = read_u64(payload, &mut offset)? as usize;
    let y_cols = read_u64(payload, &mut offset)? as usize;
    let x_unit = read_matrix(payload, &mut offset, n, num_dim)?;
    let y_mat = read_matrix(payload, &mut offset, n, y_cols)?;
    if offset != payload.len() {
        return Err(ENNError::InvalidParameter(
            "trailing bytes in frozen state".into(),
        ));
    }
    let mut next = opt.fork()?;
    next.proposals.sobol_base = sobol_base;
    restore_obs(&mut next, &x_unit, &y_mat, turbo_obs)?;
    next.warmup.budget = budget;
    next.warmup.completed = completed;
    next.warmup.generation = generation;
    next.warmup.mode = mode;
    next.warmup.reinit_left = (reinit_raw != u64::MAX).then_some(reinit_raw as usize);
    Ok(next)
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
