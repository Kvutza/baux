use ndarray::{Array2, ArrayView2};

use crate::config::InitStrategy;
use crate::error::ENNError;

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
