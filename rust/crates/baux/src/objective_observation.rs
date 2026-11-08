//! Bounded outcome vectors, separate from the scalar decision/control policy.
//!
//! Objective positions belong to the caller's fixed schema. No normalization,
//! scalarization, covariance estimate, or preference is inferred here.

/// Compile-time bound on independently measured objective channels.
pub const MAX_OBJECTIVES: usize = 8;
/// The resident optimizer's bounded observation window, not model parameters.
pub const OBJECTIVE_CAPACITY: usize = baux::semantics::neighbor::MAX_HISTORY;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObjectiveEstimate {
    pub mean: f32,
    pub variance: f32,
}

impl ObjectiveEstimate {
    pub fn validate(self) -> Result<(), String> {
        if !self.mean.is_finite() || !self.variance.is_finite() || self.variance < 0.0 {
            return Err(
                "Objective means must be finite and variances finite and nonnegative".into(),
            );
        }
        Ok(())
    }
}

/// A measured vector and an explicitly supplied scalar control observation.
/// The control is not computed from the vector by this representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveObservation {
    estimates: [ObjectiveEstimate; MAX_OBJECTIVES],
    width: u8,
    control: ObjectiveEstimate,
}

impl ObjectiveObservation {
    pub fn new(
        estimates: &[ObjectiveEstimate],
        control: ObjectiveEstimate,
    ) -> Result<Self, String> {
        if estimates.is_empty() || estimates.len() > MAX_OBJECTIVES {
            return Err(format!("Objective count must be in 1..={MAX_OBJECTIVES}"));
        }
        let mut observation = Self {
            estimates: [ObjectiveEstimate::default(); MAX_OBJECTIVES],
            width: estimates.len() as u8,
            control,
        };
        observation.estimates[..estimates.len()].copy_from_slice(estimates);
        observation.validate()?;
        Ok(observation)
    }

    pub fn scalar(mean: f32, variance: f32) -> Result<Self, String> {
        let observation = Self::scalar_adapter(mean, variance);
        observation.validate()?;
        Ok(observation)
    }

    /// Internal adapter defers validation to the existing tell boundary, so
    /// invalid scalar calls retain their established state/error ordering.
    pub fn scalar_adapter(mean: f32, variance: f32) -> Self {
        let control = ObjectiveEstimate { mean, variance };
        let mut estimates = [ObjectiveEstimate::default(); MAX_OBJECTIVES];
        estimates[0] = control;
        Self {
            estimates,
            width: 1,
            control,
        }
    }

    pub fn estimates(&self) -> &[ObjectiveEstimate] {
        &self.estimates[..usize::from(self.width)]
    }

    pub fn is_vector(&self) -> bool {
        self.width > 1
    }

    pub fn control(self) -> ObjectiveEstimate {
        self.control
    }

    pub fn validate(self) -> Result<(), String> {
        self.control.validate()?;
        for estimate in self.estimates() {
            estimate.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveRecord {
    pub identity: i64,
    pub observation: ObjectiveObservation,
}

/// Fixed storage owned by the resident search. Recording and borrowing rows
/// allocate nothing; FIFO retention never retains dense parameter vectors.
pub struct ObjectiveWindow {
    rows: [Option<ObjectiveRecord>; OBJECTIVE_CAPACITY],
    len: usize,
    capacity: usize,
    width: usize,
    incumbent: Option<ObjectiveRecord>,
}

impl ObjectiveWindow {
    pub fn new(capacity: usize) -> Self {
        assert!((1..=OBJECTIVE_CAPACITY).contains(&capacity));
        Self {
            rows: [None; OBJECTIVE_CAPACITY],
            len: 0,
            capacity,
            width: 0,
            incumbent: None,
        }
    }

    pub fn validate(&self, observation: ObjectiveObservation) -> Result<(), String> {
        observation.validate()?;
        if self.width != 0 && self.width != observation.estimates().len() {
            return Err("Objective schema width cannot change within a resident search".into());
        }
        Ok(())
    }

    pub fn record(&mut self, identity: i64, observation: ObjectiveObservation, accepted: bool) {
        let record = ObjectiveRecord {
            identity,
            observation,
        };
        self.width = observation.estimates().len();
        if self.len == self.capacity {
            self.rows[..self.capacity].rotate_left(1);
        } else {
            self.len += 1;
        }
        self.rows[self.len - 1] = Some(record);
        if accepted {
            self.incumbent = Some(record);
        }
    }

    pub fn record_slot(
        &mut self,
        slot: usize,
        identity: i64,
        observation: ObjectiveObservation,
        accepted: bool,
    ) {
        let record = ObjectiveRecord {
            identity,
            observation,
        };
        self.width = observation.estimates().len();
        self.rows[slot] = Some(record);
        self.len = self.len.max(slot + 1).min(self.capacity);
        if accepted {
            self.incumbent = Some(record);
        }
    }

    pub fn rows(&self) -> impl ExactSizeIterator<Item = &ObjectiveRecord> {
        self.rows[..self.len]
            .iter()
            .map(|row| row.as_ref().unwrap())
    }

    pub fn incumbent(&self) -> Option<&ObjectiveRecord> {
        self.incumbent.as_ref()
    }

    pub fn adopt(&mut self, identity: i64, observation: ObjectiveObservation) {
        self.incumbent = Some(ObjectiveRecord {
            identity,
            observation,
        });
    }

    pub fn retain_incumbent(&mut self) {
        self.rows.fill(None);
        self.rows[0] = self.incumbent;
        self.len = usize::from(self.incumbent.is_some());
    }
}

/// Stack-allocated const-generic multi-objective observation of compile-time
/// arity `M` ($1 \le M \le 8$).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstObservation<const M: usize> {
    pub estimates: [ObjectiveEstimate; M],
    pub control: ObjectiveEstimate,
}

impl<const M: usize> ConstObservation<M> {
    const VALID_ARITY: () = assert!(M >= 1 && M <= MAX_OBJECTIVES);

    pub fn new(
        estimates: [ObjectiveEstimate; M],
        control: ObjectiveEstimate,
    ) -> Result<Self, String> {
        let () = Self::VALID_ARITY;
        control.validate()?;
        for est in &estimates {
            est.validate()?;
        }
        Ok(Self { estimates, control })
    }

    pub fn noiseless(means: [f32; M], control_mean: f32) -> Result<Self, String> {
        let estimates = means.map(|mean| ObjectiveEstimate {
            mean,
            variance: 0.0,
        });
        Self::new(
            estimates,
            ObjectiveEstimate {
                mean: control_mean,
                variance: 0.0,
            },
        )
    }

    /// Pareto dominance check parameterized by a per-channel
    /// [`crate::sense::Sense`] array.
    pub fn dominates(&self, other: &Self, senses: &[crate::sense::Sense; M]) -> bool {
        use crate::sense::ObjectiveSense;
        let mut strictly_better = false;
        for i in 0..M {
            let sa = senses[i].score(f64::from(self.estimates[i].mean));
            let sb = senses[i].score(f64::from(other.estimates[i].mean));
            if sa < sb {
                return false;
            }
            if sa > sb {
                strictly_better = true;
            }
        }
        strictly_better
    }

    /// Stack-allocated augmented Chebyshev scalarization in canonical utility
    /// space.
    pub fn chebyshev(
        &self,
        utopia: &[f32; M],
        weights: &[f32; M],
        senses: &[crate::sense::Sense; M],
    ) -> f32 {
        use crate::sense::ObjectiveSense;
        let mut worst_gap = 0.0_f32;
        let mut sum_gap = 0.0_f32;
        for i in 0..M {
            let u = senses[i].score(f64::from(utopia[i])).raw() as f32;
            let y = senses[i].score(f64::from(self.estimates[i].mean)).raw() as f32;
            let gap = weights[i] * (u - y).max(0.0);
            worst_gap = worst_gap.max(gap);
            sum_gap += gap;
        }
        -(worst_gap + 1e-3 * sum_gap)
    }

    pub fn into_dynamic(self) -> ObjectiveObservation {
        let () = Self::VALID_ARITY;
        let mut estimates = [ObjectiveEstimate::default(); MAX_OBJECTIVES];
        estimates[..M].copy_from_slice(&self.estimates);
        ObjectiveObservation {
            estimates,
            width: M as u8,
            control: self.control,
        }
    }
}

impl<const M: usize> From<ConstObservation<M>> for ObjectiveObservation {
    fn from(value: ConstObservation<M>) -> Self {
        value.into_dynamic()
    }
}

#[cfg(test)]
mod tests;
