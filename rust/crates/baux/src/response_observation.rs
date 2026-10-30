//! Exact bookkeeping for one black-box model evaluation.
//!
//! The resident ENN models absolute outcomes at model states.  A response is a
//! separate comparison between a candidate and an explicitly named baseline.
//! Keeping the two objects distinct prevents a surrogate estimate from being
//! mislabeled as a paired function evaluation.

use crate::objective_observation::ObjectiveEstimate;
use deser::Serialize;

pub const RESPONSE_SCHEMA: &str = "baux.response-observation.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[deser(rename_all = "kebab-case")]
pub enum ResponseReference {
    Initial,
    PairedIncumbent,
    SurrogateIncumbent,
    VectorPolicy,
}

/// Realized displacement after the proposal has passed through the operator.
///
/// `metric_radius` is the norm used by the resident search.  The unweighted
/// squared displacement is reported separately because the two have different
/// units and must not be substituted for one another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProposalMeasure {
    pub seed: u64,
    pub candidate: u32,
    pub coordinates: u64,
    pub changed_coordinates: u64,
    pub squared_displacement: f64,
    pub metric_radius: f32,
    pub trust_radius: f32,
}

impl ProposalMeasure {
    pub fn validate(self) -> Result<Self, String> {
        if self.coordinates == 0 || self.changed_coordinates > self.coordinates {
            return Err("Proposal coordinate counts are inconsistent".into());
        }
        if !self.squared_displacement.is_finite()
            || self.squared_displacement < 0.0
            || !self.metric_radius.is_finite()
            || self.metric_radius < 0.0
            || !self.trust_radius.is_finite()
            || self.trust_radius < 0.0
        {
            return Err("Proposal displacement measurements must be finite and nonnegative".into());
        }
        Ok(self)
    }

    pub fn realized_survival(self) -> f64 {
        self.changed_coordinates as f64 / self.coordinates as f64
    }
}

/// Truth labels for a generation evaluation.  These counts describe distinct
/// events and are intentionally not collapsed into one generic token count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenAccounting {
    pub samples: u64,
    pub context: u64,
    pub prompt: u64,
    pub drafted: u64,
    pub evaluated: u64,
    pub repaired: u64,
    pub generated: u64,
    pub committed: u64,
}

impl TokenAccounting {
    pub fn validate(self) -> Result<Self, String> {
        let capacity = self
            .context
            .saturating_sub(self.prompt)
            .checked_mul(self.samples)
            .unwrap_or(0);
        if self.samples == 0
            || self.context < self.prompt
            || capacity < self.generated
            || self.repaired > self.evaluated
            || self.committed > self.generated
        {
            return Err("Generation token accounting is inconsistent".into());
        }
        Ok(self)
    }
}

/// One evaluated proposal and the reference used to interpret its outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResponseObservation {
    pub proposal: ProposalMeasure,
    pub reference: ResponseReference,
    pub candidate: ObjectiveEstimate,
    pub baseline: ObjectiveEstimate,
    pub improvement: ObjectiveEstimate,
    pub accepted: bool,
    pub tokens: Option<TokenAccounting>,
}

impl ResponseObservation {
    pub fn validate(self) -> Result<Self, String> {
        self.proposal.validate()?;
        self.candidate.validate()?;
        self.baseline.validate()?;
        self.improvement.validate()?;
        if let Some(tokens) = self.tokens {
            tokens.validate()?;
        }
        Ok(self)
    }
}
