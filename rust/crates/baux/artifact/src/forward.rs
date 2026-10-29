//! Exact, portable evidence from an initialized forward-program execution.
//! Raw FP16/FP32 bits preserve nonfinite failures instead of JSON-normalizing
//! them. Matching digests provide equality evidence for captured tensors,
//! not a proof of execution equivalence.

use deser::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct WeightHash {
    pub name: String,
    pub elements: u64,
    pub sha256: String,
}

pub struct WordHash {
    name: String,
    elements: u64,
    digest: Sha256,
}

impl WordHash {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            elements: 0,
            digest: Sha256::new(),
        }
    }

    pub fn elements(&self) -> u64 {
        self.elements
    }

    pub fn update(&mut self, words: &[u16]) -> Result<(), String> {
        self.elements = self
            .elements
            .checked_add(words.len() as u64)
            .ok_or("tensor digest length overflow")?;
        #[cfg(target_endian = "little")]
        {
            // SAFETY: u16 has no padding; the borrowed byte view remains valid
            // for the synchronous hash update. On-disk words are little-endian.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    words.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(words),
                )
            };
            self.digest.update(bytes);
        }
        #[cfg(target_endian = "big")]
        for word in words {
            self.digest.update(word.to_le_bytes());
        }
        Ok(())
    }

    pub fn finish(self) -> WeightHash {
        WeightHash {
            name: self.name,
            elements: self.elements,
            sha256: format!("{:x}", self.digest.finalize()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct ForwardTrace {
    pub format: String,
    pub seed: u64,
    pub input: Vec<u32>,
    pub labels: Vec<u32>,
    pub rows: usize,
    pub width: usize,
    pub weights: Vec<WeightHash>,
    pub encoded: Vec<u16>,
    pub hidden: Vec<u16>,
    pub predictions: Vec<u32>,
    pub loss_bits: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct BitDiff {
    pub elements: usize,
    pub mismatches: usize,
    pub reference_nonfinite: usize,
    pub candidate_nonfinite: usize,
    pub max_finite_ulp: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct IndexDiff {
    pub elements: usize,
    pub mismatches: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[deser(deny_unknown_fields)]
pub struct ForwardDiff {
    pub encoded: BitDiff,
    pub hidden: BitDiff,
    pub predictions: IndexDiff,
    pub losses: BitDiff,
}

impl ForwardDiff {
    pub fn exact(&self) -> bool {
        [&self.encoded, &self.hidden, &self.losses]
            .into_iter()
            .all(|diff| {
                diff.mismatches == 0
                    && diff.reference_nonfinite == 0
                    && diff.candidate_nonfinite == 0
            })
            && self.predictions.mismatches == 0
    }
}

fn ordered_u16(bits: u16) -> u16 {
    if bits & 0x8000 == 0 {
        bits ^ 0x8000
    } else {
        !bits
    }
}

fn ordered_u32(bits: u32) -> u32 {
    if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

fn words(reference: &[u16], candidate: &[u16]) -> BitDiff {
    let mut diff = BitDiff {
        elements: reference.len(),
        mismatches: 0,
        reference_nonfinite: 0,
        candidate_nonfinite: 0,
        max_finite_ulp: None,
    };
    for (&left, &right) in reference.iter().zip(candidate) {
        diff.mismatches += usize::from(left != right);
        let left_finite = left & 0x7c00 != 0x7c00;
        let right_finite = right & 0x7c00 != 0x7c00;
        diff.reference_nonfinite += usize::from(!left_finite);
        diff.candidate_nonfinite += usize::from(!right_finite);
        if left_finite && right_finite {
            let ulp = ordered_u16(left).abs_diff(ordered_u16(right)) as u64;
            diff.max_finite_ulp = Some(diff.max_finite_ulp.unwrap_or(0).max(ulp));
        }
    }
    diff
}

fn floats(reference: &[u32], candidate: &[u32]) -> BitDiff {
    let mut diff = BitDiff {
        elements: reference.len(),
        mismatches: 0,
        reference_nonfinite: 0,
        candidate_nonfinite: 0,
        max_finite_ulp: None,
    };
    for (&left, &right) in reference.iter().zip(candidate) {
        diff.mismatches += usize::from(left != right);
        let left_finite = left & 0x7f80_0000 != 0x7f80_0000;
        let right_finite = right & 0x7f80_0000 != 0x7f80_0000;
        diff.reference_nonfinite += usize::from(!left_finite);
        diff.candidate_nonfinite += usize::from(!right_finite);
        if left_finite && right_finite {
            let ulp = ordered_u32(left).abs_diff(ordered_u32(right)) as u64;
            diff.max_finite_ulp = Some(diff.max_finite_ulp.unwrap_or(0).max(ulp));
        }
    }
    diff
}

impl ForwardTrace {
    pub const FORMAT: &str = "baux.initialized-mhc4.v1";

    pub fn validate(&self) -> Result<(), String> {
        let elements = self
            .rows
            .checked_mul(self.width)
            .ok_or("forward trace shape overflow")?;
        if self.format != Self::FORMAT
            || self.rows == 0
            || self.width == 0
            || self.input.is_empty()
            || self.input.len() != self.labels.len()
            || self.predictions.len() != self.labels.len()
            || self.loss_bits.len() != self.predictions.len()
            || self.encoded.len() != elements
            || self.hidden.len() != elements
            || self.weights.is_empty()
        {
            return Err("forward trace has inconsistent format or dimensions".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for weight in &self.weights {
            if weight.name.is_empty()
                || weight.elements == 0
                || !names.insert(&weight.name)
                || weight.sha256.len() != 64
                || !weight.sha256.bytes().all(|v| v.is_ascii_hexdigit())
            {
                return Err("forward trace has invalid parameter digests".into());
            }
        }
        let expected = &baux::semantics::mhc4::probe::initialized::FAMILIES;
        if self.weights.len() != expected.len()
            || expected.iter().any(|family| {
                self.weights
                    .iter()
                    .find(|weight| weight.name == family.name)
                    .is_none_or(|weight| weight.elements != family.elements)
            })
            || self
                .weights
                .iter()
                .map(|weight| weight.elements)
                .sum::<u64>()
                != baux::semantics::shape::MHC4_PARAMS
        {
            return Err("forward trace parameter inventory differs from the mHC-4 contract".into());
        }
        Ok(())
    }

    /// Match construction before interpreting any numerical output differences.
    /// No floating-point tolerance is silently selected by the trace format.
    pub fn same_construction(&self, other: &Self) -> Result<(), String> {
        self.validate()?;
        other.validate()?;
        if self.seed != other.seed
            || self.input != other.input
            || self.labels != other.labels
            || self.rows != other.rows
            || self.width != other.width
        {
            return Err("forward trace inputs or shape differ".into());
        }
        fn sorted(trace: &ForwardTrace) -> Vec<&WeightHash> {
            let mut weights = trace.weights.iter().collect::<Vec<_>>();
            weights.sort_by(|a, b| a.name.cmp(&b.name));
            weights
        }
        let left = sorted(self);
        let right = sorted(other);
        if left.len() != right.len() {
            return Err("forward trace parameter inventories differ".into());
        }
        for (a, b) in left.iter().zip(right) {
            if *a != b {
                return Err(format!(
                    "initialized parameter tensor differs: {} versus {}",
                    a.name, b.name
                ));
            }
        }
        Ok(())
    }

    /// Compare captured outputs only after exact construction agreement.
    /// The result reports differences; it does not invent an acceptance bound.
    pub fn compare(&self, candidate: &Self) -> Result<ForwardDiff, String> {
        self.same_construction(candidate)?;
        Ok(ForwardDiff {
            encoded: words(&self.encoded, &candidate.encoded),
            hidden: words(&self.hidden, &candidate.hidden),
            predictions: IndexDiff {
                elements: self.predictions.len(),
                mismatches: self
                    .predictions
                    .iter()
                    .zip(&candidate.predictions)
                    .filter(|(left, right)| left != right)
                    .count(),
            },
            losses: floats(&self.loss_bits, &candidate.loss_bits),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ForwardTrace {
        ForwardTrace {
            format: ForwardTrace::FORMAT.into(),
            seed: 17,
            input: vec![11, 12],
            labels: vec![13, 14],
            rows: 2,
            width: baux::semantics::shape::MHC4_WIDTH as usize,
            weights: baux::semantics::mhc4::probe::initialized::FAMILIES
                .iter()
                .map(|family| WeightHash {
                    name: family.name.into(),
                    elements: family.elements,
                    sha256: "1".repeat(64),
                })
                .collect(),
            encoded: vec![0x3c00; 2 * baux::semantics::shape::MHC4_WIDTH as usize],
            hidden: vec![0x3c00; 2 * baux::semantics::shape::MHC4_WIDTH as usize],
            predictions: vec![15, 16],
            loss_bits: vec![1.0_f32.to_bits(), 2.0_f32.to_bits()],
        }
    }

    #[test]
    fn trace_inputs() {
        let left = fixture();
        left.same_construction(&left).unwrap();
        let mut right = left.clone();
        right.seed += 1;
        assert!(left.same_construction(&right).is_err());
        right = left.clone();
        right.labels[0] += 1;
        assert!(left.same_construction(&right).is_err());
        right = left.clone();
        right.weights[0].sha256 = "0".repeat(64);
        assert!(left.same_construction(&right).is_err());
        right = left.clone();
        right.weights[0].elements += 1;
        assert!(left.same_construction(&right).is_err());
        // Numerical results are deliberately not a construction comparison.
        right = left.clone();
        right.hidden[0] = 0x7c00;
        left.same_construction(&right).unwrap();
        let diff = left.compare(&right).unwrap();
        assert_eq!(diff.hidden.mismatches, 1);
        assert!(!diff.exact());
    }

    #[test]
    fn trace_invalid() {
        let valid = fixture();
        let mut invalid = valid.clone();
        invalid.weights.push(invalid.weights[0].clone());
        assert!(invalid.validate().is_err());
        invalid = valid.clone();
        invalid.weights[0].sha256 = "g".repeat(64);
        assert!(invalid.validate().is_err());
        invalid = valid.clone();
        invalid.hidden.clear();
        assert!(invalid.validate().is_err());
        invalid = valid.clone();
        invalid.predictions.clear();
        assert!(invalid.validate().is_err());
        invalid = valid.clone();
        invalid.rows = usize::MAX;
        invalid.width = 2;
        assert!(invalid.validate().is_err());
        invalid = valid;
        invalid.format = "unknown".into();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn trace_roundtrip() {
        let mut trace = fixture();
        trace.loss_bits[0] = f32::NAN.to_bits();
        let encoded = baux::artifact::json::to_vec(&trace).unwrap();
        let decoded: ForwardTrace = baux::artifact::json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.loss_bits, trace.loss_bits);
        trace.same_construction(&decoded).unwrap();
        let diff = trace.compare(&decoded).unwrap();
        assert_eq!(diff.losses.reference_nonfinite, 1);
        assert!(!diff.exact());
    }
    #[test]
    fn word_digest() {
        let words = [0, 1, 0x8000, 0xffff];
        let mut whole = WordHash::new("tensor");
        whole.update(&words).unwrap();
        let whole = whole.finish();
        let mut split = WordHash::new("tensor");
        split.update(&words[..1]).unwrap();
        split.update(&words[1..]).unwrap();
        assert_eq!(whole, split.finish());
        assert_eq!(whole.elements, 4);
        assert_eq!(
            whole.sha256,
            format!("{:x}", Sha256::digest([0, 0, 1, 0, 0, 128, 255, 255]))
        );
    }
}
