//! Tensor shape, precision, ownership, and lifetime vocabulary.

/// Stable tensor identity used by model import and perturbation streams.
/// The offset and wrapping multiplication are the existing wire convention;
/// changing either would silently select different parameter perturbations.
pub fn key(name: &str) -> u64 {
    name.as_bytes()
        .iter()
        .fold(1_469_598_103_934_665_603, |key, byte| {
            (key ^ u64::from(*byte)).wrapping_mul(1_099_511_628_211)
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F16,
    Bf16,
    F32,
    U32,
    Packed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    Scalar,
    Sequence,
    SequenceHidden,
    HeadSequenceWidth,
    RoutedRows,
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorStorage {
    Host,
    Device,
    WeightArena,
    Overlay,
    State,
    Scratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLifetime {
    Invocation,
    Round,
    Sequence,
    Model,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    Fixed(usize),
    Symbol(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: &'static str,
    pub shape: Vec<Extent>,
    pub dtype: TensorDType,
    pub layout: TensorLayout,
    pub storage: TensorStorage,
    pub lifetime: TensorLifetime,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn key_stable() {
        assert_eq!(key(""), 1_469_598_103_934_665_603);
        assert_eq!(key("blk.27.attn_q.weight"), 3_843_877_851_495_245_630);
        let keys = baux::semantics::mhc4::probe::initialized::FAMILIES
            .iter()
            .map(|family| key(family.name))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            keys.len(),
            baux::semantics::mhc4::probe::initialized::FAMILIES.len()
        );
    }
}
