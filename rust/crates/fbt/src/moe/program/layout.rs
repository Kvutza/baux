//! Compiler-facing parameter views in the canonical packed MHC4 arena.
use baux::semantics::{mhc4::layout::LAYOUT, mhc4::probe::initialized::FAMILIES, shape as d};

/// Named FP16 views; offsets and extents are in elements, not bytes.
///
/// # Returns
/// Canonically ordered `(name, offset, shape)` entries into the MHC4 parameter
/// arena. Each shape's product equals its packed slot extent. Metadata does not
/// allocate a model or transfer weights. See [`super::Resident::weights`] for
/// an explicit parameter snapshot.
pub fn layout() -> Vec<(&'static str, usize, Vec<usize>)> {
    let shapes: [&[u64]; 15] = [
        &[d::MHC4_LAYERS, d::MHC4_WIDTH, d::ROUTED_COUNT],
        &[d::MHC4_LAYERS, d::MHC4_WIDTH, d::QKV_WIDTH],
        &[d::MHC4_LAYERS, d::MHC4_WIDTH, d::MHC4_WIDTH],
        &[
            d::MHC4_LAYERS,
            d::EXPERT_COUNT,
            d::MHC4_WIDTH,
            d::GATE_WIDTH,
        ],
        &[
            d::MHC4_LAYERS,
            d::EXPERT_COUNT,
            d::EXPERT_WIDTH,
            d::MHC4_WIDTH,
        ],
        &[d::MHC4_LAYERS, d::MHC4_WIDTH],
        &[d::MHC4_LAYERS, d::MHC4_WIDTH],
        &[d::MHC4_WIDTH, baux::semantics::mhc4::VOCABULARY as u64],
        &[d::MHC4_LAYERS, d::MHC_INPUT, d::MHC_COEFS],
        &[d::MHC4_LAYERS, d::MHC_COEFS],
        &[d::MHC4_LAYERS, 4],
        &[d::MHC4_LAYERS, d::MHC_INPUT, d::MHC_COEFS],
        &[d::MHC4_LAYERS, d::MHC_COEFS],
        &[d::MHC4_LAYERS, 4],
        &[d::MHC4_WIDTH],
    ];
    FAMILIES
        .iter()
        .zip(LAYOUT.slots())
        .zip(shapes)
        .map(|((family, slot), shape)| {
            assert_eq!(shape.iter().product::<u64>(), slot.elements);
            (
                family.name,
                slot.offset as usize,
                shape.iter().map(|&n| n as usize).collect(),
            )
        })
        .collect()
}
