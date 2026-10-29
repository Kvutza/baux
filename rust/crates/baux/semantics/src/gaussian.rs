use std::fmt::Write as _;
use std::sync::LazyLock;

// These integer operations are shared with the CUDA sampler and extracted into
// Lean. Float conversion, rejection predicates and table generation remain
// separate obligations.
#[inline(always)]
pub fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

#[inline(always)]
pub fn bits(seed: u64, key: u64, element: u64, attempt: u32, stream: u32) -> u32 {
    let domain = seed as u32
        ^ ((seed >> 32) as u32).wrapping_mul(0xc2b2_ae35)
        ^ (key as u32).wrapping_mul(0x27d4_eb2f)
        ^ ((key >> 32) as u32).wrapping_mul(0x1656_67b1);
    let counter = (element as u32).wrapping_mul(0x9e37_79b9)
        ^ ((element >> 32) as u32).wrapping_mul(0x85eb_ca6b);
    mix32(domain ^ counter ^ attempt.wrapping_mul(0xd3a2_646c) ^ stream.wrapping_mul(0xfd70_46c5))
}

/// Four integer witnesses used by the CUDA retry reachability proof.
/// This checks actual table entries; it does not certify Gaussian distribution
/// or float tails.
pub fn normal_bounds(t0: u32, t192: u32, t129: u32, t221: u32) -> bool {
    t0 > 0 && t192 > 1_753_845_952 && t129 > 787_275_391 && t221 > 1_408_362_973
}

pub const ZIGGURAT_LAYERS: usize = 256;
pub const ZIGGURAT_R: f64 = 3.654_152_885_361_009;
pub const ZIGGURAT_V: f64 = 0.004_928_673_233_99;

pub struct ZigguratTables {
    pub thresholds: [u32; ZIGGURAT_LAYERS],
    pub widths: [f32; ZIGGURAT_LAYERS],
    pub densities: [f32; ZIGGURAT_LAYERS],
}

pub static ZIGGURAT: LazyLock<ZigguratTables> = LazyLock::new(|| {
    let mut thresholds = [0; ZIGGURAT_LAYERS];
    let mut widths = [0.0; ZIGGURAT_LAYERS];
    let mut densities = [0.0; ZIGGURAT_LAYERS];
    let scale = 2_147_483_648.0;
    let mut boundary = ZIGGURAT_R;
    let mut previous = boundary;
    let tail_density = (-0.5 * boundary * boundary).exp();
    let tail_width = ZIGGURAT_V / tail_density;

    thresholds[0] = (boundary / tail_width * scale) as u32;
    thresholds[1] = 0;
    widths[0] = (tail_width / scale) as f32;
    widths[ZIGGURAT_LAYERS - 1] = (boundary / scale) as f32;
    densities[0] = 1.0;
    densities[ZIGGURAT_LAYERS - 1] = tail_density as f32;

    for layer in (1..ZIGGURAT_LAYERS - 1).rev() {
        boundary =
            (-2.0 * (ZIGGURAT_V / boundary + (-0.5 * boundary * boundary).exp()).ln()).sqrt();
        thresholds[layer + 1] = (boundary / previous * scale) as u32;
        previous = boundary;
        densities[layer] = (-0.5 * boundary * boundary).exp() as f32;
        widths[layer] = (boundary / scale) as f32;
    }
    ZigguratTables {
        thresholds,
        widths,
        densities,
    }
});

pub fn metal_tables() -> String {
    fn uint_array(name: &str, values: &[u32]) -> String {
        let mut source = format!("constant uint {name}[{}] = {{", values.len());
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                source.push(',');
            }
            write!(source, "{value}u").expect("writing a String cannot fail");
        }
        source.push_str("};\n");
        source
    }

    fn float_array(name: &str, values: &[f32]) -> String {
        let mut source = format!("constant float {name}[{}] = {{", values.len());
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                source.push(',');
            }
            write!(source, "{value:?}f").expect("writing a String cannot fail");
        }
        source.push_str("};\n");
        source
    }

    let mut source = uint_array("ziggurat_thresholds", &ZIGGURAT.thresholds);
    source.push_str(&float_array("ziggurat_widths", &ZIGGURAT.widths));
    source.push_str(&float_array("ziggurat_densities", &ZIGGURAT.densities));
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_bounds() {
        let actual = [
            ZIGGURAT.thresholds[0],
            ZIGGURAT.thresholds[192],
            ZIGGURAT.thresholds[129],
            ZIGGURAT.thresholds[221],
        ];
        assert!(normal_bounds(actual[0], actual[1], actual[2], actual[3]));
        let boundary = [0, 1_753_845_952, 787_275_391, 1_408_362_973];
        for index in 0..4 {
            let mut changed = actual;
            changed[index] = boundary[index];
            assert!(!normal_bounds(
                changed[0], changed[1], changed[2], changed[3]
            ));
            changed[index] += 1;
            assert!(normal_bounds(
                changed[0], changed[1], changed[2], changed[3]
            ));
        }
    }
}
