//! Cephes `ndtri` standard normal inverse CDF and critical value helpers.

#![allow(clippy::excessive_precision)]

use crate::error::ENNError;

const S2PI: f64 = 2.506_628_274_631_000_502_42;
const EXP_M2: f64 = 0.135_335_283_236_612_691_89;

const P0: [f64; 5] = [
    -5.996_335_010_141_078_952_67e1,
    9.800_107_541_859_996_615_36e1,
    -5.667_628_574_690_702_934_39e1,
    1.393_126_093_872_796_795_03e1,
    -1.239_165_838_673_812_580_16,
];

const Q0: [f64; 8] = [
    1.954_488_583_381_417_598_34,
    4.676_279_128_988_815_384_53,
    8.636_024_213_908_905_905_75e1,
    -2.254_626_878_541_193_705_27e2,
    2.002_602_123_800_606_603_59e2,
    -8.203_722_561_683_333_399_12e1,
    1.590_562_251_262_116_955_15e1,
    -1.183_316_211_213_300_031_42,
];

const P1: [f64; 9] = [
    4.055_448_923_059_624_199_23,
    3.152_510_945_998_938_661_54e1,
    5.716_281_922_464_212_881_62e1,
    4.408_050_738_932_008_347_00e1,
    1.468_495_619_288_580_240_14e1,
    2.186_633_068_507_902_675_39,
    -1.402_560_791_713_544_958_75e-1,
    -3.504_246_268_278_482_034_18e-2,
    -8.574_567_851_546_854_136_11e-4,
];

const Q1: [f64; 8] = [
    1.577_998_832_564_667_497_31e1,
    4.539_076_351_288_792_105_84e1,
    4.131_720_382_546_720_304_40e1,
    1.504_253_856_929_075_034_08e1,
    2.504_649_462_083_094_159_79,
    -1.421_829_228_547_877_885_74e-1,
    -3.808_064_076_915_782_771_94e-2,
    -9.332_594_808_954_574_273_72e-4,
];

const P2: [f64; 9] = [
    3.237_748_917_769_460_359_70,
    6.915_228_890_689_842_116_95,
    3.938_810_252_924_744_434_15,
    1.333_034_608_158_075_423_89,
    2.014_853_895_491_790_815_38e-1,
    1.237_166_348_178_200_213_58e-2,
    3.015_815_535_082_354_160_07e-4,
    2.658_069_746_867_375_508_32e-6,
    6.239_745_391_849_832_937_30e-9,
];

const Q2: [f64; 8] = [
    6.024_270_393_647_420_142_55,
    3.679_835_638_561_608_594_03,
    1.377_020_994_890_813_302_71,
    2.162_369_935_944_966_358_90e-1,
    1.342_040_060_885_431_890_37e-2,
    3.280_144_646_821_277_391_04e-4,
    2.892_478_647_453_806_839_36e-6,
    6.790_194_080_099_812_744_25e-9,
];

#[inline]
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

#[inline]
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Inverse of the standard normal cumulative distribution function
/// $\Phi^{-1}(p)$.
pub fn ndtri(p: f64) -> f64 {
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let mut neg = true;
    let mut y = p;
    if y > (1.0 - EXP_M2) {
        y = 1.0 - y;
        neg = false;
    }
    if y > EXP_M2 {
        let yc = y - 0.5;
        let y2 = yc * yc;
        let x = yc + yc * (y2 * polevl(y2, &P0) / p1evl(y2, &Q0));
        return x * S2PI;
    }
    let x = (-2.0 * y.ln()).sqrt();
    let x0 = x - x.ln() / x;
    let z = 1.0 / x;
    let x1 = if x < 8.0 {
        z * polevl(z, &P1) / p1evl(z, &Q1)
    } else {
        z * polevl(z, &P2) / p1evl(z, &Q2)
    };
    let out = x0 - x1;
    if neg { -out } else { out }
}

/// Two-sided normal critical value $z_{(1+\alpha)/2}$ for confidence level
/// $\alpha \in (0, 1)$.
pub fn z_crit(level: f64) -> Result<f64, ENNError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(ENNError::InvalidParameter(
            "level must be in (0, 1)".to_string(),
        ));
    }
    Ok(ndtri(0.5 * (1.0 + level)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_quantiles() {
        assert_eq!(ndtri(0.5), 0.0);
        assert_eq!(ndtri(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri(1.0), f64::INFINITY);
        assert!(ndtri(-0.1).is_nan());
        assert!(ndtri(1.1).is_nan());
        assert!(ndtri(f64::NAN).is_nan());
        assert!((ndtri(0.975) - 1.959_963_984_540_054).abs() < 1e-14);
        assert!((ndtri(0.025) + 1.959_963_984_540_054).abs() < 1e-14);
        assert!((ndtri(1e-10) + 6.361_340_902_404_056).abs() < 1e-12);
        assert!((ndtri(1.0 - 1e-10) - 6.361_340_902_404_056).abs() < 1e-7);
        assert_eq!(ndtri(1.0 - 1e-6), -ndtri(1.0 - (1.0 - 1e-6)));
    }

    #[test]
    fn crit_levels() {
        let z95 = z_crit(0.95).unwrap();
        assert!((z95 - 1.959_963_984_540_054).abs() < 1e-14);
        assert!(z_crit(0.0).is_err());
        assert!(z_crit(1.0).is_err());
        assert!(z_crit(f64::NAN).is_err());
    }
}
