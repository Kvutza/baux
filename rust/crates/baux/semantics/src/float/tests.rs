use super::*;

#[test]
fn single_parts() {
    assert!(!sign_single(ONE_SINGLE));
    assert_eq!(exp_single(ONE_SINGLE), 127);
    assert_eq!(frac_single(ONE_SINGLE), 0);

    let neg_one = single_neg(ONE_SINGLE);
    assert!(sign_single(neg_one));
    assert_eq!(single_abs(neg_one), ONE_SINGLE);
}

#[test]
fn double_parts() {
    assert!(!sign_double(ONE_DOUBLE));
    assert_eq!(exp_double(ONE_DOUBLE), 1023);
    assert_eq!(frac_double(ONE_DOUBLE), 0);

    let neg_one = double_neg(ONE_DOUBLE);
    assert!(sign_double(neg_one));
    assert_eq!(double_abs(neg_one), ONE_DOUBLE);
}

#[test]
fn test_classifications() {
    assert!(single_zero(ZERO_SINGLE));
    assert!(single_zero(SIGN_SINGLE)); // -0.0
    assert!(single_finite(ONE_SINGLE));
    assert!(!single_finite(INF_SINGLE));
    assert!(!single_finite(NAN_SINGLE));
    assert!(single_nan(NAN_SINGLE));
    assert!(single_inf(INF_SINGLE));
    assert!(single_inf(single_neg(INF_SINGLE)));

    assert!(double_zero(ZERO_DOUBLE));
    assert!(double_zero(SIGN_DOUBLE)); // -0.0
    assert!(double_finite(ONE_DOUBLE));
    assert!(!double_finite(INF_DOUBLE));
    assert!(!double_finite(NAN_DOUBLE));
    assert!(double_nan(NAN_DOUBLE));
    assert!(double_inf(INF_DOUBLE));
    assert!(double_inf(double_neg(INF_DOUBLE)));
}

#[test]
fn single_comparisons() {
    let f_neg_zero = SIGN_SINGLE;
    let f_pos_zero = 0u32;
    let f_one = ONE_SINGLE;
    let f_two = 0x4000_0000u32; // 2.0f32

    assert!(!single_lt(f_pos_zero, f_neg_zero));
    assert!(!single_lt(f_neg_zero, f_pos_zero));
    assert!(single_le(f_pos_zero, f_neg_zero));
    assert!(single_le(f_neg_zero, f_pos_zero));
    assert!(single_eq(f_pos_zero, f_neg_zero));

    assert!(single_lt(f_one, f_two));
    assert!(single_le(f_one, f_two));
    assert!(!single_gt(f_one, f_two));
    assert!(!single_eq(f_one, f_two));

    // NaN comparisons
    assert!(!single_lt(NAN_SINGLE, f_one));
    assert!(!single_le(NAN_SINGLE, f_one));
    assert!(!single_gt(NAN_SINGLE, f_one));
    assert!(!single_ge(NAN_SINGLE, f_one));
    assert!(!single_eq(NAN_SINGLE, f_one));
    assert!(!single_eq(NAN_SINGLE, NAN_SINGLE));
}

#[test]
fn double_comparisons() {
    let f_neg_zero = SIGN_DOUBLE;
    let f_pos_zero = 0u64;

    assert!(!double_lt(f_pos_zero, f_neg_zero));
    assert!(!double_lt(f_neg_zero, f_pos_zero));
    assert!(double_le(f_pos_zero, f_neg_zero));
    assert!(double_le(f_neg_zero, f_pos_zero));
    assert!(double_eq(f_pos_zero, f_neg_zero));

    assert!(double_lt(ONE_DOUBLE, TWO_DOUBLE));
    assert!(double_le(ONE_DOUBLE, TWO_DOUBLE));
    assert!(!double_gt(ONE_DOUBLE, TWO_DOUBLE));
    assert!(!double_eq(ONE_DOUBLE, TWO_DOUBLE));

    // NaN comparisons
    assert!(!double_lt(NAN_DOUBLE, ONE_DOUBLE));
    assert!(!double_le(NAN_DOUBLE, ONE_DOUBLE));
    assert!(!double_gt(NAN_DOUBLE, ONE_DOUBLE));
    assert!(!double_ge(NAN_DOUBLE, ONE_DOUBLE));
    assert!(!double_eq(NAN_DOUBLE, ONE_DOUBLE));
    assert!(!double_eq(NAN_DOUBLE, NAN_DOUBLE));
}

#[test]
fn total_cmp() {
    let neg_nan = 0xffc0_0000u32;
    let pos_nan = 0x7fc0_0000u32;
    let neg_inf = single_neg(INF_SINGLE);
    let pos_inf = INF_SINGLE;
    let neg_zero = SIGN_SINGLE;
    let pos_zero = ZERO_SINGLE;
    let one = ONE_SINGLE;

    assert_eq!(single_total(neg_nan, neg_inf), -1);
    assert_eq!(single_total(neg_inf, neg_zero), -1);
    assert_eq!(single_total(neg_zero, pos_zero), -1);
    assert_eq!(single_total(pos_zero, one), -1);
    assert_eq!(single_total(one, pos_inf), -1);
    assert_eq!(single_total(pos_inf, pos_nan), -1);
    assert_eq!(single_total(one, one), 0);
}

#[test]
fn widen_cases() {
    // Zero
    assert_eq!(to_double(0), 0);
    assert_eq!(to_double(SIGN_SINGLE), SIGN_DOUBLE);

    // One
    assert_eq!(to_double(ONE_SINGLE), ONE_DOUBLE);

    // Constants
    assert_eq!(to_double((-1.0f32).to_bits()), (-1.0f64).to_bits());
    assert_eq!(to_double((0.5f32).to_bits()), (0.5f64).to_bits());
    assert_eq!(to_double((2.0f32).to_bits()), (2.0f64).to_bits());

    // Inf
    assert_eq!(to_double(INF_SINGLE), INF_DOUBLE);
    assert_eq!(to_double(single_neg(INF_SINGLE)), double_neg(INF_DOUBLE));

    // Subnormals
    let subnormals = [1u32, 2u32, 0x007f_ffffu32, 0x8000_0001u32];
    for &bits in &subnormals {
        let rust_val = (f32::from_bits(bits) as f64).to_bits();
        assert_eq!(to_double(bits), rust_val);
    }

    // Patterns across range
    let test_words = [
        0x3e80_0000u32, // 0.25
        0x42f6_0000u32, // 123.0
        0xc2f6_0000u32, // -123.0
        0x0000_0001u32, // min subnormal
        0x007f_ffffu32, // max subnormal
        0x0080_0000u32, // min normal
        0x7f7f_ffffu32, // max normal
    ];
    for &bits in &test_words {
        let rust_val = (f32::from_bits(bits) as f64).to_bits();
        assert_eq!(to_double(bits), rust_val);
    }
}
