use super::{MAX_HISTORY, MAX_PENDING, ParamBlock, Proposal, TellMode};
use crate::trust_region::TRLengthConfig;

pub(super) fn check_tellmode(current: Option<TellMode>, requested: TellMode) -> Result<(), String> {
    if current.is_some_and(|mode| mode != requested) {
        return Err(
            "Cannot mix legacy tell, tell_paired, and tell_relative on the same BF16 search".into(),
        );
    }
    if current.is_none() && requested == TellMode::PairedRelative {
        return Err("Enable paired-relative BF16 search before tell_relative".into());
    }
    Ok(())
}

pub(super) fn failure_tolerance(dimensions: usize, arms: usize) -> usize {
    let arm_count = arms as f64;
    (4.0_f64 / arm_count)
        .max(dimensions as f64 / arm_count)
        .ceil()
        .max(1.0) as usize
}

pub(super) fn check_length(length: TRLengthConfig) -> Result<(), String> {
    if [length.length_min, length.length_init, length.length_max]
        .iter()
        .any(|&value| !value.is_finite() || !(value as f32).is_finite() || value as f32 <= 0.0)
        || length.length_init < length.length_min
        || length.length_init > length.length_max
    {
        return Err(
            "BF16 radius bounds must be ordered, positive, and representable as FP32".into(),
        );
    }
    Ok(())
}

pub(super) fn correlated_length(length: TRLengthConfig) -> Result<TRLengthConfig, String> {
    check_length(length)?;
    let mut min = length.length_min as f32;
    let mut max = length.length_max as f32;
    // Accepted FP32 radii become FP64 state. Round bounds inward so this
    // round-trip cannot put the next round's state outside the caller's
    // requested interval.
    if f64::from(min) < length.length_min {
        min = min.next_up();
    }
    if f64::from(max) > length.length_max {
        max = max.next_down();
    }
    if !min.is_finite() || min >= max {
        return Err("Correlated BF16 radius bounds must contain two distinct FP32 radii".into());
    }
    let (min, max) = (f64::from(min), f64::from(max));
    Ok(TRLengthConfig::new(
        length.length_init.clamp(min, max),
        min,
        max,
    ))
}

pub(super) fn check_search(
    len: usize,
    base_value: f32,
    base_variance: f32,
    blocks: &[ParamBlock],
    capacity: usize,
    pending_capacity: usize,
) -> Result<usize, String> {
    if !base_value.is_finite() || !base_variance.is_finite() || base_variance < 0.0 {
        return Err("BF16 base value and variance are invalid".to_string());
    }
    if capacity == 0 || capacity > MAX_HISTORY {
        return Err(format!(
            "BF16 history capacity must be in 1..={MAX_HISTORY}"
        ));
    }
    if pending_capacity == 0 || pending_capacity > MAX_PENDING {
        return Err(format!(
            "BF16 pending capacity must be in 1..={MAX_PENDING}"
        ));
    }
    let mut expected = 0usize;
    for block in blocks {
        if block.offset != expected {
            return Err("BF16 blocks must form a contiguous layout".to_string());
        }
        expected = expected
            .checked_add(block.len)
            .ok_or("BF16 block layout overflow")?;
    }
    if expected == 0 || expected != len {
        return Err(format!(
            "BF16 blocks cover {expected} weights, expected {len}"
        ));
    }
    Ok(expected)
}

pub(super) fn check_tell(
    trials: &[Proposal],
    values: &[f32],
    variances: &[f32],
) -> Result<(), String> {
    if trials.is_empty() || trials.len() != values.len() || trials.len() != variances.len() {
        return Err(
            "BF16 trials, values, and variances must have equal non-zero length".to_string(),
        );
    }
    if values.iter().any(|value| !value.is_finite())
        || variances
            .iter()
            .any(|variance| !variance.is_finite() || *variance < 0.0)
    {
        return Err("BF16 values and variances must be finite".to_string());
    }
    Ok(())
}

pub(super) fn check_paired(
    correlated: bool,
    arms: usize,
    value: f32,
    variance: f32,
    incumbent_value: f32,
    incumbent_variance: f32,
) -> Result<(), String> {
    if !correlated || arms != 1 {
        return Err("tell_paired requires a single-arm correlated BF16 round".into());
    }
    if !value.is_finite()
        || !incumbent_value.is_finite()
        || !variance.is_finite()
        || variance < 0.0
        || !incumbent_variance.is_finite()
        || incumbent_variance < 0.0
    {
        return Err(
            "Paired rewards must be finite and both variances finite and nonnegative".into(),
        );
    }
    Ok(())
}

pub(super) fn check_relativeinit(
    correlated: bool,
    started: bool,
    mode: Option<TellMode>,
    capacity: usize,
    failure_tolerance: usize,
) -> Result<(), String> {
    if !correlated || started || mode.is_some() || capacity < 2 {
        return Err(
            "Enable paired-relative mode once on a fresh correlated search with capacity >= 2, \
             before any asks or tells"
                .into(),
        );
    }
    if failure_tolerance == 0 || u32::try_from(failure_tolerance).is_err() {
        return Err("Paired-relative failure_tolerance must be a positive u32".into());
    }
    Ok(())
}

pub(super) fn check_improvement(improvement: f32, variance: f32) -> Result<(), String> {
    if !improvement.is_finite() || !variance.is_finite() || variance < 0.0 {
        return Err(
            "Paired improvement must be finite and its variance finite and nonnegative".into(),
        );
    }
    Ok(())
}

pub(super) fn slot_count(capacity: usize, pending: usize) -> Result<usize, String> {
    capacity
        .checked_add(pending)
        .and_then(|slots| slots.checked_add(1))
        .ok_or("BF16 resident slot count overflow".to_string())
}

pub(super) fn cuda_blocks(blocks: &[ParamBlock]) -> Vec<baux_cuda::Bf16Leaf> {
    blocks
        .iter()
        .map(|block| baux_cuda::Bf16Leaf {
            key: block.key,
            offset: block.offset as u64,
            length: block.len as u64,
            scale: block.scale,
            weight: block.weight,
        })
        .collect()
}

#[cfg(test)]
mod radius_tests {
    use super::super::SearchState;
    use super::*;

    #[test]
    fn collapsed_bounds() {
        let ulp = f64::from(1.0_f32.next_up()) - 1.0;
        for (min, max) in [
            (1.0, 1.0),
            (1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON),
            (1.0 + ulp * 0.25, 1.0 + ulp * 0.75),
            (1.0 + ulp * 0.5, 1.0 + ulp * 1.5),
        ] {
            let config = TRLengthConfig::new((min + max) * 0.5, min, max);
            assert!(check_length(config).is_ok());
            let error = correlated_length(config).unwrap_err();
            assert!(error.contains("two distinct FP32 radii"), "{error}");
        }
    }

    #[test]
    fn rounded_bounds() {
        let config = TRLengthConfig::new(0.01, 0.0001, 0.08);
        let rounded = correlated_length(config).unwrap();
        assert_eq!(rounded.length_init, config.length_init);
        assert!(rounded.length_min >= config.length_min);
        assert!(rounded.length_max <= config.length_max);
        assert_eq!(rounded.length_min, f64::from((0.0001_f64 as f32).next_up()));
        assert_eq!(rounded.length_max, f64::from(0.08_f64 as f32));
        assert!(check_length(TRLengthConfig::new(0.01, 0.01, 0.01)).is_ok());
    }

    #[test]
    fn clamped_radius() {
        let min = 1.0 + f64::EPSILON;
        let max = 2.0 - f64::EPSILON;
        for initial in [min, max] {
            let rounded = correlated_length(TRLengthConfig::new(initial, min, max)).unwrap();
            let expected = if initial == min {
                f64::from(1.0_f32.next_up())
            } else {
                f64::from(2.0_f32.next_down())
            };
            assert_eq!(rounded.length_init, expected);
            assert!(rounded.length_init >= min && rounded.length_init <= max);
        }
    }

    #[test]
    fn adjacent_bounds() {
        let min = f64::from(1.0_f32);
        let max = f64::from(1.0_f32.next_up());
        let config = TRLengthConfig::new(min, min, max);
        assert_eq!(correlated_length(config).unwrap(), config);
    }

    #[test]
    fn accepted_bound() {
        for config in [
            TRLengthConfig::default(),
            TRLengthConfig::new(0.01, 0.0001, 0.08),
            TRLengthConfig::new(1.0, 1.0 - f64::EPSILON, 2.0 - f64::EPSILON),
            TRLengthConfig::new(1e-40, 1e-41, 1e-39),
            TRLengthConfig::new(1e37, 1e36, f64::from(f32::MAX)),
        ] {
            let rounded = correlated_length(config).unwrap();
            for factor in [0.5, 2.0] {
                let mut radius = rounded.length_init;
                for _ in 0..128 {
                    radius = f64::from(
                        ((radius * factor)
                            .max(rounded.length_min)
                            .min(rounded.length_max)) as f32,
                    );
                    assert!(radius >= config.length_min && radius <= config.length_max);
                    assert!(radius >= rounded.length_min && radius <= rounded.length_max);
                    assert!(
                        check_length(TRLengthConfig::new(
                            radius,
                            rounded.length_min,
                            rounded.length_max,
                        ))
                        .is_ok()
                    );
                }
            }
        }
    }

    #[test]
    fn bad_radii() {
        for config in [
            TRLengthConfig::new(f64::NAN, 0.001, 0.1),
            TRLengthConfig::new(0.01, 0.0, 0.1),
            TRLengthConfig::new(0.01, 0.02, 0.1),
            TRLengthConfig::new(0.01, 0.001, 0.005),
            TRLengthConfig::new(0.01, f64::MIN_POSITIVE, 0.1),
            TRLengthConfig::new(0.01, 0.001, f64::MAX),
        ] {
            assert!(correlated_length(config).is_err());
        }
    }

    #[test]
    fn radius_order() {
        let block = ParamBlock::new(0, 0, 1, 1.0, 1.0).unwrap();
        for config in [
            TRLengthConfig::new(f64::NAN, 0.001, 0.1),
            TRLengthConfig::new(0.01, 0.0, 0.1),
            TRLengthConfig::new(0.01, 0.02, 0.1),
            TRLengthConfig::new(0.01, 0.001, 0.005),
            TRLengthConfig::new(0.01, f64::MIN_POSITIVE, 0.1),
            TRLengthConfig::new(0.01, 0.001, f64::MAX),
        ] {
            let error = SearchState::new(&[0x3f80], 0.0, 0.0, vec![block], 2, 1, config)
                .err()
                .expect("Invalid bounds must fail before accessing CUDA");
            assert!(error.contains("radius bounds"), "{error}");
        }
        assert!(check_length(TRLengthConfig::default()).is_ok());
        assert!(check_length(TRLengthConfig::new(0.01, 0.01, 0.01)).is_ok());
    }
}

#[cfg(test)]
mod paired_tests {
    use super::*;

    #[test]
    fn validate_measure() {
        assert!(check_paired(true, 1, -12.0, 0.25, -13.0, 0.0).is_ok());
        assert!(check_paired(false, 1, 0.0, 0.0, 0.0, 0.0).is_err());
        for arms in [0, 2, MAX_PENDING] {
            assert!(check_paired(true, arms, 0.0, 0.0, 0.0, 0.0).is_err());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_paired(true, 1, invalid, 0.0, 0.0, 0.0).is_err());
            assert!(check_paired(true, 1, 0.0, 0.0, invalid, 0.0).is_err());
        }
        for invalid in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_paired(true, 1, 0.0, invalid, 0.0, 0.0).is_err());
            assert!(check_paired(true, 1, 0.0, 0.0, 0.0, invalid).is_err());
        }
    }

    #[test]
    fn tell_modes() {
        for mode in [TellMode::Legacy, TellMode::Paired] {
            assert!(check_tellmode(None, mode).is_ok());
            assert!(check_tellmode(Some(mode), mode).is_ok());
        }
        assert!(check_tellmode(Some(TellMode::Legacy), TellMode::Paired).is_err());
        assert!(check_tellmode(Some(TellMode::Paired), TellMode::Legacy).is_err());
        assert!(check_tellmode(None, TellMode::PairedRelative).is_err());
        for mode in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
            assert!(check_tellmode(Some(mode), mode).is_ok());
            for other in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
                assert_eq!(check_tellmode(Some(mode), other).is_ok(), mode == other);
            }
        }
    }

    #[test]
    fn mode_validation() {
        assert!(check_relativeinit(true, false, None, 2, 4).is_ok());
        assert!(check_relativeinit(true, false, None, MAX_HISTORY, u32::MAX as usize).is_ok());
        assert!(check_relativeinit(false, false, None, 2, 4).is_err());
        assert!(check_relativeinit(true, true, None, 2, 4).is_err());
        for mode in [TellMode::Legacy, TellMode::Paired, TellMode::PairedRelative] {
            assert!(check_relativeinit(true, false, Some(mode), 2, 4).is_err());
        }
        for capacity in [0, 1] {
            assert!(check_relativeinit(true, false, None, capacity, 4).is_err());
        }
        assert!(check_relativeinit(true, false, None, 2, 0).is_err());
        if usize::BITS > 32 {
            assert!(check_relativeinit(true, false, None, 2, usize::MAX).is_err());
        }
    }

    #[test]
    fn valid_improve() {
        for improvement in [-1.0, 0.0, 1.0, f32::MAX, -f32::MAX] {
            assert!(check_improvement(improvement, 0.0).is_ok());
            assert!(check_improvement(improvement, f32::MAX).is_ok());
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(check_improvement(invalid, 0.0).is_err());
            assert!(check_improvement(0.0, invalid).is_err());
        }
        assert!(check_improvement(0.0, -f32::from_bits(1)).is_err());
    }

    #[test]
    fn identity_owner() {
        let proposal = Proposal {
            owner: 1,
            id: 0,
            slot: 3,
            index: 0,
            seed: 0,
            score: 0.0,
            length: 0.0,
        };
        let foreign = Proposal {
            owner: 2,
            ..proposal
        };
        assert_ne!(proposal, foreign);
        assert_ne!(proposal, Proposal { id: 1, ..proposal });
    }
}
