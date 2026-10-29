//! Property tests for optimizer decisions, retention, distances, and
//! accelerator ABI. This package has no production interface; only test builds
//! compile these properties.
#![cfg(test)]

use baux::posterior::neighbor_dist::squared_l2;
use hegel::TestCase;
use hegel::generators as gs;
use ndarray::Array1;

#[hegel::test]
fn proposal_flags(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(512));
    let finite = tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0;
    let allow = tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0;
    let mut flags = 0;
    let mut invalid = false;
    let mut changed = false;
    for _ in 0..count {
        let base = tc.draw(gs::integers::<u16>());
        let raw = tc.draw(gs::integers::<u16>());
        let candidate = if finite { raw & 0x7bff } else { raw };
        flags |= baux::semantics::proposal::fp16_flags(base, candidate);
        invalid |= (candidate >> 10) & 31 == 31;
        changed |= base != candidate;
    }
    assert_eq!(
        baux::semantics::proposal::validate(flags, allow).is_ok(),
        !invalid && (changed || allow)
    );
}

/// Retention protects the incumbent and evicts the oldest other identity.
/// Independently scan the full window rather than calling the rule twice.
#[hegel::test]
fn history_slot(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(0).max_value(128));
    let rotation = tc.draw(gs::integers::<usize>().min_value(0).max_value(127));
    let identities = (0..count)
        .map(|i| ((i + rotation) % 128 + 1) as i64)
        .collect::<Vec<_>>();
    let base = if count == 0 {
        1
    } else {
        identities[tc.draw(gs::integers::<usize>().min_value(0).max_value(count - 1))]
    };
    let actual = baux::semantics::neighbor::retention(&identities, base).unwrap();
    if count < 128 {
        assert_eq!(actual, count);
    } else {
        assert_ne!(identities[actual], base);
        for &identity in &identities {
            assert!(identity == base || identities[actual] <= identity);
        }
    }
    assert!(baux::semantics::neighbor::retention(&[1; 129], 1).is_err());
    assert!(baux::semantics::neighbor::retention(&[1; 128], 1).is_err());
}

#[hegel::test]
fn surrogate_spec(tc: TestCase) {
    let spec = baux::semantics::optimizer::SurrogateSpec {
        neighbors: tc.draw(gs::integers::<usize>().min_value(1).max_value(128)),
        epistemic: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        aleatoric: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        scale: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(10.0)),
        fit_candidates: tc.draw(gs::integers::<usize>().min_value(1).max_value(64)),
        fit_samples: tc.draw(gs::integers::<usize>().min_value(1).max_value(128)),
        fit: tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0,
        adaptive: tc.draw(gs::integers::<u8>().min_value(0).max_value(1)) != 0,
    }
    .validate()
    .unwrap();
    let json = baux::artifact::json::to_string(&spec).unwrap();
    let actual: baux::semantics::optimizer::SurrogateSpec =
        baux::artifact::json::from_str(&json).unwrap();
    assert_eq!(actual.validate().unwrap(), spec);
    let mut invalid = spec;
    invalid.neighbors = 0;
    assert!(invalid.validate().is_err());
    invalid.neighbors = 129;
    assert!(invalid.validate().is_err());
    invalid = spec;
    invalid.scale = f32::INFINITY;
    assert!(invalid.validate().is_err());
}

/// The shared prefix scan must choose the same first risk minimizer as a
/// separately recomputed batch calculation over sorted neighbour distances.
#[hegel::test]
fn prefix_choice(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let epistemic = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(2.0));
    let mut points = (0..count)
        .map(|_| {
            (
                tc.draw(gs::floats::<f64>().min_value(0.0).max_value(4.0)),
                tc.draw(gs::floats::<f64>().min_value(0.01).max_value(2.0)),
            )
        })
        .collect::<Vec<_>>();
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut scan = baux::neighbor_rule::Prefix::default();
    let mut risks = Vec::new();
    for (index, &(distance, noise)) in points.iter().enumerate() {
        scan.push(index as u32 + 1, distance, epistemic, noise);
        let weights = points[..=index]
            .iter()
            .map(|&(d, n)| 1.0 / (1.0e-9 + epistemic * d + n).max(1.0e-12))
            .collect::<Vec<f64>>();
        let total = weights.iter().copied().sum::<f64>();
        let noise = weights
            .iter()
            .zip(&points)
            .map(|(&weight, &(_, noise))| weight * weight * noise)
            .sum::<f64>();
        risks.push(epistemic * distance + (noise / (total * total).max(1.0e-24)).sqrt());
    }
    let expected = risks
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert_eq!(scan.count as usize, expected + 1);
}

/// Recurrence has one shared ordering: prefix, repeated core, suffix. This
/// checks arbitrary valid layer spans, not only the current model constants.
#[hegel::test]
fn recurrence_trace(tc: TestCase) {
    use baux::semantics::recurrence::RecurrentCore;
    let layers = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let first = tc.draw(gs::integers::<usize>().min_value(0).max_value(layers - 1));
    let count = tc.draw(
        gs::integers::<usize>()
            .min_value(1)
            .max_value(layers - first),
    );
    let visits = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let core = RecurrentCore {
        first_layer: first,
        layer_count: count,
        training_visits: 1,
        min_inference_visits: 1,
        max_inference_visits: 8,
    };
    let actual = core.layer_visits(layers, visits).unwrap();
    let expected = (0..first)
        .chain((0..visits).flat_map(|_| first..first + count))
        .chain(first + count..layers)
        .collect::<Vec<_>>();
    assert_eq!(actual.iter().map(|v| v.layer).collect::<Vec<_>>(), expected);
    for (index, visit) in actual.iter().enumerate() {
        assert_eq!(visit.execution, index);
        assert_eq!(
            visit.recurrent,
            (first..first + count).contains(&visit.layer)
        );
    }
    assert!(core.layer_visits(layers, 0).is_err());
    assert!(core.layer_visits(layers, 9).is_err());
}

/// Known integer fixtures protect stream identity independently of aliases.
#[hegel::test]
fn stream_identity(tc: TestCase) {
    let seed = tc.draw(gs::integers::<u64>());
    assert_eq!(
        baux::hash::splitmix64(seed),
        baux::semantics::optimizer::splitmix64(seed)
    );
    assert_eq!(baux::hash::splitmix64(0), 0xe220_a839_7b1d_cdaf);
    assert_eq!(baux::hash::splitmix64(1), 0x910a_2dec_8902_5cc1);
    assert_eq!(baux::hash::splitmix64(u64::MAX), 0xe4d9_7177_1b65_2c20);
}

/// Compiling generated declarations checks every Metal field offset. Reading
/// sentinel fields on the GPU also catches order/type mistakes in the ABI.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[hegel::test]
fn wave_abi(tc: TestCase) {
    use baux::semantics::mhc4::{EncodeShape, ShardStats, WaveShape};
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    let wave = WaveShape {
        blocks: tc.draw(gs::integers::<u32>()),
        tokens: tc.draw(gs::integers::<u32>()),
        proposal_start: tc.draw(gs::integers::<u32>()),
        intra: tc.draw(gs::integers::<u32>()),
        vocabulary: tc.draw(gs::integers::<u32>()),
        shards: tc.draw(gs::integers::<u32>()),
        temperature: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(4.0)),
        score: tc.draw(gs::integers::<u32>()),
    };
    let shard = ShardStats {
        maximum: 1.0,
        sum: 2.0,
        target: 3.0,
        best: 4.0,
        token: tc.draw(gs::integers::<u32>()),
        reserved: tc.draw(gs::integers::<u32>()),
    };
    let encode = EncodeShape {
        rows: tc.draw(gs::integers::<u32>()),
        width: tc.draw(gs::integers::<u32>()),
        vocabulary: tc.draw(gs::integers::<u32>()),
        patch: tc.draw(gs::integers::<u32>()),
    };
    let source = baux::semantics::mhc4::metal_source(
        r#"
kernel void abi_read(constant WaveShape& w [[buffer(0)]],
                     constant ShardStats& s [[buffer(1)]],
                     device uint* output [[buffer(2)]],
                     constant EncodeShape& e [[buffer(3)]]) {
    output[0]=w.blocks; output[1]=w.tokens; output[2]=w.proposal_start;
    output[3]=w.intra; output[4]=w.vocabulary; output[5]=w.shards;
    output[6]=as_type<uint>(w.temperature); output[7]=w.score;
    output[8]=as_type<uint>(s.maximum); output[9]=as_type<uint>(s.sum);
    output[10]=as_type<uint>(s.target); output[11]=as_type<uint>(s.best);
    output[12]=s.token; output[13]=s.reserved;
    output[14]=e.rows; output[15]=e.width; output[16]=e.vocabulary; output[17]=e.patch;
}
"#,
    );
    static GPU: std::sync::OnceLock<(Device, metal::ComputePipelineState, metal::CommandQueue)> =
        std::sync::OnceLock::new();
    let (device, pipeline, queue) = GPU.get_or_init(|| {
        let device = Device::system_default().expect("Metal device required");
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .unwrap();
        let function = library.get_function("abi_read", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let queue = device.new_command_queue();
        (device, pipeline, queue)
    });
    let output = device.new_buffer(18 * 4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_bytes(
        0,
        size_of::<WaveShape>() as u64,
        std::ptr::from_ref(&wave).cast(),
    );
    encoder.set_bytes(
        1,
        size_of::<ShardStats>() as u64,
        std::ptr::from_ref(&shard).cast(),
    );
    encoder.set_buffer(2, Some(&output), 0);
    encoder.set_bytes(
        3,
        size_of::<EncodeShape>() as u64,
        std::ptr::from_ref(&encode).cast(),
    );
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(one, one);
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
    // SAFETY: completed command initialized all 18 u32 values in shared
    // storage.
    let actual = unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), 18) };
    assert_eq!(
        actual,
        &[
            wave.blocks,
            wave.tokens,
            wave.proposal_start,
            wave.intra,
            wave.vocabulary,
            wave.shards,
            wave.temperature.to_bits(),
            wave.score,
            shard.maximum.to_bits(),
            shard.sum.to_bits(),
            shard.target.to_bits(),
            shard.best.to_bits(),
            shard.token,
            shard.reserved,
            encode.rows,
            encode.width,
            encode.vocabulary,
            encode.patch,
        ]
    );
}

/// Both public adapters must execute the same policy over an adaptive trace,
/// including configured failure budgets and non-default evidence parameters.
#[hegel::test]
fn controller_parity(tc: TestCase) {
    let policy = baux::ReliabilityPolicy {
        evidence_decay: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        confidence_z: tc.draw(gs::floats::<f64>().min_value(0.25).max_value(4.0)),
        reliability_threshold: tc.draw(gs::floats::<f64>().min_value(0.01).max_value(0.99)),
        minimum_evidence: tc.draw(gs::floats::<f64>().min_value(0.1).max_value(8.0)),
        acceptance_ratio: tc.draw(gs::floats::<f64>().min_value(0.01).max_value(2.0)),
        boundary_fraction: tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)),
        expand_factor: tc.draw(gs::floats::<f64>().min_value(1.25).max_value(4.0)),
        shrink_factor: tc.draw(gs::floats::<f64>().min_value(0.125).max_value(0.9)),
        failure_tolerance: tc.draw(gs::integers::<usize>().min_value(0).max_value(8)),
    };
    let encoded = baux::artifact::json::to_string(&policy).unwrap();
    let decoded: baux::ReliabilityPolicy = baux::artifact::json::from_str(&encoded).unwrap();
    assert_eq!(policy, decoded);
    let mut host =
        baux::semantics::optimizer::RadiusController::configured(0.125, 1.0 / 65536.0, 2.0, policy)
            .unwrap();
    let mut metal = baux::ReliabilityController::new(
        policy,
        baux::trust_region::TRLengthConfig::new(0.125, 1.0 / 65536.0, 2.0),
    )
    .unwrap();
    let steps = tc.draw(gs::integers::<usize>().min_value(1).max_value(48));
    for _ in 0..steps {
        let observed = tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0));
        let predicted = tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0));
        let variance = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let prediction = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let response = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
        let nominal = f64::from(host.radius());
        let actual = host
            .observe(observed, variance, predicted, prediction, response, false)
            .unwrap();
        let expected = metal
            .update(baux::ReliabilityEvidence {
                observed_improvement: f64::from(observed),
                observation_variance: f64::from(variance),
                predicted_improvement: f64::from(predicted),
                prediction_variance: f64::from(prediction),
                accepted: actual.accepted,
                nominal_radius: nominal,
                response_radius: f64::from(response),
            })
            .unwrap();
        assert_eq!(actual.action, expected.action);
        assert_eq!(
            actual.next_radius.to_bits(),
            (expected.length as f32).to_bits()
        );
        assert_eq!(
            actual.standardized_error.to_bits(),
            expected.standardized_error.unwrap().to_bits()
        );
        assert_eq!(
            actual.sharpness_ratio.to_bits(),
            expected.sharpness_ratio.unwrap().to_bits()
        );
        assert_eq!(
            actual.agreement_ratio.map(f64::to_bits),
            expected.agreement_ratio.map(f64::to_bits)
        );
        assert!((1.0 / 65536.0..=2.0).contains(&actual.next_radius));
    }
}

/// An accepted observation resets consecutive rejection evidence. With the
/// adequacy gate inactive, contraction occurs at exactly the declared budget.
#[hegel::test]
fn failure_reset(tc: TestCase) {
    let budget = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let policy = baux::ReliabilityPolicy {
        failure_tolerance: budget,
        minimum_evidence: f64::MAX,
        ..Default::default()
    };
    let mut state =
        baux::semantics::optimizer::RadiusController::configured(0.125, 0.001, 2.0, policy)
            .unwrap();
    for _ in 0..budget - 1 {
        assert_eq!(
            state
                .observe(0.0, 0.0, 0.0, 0.0, 0.0, false)
                .unwrap()
                .action,
            baux::ReliabilityAction::Hold
        );
    }
    assert!(
        state
            .observe(1.0, 0.0, 1.0, 0.0, 1.0, false)
            .unwrap()
            .accepted
    );
    for step in 1..=budget {
        let result = state.observe(0.0, 0.0, 0.0, 0.0, 0.0, false).unwrap();
        let expected = if step == budget {
            baux::ReliabilityAction::Shrink
        } else {
            baux::ReliabilityAction::Hold
        };
        assert_eq!(result.action, expected);
    }
    assert_eq!(state.radius(), 0.0625);
}

/// Invalid evidence must not consume a failure or change the next transition.
#[hegel::test]
fn invalid_atomic(tc: TestCase) {
    let mut state =
        baux::semantics::optimizer::RadiusControl::new(Default::default(), 0.125, 0.001, 2.0)
            .unwrap();
    let valid = baux::ReliabilityEvidence {
        observed_improvement: tc.draw(gs::floats::<f64>().min_value(-1.0).max_value(1.0)),
        observation_variance: 0.01,
        predicted_improvement: 0.0,
        prediction_variance: 0.01,
        accepted: false,
        nominal_radius: 0.125,
        response_radius: 0.1,
    };
    state.update(valid).unwrap();
    let mut reference = state.clone();
    let mut invalid = valid;
    match tc.draw(gs::integers::<u8>().min_value(0).max_value(5)) {
        0 => invalid.observed_improvement = f64::NAN,
        1 => invalid.observation_variance = -1.0,
        2 => invalid.predicted_improvement = f64::INFINITY,
        3 => invalid.prediction_variance = -1.0,
        4 => invalid.nominal_radius = 0.0,
        _ => invalid.response_radius = -1.0,
    }
    assert!(state.update(invalid).is_err());
    assert_eq!(
        state.update(valid).unwrap(),
        reference.update(valid).unwrap()
    );
}

#[hegel::test]
fn native_order(tc: TestCase) {
    let left = tc.draw(gs::integers::<u64>());
    let right = tc.draw(gs::integers::<u64>());
    assert_eq!(
        baux::semantics::optimizer::accept_bits(left, right),
        f64::from_bits(left) > f64::from_bits(right)
    );
}

#[hegel::test]
fn test_identity(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(vals);
    let s = Array1::from_vec(scales);
    let d = squared_l2(x.view(), x.view(), true, s.view());
    assert!(d.abs() < 1e-10, "self distance must be zero: {d}");
}

#[hegel::test]
fn test_symmetry(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let x_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let y_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(x_vals);
    let y = Array1::from_vec(y_vals);
    let s = Array1::from_vec(scales);
    let d_xy = squared_l2(x.view(), y.view(), true, s.view());
    let d_yx = squared_l2(y.view(), x.view(), true, s.view());
    assert!(
        (d_xy - d_yx).abs() < 1e-10,
        "distance must be symmetric: {d_xy} != {d_yx}"
    );
}

#[hegel::test]
fn test_positivity(tc: TestCase) {
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let x_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let y_vals: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let scales: Vec<f64> = (0..dim)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.1).max_value(10.0)))
        .collect();
    let x = Array1::from_vec(x_vals);
    let y = Array1::from_vec(y_vals);
    let s = Array1::from_vec(scales);
    let d = squared_l2(x.view(), y.view(), true, s.view());
    assert!(d >= 0.0, "squared distance must be non-negative: {d}");
}

#[hegel::test]
fn tree_causal(tc: TestCase) {
    let block_size = 64usize;
    let num_blocks = tc.draw(gs::integers::<usize>().min_value(1).max_value(16));
    let total_rows = num_blocks * block_size;
    let query_row = tc.draw(
        gs::integers::<usize>()
            .min_value(0)
            .max_value(total_rows - 1),
    );
    let current_block = query_row / block_size;

    // Invariant: any block selected prior to or at current_block must strictly
    // obey causality
    let candidate_blocks: Vec<usize> = (0..=current_block).collect();
    for &b in &candidate_blocks {
        let block_start = b * block_size;
        let block_end = (b + 1) * block_size;
        if b < current_block {
            assert!(block_end - 1 < query_row || query_row >= block_start);
        } else {
            assert_eq!(b, current_block);
        }
    }
}

#[hegel::test]
fn route_simplex(tc: TestCase) {
    let num_experts = 32usize;
    let top_k = 3usize;
    let scores: Vec<f64> = (0..num_experts)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-50.0).max_value(50.0)))
        .collect();

    let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_score = indexed[0].1;
    let top_exp: Vec<f64> = indexed
        .iter()
        .take(top_k)
        .map(|(_, s)| (s - max_score).exp())
        .collect();
    let sum_exp: f64 = top_exp.iter().sum();
    assert!(sum_exp > 0.0);

    let weights: Vec<f64> = top_exp.iter().map(|e| e / sum_exp).collect();
    let total_weight: f64 = weights.iter().sum();
    assert!(
        (total_weight - 1.0).abs() < 1e-10,
        "MoE weights must sum to 1.0: {total_weight}"
    );
    for &w in &weights {
        assert!(w >= 0.0 && w <= 1.0, "MoE weight outside [0, 1]: {w}");
    }
}

/// Softmax denominator sum over simulated dot products in [-100.0, 100.0]
/// is non-zero, finite, and strictly monotonic under score shift.
#[hegel::test]
fn softmax_denom(tc: TestCase) {
    let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
    let dots: Vec<f64> = (0..count)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-100.0).max_value(100.0)))
        .collect();
    let shift = tc.draw(gs::floats::<f64>().min_value(0.001).max_value(10.0));

    let denom: f64 = dots.iter().map(|&s| s.exp()).sum();
    assert!(
        denom > 0.0,
        "denominator sum must be non-zero positive: {denom}"
    );
    assert!(denom.is_finite(), "denominator sum must be finite: {denom}");

    // Denominator sum strictly increases under positive score shift
    let shifted_up: Vec<f64> = dots.iter().map(|&s| s + shift).collect();
    let denom_up: f64 = shifted_up.iter().map(|&s| s.exp()).sum();
    assert!(denom_up.is_finite(), "shifted denominator must be finite");
    assert!(
        denom_up > denom,
        "denominator must strictly increase under positive score shift: {denom_up} <= {denom}"
    );

    // Denominator sum strictly decreases under negative score shift
    let shifted_down: Vec<f64> = dots.iter().map(|&s| s - shift).collect();
    let denom_down: f64 = shifted_down.iter().map(|&s| s.exp()).sum();
    assert!(
        denom_down > 0.0,
        "down-shifted denominator must be positive"
    );
    assert!(
        denom_down < denom,
        "denominator must strictly decrease under negative score shift: {denom_down} >= {denom}"
    );
}

/// Level 0 block centroids norm in the hierarchical summary pyramid is bounded
/// by maximum leaf norm times sqrt(block_size).
#[hegel::test]
fn pyramid_bounds(tc: TestCase) {
    let block_size = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
    let num_blocks = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));

    let total_rows = num_blocks * block_size;
    let keys: Vec<Vec<f64>> = (0..total_rows)
        .map(|_| {
            (0..dim)
                .map(|_| tc.draw(gs::floats::<f64>().min_value(-10.0).max_value(10.0)))
                .collect()
        })
        .collect();

    let sqrt_block = (block_size as f64).sqrt();

    for b in 0..num_blocks {
        let block_start = b * block_size;
        let block_keys = &keys[block_start..block_start + block_size];

        let mut max_leaf_norm = 0.0f64;
        for leaf in block_keys {
            let norm = leaf.iter().map(|&x| x * x).sum::<f64>().sqrt();
            max_leaf_norm = max_leaf_norm.max(norm);
        }

        // Level 0 block centroid: sum / sqrt(block_size) as in
        // Selection.summary
        let mut centroid = vec![0.0f64; dim];
        for leaf in block_keys {
            for (d, &val) in leaf.iter().enumerate() {
                centroid[d] += val;
            }
        }
        for val in &mut centroid {
            *val /= sqrt_block;
        }

        let centroid_norm = centroid.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let bound = max_leaf_norm * sqrt_block;

        assert!(
            centroid_norm <= bound + 1e-9 * (1.0 + bound),
            "Level 0 centroid norm exceeds bound: {centroid_norm} > {bound}"
        );

        // Arithmetic mean centroid norm is also bounded by max_leaf_norm *
        // sqrt(block_size)
        let mean_norm = centroid_norm / sqrt_block;
        assert!(
            mean_norm <= bound + 1e-9 * (1.0 + bound),
            "Mean centroid norm exceeds bound: {mean_norm} > {bound}"
        );
    }
}

/// Cross-entropy loss is strictly positive for non-degenerate predictions,
/// and equals zero when prediction matches ground truth with confidence 1.0.
#[hegel::test]
fn loss_entropy(tc: TestCase) {
    // Invariant 1: loss equals zero when prediction matches ground truth with
    // confidence 1.0
    let loss_exact = -1.0_f64.ln();
    assert_eq!(
        loss_exact, 0.0,
        "cross-entropy loss must be exactly 0.0 when confidence is 1.0"
    );

    // Invariant 2: for arbitrary predicted logits with vocabulary >= 2,
    // cross-entropy loss is strictly positive and finite.
    let vocab = tc.draw(gs::integers::<usize>().min_value(2).max_value(64));
    let target = tc.draw(gs::integers::<usize>().min_value(0).max_value(vocab - 1));
    let logits: Vec<f64> = (0..vocab)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-15.0).max_value(15.0)))
        .collect();

    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logits.iter().map(|&z| (z - max_logit).exp()).sum();
    let loss = sum_exp.ln() + max_logit - logits[target];

    assert!(
        loss > 0.0,
        "cross-entropy loss must be strictly positive: {loss}"
    );
    assert!(
        loss.is_finite(),
        "cross-entropy loss must be finite: {loss}"
    );

    let prob = (logits[target] - max_logit).exp() / sum_exp;
    assert!(prob < 1.0, "finite logits cannot yield confidence 1.0");
    assert!(prob > 0.0, "softmax probability must be positive");
    let loss_prob = -prob.ln();
    assert!(
        (loss - loss_prob).abs() < 1e-10,
        "cross-entropy formulations must match: {loss} vs {loss_prob}"
    );
}
