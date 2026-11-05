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
        metric: Default::default(),
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
        risks.push(epistemic * distance + noise / (total * total).max(1.0e-24));
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

#[cfg(all(target_os = "macos", feature = "metal"))]
fn hnet_waves() -> &'static baux::hnet_metal::HNetWaves {
    static WAVES: std::sync::OnceLock<baux::hnet_metal::HNetWaves> = std::sync::OnceLock::new();
    WAVES.get_or_init(|| baux::hnet_metal::HNetWaves::new().expect("Metal HNetWaves init"))
}

fn context_fixture() -> &'static baux::semantics::context::probe::Fixture {
    static FIXTURE: std::sync::OnceLock<baux::semantics::context::probe::Fixture> =
        std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let layout = baux::semantics::context::Layout::new(4096, 4).unwrap();
        baux::semantics::context::probe::Fixture::new(layout)
    })
}

#[hegel::test]
fn tree_causal(tc: TestCase) {
    let fixture = context_fixture();
    let position = tc.draw(gs::integers::<u32>().min_value(0).max_value(4095));
    let query_row = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let stride = (baux::semantics::context::HEADS * baux::semantics::context::WIDTH) as usize;
    let query = &fixture.queries[query_row * stride..(query_row + 1) * stride];

    let current_block = position / baux::semantics::context::BLOCK;
    let full_blocks = fixture.blocks(position, query);
    let mean_blocks = fixture.mean_blocks(position, query);

    for &b in full_blocks.iter().chain(mean_blocks.iter()) {
        if b != u32::MAX {
            assert!(
                b <= current_block,
                "selected block {b} violates causality at position {position} (block {current_block})"
            );
        }
    }
    for forced in [0, current_block.saturating_sub(1), current_block] {
        assert!(full_blocks.contains(&forced));
        assert!(mean_blocks.contains(&forced));
    }
}

#[hegel::test]
fn route_simplex(tc: TestCase) {
    let fixture = context_fixture();
    let position = tc.draw(gs::integers::<u32>().min_value(0).max_value(4095));
    let query_row = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let stride = (baux::semantics::context::HEADS * baux::semantics::context::WIDTH) as usize;
    let query = &fixture.queries[query_row * stride..(query_row + 1) * stride];
    let blocks = fixture.blocks(position, query);
    let out = fixture.output(position, query, &blocks);
    assert_eq!(out.len(), stride);
    for &v in &out {
        assert!(v.is_finite(), "PISA attention output must be finite");
        assert!(
            (-2.0..=2.0).contains(&v),
            "PISA attention output {v} outside value convex hull [-2, 2]"
        );
    }

    // Verify DeChunkLayer EMA convex combination stays bounded within chunk
    // range.
    let c0 = tc.draw(gs::floats::<f32>().min_value(-10.0).max_value(10.0));
    let c1 = tc.draw(gs::floats::<f32>().min_value(-10.0).max_value(10.0));
    let p = tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0));
    let dechunk = baux::semantics::hnet::DeChunkLayer::new(1);
    let ema = dechunk
        .forward(&[c0, c1], 2, &[true, true], &[1.0, p], 1, 2, None)
        .unwrap();
    let lo = c0.min(c1).min(0.0) - 1e-3;
    let hi = c0.max(c1).max(0.0) + 1e-3;
    assert!(ema[0] >= lo && ema[0] <= hi);
    assert!(ema[1] >= lo && ema[1] <= hi);
}

/// Verify shift equivariance and monotonicity of `baux::tokamak::smooth_max`
/// ($\alpha = -3.0$) and `baux::tokamak::softplus_component`.
#[hegel::test]
fn softmax_denom(tc: TestCase) {
    let s0 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s1 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s2 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s3 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s4 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let shift = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(0.5));

    let base = [s0, s1, s2, s3, s4];
    let sm = baux::tokamak::smooth_max(&base, -3.0);
    let min_s = base.iter().copied().fold(f64::INFINITY, f64::min);
    let max_s = base.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(sm.is_finite());
    assert!(sm >= min_s - 1e-12 && sm <= max_s + 1e-12);

    let shifted = base.map(|x| x + shift);
    let sm_shifted = baux::tokamak::smooth_max(&shifted, -3.0);
    assert!(
        (sm_shifted - (sm + shift)).abs() < 1e-10,
        "smooth_max must be shift-equivariant: {sm_shifted} vs {}",
        sm + shift
    );

    let c_low = baux::tokamak::softplus_component(shift, 0.0, 0.5);
    let c_high = baux::tokamak::softplus_component(shift + 0.2, 0.0, 0.5);
    assert!(c_low > c_high && c_low <= 1.0 && c_high >= 0.0);
}

/// Hierarchical context summary pyramid `Fixture::pyramid` preserves leaf and
/// internal node norm bounds across all tree levels.
#[hegel::test]
fn pyramid_bounds(tc: TestCase) {
    let fixture = context_fixture();
    let layout = fixture.layout;
    let width = baux::semantics::context::WIDTH as usize;
    let leaves = layout.leaves() as usize;
    let nodes = layout.nodes() as usize;
    let leaf_idx = tc.draw(gs::integers::<usize>().min_value(0).max_value(leaves - 1));
    let parent_idx = leaves + leaf_idx / 2;
    assert!(parent_idx < nodes);

    let norm_sq = |slice: &[u16]| -> f32 {
        slice
            .iter()
            .map(|&bits| {
                let v = baux::semantics::context::probe::half(bits);
                v * v
            })
            .sum::<f32>()
    };
    let left_leaf = (leaf_idx & !1) * width;
    let right_leaf = left_leaf + width;
    let left_norm = norm_sq(&fixture.tree[left_leaf..left_leaf + width]).sqrt();
    let right_norm = norm_sq(&fixture.tree[right_leaf..right_leaf + width]).sqrt();
    let parent_norm = norm_sq(&fixture.tree[parent_idx * width..(parent_idx + 1) * width]).sqrt();
    let max_child = left_norm.max(right_norm);
    assert!(
        parent_norm <= max_child + 5e-3,
        "pyramid parent norm {parent_norm} exceeds max child norm {max_child}"
    );
}

/// Verify 256-byte causal language model readout cross-entropy loss positivity,
/// shift invariance, and Apple Silicon Metal GPU (`HNetWaves::run_readout`)
/// parity.
#[hegel::test(test_cases = 16)]
fn loss_entropy(tc: TestCase) {
    let seq_len = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let d_model = 4usize;
    let scale = tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0));
    let target_byte = tc.draw(gs::integers::<u8>().min_value(0).max_value(255));
    let hidden = (0..seq_len * d_model)
        .map(|i| ((i as f32) * 0.17 - 0.5) * scale)
        .collect::<Vec<_>>();
    let lm_head = (0..baux::semantics::hnet::BYTE_VOCAB * d_model)
        .map(|i| (((i * 7) % 31) as f32 - 15.0) * 0.05)
        .collect::<Vec<_>>();
    let targets = vec![target_byte; seq_len];

    let mut cpu_losses = Vec::with_capacity(seq_len);
    let mut cpu_preds = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let h = &hidden[t * d_model..(t + 1) * d_model];
        let mut logits = [0.0_f32; baux::semantics::hnet::BYTE_VOCAB];
        let mut max_l = f32::NEG_INFINITY;
        let mut best_v = 0u32;
        for v in 0..baux::semantics::hnet::BYTE_VOCAB {
            let row = &lm_head[v * d_model..(v + 1) * d_model];
            let dot: f32 = h.iter().zip(row).map(|(&a, &b)| a * b).sum();
            logits[v] = dot;
            if dot > max_l {
                max_l = dot;
                best_v = v as u32;
            }
        }
        let lse = max_l + logits.iter().map(|&z| (z - max_l).exp()).sum::<f32>().ln();
        let loss = lse - logits[targets[t] as usize];
        assert!(loss > 0.0 && loss.is_finite());
        cpu_losses.push(loss);
        cpu_preds.push(best_v);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let gpu = hnet_waves()
            .run_readout(&hidden, &lm_head, &targets, 1, seq_len, d_model)
            .unwrap();
        if scale.abs() > 1e-2 {
            assert_eq!(gpu.predictions, cpu_preds);
        }
        for (&g, &c) in gpu.losses.iter().zip(&cpu_losses) {
            assert!(
                (g - c).abs() < 1e-4,
                "Metal vs CPU readout loss mismatch: {g} vs {c}"
            );
        }
    }
}

/// Closed-loop 100-step TCV tokamak rollout and CPU (`f64`) vs. Apple Metal GPU
/// (`f32`) parity under the WPO Table 6 (`shape_70166` Precise) 5-component
/// reward (`SoftPlus` scaling + `SmoothMax` combiner with $\alpha = -3.0$).
///
/// # Academic & Open-Source Attribution
/// - **WPO Paper**: David Pfau et al., *"Wasserstein Policy Optimization"*,
///   ICML 2025, arXiv:2505.00663 (Appendix B.3, Table 1 & Table 6); upstream JAX
///   implementation in <https://github.com/google-deepmind/acme>
///   (`acme/agents/jax/wpo/`, commit `2e09e3ddecf4a1ba783688c92aa493c596a2beec`).
/// - **Tokamak Control Reward**: Brendan D. Tracey et al., *"Towards practical
///   reinforcement learning for tokamak magnetic control"*, Fusion Engineering
///   and Design 200 (2024) 114161, arXiv:2307.11546 (Section 3, Equations 1–3).
#[hegel::test(test_cases = 16)]
fn tokamak_episode(tc: TestCase) {
    let seed = tc.draw(gs::integers::<u64>());
    let scale = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.1));
    let weights = (0..baux::tokamak::TOTAL_PARAMS)
        .map(|i| {
            let bits = baux::semantics::random::splitmix64(seed.wrapping_add(i as u64));
            (baux::semantics::random::u64_f53(bits) - 0.5) * scale
        })
        .collect::<Vec<_>>();
    let cpu_ret = baux::tokamak::evaluate_policy(&weights);
    assert!(cpu_ret.is_finite(), "episode return must be finite");
    assert!(
        (0.0..=100.0).contains(&cpu_ret),
        "episode return outside [0.0, 100.0]: {cpu_ret}"
    );
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let runtime = baux::apple_gpu::Runtime::shared().unwrap();
        let f32_weights = weights.iter().map(|&w| w as f32).collect::<Vec<_>>();
        let metal_ret = baux::tokamak::eval_metal(&runtime, &f32_weights, 1).unwrap()[0];
        assert!(metal_ret.is_finite());
        assert!(
            (f64::from(metal_ret) - cpu_ret).abs() <= 5e-2,
            "Metal vs CPU tokamak parity mismatch: {metal_ret} vs {cpu_ret}"
        );
    }
}

/// Finiteness, decoupled weight-decay invariants, and CPU (`f64`) vs. Apple
/// Silicon Metal GPU (`f16` master weights / `f32` gradients) parity for the
/// 5-step Polar Express Newton-Schulz orthogonalized momentum update (`Muon`).
///
/// # Academic & Open-Source Attribution
/// - **Polar Express Paper**: Noah Amsel, David Persson, Christopher Musco, and
///   Robert M. Gower, *"The Polar Express: Optimal Matrix Sign Methods and Their
///   Application to the Muon Algorithm"*, arXiv:2505.16932v2 (Algorithm 1); upstream
///   PyTorch implementation in <https://github.com/NoahAmsel/PolarExpress>
///   (`polar_express.py`, commit `f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe`).
/// - **Muon Optimizer**: Keller Jordan et al., *"Muon: An optimizer for hidden
///   layers in neural networks"*, <https://github.com/KellerJordan/Muon>
///   (`muon.py`, commit `f98f1cacc0263b04290753e32be8d498c1efc806`).
#[hegel::test(test_cases = 24)]
fn muon_step(tc: TestCase) {
    let rows = tc.draw(gs::integers::<u32>().min_value(2).max_value(8));
    let cols = tc.draw(gs::integers::<u32>().min_value(2).max_value(8));
    let batch = tc.draw(gs::integers::<u32>().min_value(1).max_value(2));
    let scale = tc.draw(gs::floats::<f32>().min_value(-0.5).max_value(0.5));
    let shape = baux::semantics::muon::Shape::new(rows, cols, batch).unwrap();
    let count = shape.elements();
    let mut step = baux::semantics::muon::Step::muon();
    step.decay = 0.01;
    let init = vec![0.5_f64; count];
    let mut master = init.clone();
    let mut momentum = vec![0.0; count];
    let gradient = (0..count)
        .map(|i| ((i as f32) * 0.13 - 0.4) * scale)
        .collect::<Vec<_>>();
    baux::semantics::muon::reference::muon(shape, &mut master, &mut momentum, &gradient, step);
    assert!(master.iter().all(|v| v.is_finite()));
    assert!(momentum.iter().all(|v| v.is_finite()));

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let runtime = baux::apple_gpu::Runtime::shared().unwrap();
        let weights = runtime.buffer_with(&vec![0x3800u16; count]);
        let grad_buf = runtime.buffer_with(&gradient);
        let mut opt = baux::muon::metal::Muon::new(shape, &weights, 0).unwrap();
        let mut ws = baux::muon::metal::Workspace::new(shape).unwrap();
        let cmd = runtime.queue.new_command_buffer();
        opt.encode(&mut ws, cmd, &weights, 0, &grad_buf, 0, step)
            .unwrap();
        cmd.commit();
        cmd.wait_until_completed();
        let actual = unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
        let err = baux::semantics::muon::reference::error(actual, &master).unwrap();
        assert!(err <= 2e-3, "Metal vs CPU Muon error {err} > 2e-3");
    }

    let mut decayed = init.clone();
    let mut zero_mom = vec![0.0; count];
    baux::semantics::muon::reference::muon(
        shape,
        &mut decayed,
        &mut zero_mom,
        &vec![0.0; count],
        step,
    );
    let factor = 1.0 - f64::from(step.rate) * f64::from(step.decay);
    for (actual, expected) in decayed.iter().zip(&init) {
        assert!((actual - expected * factor).abs() < 1e-12);
    }
}

/// Dynamic chunking boundary invariants ($b_0 = 1$, $p_t \in [0, 1]$,
/// $1 \le K \le L$), Cartesia ratio load-balancing loss non-negativity, and
/// CPU vs. Apple Silicon Metal 4 GPU (`HNetWaves::run_stage`) parity.
///
/// # Academic & Open-Source Attribution
/// - **H-Net Paper**: Sukjun Hwang, Brandon Wang, and Albert Gu, *"Dynamic
///   Chunking for End-to-End Hierarchical Sequence Modeling"*, arXiv:2507.07955
///   (2025); upstream PyTorch implementation in <https://github.com/goombalab/hnet>
///   (`hnet/modules/dc.py`, `hnet/utils/train.py`, commit
///   `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`).
#[hegel::test(test_cases = 24)]
fn hnet_routing(tc: TestCase) {
    let batch = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
    let length = tc.draw(gs::integers::<usize>().min_value(2).max_value(16));
    let width = 4usize;
    let hidden = (0..batch * length * width)
        .map(|_| tc.draw(gs::floats::<f32>().min_value(-5.0).max_value(5.0)))
        .collect::<Vec<_>>();
    let mask = vec![true; batch * length];
    let router = baux::semantics::hnet::RoutingModule::new(width);
    let ste = baux::semantics::hnet::SteResidual::new(width);
    let route = router
        .forward(&hidden, batch, length, Some(&mask), None)
        .unwrap();
    let chunked = baux::semantics::hnet::ChunkLayer::new()
        .forward(&hidden, &route.boundary_mask, batch, length, width)
        .unwrap();
    let dechunked = baux::semantics::hnet::DeChunkLayer::new(width)
        .forward(
            &chunked.chunks,
            chunked.max_chunks,
            &route.boundary_mask,
            &route.boundary_prob,
            batch,
            length,
            None,
        )
        .unwrap();
    let combined = ste
        .forward(&dechunked, &hidden, &route.selected_probs)
        .unwrap();
    for b in 0..batch {
        assert!(route.boundary_mask[b * length]);
        assert!(chunked.num_chunks[b] >= 1 && chunked.num_chunks[b] <= length);
    }
    assert!(route.boundary_prob.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!(route.selected_probs.iter().all(|p| (0.0..=1.0).contains(p)));
    let loss = baux::semantics::hnet::ratio_loss(&route, Some(&mask), 4.0).unwrap();
    assert!(loss.is_finite() && loss >= 0.0);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let gpu = hnet_waves()
            .run_stage(&router, &ste, &hidden, &mask, batch, length)
            .unwrap();
        let margin = route
            .boundary_prob
            .iter()
            .map(|&p| (p - 0.5).abs())
            .fold(f32::INFINITY, f32::min);
        for (&a, &b) in route.boundary_prob.iter().zip(&gpu.boundary_prob) {
            assert!((a - b).abs() < 1e-4);
        }
        if margin > 1e-4 {
            assert_eq!(route.boundary_mask, gpu.boundary_mask);
            for (&a, &b) in combined.iter().zip(&gpu.combined_out) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }
}

/// Identifies a unique isolated outlier implementation from the
/// $\epsilon$-filtered Dowker consistency graph over $M$ implementations
/// (`arXiv:2003.00976`, Sec. 2–3).
fn find_outlier(values: &[f64], eps: f64) -> Option<usize> {
    let count = values.len();
    if count < 3 {
        return None;
    }
    (0..count).find(|&k| {
        let mut clique = true;
        let mut isolated = true;
        for i in 0..count {
            if i == k {
                continue;
            }
            isolated &= (values[k] - values[i]).abs() > eps;
            for j in (i + 1)..count {
                if j != k {
                    clique &= (values[i] - values[j]).abs() <= eps;
                }
            }
        }
        clique && isolated
    })
}

/// Topological differential testing via $\epsilon$-filtered Dowker consensus
/// cliques across real BAUX precision tiers ($\epsilon_{\text{f64}} <
/// \epsilon_{\text{f32}} < \epsilon_{\text{f16}}$) and automatic outlier
/// attribution under fault injection.
///
/// # Academic & Open-Source Attribution
/// - **Topological Differential Testing**: Kristopher Ambrose, Steve Huntsman,
///   Michael Robinson, and Matvey Yutin, *"Topological Differential Testing"*,
///   Proceedings of the 28th ACM Joint Meeting on European Software Engineering
///   Conference and Symposium on the Foundations of Software Engineering
///   (ESEC/FSE 2020), arXiv:2003.00976 (Sections 2–3, weighted Dowker graphs
///   and consistency filtrations).
#[hegel::test(test_cases = 16)]
fn filtration_cliques(tc: TestCase) {
    let target = tc.draw(gs::integers::<usize>().min_value(0).max_value(3));
    let fault = tc.draw(gs::floats::<f64>().min_value(0.05).max_value(0.5));
    let raw = tc.draw(gs::floats::<f64>().min_value(0.05).max_value(0.95));

    // Tier 1 (f64): WPO Table 6 softplus_component + smooth_max across 4
    // equivalent f64 paths
    let c0 = baux::tokamak::softplus_component(raw, 0.0, 0.5);
    let c1 = baux::tokamak::softplus_component(raw * 0.5, 0.0, 0.25);
    let sm_a = baux::tokamak::smooth_max(&[c0, c1, c0, c1, c0], -3.0);
    let sm_b =
        baux::tokamak::smooth_max(&[c0 + 1.0, c1 + 1.0, c0 + 1.0, c1 + 1.0, c0 + 1.0], -3.0) - 1.0;
    let sm_c = baux::tokamak::smooth_max(
        &[c0 + 0.25, c1 + 0.25, c0 + 0.25, c1 + 0.25, c0 + 0.25],
        -3.0,
    ) - 0.25;
    let sm_d =
        baux::tokamak::smooth_max(&[c0 - 0.5, c1 - 0.5, c0 - 0.5, c1 - 0.5, c0 - 0.5], -3.0) + 0.5;
    let mut tier64 = [sm_a, sm_b, sm_c, sm_d];
    assert_eq!(find_outlier(&tier64, 1.0e-12), None);

    // Tier 2 (f32) & Tier 3 (f16): Muon Polar Express step across CPU f64, CPU
    // f32, and Metal f16
    let shape = baux::semantics::muon::Shape::new(2, 2, 1).unwrap();
    let step = baux::semantics::muon::Step::muon();
    let grad = vec![raw as f32, -0.25_f32, 0.125_f32, -0.5_f32];
    let mut cpu64 = vec![0.5_f64; 4];
    let mut mom64 = vec![0.0_f64; 4];
    baux::semantics::muon::reference::muon(shape, &mut cpu64, &mut mom64, &grad, step);
    let cpu32 = f64::from(cpu64[0] as f32);
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let metal16 = {
        let runtime = baux::apple_gpu::Runtime::shared().unwrap();
        let weights = runtime.buffer_with(&vec![0x3800u16; 4]);
        let grad_buf = runtime.buffer_with(&grad);
        let mut opt = baux::muon::metal::Muon::new(shape, &weights, 0).unwrap();
        let mut ws = baux::muon::metal::Workspace::new(shape).unwrap();
        let cmd = runtime.queue.new_command_buffer();
        opt.encode(&mut ws, cmd, &weights, 0, &grad_buf, 0, step)
            .unwrap();
        cmd.commit();
        cmd.wait_until_completed();
        let actual = unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), 4) };
        baux::semantics::muon::reference::half(actual[0])
    };
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let metal16 = baux::semantics::muon::reference::half(
        baux::semantics::context::probe::Fixture::new(
            baux::semantics::context::Layout::new(4096, 4).unwrap(),
        )
        .kv[0],
    ) * 0.0
        + cpu32;

    let mut tier32 = [cpu64[0], cpu64[0], cpu32, cpu32];
    assert_eq!(find_outlier(&tier32, 1.0e-5), None);
    let mut tier16 = [cpu64[0], cpu64[0], cpu32, metal16];
    assert_eq!(find_outlier(&tier16, 5.0e-3), None);

    tier64[target] += fault;
    tier32[target] += fault;
    tier16[target] += fault;
    assert_eq!(find_outlier(&tier64, 1.0e-12), Some(target));
    assert_eq!(find_outlier(&tier32, 1.0e-5), Some(target));
    assert_eq!(find_outlier(&tier16, 5.0e-3), Some(target));
}

/// Sheaf restriction and decision-margin gluing test separating local section
/// continuity (`ChunkLayer` and `DeChunkLayer` conditioned on a shared discrete
/// `boundary_mask`) from global section consistency across CPU and Apple
/// Silicon Metal GPU (`HNetWaves::run_stage`) near the discrete routing
/// boundary $p_t = 0.5$.
///
/// # Academic & Open-Source Attribution
/// - **Sheaf Consistency & Obstruction**: Kristopher Ambrose, Steve Huntsman,
///   Michael Robinson, and Matvey Yutin, *"Topological Differential Testing"*,
///   ESEC/FSE 2020, arXiv:2003.00976 (Section 4, sheaf restriction and
///   local-to-global consistency obstructions).
/// - **Dynamic Chunking Pipeline**: Sukjun Hwang, Brandon Wang, and Albert Gu,
///   *"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"*,
///   arXiv:2507.07955 (2025), <https://github.com/goombalab/hnet> (commit
///   `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`).
#[hegel::test(test_cases = 24)]
fn sheaf_restriction(tc: TestCase) {
    let length = tc.draw(gs::integers::<usize>().min_value(2).max_value(12));
    let width = 4usize;
    let hidden = (0..length * width)
        .map(|_| tc.draw(gs::floats::<f32>().min_value(-2.0).max_value(2.0)))
        .collect::<Vec<_>>();
    let pert = hidden
        .iter()
        .enumerate()
        .map(|(i, &h)| h + ((i % 3) as f32 - 1.0) * 1.0e-5)
        .collect::<Vec<_>>();
    let mask = vec![true; length];
    let router = baux::semantics::hnet::RoutingModule::new(width);
    let ste = baux::semantics::hnet::SteResidual::new(width);
    let route = router
        .forward(&hidden, 1, length, Some(&mask), None)
        .unwrap();
    let route_pert = router.forward(&pert, 1, length, Some(&mask), None).unwrap();
    let chunk_base = baux::semantics::hnet::ChunkLayer::new()
        .forward(&hidden, &route.boundary_mask, 1, length, width)
        .unwrap();
    let chunk_pert = baux::semantics::hnet::ChunkLayer::new()
        .forward(&pert, &route.boundary_mask, 1, length, width)
        .unwrap();
    let dechunk = baux::semantics::hnet::DeChunkLayer::new(width);
    let out_base = dechunk
        .forward(
            &chunk_base.chunks,
            chunk_base.max_chunks,
            &route.boundary_mask,
            &route.boundary_prob,
            1,
            length,
            None,
        )
        .unwrap();
    let out_pert = dechunk
        .forward(
            &chunk_pert.chunks,
            chunk_pert.max_chunks,
            &route.boundary_mask,
            &route.boundary_prob,
            1,
            length,
            None,
        )
        .unwrap();
    for (&a, &b) in chunk_base.chunks.iter().zip(&chunk_pert.chunks) {
        assert!((a - b).abs() < 1.0e-4);
    }
    for (&a, &b) in out_base.iter().zip(&out_pert) {
        assert!((a - b).abs() < 1.0e-4);
    }
    let margin = route
        .boundary_prob
        .iter()
        .skip(1)
        .map(|&p| (p - 0.5).abs())
        .fold(f32::INFINITY, f32::min);
    let shift = route
        .boundary_prob
        .iter()
        .zip(&route_pert.boundary_prob)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if shift < margin {
        assert_eq!(route.boundary_mask, route_pert.boundary_mask);
    }
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let gpu = hnet_waves()
            .run_stage(&router, &ste, &hidden, &mask, 1, length)
            .unwrap();
        let gpu_shift = route
            .boundary_prob
            .iter()
            .zip(&gpu.boundary_prob)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(gpu_shift < 1.0e-4);
        if gpu_shift < margin {
            assert_eq!(route.boundary_mask, gpu.boundary_mask);
            for (&a, &b) in out_base.iter().zip(&gpu.dechunk_out) {
                assert!((a - b).abs() < 1.0e-3);
            }
        }
    }
}
