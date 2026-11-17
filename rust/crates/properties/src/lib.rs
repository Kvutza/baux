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
fn hnet_waves() -> &'static baux_fbt::hnet_metal::HNetWaves {
    static WAVES: std::sync::OnceLock<baux_fbt::hnet_metal::HNetWaves> = std::sync::OnceLock::new();
    WAVES.get_or_init(|| baux_fbt::hnet_metal::HNetWaves::new().expect("Metal HNetWaves init"))
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

/// Verify shift equivariance and monotonicity of `baux_tokamak::smooth_max`
/// ($\alpha = -3.0$) and `baux_tokamak::softplus_component`.
#[hegel::test]
fn softmax_denom(tc: TestCase) {
    let s0 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s1 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s2 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s3 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let s4 = tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0));
    let shift = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(0.5));

    let base = [s0, s1, s2, s3, s4];
    let sm = baux_tokamak::smooth_max(&base, -3.0);
    let min_s = base.iter().copied().fold(f64::INFINITY, f64::min);
    let max_s = base.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(sm.is_finite());
    assert!(sm >= min_s - 1e-12 && sm <= max_s + 1e-12);

    let shifted = base.map(|x| x + shift);
    let sm_shifted = baux_tokamak::smooth_max(&shifted, -3.0);
    assert!(
        (sm_shifted - (sm + shift)).abs() < 1e-10,
        "smooth_max must be shift-equivariant: {sm_shifted} vs {}",
        sm + shift
    );

    let c_low = baux_tokamak::softplus_component(shift, 0.0, 0.5);
    let c_high = baux_tokamak::softplus_component(shift + 0.2, 0.0, 0.5);
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
    let weights = (0..baux_tokamak::TOTAL_PARAMS)
        .map(|i| {
            let bits = baux::semantics::random::splitmix64(seed.wrapping_add(i as u64));
            (baux::semantics::random::u64_f53(bits) - 0.5) * scale
        })
        .collect::<Vec<_>>();
    let cpu_ret = baux_tokamak::evaluate_policy(&weights);
    assert!(cpu_ret.is_finite(), "episode return must be finite");
    assert!(
        (0.0..=100.0).contains(&cpu_ret),
        "episode return outside [0.0, 100.0]: {cpu_ret}"
    );
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let runtime = baux::apple_gpu::Runtime::shared().unwrap();
        let f32_weights = weights.iter().map(|&w| w as f32).collect::<Vec<_>>();
        let metal_ret = baux_tokamak::eval_metal(&runtime, &f32_weights, 1).unwrap()[0];
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
        let mut opt = baux_fbt::muon::metal::Muon::new(shape, &weights, 0).unwrap();
        let mut ws = baux_fbt::muon::metal::Workspace::new(shape).unwrap();
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
/// cliques across real baux precision tiers ($\epsilon_{\text{f64}} <
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
    let c0 = baux_tokamak::softplus_component(raw, 0.0, 0.5);
    let c1 = baux_tokamak::softplus_component(raw * 0.5, 0.0, 0.25);
    let sm_a = baux_tokamak::smooth_max(&[c0, c1, c0, c1, c0], -3.0);
    let sm_b =
        baux_tokamak::smooth_max(&[c0 + 1.0, c1 + 1.0, c0 + 1.0, c1 + 1.0, c0 + 1.0], -3.0) - 1.0;
    let sm_c = baux_tokamak::smooth_max(
        &[c0 + 0.25, c1 + 0.25, c0 + 0.25, c1 + 0.25, c0 + 0.25],
        -3.0,
    ) - 0.25;
    let sm_d =
        baux_tokamak::smooth_max(&[c0 - 0.5, c1 - 0.5, c0 - 0.5, c1 - 0.5, c0 - 0.5], -3.0) + 0.5;
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
        let mut opt = baux_fbt::muon::metal::Muon::new(shape, &weights, 0).unwrap();
        let mut ws = baux_fbt::muon::metal::Workspace::new(shape).unwrap();
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

/// Stateless `DelayedPool<T>` RAASP forced-coordinate perturbation, box bounds,
/// and `O(k_pert)` sparse-delta ARD squared L2 equivalence against dense L2.
#[hegel::test]
fn raasp_sparse(tc: TestCase) {
    use baux::semantics::key::Key;
    use baux::semantics::neighbor::{ard_dist, simd_l2};
    use baux::semantics::pool::{DelayedPool, cand_key, forced_dim, raasp_pack, sparse_dist};

    let cands = tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(32));
    let num_pert = tc.draw(gs::integers::<usize>().min_value(1).max_value(dim));
    let seed = tc.draw(gs::integers::<u64>());
    let key = Key::new(seed);

    let mut center = Vec::with_capacity(dim);
    let mut lo = Vec::with_capacity(dim);
    let mut hi = Vec::with_capacity(dim);
    let mut target = Vec::with_capacity(dim);
    let mut weights = Vec::with_capacity(dim);
    for _ in 0..dim {
        let c = tc.draw(gs::floats::<f64>().min_value(0.1).max_value(0.9));
        let span = tc.draw(gs::floats::<f64>().min_value(0.01).max_value(0.1));
        center.push(c);
        lo.push((c - span).max(0.0));
        hi.push((c + span).min(1.0));
        target.push(tc.draw(gs::floats::<f64>().min_value(0.0).max_value(1.0)));
        weights.push(tc.draw(gs::floats::<f64>().min_value(0.2).max_value(5.0)));
    }

    let pool = DelayedPool::Raasp {
        key,
        center: &center,
        lower: &lo,
        upper: &hi,
        num_candidates: cands,
        num_pert,
    };
    let center32: Vec<f32> = center.iter().map(|&v| v as f32).collect();
    let lo32: Vec<f32> = lo.iter().map(|&v| v as f32).collect();
    let hi32: Vec<f32> = hi.iter().map(|&v| v as f32).collect();
    let pool32 = DelayedPool::<f32>::Raasp {
        key,
        center: &center32,
        lower: &lo32,
        upper: &hi32,
        num_candidates: cands,
        num_pert,
    };
    let base_ard = ard_dist(&center, &target, &weights);
    let base_unw = simd_l2(&center, &target);

    let mut row = vec![0.0_f64; dim];
    let mut idx_buf = vec![0usize; dim];
    let mut val_buf = vec![0.0_f64; dim];
    for c in 0..cands {
        pool.write_row(c, &mut row);
        let forced = forced_dim(cand_key(key, c), dim);
        let k_pert = raasp_pack(key, c, &lo, &hi, num_pert, &mut idx_buf, &mut val_buf).unwrap();
        assert!(k_pert >= 1 && idx_buf[..k_pert].contains(&forced));
        for t in 0..k_pert {
            let j = idx_buf[t];
            let val = val_buf[t];
            assert!(val >= lo[j] && val <= hi[j]);
            assert_eq!(row[j].to_bits(), val.to_bits());
        }
        for j in 0..dim {
            let v32 = pool32.eval(c, j);
            assert!(v32 >= lo32[j] && v32 <= hi32[j]);
        }

        let sp_ard = sparse_dist(
            base_ard,
            &center,
            &target,
            Some(&weights),
            &idx_buf[..k_pert],
            &val_buf[..k_pert],
        );
        let dense_ard = ard_dist(&row, &target, &weights);
        assert!(
            (sp_ard - dense_ard).abs() < 1.0e-9,
            "sparse ARD L2 {sp_ard} != dense {dense_ard}"
        );

        let sp_unw = sparse_dist(
            base_unw,
            &center,
            &target,
            None,
            &idx_buf[..k_pert],
            &val_buf[..k_pert],
        );
        let dense_unw = simd_l2(&row, &target);
        assert!(
            (sp_unw - dense_unw).abs() < 1.0e-9,
            "sparse L2 {sp_unw} != dense {dense_unw}"
        );
    }
}

/// Stateless Gray-code Sobol + hash-Owen scramble preserves exact 1D `(0, m,
/// 1)` dyadic stratification on every `2^m`-aligned block.
#[hegel::test]
fn sobol_owen(tc: TestCase) {
    use baux::semantics::key::Key;
    use baux::semantics::pool::{DelayedPool, joe_kuo};

    let m = tc.draw(gs::integers::<u32>().min_value(1).max_value(5));
    let n = 1usize << m;
    let dim = tc.draw(gs::integers::<usize>().min_value(0).max_value(15));
    let block = tc.draw(gs::integers::<u32>().min_value(0).max_value(16));
    let base_idx = block << m;
    let seed = tc.draw(gs::integers::<u64>());

    let lo = vec![0.0_f64; dim + 1];
    let hi = vec![1.0_f64; dim + 1];
    let dirs: Vec<[u32; 32]> = (0..=dim).map(joe_kuo).collect();
    let pool = DelayedPool::Sobol {
        key: Key::new(seed),
        lower: &lo,
        upper: &hi,
        dirs: &dirs,
        base_idx,
        num_candidates: n,
        scramble: true,
    };
    let mut bins = vec![0usize; n];
    for c in 0..n {
        let u = pool.eval(c, dim);
        assert!((0.0..1.0).contains(&u), "Sobol draw {u} out of [0, 1)");
        let b = ((u * n as f64).floor() as usize).min(n - 1);
        bins[b] += 1;
    }
    assert!(
        bins.iter().all(|&cnt| cnt == 1),
        "Owen-scrambled Sobol violated (0, {m}, 1) stratification: {bins:?}"
    );
}

/// Cycle-walking Feistel Latin Hypercube Design (`eval_lhd`) produces an exact
/// `0..N` bin permutation in every coordinate without allocating tables.
#[hegel::test]
fn lhd_bijection(tc: TestCase) {
    use baux::semantics::key::Key;
    use baux::semantics::pool::{DelayedPool, lhd_bin};

    let n = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
    let num_dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
    let seed = tc.draw(gs::integers::<u64>());
    let key = Key::new(seed);
    let lo = vec![0.0_f64; num_dim];
    let hi = vec![1.0_f64; num_dim];
    let pool = DelayedPool::Lhd {
        key,
        lower: &lo,
        upper: &hi,
        num_candidates: n,
    };

    for j in 0..num_dim {
        let mut seen = vec![false; n];
        for c in 0..n {
            let bin = lhd_bin(key, c, n, j);
            assert!(bin < n && !seen[bin], "duplicate LHD bin {bin} at dim {j}");
            seen[bin] = true;
            let u = pool.eval(c, j);
            let b_lo = bin as f64 / n as f64;
            let b_hi = (bin + 1) as f64 / n as f64;
            assert!(
                u >= b_lo && u <= b_hi,
                "LHD draw {u} outside stratum [{b_lo}, {b_hi}]"
            );
        }
    }
}

/// Generic `TopK<T>`, `ConstParamTy` posterior specialization, non-Euclidean
/// `loo_slice`, `scale_prior` `[10^-3, 10^3]`, UCB `beta = 1.0` default, and
/// NaN-safe `BetterScore<T>` selection.
#[hegel::test]
fn topk_score(tc: TestCase) {
    use baux::semantics::key::Key;
    use baux::semantics::neighbor::query::{posterior_const, posterior_eval};
    use baux::semantics::neighbor::{DistanceMetric, MetricKind, TopK, UncertaintyLaw};
    use baux::semantics::optimizer::surrogate::{
        AcquisitionRule, BetterScore, DEFAULT_BETA, loo_slice, scale_prior, select_pareto,
        select_topk,
    };

    assert_eq!(DEFAULT_BETA, 1.0);
    assert_eq!(
        AcquisitionRule::default(),
        AcquisitionRule::Ucb { beta: 1.0 }
    );

    let pos_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
    assert_eq!(BetterScore(pos_nan), BetterScore(neg_nan));
    assert!(BetterScore(pos_nan) < BetterScore(f64::NEG_INFINITY));
    assert!(BetterScore(neg_nan) < BetterScore(f64::NEG_INFINITY));
    assert!(BetterScore(f32::NAN) < BetterScore(f32::NEG_INFINITY));

    let n = tc.draw(gs::integers::<usize>().min_value(3).max_value(24));
    let k = tc.draw(gs::integers::<usize>().min_value(1).max_value(n.min(8)));
    let seed = tc.draw(gs::integers::<u64>());
    let key = Key::new(seed);
    let (epi_p, ale_p) = scale_prior(key, 0, 1.0_f64, true);
    assert!((1.0e-3..=1.0e3).contains(&epi_p) && (1.0e-6..=1.0).contains(&ale_p));

    let dists: Vec<f64> = (0..n)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.0).max_value(4.0)))
        .collect();
    let ys: Vec<f64> = (0..n)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(-5.0).max_value(5.0)))
        .collect();
    let vars: Vec<f64> = (0..n)
        .map(|_| tc.draw(gs::floats::<f64>().min_value(0.0).max_value(0.5)))
        .collect();

    let mut top = TopK::<f64, 8>::new(k);
    for (idx, &d) in dists.iter().enumerate() {
        top.push(d, idx as u32);
    }
    let mut sorted: Vec<(f64, u32)> = dists
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| (d, i as u32))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    assert_eq!(top.len(), k);
    for (idx, &(d, slot)) in top.as_slice().iter().enumerate() {
        assert_eq!(slot, sorted[idx].1);
        assert_eq!(d.to_bits(), sorted[idx].0.to_bits());
    }

    let scales = vec![0.25_f64; n];
    let p_const =
        posterior_const::<{ MetricKind::Hyperbolic }, { UncertaintyLaw::BiasVariance }, f64>(
            &dists,
            Some(&scales),
            0.25,
            &ys,
            Some(&vars),
            k,
            0.1,
            0.05,
            1.0,
            None,
            true,
        );
    let p_dyn = posterior_eval(
        DistanceMetric::Hyperbolic { local_k: 2 },
        UncertaintyLaw::BiasVariance,
        &dists,
        Some(&scales),
        0.25,
        &ys,
        Some(&vars),
        k,
        0.1,
        0.05,
        1.0,
        None,
        true,
    );
    assert_eq!(p_const, p_dyn);
    assert!(p_const.mu.is_finite() && p_const.se_total > 0.0);

    let mut mat = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            mat[r * n + c] = if r == c {
                0.0
            } else {
                (dists[r] - dists[c]).powi(2) + 0.01
            };
        }
    }
    let ll = loo_slice(
        DistanceMetric::Conformal { local_k: 2 },
        UncertaintyLaw::BiasVariance,
        &mat,
        &ys,
        &vars,
        &[0, 1],
        k,
        0.1,
        0.05,
        1.0,
        true,
    );
    assert!(ll.is_finite());

    let mut scores = ys.clone();
    scores[0] = pos_nan;
    scores[1] = f64::NEG_INFINITY;
    let mut chosen = vec![0usize; 2];
    let got = select_topk(&scores, key, &mut chosen);
    assert_eq!(got, 2);
    assert_ne!(chosen[0], 0, "NaN score must never win top-1 selection");
    assert_ne!(chosen[0], chosen[1]);
    assert!(BetterScore(scores[chosen[0]]) >= BetterScore(scores[chosen[1]]));

    let ses: Vec<f64> = vars.iter().map(|&v| v + 0.01).collect();
    let mut p_out = vec![0usize; k];
    let p_got = select_pareto(&scores, &ses, key, &mut p_out);
    assert_eq!(p_got, k);
}

/// `TurboState` failure-tolerance scaling, restart reset law, and arbitrary-`K`
/// `RoundPlan` ABI packing (`MODE_MASKED` vs `MODE_SOBOL`) without fixed
/// 4-candidate bounds.
#[hegel::test]
fn turbo_plan(tc: TestCase) {
    use baux::semantics::abi::{MODE_MASKED, MODE_SOBOL};
    use baux::semantics::key::Key;
    use baux::semantics::neighbor::{MetricKind, UncertaintyLaw};
    use baux::semantics::optimizer::surrogate::AcquisitionRule;
    use baux::semantics::optimizer::trust_region::{
        LifecyclePhase, PoolSampler, RadiusAction, RoundPlan, SurrogateMode, TurboState,
        default_pool, turbo_tolerance,
    };

    let dim = tc.draw(gs::integers::<usize>().min_value(1).max_value(64));
    let arms = tc.draw(gs::integers::<usize>().min_value(1).max_value(8));
    let cands = tc.draw(gs::integers::<usize>().min_value(5).max_value(2048));
    let seed = tc.draw(gs::integers::<u64>());

    let expected_tol = ((4.0 / arms as f64).max(dim as f64 / arms as f64)).ceil() as usize;
    assert_eq!(
        turbo_tolerance(dim as f64, arms) as usize,
        expected_tol.max(1)
    );
    let state = TurboState::for_dims(dim, arms, 0.8, 0.25, 1.6, None);
    assert_eq!(state.failure_limit, expected_tol.max(1));
    let overridden = TurboState::for_dims(dim, arms, 0.8, 0.25, 1.6, Some(7));
    assert_eq!(overridden.failure_limit, 7);

    let mut cur = TurboState::new(0.4, 0.25, 1.6, 2);
    let (next1, act1, res1) = cur.next_step(false);
    assert_eq!((act1, res1), (RadiusAction::Hold, false));
    let (next2, act2, res2) = next1.next_step(false);
    assert_eq!((act2, res2), (RadiusAction::Shrink, true));
    assert_eq!(next2.length, 0.4);
    cur = next2;
    for _ in 0..2 {
        let (n, a, r) = cur.next_step(true);
        assert_eq!((a, r), (RadiusAction::Hold, false));
        cur = n;
    }
    let (expanded, act3, res3) = cur.next_step(true);
    assert_eq!((act3, res3), (RadiusAction::Expand, false));
    assert!((expanded.length - 0.8).abs() < 1.0e-12);

    assert!(default_pool(dim) >= 10 && default_pool(dim) <= 5000);
    let mut plan = RoundPlan::turbo_enn(
        Key::new(seed),
        3,
        dim,
        cands,
        arms,
        state,
        SurrogateMode::Enn {
            neighbors: 16,
            epistemic: 0.05,
            aleatoric: 0.05,
            scale: 1.0,
            metric: MetricKind::Euclidean,
            law: UncertaintyLaw::BiasVariance,
            noise_aware: true,
        },
        AcquisitionRule::default(),
    );
    let pd = plan.pool_dispatch(4, 24, 2);
    let sh = plan.select_header(24, 2, 4, true);
    assert_eq!(pd.num_candidates, cands as u32);
    assert_eq!(sh.num_candidates, cands as u32);
    assert_eq!(sh.num_arms, arms as u32);
    assert_eq!(sh.beta, 1.0);

    plan.phase = LifecyclePhase::Active {
        sampler: PoolSampler::Sobol { base_idx: 0 },
        surrogate: SurrogateMode::NoSurrogate,
    };
    assert_eq!(plan.abi_mode(), MODE_SOBOL);
    plan.phase = LifecyclePhase::Active {
        sampler: PoolSampler::SobolMasked {
            base_idx: 0,
            prob: 0.5,
        },
        surrogate: SurrogateMode::NoSurrogate,
    };
    assert_eq!(plan.abi_mode(), MODE_MASKED);
}

/// `ybounds` warp/inverse/Jacobian roundtrip and `morbo` simplex, NDS, and 2D
/// hypervolume monotonicity laws.
#[hegel::test]
fn warp_morbo(tc: TestCase) {
    use baux::semantics::key::Key;
    use baux::semantics::morbo::{chebyshev, dominates, hv_pair, nds_ranks, sample_simplex};
    use baux::semantics::ybounds::{inv_scalar, jac_inv, natural_point, warp_scalar};

    let lo_b = tc.draw(gs::floats::<f64>().min_value(-10.0).max_value(0.0));
    let span = tc.draw(gs::floats::<f64>().min_value(0.5).max_value(10.0));
    let hi_b = lo_b + span;
    let frac = tc.draw(gs::floats::<f64>().min_value(0.05).max_value(0.95));
    let y = lo_b + frac * span;

    for (lo, hi) in [
        (lo_b, f64::INFINITY),
        (f64::NEG_INFINITY, hi_b),
        (lo_b, hi_b),
    ] {
        let z = warp_scalar(y, lo, hi);
        let back = inv_scalar(z, lo, hi);
        assert!(
            (back - y).abs() < 1.0e-7,
            "ybounds roundtrip failed for ({lo}, {hi}): {y} -> {z} -> {back}"
        );
        let eps = 1.0e-6;
        let fd = (inv_scalar(z + eps, lo, hi) - inv_scalar(z - eps, lo, hi)) / (2.0 * eps);
        let jac = jac_inv(z, lo, hi);
        assert!(
            (fd - jac).abs() <= 1.0e-4 * jac.abs().max(1.0),
            "ybounds Jacobian mismatch: {jac} vs fd {fd}"
        );
        let (nat_mu, nat_se, nat_epi, nat_ale) = natural_point(z, 0.2, 0.15, 0.1, lo, hi);
        assert!(
            (nat_mu - back).abs() < 1.0e-4 && nat_se >= 0.0 && nat_epi >= 0.0 && nat_ale >= 0.0
        );
    }

    let seed = tc.draw(gs::integers::<u64>());
    let mut weights = [0.0_f64; 3];
    sample_simplex(Key::new(seed), &mut weights);
    let w_sum: f64 = weights.iter().sum();
    assert!((w_sum - 1.0).abs() < 1.0e-12 && weights.iter().all(|&w| w > 0.0));

    let s = chebyshev(&[1.0, 2.0, 3.0], &[0.0; 3], &[4.0; 3], &weights, 0.05, true);
    assert!(s.is_finite());

    let objs = [1.0, 1.0, 2.0, 2.0, 0.5, 2.5];
    let mut ranks = [0u32; 3];
    nds_ranks(&objs, 3, 2, &mut ranks);
    assert!(dominates(&objs[2..4], &objs[0..2]));
    assert_eq!(ranks[1], 0);
    assert_eq!(ranks[2], 0);
    assert!(ranks[0] >= 1);

    let mut scratch = [(0.0, 0.0); 3];
    let hv_sub = hv_pair(&objs[0..2], 1, 0.0, 0.0, &mut scratch[..1]);
    let hv_all = hv_pair(&objs, 3, 0.0, 0.0, &mut scratch);
    assert!(hv_all >= hv_sub && hv_sub > 0.0);
}

/// GPU layout verification for tiled arbitrary-`num_candidates` `abi` structs
/// (`PoolDispatch`, `SelectHeader`, `ArmDecision`, `TileSpan`) on Apple Metal.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[hegel::test]
fn pool_abi(tc: TestCase) {
    use baux::semantics::abi::{ArmDecision, PoolDispatch, SelectHeader, TileSpan, metal_source};
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};

    let pd = PoolDispatch {
        key_hi: tc.draw(gs::integers::<u64>()),
        key_lo: tc.draw(gs::integers::<u64>()),
        root_seed: tc.draw(gs::integers::<u64>()),
        stream_seed: tc.draw(gs::integers::<u64>()),
        radius: tc.draw(gs::floats::<f32>().min_value(0.01).max_value(2.0)),
        prob_pert: tc.draw(gs::floats::<f32>().min_value(0.0).max_value(1.0)),
        num_candidates: tc.draw(gs::integers::<u32>().min_value(1).max_value(10_000)),
        num_arms: tc.draw(gs::integers::<u32>().min_value(1).max_value(16)),
        num_dim: tc.draw(gs::integers::<u32>().min_value(1).max_value(4096)),
        num_pert: 8,
        base_idx: tc.draw(gs::integers::<u32>()),
        tiles: 4,
        history: 24,
        mode: tc.draw(gs::integers::<u32>().min_value(0).max_value(5)),
        base_slot: 0,
        blocks: 1,
        initialize: 0,
        scramble: 1,
    };
    let sh = SelectHeader {
        root_seed: pd.root_seed,
        acq_seed: tc.draw(gs::integers::<u64>()),
        epistemic_scale: 0.05,
        aleatoric_scale: 0.05,
        y_scale: 1.0,
        beta: 1.0,
        radius: pd.radius,
        axis_base: 1.0,
        axis_weight: 1.0,
        morbo_alpha: 0.05,
        num_candidates: pd.num_candidates,
        num_arms: pd.num_arms,
        neighbors: 16,
        history: 24,
        acquisition: 0,
        tiles: 4,
        mode: pd.mode,
        resident_history: 24,
        implicit_history: 0,
        forced_candidate: baux::semantics::abi::NO_FORCED,
        distance_metric: 0,
        scale_neighbors: 16,
        incumbent_index: 0,
        candidate_floor: 0,
        adaptive_k: 0,
        uncertainty_law: 1,
        axis_enabled: 0,
        num_metrics: 2,
        has_yvar: 1,
        has_ybounds: 1,
    };
    let ad = ArmDecision {
        root_seed: pd.root_seed,
        seed: pd.stream_seed,
        index: tc.draw(gs::integers::<u32>()),
        arm: 0,
        valid: 1,
        mode: pd.mode,
        radius: pd.radius,
        score: 2.5,
        pred_mean: 1.5,
        pred_se: 1.0,
        pred_epi: 0.8,
        pred_ale: 0.6,
        incumbent_mean: 1.0,
        incumbent_se: 0.2,
        neighbors: 16,
        incumbent_k: 16,
    };
    let ts = TileSpan {
        leaf: 3,
        start: 64,
        length: 128,
        pad: 0,
    };
    let src = metal_source(
        r#"
kernel void check_pool(constant PoolDispatch& p [[buffer(0)]],
                       constant SelectHeader& h [[buffer(1)]],
                       constant ArmDecision& a [[buffer(2)]],
                       constant TileSpan& t [[buffer(3)]],
                       device uint* out [[buffer(4)]]) {
    out[0] = uint(p.key_hi & 0xfffffffful);
    out[1] = p.num_candidates;
    out[2] = p.num_dim;
    out[3] = as_type<uint>(p.prob_pert);
    out[4] = as_type<uint>(p.radius);
    out[5] = uint(h.acq_seed >> 32);
    out[6] = h.num_arms;
    out[7] = h.forced_candidate;
    out[8] = as_type<uint>(h.beta);
    out[9] = h.has_ybounds;
    out[10] = a.index;
    out[11] = as_type<uint>(a.pred_ale);
    out[12] = t.length;
    out[13] = t.leaf;
}
"#,
    );
    static PIPE: std::sync::OnceLock<(Device, metal::ComputePipelineState, metal::CommandQueue)> =
        std::sync::OnceLock::new();
    let (dev, pipe, q) = PIPE.get_or_init(|| {
        let dev = Device::system_default().expect("Metal device");
        let lib = dev
            .new_library_with_source(&src, &CompileOptions::new())
            .unwrap();
        let fun = lib.get_function("check_pool", None).unwrap();
        let pipe = dev.new_compute_pipeline_state_with_function(&fun).unwrap();
        let q = dev.new_command_queue();
        (dev, pipe, q)
    });
    let out = dev.new_buffer(14 * 4, MTLResourceOptions::StorageModeShared);
    let cmd = q.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipe);
    enc.set_bytes(
        0,
        size_of::<PoolDispatch>() as u64,
        std::ptr::from_ref(&pd).cast(),
    );
    enc.set_bytes(
        1,
        size_of::<SelectHeader>() as u64,
        std::ptr::from_ref(&sh).cast(),
    );
    enc.set_bytes(
        2,
        size_of::<ArmDecision>() as u64,
        std::ptr::from_ref(&ad).cast(),
    );
    enc.set_bytes(
        3,
        size_of::<TileSpan>() as u64,
        std::ptr::from_ref(&ts).cast(),
    );
    enc.set_buffer(4, Some(&out), 0);
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(one, one);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
    let got = unsafe { std::slice::from_raw_parts(out.contents().cast::<u32>(), 14) };
    assert_eq!(
        got,
        &[
            pd.key_hi as u32,
            pd.num_candidates,
            pd.num_dim,
            pd.prob_pert.to_bits(),
            pd.radius.to_bits(),
            (sh.acq_seed >> 32) as u32,
            sh.num_arms,
            sh.forced_candidate,
            sh.beta.to_bits(),
            sh.has_ybounds,
            ad.index,
            ad.pred_ale.to_bits(),
            ts.length,
            ts.leaf,
        ]
    );
}
