use super::resources::{half_bits, mhc_bias};
use super::*;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

#[derive(Debug, Clone, Copy)]
pub(super) struct Mhc4Initialization {
    policy: baux::semantics::mhc4::Initialization,
}

impl std::ops::Deref for Mhc4Initialization {
    type Target = baux::semantics::mhc4::Initialization;
    fn deref(&self) -> &Self::Target {
        &self.policy
    }
}

impl Mhc4Initialization {
    pub(super) fn new(architecture: ResidualArchitecture) -> Result<Self, String> {
        if !architecture.is_mhc4() {
            return Err("fbt-pisa1-mhc4 initialization requires model fbt-pisa1-mhc4-v1".into());
        }

        Ok(Self {
            policy: baux::semantics::mhc4::Initialization::new(),
        })
    }
}

/// Draw independent Gaussian columns and normalize each column to an exact
/// FP32 norm before the single FP16 rounding step. Matrices use row-major
/// `[input, output]` storage, matching the Metal projection kernels.
fn fill_wide(values: &mut [u16], input: usize, output: usize, seed: u64, column_norm: f32) {
    use rayon::prelude::*;
    const STRIPE: usize = 4096;
    let base_addr = values.as_mut_ptr() as usize;
    let stripes = output.div_ceil(STRIPE);
    (0..stripes).into_par_iter().for_each(move |stripe| {
        let col_start = stripe * STRIPE;
        let cols = (output - col_start).min(STRIPE);
        let mut sub = vec![0.0_f32; input * cols];
        let mut norms = vec![0.0_f64; cols];
        let stripe_seed = crate::hash::splitmix64(seed.wrapping_add(stripe as u64));
        let mut random = StdRng::seed_from_u64(stripe_seed);
        for row in 0..input {
            let row_off = row * cols;
            for col in 0..cols {
                let sample: f32 = StandardNormal.sample(&mut random);
                sub[row_off + col] = sample;
                norms[col] += f64::from(sample) * f64::from(sample);
            }
        }
        for norm in &mut norms {
            *norm = f64::from(column_norm) / norm.sqrt();
        }
        let ptr = base_addr as *mut u16;
        for row in 0..input {
            let out_off = row * output + col_start;
            let row_off = row * cols;
            for col in 0..cols {
                let scaled = (f64::from(sub[row_off + col]) * norms[col]) as f32;
                unsafe { *ptr.add(out_off + col) = half_bits(scaled) };
            }
        }
    });
}

fn fill_spherical(values: &mut [u16], input: usize, output: usize, seed: u64, column_norm: f32) {
    let size = input * output;
    if values.len() == size {
        if output > 4096 {
            fill_wide(values, input, output, seed, column_norm);
            return;
        }
        let mut random = StdRng::seed_from_u64(seed);
        baux::semantics::mhc4::columns(
            values,
            input,
            output,
            column_norm,
            || StandardNormal.sample(&mut random),
            half_bits,
        )
        .expect("fixed mHC4 Gaussian column initialization");
        return;
    }

    use rayon::prelude::*;
    let chunks: Vec<_> = values.chunks_exact_mut(size).collect();
    chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(index, chunk)| {
            let chunk_seed = crate::hash::splitmix64(seed.wrapping_add(index as u64));
            let mut random = StdRng::seed_from_u64(chunk_seed);
            baux::semantics::mhc4::columns(
                chunk,
                input,
                output,
                column_norm,
                || StandardNormal.sample(&mut random),
                half_bits,
            )
            .expect("fixed mHC4 Gaussian column initialization");
        });
}

fn spherical(
    runtime: &Runtime,
    matrices: usize,
    input: usize,
    output: usize,
    seed: u64,
    column_norm: f32,
) -> Buffer {
    let elements = matrices
        .checked_mul(input)
        .and_then(|v| v.checked_mul(output))
        .expect("fixed model tensor dimensions must fit usize");
    let buffer = runtime.buffer::<u16>(elements);
    let values =
        unsafe { std::slice::from_raw_parts_mut(buffer.contents().cast::<u16>(), elements) };
    fill_spherical(values, input, output, seed, column_norm);
    buffer
}

fn mhc_controls(runtime: &Runtime, layers: usize, scale: f32, lambda: f32) -> Buffer {
    let mut values = Vec::with_capacity(layers * 4);
    for _ in 0..layers {
        values.extend([
            half_bits(scale),
            half_bits(scale),
            half_bits(scale),
            half_bits(lambda),
        ]);
    }
    runtime.buffer_with(&values)
}

fn save_triple(path: &std::path::Path, g: &[u16], d: &[u16], r: &[u16]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!(".triple-{}.tmp", std::process::id()));
    let Ok(mut out) = std::fs::File::create(&tmp) else {
        return;
    };
    let write_slice = |f: &mut std::fs::File, s: &[u16]| {
        use std::io::Write;
        let bytes = unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 2) };
        f.write_all(bytes)
    };
    if write_slice(&mut out, g).is_ok()
        && write_slice(&mut out, d).is_ok()
        && write_slice(&mut out, r).is_ok()
    {
        let _ = std::fs::rename(&tmp, path);
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn cached_triple(runtime: &Runtime, seed: u64, down_norm: f32) -> (Buffer, Buffer, Buffer) {
    let layers = MODEL_LAYERS as usize;
    let experts = routing::MOE_EXPERTS as usize;
    let width = WIDTH as usize;
    let expert_width = routing::ROUTED_WIDTH as usize;
    let gate_up = routing::MOE_GATEUP as usize;
    let vocab = VOCAB as usize;
    let g_elems = layers * experts * width * gate_up;
    let d_elems = layers * experts * expert_width * width;
    let r_elems = width * vocab;
    let (g_bytes, d_bytes, r_bytes) = (g_elems * 2, d_elems * 2, r_elems * 2);
    let total = (g_bytes + d_bytes + r_bytes) as u64;
    let path = std::path::PathBuf::from(format!(".cache/baux/weights/mhc4-big-v1-{seed:016x}.bin"));
    if let Ok(file) = std::fs::File::open(&path)
        && file.metadata().map(|m| m.len() == total).unwrap_or(false)
        && let (Some(g), Some(d), Some(r)) = (
            runtime.mmap_buffer(&file, 0, g_bytes),
            runtime.mmap_buffer(&file, g_bytes as i64, d_bytes),
            runtime.mmap_buffer(&file, (g_bytes + d_bytes) as i64, r_bytes),
        )
    {
        return (g, d, r);
    }
    let gate_up_buf = runtime.buffer::<u16>(g_elems);
    let down_buf = runtime.buffer::<u16>(d_elems);
    let readout_buf = runtime.buffer::<u16>(r_elems);
    let g_slice =
        unsafe { std::slice::from_raw_parts_mut(gate_up_buf.contents().cast::<u16>(), g_elems) };
    let d_slice =
        unsafe { std::slice::from_raw_parts_mut(down_buf.contents().cast::<u16>(), d_elems) };
    let r_slice =
        unsafe { std::slice::from_raw_parts_mut(readout_buf.contents().cast::<u16>(), r_elems) };
    std::thread::scope(|scope| {
        scope.spawn(|| {
            fill_spherical(
                g_slice,
                width,
                gate_up,
                crate::hash::splitmix64(seed ^ 0xa54f_f53a),
                1.0,
            );
        });
        scope.spawn(|| {
            fill_spherical(
                d_slice,
                expert_width,
                width,
                crate::hash::splitmix64(seed ^ 0x510e_527f),
                down_norm,
            );
        });
        scope.spawn(|| {
            fill_spherical(
                r_slice,
                width,
                vocab,
                crate::hash::splitmix64(seed ^ 0x9b05_688c),
                1.0,
            );
        });
    });
    save_triple(&path, g_slice, d_slice, r_slice);
    (gate_up_buf, down_buf, readout_buf)
}

impl CandidateWeights {
    pub(super) fn block_scale(
        &self,
        policy: Option<Mhc4Initialization>,
        shape: crate::config::TrustRegionShape,
        family: &str,
        expert: Option<usize>,
        initial_rms: f64,
    ) -> Result<(f32, f32), String> {
        let multiplier = if shape == crate::config::TrustRegionShape::TensorFamilyStatic {
            baux::semantics::mhc4::expert_multiplier(family, expert)
        } else {
            1.0
        };
        if let Some(policy) = policy {
            return Ok((policy.proposal_scale(family)? * multiplier, multiplier));
        }
        let floor = if matches!(family, "mhc_attention_control" | "mhc_moe_control") {
            0.02
        } else {
            1e-6
        };
        Ok((initial_rms.max(floor) as f32 * multiplier, multiplier))
    }

    pub(super) fn mhc4_weights(
        runtime: &Runtime,
        seed: u64,
        architecture: ResidualArchitecture,
    ) -> Result<Self, String> {
        let policy = Mhc4Initialization::new(architecture)?;
        let layers = MODEL_LAYERS as usize;
        let width = WIDTH as usize;
        let qkv_width = QKV_WIDTH as usize;
        let draw = |matrices, input, output, domain, norm| {
            spherical(
                runtime,
                matrices,
                input,
                output,
                crate::hash::splitmix64(seed ^ domain),
                norm,
            )
        };
        let (gate_up_buf, down_buf, readout_buf) = cached_triple(runtime, seed, policy.down_norm());

        Ok(Self {
            architecture,
            initialization: crate::config::ModelInitialization::FbtPisa1Mhc4,
            router: draw(
                layers,
                width,
                routing::ROUTED_EXPERTS as usize,
                0x6a09_e667,
                policy.router_column_norm,
            ),
            qkv: draw(layers, width, qkv_width, 0xbb67_ae85, 1.0),
            output: draw(layers, width, width, 0x3c6e_f372, policy.residual_gain),
            gate_up: gate_up_buf,
            down: down_buf,
            attention_norm: filled(runtime, layers * width, 0x3c00),
            ffn_norm: filled(runtime, layers * width, 0x3c00),
            // Tied embedding/readout token columns have unit norm, giving
            // unit logit variance after RMS normalization.
            readout: readout_buf,
            mask_embed: draw(1, width, 1, 0x4d41_534b, 1.0),
            // Each column has unit norm across the 64 within-block positions.
            // The same basis gates fine-token embeddings during compression
            // and supplies the position component to the local decoder.
            index_query: draw(1, 64, width, 0x494e_4458, 1.0),
            feedback_state: draw(1, width, width, 0x1f83_d9ab, 1.0),
            feedback_gate: draw(1, width, width, 0x5be0_cd19, 1.0),
            mhc_attention_predictor: draw(
                layers,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
                0x428a_2f98,
                1.0,
            ),
            mhc_attention_bias: mhc_bias(runtime, layers),
            mhc_attention_control: mhc_controls(
                runtime,
                layers,
                policy.predictor_control,
                policy.transport_lambda,
            ),
            mhc_moe_predictor: draw(
                layers,
                MHC_INPUT as usize,
                MHC_COEFFICIENTS as usize,
                0x7137_4491,
                1.0,
            ),
            mhc_moe_bias: mhc_bias(runtime, layers),
            mhc_moe_control: mhc_controls(
                runtime,
                layers,
                policy.predictor_control,
                policy.transport_lambda,
            ),
            final_norm: filled(runtime, width, 0x3c00),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mhc4_scales() {
        let policy = Mhc4Initialization::new(ResidualArchitecture::Mhc4).unwrap();
        assert_eq!(policy.residual_depth, 22.0);
        assert_eq!(policy.sites, 14);
        assert!((policy.residual_gain - 22.0f32.sqrt().recip()).abs() < 1.0e-7);
        assert!((policy.transport_lambda - 1.0 / 112.0).abs() < 1.0e-7);

        let retained = (1.0 - policy.transport_lambda).powi(policy.sites as i32);
        assert!((retained - 0.882).abs() < 0.001);
    }

    #[test]
    fn router_scale() {
        let policy = Mhc4Initialization::new(ResidualArchitecture::Mhc4).unwrap();
        let expected_max =
            policy.router_column_norm * (2.0 * (routing::ROUTED_EXPERTS as f32).ln()).sqrt();
        assert!((expected_max - 1.0).abs() < 1.0e-6);
    }
}
