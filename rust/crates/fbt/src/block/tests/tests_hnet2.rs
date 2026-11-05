//! End-to-end verification and 1M-byte benchmark for 2-stage 512-d learned
//! H-Net.

use super::{hnet_batch, router_mode, *};
use crate::apple_gpu::buffer_slice;
use baux::semantics::hnet::{HNet2Params, hnet2_forward, hnet2_penalty, hnet2_reverse};

fn read_f16(buf: &BufferRef, offset: u64, count: usize) -> Vec<f32> {
    let raw = unsafe {
        std::slice::from_raw_parts(
            buf.contents()
                .cast::<u8>()
                .add(offset as usize)
                .cast::<u16>(),
            count,
        )
    };
    raw.iter()
        .map(|&x| crate::context::probe::half(x))
        .collect()
}

struct OwnedParams {
    u0_table: Vec<f32>,
    gates_table: Vec<f32>,
    hnet_all: Vec<f32>,
    attn_all: Vec<f32>,
    ffn_all: Vec<f32>,
    final_norm: Vec<f32>,
    readout: Vec<f32>,
    gamma1: f32,
    gamma3: f32,
}

impl OwnedParams {
    fn as_ref(&self) -> HNet2Params<'_> {
        let d = WIDTH as usize;
        HNet2Params {
            d_model: d,
            u0_table: &self.u0_table,
            gates_table: &self.gates_table,
            w_q: &self.hnet_all[0..d * d],
            w_k: &self.hnet_all[d * d..2 * d * d],
            w_res: &self.hnet_all[2 * d * d..3 * d * d],
            b_res: &self.hnet_all[3 * d * d..3 * d * d + d],
            attn_norm1: &self.attn_all[d..2 * d],
            ffn_norm1: &self.ffn_all[d..2 * d],
            attn_norm3: &self.attn_all[3 * d..4 * d],
            ffn_norm3: &self.ffn_all[3 * d..4 * d],
            ffn_norm4: &self.ffn_all[4 * d..5 * d],
            final_norm: &self.final_norm,
            readout: &self.readout,
            gamma1: self.gamma1,
            gamma3: self.gamma3,
        }
    }
}

fn extract_params(scorer: &BlockDecoder, weights: CandidateRow<'_>) -> Result<OwnedParams, String> {
    let hnet2 = scorer.buffers.hnet2()?;
    let d = WIDTH as usize;
    let v = 256usize;
    let scalars = buffer_slice::<f32>(&hnet2.scalars, 4)?;
    Ok(OwnedParams {
        u0_table: buffer_slice::<f32>(&hnet2.u0_table, v * d)?.to_vec(),
        gates_table: buffer_slice::<f32>(&hnet2.gates_table, v * 2)?.to_vec(),
        hnet_all: read_f16(weights.buffer, weights.hnet, 3 * d * d + d),
        attn_all: read_f16(weights.buffer, weights.attention_norm, 5 * d),
        ffn_all: read_f16(weights.buffer, weights.ffn_norm, 5 * d),
        final_norm: read_f16(weights.buffer, weights.final_norm, d),
        readout: read_f16(weights.buffer, weights.readout, d * v),
        gamma1: scalars[0],
        gamma3: scalars[1],
    })
}

fn check_hnet2(
    runtime: &Runtime,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    bytes: &[u8],
) -> Result<(usize, usize), String> {
    let batch = hnet_batch(bytes);
    let _ = scorer.buffers.hnet2()?;
    let scored = scorer.score_batch(runtime, decoder, weights, &batch)?;
    assert!(scored.quality.mean_nll.is_finite());
    verify_oracle(scorer, weights, &batch.tokens, &batch.labels)
}

fn verify_oracle(
    scorer: &BlockDecoder,
    weights: CandidateRow<'_>,
    tokens: &[u32],
    labels: &[u32],
) -> Result<(usize, usize), String> {
    let n = tokens.len();
    let d = WIDTH as usize;
    let hnet2 = scorer.buffers.hnet2()?;
    let (m0, m1) = hnet2.counts()?;
    let (m0, m1) = (m0 as usize, m1 as usize);
    let owned = extract_params(scorer, weights)?;
    let params = owned.as_ref();
    let fwd = hnet2_forward(&params, tokens)?;
    assert_eq!(m0, fwd.origins0.len(), "Stage-0 chunk count mismatch");
    assert_eq!(m1, fwd.origins1.len(), "Stage-1 chunk count mismatch");
    assert_eq!(
        buffer_slice::<u32>(&hnet2.origins0, m0)?,
        fwd.origins0.as_slice()
    );
    assert_eq!(buffer_slice::<u32>(&hnet2.plug0, n)?, fwd.plug0.as_slice());
    assert_eq!(
        buffer_slice::<u32>(&hnet2.origins1, m1)?,
        fwd.origins1.as_slice()
    );
    assert_eq!(buffer_slice::<u32>(&hnet2.plug1, m0)?, fwd.plug1.as_slice());

    let probs0 = buffer_slice::<f32>(&hnet2.probs0, n)?;
    for i in 0..n {
        assert!(
            (probs0[i] - fwd.probs0[i]).abs() < 1e-4,
            "prob0 mismatch at {i}"
        );
    }
    let probs1 = buffer_slice::<f32>(&hnet2.probs1, m0)?;
    for c in 0..m0 {
        assert!(
            (probs1[c] - fwd.probs1[c]).abs() < 1e-4,
            "prob1 mismatch at {c}"
        );
    }
    let saved_h1 = buffer_slice::<f32>(&hnet2.tile_z0, m0 * d)?;
    for i in 0..m0 * d {
        assert!((saved_h1[i] - fwd.h1[i]).abs() < 1e-4, "h1 mismatch at {i}");
    }
    verify_reverse(scorer, &params, &fwd, tokens, labels)?;
    Ok((m0, m1))
}

fn verify_reverse(
    scorer: &BlockDecoder,
    params: &HNet2Params<'_>,
    fwd: &baux::semantics::hnet::HNet2Forward,
    tokens: &[u32],
    labels: &[u32],
) -> Result<(), String> {
    let n = tokens.len();
    let d = WIDTH as usize;
    let (m0, m1) = (fwd.origins0.len(), fwd.origins1.len());
    let hnet2 = scorer.buffers.hnet2()?;
    for (idx, &c0) in fwd.origins1.iter().enumerate() {
        assert!((c0 as usize) < m0);
        if idx > 0 {
            assert!(c0 > fwd.origins1[idx - 1]);
        }
    }
    let core_f16 = buffer_slice::<u16>(&hnet2.core_h1, m1 * d)?;
    let core_f32: Vec<f32> = core_f16
        .iter()
        .map(|&x| crate::context::probe::half(x))
        .collect();
    let rev = hnet2_reverse(params, fwd, &core_f32, tokens, labels)?;
    let ema1 = buffer_slice::<f32>(&hnet2.ema1, m1 * d)?;
    for i in 0..m1 * d {
        assert!((ema1[i] - rev.ema1[i]).abs() < 1e-4, "ema1 mismatch at {i}");
    }
    let ema0 = buffer_slice::<f32>(&hnet2.chunks_h0, m0 * d)?;
    for i in 0..m0 * d {
        assert!((ema0[i] - rev.ema0[i]).abs() < 1e-4, "ema0 mismatch at {i}");
    }
    let losses = buffer_slice::<f32>(&scorer.buffers.losses, n)?;
    let preds = buffer_slice::<u32>(&scorer.proposals, n)?;
    assert_eq!(preds, rev.predictions.as_slice());
    for i in 0..n {
        assert!(
            (losses[i] - rev.losses[i]).abs() < 1e-3,
            "loss mismatch at {i}"
        );
    }
    let st = hnet2.last_stats.get();
    let (_, pen) = hnet2_penalty(&fwd.probs0, m0, &fwd.probs1, m1);
    assert!((st.total_penalty - pen).abs() < 1e-5);
    Ok(())
}

#[test]
fn hnet2_oracle() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let d = WIDTH as usize;
        let orig_hnet = buffer_slice::<u16>(&model.hnet, 3 * d * d + d)?.to_vec();
        let decoder = decode::Decoder::with_visits(&runtime, CONTEXT, false, 1)?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;

        let utf8 = "Hello 🌍 — Δx = ∑ f(x_i)! Two-stage dynamic chunking over UTF-8 bytes.\n"
            .as_bytes()
            .repeat(12);
        let flat = model.flatten(&runtime)?;
        let (m0, m1) = check_hnet2(
            &runtime,
            &scorer,
            &decoder,
            model.row(&flat)?,
            &utf8[..=512],
        )?;
        assert!(
            m0 > 64 && m0 < 256 && m1 > 24 && m1 < m0,
            "utf8 m0={m0} m1={m1}"
        );

        let irreg: Vec<u8> = (0..=300).map(|i| ((i * 61 + 29) % 256) as u8).collect();
        let (im0, im1) = check_hnet2(&runtime, &scorer, &decoder, model.row(&flat)?, &irreg)?;
        assert!(
            im0 > 32 && im0 < 200 && im1 > 12 && im1 < im0,
            "irreg im0={im0} im1={im1}"
        );

        check_tiling(
            &runtime,
            &scorer,
            &decoder,
            model.row(&flat)?,
            &utf8[..=512],
        )?;
        drop(flat);
        check_regimes(&model, &orig_hnet, &scorer, &decoder, &irreg[..=256])
    })
}

fn check_tiling(
    runtime: &Runtime,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    bytes: &[u8],
) -> Result<(), String> {
    let n = bytes.len() - 1;
    let d = WIDTH as usize;
    let batch = hnet_batch(bytes);
    scorer.score_batch(runtime, decoder, weights, &batch)?;
    let hnet2 = scorer.buffers.hnet2()?;
    let (m0_a, m1_a) = hnet2.counts()?;
    let orig0_a = buffer_slice::<u32>(&hnet2.origins0, m0_a as usize)?.to_vec();
    let orig1_a = buffer_slice::<u32>(&hnet2.origins1, m1_a as usize)?.to_vec();
    let probs0_a = buffer_slice::<f32>(&hnet2.probs0, n)?.to_vec();
    let probs1_a = buffer_slice::<f32>(&hnet2.probs1, m0_a as usize)?.to_vec();
    let ema0_a = buffer_slice::<f32>(&hnet2.chunks_h0, m0_a as usize * d)?.to_vec();
    let loss_a = buffer_slice::<f32>(&scorer.buffers.losses, n)?.to_vec();

    scorer.hnet2_tiled(decoder, weights, &[100, 150, 93, 169], 1)?;
    let (m0_b, m1_b) = hnet2.counts()?;
    assert_eq!((m0_a, m1_a), (m0_b, m1_b));
    assert_eq!(
        orig0_a,
        buffer_slice::<u32>(&hnet2.origins0, m0_b as usize)?
    );
    assert_eq!(
        orig1_a,
        buffer_slice::<u32>(&hnet2.origins1, m1_b as usize)?
    );
    let probs0_b = buffer_slice::<f32>(&hnet2.probs0, n)?;
    let probs1_b = buffer_slice::<f32>(&hnet2.probs1, m0_b as usize)?;
    let ema0_b = buffer_slice::<f32>(&hnet2.chunks_h0, m0_b as usize * d)?;
    let loss_b = buffer_slice::<f32>(&scorer.buffers.losses, n)?;
    for i in 0..n {
        assert!((probs0_a[i] - probs0_b[i]).abs() < 1e-5);
        assert!((loss_a[i] - loss_b[i]).abs() < 1e-4);
    }
    for c in 0..m0_a as usize {
        assert!((probs1_a[c] - probs1_b[c]).abs() < 1e-5);
    }
    for i in 0..ema0_a.len() {
        assert!((ema0_a[i] - ema0_b[i]).abs() < 1e-4);
    }
    Ok(())
}

fn check_regimes(
    model: &CandidateWeights,
    orig_hnet: &[u16],
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    bytes: &[u8],
) -> Result<(), String> {
    let runtime = Runtime::shared()?;
    let n = bytes.len() - 1;
    let d = WIDTH as usize;
    router_mode(model, orig_hnet, None, -1)?;
    let flat_open = model.flatten(&runtime)?;
    let (om0, om1) = check_hnet2(&runtime, scorer, decoder, model.row(&flat_open)?, bytes)?;
    drop(flat_open);
    assert_eq!((om0, om1), (n, n));

    router_mode(model, orig_hnet, None, 1)?;
    let flat_closed = model.flatten(&runtime)?;
    let (cm0, cm1) = check_hnet2(&runtime, scorer, decoder, model.row(&flat_closed)?, bytes)?;
    assert_eq!((cm0, cm1), (1, 1));
    crate::apple_gpu::slice_mut::<u16>(&model.hnet, 3 * d * d + d)?.copy_from_slice(orig_hnet);
    Ok(())
}

#[test]
fn hnet2_causality() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let flat = model.flatten(&runtime)?;
        let weights = model.row(&flat)?;
        let decoder = decode::Decoder::with_visits(&runtime, CONTEXT, false, 1)?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;
        let hnet2 = scorer.buffers.hnet2()?;

        let (n, split) = (512usize, 256usize);
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 73 + 19) % 256) as u8).collect();
        let mut batch = hnet_batch(&bytes);
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let probs0_a = buffer_slice::<f32>(&hnet2.probs0, split)?.to_vec();
        let plug0_a = buffer_slice::<u32>(&hnet2.plug0, split)?.to_vec();
        let c0_split = plug0_a[split - 1] as usize + 1;
        let probs1_a = buffer_slice::<f32>(&hnet2.probs1, c0_split)?.to_vec();
        let plug1_a = buffer_slice::<u32>(&hnet2.plug1, c0_split)?.to_vec();
        let loss_a = buffer_slice::<f32>(&scorer.buffers.losses, split)?.to_vec();

        for tok in &mut batch.tokens[split..] {
            *tok = (*tok + 113) % 256;
        }
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        assert_eq!(plug0_a, buffer_slice::<u32>(&hnet2.plug0, split)?);
        assert_eq!(plug1_a, buffer_slice::<u32>(&hnet2.plug1, c0_split)?);
        let probs0_b = buffer_slice::<f32>(&hnet2.probs0, split)?;
        let probs1_b = buffer_slice::<f32>(&hnet2.probs1, c0_split)?;
        let loss_b = buffer_slice::<f32>(&scorer.buffers.losses, split)?;
        for i in 0..split {
            assert!((probs0_a[i] - probs0_b[i]).abs() < 1e-6);
            assert!((loss_a[i] - loss_b[i]).abs() < 1e-4);
        }
        for c in 0..c0_split {
            assert!((probs1_a[c] - probs1_b[c]).abs() < 1e-6);
        }
        Ok(())
    })
}

#[test]
fn hnet2_interior() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let d = WIDTH as usize;
        let flat = model.flatten(&runtime)?;
        let weights = model.row(&flat)?;
        let decoder = decode::Decoder::with_visits(&runtime, CONTEXT, false, 1)?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;
        let hnet2 = scorer.buffers.hnet2()?;

        let n = 1024usize;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 29 + 7) % 251) as u8).collect();
        let mut batch = hnet_batch(&bytes);
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let (m0, _) = hnet2.counts()?;
        let origins0 = buffer_slice::<u32>(&hnet2.origins0, m0 as usize)?.to_vec();
        let (c0_idx, int_t) = (0..(m0 as usize).saturating_sub(1))
            .find_map(|c| {
                let (s, e) = (origins0[c] as usize, origins0[c + 1] as usize);
                (e - s >= 2).then_some((c, s + 1))
            })
            .ok_or("expected multi-byte Stage-0 chunk")?;
        let h1_before = buffer_slice::<f32>(&hnet2.tile_z0, (c0_idx + 2) * d)?
            [(c0_idx + 1) * d..(c0_idx + 2) * d]
            .to_vec();
        let loss_before = buffer_slice::<f32>(&scorer.buffers.losses, n)?.to_vec();

        batch.tokens[int_t] = (batch.tokens[int_t] + 97) % 251;
        scorer.score_batch(&runtime, &decoder, weights, &batch)?;
        let h1_after = &buffer_slice::<f32>(&hnet2.tile_z0, (c0_idx + 2) * d)?
            [(c0_idx + 1) * d..(c0_idx + 2) * d];
        let loss_after = buffer_slice::<f32>(&scorer.buffers.losses, n)?;
        assert_ne!(
            h1_before, h1_after,
            "interior byte must change Stage-0/1 state"
        );
        assert!((int_t..n).any(|t| (loss_before[t] - loss_after[t]).abs() > 1e-4));
        for t in 0..int_t {
            assert!((loss_before[t] - loss_after[t]).abs() < 1e-4);
        }
        check_s1(&runtime, &scorer, &decoder, weights, &bytes)
    })
}

fn check_s1(
    runtime: &Runtime,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    weights: CandidateRow<'_>,
    bytes: &[u8],
) -> Result<(), String> {
    let d = WIDTH as usize;
    let mut batch = hnet_batch(bytes);
    scorer.score_batch(runtime, decoder, weights, &batch)?;
    let hnet2 = scorer.buffers.hnet2()?;
    let (m0, m1) = hnet2.counts()?;
    let origins0 = buffer_slice::<u32>(&hnet2.origins0, m0 as usize)?.to_vec();
    let origins1 = buffer_slice::<u32>(&hnet2.origins1, m1 as usize)?.to_vec();
    let (c1_idx, int_c0) = (0..(m1 as usize).saturating_sub(1))
        .find_map(|c1| {
            let (s, e) = (origins1[c1] as usize, origins1[c1 + 1] as usize);
            (e - s >= 2).then_some((c1, s + 1))
        })
        .ok_or("expected multi-chunk Stage-1 super-chunk")?;
    let byte_t = origins0[int_c0] as usize;
    let s1_before = buffer_slice::<u16>(&hnet2.chunks_h1, (c1_idx + 2) * d)?
        [(c1_idx + 1) * d..(c1_idx + 2) * d]
        .to_vec();
    batch.tokens[byte_t] = (batch.tokens[byte_t] + 83) % 251;
    scorer.score_batch(runtime, decoder, weights, &batch)?;
    let s1_after = &buffer_slice::<u16>(&hnet2.chunks_h1, (c1_idx + 2) * d)?
        [(c1_idx + 1) * d..(c1_idx + 2) * d];
    assert_ne!(
        s1_before, s1_after,
        "interior Stage-0 chunk must change Stage-1 state"
    );
    Ok(())
}

#[test]
fn hnet2_bo() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let d = WIDTH as usize;
        let orig_hnet = buffer_slice::<u16>(&model.hnet, 3 * d * d + d)?.to_vec();
        let decoder = decode::Decoder::with_visits(&runtime, CONTEXT, false, 1)?;
        let scorer = BlockDecoder::with_architecture(&runtime, CONTEXT, arch)?;
        let hnet2 = scorer.buffers.hnet2()?;
        let n = CONTEXT as usize;
        let bytes: Vec<u8> = (0..=n).map(|i| ((i * 41 + 13) % 256) as u8).collect();
        let batch = hnet_batch(&bytes);

        let st_bal = {
            let flat = model.flatten(&runtime)?;
            scorer.score_batch(&runtime, &decoder, model.row(&flat)?, &batch)?;
            hnet2.last_stats.get()
        };
        assert!(
            (768..=1344).contains(&st_bal.stage0_chunks),
            "bal m0={} m1={}",
            st_bal.stage0_chunks,
            st_bal.stage1_chunks
        );
        assert!(
            (350..=700).contains(&st_bal.stage1_chunks),
            "bal m0={} m1={}",
            st_bal.stage0_chunks,
            st_bal.stage1_chunks
        );

        router_mode(&model, &orig_hnet, None, 0)?;
        let st_uncal = {
            let flat = model.flatten(&runtime)?;
            scorer.score_batch(&runtime, &decoder, model.row(&flat)?, &batch)?;
            hnet2.last_stats.get()
        };
        assert!(st_uncal.total_penalty > st_bal.total_penalty);

        router_mode(&model, &orig_hnet, None, -1)?;
        let st_open = {
            let flat = model.flatten(&runtime)?;
            scorer.score_batch(&runtime, &decoder, model.row(&flat)?, &batch)?;
            hnet2.last_stats.get()
        };
        assert_eq!((st_open.stage0_chunks, st_open.stage1_chunks), (4096, 4096));
        assert!(st_open.total_penalty > st_bal.total_penalty * 2.0);

        router_mode(&model, &orig_hnet, None, 1)?;
        let st_closed = {
            let flat = model.flatten(&runtime)?;
            scorer.score_batch(&runtime, &decoder, model.row(&flat)?, &batch)?;
            hnet2.last_stats.get()
        };
        assert_eq!((st_closed.stage0_chunks, st_closed.stage1_chunks), (1, 1));
        assert!(st_closed.total_penalty > st_bal.total_penalty);

        crate::apple_gpu::slice_mut::<u16>(&model.hnet, 3 * d * d + d)?.copy_from_slice(&orig_hnet);
        bo_step(&runtime, model, &scorer, &decoder, &batch)
    })
}

fn bo_step(
    runtime: &Runtime,
    model: CandidateWeights,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    batch: &crate::pretrain_data::CausalBatch,
) -> Result<(), String> {
    let init_score = {
        let flat = model.flatten(runtime)?;
        scorer.score_batch(runtime, decoder, model.row(&flat)?, batch)?
    };
    let run = crate::config::parse_tune(
        "version = 4\nexperiment = \"pretrain\"\nmodel = \"fbt-pisa1-hnet-mhc4-v1\"\n\
         recipe = \"fineweb-bytes-100m-v1\"\nobjective = \"cross-entropy\"\n\
         window-tokens = 1048576\nrounds = 4\nround-target-ms = 680\n\
         [optimizer]\nalgorithm = \"operator-bo\"\n\
         distribution = \"gaussian\"\n\
         [optimizer.surrogate]\nneighbors = 8\nepistemic-scale = 0.1\n\
         aleatoric-scale = 0.05\ny-scale = 0.005\n\
         [optimizer.trust-region]\nshape = \"tensor-family-static\"\n\
         initial = 0.004\nmin = 0.001\nmax = 0.008\n",
    )?;
    let layout = Mhc4Layout::from_weights(&model, 0);
    let (mut search, _) = model.search_shaped(
        run.proposal_kernel(),
        run.length(),
        crate::config::TrustRegionShape::TensorFamilyStatic,
    )?;
    drop(model);
    let policy = run.resident_enn(run.acquisition_seed())?;
    search.configure_enn(policy)?;
    search.observe_objectives(init_score.observation)?;
    let mut surrogate = baux::semantics::optimizer::Surrogate::with_spec(
        -init_score.quality.mean_nll,
        0.0,
        run.acquisition_seed(),
        policy.surrogate_spec()?,
    )?;
    let before = buffer_slice::<u16>(&search.base_buffer(), 1024)?.to_vec();
    let geom = search.sample_distances(101)?;
    assert_eq!(geom.len(), 1);
    let sel = surrogate.select_pool(
        &geom,
        0,
        baux::semantics::optimizer::AcquisitionRule::Ucb { beta: 0.0 },
        None,
    )?;
    let (cand_buf, proposal) = search.materialize_selection(101, &sel)?;
    let cand_score = scorer.score_batch(runtime, decoder, layout.view(&cand_buf), batch)?;
    let slot = surrogate.append(&geom[0], -cand_score.quality.mean_nll, 0.0, true)?;
    search.commit_host(&proposal, slot, cand_score.observation, true, 0.004)?;
    assert_eq!(search.sync()?, vec![true]);
    let base_buf = search.base_buffer();
    let after = buffer_slice::<u16>(&base_buf, 1024)?;
    assert_ne!(before.as_slice(), after);
    Ok(())
}

#[test]
#[ignore]
fn hnet2_1m() -> Result<(), String> {
    autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        let arch = ResidualArchitecture::HnetMhc4;
        let ctx: u32 = 1_048_576;
        let model = CandidateWeights::initialized(
            &runtime,
            17,
            crate::config::ModelInitialization::FbtPisa1Mhc4,
            arch,
        )?;
        let decoder = decode::Decoder::with_visits(&runtime, CONTEXT, false, 1)?;
        let scorer = BlockDecoder::with_architecture(&runtime, ctx, arch)?;
        let _ = scorer.buffers.hnet2()?;
        let bytes: Vec<u8> = (0..=ctx as usize)
            .map(|i| ((i.wrapping_mul(41).wrapping_add(13)) % 256) as u8)
            .collect();
        let batch = hnet_batch(&bytes);
        bench_1m(&runtime, model, &scorer, &decoder, &batch)
    })
}

fn bench_1m(
    runtime: &Runtime,
    model: CandidateWeights,
    scorer: &BlockDecoder,
    decoder: &decode::Decoder,
    batch: &crate::pretrain_data::CausalBatch,
) -> Result<(), String> {
    let run = crate::config::parse_tune(
        "version = 4\nexperiment = \"pretrain\"\nmodel = \"fbt-pisa1-hnet-mhc4-v1\"\n\
         recipe = \"fineweb-bytes-100m-v1\"\nobjective = \"cross-entropy\"\n\
         window-tokens = 1048576\nrounds = 5\nround-target-ms = 680\n\
         [optimizer]\nalgorithm = \"operator-bo\"\n\
         distribution = \"gaussian\"\n\
         [optimizer.surrogate]\nneighbors = 8\nepistemic-scale = 0.1\n\
         aleatoric-scale = 0.05\ny-scale = 0.005\n\
         [optimizer.trust-region]\nshape = \"tensor-family-static\"\n\
         initial = 0.004\nmin = 0.001\nmax = 0.008\n",
    )?;
    let layout = Mhc4Layout::from_weights(&model, 0);
    let (mut search, _) = model.search_shaped(
        run.proposal_kernel(),
        run.length(),
        crate::config::TrustRegionShape::TensorFamilyStatic,
    )?;
    drop(model);
    let policy = run.resident_enn(run.acquisition_seed())?;
    search.configure_enn(policy)?;
    let init = scorer.score_batch(runtime, decoder, layout.view(&search.base_buffer()), batch)?;
    search.observe_objectives(init.observation)?;
    let mut surrogate = baux::semantics::optimizer::Surrogate::with_spec(
        -init.quality.mean_nll,
        0.0,
        run.acquisition_seed(),
        policy.surrogate_spec()?,
    )?;
    for step in 0..=5u64 {
        let wall_t0 = Instant::now();
        let ask_t0 = Instant::now();
        let geom = search.sample_distances(200 + step)?;
        let sel = surrogate.select_pool(
            &geom,
            0,
            baux::semantics::optimizer::AcquisitionRule::Ucb { beta: 0.0 },
            None,
        )?;
        let (cand_buf, proposal) = search.materialize_selection(200 + step, &sel)?;
        let ask_ms = ask_t0.elapsed().as_secs_f64() * 1000.0;
        let cand = scorer.score_batch(runtime, decoder, layout.view(&cand_buf), batch)?;
        let tell_t0 = Instant::now();
        let slot = surrogate.append(&geom[0], -cand.quality.mean_nll, 0.0, step == 0)?;
        search.commit_host(&proposal, slot, cand.observation, step == 0, 0.004)?;
        let _ = search.sync()?;
        let tell_ms = tell_t0.elapsed().as_secs_f64() * 1000.0;
        let wall_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
        log_1m(scorer, step, ask_ms, tell_ms, wall_ms);
    }
    Ok(())
}

fn log_1m(scorer: &BlockDecoder, step: u64, ask_ms: f64, tell_ms: f64, wall_ms: f64) {
    let runtime = Runtime::shared().unwrap();
    let st = scorer.buffers.hnet2().unwrap().last_stats.get();
    let alloc_mib = (runtime.device.current_allocated_size() as f64) / (1024.0 * 1024.0);
    eprintln!(
        "HNET2_1M_BENCH step={step} bytes={} m0={} m1={} aligned_m1={} f0={:.4} f1={:.4} pen={:.6} prep_ms={:.2} core_ms={:.2} rev_ms={:.2} ask_ms={:.2} tell_ms={:.2} wall_ms={:.2} alloc_mib={:.1}",
        st.total_bytes,
        st.stage0_chunks,
        st.stage1_chunks,
        st.aligned_m1,
        st.f0,
        st.f1,
        st.total_penalty,
        st.prep_seconds * 1000.0,
        st.core_seconds * 1000.0,
        st.rev_seconds * 1000.0,
        ask_ms,
        tell_ms,
        wall_ms,
        alloc_mib
    );
    assert!((180_000..=350_000).contains(&st.stage0_chunks));
    assert!((80_000..=185_000).contains(&st.stage1_chunks));
    assert!(
        alloc_mib <= 6656.0,
        "alloc_mib {alloc_mib:.1} exceeds 6656 MiB"
    );
    if step > 0 {
        assert!(
            wall_ms < 1000.0,
            "1M round wall_ms {wall_ms:.2} exceeds 1000 ms"
        );
    }
}
