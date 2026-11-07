//! Differential fixtures executed from pinned upstream Python definitions.
//!
//! # Academic & Open-Source Attribution
//! - **Paper**: Sukjun Hwang, Brandon Wang, and Albert Gu, *"Dynamic Chunking
//!   for End-to-End Hierarchical Sequence Modeling"*, arXiv:2507.07955 (2025).
//! - **Upstream Repository**: Goomba Lab / Cartesia AI, <https://github.com/goombalab/hnet>
//!   (commit `3673fe1217ebeb0d1438c7c71d58d32bdd190ec2`, files
//!   `hnet/modules/dc.py`, `hnet/models/hnet.py`, `hnet/utils/tokenizers.py`,
//!   `hnet/utils/train.py`).
use super::*;
use baux::artifact::toml::{self, Value};

pub(super) fn fixtures() -> Value {
    let root = option_env!("CARGO_MANIFEST_DIR")
        .map(std::path::Path::new)
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .unwrap_or(std::path::Path::new("."));
    if root.join(".cache/baux/upstream/hnet").exists() {
        crate::artifact::merkle::verify_lock(&root, "hnet", None).expect("verify hnet.lock.toml");
    }
    let value: Value = toml::from_str(include_str!("fixtures.toml")).unwrap();
    assert_eq!(
        value["revision"].as_str(),
        Some("3673fe1217ebeb0d1438c7c71d58d32bdd190ec2")
    );
    assert_eq!(value["full_model_parity"].as_bool(), Some(false));
    value
}

pub(super) fn floats(value: &Value) -> Vec<f32> {
    value
        .as_seq()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn bools(value: &Value) -> Vec<bool> {
    value
        .as_seq()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect()
}

pub(super) fn size(value: &Value) -> usize {
    value.as_u64().unwrap() as usize
}

pub(super) fn close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-6 * (1.0 + e.abs()),
            "{label}[{index}]: actual={a} upstream={e}"
        );
    }
}

fn check_routing(waves: &HNetWaves, case: &Value) -> Result<(), String> {
    let (batch, length, width) = (
        size(&case["batch"]),
        size(&case["length"]),
        size(&case["width"]),
    );
    let router = RoutingModule::with_weights(width, floats(&case["q"]), floats(&case["k"]))?;
    let hidden = floats(&case["hidden"]);
    let mask = bools(&case["mask"]);
    let expected = bools(&case["boundary"]);
    let route = router.forward(&hidden, batch, length, Some(&mask), None)?;
    assert_eq!(
        route.boundary_mask, expected,
        "CPU boundary: {:?}",
        case["name"]
    );
    close(
        "CPU probability",
        &route.boundary_prob,
        &floats(&case["probability"]),
    );
    close(
        "CPU selected",
        &route.selected_probs,
        &floats(&case["selected"]),
    );
    let chunks = ChunkLayer::new().forward(&hidden, &route.boundary_mask, batch, length, width)?;
    close("CPU chunks", &chunks.chunks, &floats(&case["chunks"]));
    assert_eq!(chunks.next_mask, bools(&case["chunk_mask"]));
    let gpu = waves.run_stage(
        &router,
        &SteResidual::new(width),
        &hidden,
        &mask,
        batch,
        length,
    )?;
    assert_eq!(
        gpu.boundary_mask, expected,
        "GPU boundary: {:?}",
        case["name"]
    );
    close(
        "GPU probability",
        &gpu.boundary_prob,
        &floats(&case["probability"]),
    );
    close(
        "GPU selected",
        &gpu.selected_probs,
        &floats(&case["selected"]),
    );
    let expected_counts: Vec<_> = case["chunk_count"]
        .as_seq()
        .unwrap()
        .iter()
        .map(size)
        .collect();
    assert_eq!(gpu.num_chunks, expected_counts);
    let maximum = *expected_counts.iter().max().unwrap();
    let expected_chunks = floats(&case["chunks"]);
    for b in 0..batch {
        close(
            "GPU chunks",
            &gpu.chunks[b * length * width..b * length * width + maximum * width],
            &expected_chunks[b * maximum * width..(b + 1) * maximum * width],
        );
    }
    Ok(())
}

#[test]
fn routing() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let data = fixtures();
        let waves = HNetWaves::new()?;
        for case in data["routing"].as_seq().unwrap() {
            check_routing(&waves, case)?;
        }
        let router = RoutingModule::new(4);
        let mut state = router.allocate_state(1);
        for step in data["recurrent"].as_seq().unwrap() {
            let actual = router.step(&floats(&step["hidden"]), &mut state)?;
            close(
                "recurrent probability",
                &actual.boundary_prob,
                &floats(&step["probability"]),
            );
            close(
                "recurrent selected",
                &actual.selected_probs,
                &floats(&step["selected"]),
            );
            assert_eq!(actual.boundary_mask, bools(&step["boundary"]));
            assert_eq!(state.has_seen, bools(&step["seen"]));
            close(
                "recurrent state",
                &state.last_hidden,
                &floats(&step["last_hidden"]),
            );
        }
        Ok(())
    })
}

#[test]
fn ste() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let data = fixtures();
        let case = &data["ste"];
        let gradient = floats(&case["gradient"]);
        let dechunk = floats(&case["dechunk"]);
        let probability = floats(&case["probability"]);
        let expected = floats(&case["expected"]);
        let cpu = SteResidual::new(4).backward_prob(&gradient, &dechunk, &probability)?;
        let gpu = HNetWaves::new()?.backward_ste(&gradient, &dechunk, &probability, 1, 4, 4)?;
        close("CPU STE", &cpu, &expected);
        close("GPU STE", &gpu, &expected);
        Ok(())
    })
}

#[test]
fn tokenizer() {
    let data = fixtures();
    let tok = ByteTokenizer::new();
    for case in data["tokenizer"].as_seq().unwrap() {
        let text = case["text"].as_str().unwrap();
        let bos = case["add_bos"].as_bool().unwrap();
        let eos = case["add_eos"].as_bool().unwrap();
        let expected: Vec<u8> = case["tokens"]
            .as_seq()
            .unwrap()
            .iter()
            .map(|v| size(v) as u8)
            .collect();
        let encoded = tok.encode(text, bos, eos);
        assert_eq!(encoded, expected);
        if !bos && !eos {
            assert_eq!(tok.decode(&encoded).unwrap(), text);
        }
    }
}

#[test]
fn ratio() -> Result<(), String> {
    let data = fixtures();
    for case in data["ratio_loss"].as_seq().unwrap() {
        let probability = floats(&case["probability"]);
        let boundary = bools(&case["boundary"]);
        let selected = floats(&case["selected"]);
        let target = case["target_ratio"].as_f64().unwrap() as f32;
        let expected = case["loss"].as_f64().unwrap() as f32;
        let length = probability.len();
        let output = RoutingOutput {
            boundary_prob: probability,
            boundary_mask: boundary,
            selected_probs: selected,
            batch_size: 1,
            seq_len: length,
        };
        let actual = ratio_loss(&output, None, target)?;
        close("ratio loss", &[actual], &[expected]);
    }
    Ok(())
}
