//! Differential fixtures executed from pinned upstream Python definitions.
use super::*;
use baux::artifact::json::{self, Value};

pub(super) fn fixtures() -> Value {
    let value: Value = json::from_str(include_str!("fixtures.json")).unwrap();
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
