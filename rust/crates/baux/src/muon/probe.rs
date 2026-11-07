//! Optimizer-only timings. Analytic model backward and generation are excluded.
use super::{
    Shape, Step,
    metal::{Muon, Workspace},
};
use crate::apple_gpu::{Runtime, gpu_seconds};
use baux::artifact::json::{Value, json};
use metal::MTLCommandBufferStatus;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn measure() -> Result<Value, String> {
    let runtime = Runtime::shared()?;
    let mut records = Vec::new();
    for (rows, cols, batch) in [(32, 48, 2), (512, 640, 5), (512, 512, 65)] {
        let shape = Shape::new(rows, cols, batch)?;
        let count = shape.elements();
        let weights = runtime.buffer_with(&vec![0x3800u16; count]);
        let gradient = runtime.buffer::<f32>(count);
        let mut optimizer = Muon::new(shape, &weights, 0)?;
        let mut workspace = Workspace::new(shape)?;
        for repeat in 0..4 {
            // A fixed arithmetic fixture, not a model gradient or training
            // result.
            let values =
                unsafe { std::slice::from_raw_parts_mut(gradient.contents().cast::<f32>(), count) };
            for (index, value) in values.iter_mut().enumerate() {
                *value = ((index * 17 + index / 7) % 127) as f32 / 127.0 - 0.5;
            }
            let command = runtime.queue.new_command_buffer();
            let start = Instant::now();
            optimizer.encode(
                &mut workspace,
                command,
                &weights,
                0,
                &gradient,
                0,
                Step::default(),
            )?;
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err("Muon probe GPU execution failed".into());
            }
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let values =
                unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
            let invalid = values
                .iter()
                .filter(|value| **value & 0x7c00 == 0x7c00)
                .count();
            records.push(json!({
                "rows": rows,
                "cols": cols,
                "batch": batch,
                "repeat": repeat,
                "warmup": repeat == 0,
                "encode-ms": encode_ms,
                "wall-ms": wall_ms,
                "gpu-ms": gpu_seconds(command).map(|s| s * 1000.0),
                "state-bytes": optimizer.state_bytes(),
                "workspace-bytes": workspace.bytes(),
                "invalid-weights": invalid,
            }));
        }
    }
    Ok(json!({
        "schema": "baux.muon-kernels.v1",
        "hardware": runtime.device.name(),
        "variant": "polar-express-5",
        "full-training-loop": false,
        "gradient": "fixed-arithmetic-fixture",
        "records": records,
        "parity": verify()?,
    }))
}

pub fn verify() -> Result<Value, String> {
    use super::reference;
    let runtime = Runtime::shared()?;
    let mut records = Vec::new();
    for (rows, cols) in [(3, 5), (5, 3), (8, 8)] {
        let shape = Shape::new(rows, cols, 2)?;
        let count = shape.elements();
        let weights = runtime.buffer_with(&vec![0x3800u16; count]);
        let mut workspace = Workspace::new(shape)?;
        let mut optimizer = Muon::new(shape, &weights, 0)?;
        let mut expected = vec![0.5f64; count];
        let mut momentum = vec![0.0f64; count];
        for round in 0..4 {
            let fixture = reference::fixture(shape, round);
            let gradient = runtime.buffer_with(&fixture);
            let step = Step {
                decay: 0.01,
                ..Step::default()
            };
            reference::muon(shape, &mut expected, &mut momentum, &fixture, step);
            let command = runtime.queue.new_command_buffer();
            optimizer.encode(&mut workspace, command, &weights, 0, &gradient, 0, step)?;
            command.commit();
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err("Muon parity GPU command failed".into());
            }
            let actual =
                unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
            records.push(json!({"optimizer":"muon","rows":rows,"cols":cols,"round":round,"max-abs-error":reference::error(actual,&expected)?}));
        }
    }
    verify_adam(&runtime, &mut records)?;
    check_locks(&repo_root())?;
    Ok(json!({"reference":"fp64","tolerance":0.001,"passed":true,"records":records}))
}

fn repo_root() -> PathBuf {
    let mut cur = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    while !cur.join("research/upstream").exists() {
        if let Some(parent) = cur.parent() {
            cur = parent.to_path_buf();
        } else {
            return PathBuf::from(".");
        }
    }
    cur
}

fn check_locks(root: &Path) -> Result<(), String> {
    let muon_dir = root.join(".cache/baux/upstream/Muon");
    let polar_dir = root.join(".cache/baux/upstream/PolarExpress");
    let mhc_dir = root.join(".cache/baux/upstream/mHC");
    if muon_dir.exists() && polar_dir.exists() && mhc_dir.exists() {
        crate::artifact::merkle::verify_lock(root, "muon", None)?;
        crate::artifact::merkle::verify_lock(root, "polar_express", None)?;
        crate::artifact::merkle::verify_lock(root, "mhc", None)?;
        crate::artifact::merkle::verify_lock(root, "acme", None)?;
    }
    Ok(())
}

#[test]
fn upstream_locks() -> Result<(), String> {
    let root = repo_root();
    check_locks(&root)
}

fn verify_adam(runtime: &Runtime, records: &mut Vec<Value>) -> Result<(), String> {
    use super::{metal::Adam, reference};
    let shape = Shape::new(4, 8, 2)?;
    let count = shape.elements();
    let weights = runtime.buffer_with(&vec![0x3800u16; count]);
    let mut optimizer = Adam::new(&weights, 0, count as u32)?;
    let mut expected = vec![0.5f64; count];
    let mut first = vec![0.0f64; count];
    let mut second = vec![0.0f64; count];
    for round in 0..4 {
        let fixture = reference::fixture(shape, round);
        let gradient = runtime.buffer_with(&fixture);
        let step = Step {
            rate: 0.001,
            momentum: 0.9,
            decay: 0.01,
            ..Step::default()
        };
        reference::adam(
            &mut expected,
            &mut first,
            &mut second,
            &fixture,
            step.adam(round as u32 + 1)?,
        );
        let command = runtime.queue.new_command_buffer();
        optimizer.encode(command, &weights, 0, &gradient, 0, step)?;
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("AdamW parity GPU command failed".into());
        }
        let actual = unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
        records.push(json!({"optimizer":"adamw","round":round,"max-abs-error":reference::error(actual,&expected)?}));
    }
    Ok(())
}

#[cfg(test)]
fn floats(value: &Value) -> Vec<f64> {
    value
        .as_seq()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

#[cfg(test)]
fn check_remez(remez: &Value) {
    use super::POLAR;
    let raw = remez["paper_raw_coefficients"].as_seq().unwrap();
    let polar = remez["paper_polar_coefficients"].as_seq().unwrap();
    let repo_raw = remez["repo_raw_coefficients"].as_seq().unwrap();
    let repo = remez["repo_coefficients"].as_seq().unwrap();
    assert_eq!(raw.len(), POLAR.len());
    assert_eq!(polar.len(), POLAR.len());
    let s1 = 1.01_f32;
    let s3 = s1 * s1 * s1;
    let s5 = s3 * s1 * s1;
    for (idx, coeff) in POLAR.iter().enumerate() {
        let r = floats(&raw[idx]);
        let p = floats(&polar[idx]);
        assert!((coeff.a - (r[0] as f32 / s1)).abs() < 1e-6);
        assert!((coeff.b - (r[1] as f32 / s3)).abs() < 1e-6);
        assert!((coeff.c - (r[2] as f32 / s5)).abs() < 1e-6);
        assert!((f64::from(coeff.a) - p[0]).abs() < 1e-6);
        assert!((f64::from(coeff.b) - p[1]).abs() < 1e-6);
        assert!((f64::from(coeff.c) - p[2]).abs() < 1e-6);
        let rr = floats(&repo_raw[idx]);
        let rc = floats(&repo[idx]);
        let factor = if idx + 1 < POLAR.len() { 1.01 } else { 1.0 };
        assert!((rc[0] - rr[0] / factor).abs() < 1e-12);
        assert!((rc[1] - rr[1] / factor.powi(3)).abs() < 1e-12);
        assert!((rc[2] - rr[2] / factor.powi(5)).abs() < 1e-12);
    }
}

#[cfg(test)]
fn check_case(runtime: &Runtime, case: &Value) -> Result<(), String> {
    use super::reference;
    let rows = case["rows"].as_u64().unwrap() as u32;
    let cols = case["cols"].as_u64().unwrap() as u32;
    let batch = case["batch"].as_u64().unwrap() as u32;
    let shape = Shape::new(rows, cols, batch)?;
    let count = shape.elements();
    let weights = runtime.buffer_with(&vec![0x3800u16; count]);
    let mut workspace = Workspace::new(shape)?;
    let mut optimizer = Muon::new(shape, &weights, 0)?;
    let mut expected = vec![0.5f64; count];
    let mut momentum = vec![0.0f64; count];
    let step = Step {
        rate: case["rate"].as_f64().unwrap() as f32,
        momentum: case["momentum"].as_f64().unwrap() as f32,
        decay: case["decay"].as_f64().unwrap() as f32,
        epsilon: case["epsilon"].as_f64().unwrap() as f32,
        ..Step::default()
    };
    for round in case["rounds"].as_seq().unwrap() {
        let grad: Vec<f32> = floats(&round["gradient"])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let target = floats(&round["weights"]);
        let target_mom = floats(&round["momentum"]);
        let traj = round["trajectory"].as_seq().unwrap();
        assert_eq!(traj.len(), 5);
        let step5 = floats(&traj[4]);
        let fp64 = floats(&round["polar_fp64"]);
        let bf16 = floats(&round["polar_bf16"]);
        for (&s5, &f64v) in step5.iter().zip(&fp64) {
            assert!((s5 - f64v).abs() < 1e-4);
        }
        for (&s5, &b16) in step5.iter().zip(&bf16) {
            assert!((s5 - b16).abs() < 0.15);
        }
        reference::muon(shape, &mut expected, &mut momentum, &grad, step);
        for (&a, &e) in expected.iter().zip(&target) {
            assert!((a - e).abs() < 1e-6, "muon weight diff: {a} vs {e}");
        }
        for (&a, &e) in momentum.iter().zip(&target_mom) {
            assert!((a - e).abs() < 1e-6, "muon momentum diff: {a} vs {e}");
        }
        let grad_buf = runtime.buffer_with(&grad);
        let command = runtime.queue.new_command_buffer();
        optimizer.encode(&mut workspace, command, &weights, 0, &grad_buf, 0, step)?;
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("upstream Muon parity GPU command failed".into());
        }
        let actual = unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
        reference::error(actual, &target)?;
    }
    Ok(())
}

#[cfg(test)]
fn check_adam(runtime: &Runtime, case: &Value) -> Result<(), String> {
    use super::{metal::Adam, reference};
    let rows = case["rows"].as_u64().unwrap() as u32;
    let cols = case["cols"].as_u64().unwrap() as u32;
    let batch = case["batch"].as_u64().unwrap() as u32;
    let shape = Shape::new(rows, cols, batch)?;
    let count = shape.elements();
    let weights = runtime.buffer_with(&vec![0x3800u16; count]);
    let mut optimizer = Adam::new(&weights, 0, count as u32)?;
    let mut expected = vec![0.5f64; count];
    let mut first = vec![0.0f64; count];
    let mut second = vec![0.0f64; count];
    let step = Step {
        rate: case["rate"].as_f64().unwrap() as f32,
        momentum: case["momentum"].as_f64().unwrap() as f32,
        beta2: case["beta2"].as_f64().unwrap() as f32,
        decay: case["decay"].as_f64().unwrap() as f32,
        epsilon: case["epsilon"].as_f64().unwrap() as f32,
        ..Step::default()
    };
    for (idx, round) in case["rounds"].as_seq().unwrap().iter().enumerate() {
        let grad: Vec<f32> = floats(&round["gradient"])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let target = floats(&round["weights"]);
        reference::adam(
            &mut expected,
            &mut first,
            &mut second,
            &grad,
            step.adam(idx as u32 + 1)?,
        );
        for (&a, &e) in expected.iter().zip(&target) {
            assert!((a - e).abs() < 1e-6, "adam weight diff: {a} vs {e}");
        }
        let grad_buf = runtime.buffer_with(&grad);
        let command = runtime.queue.new_command_buffer();
        optimizer.encode(command, &weights, 0, &grad_buf, 0, step)?;
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("upstream AdamW parity GPU command failed".into());
        }
        let actual = unsafe { std::slice::from_raw_parts(weights.contents().cast::<u16>(), count) };
        reference::error(actual, &target)?;
    }
    Ok(())
}

/// Differential parity against pinned upstream `PolarExpress`
/// (`optimal_quintic`, `optimal_composition`, `PolarExpress`) and `Muon`
/// (`muon_update`, `adam_update`).
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
#[test]
fn upstream_polar() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let data: Value = baux::artifact::json::from_str(include_str!("fixtures.json")).unwrap();
        assert_eq!(
            data["polar_revision"].as_str(),
            Some("f717e2c6b3ef6f67a5e00ab7f5371a85377e6fbe")
        );
        assert_eq!(
            data["muon_revision"].as_str(),
            Some("f98f1cacc0263b04290753e32be8d498c1efc806")
        );
        check_remez(&data["remez"]);
        let runtime = Runtime::shared()?;
        for case in data["cases"].as_seq().unwrap() {
            check_case(&runtime, case)?;
        }
        check_adam(&runtime, &data["adam"])?;
        Ok(())
    })
}
