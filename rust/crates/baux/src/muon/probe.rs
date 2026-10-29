//! Optimizer-only timings. Analytic model backward and generation are excluded.
use super::{
    Shape, Step,
    metal::{Muon, Workspace},
};
use crate::apple_gpu::{Runtime, gpu_seconds};
use baux::artifact::json::{Value, json};
use metal::MTLCommandBufferStatus;
use std::time::Instant;

pub fn measure() -> Result<Value, String> {
    let runtime = Runtime::shared()?;
    let mut records = Vec::new();
    for (rows, cols, batch) in [(32, 48, 2), (512, 640, 5), (512, 432, 626)] {
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
    Ok(json!({"reference":"fp64","tolerance":0.001,"passed":true,"records":records}))
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
