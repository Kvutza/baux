//! Selected-logit derivatives, with identity adjoints at half casts.
use super::*;
use crate::apple_gpu::{bind_buffers, buffer_slice, encode_value};

fn probabilities(logits: &[f64; 3]) -> [f64; 3] {
    let sigmoid = logits.map(|x| (1.0 / (1.0 + (-x).exp())).max(1e-20));
    let total = sigmoid.iter().sum::<f64>();
    sigmoid.map(|x| x / total)
}

fn qualify(runtime: &Runtime, logits: [f64; 3], upstream: [f32; 3]) -> Result<(), String> {
    let experts = routing::ROUTED_EXPERTS as usize;
    let mut scores = vec![resources::half_bits(-64.0); experts];
    for (i, value) in logits.iter().enumerate() {
        scores[i] = resources::half_bits(*value as f32);
    }
    let score_buffer = runtime.buffer_with(&scores);
    let selected = runtime.buffer::<u32>(3);
    let weights = runtime.buffer::<u16>(3);
    let margin = runtime.buffer::<f32>(1);
    let shape = routing::Top3RouteShape::new(1, WIDTH);
    let forward = runtime.precise(
        include_str!("../fbt_routing.metal"),
        "router reference",
        "fbt_moe_select_top3",
    )?;
    let backward = runtime.precise(
        include_str!("../fbt_routing_grad.metal"),
        "router adjoint",
        "moe_router_score_grad",
    )?;
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&forward);
    bind_buffers(
        &encoder,
        &[
            (&score_buffer, 0),
            (&selected, 0),
            (&weights, 0),
            (&margin, 0),
        ],
    );
    encode_value(&encoder, 4, &shape);
    encoder.dispatch_thread_groups(thread_group(1), thread_group(32));
    encoder.end_encoding();
    complete(command)?;
    let ids = buffer_slice::<u32>(&selected, 3)?;
    let ordered: [f64; 3] = std::array::from_fn(|i| logits[ids[i] as usize]);
    for (&actual, expected) in buffer_slice::<u16>(&weights, 3)?
        .iter()
        .zip(probabilities(&ordered))
    {
        assert_eq!(actual, resources::half_bits(expected as f32));
    }
    let upstream_buffer = runtime.buffer_with(&upstream);
    let gradient = runtime.buffer::<f32>(3);
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&backward);
    bind_buffers(
        &encoder,
        &[
            (&score_buffer, 0),
            (&selected, 0),
            (&upstream_buffer, 0),
            (&gradient, 0),
        ],
    );
    encode_value(&encoder, 4, &shape);
    encoder.dispatch_thread_groups(thread_group(1), thread_group(128));
    encoder.end_encoding();
    complete(command)?;
    let actual = buffer_slice::<f32>(&gradient, 3)?;
    let loss = |x: &[f64; 3]| {
        probabilities(x)
            .iter()
            .zip(upstream)
            .map(|(p, g)| p * f64::from(g))
            .sum::<f64>()
    };
    for step in [1e-3, 3e-4, 1e-4] {
        for index in 0..3 {
            let mut plus = ordered;
            let mut minus = ordered;
            plus[index] += step;
            minus[index] -= step;
            let expected = (loss(&plus) - loss(&minus)) / (2.0 * step);
            assert!(
                (f64::from(actual[index]) - expected).abs() < 2e-7,
                "logits={ordered:?} upstream={upstream:?} index={index} actual={} expected={expected}",
                actual[index]
            );
        }
    }
    Ok(())
}

#[test]
fn selected_adjoint() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        qualify(&runtime, [0.0; 3], [1.0; 3])?;
        qualify(&runtime, [0.125, -1.25, 0.375], [0.75, -1.25, 0.5])?;
        qualify(&runtime, [-50.0, -51.0, -52.0], [0.75, -1.25, 0.5])?;
        qualify(&runtime, [20.0, 16.0, 12.0], [0.75, -1.25, 0.5])?;
        println!("ROUTER_ADJOINT fixtures=4 coordinates=12 step_sizes=3 tolerance=2e-7");
        Ok(())
    })
}

fn projection(runtime: &Runtime, rows: usize) -> Result<(), String> {
    let width = WIDTH as usize;
    let columns = routing::MOE_GATEUP as usize;
    let first = 3usize;
    let count = first + rows + 2;
    let mut input = vec![0u16; count * width];
    let mut weights = vec![0u16; 3 * width * columns];
    for row in 0..count {
        for (j, column) in [0, 17, 511].into_iter().enumerate() {
            input[row * width + column] = resources::half_bits(((row + j) % 7 + 1) as f32 / 8.0);
        }
    }
    for input in [0, 17, 511] {
        for output in 0..columns {
            weights[(2 * width + input) * columns + output] =
                resources::half_bits(((input + output) % 13) as f32 / 16.0 - 0.375);
        }
    }
    let input_buffer = runtime.buffer_with(&input);
    let weight_buffer = runtime.buffer_with(&weights);
    let tiles = runtime.buffer_with(&[1u32, first as u32, rows as u32, 0]);
    let output = runtime.buffer_with(&vec![0x7e00u16; count * columns]);
    let pipeline = runtime.pipeline_metal4(
        include_str!("../fbt_moe_routing_tensorops.metal"),
        "routed projection replay",
        "fbt_moe_routed_gate_projection",
        &[],
    )?;
    let shape = routing::Top3RouteShape::new(rows as u32, WIDTH);
    let command = runtime.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    routing::tensor_encode(
        &encoder,
        &pipeline,
        &[
            (&input_buffer, 0),
            (&weight_buffer, 0),
            (&tiles, 0),
            (&output, 0),
        ],
        std::ptr::from_ref(&shape).cast(),
        size_of::<routing::Top3RouteShape>() as u64,
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    )?;
    encoder.end_encoding();
    complete(command)?;
    let actual = buffer_slice::<u16>(&output, count * columns)?;
    assert!(actual[..first * columns].iter().all(|&x| x == 0x7e00));
    assert!(
        actual[(first + rows) * columns..]
            .iter()
            .all(|&x| x == 0x7e00)
    );
    for row in first..first + rows {
        for column in 0..columns {
            let expected: f32 = [0, 17, 511]
                .into_iter()
                .map(|i| {
                    decode_half(input[row * width + i]) as f32
                        * decode_half(weights[(2 * width + i) * columns + column]) as f32
                })
                .sum();
            assert_eq!(
                actual[row * columns + column],
                resources::half_bits(expected),
                "row={row} column={column}"
            );
        }
    }
    println!("PROJECTION_REPLAY rows={rows} columns={columns} exact=true");
    Ok(())
}

#[test]
fn projection_replay() -> Result<(), String> {
    metal::objc::rc::autoreleasepool(|| {
        let runtime = Runtime::shared()?;
        for rows in [1, 16, 17, 63, 64] {
            projection(&runtime, rows)?;
        }
        Ok(())
    })
}
