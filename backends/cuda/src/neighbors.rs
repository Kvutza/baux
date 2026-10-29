//! Device parity and latency for the bounded FP16 neighbour geometry.
use super::*;
use baux_cuda_device::NeighborShape;

pub fn parity() -> CudaResult<()> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    // SAFETY: generated bindings load this source revision's embedded module.
    let module = unsafe { trials::load(&context) }.map_err(cuda_error)?;
    let shape = NeighborShape {
        samples: 51_437,
        history: 64,
        candidates: 4,
    };
    let values = |rows: u32, salt: u32| {
        (0..rows * shape.samples)
            .map(|i| 0x3400_u16 + ((i.wrapping_mul(salt) ^ (i >> 7)) % 1024) as u16)
            .collect::<Vec<_>>()
    };
    let pool = values(shape.candidates, 17);
    let history = values(shape.history, 31);
    let weights = (0..shape.samples)
        .map(|i| 1.0 + (i % 3) as f32)
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for candidate in pool.chunks_exact(shape.samples as usize) {
        for row in history.chunks_exact(shape.samples as usize) {
            expected.push(
                candidate
                    .iter()
                    .zip(row)
                    .zip(&weights)
                    .map(|((&a, &b), &weight)| {
                        let delta = f64::from(a & 1023) / 4096.0 - f64::from(b & 1023) / 4096.0;
                        delta * delta * f64::from(weight)
                    })
                    .sum::<f64>(),
            );
        }
    }
    let pool = DeviceBuffer::from_host(&stream, &pool).map_err(cuda_error)?;
    let history = DeviceBuffer::from_host(&stream, &history).map_err(cuda_error)?;
    let weights = DeviceBuffer::from_host(&stream, &weights).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(&stream, expected.len()).map_err(cuda_error)?;
    let launch = module
        .prepare_neighbor_distances(LaunchConfig1D::new(expected.len() as u32, 256, 0))
        .map_err(cuda_error)?;
    module
        .neighbor_distances(
            &stream,
            &launch,
            &pool,
            &history,
            &weights,
            &mut output,
            shape,
        )
        .map_err(cuda_error)?;
    let start = timing_event(&stream)?;
    for _ in 0..4 {
        module
            .neighbor_distances(
                &stream,
                &launch,
                &pool,
                &history,
                &weights,
                &mut output,
                shape,
            )
            .map_err(cuda_error)?;
    }
    let end = timing_event(&stream)?;
    let actual = output.to_host_vec(&stream).map_err(cuda_error)?;
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - b).abs() / b.max(1.0e-12))
        .fold(0.0, f64::max);
    if max_error > 1.0e-5 {
        return Err(format!(
            "FP16 neighbour geometry relative error {max_error}"
        ));
    }
    println!(
        "NEIGHBOR_GEOMETRY parity=true samples={} history={} candidates={} kernel_ms={:.6} \
         max_relative_error={max_error:.9} full_bo_loop=false",
        shape.samples,
        shape.history,
        shape.candidates,
        start.elapsed_ms(&end).map_err(cuda_error)? / 4.0
    );
    Ok(())
}
