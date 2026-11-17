//! Device parity and latency for the bounded FP32 neighbour geometry.
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
            .map(|i| 0.25_f32 + ((i.wrapping_mul(salt) ^ (i >> 7)) % 1024) as f32 / 4096.0)
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
                        let delta = f64::from(a) - f64::from(b);
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

pub fn pool_parity(candidates: usize, arms: usize) -> CudaResult<()> {
    use baux::semantics::key::Key;
    use baux::semantics::neighbor::{MetricKind, UncertaintyLaw};
    use baux::semantics::optimizer::surrogate::AcquisitionRule;
    use baux::semantics::optimizer::trust_region::{RoundPlan, SurrogateMode, TurboState};

    let dims = 40usize;
    let hist = 48usize;
    let cands = candidates.max(8);
    let num_arms = arms.clamp(1, cands);
    let rows: Vec<f32> = (0..hist * dims)
        .map(|i| ((i.wrapping_mul(37) ^ (i >> 3)) % 1024) as f32 / 1024.0)
        .collect();
    let outcomes: Vec<f32> = (0..hist)
        .map(|r| -0.5 + ((r * 19 + 7) % 64) as f32 / 32.0)
        .collect();
    let center = rows[..dims].to_vec();
    let state = TurboState::for_dims(dims, num_arms, 0.8, 0.01, 1.6, None);
    let plan = RoundPlan::turbo_enn(
        Key::new(2026),
        1,
        dims,
        cands,
        num_arms,
        state,
        SurrogateMode::Enn {
            neighbors: 12,
            epistemic: 0.05,
            aleatoric: 0.05,
            scale: 1.0,
            metric: MetricKind::Euclidean,
            law: UncertaintyLaw::InversePrecision,
            noise_aware: false,
        },
        AcquisitionRule::default(),
    );

    let index = CudaIndex::with_mode(dims, &rows, CudaKnnMode::Kernel)?;
    let mut scratch = index.new_scratch()?;
    let fenced = index.propose_fenced(&mut scratch, &plan, &center, &outcomes, None, 1, 0, None)?;
    let actual = fenced.materialize(None)?;
    let expected = crate::simulate_round(&rows, hist, &plan, &center, &outcomes, None, 1, 0, None)?;
    if actual.selected != expected.selected {
        return Err(format!(
            "CUDA DelayedPool selected {:?} != CPU {:?}",
            actual.selected, expected.selected
        ));
    }
    let max_row_err = actual
        .rows
        .iter()
        .zip(&expected.rows)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    if max_row_err > 1.0e-5 {
        return Err(format!(
            "CUDA DelayedPool materialized row error {max_row_err}"
        ));
    }
    Ok(())
}
