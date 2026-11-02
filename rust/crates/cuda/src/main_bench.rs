use std::io;

use baux_cuda::target_name;
use baux_cuda_device::trials;
use cuda_core::{CudaContext, DeviceBuffer};

use super::main_parity::{
    AppResult, Leaf, compare_rows, expected_row, launch_materialize, make_base, parse_arg,
    row_bytes,
};

pub(super) fn benchmark(args: &[String]) -> AppResult<()> {
    if args.first().is_some_and(|value| value == "muon") {
        if args.len() != 1 {
            return Err(io::Error::other("bench muon accepts no additional arguments").into());
        }
        let record = baux_cuda::muon::measure().map_err(io::Error::other)?;
        println!("{}", baux_cuda::artifact::json::pretty_string(&record)?);
        return Ok(());
    }
    let elements = parse_arg(args, 0, 16 * 1024 * 1024, "elements")?;
    let iterations = parse_arg(args, 1, 100, "iterations")?;
    let bits: u8 = parse_arg(args, 2, 4, "bits")?;
    if iterations == 0 {
        return Err(io::Error::other("iterations must be positive").into());
    }
    if bits != 4 && bits != 8 {
        return Err(io::Error::other("bits must be 4 or 8").into());
    }

    let scale = if bits == 4 { 0.125 } else { 0.03125 };
    let leaves = vec![Leaf::new(0, elements, bits, scale, 0.75)?];
    let base = make_base(row_bytes(&leaves));
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel
    // artifact.
    let module = unsafe { trials::load(&context) }?;
    let base_device = DeviceBuffer::from_host(&stream, &base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(&stream, base.len())?;
    let length = 0.8;

    for seed in 0..5 {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            seed,
        )?;
    }
    stream.synchronize()?;

    let event_flags = cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
    let start = stream.record_event(Some(event_flags))?;
    for iteration in 0..iterations {
        launch_materialize(
            &module,
            &stream,
            &base_device,
            &mut output_device,
            &leaves,
            length,
            10_000 + iteration as u64,
        )?;
    }
    let end = stream.record_event(Some(event_flags))?;
    let total_ms = f64::from(start.elapsed_ms(&end)?);
    let output = output_device.to_host_vec(&stream)?;
    let final_seed = 10_000 + iterations as u64 - 1;
    let expected = expected_row(&base, &leaves, length, final_seed)?;
    compare_rows(&expected, &output, final_seed, length)?;

    let kernel_ms = total_ms / iterations as f64;
    let bytes = (base.len() as f64) * 2.0 * iterations as f64;
    let gib_s = bytes / (total_ms / 1_000.0) / (1024.0 * 1024.0 * 1024.0);
    let target = target_name(&context).map_err(io::Error::other)?;
    println!(
        "BENCH ok=true target={target} elements={elements} bits={bits} iterations={iterations} \
         row_bytes={} kernel_ms={kernel_ms:.6} gib_s={gib_s:.3}",
        base.len()
    );
    Ok(())
}
