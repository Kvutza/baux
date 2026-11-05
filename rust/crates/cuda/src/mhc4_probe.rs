//! Run the shared fixture through production mHC-4 kernels.
use super::capture::{CudaResult, cuda_error, read_prefix};
use super::fbt_model;
use baux::semantics::mhc4 as mhc_sem;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

pub(crate) fn check() -> CudaResult<()> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = super::capture::load_model(&context)?;
    check_objective(&stream, &module)?;
    check_encode(&stream, &module)?;
    context.check_err().map_err(cuda_error)?;
    Ok(())
}

fn check_objective(stream: &CudaStream, module: &fbt_model::LoadedModule) -> CudaResult<()> {
    for count in [1, 65, 131, 1025] {
        for paired in [false, true] {
            let fixture = mhc_sem::probe::objective::Fixture::new(count, paired);
            let losses = DeviceBuffer::from_host(stream, &fixture.losses).map_err(cuda_error)?;
            let baseline =
                DeviceBuffer::from_host(stream, &fixture.baseline).map_err(cuda_error)?;
            let generated =
                DeviceBuffer::from_host(stream, &fixture.generated).map_err(cuda_error)?;
            let targets = DeviceBuffer::from_host(stream, &fixture.targets).map_err(cuda_error)?;
            let mut stats = DeviceBuffer::zeroed(stream, 1).map_err(cuda_error)?;
            let mut patches =
                DeviceBuffer::zeroed(stream, count.div_ceil(64)).map_err(cuda_error)?;
            let launch = module
                .prepare_mhc4_objective(LaunchConfig1D::new(1, 256, 0))
                .map_err(cuda_error)?;
            module
                .mhc4_objective(
                    stream,
                    &launch,
                    &losses,
                    &generated,
                    &targets,
                    &mut stats,
                    &baseline,
                    fixture.shape,
                    &mut patches,
                )
                .map_err(cuda_error)?;
            let actual = read_prefix(&stats, stream, 1)?[0];
            let patches = if paired {
                read_prefix(&patches, stream, count.div_ceil(64))?
            } else {
                Vec::new()
            };
            fixture.verify(&actual, &patches)?;
        }
    }
    println!(
        "MHC4_OBJECTIVE_PARITY ok=true tokens=1,65,131,1025 paired=true,false partial=true \
         token_differences=exact"
    );
    Ok(())
}

fn check_encode(stream: &CudaStream, module: &fbt_model::LoadedModule) -> CudaResult<()> {
    let fixture = mhc_sem::probe::encode::Fixture::new();
    let widen = |slice: &[u16]| -> Vec<f32> {
        slice
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect()
    };
    let readout =
        DeviceBuffer::from_host(stream, &widen(&fixture.transposed())).map_err(cuda_error)?;
    let tokens = DeviceBuffer::from_host(stream, &fixture.tokens).map_err(cuda_error)?;
    let positions =
        DeviceBuffer::from_host(stream, &widen(&fixture.positions)).map_err(cuda_error)?;
    let mask = DeviceBuffer::from_host(stream, &widen(&fixture.mask)).map_err(cuda_error)?;
    let total = mhc_sem::probe::encode::ROWS * mhc_sem::probe::encode::WIDTH;
    let mut output = DeviceBuffer::<f32>::zeroed(stream, total).map_err(cuda_error)?;
    let launch = module
        .prepare_mhc4_encode(LaunchConfig1D::new(1, 256, 0))
        .map_err(cuda_error)?;
    module
        .mhc4_encode(
            stream,
            &launch,
            &readout,
            &tokens,
            &positions,
            &mask,
            &mut output,
            baux_cuda_device::Mhc4EncodeShape {
                rows: mhc_sem::probe::encode::ROWS as u32,
                width: mhc_sem::probe::encode::WIDTH as u32,
                vocabulary: mhc_sem::probe::encode::VOCABULARY as u32,
                patch: mhc_sem::probe::encode::PATCH as u32,
            },
        )
        .map_err(cuda_error)?;
    let narrowed = read_prefix(&output, stream, total)?
        .into_iter()
        .map(|v| half::f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    fixture.verify(&narrowed)?;
    println!("MHC4_ENCODE_PARITY ok=true patches=full,partial,empty,masked fp16=exact");
    Ok(())
}
