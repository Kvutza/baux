//! Run the shared fixture through production hierarchical mHC-4 kernels.
use super::{CudaResult, cuda_error, fbt_model, read_prefix};
use baux::semantics::mhc4 as mhc_sem;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};

pub(crate) fn check() -> CudaResult<()> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let module = super::capture::load_model(&context)?;
    check_objective(&stream, &module)?;
    check_encode(&stream, &module)?;
    check_wave(&stream, &module)?;
    context.check_err().map_err(cuda_error)?;
    println!(
        "MHC4_WAVE_PARITY ok=true tokens=65 patches=2 partial=true temperatures=0,0.8 \
         token_decisions=exact"
    );
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
    let readout = DeviceBuffer::from_host(stream, &fixture.transposed()).map_err(cuda_error)?;
    let tokens = DeviceBuffer::from_host(stream, &fixture.tokens).map_err(cuda_error)?;
    let positions = DeviceBuffer::from_host(stream, &fixture.positions).map_err(cuda_error)?;
    let mask = DeviceBuffer::from_host(stream, &fixture.mask).map_err(cuda_error)?;
    let total = mhc_sem::probe::encode::ROWS * mhc_sem::probe::encode::WIDTH;
    let mut output = DeviceBuffer::zeroed(stream, total).map_err(cuda_error)?;
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
    fixture.verify(&read_prefix(&output, stream, total)?)?;
    println!("MHC4_ENCODE_PARITY ok=true patches=full,partial,empty,masked fp16=exact");
    Ok(())
}

fn check_wave(stream: &CudaStream, module: &fbt_model::LoadedModule) -> CudaResult<()> {
    let fixture = mhc_sem::probe::Fixture::new();
    let hidden = DeviceBuffer::from_host(stream, &fixture.hidden).map_err(cuda_error)?;
    let qkv = DeviceBuffer::from_host(stream, &fixture.qkv).map_err(cuda_error)?;
    let positions = DeviceBuffer::from_host(stream, &fixture.positions).map_err(cuda_error)?;
    let readout = DeviceBuffer::from_host(stream, &fixture.readout).map_err(cuda_error)?;
    let expected = DeviceBuffer::from_host(stream, &fixture.targets).map_err(cuda_error)?;
    let seeds = DeviceBuffer::from_host(stream, &[fixture.seed]).map_err(cuda_error)?;
    let mut centroids =
        DeviceBuffer::zeroed(stream, fixture.centroids.len()).map_err(cuda_error)?;
    let mut output = DeviceBuffer::zeroed(stream, fixture.targets.len()).map_err(cuda_error)?;
    let mut losses = DeviceBuffer::zeroed(stream, fixture.targets.len()).map_err(cuda_error)?;
    let centroid_launch = module
        .prepare_mhc4_centroids(LaunchConfig1D::new(
            fixture.centroids.len().div_ceil(256) as u32,
            256,
            0,
        ))
        .map_err(cuda_error)?;
    let wave_launch = module
        .prepare_mhc4_expand(LaunchConfig1D::new(2, 64, 0))
        .map_err(cuda_error)?;
    for temperature in [0.0, 0.8] {
        let shape = mhc_sem::shape(mhc_sem::PATCH, fixture.targets.len(), temperature, true)?;
        module
            .mhc4_centroids(stream, &centroid_launch, &readout, &mut centroids, shape)
            .map_err(cuda_error)?;
        module
            .mhc4_expand(
                stream,
                &wave_launch,
                &hidden,
                &qkv,
                &positions,
                &readout,
                &centroids,
                &seeds,
                &expected,
                &mut output,
                &mut losses,
                shape,
            )
            .map_err(cuda_error)?;
        fixture.verify(
            temperature,
            &read_prefix(&output, stream, fixture.targets.len())?,
            &read_prefix(&losses, stream, fixture.targets.len())?,
            &read_prefix(&centroids, stream, fixture.centroids.len())?,
        )?;
    }
    Ok(())
}
