//! Runtime-assembled PTX recipes, checked against independent FP16 arithmetic.
use crate::capture::{CudaResult, cuda_error, timing_event};
use baux_cuda_device::{MatmulShape, fbt_model};
use cuda_core::{
    CudaContext, CudaFunction, CudaStream, DeviceBuffer, DeviceCopy,
    simt::{LaunchConfig2D, launch_kernel_on_stream},
};
use ptx_synth::{GemmConfig, TargetArch, emit_gemm16, emit_gemm32};
use std::{
    ffi::c_void,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_IMAGE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Gemm {
    function: CudaFunction,
    output_bytes: usize,
}

impl Gemm {
    pub(crate) fn from_env(context: &Arc<CudaContext>) -> CudaResult<Option<Self>> {
        match std::env::var("BAUX_SYNTH_GEMM") {
            Ok(value) => {
                let k = value
                    .parse()
                    .map_err(|_| "BAUX_SYNTH_GEMM must be 8, 16, 32, or 64")?;
                let kernel = Self::new(context, k)?;
                kernel.check(&context.default_stream())?;
                Ok(Some(kernel))
            }
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn new(context: &Arc<CudaContext>, k: u32) -> CudaResult<Self> {
        Self::compile(context, k, false)
    }

    pub(crate) fn for_train(context: &Arc<CudaContext>) -> CudaResult<Self> {
        Self::compile(context, 32, true)
    }

    fn compile(context: &Arc<CudaContext>, k: u32, fp32: bool) -> CudaResult<Self> {
        if ![8, 16, 32, 64].contains(&k) {
            return Err("PTX GEMM staging depth must be 8, 16, 32, or 64".into());
        }
        let name = format!("baux_gemm_k{k}{}", if fp32 { "_f32" } else { "" });
        let (major, minor) = context.compute_capability().map_err(cuda_error)?;
        let target = TargetArch::from_compute(major, minor).ok_or_else(|| {
            format!("PTX GEMM does not support compute capability {major}.{minor}")
        })?;
        let config = GemmConfig {
            name: name.clone(),
            target,
            tile_k: k,
        };
        let ptx = if fp32 {
            emit_gemm32(&config)
        } else {
            emit_gemm16(&config)
        };
        if cfg!(target_os = "macos") {
            let image = crate::capture::driver_image(ptx.as_bytes());
            let module = context.load_module_from_image(&image).map_err(cuda_error)?;
            let function = module.load_function(&name).map_err(cuda_error)?;
            if crate::capture::enabled() {
                crate::capture::assembled(
                    context,
                    name,
                    target.ptx_target().into(),
                    ptx.as_bytes(),
                    ptx.as_bytes(),
                    Vec::new(),
                )?;
            }
            return Ok(Self {
                function,
                output_bytes: if fp32 { 4 } else { 2 },
            });
        }
        let sequence = NEXT_IMAGE.fetch_add(1, Ordering::Relaxed);
        let source = format!("/tmp/{name}-{}-{sequence}.ptx", std::process::id());
        let binary = format!("/tmp/{name}-{}-{sequence}.cubin", std::process::id());
        std::fs::write(&source, &ptx).map_err(cuda_error)?;
        let arch = format!("-arch={}", target.ptx_target());
        let assembled = std::process::Command::new("ptxas")
            .args([&arch, "-O3", "--warn-on-spills", &source, "-o", &binary])
            .output()
            .map_err(cuda_error)?;
        if !assembled.status.success() {
            return Err(format!(
                "PTX GEMM assembly failed: {}",
                String::from_utf8_lossy(&assembled.stderr)
            ));
        }
        let image = if crate::capture::enabled() {
            Some(std::fs::read(&binary).map_err(cuda_error)?)
        } else {
            None
        };
        let module = match &image {
            Some(bytes) => context.load_module_from_image(&crate::capture::driver_image(bytes)),
            None => context.load_module_from_file(&binary),
        }
        .map_err(cuda_error)?;
        let function = module.load_function(&name).map_err(cuda_error)?;
        if let Some(image) = image {
            crate::capture::assembled(
                context,
                name,
                target.ptx_target().into(),
                ptx.as_bytes(),
                &image,
                vec![
                    arch,
                    "-O3".into(),
                    "--warn-on-spills".into(),
                    source,
                    "-o".into(),
                    binary,
                ],
            )?;
        }
        Ok(Self {
            function,
            output_bytes: if fp32 { 4 } else { 2 },
        })
    }

    pub(crate) fn launch(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> CudaResult<()> {
        self.range(stream, a, b, c, m, n, k, 0)
    }

    pub(crate) fn range(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
        first: usize,
    ) -> CudaResult<()> {
        let offset = first.checked_mul(k).ok_or("PTX GEMM offset overflow")?;
        self.product(stream, a, b, c, [m, n, k], [offset, 0, 0])
    }

    pub(crate) fn product<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<T>,
        shape: [usize; 3],
        offsets: [usize; 3],
    ) -> CudaResult<()> {
        self.batched(stream, a, b, c, shape, offsets, 1)
    }

    pub(crate) fn batch<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<T>,
        shape: [usize; 3],
        batch: u32,
    ) -> CudaResult<()> {
        if self.output_bytes != 4 {
            return Err("batched GEMM requires the FP32 recipe".into());
        }
        self.batched(stream, a, b, c, shape, [0; 3], batch)
    }

    fn batched<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &DeviceBuffer<T>,
        shape: [usize; 3],
        offsets: [usize; 3],
        batch: u32,
    ) -> CudaResult<()> {
        let [m, n, k] = shape;
        if size_of::<T>() != self.output_bytes
            || batch == 0
            || batch > 65535
            || m == 0
            || n == 0
            || k == 0
            || m.checked_mul(k)
                .and_then(|v| v.checked_mul(batch as usize))
                .and_then(|v| v.checked_add(offsets[0]))
                .is_none_or(|v| v > a.len())
            || k.checked_mul(n)
                .and_then(|v| v.checked_mul(batch as usize))
                .and_then(|v| v.checked_add(offsets[1]))
                .is_none_or(|v| v > b.len())
            || m.checked_mul(n)
                .and_then(|v| v.checked_mul(batch as usize))
                .and_then(|v| v.checked_add(offsets[2]))
                .is_none_or(|v| v > c.len())
            || !Arc::ptr_eq(a.context(), stream.context())
            || !Arc::ptr_eq(b.context(), stream.context())
            || !Arc::ptr_eq(c.context(), stream.context())
            || a.cu_deviceptr() == c.cu_deviceptr()
            || b.cu_deviceptr() == c.cu_deviceptr()
        {
            return Err("PTX GEMM buffer extent mismatch".into());
        }
        let mut ap = a.cu_deviceptr() + (offsets[0] * 2) as u64;
        let mut bp = b.cu_deviceptr() + (offsets[1] * 2) as u64;
        let mut cp = c.cu_deviceptr() + (offsets[2] * self.output_bytes) as u64;
        let mut m = u32::try_from(m).map_err(cuda_error)?;
        let mut n = u32::try_from(n).map_err(cuda_error)?;
        let mut k = u32::try_from(k).map_err(cuda_error)?;
        let mut args = [
            (&mut ap as *mut u64).cast::<c_void>(),
            (&mut bp as *mut u64).cast::<c_void>(),
            (&mut cp as *mut u64).cast::<c_void>(),
            (&mut m as *mut u32).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut k as *mut u32).cast::<c_void>(),
        ];
        // SAFETY: checked matrix types, extents and contexts match the PTX ABI;
        // one block covers 64x16 outputs, including predicated boundary tiles.
        unsafe {
            launch_kernel_on_stream(
                &self.function,
                (n.div_ceil(16), m.div_ceil(64), batch),
                (256, 1, 1),
                0,
                stream,
                &mut args,
            )
            .map_err(cuda_error)
        }
    }

    fn check(&self, stream: &CudaStream) -> CudaResult<()> {
        for (m, n, k) in [(1, 1, 1), (63, 625, 216), (65, 33, 9), (129, 81, 65)] {
            let a = data(m * k, 17);
            let b = data(k * n, 29);
            let ad = DeviceBuffer::from_host(stream, &a).map_err(cuda_error)?;
            let bd = DeviceBuffer::from_host(stream, &b).map_err(cuda_error)?;
            let cd = DeviceBuffer::zeroed(stream, m * n).map_err(cuda_error)?;
            self.launch(stream, &ad, &bd, &cd, m, n, k)?;
            let actual = cd.to_host_vec(stream).map_err(cuda_error)?;
            for row in 0..m {
                for col in 0..n {
                    let expected = (0..k)
                        .map(|i| {
                            half::f16::from_bits(a[row * k + i]).to_f32()
                                * half::f16::from_bits(b[i * n + col]).to_f32()
                        })
                        .sum::<f32>();
                    let expected = half::f16::from_f32(expected).to_bits();
                    let observed = actual[row * n + col];
                    if half::f16::from_bits(observed).to_f32()
                        != half::f16::from_bits(expected).to_f32()
                    {
                        return Err(format!(
                            "PTX GEMM parity failed at {m}x{n}x{k} [{row},{col}]: {observed:04x} \
                             != {expected:04x}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

fn data(len: usize, salt: usize) -> Vec<u16> {
    (0..len)
        .map(|i| half::f16::from_f32(((i * salt + i / 7) % 31) as f32 / 16.0 - 0.9375).to_bits())
        .collect()
}

pub(crate) fn matmul(
    synth: Option<&Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    columns: usize,
    inner: usize,
) -> CudaResult<()> {
    matmul_at(
        synth, module, stream, input, weights, output, rows, columns, inner, 0,
    )
}

pub(crate) fn matmul_at(
    _synth: Option<&Gemm>,
    module: &fbt_model::LoadedModule,
    stream: &CudaStream,
    input: &DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    rows: usize,
    columns: usize,
    inner: usize,
    first: usize,
) -> CudaResult<()> {
    if first
        .checked_add(rows)
        .and_then(|v| v.checked_mul(inner))
        .is_none_or(|v| v > input.len())
    {
        return Err("matmul input range exceeds allocation".into());
    }
    let shape = MatmulShape {
        rows: rows as u32,
        columns: columns as u32,
        inner: inner as u32,
        input_stride: inner as u32,
        weight_stride: columns as u32,
        output_stride: columns as u32,
        input_offset: (first * inner) as u64,
        weight_offset: 0,
        output_offset: 0,
    };
    let launch = module
        .prepare_matmul_turing(LaunchConfig2D::new(
            (columns.div_ceil(16) as u32, rows.div_ceil(64) as u32),
            (256, 1),
            0,
        ))
        .map_err(cuda_error)?;
    module
        .matmul_turing(
            stream,
            &launch,
            input,
            weights,
            cuda_host::RowWidth::new(output, columns as u32),
            shape,
        )
        .map_err(cuda_error)
}

pub fn bench() -> CudaResult<()> {
    println!("{}", measure()?);
    Ok(())
}

fn widen_upload(stream: &CudaStream, host: &[u16]) -> CudaResult<DeviceBuffer<f32>> {
    let values: Vec<f32> = host
        .iter()
        .map(|&v| half::f16::from_bits(v).to_f32())
        .collect();
    DeviceBuffer::from_host(stream, &values).map_err(cuda_error)
}

pub(crate) fn measure() -> CudaResult<String> {
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let baseline = crate::capture::load_model(&context)?;
    let mut records = Vec::new();
    for depth in [8, 16, 32, 64] {
        let kernel = Gemm::new(&context, depth)?;
        kernel.check(&stream)?;
        for n in [512, 625, 640, 8192] {
            let (m, k) = (4096, 512);
            let a_host = data(m * k, 17);
            let b_host = data(k * n, 29);
            let a = DeviceBuffer::from_host(&stream, &a_host).map_err(cuda_error)?;
            let b = DeviceBuffer::from_host(&stream, &b_host).map_err(cuda_error)?;
            let a_f32 = widen_upload(&stream, &a_host)?;
            let b_f32 = widen_upload(&stream, &b_host)?;
            let c = DeviceBuffer::zeroed(&stream, m * n).map_err(cuda_error)?;
            let mut reference = DeviceBuffer::<f32>::zeroed(&stream, m * n).map_err(cuda_error)?;
            kernel.launch(&stream, &a, &b, &c, m, n, k)?;
            stream.synchronize().map_err(cuda_error)?;
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = timing_event(&stream)?;
                kernel.launch(&stream, &a, &b, &c, m, n, k)?;
                let end = timing_event(&stream)?;
                stream.synchronize().map_err(cuda_error)?;
                samples.push(start.elapsed_ms(&end).map_err(cuda_error)?);
            }
            samples.sort_by(f32::total_cmp);
            matmul(
                None,
                &baseline,
                &stream,
                &a_f32,
                &b_f32,
                &mut reference,
                m,
                n,
                k,
            )?;
            stream.synchronize().map_err(cuda_error)?;
            let mut control = Vec::new();
            for _ in 0..5 {
                let start = timing_event(&stream)?;
                matmul(
                    None,
                    &baseline,
                    &stream,
                    &a_f32,
                    &b_f32,
                    &mut reference,
                    m,
                    n,
                    k,
                )?;
                let end = timing_event(&stream)?;
                stream.synchronize().map_err(cuda_error)?;
                control.push(start.elapsed_ms(&end).map_err(cuda_error)?);
            }
            control.sort_by(f32::total_cmp);
            let actual = c.to_host_vec(&stream).map_err(cuda_error)?;
            let expected = reference.to_host_vec(&stream).map_err(cuda_error)?;
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|&(&a, &b)| a != half::f16::from_f32(b).to_bits())
                .count();
            if mismatches != 0 {
                return Err(format!(
                    "PTX GEMM differs from CUDA-Oxide on {m}x{n}x{k}: {mismatches} elements"
                ));
            }
            records.push(baux::artifact::json::json!({"tile_k":depth,"m":m,"n":n,"k":k,"samples_ms":samples,"median_ms":samples[2],"baseline_ms":control[2],"speedup":control[2]/samples[2],"cpu_edge_parity":true,"baseline_mismatches":mismatches}));
        }
    }
    baux::artifact::json::pretty_string(&baux::artifact::json::json!({"schema":"baux.ptx.gemm.v1","hardware":context.device_name().map_err(cuda_error)?,"kernels":records,"full_generation_loop":false})).map_err(cuda_error)
}
