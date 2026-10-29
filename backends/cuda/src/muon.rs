//! Native CUDA optimizer updates. Model backward supplies the analytic
//! gradients.
use crate::{CudaResult, cuda_error};
pub use baux::semantics::muon::reference;
pub use baux_cuda_device::muon::{Shape, Step};
use baux_cuda_device::muon::{spec::POLAR, updates};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

macro_rules! run {
    ($module:expr, $prepare:ident, $kernel:ident, $grid:expr, $stream:expr $(, $arg:expr)*) => {{
        let launch = $module.$prepare($grid).map_err(cuda_error)?;
        $module.$kernel($stream, &launch $(, $arg)*).map_err(cuda_error)
    }};
}

fn config(count: usize) -> LaunchConfig1D {
    LaunchConfig1D::new((count as u32).div_ceil(256), 256, 0)
}

pub struct Workspace {
    capacity: Shape,
    kernels: updates::LoadedModule,
    gemm: crate::gemm::Gemm,
    x: [DeviceBuffer<u16>; 2],
    transpose: DeviceBuffer<u16>,
    gram: DeviceBuffer<u16>,
    polynomial: DeviceBuffer<u16>,
    product: DeviceBuffer<f32>,
    partials: DeviceBuffer<f32>,
    norms: DeviceBuffer<f32>,
}

impl Workspace {
    pub fn new(stream: &CudaStream, capacity: Shape) -> CudaResult<Self> {
        let capacity = Shape::new(capacity.rows, capacity.cols, capacity.batch)?;
        let count = capacity.elements();
        let wide = capacity.wide();
        let gram = wide.rows as usize * wide.rows as usize * wide.batch as usize;
        // SAFETY: generated bindings load their own embedded device module.
        let kernels = unsafe { updates::load(stream.context()) }.map_err(cuda_error)?;
        Ok(Self {
            capacity,
            kernels,
            gemm: crate::gemm::Gemm::for_train(stream.context())?,
            x: [
                DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?,
                DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?,
            ],
            transpose: DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?,
            gram: DeviceBuffer::zeroed(stream, gram).map_err(cuda_error)?,
            polynomial: DeviceBuffer::zeroed(stream, gram).map_err(cuda_error)?,
            product: DeviceBuffer::zeroed(stream, count.max(gram)).map_err(cuda_error)?,
            partials: DeviceBuffer::zeroed(stream, (capacity.chunks() * capacity.batch) as usize)
                .map_err(cuda_error)?,
            norms: DeviceBuffer::zeroed(stream, capacity.batch as usize).map_err(cuda_error)?,
        })
    }
    fn validate(&self, stream: &CudaStream, shape: Shape) -> CudaResult<()> {
        let wide = shape.wide();
        if shape.elements() > self.capacity.elements()
            || wide.rows as usize * wide.rows as usize * wide.batch as usize > self.gram.len()
            || shape.chunks() * shape.batch > self.capacity.chunks() * self.capacity.batch
            || shape.batch > self.capacity.batch
            || !std::sync::Arc::ptr_eq(self.product.context(), stream.context())
        {
            return Err("Muon workspace capacity or context mismatch".into());
        }
        Ok(())
    }
    pub fn bytes(&self) -> usize {
        self.x.iter().map(|b| b.len() * 2).sum::<usize>()
            + self.transpose.len() * 2
            + self.gram.len() * 2
            + self.polynomial.len() * 2
            + self.product.len() * 4
            + self.partials.len() * 4
            + self.norms.len() * 4
    }
}

pub struct Muon {
    shape: Shape,
    master: DeviceBuffer<f32>,
    momentum: DeviceBuffer<f32>,
    weights: u64,
}

impl Muon {
    pub fn new(
        stream: &CudaStream,
        workspace: &Workspace,
        shape: Shape,
        weights: &DeviceBuffer<u16>,
    ) -> CudaResult<Self> {
        let shape = Shape::new(shape.rows, shape.cols, shape.batch)?;
        workspace.validate(stream, shape)?;
        let count = shape.elements();
        if weights.len() != count || !std::sync::Arc::ptr_eq(weights.context(), stream.context()) {
            return Err("Muon weight extent or context mismatch".into());
        }
        let mut master = DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?;
        let mut momentum = DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?;
        run!(
            workspace.kernels,
            prepare_init,
            init,
            config(count),
            stream,
            weights,
            &mut master,
            &mut momentum,
            count as u32
        )?;
        Ok(Self {
            shape,
            master,
            momentum,
            weights: weights.cu_deviceptr(),
        })
    }
    /// Consume FP32 gradients after every recurrent contribution has been
    /// summed. Scratch is reused across banks; no host copies or
    /// synchronization here.
    pub fn step(
        &mut self,
        stream: &CudaStream,
        workspace: &mut Workspace,
        weights: &mut DeviceBuffer<u16>,
        gradient: &mut DeviceBuffer<f32>,
        mut step: Step,
    ) -> CudaResult<()> {
        step.validate()?;
        workspace.validate(stream, self.shape)?;
        let count = self.shape.elements();
        if weights.cu_deviceptr() != self.weights
            || weights.len() != count
            || gradient.len() != count
            || !std::sync::Arc::ptr_eq(self.master.context(), stream.context())
            || !std::sync::Arc::ptr_eq(gradient.context(), stream.context())
        {
            return Err("Muon binding, gradient extent, or context mismatch".into());
        }
        step.scale *= 0.2 * (self.shape.rows.max(self.shape.cols) as f32).sqrt();
        let chunks = self.shape.chunks() * self.shape.batch;
        run!(
            workspace.kernels,
            prepare_momentum,
            momentum,
            config(chunks as usize * 256),
            stream,
            gradient,
            &mut self.momentum,
            &mut workspace.partials,
            self.shape,
            step
        )?;
        run!(
            workspace.kernels,
            prepare_norm,
            norm,
            config(self.shape.batch as usize * 256),
            stream,
            &workspace.partials,
            &mut workspace.norms,
            self.shape,
            step
        )?;
        run!(
            workspace.kernels,
            prepare_normalize,
            normalize,
            config(count),
            stream,
            gradient,
            &workspace.norms,
            &mut workspace.x[0],
            self.shape
        )?;
        let wide = self.shape.wide();
        let [rows, cols] = [wide.rows as usize, wide.cols as usize];
        let gram_count = rows * rows * wide.batch as usize;
        for (index, polynomial) in POLAR.iter().copied().enumerate() {
            let (x, next) = if index % 2 == 0 {
                let (a, b) = workspace.x.split_at_mut(1);
                (&a[0], &mut b[0])
            } else {
                let (a, b) = workspace.x.split_at_mut(1);
                (&b[0], &mut a[0])
            };
            let tiles = wide.rows.div_ceil(16) * wide.cols.div_ceil(16) * wide.batch;
            run!(
                workspace.kernels,
                prepare_transpose,
                transpose,
                config(tiles as usize * 256),
                stream,
                x,
                &mut workspace.transpose,
                wide
            )?;
            workspace.gemm.batch(
                stream,
                x,
                &workspace.transpose,
                &workspace.product,
                [rows, rows, cols],
                wide.batch,
            )?;
            run!(
                workspace.kernels,
                prepare_cast,
                cast,
                config(gram_count),
                stream,
                &workspace.product,
                &mut workspace.gram,
                gram_count as u32
            )?;
            workspace.gemm.batch(
                stream,
                &workspace.gram,
                &workspace.gram,
                &workspace.product,
                [rows, rows, rows],
                wide.batch,
            )?;
            run!(
                workspace.kernels,
                prepare_polynomial,
                polynomial,
                config(gram_count),
                stream,
                &workspace.gram,
                &workspace.product,
                &mut workspace.polynomial,
                gram_count as u32,
                polynomial
            )?;
            workspace.gemm.batch(
                stream,
                &workspace.polynomial,
                x,
                &workspace.product,
                [rows, cols, rows],
                wide.batch,
            )?;
            run!(
                workspace.kernels,
                prepare_axpy,
                axpy,
                config(count),
                stream,
                x,
                &workspace.product,
                next,
                count as u32,
                polynomial
            )?;
        }
        run!(
            workspace.kernels,
            prepare_update,
            update,
            config(count),
            stream,
            &workspace.x[POLAR.len() % 2],
            &mut self.master,
            weights,
            self.shape,
            step
        )
    }
    pub fn state_bytes(&self) -> usize {
        self.master.len() * 4 + self.momentum.len() * 4
    }
}
pub struct Adam {
    kernels: updates::LoadedModule,
    master: DeviceBuffer<f32>,
    first: DeviceBuffer<f32>,
    second: DeviceBuffer<f32>,
    weights: u64,
    updates: u32,
}

impl Adam {
    pub fn new(stream: &CudaStream, weights: &DeviceBuffer<u16>) -> CudaResult<Self> {
        let count = weights.len();
        if count == 0
            || count > u32::MAX as usize
            || !std::sync::Arc::ptr_eq(weights.context(), stream.context())
        {
            return Err("AdamW weight extent or context mismatch".into());
        }
        // SAFETY: generated bindings load their own embedded device module.
        let kernels = unsafe { updates::load(stream.context()) }.map_err(cuda_error)?;
        let mut master = DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?;
        let mut first = DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?;
        let second = DeviceBuffer::zeroed(stream, count).map_err(cuda_error)?;
        run!(
            kernels,
            prepare_init,
            init,
            config(count),
            stream,
            weights,
            &mut master,
            &mut first,
            count as u32
        )?;
        Ok(Self {
            kernels,
            master,
            first,
            second,
            weights: weights.cu_deviceptr(),
            updates: 0,
        })
    }

    pub fn step(
        &mut self,
        stream: &CudaStream,
        weights: &mut DeviceBuffer<u16>,
        gradient: &DeviceBuffer<f32>,
        step: Step,
    ) -> CudaResult<()> {
        let updates = self
            .updates
            .checked_add(1)
            .ok_or("AdamW update count overflow")?;
        let step = step.adam(updates)?;
        let count = self.master.len();
        if weights.cu_deviceptr() != self.weights
            || weights.len() != count
            || gradient.len() != count
            || !std::sync::Arc::ptr_eq(self.master.context(), stream.context())
            || !std::sync::Arc::ptr_eq(gradient.context(), stream.context())
        {
            return Err("AdamW binding, gradient extent, or context mismatch".into());
        }
        run!(
            self.kernels,
            prepare_adam,
            adam,
            config(count),
            stream,
            gradient,
            &mut self.first,
            &mut self.second,
            &mut self.master,
            weights,
            count as u32,
            step
        )?;
        self.updates = updates;
        Ok(())
    }
}
/// Timings for model-shaped matrix banks with a fixed arithmetic gradient
/// fixture. This deliberately reports full-training-loop=false.
pub fn measure() -> CudaResult<baux::artifact::json::Value> {
    use crate::timing_event;
    use baux::artifact::json::json;
    use std::time::Instant;
    let context = cuda_core::CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let mut records = Vec::new();
    for (rows, cols, batch) in [(32, 48, 2), (512, 640, 5), (512, 432, 626)] {
        let shape = Shape::new(rows, cols, batch)?;
        let count = shape.elements();
        let mut weights =
            DeviceBuffer::from_host(&stream, &vec![0x3800u16; count]).map_err(cuda_error)?;
        let fixture = (0..count)
            .map(|i| ((i * 17 + i / 7) % 127) as f32 / 127.0 - 0.5)
            .collect::<Vec<_>>();
        let mut gradient = DeviceBuffer::from_host(&stream, &fixture).map_err(cuda_error)?;
        let mut workspace = Workspace::new(&stream, shape)?;
        let mut optimizer = Muon::new(&stream, &workspace, shape, &weights)?;
        for repeat in 0..4 {
            gradient
                .copy_from_host(&stream, &fixture)
                .map_err(cuda_error)?;
            stream.synchronize().map_err(cuda_error)?;
            let start = Instant::now();
            let gpu_start = timing_event(&stream)?;
            optimizer.step(
                &stream,
                &mut workspace,
                &mut weights,
                &mut gradient,
                Step::default(),
            )?;
            let gpu_end = timing_event(&stream)?;
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            stream.synchronize().map_err(cuda_error)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let observed = weights.to_host_vec(&stream).map_err(cuda_error)?;
            let invalid = observed.iter().filter(|v| **v & 0x7c00 == 0x7c00).count();
            records.push(json!({
                "rows": rows,
                "cols": cols,
                "batch": batch,
                "repeat": repeat,
                "warmup": repeat == 0,
                "encode-ms": encode_ms,
                "wall-ms": wall_ms,
                "gpu-ms": gpu_start.elapsed_ms(&gpu_end).map_err(cuda_error)?,
                "state-bytes": optimizer.state_bytes(),
                "workspace-bytes": workspace.bytes(),
                "invalid-weights": invalid,
            }));
        }
    }
    Ok(json!({
        "schema": "baux.muon-kernels.v1",
        "hardware": context.device_name().map_err(cuda_error)?,
        "variant": "polar-express-5",
        "full-training-loop": false,
        "gradient": "fixed-arithmetic-fixture",
        "records": records,
        "parity": verify()?,
    }))
}

pub fn verify() -> CudaResult<baux::artifact::json::Value> {
    use baux::artifact::json::json;
    let context = cuda_core::CudaContext::new(0).map_err(cuda_error)?;
    let stream = context.default_stream();
    let mut records = Vec::new();
    for (rows, cols) in [(3, 5), (5, 3), (8, 8)] {
        let shape = Shape::new(rows, cols, 2)?;
        let count = shape.elements();
        let mut weights =
            DeviceBuffer::from_host(&stream, &vec![0x3800u16; count]).map_err(cuda_error)?;
        let mut workspace = Workspace::new(&stream, shape)?;
        let mut optimizer = Muon::new(&stream, &workspace, shape, &weights)?;
        let mut expected = vec![0.5f64; count];
        let mut momentum = vec![0.0f64; count];
        for round in 0..4 {
            let fixture = reference::fixture(shape, round);
            let mut gradient = DeviceBuffer::from_host(&stream, &fixture).map_err(cuda_error)?;
            let step = Step {
                decay: 0.01,
                ..Step::default()
            };
            reference::muon(shape, &mut expected, &mut momentum, &fixture, step);
            optimizer.step(&stream, &mut workspace, &mut weights, &mut gradient, step)?;
            let actual = weights.to_host_vec(&stream).map_err(cuda_error)?;
            records.push(json!({"optimizer":"muon","rows":rows,"cols":cols,"round":round,"max-abs-error":reference::error(&actual,&expected)?}));
        }
    }
    verify_adam(&stream, &mut records)?;
    Ok(json!({"reference":"fp64","tolerance":0.001,"passed":true,"records":records}))
}

fn verify_adam(
    stream: &cuda_core::CudaStream,
    records: &mut Vec<baux::artifact::json::Value>,
) -> CudaResult<()> {
    use baux::artifact::json::json;
    let shape = Shape::new(4, 8, 2)?;
    let count = shape.elements();
    let mut weights =
        DeviceBuffer::from_host(stream, &vec![0x3800u16; count]).map_err(cuda_error)?;
    let mut optimizer = Adam::new(stream, &weights)?;
    let mut expected = vec![0.5f64; count];
    let mut first = vec![0.0f64; count];
    let mut second = vec![0.0f64; count];
    for round in 0..4 {
        let fixture = reference::fixture(shape, round);
        let gradient = DeviceBuffer::from_host(stream, &fixture).map_err(cuda_error)?;
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
        optimizer.step(stream, &mut weights, &gradient, step)?;
        let actual = weights.to_host_vec(stream).map_err(cuda_error)?;
        records.push(json!({"optimizer":"adamw","round":round,"max-abs-error":reference::error(&actual,&expected)?}));
    }
    Ok(())
}
