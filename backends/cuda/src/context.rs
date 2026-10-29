//! Persistent single-sequence KV cache with bounded PISA query scratch.
use crate::{CudaResult, copy_prefix, cuda_error, read_prefix, timing_event};
pub use baux::semantics::context as layout;
use baux_cuda_device::{PisaShape, fbt_model};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use layout::{BLOCK, HEADS, Layout, Output, SELECTED, WIDTH};
use std::sync::Arc;
use std::time::Instant;

pub struct ContextCache {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: fbt_model::LoadedModule,
    layout: Layout,
    kv: DeviceBuffer<u16>,
    tree: DeviceBuffer<u16>,
    queries: DeviceBuffer<u16>,
    blocks: DeviceBuffer<u32>,
    output: DeviceBuffer<u16>,
    filled: u32,
}

impl ContextCache {
    pub fn new(layout: Layout) -> CudaResult<Self> {
        let layout = Layout::new(layout.tokens, layout.queries)?;
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let stream = context.default_stream();
        let module = crate::capture::load_model(&context)?;
        Ok(Self {
            kv: DeviceBuffer::zeroed(&stream, (layout.tokens * 2 * WIDTH) as usize)
                .map_err(cuda_error)?,
            tree: DeviceBuffer::zeroed(&stream, (layout.nodes() * WIDTH) as usize)
                .map_err(cuda_error)?,
            queries: DeviceBuffer::zeroed(&stream, (layout.queries * HEADS * WIDTH) as usize)
                .map_err(cuda_error)?,
            blocks: DeviceBuffer::zeroed(&stream, (layout.queries * SELECTED) as usize)
                .map_err(cuda_error)?,
            output: DeviceBuffer::zeroed(&stream, (layout.queries * HEADS * WIDTH) as usize)
                .map_err(cuda_error)?,
            context,
            stream,
            module,
            layout,
            filled: 0,
        })
    }

    fn rebuild(&mut self, mut first: u32, mut end: u32) -> CudaResult<()> {
        let launch = self
            .module
            .prepare_cache_leaves(LaunchConfig1D::new(end - first, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .cache_leaves(
                &self.stream,
                &launch,
                &self.kv,
                &mut self.tree,
                first,
                end - first,
            )
            .map_err(cuda_error)?;
        if self.layout.leaves() > 1 && end - first <= 64 {
            let launch = self
                .module
                .prepare_cache_ancestors(LaunchConfig1D::new(1, 64, 0))
                .map_err(cuda_error)?;
            return self
                .module
                .cache_ancestors(
                    &self.stream,
                    &launch,
                    &mut self.tree,
                    self.layout.leaves(),
                    first,
                    end,
                )
                .map_err(cuda_error);
        }
        let (mut child, mut parent, mut count) = (0, self.layout.leaves(), self.layout.leaves());
        while count > 1 {
            first /= 2;
            end = end.div_ceil(2);
            let launch = self
                .module
                .prepare_cache_parents(LaunchConfig1D::new(end - first, 64, 0))
                .map_err(cuda_error)?;
            self.module
                .cache_parents(
                    &self.stream,
                    &launch,
                    &mut self.tree,
                    child,
                    parent,
                    first,
                    end - first,
                )
                .map_err(cuda_error)?;
            child = parent;
            count /= 2;
            parent += count;
        }
        Ok(())
    }
}

impl layout::probe::Cache for ContextCache {
    fn write(&mut self, start: u32, values: &[u16]) -> CudaResult<f64> {
        let rows = u32::try_from(values.len() / (2 * WIDTH) as usize).map_err(|e| e.to_string())?;
        let end = start.checked_add(rows).ok_or("KV write range overflow")?;
        if values.is_empty()
            || values.len() % (2 * WIDTH) as usize != 0
            || start > self.filled
            || end > self.layout.tokens
        {
            return Err("KV write must extend or repair a contiguous in-range prefix".into());
        }
        // SAFETY: the checked contiguous range lies in this allocated buffer.
        unsafe {
            cuda_core::simt::memory::memcpy_htod_async(
                self.kv.cu_deviceptr() + u64::from(start) * u64::from(2 * WIDTH) * 2,
                values.as_ptr(),
                std::mem::size_of_val(values),
                self.stream.cu_stream(),
            )
            .map_err(cuda_error)?;
        }
        let begin = timing_event(&self.stream)?;
        self.rebuild(start / BLOCK, end.div_ceil(BLOCK))?;
        let finish = timing_event(&self.stream)?;
        self.stream.synchronize().map_err(cuda_error)?;
        self.context.check_err().map_err(cuda_error)?;
        self.filled = self.filled.max(end);
        Ok(f64::from(begin.elapsed_ms(&finish).map_err(cuda_error)?))
    }

    fn query(&mut self, start: u32, queries: &[u16]) -> CudaResult<Output> {
        let rows =
            u32::try_from(queries.len() / (HEADS * WIDTH) as usize).map_err(|e| e.to_string())?;
        self.layout.range(start, rows)?;
        if queries.len() % (HEADS * WIDTH) as usize != 0 || start + rows > self.filled {
            return Err(
                "queries must cover complete heads within the initialized KV prefix".into(),
            );
        }
        let wall = Instant::now();
        let shape = PisaShape::cached(rows, self.layout.tokens, start).map_err(str::to_string)?;
        copy_prefix(&self.queries, queries, &self.stream)?;
        let begin = timing_event(&self.stream)?;
        let launch = self
            .module
            .prepare_pisa_select(LaunchConfig1D::new(rows, 64, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_select(
                &self.stream,
                &launch,
                &self.queries,
                &self.tree,
                &mut self.blocks,
                shape,
            )
            .map_err(cuda_error)?;
        let launch = self
            .module
            .prepare_pisa_multihead(LaunchConfig1D::new(rows * 2, 256, 0))
            .map_err(cuda_error)?;
        self.module
            .pisa_multihead(
                &self.stream,
                &launch,
                &self.queries,
                &self.kv,
                &self.blocks,
                &mut self.output,
                shape,
            )
            .map_err(cuda_error)?;
        let finish = timing_event(&self.stream)?;
        let blocks = read_prefix(&self.blocks, &self.stream, (rows * SELECTED) as usize)?;
        let values = read_prefix(&self.output, &self.stream, queries.len())?;
        self.context.check_err().map_err(cuda_error)?;
        Ok(Output {
            blocks,
            values,
            device_ms: f64::from(begin.elapsed_ms(&finish).map_err(cuda_error)?),
            wall_ms: wall.elapsed().as_secs_f64() * 1000.0,
        })
    }

    fn tree(&mut self) -> CudaResult<Vec<u16>> {
        read_prefix(
            &self.tree,
            &self.stream,
            (self.layout.nodes() * WIDTH) as usize,
        )
    }
}

fn parse_arg(args: &[String], index: usize, default: u32) -> CudaResult<u32> {
    args.get(index).map_or(Ok(default), |value| {
        value.parse::<u32>().map_err(cuda_error)
    })
}

pub fn run(args: &[String]) -> CudaResult<()> {
    if args.len() > 3 {
        return Err("usage: ./baux cuda context [TOKENS=1048576] [QUERIES=128] [REPEATS=5]".into());
    }
    let layout = Layout::new(parse_arg(args, 0, 1_048_576)?, parse_arg(args, 1, 128)?)?;
    let mut cache = ContextCache::new(layout)?;
    let report = layout::probe::run(&mut cache, layout, parse_arg(args, 2, 5)?)?;
    println!("{}", baux::artifact::json::pretty_string(&baux::artifact::json::json!({
        "stage": "cuda-cached-pisa-attention", "context": layout.tokens, "queries": layout.queries,
        "kv-bytes-per-visit": layout.kv_bytes(), "tree-bytes-per-visit": layout.tree_bytes(),
        "query-work-bytes": layout.work_bytes(), "selected-token-budget": 512,
        "tree-build-ms": report.build_ms, "repair-ms": report.repair_ms,
        "median-query-device-ms": report.device_ms, "median-query-wall-ms": report.wall_ms,
        "max-abs-error": report.error, "checked-ranges": report.ranges,
        "full-model": false, "learned-capability": false
    })).map_err(|e| e.to_string())?);
    Ok(())
}
