use crate::{CudaResult, cuda_error};
use cuda_core::{CudaContext, CudaStream, IntoResult, sys};
use std::mem::MaybeUninit;
use std::sync::Arc;

const RELAXED: sys::CUstreamCaptureMode = 2;

/// An instantiated CUDA Graph whose recorded pointers remain owned by the
/// model. Inputs may change in-place between replays; allocations may not.
pub(crate) struct CudaGraph {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
}

impl CudaGraph {
    pub(crate) fn capture(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        record: impl FnOnce() -> CudaResult<()>,
    ) -> CudaResult<Self> {
        context.bind_to_thread().map_err(cuda_error)?;
        unsafe {
            sys::cuStreamBeginCapture_v2(stream.cu_stream(), RELAXED)
                .result()
                .map_err(cuda_error)?;
        }
        let recorded = record();
        let captured = unsafe {
            let mut graph = MaybeUninit::<sys::CUgraph>::uninit();
            sys::cuStreamEndCapture(stream.cu_stream(), graph.as_mut_ptr())
                .result()
                .map(|()| graph.assume_init())
        };
        let graph = match (recorded, captured) {
            (Ok(()), Ok(graph)) if !graph.is_null() => graph,
            (Err(error), Ok(graph)) => {
                destroy_graph(graph);
                return Err(error);
            }
            (Err(error), Err(_)) => return Err(error),
            (Ok(()), Err(error)) => return Err(cuda_error(error)),
            (Ok(()), Ok(_)) => return Err("CUDA graph capture returned a null graph".into()),
        };
        let executable = unsafe {
            let mut executable = MaybeUninit::<sys::CUgraphExec>::uninit();
            if let Err(error) =
                sys::cuGraphInstantiateWithFlags(executable.as_mut_ptr(), graph, 0).result()
            {
                destroy_graph(graph);
                return Err(cuda_error(error));
            }
            executable.assume_init()
        };
        let result = Self {
            context,
            stream,
            graph,
            executable,
        };
        unsafe {
            sys::cuGraphUpload(result.executable, result.stream.cu_stream())
                .result()
                .map_err(cuda_error)?;
        }
        result.stream.synchronize().map_err(cuda_error)?;
        Ok(result)
    }

    pub(crate) fn launch(&self) -> CudaResult<()> {
        self.context.bind_to_thread().map_err(cuda_error)?;
        unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()) }
            .result()
            .map_err(cuda_error)
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if !self.executable.is_null() {
            let _ = unsafe { sys::cuGraphExecDestroy(self.executable) }.result();
        }
        destroy_graph(self.graph);
    }
}

fn destroy_graph(graph: sys::CUgraph) {
    if !graph.is_null() {
        let _ = unsafe { sys::cuGraphDestroy(graph) }.result();
    }
}
