//! Production FBT native-kernel integration with TVM Relax. Existing Metal
//! kernels remain the implementation; TVM owns invocation and typed returns.
//! Metal allocations and optimizer mutation leases cross via DLPack, not CPU.
use super::*;
use baux::portable::{TvmExecutable, TvmTensor};

fn buffers(inputs: &[TvmTensor]) -> Result<Vec<metal::Buffer>, String> {
    inputs
        .iter()
        .map(|tensor| {
            let (pointer, offset) = unsafe { tensor.metal()? };
            if offset != 0 {
                return Err("FBT requires whole Metal allocations".into());
            }
            // The tensor retains the producer and mutation lease through the
            // call.
            Ok(unsafe { metal::BufferRef::from_ptr(pointer.cast()) }.to_owned())
        })
        .collect()
}

pub(super) fn forward(
    state: Arc<Mutex<State>>,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
) -> Result<Executable, String> {
    let compiled =
        TvmExecutable::native(inputs.clone(), outputs.clone(), "metal", move |inputs| {
            metal::objc::rc::autoreleasepool(|| {
                let buffers = buffers(inputs)?;
                let mut state = state.lock().map_err(|_| "FBT scorer mutex poisoned")?;
                state.execute_on(&buffers)?;
                state
                    .outputs()?
                    .into_iter()
                    .map(|(buffer, count)| {
                        let pointer = buffer.as_ptr().cast();
                        // outputs() completed its copy and retains no writable
                        // alias.
                        unsafe {
                            TvmTensor::import_metal(
                                tensor(count, "float32")?,
                                pointer,
                                0,
                                Arc::new(buffer),
                            )
                        }
                    })
                    .collect()
            })
        })?;
    Executable::native("metal", inputs, outputs, move |program, inputs| {
        let tensors = inputs
            .iter()
            .map(|input| {
                let (pointer, offset) = unsafe { input.metal()? };
                // Keep the complete portable input, including its optimizer
                // lease, alive until TVM has released all
                // tensor aliases.
                unsafe {
                    TvmTensor::import_metal(
                        input.spec().clone(),
                        pointer,
                        offset,
                        Arc::new(input.clone()),
                    )
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        let result = compiled.run(&tensors.iter().collect::<Vec<_>>())?;
        result
            .into_iter()
            .map(|value| {
                let (pointer, offset) = unsafe { value.metal()? };
                unsafe {
                    program.import_metal(
                        value.spec().clone(),
                        pointer,
                        usize::try_from(offset).map_err(|e| e.to_string())?,
                        Arc::new(value),
                    )
                }
            })
            .collect()
    })
}
