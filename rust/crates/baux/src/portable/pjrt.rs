//! Public PJRT C API execution with the same resident ownership contract.
use super::*;

// Public schemas at OpenXLA d867969800d4a8fd045505eabc65e1e66f30794f:
// xla/pjrt/proto/compile_options.proto and xla/xla_data.proto.
// Input ShapeProtos make XLA validate declared types and shapes at compilation;
// execution alone only checks allocation size, allowing equal-width type
// errors.
fn varint(mut value: u64, output: &mut Vec<u8>) {
    while value >= 128 {
        output.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    output.push(value as u8);
}

fn options(inputs: &[Tensor]) -> Result<Vec<u8>> {
    // Field 3: build options, with one replica (4) and one partition (5).
    let mut output = vec![0x1a, 4, 0x20, 1, 0x28, 1];
    const TYPES: [u8; 9] = [1, 4, 5, 8, 9, 10, 16, 11, 12];
    for input in inputs {
        let kind = layout(input)?.0 as usize;
        let mut shape = vec![0x10, TYPES[kind]]; // ShapeProto.element_type (2).
        for &dimension in &input.shape {
            let dimension = i64::try_from(dimension).map_err(|_| "dimension exceeds int64")?;
            shape.push(0x18); // ShapeProto.dimensions (3), unpacked varint.
            varint(dimension as u64, &mut shape);
        }
        output.push(0x0a); // CompileOptionsProto.argument_layouts (1).
        varint(shape.len() as u64, &mut output);
        output.extend(shape);
    }
    Ok(output)
}

impl Executable {
    /// Compile StableHLO with an entry named `main` using native XLA on CPU.
    /// Inputs are never donated; outputs remain valid across later calls.
    pub fn compile_xla(
        source: &str,
        device: &str,
        inputs: Vec<Tensor>,
        outputs: Vec<Tensor>,
    ) -> Result<Self> {
        for spec in inputs.iter().chain(&outputs) {
            layout(spec)?;
        }
        let device = CString::new(device).map_err(|e| e.to_string())?;
        let options = options(&inputs)?;
        let bridge = raw::pjrt()?;
        let mut pointer = std::ptr::null_mut();
        let mut error = [0; 2048];
        // Native compilation finishes before returning. No borrowed source
        // or device pointer survives this call; failures clean up the session.
        let code = unsafe {
            (bridge.compile)(
                source.as_ptr(),
                source.len(),
                device.as_ptr(),
                options.as_ptr(),
                options.len(),
                inputs.len(),
                &mut pointer,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        raw::status(code, &error)?;
        Ok(Self {
            owner: Arc::new(Session(Mutex::new(raw::Context(pointer, &bridge.api)))),
            inputs: inputs.into(),
            outputs: outputs.into(),
            queue: pending::Queue::new()?,
            kernel: None,
        })
    }
}
