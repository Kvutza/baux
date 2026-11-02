use std::error::Error;
use std::io;

use baux_cuda::{device_target, embed_parity, feedback_parity, target_name};
use baux_cuda_device::trials;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, simt::LaunchConfig};

pub(crate) type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Leaf {
    pub(crate) offset: usize,
    pub(crate) length: usize,
    pub(crate) bits: u8,
    pub(crate) scale: f32,
    pub(crate) radius: f32,
}

impl Leaf {
    pub(crate) fn new(
        offset: usize,
        length: usize,
        bits: u8,
        scale: f32,
        radius: f32,
    ) -> AppResult<Self> {
        if length == 0 {
            return Err(io::Error::other("leaf length must be positive").into());
        }
        if bits != 4 && bits != 8 {
            return Err(io::Error::other("leaf bits must be 4 or 8").into());
        }
        if !scale.is_finite() || scale <= 0.0 || !radius.is_finite() || radius <= 0.0 {
            return Err(io::Error::other("leaf scale and radius must be positive").into());
        }
        Ok(Self {
            offset,
            length,
            bits,
            scale,
            radius,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LaunchLeaf {
    pub(crate) byte_offset: u32,
    pub(crate) element_offset: u32,
    pub(crate) length: u32,
    pub(crate) bits: u32,
    pub(crate) whole: u32,
    pub(crate) threshold: u32,
}

impl LaunchLeaf {
    pub(crate) fn new(leaf: Leaf, byte_offset: usize, length: f32) -> AppResult<Self> {
        let max_code = (1_u32 << leaf.bits) - 1;
        let amplitude = (length * leaf.radius / leaf.scale).clamp(0.0, max_code as f32);
        let whole = amplitude.floor() as u32;
        let threshold = if whole == max_code {
            0
        } else {
            ((amplitude - whole as f32) * (u32::MAX as f32)) as u32
        };
        Ok(Self {
            byte_offset: byte_offset.try_into()?,
            element_offset: leaf.offset.try_into()?,
            length: leaf.length.try_into()?,
            bits: u32::from(leaf.bits),
            whole,
            threshold,
        })
    }

    pub(crate) fn work_items(self) -> u32 {
        if self.bits == 4 {
            self.length.div_ceil(2)
        } else {
            self.length
        }
    }

    pub(crate) fn row_bytes(self) -> usize {
        self.work_items() as usize
    }
}

pub(crate) fn check_case(
    context: &CudaContext,
    stream: &CudaStream,
    module: &trials::LoadedModule,
    base: &[u8],
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let base_device = DeviceBuffer::from_host(stream, base)?;
    let mut output_device = DeviceBuffer::<u8>::zeroed(stream, base.len())?;
    launch_materialize(
        module,
        stream,
        &base_device,
        &mut output_device,
        leaves,
        length,
        seed,
    )?;
    let output = output_device.to_host_vec(stream)?;
    let expected = expected_row(base, leaves, length, seed)?;
    compare_rows(&expected, &output, seed, length)?;
    context.check_err()?;
    Ok(())
}

pub(crate) fn launch_materialize(
    module: &trials::LoadedModule,
    stream: &CudaStream,
    base: &DeviceBuffer<u8>,
    output: &mut DeviceBuffer<u8>,
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<()> {
    let mut byte_offset = 0;
    for &leaf in leaves {
        let launch = LaunchLeaf::new(leaf, byte_offset, length)?;
        // SAFETY: there is one thread per leaf byte, leaf byte ranges do not
        // overlap, and the input and output buffers contain the
        // complete encoded row.
        unsafe {
            module.materialize(
                stream,
                LaunchConfig::for_num_elems(launch.work_items()),
                base,
                output,
                launch.byte_offset,
                launch.element_offset,
                launch.length,
                launch.bits,
                seed as u32,
                (seed >> 32) as u32,
                launch.whole,
                launch.threshold,
            )
        }?;
        byte_offset += launch.row_bytes();
    }
    Ok(())
}

pub(crate) fn expected_row(
    base: &[u8],
    leaves: &[Leaf],
    length: f32,
    seed: u64,
) -> AppResult<Vec<u8>> {
    let mut row = vec![0_u8; base.len()];
    let mut byte_offset = 0;
    for &leaf in leaves {
        let step = LaunchLeaf::new(leaf, byte_offset, length)?;
        for element in 0..leaf.length {
            let byte = byte_offset + if leaf.bits == 4 { element / 2 } else { element };
            let shift = if leaf.bits == 4 { (element & 1) * 4 } else { 0 };
            let mask = if leaf.bits == 4 { 0x0f } else { 0xff };
            let code = u32::from((base[byte] >> shift) & mask);
            let value = cpu_perturb(
                code,
                seed,
                leaf.offset as u32 + element as u32,
                step.bits,
                step.whole,
                step.threshold,
            );
            row[byte] |= (value as u8) << shift;
        }
        byte_offset += step.row_bytes();
    }
    Ok(row)
}

fn cpu_perturb(code: u32, seed: u64, element: u32, bits: u32, whole: u32, threshold: u32) -> u32 {
    let random = cpu_hash(seed, element);
    let amount = whole + u32::from((random >> 1) < (threshold >> 1));
    if amount == 0 {
        return code;
    }
    let max_code = (1_u32 << bits) - 1;
    if random & 1 == 0 {
        if code >= amount {
            code - amount
        } else {
            (code + amount).min(max_code)
        }
    } else if code + amount <= max_code {
        code + amount
    } else {
        code.saturating_sub(amount)
    }
}

fn cpu_hash(seed: u64, element: u32) -> u32 {
    let mut value = (seed as u32) ^ element.wrapping_mul(0x9e37_79b9);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= (seed >> 32) as u32;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 15)
}

pub(crate) fn compare_rows(
    expected: &[u8],
    actual: &[u8],
    seed: u64,
    length: f32,
) -> AppResult<()> {
    if expected == actual {
        return Ok(());
    }
    let mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .unwrap_or(expected.len().min(actual.len()));
    Err(io::Error::other(format!(
        "CUDA parity failed at byte {mismatch} for seed={seed} length={length}: \
         expected={:?} actual={:?}",
        expected.get(mismatch),
        actual.get(mismatch)
    ))
    .into())
}

pub(crate) fn row_bytes(leaves: &[Leaf]) -> usize {
    leaves
        .iter()
        .map(|leaf| {
            if leaf.bits == 4 {
                leaf.length.div_ceil(2)
            } else {
                leaf.length
            }
        })
        .sum()
}

pub(crate) fn make_base(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
        .collect()
}

pub(crate) fn parse_arg<T>(args: &[String], index: usize, default: T, name: &str) -> AppResult<T>
where
    T: std::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    args.get(index)
        .map(|value| value.parse().map_err(Into::into))
        .unwrap_or(Ok(default))
        .map_err(|error: Box<dyn Error + Send + Sync>| {
            io::Error::other(format!("invalid {name}: {error}")).into()
        })
}

pub(super) fn parity() -> AppResult<()> {
    baux_cuda::proposal_check().map_err(io::Error::other)?;
    baux_cuda::neighbors::parity().map_err(io::Error::other)?;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: the generated bindings load the matching embedded kernel
    // artifact.
    let module = unsafe { trials::load(&context) }?;
    let layouts = [
        vec![Leaf::new(0, 257, 4, 0.125, 0.75)?],
        vec![Leaf::new(0, 259, 8, 0.03125, 0.5)?],
        vec![
            Leaf::new(0, 257, 4, 0.125, 0.75)?,
            Leaf::new(257, 259, 8, 0.03125, 0.5)?,
        ],
    ];
    let mut cases = 0;
    for leaves in layouts {
        let base = make_base(row_bytes(&leaves));
        for length in [0.0, 0.8, 8.0] {
            for seed in [0_u64, 1, 0x0123_4567_89ab_cdef, u64::MAX] {
                check_case(&context, &stream, &module, &base, &leaves, length, seed)?;
                cases += 1;
            }
        }
    }
    embed_parity().map_err(io::Error::other)?;
    feedback_parity().map_err(io::Error::other)?;
    let target = target_name(&context).map_err(io::Error::other)?;
    println!("PARITY ok=true cases={cases} target={target}");
    Ok(())
}
