#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionKind {
    Ucb,
    Thompson,
    Pareto,
}

impl AcquisitionKind {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "ucb" => Ok(Self::Ucb),
            "thompson" => Ok(Self::Thompson),
            "pareto" => Ok(Self::Pareto),
            other => Err(format!(
                "unknown acquisition {other:?}; expected 'ucb', 'thompson', or 'pareto'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Auto,
    Cpu,
    Metal,
    OpenCl,
    Cuda,
}

impl ComputeDevice {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "opencl" | "ocl" => Ok(Self::OpenCl),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown compute device {other:?}; expected 'auto', 'cpu', 'metal', 'opencl', or \
                 'cuda'"
            )),
        }
    }
}

/// Noise for observation rows, independent of candidate ordering and batching.
pub(crate) fn thompson_draws(count: usize, seed: u64) -> Vec<f32> {
    (0..count)
        .map(|row| crate::hash::normal_metric(seed, row as i64, 0) as f32)
        .collect()
}

/// Resident histories can reorder rows; their storage slots identify
/// observations.
pub(crate) fn history_draws(history: &[(usize, f32)], seed: u64) -> Vec<f32> {
    history
        .iter()
        .map(|&(slot, _)| crate::hash::normal_metric(seed, slot as i64, 0) as f32)
        .collect()
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
    all(target_os = "macos", feature = "metal"),
    feature = "opencl"
))]
pub(crate) fn acquisition_code(acquisition: AcquisitionKind) -> u32 {
    match acquisition {
        AcquisitionKind::Ucb => 0,
        AcquisitionKind::Thompson => 1,
        AcquisitionKind::Pareto => 2,
    }
}
