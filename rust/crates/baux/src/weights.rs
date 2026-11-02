#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionKind {
    Ucb,
    /// One sampled function with shared observation noise, as in ENN posterior
    /// draws.
    Thompson,
    /// Legacy scalar mean-plus-uncertainty heuristic for materialized weight
    /// selection.
    ///
    /// This is intentionally distinct from
    /// `crate::acquisition::ParetoAcquisition`, which performs
    /// multiobjective Pareto-front selection.
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
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "unknown compute device {other:?}; expected 'auto', 'cpu', 'metal', or 'cuda'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    Int4 = 0,
    Int8 = 1,
    Fp4E2M1 = 2,
    Fp8E4M3 = 3,
    Fp8E5M2 = 4,
}

impl EncodingType {
    pub fn parse(bits: u8, mode: Option<&str>) -> Result<Self, String> {
        match (bits, mode.map(|s| s.trim().to_ascii_lowercase()).as_deref()) {
            (4, None | Some("int") | Some("int4")) => Ok(Self::Int4),
            (8, None | Some("int") | Some("int8")) => Ok(Self::Int8),
            (4, Some("fp4") | Some("fp4_e2m1") | Some("e2m1")) => Ok(Self::Fp4E2M1),
            (8, Some("fp8") | Some("fp8_e4m3") | Some("e4m3")) => Ok(Self::Fp8E4M3),
            (8, Some("fp8_e5m2") | Some("e5m2")) => Ok(Self::Fp8E5M2),
            _ => Err(format!("unsupported encoding bits={bits}, mode={mode:?}")),
        }
    }
}

#[cfg(any(
    all(target_os = "linux", target_arch = "x86_64", feature = "cuda"),
    all(target_os = "macos", feature = "metal")
))]
pub fn acquisition_code(acquisition: AcquisitionKind) -> u32 {
    match acquisition {
        AcquisitionKind::Ucb => 0,
        AcquisitionKind::Thompson => 1,
        AcquisitionKind::Pareto => 2,
    }
}
