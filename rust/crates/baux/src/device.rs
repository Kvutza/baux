//! Unified execution device abstraction (Candle-style).

use crate::capability::Backend;
use crate::index::IndexDriver;
use crate::weights::ComputeDevice;
use deser::{Deserialize, Serialize};

/// Execution device where tensors, parameters, and surrogates reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[deser(rename_all = "snake_case")]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
    OpenCl,
}

/// Device memory capacity and allocation snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMemory {
    pub name: String,
    pub has_unified_memory: bool,
    pub recommended_working_set_bytes: u64,
    pub allocated_bytes: u64,
    pub max_buffer_bytes: u64,
}

impl Device {
    pub const fn cpu() -> Self {
        Self::Cpu
    }

    pub fn metal() -> Result<Self, String> {
        if Backend::Metal.compiled() {
            Ok(Self::Metal)
        } else {
            Err("metal backend was not compiled into this library".into())
        }
    }

    pub fn cuda() -> Result<Self, String> {
        if Backend::Cuda.compiled() {
            Ok(Self::Cuda)
        } else {
            Err("cuda backend was not compiled into this library".into())
        }
    }

    pub const fn auto() -> Self {
        if Backend::Metal.compiled() {
            Self::Metal
        } else if Backend::Cuda.compiled() {
            Self::Cuda
        } else {
            Self::Cpu
        }
    }

    pub fn parse(name: &str) -> Result<Self, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::auto()),
            "cpu" | "exact" | "flat" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" | "cuvs" | "cuvs_cagra" => Ok(Self::Cuda),
            "opencl" | "ocl" => Ok(Self::OpenCl),
            other => Err(format!(
                "unknown device {other:?}; expected 'auto', 'cpu', 'metal', 'cuda', or 'opencl'"
            )),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::OpenCl => "opencl",
        }
    }

    pub const fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub const fn is_metal(self) -> bool {
        matches!(self, Self::Metal)
    }

    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda)
    }

    pub const fn compiled(self) -> bool {
        self.backend().compiled()
    }

    pub const fn backend(self) -> Backend {
        match self {
            Self::Cpu => Backend::Cpu,
            Self::Metal => Backend::Metal,
            Self::Cuda => Backend::Cuda,
            Self::OpenCl => Backend::OpenCl,
        }
    }

    pub const fn index_driver(self) -> IndexDriver {
        match self {
            Self::Cpu => IndexDriver::Exact,
            Self::Metal => IndexDriver::Metal,
            Self::Cuda => IndexDriver::Cuda,
            Self::OpenCl => IndexDriver::OpenCl,
        }
    }

    pub fn probe(self) -> Result<(), String> {
        self.backend().probe()
    }

    pub fn memory_info(self) -> Result<DeviceMemory, String> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal => {
                let info = crate::resident_metal::memory_info()?;
                Ok(DeviceMemory {
                    name: info.name,
                    has_unified_memory: info.has_unified_memory,
                    recommended_working_set_bytes: info.recommended_max_working_set_size,
                    allocated_bytes: info.current_allocated_size,
                    max_buffer_bytes: info.max_buffer_length,
                })
            }
            _ => Err("memory_info is only supported on active accelerator devices".into()),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

impl From<Backend> for Device {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Cpu => Self::Cpu,
            Backend::Metal => Self::Metal,
            Backend::Cuda => Self::Cuda,
            Backend::OpenCl => Self::OpenCl,
        }
    }
}

impl From<Device> for Backend {
    fn from(device: Device) -> Self {
        device.backend()
    }
}

impl From<Device> for IndexDriver {
    fn from(device: Device) -> Self {
        device.index_driver()
    }
}

impl From<ComputeDevice> for Device {
    fn from(device: ComputeDevice) -> Self {
        match device {
            ComputeDevice::Auto => Self::auto(),
            ComputeDevice::Cpu => Self::Cpu,
            ComputeDevice::Metal => Self::Metal,
            ComputeDevice::Cuda => Self::Cuda,
            ComputeDevice::OpenCl => Self::OpenCl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parse() {
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("EXACT").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("metal").unwrap(), Device::Metal);
        assert_eq!(Device::parse("cuvs_cagra").unwrap(), Device::Cuda);
        assert_eq!(Device::parse("ocl").unwrap(), Device::OpenCl);
        assert_eq!(Device::parse("auto").unwrap(), Device::auto());
        assert!(Device::parse("unknown").is_err());
    }

    #[test]
    fn device_conversions() {
        assert_eq!(Device::default(), Device::Cpu);
        assert!(Device::cpu().is_cpu());
        assert!(Device::Metal.is_metal());
        assert!(Device::Cuda.is_cuda());
        assert_eq!(Backend::from(Device::Metal), Backend::Metal);
        assert_eq!(Device::from(Backend::Cuda), Device::Cuda);
        assert_eq!(IndexDriver::from(Device::Cpu), IndexDriver::Exact);
        assert_eq!(Device::from(ComputeDevice::Auto), Device::auto());
    }
}
