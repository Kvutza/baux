use cuda_core::DeviceCopy;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NeighborShape {
    pub samples: u32,
    pub history: u32,
    pub candidates: u32,
}

// SAFETY: repr(C) and scalar fields only.
unsafe impl DeviceCopy for NeighborShape {}
