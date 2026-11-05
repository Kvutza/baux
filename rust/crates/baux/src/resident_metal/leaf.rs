#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leaf {
    pub key: u64,
    pub offset: u64,
    pub length: u64,
    pub scale: f32,
    pub weight: f32,
    pub address: u32,
    pub pad: u32,
}
