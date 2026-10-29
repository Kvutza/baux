pub use crate::weights::{EncodingType, decode_code};
#[cfg(test)]
pub use crate::weights::{FP4_LUT, decode_e4m3, decode_e5m2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameter {
    pub offset: usize,
    pub length: usize,
    pub bits: u8,
    pub encoding: EncodingType,
    pub scale: f32,
    pub weight: f32,
    pub radius: f32,
}

impl Parameter {
    pub fn new(
        offset: usize,
        length: usize,
        bits: u8,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        Self::encoded(
            offset,
            length,
            bits,
            EncodingType::parse(bits, None)?,
            scale,
            weight,
            radius,
        )
    }

    pub fn encoded(
        offset: usize,
        length: usize,
        bits: u8,
        encoding: EncodingType,
        scale: f32,
        weight: f32,
        radius: f32,
    ) -> Result<Self, String> {
        if length == 0 {
            return Err("leaf length must be positive".to_string());
        }
        if bits != 4 && bits != 8 {
            return Err(format!("leaf bits must be 4 or 8, got {bits}"));
        }
        let encoded_bits = match encoding {
            EncodingType::Int4 | EncodingType::Fp4E2M1 => 4,
            _ => 8,
        };
        if bits != encoded_bits {
            return Err("encoding does not match parameter bit width".into());
        }
        offset
            .checked_add(length)
            .ok_or("parameter range overflow")?;
        for (name, value) in [("scale", scale), ("weight", weight), ("radius", radius)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        Ok(Self {
            offset,
            length,
            bits,
            encoding,
            scale,
            weight,
            radius,
        })
    }

    pub(super) fn bytes(self) -> usize {
        match self.bits {
            4 => self.length.div_ceil(2),
            8 => self.length,
            _ => unreachable!("leaf width is checked at construction"),
        }
    }
}
