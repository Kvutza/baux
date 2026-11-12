//! Hardened binary checkpoint primitives with bounded allocation and BLAKE3
//! authentication.

use ndarray::{Array1, Array2};

use crate::error::ENNError;

pub(crate) const FREEZE_MAGIC: &[u8; 8] = b"BAUXFRZ2";
pub(crate) const SNAPSHOT_VERSION: u64 = 2;

pub(crate) const MAX_BYTES: usize = 512 * 1024 * 1024;
pub(crate) const MAX_DIM: usize = 4_000_000;
pub(crate) const MAX_OBS: usize = 2_000_000;
pub(crate) const MAX_METRICS: usize = 4_096;
pub(crate) const MAX_REGIONS: usize = 4_096;
pub(crate) const MAX_PENDING: usize = 1_000_000;
pub(crate) const MAX_SOBOL: u64 = 64_000_000;
pub(crate) const MAX_TEXT: usize = 4_096;

pub(crate) fn bad_state(msg: impl Into<String>) -> ENNError {
    ENNError::InvalidParameter(msg.into())
}

#[inline]
pub(crate) fn write_header(buf: &mut Vec<u8>) {
    buf.extend_from_slice(FREEZE_MAGIC);
    write_word(buf, SNAPSHOT_VERSION);
}

#[inline]
pub(crate) fn seal_frame(buf: &mut Vec<u8>) {
    let tag = crate::artifact::blake3::hash(buf);
    buf.extend_from_slice(&tag);
}

#[inline]
pub(crate) fn write_byte(buf: &mut Vec<u8>, val: u8) {
    buf.push(val);
}

#[inline]
pub(crate) fn write_bool(buf: &mut Vec<u8>, val: bool) {
    buf.push(u8::from(val));
}

#[inline]
pub(crate) fn write_int(buf: &mut Vec<u8>, val: i32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub(crate) fn write_uint(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub(crate) fn write_word(buf: &mut Vec<u8>, val: u64) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub(crate) fn write_quad(buf: &mut Vec<u8>, val: u128) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline]
pub(crate) fn write_len(buf: &mut Vec<u8>, val: usize) {
    buf.extend_from_slice(&(val as u64).to_le_bytes());
}

#[inline]
pub(crate) fn write_real(buf: &mut Vec<u8>, val: f64) {
    buf.extend_from_slice(&val.to_bits().to_le_bytes());
}

pub(crate) fn write_slice(buf: &mut Vec<u8>, vals: &[f64]) {
    write_len(buf, vals.len());
    for &v in vals {
        write_real(buf, v);
    }
}

pub(crate) fn write_arr(buf: &mut Vec<u8>, arr: &Array1<f64>) {
    write_len(buf, arr.len());
    for &v in arr.iter() {
        write_real(buf, v);
    }
}

pub(crate) fn write_mat(buf: &mut Vec<u8>, mat: &Array2<f64>) {
    write_len(buf, mat.nrows());
    write_len(buf, mat.ncols());
    for row in mat.rows() {
        for &v in row.iter() {
            write_real(buf, v);
        }
    }
}

pub(crate) fn write_str(buf: &mut Vec<u8>, text: &str) {
    let raw = text.as_bytes();
    write_len(buf, raw.len());
    buf.extend_from_slice(raw);
}

pub(crate) struct WireReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WireReader<'a> {
    pub(crate) fn open_frame(bytes: &'a [u8]) -> Result<Self, ENNError> {
        if bytes.len() > MAX_BYTES {
            return Err(bad_state("frozen state exceeds maximum allowed byte size"));
        }
        if bytes.len() < 48 {
            return Err(bad_state("frozen state buffer too short"));
        }
        let (payload, tag) = bytes.split_at(bytes.len() - 32);
        if &payload[..8] != FREEZE_MAGIC {
            return Err(bad_state("invalid or unsupported frozen state magic"));
        }
        if crate::artifact::blake3::hash(payload).as_slice() != tag {
            return Err(bad_state("frozen optimizer BLAKE3 integrity check failed"));
        }
        let mut reader = Self {
            bytes: payload,
            pos: 8,
        };
        let version = reader.read_word()?;
        if version != SNAPSHOT_VERSION {
            return Err(bad_state("unsupported frozen state wire version"));
        }
        Ok(reader)
    }

    pub(crate) fn from_slice(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn finish(&self) -> Result<(), ENNError> {
        if self.pos != self.bytes.len() {
            return Err(bad_state("trailing bytes in frozen state payload"));
        }
        Ok(())
    }

    pub(crate) fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], ENNError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| bad_state("frozen state read offset overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| bad_state("truncated frozen state payload"))?;
        self.pos = end;
        Ok(slice)
    }

    pub(crate) fn read_byte(&mut self) -> Result<u8, ENNError> {
        Ok(self.read_bytes(1)?[0])
    }

    pub(crate) fn read_bool(&mut self) -> Result<bool, ENNError> {
        match self.read_byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad_state("invalid boolean tag in frozen state")),
        }
    }

    pub(crate) fn read_int(&mut self) -> Result<i32, ENNError> {
        let raw = self.read_bytes(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(raw);
        Ok(i32::from_le_bytes(arr))
    }

    pub(crate) fn read_uint(&mut self) -> Result<u32, ENNError> {
        let raw = self.read_bytes(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(raw);
        Ok(u32::from_le_bytes(arr))
    }

    pub(crate) fn read_word(&mut self) -> Result<u64, ENNError> {
        let raw = self.read_bytes(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(raw);
        Ok(u64::from_le_bytes(arr))
    }

    pub(crate) fn read_quad(&mut self) -> Result<u128, ENNError> {
        let raw = self.read_bytes(16)?;
        let mut arr = [0u8; 16];
        arr.copy_from_slice(raw);
        Ok(u128::from_le_bytes(arr))
    }

    pub(crate) fn read_len(&mut self, max_len: usize) -> Result<usize, ENNError> {
        let raw = self.read_word()?;
        let val =
            usize::try_from(raw).map_err(|_| bad_state("frozen length exceeds platform usize"))?;
        if val > max_len {
            return Err(bad_state("frozen length exceeds safety limit"));
        }
        Ok(val)
    }

    pub(crate) fn read_real(&mut self) -> Result<f64, ENNError> {
        Ok(f64::from_bits(self.read_word()?))
    }

    pub(crate) fn read_finite(&mut self) -> Result<f64, ENNError> {
        let val = self.read_real()?;
        if !val.is_finite() {
            return Err(bad_state("non-finite float in frozen state"));
        }
        Ok(val)
    }

    pub(crate) fn read_vec(
        &mut self,
        max_len: usize,
        require_finite: bool,
    ) -> Result<Vec<f64>, ENNError> {
        let len = self.read_len(max_len)?;
        let byte_len = len
            .checked_mul(8)
            .ok_or_else(|| bad_state("frozen vector byte size overflow"))?;
        let raw = self.read_bytes(byte_len)?;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| bad_state("frozen vector allocation failed"))?;
        for chunk in raw.chunks_exact(8) {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            let val = f64::from_bits(u64::from_le_bytes(arr));
            if require_finite && !val.is_finite() {
                return Err(bad_state("non-finite element in frozen vector"));
            }
            out.push(val);
        }
        Ok(out)
    }

    pub(crate) fn read_arr(
        &mut self,
        expected_len: usize,
        require_finite: bool,
    ) -> Result<Array1<f64>, ENNError> {
        let vals = self.read_vec(expected_len, require_finite)?;
        if vals.len() != expected_len {
            return Err(bad_state("frozen array length mismatch"));
        }
        Ok(Array1::from_vec(vals))
    }

    pub(crate) fn read_mat(
        &mut self,
        max_rows: usize,
        max_cols: usize,
        require_finite: bool,
    ) -> Result<Array2<f64>, ENNError> {
        let rows = self.read_len(max_rows)?;
        let cols = self.read_len(max_cols)?;
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| bad_state("frozen matrix element count overflow"))?;
        let byte_len = total
            .checked_mul(8)
            .ok_or_else(|| bad_state("frozen matrix byte size overflow"))?;
        let raw = self.read_bytes(byte_len)?;
        let mut vals = Vec::new();
        vals.try_reserve_exact(total)
            .map_err(|_| bad_state("frozen matrix allocation failed"))?;
        for chunk in raw.chunks_exact(8) {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            let val = f64::from_bits(u64::from_le_bytes(arr));
            if require_finite && !val.is_finite() {
                return Err(bad_state("non-finite element in frozen matrix"));
            }
            vals.push(val);
        }
        Array2::from_shape_vec((rows, cols), vals).map_err(|e| bad_state(e.to_string()))
    }

    pub(crate) fn read_str(&mut self, max_len: usize) -> Result<String, ENNError> {
        let len = self.read_len(max_len)?;
        let raw = self.read_bytes(len)?;
        let s =
            std::str::from_utf8(raw).map_err(|_| bad_state("invalid UTF-8 in frozen string"))?;
        Ok(s.to_owned())
    }
}
