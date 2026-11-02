use baux::semantics::optimizer::param_tree::{ParamFamily, ParamTree};
use std::io::{BufWriter, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"BAUXUPD1";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 24;
const RECORD_BYTES: usize = 80;

/// In-place online sufficient statistics for one contiguous parameter block
/// (`BAUXUPD1`).
///
/// Mirrors `llama-imatrix` moment accumulation and `Burn` structural
/// `ParamFamily` indexing: each block is identified by its `[offset .. offset +
/// elements]` span in the flat weight arena and its 17-family [`ParamFamily`]
/// tag, with zero architecture-specific string keys and zero per-round heap
/// allocations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct BlockStats {
    pub offset: u32,
    pub elements: u32,
    pub family: u8,
    pub reserved: [u8; 3],
    pub proposed_rounds: u32,
    pub accepted_rounds: u32,
    pub initial_rms: f32,
    pub final_rms: f32,
    pub last_scale: f32,
    pub scale_sum: f64,
    pub proposed_changed: u64,
    pub accepted_changed: u64,
    pub proposed_sq_change: f64,
    pub accepted_sq_change: f64,
    pub net_sq_change: f64,
}

impl BlockStats {
    pub fn proposed_rms(&self) -> f64 {
        if self.proposed_rounds == 0 || self.elements == 0 {
            return 0.0;
        }
        (self.proposed_sq_change / (f64::from(self.proposed_rounds) * f64::from(self.elements)))
            .sqrt()
    }

    pub fn accepted_rms(&self) -> f64 {
        if self.accepted_rounds == 0 || self.elements == 0 {
            return 0.0;
        }
        (self.accepted_sq_change / (f64::from(self.accepted_rounds) * f64::from(self.elements)))
            .sqrt()
    }

    fn to_bytes(self) -> [u8; RECORD_BYTES] {
        let mut out = [0u8; RECORD_BYTES];
        out[0..4].copy_from_slice(&self.offset.to_le_bytes());
        out[4..8].copy_from_slice(&self.elements.to_le_bytes());
        out[8] = self.family;
        out[9..12].copy_from_slice(&self.reserved);
        out[12..16].copy_from_slice(&self.proposed_rounds.to_le_bytes());
        out[16..20].copy_from_slice(&self.accepted_rounds.to_le_bytes());
        out[20..24].copy_from_slice(&self.initial_rms.to_le_bytes());
        out[24..28].copy_from_slice(&self.final_rms.to_le_bytes());
        out[28..32].copy_from_slice(&self.last_scale.to_le_bytes());
        out[32..40].copy_from_slice(&self.scale_sum.to_le_bytes());
        out[40..48].copy_from_slice(&self.proposed_changed.to_le_bytes());
        out[48..56].copy_from_slice(&self.accepted_changed.to_le_bytes());
        out[56..64].copy_from_slice(&self.proposed_sq_change.to_le_bytes());
        out[64..72].copy_from_slice(&self.accepted_sq_change.to_le_bytes());
        out[72..80].copy_from_slice(&self.net_sq_change.to_le_bytes());
        out
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != RECORD_BYTES {
            return Err("invalid BAUXUPD1 block record size".into());
        }
        let u32_le = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let u64_le = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        let f32_le = |off: usize| f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let f64_le = |off: usize| f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        Ok(Self {
            offset: u32_le(0),
            elements: u32_le(4),
            family: bytes[8],
            reserved: [bytes[9], bytes[10], bytes[11]],
            proposed_rounds: u32_le(12),
            accepted_rounds: u32_le(16),
            initial_rms: f32_le(20),
            final_rms: f32_le(24),
            last_scale: f32_le(28),
            scale_sum: f64_le(32),
            proposed_changed: u64_le(40),
            accepted_changed: u64_le(48),
            proposed_sq_change: f64_le(56),
            accepted_sq_change: f64_le(64),
            net_sq_change: f64_le(72),
        })
    }
}

/// Aggregated sufficient statistics folded over the 17-family [`ParamTree`]
/// algebra.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct FamilyStats {
    pub blocks: u32,
    pub elements: u64,
    pub proposed_changed: u64,
    pub accepted_changed: u64,
    pub proposed_sq_change: f64,
    pub accepted_sq_change: f64,
    pub net_sq_change: f64,
}

/// Fixed-size `O(B)` online parameter update accumulator serialized as
/// `BAUXUPD1`.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct UpdateLog {
    pub rounds: u32,
    pub accepted_rounds: u32,
    pub blocks: Vec<BlockStats>,
}

impl UpdateLog {
    pub fn push_block(
        &mut self,
        offset: usize,
        elements: usize,
        family: ParamFamily,
        initial_rms: f64,
        proposal_scale: f32,
    ) -> Result<(), String> {
        let offset = u32::try_from(offset).map_err(|_| "block offset exceeds u32")?;
        let elements = u32::try_from(elements).map_err(|_| "block element count exceeds u32")?;
        if !initial_rms.is_finite()
            || initial_rms < 0.0
            || !proposal_scale.is_finite()
            || proposal_scale < 0.0
        {
            return Err("invalid initial block statistics".into());
        }
        let rms = initial_rms as f32;
        self.blocks.push(BlockStats {
            offset,
            elements,
            family: family.index() as u8,
            reserved: [0; 3],
            proposed_rounds: 0,
            accepted_rounds: 0,
            initial_rms: rms,
            final_rms: rms,
            last_scale: proposal_scale,
            scale_sum: 0.0,
            proposed_changed: 0,
            accepted_changed: 0,
            proposed_sq_change: 0.0,
            accepted_sq_change: 0.0,
            net_sq_change: 0.0,
        });
        Ok(())
    }

    pub fn push(
        &mut self,
        index: u32,
        seed: u64,
        radius: f32,
        accepted: bool,
        changes: &[(u64, f64)],
    ) -> Result<(), String> {
        let scales: Vec<f32> = self.blocks.iter().map(|b| b.last_scale).collect();
        self.push_scaled(index, seed, radius, accepted, changes, &scales)
    }

    pub fn push_scaled(
        &mut self,
        _index: u32,
        _seed: u64,
        radius: f32,
        accepted: bool,
        changes: &[(u64, f64)],
        scales: &[f32],
    ) -> Result<(), String> {
        if changes.len() != self.blocks.len()
            || scales.len() != self.blocks.len()
            || scales.iter().any(|s| !s.is_finite() || *s < 0.0)
            || !radius.is_finite()
            || self
                .blocks
                .iter()
                .zip(changes)
                .any(|(block, &(changed, squared))| {
                    changed > u64::from(block.elements) || !squared.is_finite() || squared < 0.0
                })
        {
            return Err("Invalid per-tensor update statistics".into());
        }
        self.rounds = self.rounds.saturating_add(1);
        self.accepted_rounds = self.accepted_rounds.saturating_add(u32::from(accepted));
        for ((block, &(changed, squared)), &scale) in
            self.blocks.iter_mut().zip(changes).zip(scales)
        {
            block.last_scale = scale;
            if changed > 0 || squared > 0.0 {
                block.proposed_rounds = block.proposed_rounds.saturating_add(1);
                block.proposed_changed = block.proposed_changed.saturating_add(changed);
                block.proposed_sq_change += squared;
                block.scale_sum += f64::from(scale);
                if accepted {
                    block.accepted_rounds = block.accepted_rounds.saturating_add(1);
                    block.accepted_changed = block.accepted_changed.saturating_add(changed);
                    block.accepted_sq_change += squared;
                }
            }
        }
        Ok(())
    }

    /// Fold block-level accumulators into the 17-family [`ParamTree`] algebra.
    pub fn family_tree(&self) -> ParamTree<FamilyStats> {
        let mut tree = ParamTree::from_fn(|_| FamilyStats::default());
        for block in &self.blocks {
            if let Some(slot) = tree.values.get_mut(usize::from(block.family)) {
                slot.blocks += 1;
                slot.elements += u64::from(block.elements);
                slot.proposed_changed += block.proposed_changed;
                slot.accepted_changed += block.accepted_changed;
                slot.proposed_sq_change += block.proposed_sq_change;
                slot.accepted_sq_change += block.accepted_sq_change;
                slot.net_sq_change += block.net_sq_change;
            }
        }
        tree
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)?;
        writer.flush().map_err(|e| e.to_string())
    }

    pub fn write_to(&self, writer: &mut impl Write) -> Result<(), String> {
        let count = u32::try_from(self.blocks.len()).map_err(|_| "block count exceeds u32")?;
        let mut header = [0u8; HEADER_BYTES];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&self.rounds.to_le_bytes());
        header[16..20].copy_from_slice(&self.accepted_rounds.to_le_bytes());
        header[20..24].copy_from_slice(&count.to_le_bytes());
        writer.write_all(&header).map_err(|e| e.to_string())?;
        for block in &self.blocks {
            writer
                .write_all(&block.to_bytes())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_BYTES || &bytes[0..8] != MAGIC {
            return Err("invalid BAUXUPD1 header".into());
        }
        let u32_le = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let version = u32_le(8);
        if version != VERSION {
            return Err(format!("unsupported BAUXUPD1 version {version}"));
        }
        let rounds = u32_le(12);
        let accepted_rounds = u32_le(16);
        let count = u32_le(20) as usize;
        let expected = HEADER_BYTES
            .checked_add(
                count
                    .checked_mul(RECORD_BYTES)
                    .ok_or("BAUXUPD1 size overflow")?,
            )
            .ok_or("BAUXUPD1 size overflow")?;
        if bytes.len() != expected {
            return Err(format!(
                "BAUXUPD1 byte length {} does not match expected {expected}",
                bytes.len()
            ));
        }
        let blocks = bytes[HEADER_BYTES..]
            .chunks_exact(RECORD_BYTES)
            .map(BlockStats::from_bytes)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            rounds,
            accepted_rounds,
            blocks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_metrics() {
        let mut log = UpdateLog::default();
        log.push_block(0, 4, ParamFamily::RoutedDown, 2.0, 2.0)
            .unwrap();
        // Realized changes [0, 0, 0.5, -0.5] -> changed = 2, squared = 0.5.
        log.push(1, 17, 0.25, false, &[(2, 0.5)]).unwrap();
        log.push(2, 18, 0.25, true, &[(0, 0.0)]).unwrap();
        log.push(3, 19, 0.25, true, &[(2, 0.5)]).unwrap();

        let mut bytes = Vec::new();
        log.write_to(&mut bytes).unwrap();
        assert_eq!(bytes.len(), HEADER_BYTES + RECORD_BYTES);

        let decoded = UpdateLog::decode(&bytes).unwrap();
        assert_eq!(decoded, log);
        assert_eq!(decoded.rounds, 3);
        assert_eq!(decoded.accepted_rounds, 2);

        let block = &decoded.blocks[0];
        assert_eq!(block.proposed_rounds, 2);
        assert_eq!(block.accepted_rounds, 1);
        assert_eq!(block.proposed_changed, 4);
        assert_eq!(block.accepted_changed, 2);
        assert!((block.proposed_sq_change - 1.0).abs() < 1e-12);
        assert!((block.accepted_sq_change - 0.5).abs() < 1e-12);
        assert!((block.proposed_rms() - 0.125f64.sqrt()).abs() < 1e-12);
        assert!((block.accepted_rms() - 0.125f64.sqrt()).abs() < 1e-12);

        let tree = decoded.family_tree();
        assert_eq!(tree.get(ParamFamily::RoutedDown).blocks, 1);
        assert_eq!(tree.get(ParamFamily::RoutedDown).accepted_changed, 2);
        assert_eq!(tree.get(ParamFamily::TokenEmbed).blocks, 0);

        assert!(log.push(4, 20, 0.25, false, &[(5, 0.0)]).is_err());
    }
}
