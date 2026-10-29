use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RouteStats {
    pub(crate) active_experts: usize,
    pub(crate) routed_rows: usize,
    pub(crate) routed_tiles: usize,
    /// Bit `i` identifies routed expert tensor `i + 1`; tensor zero is shared.
    pub(crate) expert_mask: [u64; ROUTE_WORDS],
}

impl FineGrainedMoeBuffers {
    /// Counts describe the final site; the mask is the union over every MoE
    /// site in the completed forward pass.
    pub(crate) fn route_stats(&self) -> RouteStats {
        let loads = unsafe {
            std::slice::from_raw_parts(
                self.expert_loads.contents().cast::<u32>(),
                ROUTED_EXPERTS as usize,
            )
        };
        let dispatch = unsafe {
            std::slice::from_raw_parts(
                self.routed_dispatch.contents().cast::<u32>(),
                DISPATCH_WORDS,
            )
        };
        let mask = unsafe {
            std::slice::from_raw_parts(self.route_mask.contents().cast::<u32>(), MASK_WORDS)
        };
        let mut expert_mask = [0; ROUTE_WORDS];
        for (word, pair) in expert_mask.iter_mut().zip(mask.chunks(2)) {
            *word = u64::from(pair[0]) | (u64::from(pair.get(1).copied().unwrap_or(0)) << 32);
        }
        RouteStats {
            active_experts: expert_mask
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum(),
            routed_rows: loads.iter().map(|&load| load as usize).sum(),
            routed_tiles: dispatch[1] as usize,
            expert_mask,
        }
    }

    pub(crate) fn reset_routes(&self, command: &CommandBufferRef) {
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(
            &self.route_mask,
            metal::NSRange::new(0, self.route_mask.length()),
            0,
        );
        blit.end_encoding();
    }
}
