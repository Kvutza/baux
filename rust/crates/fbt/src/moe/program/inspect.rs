//! Explicit tensor snapshots for compiler parity diagnosis.
use super::*;

fn copy(buffer: &BufferRef, bytes: usize) -> Vec<u8> {
    assert!(bytes <= buffer.length() as usize);
    // Called after command completion under the State mutex; all scorer
    // allocations are shared Metal storage and remain owned by State.
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), bytes) }.to_vec()
}

impl Resident {
    /// Replay the full native batch and snapshot a prefix of each visit's
    /// residual state, normalized FFN input, router logits and expert IDs.
    /// Keeping the native row count preserves its matrix/packing schedules.
    /// This is an explicit host transfer, not the forward hot path.
    pub fn inspect(&self, tokens: &[u32], rows: usize) -> Result<Vec<(String, Vec<u8>)>, String> {
        if rows == 0 || rows > CONTEXT as usize || rows % 128 != 0 {
            return Err("inspection requires an aligned prefix within one sequence".into());
        }
        let batch = self.batch(tokens, tokens, &vec![1; ROWS as usize])?;
        let leased = self.search.program_base(&self.forward)?;
        let weights = metal(&leased)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| "model state mutex poisoned")?;
        state.buffers.tokens = metal(&batch[0])?;
        let mut outputs = Vec::new();
        for visit in baux::semantics::mhc4::layout::MHC4_VISITS {
            let step = visit.step;
            let command = state.runtime.queue.new_command_buffer();
            scorer_replay::replay_site(
                command,
                &state.pipelines,
                &state.tensorops,
                &state.pisa,
                &state.buffers,
                state.layout.view(&weights),
                ROWS,
                scorer::ReplaySite::Moe(step),
            )?;
            complete(command)?;
            for (name, buffer, bytes) in [
                (
                    "streams",
                    &state.buffers.mhc_streams[0],
                    rows * 4 * WIDTH as usize * 2,
                ),
                (
                    "input",
                    &state.buffers.normalized,
                    rows * WIDTH as usize * 2,
                ),
                ("branch", &state.buffers.output, rows * WIDTH as usize * 2),
                (
                    "scores",
                    &state.buffers.fine_grained.scores,
                    rows * routing::ROUTED_EXPERTS as usize * 2,
                ),
                (
                    "experts",
                    &state.buffers.fine_grained.route_experts,
                    rows * 3 * 4,
                ),
            ] {
                outputs.push((format!("{step}/{name}"), copy(buffer, bytes)));
            }
        }
        Ok(outputs)
    }
}
