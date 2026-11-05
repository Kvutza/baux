use super::scorer::*;
use super::*;
use metal::MTLDispatchType;
use std::cell::RefCell;

pub(crate) fn replay_site(
    command: &CommandBufferRef,
    pipelines: &Pipelines,
    tensorops: &TensorOpsPipelines,
    pisa: &Pisa1,
    buffers: &Buffers,
    weights: CandidateRow<'_>,
    rows: u32,
    target: ReplaySite,
) -> Result<(), String> {
    if rows == 0 || rows > buffers.rows || rows % 128 != 0 {
        return Err("invalid training replay extent".into());
    }
    buffers.fine_grained.reset_routes(command);
    let scorer = Scorer {
        command,
        encoder: RefCell::new(Some(
            command.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial),
        )),
        pipelines,
        tensorops,
        pisa1: pisa,
        buffers,
        weights,
        row_start: 0,
        rows,
        active_rows: std::cell::Cell::new(rows),
        cache: None,
        dirty: false,
        trace: None,
        feedback: crate::config::FeedbackTransition::Identity,
        diffusion: None,
        cache_only: false,
    };
    let result = scorer.encode_to(target);
    if let Some(encoder) = scorer.encoder.borrow_mut().take() {
        encoder.end_encoding();
    }
    result
}
