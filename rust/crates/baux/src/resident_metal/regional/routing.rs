//! Cross-region relevance routing, box/ball containment, and import flushing.

use super::*;

pub fn restarts(search: &RegionalSearch, slot: usize) -> Result<usize, String> {
    search.states.get(slot).ok_or("unknown region")?.restarts()
}

fn decode_word(bits: u16, fp16: bool) -> f32 {
    if fp16 {
        crate::shm_slab::decode_half(bits)
    } else {
        f32::from_bits(u32::from(bits) << 16)
    }
}

fn contains_resident(state: &SearchState, incoming: &Buffer) -> Result<bool, String> {
    state.check_healthy()?;
    if state.async_command.is_some() {
        return Err("Finish the resident GPU command before measuring an import".into());
    }
    if state.history == 0 {
        return Err("Measure the regional incumbent before importing observations".into());
    }
    if incoming.length() != state.row_bytes() {
        return Err("Imported resident row has the wrong byte length".into());
    }
    let inc =
        unsafe { std::slice::from_raw_parts(incoming.contents().cast::<u16>(), state.dimensions) };
    let base = unsafe {
        std::slice::from_raw_parts(state.base.contents().cast::<u16>(), state.dimensions)
    };
    let len = state.length as f32;
    let fp16 = state.independent_fp16;
    for block in &state.blocks {
        let half = (block.scale * len).max(1.0e-12);
        for idx in block.offset..block.offset + block.len {
            let x = decode_word(inc[idx], fp16);
            let c = decode_word(base[idx], fp16);
            if !x.is_finite() || (x - c).abs() > half {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

pub(super) fn routes(
    search: &RegionalSearch,
    source: Option<usize>,
    incoming: &Buffer,
) -> Result<Vec<usize>, String> {
    if search.sharing == SharingPolicy::OriginOnly {
        return Ok(Vec::new());
    }
    if search.sharing == SharingPolicy::BoxContain {
        let mut out = Vec::new();
        for slot in 0..search.len() {
            if Some(slot) != source {
                let state = search.states.get(slot).ok_or("unknown region")?;
                if contains_resident(state, incoming)? {
                    out.push(slot);
                }
            }
        }
        return Ok(out);
    }
    let mut distances_and_radii2 = Vec::with_capacity(search.len());
    for slot in 0..search.len() {
        let state = search.states.get(slot).ok_or("unknown region")?;
        let distance = f64::from(state.measure_center(incoming)?);
        let scale = state
            .blocks
            .iter()
            .map(|block| {
                block.len as f64 * f64::from(block.weight) * f64::from(block.scale).powi(2)
            })
            .sum::<f64>();
        let radius2 = state.length.powi(2) * scale;
        distances_and_radii2.push((distance, radius2));
    }
    Ok(search
        .sharing
        .route_distances(source, &distances_and_radii2))
}

pub(super) fn flush_region(
    search: &mut RegionalSearch,
    slot: usize,
    adopt: bool,
) -> Result<(), String> {
    let state = search.states.get_mut(slot).ok_or("unknown region")?;
    state.check_idle()?;
    let has_surrogate = state.has_surrogate();
    while let Some(entry) = search.pending_imports[slot].first() {
        let row = entry.row.clone();
        let observation = entry.observation;
        let from_region = entry.from_region;
        if has_surrogate {
            search.states[slot].import_objectives(&row, observation)?;
        }
        if (!has_surrogate || adopt)
            && from_region
            && search.states[slot].adopt_incumbent(&row, observation)?
        {
            search
                .ledger
                .adopt_shared(slot, f64::from(observation.control().mean))
                .map_err(|error| error.to_string())?;
        }
        search.pending_imports[slot].remove(0);
        search.delivered_imports += 1;
    }
    Ok(())
}
