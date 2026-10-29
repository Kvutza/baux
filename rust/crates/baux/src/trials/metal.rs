use std::ffi::c_void;
use std::sync::Arc;

extern crate metal as metal_crate;

use metal_crate::{Buffer, ComputePipelineState, MTLSize};

use super::{Center, LeafStep, Parameter, RoundConfig, SparseEdit, Tile, make_steps, make_tiles};
use crate::apple_gpu::{Runtime, thread_group};

#[path = "metal/metal_ops.rs"]
mod metal_ops;

const THREADS: u64 = 256;
const HISTORY_BATCH: usize = 8;
const SOURCE: &str = include_str!("trials.metal");

#[repr(C)]
#[derive(Clone, Copy)]
struct Seed {
    low: u32,
    high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    row_stride: u32,
    history: u32,
    candidates: u32,
    leaves: u32,
    tiles: u32,
    neighbors: u32,
    base_slot: u32,
    trial_slot: u32,
    center_count: u32,
    acquisition: u32,
    epistemic_scale: f32,
    aleatoric_scale: f32,
    y_scale: f32,
    beta: f32,
    num_pert: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CenterStep {
    parent: u32,
    seed: Seed,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MultiTrParams {
    num_regions: u32,
    candidates_per_region: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RowSumParams {
    row_stride: u32,
    row_bytes: u32,
    slot: u32,
    pad: u32,
}

struct Scratch {
    params: Buffer,
    history_slots: Buffer,
    outcomes: Buffer,
    seeds: Buffer,
    draws: Buffer,
    scores: Buffer,
    partials: Buffer,
    base_distances: Buffer,
    choice: Buffer,
    selected_scores: Buffer,
    leaves: Buffer,
    tiles: Buffer,
    centers: Buffer,
    candidate_centers: Buffer,
    edits: Buffer,
    row_sum: Buffer,
    candidate_capacity: usize,
    center_capacity: usize,
    edit_capacity: usize,
}

#[derive(Default)]
struct Resident {
    history: Vec<(usize, f32)>,
    steps: Vec<LeafStep>,
    centers: Vec<Center>,
    region_centers: Vec<usize>,
    candidates_per_region: usize,
}

pub(super) struct Engine {
    runtime: Arc<Runtime>,
    rows: Buffer,
    row_bytes: usize,
    row_stride: usize,
    tile_count: usize,
    distance: ComputePipelineState,
    base_distance: ComputePipelineState,
    score: ComputePipelineState,
    score_sparse: ComputePipelineState,
    fill_seeds: ComputePipelineState,
    fill_edits: ComputePipelineState,
    pick: ComputePipelineState,
    multi_tr_pick: ComputePipelineState,
    row_sum: ComputePipelineState,
    write: ComputePipelineState,
    write_sparse: ComputePipelineState,
    scratch: Scratch,
    resident: Resident,
    region: Option<Buffer>,
    radii: Buffer,
    region_steps: ComputePipelineState,
    state: Option<Buffer>,
    capacity: usize,
    configure: ComputePipelineState,
    advance: ComputePipelineState,
    copy_row: ComputePipelineState,
    shadow: Buffer,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Parameter], slots: usize) -> Result<Self, String> {
        let runtime = Runtime::shared()?;
        let source = format!(
            "{SOURCE}\n{}\n#define REGION_METAL\n{}\n#define DEVICE device\n{}\n{}",
            include_str!("../search/arithmetic.cl"),
            include_str!("region_steps.cl"),
            include_str!("../search/adaptation.cl"),
            include_str!("control.cl")
        );
        let pipeline = |name| runtime.precise(&source, "trial", name);
        let configure = pipeline("configure")?;
        let advance = pipeline("advance")?;
        let copy_row = pipeline("copy_row")?;
        let region_steps = pipeline("region_steps")?;
        let distance = pipeline("distance_trials")?;
        let base_distance = pipeline("base_distance")?;
        let score = pipeline("score_trials")?;
        let score_sparse = pipeline("score_sparse")?;
        let fill_seeds = pipeline("fill_seeds")?;
        let fill_edits = pipeline("fill_edits")?;
        let pick = pipeline("pick_trial")?;
        let multi_tr_pick = pipeline("multi_tr_pick_trials")?;
        let row_sum = pipeline("row_sum")?;
        let write = pipeline("write_trial")?;
        let write_sparse = pipeline("write_sparse")?;
        let row_stride = base
            .len()
            .checked_add(3)
            .ok_or("model row stride overflow")?
            & !3;
        let rows = shared(
            &runtime,
            slots
                .checked_mul(row_stride)
                .ok_or("model row arena size overflow")?,
            "model rows",
        )?;
        copy_to(&rows, base);
        let (scratch, tile_count) = alloc_scratch(&runtime, leaves, slots)?;
        let radii = runtime.buffer_with(&leaves.iter().map(|leaf| leaf.radius).collect::<Vec<_>>());
        let shadow = runtime.buffer::<u64>(16);
        Ok(Self {
            runtime,
            rows,
            row_bytes: base.len(),
            row_stride,
            tile_count,
            distance,
            base_distance,
            score,
            score_sparse,
            fill_seeds,
            fill_edits,
            pick,
            multi_tr_pick,
            row_sum,
            write,
            write_sparse,
            scratch,
            resident: Resident::default(),
            region: None,
            state: None,
            capacity: 0,
            configure,
            advance,
            copy_row,
            shadow,
            radii,
            region_steps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        self.ensure_candidates(seeds.len())?;
        self.write_seeds(seeds);
        self.run_dense(
            base_slot,
            history,
            trial_slot,
            seeds.len(),
            leaves,
            config,
            materialize_row,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_stream(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        base_seed: u64,
        count: usize,
        leaves: &[Parameter],
        config: RoundConfig,
        materialize_row: bool,
    ) -> Result<(usize, u64, f32), String> {
        self.ensure_candidates(count)?;
        self.fill_seeds(base_seed, count)?;
        let (index, score) = self.run_dense(
            base_slot,
            history,
            trial_slot,
            count,
            leaves,
            config,
            materialize_row,
        )?;
        Ok((index, super::cpu::seed_at(base_seed, index as u32), score))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_dense(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        count: usize,
        leaves: &[Parameter],
        config: RoundConfig,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        let distance_groups = distance_groups(count, history_count, self.tile_count)?;
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        self.sync_history(history)?;
        self.sync_steps(&steps);
        self.sync_draws(config);

        let params = Params {
            row_stride: to_u32(self.row_stride, "row stride")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(count, "candidate count")?,
            leaves: to_u32(leaves.len(), "leaf count")?,
            tiles: to_u32(self.tile_count, "tile count")?,
            neighbors: to_u32(config.surrogate.neighbors, "neighbor count")?,
            base_slot: to_u32(base_slot, "base slot")?,
            trial_slot: to_u32(trial_slot, "trial slot")?,
            center_count: 0,
            acquisition: crate::weights::acquisition_code(config.acquisition.kind),
            epistemic_scale: config.surrogate.epistemic_scale,
            aleatoric_scale: config.surrogate.aleatoric_scale,
            y_scale: config.surrogate.output_scale,
            beta: config.acquisition.beta,
            num_pert: 0,
        };
        copy_to(&self.scratch.params, &[params]);
        let params = self.scratch.params.clone();

        let command = self.runtime.queue.new_command_buffer();
        self.encode_scores(command, &params, distance_groups, count);
        self.encode_pick(command, &params, count);
        if materialize_row {
            self.encode_write(command, &params);
        }
        command.commit();
        command.wait_until_completed();
        let index = read_one::<u32>(&self.scratch.choice) as usize;
        if index == u32::MAX as usize {
            return Err("proposal rejected: insufficient history or unavailable slot".into());
        }
        Ok((index, read_one::<f32>(&self.scratch.selected_scores)))
    }

    pub(super) fn encode_scores(
        &self,
        command: &metal_crate::CommandBufferRef,
        params: &Buffer,
        distance_groups: u64,
        candidates: usize,
    ) {
        self.encode_region(command);
        self.encode_state(command, params);
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.history_slots), 0);
        encoder.set_buffer(2, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        encoder.set_buffer(5, Some(&self.scratch.partials), 0);
        encoder.set_buffer(6, Some(&self.scratch.centers), 0);
        encoder.set_buffer(7, Some(&self.scratch.candidate_centers), 0);
        set_params(&encoder, 8, params);
        encoder.dispatch_thread_groups(thread_group(distance_groups), thread_group(THREADS));
        encoder.end_encoding();

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.score);
        encoder.set_buffer(0, Some(&self.scratch.partials), 0);
        encoder.set_buffer(1, Some(&self.scratch.outcomes), 0);
        encoder.set_buffer(2, Some(&self.scratch.draws), 0);
        encoder.set_buffer(3, Some(&self.scratch.scores), 0);
        set_params(&encoder, 4, params);
        encoder.set_buffer(5, Some(&self.scratch.history_slots), 0);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: candidates as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();
    }

    pub(super) fn encode_pick(
        &self,
        command: &metal_crate::CommandBufferRef,
        params: &Buffer,
        candidates: usize,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pick);
        encoder.set_buffer(0, Some(&self.scratch.scores), 0);
        encoder.set_buffer(1, Some(&self.scratch.choice), 0);
        encoder.set_buffer(2, Some(&self.scratch.selected_scores), 0);
        set_params(&encoder, 3, params);
        encoder
            .dispatch_thread_groups(thread_group(1), thread_group(selection_threads(candidates)));
        encoder.end_encoding();
    }

    pub(super) fn encode_write(&self, command: &metal_crate::CommandBufferRef, params: &Buffer) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.write);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.seeds), 0);
        encoder.set_buffer(2, Some(&self.scratch.choice), 0);
        encoder.set_buffer(3, Some(&self.scratch.leaves), 0);
        encoder.set_buffer(4, Some(&self.scratch.tiles), 0);
        set_params(&encoder, 5, params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.tile_count as u64,
                height: 1,
                depth: 1,
            },
            thread_group(THREADS),
        );
        encoder.end_encoding();
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Vec<u8> {
        let start = slot * self.row_stride;
        unsafe {
            std::slice::from_raw_parts(self.rows.contents().cast::<u8>().add(start), row_bytes)
                .to_vec()
        }
    }

    pub(super) fn row_buffer(&self, slot: usize) -> Result<(Buffer, usize), String> {
        if self.row_bytes == 0 || slot >= self.rows.length() as usize / self.row_stride {
            return Err(format!("model row slot {slot} is out of range"));
        }
        Ok((self.rows.to_owned(), slot * self.row_stride))
    }

    pub(super) fn byte_sum(&self, slot: usize) -> Result<u64, String> {
        self.row_buffer(slot)?;
        let params = RowSumParams {
            row_stride: to_u32(self.row_stride, "row stride")?,
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            slot: to_u32(slot, "row slot")?,
            pad: 0,
        };
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.row_sum);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.row_sum), 0);
        sum_params(&encoder, 2, &params);
        encoder.dispatch_thread_groups(thread_group(1), thread_group(THREADS));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(read_one::<u64>(&self.scratch.row_sum))
    }

    pub(super) fn write(&self, slot: usize, row: &[u8]) {
        let start = slot * self.row_stride;
        unsafe {
            std::ptr::copy_nonoverlapping(
                row.as_ptr(),
                self.rows.contents().cast::<u8>().add(start),
                row.len(),
            );
        }
    }

    fn ensure_candidates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.scratch.candidate_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.seeds = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<Seed>()),
            "seeds",
        )?;
        self.scratch.scores = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<f32>()),
            "scores",
        )?;
        self.scratch.choice = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<u32>()),
            "choices",
        )?;
        self.scratch.selected_scores = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<f32>()),
            "selected scores",
        )?;
        self.scratch.candidate_centers = shared(
            &self.runtime,
            capacity.saturating_mul(size_of::<u32>()),
            "candidate centers",
        )?;
        let partial_count = capacity
            .checked_mul(super::MAX_HISTORY)
            .and_then(|value| value.checked_mul(self.tile_count))
            .ok_or("partial distance buffer size overflow")?;
        self.scratch.partials = shared(
            &self.runtime,
            partial_count.saturating_mul(size_of::<f32>()),
            "partial distances",
        )?;
        self.scratch.candidate_capacity = capacity;
        self.resident.region_centers.clear();
        Ok(())
    }

    fn sync_history(&mut self, history: &[(usize, f32)]) -> Result<(), String> {
        if self.state.is_some() {
            return Ok(());
        }
        let prefix = self
            .resident
            .history
            .iter()
            .zip(history)
            .take_while(|((old_slot, old_value), (slot, value))| {
                old_slot == slot && old_value.to_bits() == value.to_bits()
            })
            .count();
        if prefix == history.len() && history.len() == self.resident.history.len() {
            return Ok(());
        }
        let start = if prefix == self.resident.history.len() {
            prefix
        } else {
            0
        };
        for (index, &(slot, value)) in history.iter().enumerate().skip(start) {
            let slot = to_u32(slot, "history slot")?;
            copy_one(&self.scratch.history_slots, index, slot);
            copy_one(&self.scratch.outcomes, index, value);
        }
        self.resident.history.clear();
        self.resident.history.extend_from_slice(history);
        Ok(())
    }

    pub(super) fn bind_region(&mut self, region: Buffer) {
        self.region = Some(region);
    }

    fn encode_region(&self, command: &metal_crate::CommandBufferRef) {
        if let Some(region) = &self.region {
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.region_steps);
            encoder.set_buffer(0, Some(region), 0);
            encoder.set_buffer(1, Some(&self.scratch.leaves), 0);
            encoder.set_buffer(2, Some(&self.radii), 0);
            let count = self.radii.length() / size_of::<f32>() as u64;
            encoder.dispatch_threads(thread_group(count), thread_group(1));
            encoder.end_encoding();
        }
    }

    fn sync_steps(&mut self, steps: &[LeafStep]) {
        if self.region.is_some() || self.resident.steps == steps {
            return;
        }
        copy_to(&self.scratch.leaves, steps);
        self.resident.steps.clear();
        self.resident.steps.extend_from_slice(steps);
    }

    fn sync_draws(&self, config: RoundConfig) {
        if config.acquisition.kind == crate::weights::AcquisitionKind::Thompson {
            // Cover every resident slot without reading device-managed history.
            let slots = self.scratch.draws.length() as usize / size_of::<f32>();
            let draws = crate::weights::thompson_draws(slots, config.acquisition.seed);
            copy_to(&self.scratch.draws, &draws);
        }
    }

    fn write_seeds(&self, seeds: &[u64]) {
        debug_assert_eq!(size_of::<Seed>(), size_of::<u64>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                seeds.as_ptr().cast::<u8>(),
                self.scratch.seeds.contents().cast::<u8>(),
                std::mem::size_of_val(seeds),
            );
        }
    }

    fn fill_seeds(&self, base_seed: u64, count: usize) -> Result<(), String> {
        let seed = Seed {
            low: base_seed as u32,
            high: (base_seed >> 32) as u32,
        };
        let candidates = to_u32(count, "candidate count")?;
        let command = self.runtime.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.fill_seeds);
        encoder.set_buffer(0, Some(&self.scratch.seeds), 0);
        encoder.set_bytes(
            1,
            size_of::<Seed>() as u64,
            (&seed as *const Seed).cast::<c_void>(),
        );
        encoder.set_bytes(
            2,
            size_of::<u32>() as u64,
            (&candidates as *const u32).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(thread_group(count as u64), thread_group(1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(())
    }
}

fn alloc_scratch(
    runtime: &Runtime,
    leaves: &[Parameter],
    slots: usize,
) -> Result<(Scratch, usize), String> {
    let tiles = make_tiles(leaves);
    let scratch = Scratch {
        params: runtime.buffer::<Params>(1),
        history_slots: shared(
            runtime,
            super::MAX_HISTORY * size_of::<u32>(),
            "history slots",
        )?,
        outcomes: shared(runtime, super::MAX_HISTORY * size_of::<f32>(), "outcomes")?,
        seeds: shared(runtime, size_of::<Seed>(), "seeds")?,
        draws: shared(runtime, slots * size_of::<f32>(), "draws")?,
        scores: shared(runtime, size_of::<f32>(), "scores")?,
        partials: shared(
            runtime,
            super::MAX_HISTORY
                .saturating_mul(tiles.len())
                .saturating_mul(size_of::<f32>()),
            "partial distances",
        )?,
        base_distances: shared(
            runtime,
            super::MAX_HISTORY.saturating_mul(size_of::<f32>()),
            "base distances",
        )?,
        choice: shared(runtime, size_of::<u32>(), "choice")?,
        selected_scores: shared(runtime, size_of::<f32>(), "selected scores")?,
        leaves: shared(
            runtime,
            leaves.len().saturating_mul(size_of::<LeafStep>()),
            "leaves",
        )?,
        tiles: shared(
            runtime,
            tiles.len().saturating_mul(size_of::<Tile>()),
            "tiles",
        )?,
        centers: shared(runtime, size_of::<CenterStep>(), "centers")?,
        candidate_centers: shared(runtime, size_of::<u32>(), "candidate centers")?,
        edits: shared(runtime, size_of::<SparseEdit>(), "sparse edits")?,
        row_sum: shared(runtime, size_of::<u64>(), "row sum")?,
        candidate_capacity: 1,
        center_capacity: 1,
        edit_capacity: 1,
    };
    copy_to(&scratch.tiles, &tiles);
    copy_to(&scratch.leaves, &make_steps(leaves, 0.0));
    Ok((scratch, tiles.len()))
}

fn shared(runtime: &Runtime, bytes: usize, name: &str) -> Result<Buffer, String> {
    if bytes == 0 {
        return Err(format!("{name} buffer cannot be empty"));
    }
    let max_bytes = runtime.device.max_buffer_length();
    if bytes as u64 > max_bytes as u64 {
        return Err(format!(
            "{name} buffer requires {bytes} bytes, exceeding the Metal device limit of \
             {max_bytes} bytes"
        ));
    }
    let buffer = runtime.buffer::<u8>(bytes);
    if buffer.length() < bytes as u64 || buffer.contents().is_null() {
        return Err(format!(
            "Metal could not allocate the {name} buffer ({bytes} bytes)"
        ));
    }
    Ok(buffer)
}

fn copy_to<T>(buffer: &Buffer, values: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().cast::<u8>(),
            std::mem::size_of_val(values),
        );
    }
}

fn copy_one<T: Copy>(buffer: &Buffer, index: usize, value: T) {
    unsafe {
        buffer.contents().cast::<T>().add(index).write(value);
    }
}

fn read_one<T: Copy>(buffer: &Buffer) -> T {
    unsafe { *buffer.contents().cast::<T>() }
}

fn read_slice<T: Copy>(buffer: &Buffer, len: usize) -> &[T] {
    unsafe { std::slice::from_raw_parts(buffer.contents().cast::<T>(), len) }
}

fn set_params(encoder: &metal_crate::ComputeCommandEncoderRef, index: u64, params: &Buffer) {
    encoder.set_buffer(index, Some(params), 0);
}

fn sum_params(encoder: &metal_crate::ComputeCommandEncoderRef, index: u64, params: &RowSumParams) {
    encoder.set_bytes(
        index,
        size_of::<RowSumParams>() as u64,
        (params as *const RowSumParams).cast::<c_void>(),
    );
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}

fn distance_groups(candidates: usize, history: usize, tiles: usize) -> Result<u64, String> {
    candidates
        .div_ceil(2)
        .checked_mul(history.div_ceil(HISTORY_BATCH))
        .and_then(|value| value.checked_mul(tiles))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or("distance dispatch size overflow".to_string())
}

fn selection_threads(candidates: usize) -> u64 {
    candidates.next_power_of_two().clamp(32, THREADS as usize) as u64
}

const fn size_of<T>() -> usize {
    std::mem::size_of::<T>()
}
