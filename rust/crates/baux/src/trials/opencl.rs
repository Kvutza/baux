use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, ClMem};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_mem_flags};

use super::{Center, LeafStep, Parameter, RoundConfig, SparseEdit, Tile, make_steps, make_tiles};

#[path = "opencl/opencl_ops.rs"]
mod opencl_ops;

const THREADS: usize = 256;
const SOURCE: &str = include_str!("trials.cl");

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Seed {
    low: u32,
    high: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Params {
    row_bytes: u32,
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
struct RowSumParams {
    row_bytes: u32,
    slot: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MultiTrParams {
    num_regions: u32,
    candidates_per_region: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CenterStep {
    parent: u32,
    seed: Seed,
}

struct Scratch {
    params: std::sync::Arc<Buffer<Params>>,
    history_slots: Buffer<u32>,
    outcomes: Buffer<f32>,
    seeds: Buffer<Seed>,
    draws: Buffer<f32>,
    scores: Buffer<f32>,
    partials: Buffer<f32>,
    choice: Buffer<u32>,
    leaves: Buffer<LeafStep>,
    tiles: Buffer<Tile>,
    centers: Buffer<CenterStep>,
    candidate_centers: Buffer<u32>,
    edits: Buffer<SparseEdit>,
    base_distances: Buffer<f32>,
    row_sum: Buffer<u64>,
    selected_scores: Buffer<f32>,
    candidate_capacity: usize,
    center_capacity: usize,
    edit_capacity: usize,
}

/// Borrowed OpenCL row binding for a resident candidate.
///
/// The queue and context borrows keep the OpenCL runtime alive for as long as
/// the evaluator needs the row, while the cloned buffer handle keeps the device
/// allocation alive independently of the owning search.
pub(crate) struct ResidentRow<'a> {
    context: &'a Context,
    queue: &'a CommandQueue,
    rows: &'a Buffer<u8>,
    offset: usize,
    row_bytes: usize,
}

impl<'a> ResidentRow<'a> {
    pub fn context(&self) -> &'a Context {
        self.context
    }

    pub fn queue(&self) -> &'a CommandQueue {
        self.queue
    }

    pub fn buffer(&self) -> &'a Buffer<u8> {
        self.rows
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    #[allow(dead_code)]
    pub fn end(&self) -> usize {
        self.offset + self.row_bytes
    }
}

pub(super) struct Engine {
    pub(super) context: Context,
    queue: CommandQueue,
    rows: Buffer<u8>,
    row_bytes: usize,
    slots: usize,
    tile_count: usize,
    distance: Kernel,
    base_distance: Kernel,
    score: Kernel,
    score_sparse: Kernel,
    fill_seeds: Kernel,
    fill_edits: Kernel,
    pick: Kernel,
    multi_tr_pick: Kernel,
    write: Kernel,
    write_sparse: Kernel,
    row_sum: Kernel,
    scratch: Scratch,
    uploaded_history: Vec<(u32, u32)>,
    uploaded_leaves: Vec<LeafStep>,
    region: Option<std::sync::Arc<Buffer<u64>>>,
    radii: Buffer<f32>,
    leaf_count: usize,
    region_steps: Kernel,
    state: Option<Buffer<u32>>,
    capacity: usize,
    configure: Kernel,
    advance: Kernel,
    copy_row: Kernel,
    shadow: Buffer<u64>,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Parameter], slots: usize) -> Result<Self, String> {
        let (context, queue, program) = open_device()?;
        let kernel = |name: &str| {
            Kernel::create(&program, name)
                .map_err(|error| format!("missing OpenCL trial kernel {name}: {error}"))
        };
        let distance = kernel("distance_trials")?;
        let base_distance = kernel("base_distance")?;
        let score = kernel("score_trials")?;
        let score_sparse = kernel("score_sparse")?;
        let fill_seeds = kernel("fill_seeds")?;
        let fill_edits = kernel("fill_edits")?;
        let pick = kernel("pick_trial")?;
        let multi_tr_pick = kernel("multi_tr_pick_trials")?;
        let write = kernel("write_trial")?;
        let write_sparse = kernel("write_sparse")?;
        let row_sum = kernel("row_sum")?;
        let region_steps =
            Kernel::create(&program, "region_steps").map_err(|error| error.to_string())?;
        let configure = Kernel::create(&program, "configure").map_err(|error| error.to_string())?;
        let advance = Kernel::create(&program, "advance").map_err(|error| error.to_string())?;
        let copy_row = Kernel::create(&program, "copy_row").map_err(|error| error.to_string())?;
        let (scratch, rows, radii, shadow, tile_count) =
            alloc_scratch(&context, &queue, base, leaves, slots)?;
        Ok(Self {
            region: None,
            state: None,
            capacity: 0,
            configure,
            advance,
            copy_row,
            shadow,
            radii,
            leaf_count: leaves.len(),
            region_steps,
            context,
            queue,
            rows,
            row_bytes: base.len(),
            slots,
            tile_count,
            distance,
            base_distance,
            score,
            score_sparse,
            fill_seeds,
            fill_edits,
            pick,
            multi_tr_pick,
            write,
            write_sparse,
            row_sum,
            scratch,
            uploaded_history: Vec::new(),
            uploaded_leaves: Vec::new(),
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
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        self.ensure_candidates(seeds.len())?;
        let distance_groups = seeds
            .len()
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let seeds: Vec<Seed> = seeds
            .iter()
            .map(|&seed| Seed {
                low: seed as u32,
                high: (seed >> 32) as u32,
            })
            .collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        self.write_inputs(&history_slots, &outcomes, &seeds, &steps)?;
        self.fill_draws(config)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            history: to_u32(history_count, "history length")?,
            candidates: to_u32(seeds.len(), "candidate count")?,
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
        let params = self.parameters(params)?;
        self.finish_dense(&params, distance_groups, seeds.len(), materialize_row)
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
        let history_count = if self.state.is_some() {
            self.capacity
        } else {
            history.len()
        };
        self.ensure_candidates(count)?;
        let distance_groups = count
            .div_ceil(2)
            .checked_mul(self.tile_count)
            .ok_or("distance dispatch size overflow")?;
        let history_slots: Vec<u32> = history
            .iter()
            .map(|&(slot, _)| to_u32(slot, "history slot"))
            .collect::<Result<_, _>>()?;
        let outcomes: Vec<f32> = history.iter().map(|&(_, value)| value).collect();
        let steps = if self.region.is_some() {
            Vec::new()
        } else {
            make_steps(leaves, config.proposal.radius)
        };
        self.write_static(&history_slots, &outcomes, &steps)?;
        self.fill_seeds(base_seed, count)?;
        self.fill_draws(config)?;

        let params = Params {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
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
        let params = self.parameters(params)?;
        let (index, score) = self.finish_dense(&params, distance_groups, count, materialize_row)?;
        Ok((index, super::cpu::seed_at(base_seed, index as u32), score))
    }

    fn finish_dense(
        &self,
        params: &Buffer<Params>,
        distance_groups: usize,
        candidates: usize,
        materialize_row: bool,
    ) -> Result<(usize, f32), String> {
        self.run_scores(params, distance_groups, candidates)?;
        unsafe {
            ExecuteKernel::new(&self.pick)
                .set_arg(&self.scratch.scores)
                .set_arg(&self.scratch.choice)
                .set_arg(&self.scratch.selected_scores)
                .set_arg(params)
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial selection: {error}"))?;
            if materialize_row {
                ExecuteKernel::new(&self.write)
                    .set_arg(&self.rows)
                    .set_arg(&self.scratch.seeds)
                    .set_arg(&self.scratch.choice)
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.scratch.tiles)
                    .set_arg(params)
                    .set_global_work_size(self.tile_count * THREADS)
                    .set_local_work_size(THREADS)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| format!("failed to launch OpenCL trial write: {error}"))?;
            }
        }
        let mut choice = [0u32];
        let mut score = [0.0f32];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.choice, CL_BLOCKING, 0, &mut choice, &[])
                .map_err(|error| format!("failed to read OpenCL trial choice: {error}"))?;
            self.queue
                .enqueue_read_buffer(
                    &self.scratch.selected_scores,
                    CL_BLOCKING,
                    0,
                    &mut score,
                    &[],
                )
                .map_err(|error| format!("failed to read OpenCL selected score: {error}"))?;
        }
        let index = choice[0] as usize;
        if index >= candidates {
            return Err("OpenCL trial choice exceeds candidate count".into());
        }
        Ok((index, score[0]))
    }

    pub(super) fn read(&self, slot: usize, row_bytes: usize) -> Result<Vec<u8>, String> {
        let mut row = vec![0u8; row_bytes];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.rows, CL_BLOCKING, slot * row_bytes, &mut row, &[])
                .map_err(|error| format!("failed to read OpenCL trial row: {error}"))?
                .wait()
                .map_err(|error| format!("failed waiting for OpenCL trial row: {error}"))?;
        }
        Ok(row)
    }

    pub(super) fn device_view(&self, slot: usize) -> Result<ResidentRow<'_>, String> {
        if self.row_bytes == 0 {
            return Err("model row bytes must be positive".to_string());
        }
        let allocation_len = self
            .rows
            .size()
            .map_err(|error| format!("failed to query OpenCL model row size: {error}"))?
            as usize;
        let offset = slot
            .checked_mul(self.row_bytes)
            .ok_or("model row offset overflow")?;
        let end = offset
            .checked_add(self.row_bytes)
            .ok_or("model row extent overflow")?;
        if end > allocation_len {
            return Err(format!("model row slot {slot} is out of range"));
        }
        Ok(ResidentRow {
            context: &self.context,
            queue: &self.queue,
            rows: &self.rows,
            offset,
            row_bytes: self.row_bytes,
        })
    }

    pub(super) fn byte_sum(&self, slot: usize) -> Result<u64, String> {
        self.device_view(slot)?;
        let params = RowSumParams {
            row_bytes: to_u32(self.row_bytes, "row bytes")?,
            slot: to_u32(slot, "row slot")?,
            pad0: 0,
            pad1: 0,
        };
        unsafe {
            ExecuteKernel::new(&self.row_sum)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.row_sum)
                .set_arg(&params)
                .set_global_work_size(THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL resident row sum: {error}"))?;
        }
        let mut sum = [0u64];
        unsafe {
            self.queue
                .enqueue_read_buffer(&self.scratch.row_sum, CL_BLOCKING, 0, &mut sum, &[])
                .map_err(|error| format!("failed to read OpenCL resident row sum: {error}"))?;
        }
        Ok(sum[0])
    }

    pub(super) fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.rows, CL_BLOCKING, slot * self.row_bytes, row, &[])
                .map_err(|error| format!("failed to write OpenCL trial row: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn ensure_candidates(&mut self, count: usize) -> Result<(), String> {
        if count <= self.scratch.candidate_capacity {
            return Ok(());
        }
        let capacity = count.next_power_of_two();
        self.scratch.seeds = buffer(&self.context, capacity, CL_MEM_READ_ONLY, "seeds")?;
        self.scratch.scores = buffer(&self.context, capacity, CL_MEM_READ_WRITE, "scores")?;
        self.scratch.choice = buffer(&self.context, capacity, CL_MEM_READ_WRITE, "choices")?;
        self.scratch.selected_scores = buffer(
            &self.context,
            capacity,
            CL_MEM_READ_WRITE,
            "selected scores",
        )?;
        self.scratch.candidate_centers = buffer(
            &self.context,
            capacity,
            CL_MEM_READ_ONLY,
            "candidate centers",
        )?;
        let partial_count = capacity
            .checked_mul(super::MAX_HISTORY)
            .and_then(|value| value.checked_mul(self.tile_count))
            .ok_or("partial distance buffer size overflow")?;
        self.scratch.partials = buffer(
            &self.context,
            partial_count,
            CL_MEM_READ_WRITE,
            "partial distances",
        )?;
        self.scratch.candidate_capacity = capacity;
        Ok(())
    }

    pub(super) fn run_scores(
        &self,
        params: &Buffer<Params>,
        distance_groups: usize,
        candidates: usize,
    ) -> Result<(), String> {
        unsafe {
            ExecuteKernel::new(&self.distance)
                .set_arg(&self.rows)
                .set_arg(&self.scratch.history_slots)
                .set_arg(&self.scratch.seeds)
                .set_arg(&self.scratch.leaves)
                .set_arg(&self.scratch.tiles)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.centers)
                .set_arg(&self.scratch.candidate_centers)
                .set_arg(params)
                .set_global_work_size(
                    distance_groups
                        .checked_mul(THREADS)
                        .ok_or("distance work size overflow")?,
                )
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial distances: {error}"))?;
            ExecuteKernel::new(&self.score)
                .set_arg(&self.scratch.partials)
                .set_arg(&self.scratch.outcomes)
                .set_arg(&self.scratch.draws)
                .set_arg(&self.scratch.scores)
                .set_arg(params)
                .set_arg(&self.scratch.history_slots)
                .set_global_work_size(candidates * THREADS)
                .set_local_work_size(THREADS)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL trial scoring: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn write_inputs(
        &mut self,
        history_slots: &[u32],
        outcomes: &[f32],
        seeds: &[Seed],
        leaves: &[LeafStep],
    ) -> Result<(), String> {
        self.write_static(history_slots, outcomes, leaves)?;
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.seeds, CL_BLOCKING, 0, seeds, &[])
                .map_err(|error| format!("failed to write OpenCL trial seeds: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn write_static(
        &mut self,
        history_slots: &[u32],
        outcomes: &[f32],
        leaves: &[LeafStep],
    ) -> Result<(), String> {
        // Borrowed host slices must remain alive until each upload completes,
        // including when a later enqueue fails. Keep blocking writes until an
        // event-owned staging abstraction can guarantee that on every exit.
        let prefix = self
            .uploaded_history
            .iter()
            .zip(history_slots.iter().zip(outcomes))
            .take_while(|((old_slot, old_value), (slot, value))| {
                old_slot == *slot && *old_value == value.to_bits()
            })
            .count();
        unsafe {
            if self.state.is_none() && prefix < history_slots.len() {
                self.queue
                    .enqueue_write_buffer(
                        &mut self.scratch.history_slots,
                        CL_BLOCKING,
                        prefix * std::mem::size_of::<u32>(),
                        &history_slots[prefix..],
                        &[],
                    )
                    .map_err(|error| format!("failed to write OpenCL history slots: {error}"))?;
                self.queue
                    .enqueue_write_buffer(
                        &mut self.scratch.outcomes,
                        CL_BLOCKING,
                        prefix * std::mem::size_of::<f32>(),
                        &outcomes[prefix..],
                        &[],
                    )
                    .map_err(|error| format!("failed to write OpenCL outcomes: {error}"))?;
            }
        }
        self.sync_leaves(leaves)?;
        self.uploaded_history = history_slots
            .iter()
            .zip(outcomes)
            .map(|(&slot, value)| (slot, value.to_bits()))
            .collect();
        Ok(())
    }

    pub(super) fn fill_seeds(&self, base_seed: u64, count: usize) -> Result<(), String> {
        let seed = Seed {
            low: base_seed as u32,
            high: (base_seed >> 32) as u32,
        };
        unsafe {
            ExecuteKernel::new(&self.fill_seeds)
                .set_arg(&self.scratch.seeds)
                .set_arg(&seed)
                .set_arg(&to_u32(count, "candidate count")?)
                .set_global_work_size(count)
                .set_local_work_size(1)
                .enqueue_nd_range(&self.queue)
                .map_err(|error| format!("failed to launch OpenCL seed fill: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn fill_draws(&mut self, config: RoundConfig) -> Result<(), String> {
        if config.acquisition.kind != crate::weights::AcquisitionKind::Thompson {
            return Ok(());
        }
        // Cover every resident slot without reading device-managed history.
        let draws = crate::weights::thompson_draws(self.slots, config.acquisition.seed);
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.draws, CL_BLOCKING, 0, &draws, &[])
                .map_err(|error| format!("failed to write OpenCL Thompson draws: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn bind_region(&mut self, region: std::sync::Arc<Buffer<u64>>) {
        self.region = Some(region);
    }

    pub(super) fn sync_leaves(&mut self, leaves: &[LeafStep]) -> Result<(), String> {
        if let Some(region) = &self.region {
            unsafe {
                ExecuteKernel::new(&self.region_steps)
                    .set_arg(region.as_ref())
                    .set_arg(&self.scratch.leaves)
                    .set_arg(&self.radii)
                    .set_global_work_size(self.leaf_count)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
            return Ok(());
        }
        if self.uploaded_leaves == leaves {
            return Ok(());
        }
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut self.scratch.leaves, CL_BLOCKING, 0, leaves, &[])
                .map_err(|error| format!("failed to write OpenCL leaves: {error}"))?;
        }
        self.uploaded_leaves.clear();
        self.uploaded_leaves.extend_from_slice(leaves);
        Ok(())
    }

    pub(super) fn parameters(
        &mut self,
        value: Params,
    ) -> Result<std::sync::Arc<Buffer<Params>>, String> {
        let params = std::sync::Arc::get_mut(&mut self.scratch.params)
            .ok_or("dispatch parameters are still borrowed")?;
        unsafe {
            self.queue
                .enqueue_write_buffer(params, CL_BLOCKING, 0, &[value], &[])
                .map_err(|error| error.to_string())?;
            if let Some(state) = &self.state {
                ExecuteKernel::new(&self.configure)
                    .set_arg(state)
                    .set_arg(&self.scratch.history_slots)
                    .set_arg(params)
                    .set_arg(&self.scratch.choice)
                    .set_global_work_size(1)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
                ExecuteKernel::new(&self.copy_row)
                    .set_arg(&self.rows)
                    .set_arg(params)
                    .set_global_work_size(self.row_bytes)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|error| error.to_string())?;
            }
        }
        Ok(self.scratch.params.clone())
    }
}

fn open_device() -> Result<(Context, CommandQueue, Program), String> {
    let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|error| format!("failed to enumerate OpenCL GPU devices: {error}"))?
        .into_iter()
        .next()
        .or_else(|| {
            get_all_devices(CL_DEVICE_TYPE_CPU)
                .ok()
                .and_then(|devices| devices.into_iter().next())
        })
        .ok_or("no OpenCL GPU or CPU device found")?;
    let device = Device::new(device_id);
    let context = Context::from_device(&device)
        .map_err(|error| format!("failed to create OpenCL context: {error}"))?;
    let queue = CommandQueue::create_default(&context, 0)
        .map_err(|error| format!("failed to create OpenCL command queue: {error}"))?;
    let source = format!(
        "{SOURCE}\n{}\n{}\n#define DEVICE __global\n{}\n{}",
        include_str!("../search/arithmetic.cl"),
        include_str!("region_steps.cl"),
        include_str!("../search/adaptation.cl"),
        include_str!("control.cl")
    );
    let program = Program::create_and_build_from_source(&context, &source, "")
        .map_err(|error| format!("failed to build OpenCL trial kernels: {error}"))?;
    Ok((context, queue, program))
}

#[allow(clippy::type_complexity)]
fn alloc_scratch(
    context: &Context,
    queue: &CommandQueue,
    base: &[u8],
    leaves: &[Parameter],
    slots: usize,
) -> Result<(Scratch, Buffer<u8>, Buffer<f32>, Buffer<u64>, usize), String> {
    let mut radii = buffer(context, leaves.len(), CL_MEM_READ_ONLY, "radii")?;
    let shadow = buffer(context, 16, CL_MEM_READ_WRITE, "region staging")?;
    let tiles = make_tiles(leaves);
    let mut rows = buffer::<u8>(
        context,
        slots.saturating_mul(base.len()),
        CL_MEM_READ_WRITE,
        "model rows",
    )?;
    let mut scratch = Scratch {
        params: std::sync::Arc::new(buffer(
            context,
            1,
            CL_MEM_READ_WRITE,
            "dispatch parameters",
        )?),
        history_slots: buffer(
            context,
            super::MAX_HISTORY,
            CL_MEM_READ_WRITE,
            "history slots",
        )?,
        outcomes: buffer(context, super::MAX_HISTORY, CL_MEM_READ_WRITE, "outcomes")?,
        seeds: buffer(context, 1, CL_MEM_READ_ONLY, "seeds")?,
        draws: buffer(context, slots, CL_MEM_READ_ONLY, "draws")?,
        scores: buffer(context, 1, CL_MEM_READ_WRITE, "scores")?,
        partials: buffer(
            context,
            super::MAX_HISTORY.saturating_mul(tiles.len()),
            CL_MEM_READ_WRITE,
            "partial distances",
        )?,
        choice: buffer(context, 1, CL_MEM_READ_WRITE, "choice")?,
        leaves: buffer(context, leaves.len(), CL_MEM_READ_WRITE, "leaves")?,
        tiles: buffer(context, tiles.len(), CL_MEM_READ_ONLY, "tiles")?,
        centers: buffer(context, 1, CL_MEM_READ_ONLY, "centers")?,
        candidate_centers: buffer(context, 1, CL_MEM_READ_ONLY, "candidate centers")?,
        edits: buffer(context, 1, CL_MEM_READ_ONLY, "sparse edits")?,
        base_distances: buffer(
            context,
            super::MAX_HISTORY,
            CL_MEM_READ_WRITE,
            "base distances",
        )?,
        row_sum: buffer(context, 1, CL_MEM_READ_WRITE, "row sum")?,
        selected_scores: buffer(context, 1, CL_MEM_READ_WRITE, "selected scores")?,
        candidate_capacity: 1,
        center_capacity: 1,
        edit_capacity: 1,
    };
    unsafe {
        queue
            .enqueue_write_buffer(
                &mut radii,
                CL_BLOCKING,
                0,
                &leaves.iter().map(|leaf| leaf.radius).collect::<Vec<_>>(),
                &[],
            )
            .map_err(|error| error.to_string())?;
        queue
            .enqueue_write_buffer(&mut rows, CL_BLOCKING, 0, base, &[])
            .map_err(|error| format!("failed to write OpenCL base row: {error}"))?;
        queue
            .enqueue_write_buffer(&mut scratch.tiles, CL_BLOCKING, 0, &tiles, &[])
            .map_err(|error| format!("failed to write OpenCL tiles: {error}"))?;
        queue
            .enqueue_write_buffer(
                &mut scratch.leaves,
                CL_BLOCKING,
                0,
                &make_steps(leaves, 0.0),
                &[],
            )
            .map_err(|error| error.to_string())?;
    }
    Ok((scratch, rows, radii, shadow, tiles.len()))
}

fn buffer<T>(
    context: &Context,
    len: usize,
    flags: cl_mem_flags,
    name: &str,
) -> Result<Buffer<T>, String> {
    if len == 0 {
        return Err(format!("{name} buffer cannot be empty"));
    }
    unsafe {
        Buffer::<T>::create(context, flags, len, ptr::null_mut())
            .map_err(|error| format!("failed to allocate OpenCL {name}: {error}"))
    }
}

fn to_u32(value: usize, name: &str) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| format!("{name} exceeds u32 range"))
}
