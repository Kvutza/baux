use super::*;
use std::sync::atomic::Ordering;

impl SearchState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base: &[u16],
        base_value: f32,
        base_variance: f32,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                Ok(base),
                Some((base_value, base_variance)),
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                ProposalKernel::Gaussian,
            )
        })
    }

    pub(crate) fn new_unscored(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                Ok(base),
                None,
                blocks,
                capacity,
                pending_capacity,
                length,
                false,
                false,
                ProposalKernel::Gaussian,
            )
        })
    }

    /// Full FP16 weight rows with independent per-coordinate innovations.
    pub(crate) fn new_fp16(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalKernel,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                Ok(base),
                None,
                blocks,
                capacity,
                1,
                length,
                true,
                false,
                perturbation,
            )
        })
    }

    pub(crate) fn new_implicit(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalKernel,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                Ok(base),
                None,
                blocks,
                resident_capacity,
                1,
                length,
                true,
                true,
                perturbation,
            )
        })
    }

    pub(crate) fn new_mapped(
        base_buf: Buffer,
        dimensions: usize,
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalKernel,
    ) -> Result<Self, String> {
        autoreleasepool(|| {
            Self::new_inner(
                Err((base_buf, dimensions)),
                None,
                blocks,
                resident_capacity,
                1,
                length,
                true,
                true,
                perturbation,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_inner(
        base_input: Result<&[u16], (Buffer, usize)>,
        initial: Option<(f32, f32)>,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
        independent_fp16: bool,
        implicit_history: bool,
        perturbation: ProposalKernel,
    ) -> Result<Self, String> {
        let (dimensions, latent_init, length) = match &base_input {
            Ok(slice) => (
                slice.len(),
                false,
                validate_base(
                    slice,
                    initial,
                    capacity,
                    pending_capacity,
                    length,
                    independent_fp16,
                )?,
            ),
            Err((_, dims)) => (*dims, true, checked_length(length)?),
        };
        let res_cap = if latent_init { 1 } else { capacity };
        let window_cap = if implicit_history {
            MAX_HISTORY
        } else {
            capacity
        };
        let (tiles, offsets, leaves) = make_layout(&blocks, dimensions)?;
        let exact_history = independent_fp16 && implicit_history && !latent_init;
        let (partial_count, aggregate_count, geometry_count, replay_count, sizes, resident_bytes) =
            allocation_sizes(
                dimensions,
                blocks.len(),
                tiles.len(),
                offsets.len(),
                res_cap,
                exact_history,
            )?;
        let runtime = Runtime::shared()?;
        preflight(&runtime, &sizes, resident_bytes)?;
        let [
            propose_pipeline,
            initial_pipeline,
            pool_pipeline,
            reduction_pipeline,
            replay_short_pipeline,
            replay_reduction_pipeline,
            selection_pipeline,
            materialize_pipeline,
            reference_pipeline,
            rms_pipeline,
        ] = compile_pipelines(&runtime, independent_fp16, perturbation, latent_init)?;
        let (base_buf, history_rows, replay_origin) = match base_input {
            Ok(slice) => (
                runtime.buffer_with(slice),
                (0..capacity)
                    .map(|_| runtime.buffer::<u16>(dimensions))
                    .collect(),
                exact_history.then(|| runtime.buffer_with(slice)),
            ),
            Err((buf, _)) => (buf.clone(), vec![buf.clone()], exact_history.then(|| buf)),
        };
        let (replay_scale_count, replay_family_count) = if exact_history {
            (MAX_HISTORY * blocks.len(), 4 * MAX_HISTORY * FAMILIES)
        } else {
            (1, 1)
        };
        let mut trust = TurboTrustRegion::new(dimensions, length);
        trust.set_arms(1);
        let mut state = Self {
            independent_fp16,
            perturbation,
            leaves_gpu: runtime.buffer_with(&leaves),
            tiles_gpu: runtime.buffer_with(&tiles),
            offsets_gpu: runtime.buffer_with(&offsets),
            replay_origin,
            anchor: history_rows[0].clone(),
            rejected: history_rows[history_rows.len().min(2) - 1].clone(),
            history_rows,
            base: base_buf,
            proposal: runtime.buffer::<u16>(dimensions),
            reference: None,
            reference_scales: runtime.buffer_with(&vec![1.0f32; blocks.len()]),
            reference_partials: runtime.buffer::<f32>(tiles.len()),
            partials: runtime.buffer::<Partial>(partial_count),
            pool_aggregates: runtime.buffer::<Partial>(aggregate_count),
            decision: runtime.buffer::<Decision>(1),
            pool_distances: runtime.buffer::<f32>(5 * MAX_HISTORY),
            replay_steps: runtime.buffer::<ReplayStep>(MAX_HISTORY),
            replay_scales: runtime.buffer::<f32>(replay_scale_count),
            replay_partials: runtime.buffer::<f32>(replay_count),
            replay_components: runtime.buffer::<f32>(replay_count),
            pool_family_distances: runtime.buffer::<f32>(replay_family_count),
            family_groups: runtime.buffer_with(&vec![0u32; blocks.len()]),
            family_base_weights: runtime
                .buffer_with(&blocks.iter().map(|block| block.weight).collect::<Vec<_>>()),
            pool_geometry: runtime.buffer::<f32>(geometry_count),
            runtime,
            blocks,
            tiles,
            propose_pipeline,
            initial_pipeline,
            pool_pipeline,
            reduction_pipeline,
            replay_short_pipeline,
            replay_reduction_pipeline,
            selection_pipeline,
            materialize_pipeline,
            reference_pipeline,
            rms_pipeline,
            dimensions,
            resident_bytes,
            length_config: length,
            length: length.length_init,
            best: f32::NEG_INFINITY,
            best_variance: 0.0,
            objective_selector: None,
            pool_layout: if latent_init {
                crate::procedural_pool::ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()?
            } else {
                crate::procedural_pool::ProceduralPool::legacy()
            },
            proposal_method: crate::procedural_pool::ProposalMethod::Additive,
            objective_history: crate::objective_observation::ObjectiveWindow::new(window_cap),
            outcomes: vec![0.0; window_cap],
            variances: vec![0.0; window_cap],
            identities: vec![1; window_cap],
            observation: 1,
            base_id: 1,
            trust,
            reliability: None,
            trust_observations: 0,
            paired_evidence_sum: 0.0,
            paired_evidence_count: 0,
            restart_count: 0,
            history: 0,
            resident_history: 0,
            resident_identities: vec![1; res_cap],
            pairwise_distances: vec![0.0; MAX_HISTORY * MAX_HISTORY],
            distance_scaling: crate::config::DistanceScaling::Global,
            local_scale_neighbors: 8,
            family: None,
            metric_gpu: None,
            implicit_history,
            exact_history,
            latent_history: latent_init,
            neighbors: None,
            calibration: crate::calibration::Calibration::default(),
            initial_observations: 0,
            fit_candidates: 0,
            fit_samples: 0,
            fit_params: false,
            adapt_neighbors: false,
            fit_seed: 0,
            fitter: None,
            fitted_enn: None,
            failures: 0,
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            next_id: 0,
            pending: None,
            evaluation: None,
            queued: None,
            reference_seed: None,
            relative: false,
            started: false,
            poisoned: false,
            profiling: false,
            last_profile: None,
            async_command: None,
            profile_commands: Vec::new(),
            tell_profile: None,
            axis: None,
        };
        if !latent_init {
            state.copy(&state.base, &state.anchor)?;
        }
        if let Some((value, variance)) = initial {
            state.observe_initial(value, variance)?;
        }
        Ok(state)
    }
}

fn allocation_sizes(
    dimensions: usize,
    blocks: usize,
    tiles: usize,
    offsets: usize,
    capacity: usize,
    exact_history: bool,
) -> Result<(usize, usize, usize, usize, [u64; 18], u64), String> {
    let row_bytes = bytes::<u16>(dimensions)?;
    let partial_count = tiles
        .checked_mul(4)
        .and_then(|n| n.checked_mul(capacity.div_ceil(2)))
        .ok_or("BF16 partial count overflow")?;
    let aggregate_count = 4usize
        .checked_mul(capacity.div_ceil(2))
        .ok_or("BF16 aggregate count overflow")?;
    let geometry_count = tiles
        .checked_mul(POOL_METRICS)
        .ok_or("BF16 geometry count overflow")?;
    let replay_count = if exact_history {
        tiles
            .checked_mul(4 * MAX_HISTORY)
            .ok_or("BF16 replay partial count overflow")?
    } else {
        1
    };
    let replay_scales = if exact_history {
        MAX_HISTORY
            .checked_mul(blocks)
            .ok_or("BF16 replay scale count overflow")?
    } else {
        1
    };
    let replay_families = if exact_history {
        4 * MAX_HISTORY * FAMILIES
    } else {
        1
    };
    let sizes = [
        row_bytes,
        bytes::<Leaf>(blocks)?,
        bytes::<Tile>(tiles)?,
        bytes::<u32>(offsets)?,
        bytes::<f32>(blocks)?,
        bytes::<f32>(tiles)?,
        bytes::<Partial>(partial_count)?,
        bytes::<Partial>(aggregate_count)?,
        bytes::<Decision>(1)?,
        bytes::<f32>(5 * MAX_HISTORY)?,
        bytes::<f32>(geometry_count)?,
        bytes::<ReplayStep>(MAX_HISTORY)?,
        bytes::<f32>(replay_scales)?,
        bytes::<f32>(replay_count)?,
        bytes::<f32>(replay_count)?,
        bytes::<f32>(replay_families)?,
        bytes::<u32>(blocks)?,
        bytes::<f32>(blocks)?,
    ];
    let resident_bytes = sizes.iter().try_fold(
        row_bytes
            .checked_mul(capacity as u64 + 2 + u64::from(exact_history))
            .ok_or("BF16 memory overflow")?,
        |sum, size| sum.checked_add(*size).ok_or("BF16 memory overflow"),
    )?;
    Ok((
        partial_count,
        aggregate_count,
        geometry_count,
        replay_count,
        sizes,
        resident_bytes,
    ))
}

pub(super) fn search_source(independent_fp16: bool, perturbation: ProposalKernel) -> String {
    let mut defines = format!("#define BAUX_HISTORY {MAX_HISTORY}\n");
    if independent_fp16 {
        defines.push_str("#define FP16_INDEPENDENT\n");
    }
    if perturbation == ProposalKernel::Rademacher {
        defines.push_str("#define RADEMACHER_ONLY\n");
    }
    #[cfg(test)]
    defines.push_str("#define POOL_GEOMETRY_DIAGNOSTICS\n");
    let tables = if perturbation == ProposalKernel::Gaussian {
        metal_tables()
    } else {
        String::new()
    };
    format!(
        "{defines}{}",
        SOURCE.replacen("// BAUX_ZIGGURAT_TABLES", &tables, 1)
    )
}

fn compile_pipelines(
    runtime: &Runtime,
    independent_fp16: bool,
    perturbation: ProposalKernel,
    single_pool: bool,
) -> Result<[ComputePipelineState; 10], String> {
    let base_source = search_source(independent_fp16, perturbation);
    let source = if single_pool {
        format!("#define PROCEDURAL_SLOTS 1\n#define PROCEDURAL_COUNT 1\n{base_source}")
    } else {
        base_source
    };
    let pipelines = [
        runtime.pipeline(&source, "Dense search", "bf16_propose")?,
        runtime.pipeline(&source, "Dense search", "bf16_propose_initial")?,
        runtime.pipeline(&source, "Dense search", "bf16_propose_pool")?,
        runtime.precise(&source, "Dense search", "bf16_reduce_pool")?,
        runtime.precise(&source, "Dense search", "bf16_replay_short")?,
        runtime.precise(&source, "Dense search", "bf16_reduce_replay")?,
        runtime.precise(&source, "Dense search", "bf16_select")?,
        runtime.pipeline(&source, "Dense search", "bf16_materialize")?,
        runtime.precise(&source, "Dense search", "bf16_reference")?,
        runtime.precise(&source, "Dense search", "bf16_reference_rms")?,
    ];
    for pipeline in &pipelines {
        if pipeline.max_total_threads_per_threadgroup() < 256 {
            return Err("BF16 Metal kernels require 256 threads per group".into());
        }
        if pipeline.thread_execution_width() != 32 {
            return Err("BF16 Metal reductions require 32-lane SIMD groups".into());
        }
    }
    Ok(pipelines)
}

fn validate_base(
    base: &[u16],
    initial: Option<(f32, f32)>,
    capacity: usize,
    pending_capacity: usize,
    length: TRLengthConfig,
    independent_fp16: bool,
) -> Result<TRLengthConfig, String> {
    let length = checked_length(length)?;
    if capacity == 0 || capacity > MAX_HISTORY || pending_capacity != 1 {
        return Err(
            "Metal correlated search requires capacity in 1..=128 and max_pending=1".into(),
        );
    }
    if let Some((value, variance)) = initial {
        check_scores(&[value], &[variance])?;
    }
    let exponent_mask = if independent_fp16 { 0x7c00 } else { 0x7f80 };
    let has_nonfinite = if base.len() >= 65_536 {
        use rayon::prelude::*;
        base.par_chunks(65_536).any(|chunk| {
            chunk
                .iter()
                .any(|&bits| bits & exponent_mask == exponent_mask)
        })
    } else {
        base.iter()
            .any(|bits| bits & exponent_mask == exponent_mask)
    };
    if has_nonfinite {
        return Err("Base weights must be finite".into());
    }
    Ok(length)
}
