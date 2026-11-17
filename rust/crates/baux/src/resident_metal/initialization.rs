use super::*;

pub(super) struct PoolBufferPlan {
    pub(super) partial_count: usize,
    pub(super) aggregate_count: usize,
    pub(super) distance_count: usize,
    pub(super) geometry_count: usize,
    pub(super) cand_count: usize,
    pub(super) sizes: [u64; 12],
    pub(super) resident_bytes: u64,
}

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
                ProposalDistribution::Gaussian,
            )
        })
    }

    pub fn new_unscored(
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
                ProposalDistribution::Gaussian,
            )
        })
    }

    /// Full FP16 weight rows with independent per-coordinate innovations.
    pub fn new_fp16(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalDistribution,
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

    pub fn new_implicit(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalDistribution,
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

    pub fn new_mapped(
        base_buf: Buffer,
        dimensions: usize,
        blocks: Vec<ParamBlock>,
        resident_capacity: usize,
        length: TRLengthConfig,
        perturbation: ProposalDistribution,
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
    pub fn new_inner(
        base_input: Result<&[u16], (Buffer, usize)>,
        initial: Option<(f32, f32)>,
        blocks: Vec<ParamBlock>,
        capacity: usize,
        pending_capacity: usize,
        length: TRLengthConfig,
        independent_fp16: bool,
        implicit_history: bool,
        perturbation: ProposalDistribution,
    ) -> Result<Self, String> {
        let single_pool = matches!(base_input, Err(_));
        if implicit_history && capacity == 0 {
            return Err("Metal implicit history requires resident_capacity >= 1".into());
        }
        let (dimensions, length) = match &base_input {
            Ok(slice) => (
                slice.len(),
                validate_base(
                    slice,
                    initial,
                    capacity,
                    pending_capacity,
                    length,
                    independent_fp16,
                )?,
            ),
            Err((_, dims)) => (*dims, checked_length(length)?),
        };
        let res_cap = capacity;
        let window_cap = if implicit_history {
            MAX_HISTORY
        } else {
            capacity.max(1)
        };
        let pool_layout = if single_pool {
            crate::procedural_pool::ProceduralPoolConfig {
                arms: 1,
                candidates_per_arm: 1,
            }
            .resolve_resident()?
        } else {
            crate::procedural_pool::ProceduralPool::legacy()
        };
        let (tiles, offsets, leaves) = make_layout(&blocks, dimensions)?;
        let plan = allocation_sizes(
            dimensions,
            blocks.len(),
            tiles.len(),
            offsets.len(),
            res_cap,
            pool_layout.count() as usize,
        )?;
        let runtime = Runtime::shared()?;
        preflight(&runtime, &plan.sizes, plan.resident_bytes)?;
        let [
            propose_pipeline,
            initial_pipeline,
            pool_pipeline,
            reduction_pipeline,
            score_pipeline,
            selection_pipeline,
            materialize_pipeline,
            import_pipeline,
            import_reduction,
        ] = compile_pipelines(&runtime, independent_fp16, perturbation)?;
        let (base_buf, history_rows) = match base_input {
            Ok(slice) => {
                let base_buf = Arc::new(runtime.buffer_with(slice));
                let history_rows = (0..capacity)
                    .map(|_| Arc::new(runtime.buffer::<u16>(dimensions)))
                    .collect();
                (base_buf, history_rows)
            }
            Err((buf, _)) => {
                let mut history_rows = Vec::with_capacity(capacity);
                let base_buf = Arc::new(buf);
                history_rows.push(Arc::clone(&base_buf));
                for _ in 1..capacity {
                    history_rows.push(Arc::new(runtime.buffer::<u16>(dimensions)));
                }
                (base_buf, history_rows)
            }
        };
        let mut trust = TurboTrustRegion::new(dimensions, length);
        trust.set_arms(pool_layout.arms() as usize);
        let sampler = manifold::ResidentSampler::from_distribution(perturbation);
        let mut state = Self {
            independent_fp16,
            perturbation,
            sampler,
            num_pert: dimensions.max(1),
            sobol_offset: 0,
            lhd_points: 1,
            box_bounded: false,
            noise_aware: true,
            adapt_controller: true,
            success_tolerance_override: None,
            leaves_gpu: runtime.buffer_with(&leaves),
            tiles_gpu: runtime.buffer_with(&tiles),
            offsets_gpu: runtime.buffer_with(&offsets),
            dummy_meta_gpu: runtime.buffer_with(&[DimMeta::default()]),
            dummy_sobol_gpu: runtime.buffer_with(&[0u32; 32]),
            sobol_dirs_gpu: None,
            manifold: None,
            history_rows,
            base: base_buf,
            proposal: Arc::new(runtime.buffer::<u16>(dimensions)),
            partials: Arc::new(runtime.buffer::<Partial>(plan.partial_count)),
            pool_aggregates: Arc::new(runtime.buffer::<Partial>(plan.aggregate_count)),
            candidate_scores: Arc::new(runtime.buffer::<CandidateScore>(plan.cand_count)),
            selected_indices_gpu: Arc::new(runtime.buffer::<u32>(plan.cand_count)),
            axis_candidates_gpu: Arc::new(runtime.buffer::<f32>(plan.cand_count)),
            decision: Arc::new(runtime.buffer::<Decision>(1)),
            pool_distances: Arc::new(runtime.buffer::<f32>(plan.distance_count)),
            pool_geometry: Arc::new(runtime.buffer::<f32>(plan.geometry_count)),
            runtime,
            blocks,
            tiles,
            propose_pipeline,
            initial_pipeline,
            pool_pipeline,
            reduction_pipeline,
            score_pipeline,
            selection_pipeline,
            materialize_pipeline,
            import_pipeline,
            import_reduction,
            dimensions,
            resident_bytes: plan.resident_bytes,
            length_config: length,
            length: length.length_init,
            best: f32::NEG_INFINITY,
            best_variance: 0.0,
            objective_selector: None,
            pool_layout,
            objective_history: crate::objective_observation::ObjectiveWindow::new(window_cap),
            outcomes: vec![0.0; window_cap],
            variances: vec![0.0; window_cap],
            identities: vec![1; window_cap],
            observation: 1,
            base_id: 1,
            trust,
            trust_observations: 0,
            restart_count: 0,
            history: 0,
            resident_history: 0,
            resident_identities: vec![1; res_cap.max(1)],
            pairwise_distances: vec![0.0; MAX_HISTORY * MAX_HISTORY],
            distance_metric: crate::config::DistanceMetric::Euclidean,
            local_scale_neighbors: 8,
            implicit_history,
            initial_observations: 0,
            fit_candidates: 0,
            fit_samples: 0,
            fit_params: false,
            adapt_neighbors: false,
            fit_seed: 0,
            fitter: None,
            fitted_enn: None,
            next_id: 0,
            pending: None,
            evaluation: None,
            queued: None,
            started: false,
            profiling: false,
            last_profile: None,
            async_command: None,
            profile_commands: Vec::new(),
            tell_profile: None,
            axis: None,
        };
        if !single_pool && !state.history_rows.is_empty() {
            state.copy(&*state.base, &*state.history_rows[0])?;
        }
        if let Some((value, variance)) = initial {
            state = state.observe_initial(value, variance)?;
        }
        Ok(state)
    }
}

pub(super) fn metric_count(pool_count: usize) -> Result<usize, String> {
    if cfg!(test) {
        pool_count
            .max(4)
            .checked_add(6)
            .ok_or_else(|| "BF16 geometry metric overflow".into())
    } else {
        Ok(pool_count.max(1))
    }
}

pub(super) fn allocation_sizes(
    dimensions: usize,
    blocks: usize,
    tiles: usize,
    offsets: usize,
    capacity: usize,
    pool_count: usize,
) -> Result<PoolBufferPlan, String> {
    let pool_count = pool_count.max(1);
    let row_bytes = bytes::<u16>(dimensions)?;
    let pairs = capacity.max(1).div_ceil(2);
    let partial_count = tiles
        .checked_mul(pool_count)
        .and_then(|n| n.checked_mul(pairs))
        .ok_or("BF16 partial count overflow")?;
    let aggregate_count = pool_count
        .checked_mul(pairs)
        .ok_or("BF16 aggregate count overflow")?;
    let distance_count = pool_count
        .checked_mul(MAX_HISTORY)
        .ok_or("BF16 pool distance count overflow")?
        .max(2 * MAX_HISTORY + 1);
    let geometry_metrics = metric_count(pool_count)?;
    let geometry_count = tiles
        .checked_mul(geometry_metrics)
        .ok_or("BF16 geometry count overflow")?;
    let cand_count = pool_count.max(4);
    let sizes = [
        row_bytes,
        bytes::<Leaf>(blocks)?,
        bytes::<Tile>(tiles)?,
        bytes::<u32>(offsets)?,
        bytes::<Partial>(partial_count)?,
        bytes::<Partial>(aggregate_count)?,
        bytes::<CandidateScore>(cand_count)?,
        bytes::<u32>(cand_count)?,
        bytes::<f32>(cand_count)?,
        bytes::<Decision>(1)?,
        bytes::<f32>(distance_count)?,
        bytes::<f32>(geometry_count)?,
    ];
    let resident_bytes = sizes.iter().try_fold(
        row_bytes
            .checked_mul(capacity as u64 + 2)
            .ok_or("BF16 memory overflow")?,
        |sum, size| sum.checked_add(*size).ok_or("BF16 memory overflow"),
    )?;
    Ok(PoolBufferPlan {
        partial_count,
        aggregate_count,
        distance_count,
        geometry_count,
        cand_count,
        sizes,
        resident_bytes,
    })
}

pub fn search_source(independent_fp16: bool, _perturbation: ProposalDistribution) -> String {
    let mut defines = format!("#define BAUX_HISTORY {MAX_HISTORY}\n");
    if independent_fp16 {
        defines.push_str("#define FP16_INDEPENDENT\n");
    }
    #[cfg(test)]
    defines.push_str("#define POOL_GEOMETRY_DIAGNOSTICS\n");
    let mut tables = baux::semantics::abi::metal_abi();
    tables.push_str(&metal_tables());
    format!(
        "{defines}{}",
        SOURCE.replacen("// BAUX_ZIGGURAT_TABLES", &tables, 1)
    )
}

fn compile_pipelines(
    runtime: &Runtime,
    independent_fp16: bool,
    perturbation: ProposalDistribution,
) -> Result<[ComputePipelineState; 9], String> {
    let source = search_source(independent_fp16, perturbation);
    let pipelines = [
        runtime.pipeline(&source, "Dense search", "dense_propose")?,
        runtime.pipeline(&source, "Dense search", "dense_propose_initial")?,
        runtime.pipeline(&source, "Dense search", "dense_propose_pool")?,
        runtime.precise(&source, "Dense search", "dense_reduce_pool")?,
        runtime.precise(&source, "Dense search", "dense_score_candidates")?,
        runtime.precise(&source, "Dense search", "dense_select")?,
        runtime.pipeline(&source, "Dense search", "dense_materialize")?,
        runtime.precise(&source, "Dense search", "import_partial")?,
        runtime.precise(&source, "Dense search", "import_reduce")?,
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
    if capacity > MAX_HISTORY || pending_capacity != 1 {
        return Err(
            "Metal correlated search requires capacity in 0..=128 and max_pending=1".into(),
        );
    }
    if let Some((value, variance)) = initial {
        check_scores(&[value], &[variance])?;
    }
    let exponent_mask = if independent_fp16 { 0x7c00 } else { 0x7f80 };
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(target_arch = "wasm32")]
    let has_nonfinite = base
        .iter()
        .any(|bits| bits & exponent_mask == exponent_mask);
    if has_nonfinite {
        return Err("Base weights must be finite".into());
    }
    Ok(length)
}
