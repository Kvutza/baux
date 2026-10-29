use super::{Center, LeafStep, Parameter, RoundConfig, SparseEdit, Tile, make_steps, make_tiles};

fn cuda_config(config: RoundConfig) -> baux_cuda::RoundConfig {
    baux_cuda::RoundConfig {
        neighbors: config.surrogate.neighbors,
        acquisition: crate::weights::acquisition_code(config.acquisition.kind),
        epistemic_scale: config.surrogate.epistemic_scale,
        aleatoric_scale: config.surrogate.aleatoric_scale,
        y_scale: config.surrogate.output_scale,
        beta: config.acquisition.beta,
        adaptive_neighbors: false,
    }
}

pub(super) struct Engine {
    inner: baux_cuda::TrialEngine,
}

impl Engine {
    pub(super) fn new(base: &[u8], leaves: &[Parameter], slots: usize) -> Result<Self, String> {
        let steps = make_steps(leaves, 0.0);
        let tiles = make_tiles(leaves);
        Ok(Self {
            inner: baux_cuda::TrialEngine::new(
                base,
                &steps.iter().copied().map(cuda_leaf).collect::<Vec<_>>(),
                &tiles.iter().copied().map(cuda_tile).collect::<Vec<_>>(),
                slots,
            )?,
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
        let history_slots = history
            .iter()
            .map(|&(slot, _)| {
                u32::try_from(slot).map_err(|_| format!("CUDA history slot {slot} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = history.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let draws = crate::weights::history_draws(history, config.acquisition.seed);
        let steps = make_steps(leaves, config.proposal.radius)
            .into_iter()
            .map(cuda_leaf)
            .collect::<Vec<_>>();
        let (index, score) = self.inner.ask(
            base_slot,
            &history_slots,
            &outcomes,
            trial_slot,
            seeds,
            &draws,
            &steps,
            cuda_config(config),
            materialize_row,
        )?;
        Ok((index, score))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_sparse(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        trial_slot: usize,
        seeds: &[u64],
        edits: &[SparseEdit],
        num_pert: usize,
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<(usize, f32), String> {
        let history_slots = history
            .iter()
            .map(|&(slot, _)| {
                u32::try_from(slot).map_err(|_| "CUDA history slot exceeds u32".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = history.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let draws = crate::weights::history_draws(history, config.acquisition.seed);
        let steps = make_steps(leaves, config.proposal.radius)
            .into_iter()
            .map(cuda_leaf)
            .collect::<Vec<_>>();
        let edits = edits
            .iter()
            .map(|edit| baux_cuda::SparseEdit {
                leaf: edit.leaf,
                element: edit.element,
            })
            .collect::<Vec<_>>();
        self.inner.ask_sparse(
            base_slot,
            &history_slots,
            &outcomes,
            trial_slot,
            seeds,
            &draws,
            &edits,
            num_pert,
            &steps,
            cuda_config(config),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_regions(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            num_regions,
            seeds_per_region,
            None,
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ask_centers(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        seeds_per_region: usize,
        centers: &[Center],
        region_centers: &[usize],
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        self.regions_core(
            base_slot,
            history,
            region_centers.len(),
            seeds_per_region,
            Some((centers, region_centers)),
            seeds,
            leaves,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn regions_core(
        &mut self,
        base_slot: usize,
        history: &[(usize, f32)],
        num_regions: usize,
        seeds_per_region: usize,
        tree: Option<(&[Center], &[usize])>,
        seeds: &[u64],
        leaves: &[Parameter],
        config: RoundConfig,
    ) -> Result<Vec<(usize, f32)>, String> {
        let history_slots = history
            .iter()
            .map(|&(slot, _)| {
                u32::try_from(slot).map_err(|_| format!("CUDA history slot {slot} exceeds u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = history.iter().map(|&(_, value)| value).collect::<Vec<_>>();
        let draws = crate::weights::history_draws(history, config.acquisition.seed);
        let steps = make_steps(leaves, config.proposal.radius)
            .into_iter()
            .map(cuda_leaf)
            .collect::<Vec<_>>();
        let (centers, region_centers) = match tree {
            Some((centers, region_centers)) => {
                let centers = centers
                    .iter()
                    .map(cuda_center)
                    .collect::<Result<Vec<_>, String>>()?;
                let region_centers = region_centers
                    .iter()
                    .map(|&center| {
                        u32::try_from(center)
                            .map_err(|_| "CUDA region center exceeds u32".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (centers, region_centers)
            }
            None => (Vec::new(), Vec::new()),
        };
        self.inner.ask_multi(
            base_slot,
            &history_slots,
            &outcomes,
            num_regions,
            seeds_per_region,
            seeds,
            &draws,
            &centers,
            &region_centers,
            &steps,
            cuda_config(config),
        )
    }

    pub(super) fn materialize(
        &mut self,
        base_slot: usize,
        trial_slot: usize,
        seed: u64,
        steps: &[LeafStep],
    ) -> Result<(), String> {
        let steps = steps.iter().copied().map(cuda_leaf).collect::<Vec<_>>();
        self.inner.materialize(base_slot, trial_slot, seed, &steps)
    }

    pub(super) fn read(&self, slot: usize) -> Result<Vec<u8>, String> {
        self.inner.read(slot)
    }

    pub(super) fn device_row(&self, slot: usize) -> Result<(u64, usize, usize), String> {
        self.inner.device_row(slot)
    }

    pub(super) fn device_rows(&self, slots: &[usize]) -> Result<Vec<(u64, usize, usize)>, String> {
        self.inner.device_rows(slots)
    }

    pub(super) fn write(&mut self, slot: usize, row: &[u8]) -> Result<(), String> {
        self.inner.write(slot, row)
    }
}

fn cuda_leaf(step: LeafStep) -> baux_cuda::Leaf {
    baux_cuda::Leaf {
        byte_offset: step.byte_offset,
        element_offset: step.element_offset,
        length: step.length,
        bits: step.bits,
        encoding: step.encoding,
        scale: step.scale,
        weight: step.weight,
        whole: step.whole,
        threshold: step.threshold,
    }
}

fn cuda_tile(tile: Tile) -> baux_cuda::Tile {
    baux_cuda::Tile {
        leaf: tile.leaf,
        start: tile.start,
        length: tile.length,
        pad: tile.pad,
    }
}

fn cuda_center(center: &Center) -> Result<baux_cuda::CenterStep, String> {
    let parent = match center.parent {
        Some(parent) => {
            u32::try_from(parent).map_err(|_| "CUDA center parent exceeds u32".to_string())?
        }
        None => u32::MAX,
    };
    Ok(baux_cuda::CenterStep {
        parent,
        seed: baux_cuda::Seed {
            low: center.seed as u32,
            high: (center.seed >> 32) as u32,
        },
    })
}
