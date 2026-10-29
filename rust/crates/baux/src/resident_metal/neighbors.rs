//! Stratified coordinate quadrature for bounded model-state neighbour history.
//! Every proposal still materializes all coordinates. These sampled distances
//! are estimates of the weighted weight-space metric, not exact distances or
//! a proved sufficient representation of the objective.
use super::*;
use crate::apple_gpu::thread_group;

#[inline(always)]
fn decode_fp16(bits: u16) -> f64 {
    let exponent = (bits >> 10) & 31;
    let fraction = f64::from(bits & 1023);
    let value = match exponent {
        0 => fraction * 2.0f64.powi(-24),
        31 => f64::NAN,
        _ => (1024.0 + fraction) * 2.0f64.powi(i32::from(exponent) - 25),
    };
    if bits & 0x8000 == 0 { value } else { -value }
}

#[inline(always)]
fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    element: u64,
    leaf: u32,
    mass: f32,
}

pub(super) struct Neighbors {
    points: Vec<Point>,
    indices: Vec<usize>,
    descriptors: Buffer,
    history: Buffer,
    pool: Buffer,
    components: Buffer,
    sample: ComputePipelineState,
    distance: ComputePipelineState,
    history_coords: Vec<f64>,
    pool_coords: Vec<f64>,
    pool_deltas: Buffer,
}

impl Neighbors {
    pub(super) fn new(state: &SearchState) -> Result<Self, String> {
        let mut points = Vec::new();
        let mut indices = Vec::new();
        for (leaf, block) in state.blocks.iter().enumerate() {
            let count = (8192usize.saturating_mul(block.len) / state.dimensions)
                .max(8)
                .min(block.len);
            for stratum in 0..count {
                let start = stratum * block.len / count;
                let end = (stratum + 1) * block.len / count;
                let element = start
                    + crate::hash::splitmix64(
                        block.key ^ (stratum as u64).wrapping_mul(0x9e3779b97f4a7c15),
                    ) as usize
                        % (end - start);
                points.push(Point {
                    element: element as u64,
                    leaf: leaf as u32,
                    mass: (end - start) as f32,
                });
                indices.push(block.offset + element);
            }
        }
        let source = state.procedural_source(state.pool_layout);
        let n_pts = points.len();
        Ok(Self {
            descriptors: state.runtime.buffer_with(&points),
            history: state.runtime.buffer::<u16>(points.len() * MAX_HISTORY),
            pool: state.runtime.buffer::<u16>(points.len() * 4),
            components: state.runtime.buffer::<f32>(4 * MAX_HISTORY * FAMILIES),
            sample: state
                .runtime
                .pipeline(&source, "Neighbour geometry", "bf16_sample")?,
            distance: state.runtime.precise(
                &source,
                "Neighbour geometry",
                "bf16_sample_distances",
            )?,
            history_coords: vec![0.0_f64; n_pts * MAX_HISTORY],
            pool_coords: vec![0.0_f64; n_pts * 4],
            pool_deltas: state.runtime.buffer::<f32>(n_pts * 4),
            points,
            indices,
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.descriptors.length()
            + self.history.length()
            + self.pool.length()
            + self.components.length()
    }

    pub(super) fn capture(&mut self, row: usize, model: &Buffer, independent_fp16: bool) {
        assert!(row < MAX_HISTORY);
        // Called only while the search is idle or after the proposal completed.
        let n_pts = self.points.len();
        unsafe {
            for (sample, &index) in self.indices.iter().enumerate() {
                let bits = model.contents().cast::<u16>().add(index).read();
                self.history
                    .contents()
                    .cast::<u16>()
                    .add(row * n_pts + sample)
                    .write(bits);
                let coord = if independent_fp16 {
                    decode_fp16(bits)
                } else {
                    f64::from(decode_bf16(bits))
                };
                self.history_coords[row * n_pts + sample] = coord;
            }
        }
    }

    pub(super) fn archive(&mut self, row: usize, candidate: usize) {
        assert!(row < MAX_HISTORY && candidate < 4);
        let n_pts = self.points.len();
        self.history_coords[row * n_pts..(row + 1) * n_pts]
            .copy_from_slice(&self.pool_coords[candidate * n_pts..(candidate + 1) * n_pts]);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.pool
                    .contents()
                    .cast::<u16>()
                    .add(candidate * self.points.len()),
                self.history
                    .contents()
                    .cast::<u16>()
                    .add(row * self.points.len()),
                self.points.len(),
            );
        }
    }

    pub(super) fn encode(&self, state: &SearchState, command: &CommandBufferRef, root: u64) {
        let params = Params {
            stream_seed: root,
            tiles: self.points.len() as u32,
            history: state.history as u32,
            initialize: u32::from(state.family.is_some()),
            ..state.pool_params(root)
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.sample);
        for (i, buffer) in [
            &state.base,
            &state.leaves_gpu,
            &self.descriptors,
            &self.pool,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            4,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.set_buffer(5, Some(&self.pool_deltas), 0);
        encoder.dispatch_thread_groups(
            thread_group(self.points.len().div_ceil(256) as u64),
            thread_group(256),
        );
        encoder.end_encoding();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance);
        for (i, buffer) in [
            &self.pool,
            &self.history,
            &self.descriptors,
            &state.leaves_gpu,
            &state.pool_distances,
            &state.family_base_weights,
            &state.family_groups,
            &self.components,
        ]
        .iter()
        .enumerate()
        {
            encoder.set_buffer(i as u64, Some(buffer), 0);
        }
        encoder.set_bytes(
            8,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_thread_groups(thread_group((4 * state.history) as u64), thread_group(256));
        encoder.end_encoding();
    }

    pub(super) fn compute_distances(&mut self, state: &SearchState) -> Vec<Vec<f64>> {
        let n_pts = self.points.len();
        let base_index = state.identities[..state.history]
            .iter()
            .position(|&identity| identity == state.base_id)
            .unwrap_or(0);
        let deltas = read::<f32>(&self.pool_deltas, 4 * n_pts);
        for cand in 0..4 {
            let cand_deltas = &deltas[cand * n_pts..(cand + 1) * n_pts];
            let base_history = &self.history_coords[base_index * n_pts..(base_index + 1) * n_pts];
            let cand_pool = &mut self.pool_coords[cand * n_pts..(cand + 1) * n_pts];
            for s in 0..n_pts {
                cand_pool[s] = base_history[s] + f64::from(cand_deltas[s]);
            }
        }

        let mut matrix_f64 = vec![vec![0.0_f64; state.history]; 4];
        let mut f32_dists = vec![0.0_f32; 4 * state.history];
        let mut comps = vec![[0.0_f32; FAMILIES]; 4 * state.history];

        let fam_groups = read::<u32>(&state.family_groups, state.blocks.len());
        let fam_weights = read::<f32>(&state.family_base_weights, state.blocks.len());

        for cand in 0..4 {
            let cand_coords = &self.pool_coords[cand * n_pts..(cand + 1) * n_pts];
            for row in 0..state.history {
                let hist_coords = &self.history_coords[row * n_pts..(row + 1) * n_pts];
                let mut sum = 0.0_f64;
                let mut grouped = [0.0_f64; FAMILIES];
                for (s, pt) in self.points.iter().enumerate() {
                    let d = cand_coords[s] - hist_coords[s];
                    let term = d
                        * d
                        * f64::from(pt.mass)
                        * f64::from(state.blocks[pt.leaf as usize].weight);
                    sum += term;
                    if state.family.is_some() {
                        let g = fam_groups[pt.leaf as usize] as usize;
                        if g < FAMILIES {
                            let g_term = d
                                * d
                                * f64::from(pt.mass)
                                * f64::from(fam_weights[pt.leaf as usize]);
                            grouped[g] += g_term;
                        }
                    }
                }
                matrix_f64[cand][row] = sum;
                f32_dists[cand * state.history + row] = sum as f32;
                if state.family.is_some() {
                    for g in 0..FAMILIES {
                        comps[cand * state.history + row][g] = grouped[g] as f32;
                    }
                }
            }
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                f32_dists.as_ptr(),
                state.pool_distances.contents().cast::<f32>(),
                f32_dists.len(),
            );
            if state.family.is_some() {
                std::ptr::copy_nonoverlapping(
                    comps.as_ptr().cast::<f32>(),
                    self.components.contents().cast::<f32>(),
                    comps.len() * FAMILIES,
                );
            }
        }

        matrix_f64
    }

    pub(super) fn components(&self, candidate: usize, history: usize) -> Vec<[f32; FAMILIES]> {
        let values = read::<f32>(&self.components, 4 * history * FAMILIES);
        values[candidate * history * FAMILIES..(candidate + 1) * history * FAMILIES]
            .chunks_exact(FAMILIES)
            .map(|row| row.try_into().unwrap())
            .collect()
    }
}

impl SearchState {
    pub fn neighbor_error(&self, round: &Proposals) -> Option<f64> {
        self.neighbors.as_ref()?;
        let sampled = round
            .history_distances
            .iter()
            .find(|(id, _)| *id == round.base_id)?
            .1;
        let exact: f64 = round
            .changes
            .iter()
            .zip(&self.blocks)
            .map(|(change, block)| change.1 * f64::from(block.weight))
            .sum();
        (exact > 0.0).then(|| (f64::from(sampled) - exact).abs() / exact)
    }

    pub fn neighbor_samples(&self) -> usize {
        self.neighbors
            .as_ref()
            .map_or(0, |neighbors| neighbors.points.len())
    }

    pub fn calibration_info(&self) -> (usize, f32, Option<f64>) {
        (
            self.calibration.count(),
            self.calibration.multiplier(),
            self.calibration.coverage(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistanceScaling, HistoryGeometry, ResidentEnnConfig};
    use crate::procedural_pool::{ProceduralPoolConfig, ProposalMethod};

    fn search(arms: u32, slots: u32, mode: ProposalKernel, method: ProposalMethod) -> SearchState {
        // Include odd tensor starts, singleton leaves, tile boundaries, and
        // tails.
        let lengths = [TILE_ELEMENTS + 3, 1, 10, 19];
        let mut base = Vec::new();
        let mut blocks = Vec::new();
        for (leaf, length) in lengths.into_iter().enumerate() {
            blocks.push(ParamBlock::new(71 + leaf as u64, base.len(), length, 0.25, 1.0).unwrap());
            base.extend((0..length).map(|index| {
                let magnitude = 0x3000 + ((index * 17 + leaf * 31) % 2048) as u16;
                magnitude | (((index + leaf) & 1) as u16) << 15
            }));
        }
        let mut search =
            SearchState::new_implicit(&base, blocks, 1, TRLengthConfig::new(0.1, 0.001, 0.4), mode)
                .unwrap();
        search
            .configure_enn(ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms,
                    candidates_per_arm: slots,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: method,
                ask: {
                    let mut config = RoundConfig::default();
                    config.surrogate.neighbors = 1;
                    config
                },
                num_candidates: 4,
                num_samples: 2,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: DistanceScaling::Global,
                history_geometry: HistoryGeometry::Latent,
                local_scale_neighbors: 1,
            })
            .unwrap();
        search.observe_initial(0.0, 0.0).unwrap();
        search
    }

    #[test]
    fn sample_parity() {
        autoreleasepool(|| {
            for (arms, slots) in [(1, 1), (1, 4), (2, 2), (4, 1)] {
                for mode in [ProposalKernel::Gaussian, ProposalKernel::Rademacher] {
                    for method in [
                        ProposalMethod::Additive,
                        ProposalMethod::PairMix,
                        ProposalMethod::QuadraticFlow,
                    ] {
                        let mut search = search(arms, slots, mode, method);
                        for iteration in 0..8 {
                            let candidate = iteration % search.pool_layout.count() as usize;
                            let root = crate::hash::splitmix64(iteration as u64 + 71);
                            let direct = search.test_candidate(root, candidate).unwrap();
                            let direct = read::<u16>(&direct, search.dimensions);
                            let command = search.runtime.queue.new_command_buffer();
                            search.encode_proposal(
                                command,
                                Params {
                                    history: 1,
                                    base_slot: 0,
                                    ..search.params(root, candidate)
                                },
                            );
                            finish(command).unwrap();
                            assert_eq!(read::<u16>(&search.proposal, search.dimensions), direct);
                            if method == ProposalMethod::PairMix {
                                assert_eq!(search.pool_params(root).stream_seed, root);
                            }
                            search
                                .begin_forced(root, RoundConfig::default(), candidate)
                                .unwrap();
                            let round = search.finish_ask().unwrap();
                            let actual = read::<u16>(&search.proposal, search.dimensions);
                            assert_eq!(actual, direct);
                            let neighbors = search.neighbors.as_ref().unwrap();
                            let count = neighbors.points.len();
                            let pool = read::<u16>(&neighbors.pool, 4 * count);
                            let observed = neighbors
                                .indices
                                .iter()
                                .map(|&index| actual[index])
                                .collect::<Vec<_>>();
                            for (sample, &expected) in observed.iter().enumerate() {
                                assert_eq!(
                                    pool[candidate * count + sample],
                                    expected,
                                    "sample/materializer mismatch: {arms}x{slots} {mode:?} \
                                     {method:?} candidate={candidate} iteration={iteration} \
                                     sample={sample}"
                                );
                            }
                            let row = search.history;
                            search.tell_noisy(&round, -1.0, 0.0).unwrap();
                            search.sync().unwrap();
                            let neighbors = search.neighbors.as_ref().unwrap();
                            let history = read::<u16>(&neighbors.history, (row + 1) * count);
                            assert_eq!(&history[row * count..], observed);
                        }
                    }
                }
            }
        });
    }

    fn decode_half(bits: u16) -> f32 {
        let sign = if (bits & 0x8000) != 0 { -1.0 } else { 1.0 };
        let exp = ((bits >> 10) & 0x1f) as i32;
        let frac = (bits & 0x03ff) as f32 / 1024.0;
        sign * (1.0 + frac) * 2.0f32.powi(exp - 15)
    }

    fn check_fiber(sym: &[u16], ind: &[u16], scale: f32) {
        for (s_chunk, i_chunk) in sym.chunks_exact(4).zip(ind.chunks_exact(4)) {
            let raw: Vec<f32> = i_chunk.iter().map(|&b| decode_half(b) - 1.0).collect();
            let mu4: f32 = raw.iter().sum::<f32>() / 4.0;
            let sum: f32 = s_chunk.iter().map(|&b| decode_half(b) - 1.0).sum();
            assert!(sum.abs() < 0.01, "fiber sum {sum} exceeds tolerance");
            for (&s_b, &r) in s_chunk.iter().zip(&raw) {
                let actual = decode_half(s_b) - 1.0;
                let expected = scale * (r - mu4);
                assert!(
                    (actual - expected).abs() < 0.01,
                    "mismatch: {actual} vs {expected}"
                );
            }
        }
    }

    fn check_hierarchy(sym: &[u16], ind: &[u16]) {
        for (s_chunk, i_chunk) in sym.chunks_exact(8).zip(ind.chunks_exact(8)) {
            let raw: Vec<f32> = i_chunk.iter().map(|&b| decode_half(b) - 1.0).collect();
            let mu8: f32 = raw.iter().sum::<f32>() / 8.0;
            let s_del: Vec<f32> = s_chunk.iter().map(|&b| decode_half(b) - 1.0).collect();
            let s_mean: f32 = s_del.iter().sum::<f32>() / 8.0;
            assert!((s_mean - 0.50 * mu8).abs() < 0.01, "mean mismatch");
            for (&act, &r) in s_del.iter().zip(&raw) {
                let residual = act - s_mean;
                let exp_res = 1.05220856 * (r - mu8);
                assert!((residual - exp_res).abs() < 0.01, "residual mismatch");
            }
        }
    }

    fn symmetry_search(
        base: &[u16],
        blocks: Vec<ParamBlock>,
        method: ProposalMethod,
    ) -> SearchState {
        let mut search = SearchState::new_implicit(
            base,
            blocks,
            1,
            TRLengthConfig::new(0.1, 0.001, 0.4),
            ProposalKernel::Gaussian,
        )
        .unwrap();
        search
            .configure_enn(ResidentEnnConfig {
                pool: ProceduralPoolConfig {
                    arms: 1,
                    candidates_per_arm: 1,
                }
                .resolve_resident()
                .unwrap(),
                proposal_method: method,
                ask: {
                    let mut config = RoundConfig::default();
                    config.surrogate.neighbors = 1;
                    config
                },
                num_candidates: 4,
                num_samples: 2,
                fit_params: false,
                adapt_neighbors: false,
                distance_scaling: DistanceScaling::Global,
                history_geometry: HistoryGeometry::Latent,
                local_scale_neighbors: 1,
            })
            .unwrap();
        search.observe_initial(0.0, 0.0).unwrap();
        search
    }

    #[test]
    fn symmetry_quotient() {
        autoreleasepool(|| {
            let base = vec![0x3c00u16; 4 * 640];
            let blocks = vec![
                ParamBlock::new(101, 0, 640, 0.25, 1.0)
                    .unwrap()
                    .with_address(ProgramAddress::new(0, Some(0), None).unwrap()),
                ParamBlock::new(102, 640, 640, 0.25, 1.0)
                    .unwrap()
                    .with_address(ProgramAddress::new(8, Some(0), None).unwrap()),
                ParamBlock::new(103, 1280, 640, 0.25, 1.0)
                    .unwrap()
                    .with_address(ProgramAddress::new(7, None, None).unwrap()),
                ParamBlock::new(104, 1920, 640, 0.25, 1.0)
                    .unwrap()
                    .with_address(ProgramAddress::new(1, Some(0), None).unwrap()),
            ];
            let mut sym_search =
                symmetry_search(&base, blocks.clone(), ProposalMethod::QuadraticFlow);
            let mut ind_search = symmetry_search(&base, blocks, ProposalMethod::Additive);
            let sym_direct = sym_search.test_candidate(42, 0).unwrap();
            let ind_direct = ind_search.test_candidate(42, 0).unwrap();
            let sym = read::<u16>(&sym_direct, 4 * 640);
            let ind = read::<u16>(&ind_direct, 4 * 640);
            let sqrt_4_3 = (4.0f32 / 3.0f32).sqrt();

            check_fiber(&sym[0..640], &ind[0..640], sqrt_4_3);
            check_fiber(&sym[640..1280], &ind[640..1280], sqrt_4_3);
            check_hierarchy(&sym[1280..1920], &ind[1280..1920]);
            assert_eq!(&sym[1920..2560], &ind[1920..2560]);
        });
    }
}
