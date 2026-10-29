use std::sync::Arc;
use std::time::{Duration, Instant};

extern crate metal as mtl;
use mtl::{ComputePipelineState, MTLSize};
use ndarray::{Array2, ArrayView2};

use super::metal_plan::Plan;
use super::{KnnPlan, KnnProfile};
use super::{check_rows, flatten_f32, pad_k};
use crate::apple_gpu::Runtime;
use crate::error::IndexError;

const SOURCE: &str = include_str!("metal_index.metal");

mod posterior;
pub(crate) use self::posterior::MetalPosterior;
mod scratch;
use self::scratch::*;

pub(crate) struct MetalIndex {
    runtime: Arc<Runtime>,
    distance: Vec<ComputePipelineState>,
    distance_choice: Option<usize>,
    topk16: ComputePipelineState,
    fused16: ComputePipelineState,
    tree16: ComputePipelineState,
    tiled16: ComputePipelineState,
    simd16: ComputePipelineState,
    gram16: ComputePipelineState,
    reduce16: ComputePipelineState,
    wide: ComputePipelineState,
    fold: ComputePipelineState,
    local_topk: ComputePipelineState,
    merge: ComputePipelineState,
    init_results: ComputePipelineState,
    rows: metal::Buffer,
    row_norms: metal::Buffer,
    host_rows: Vec<f32>,
    host_norms: Vec<f32>,
    row_capacity: usize,
    scratch: Scratch,
    num_dim: usize,
    plan: Plan,
    request: KnnPlan,
    profile: Option<KnnProfile>,
}

impl MetalIndex {
    pub(super) fn supports_neighbors(k: usize) -> bool {
        (1..=MAX_K).contains(&k)
    }

    pub(crate) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::new_plan(num_dim, train, KnnPlan::Measured)
    }

    pub(crate) fn new_plan(
        num_dim: usize,
        train: &ArrayView2<f64>,
        request: KnnPlan,
    ) -> Result<Self, IndexError> {
        let runtime = Runtime::shared().map_err(IndexError::InvalidParameter)?;
        let pipeline = |name: &str| -> Result<ComputePipelineState, IndexError> {
            let result = runtime.pipeline(SOURCE, "index", name);
            result.map_err(IndexError::InvalidParameter)
        };
        let names = &[
            "distance_rows",
            "distance_rows_2",
            "distance_rows_4",
            "distance_simd",
        ];
        let distance = names
            .iter()
            .map(|name| pipeline(name))
            .collect::<Result<Vec<_>, _>>()?;
        let topk16 = pipeline("topk_16")?;
        let fused16 = pipeline("l2_topk_16")?;
        let tree16 = pipeline("l2_topk_16_batch")?;
        let tiled16 = pipeline("l2_topk_16_tiled")?;
        let simd16 = pipeline("l2_topk_16_simd")?;
        let gram16 = pipeline("l2_topk_16_gram")?;
        let reduce16 = pipeline("reduce_topk_16")?;
        let wide = pipeline("l2_topk_batch")?;
        let fold = pipeline("fold_topk")?;
        let local_topk = pipeline("local_topk")?;
        let merge = pipeline("merge_topk")?;
        let init_results = pipeline("init_results")?;
        let scratch = Scratch::new(&runtime);
        let mut index = Self {
            rows: runtime.buffer::<u32>(1),
            row_norms: runtime.buffer::<u32>(1),
            runtime,
            distance,
            distance_choice: None,
            topk16,
            fused16,
            tree16,
            tiled16,
            simd16,
            gram16,
            reduce16,
            wide,
            fold,
            local_topk,
            merge,
            init_results,
            host_rows: Vec::new(),
            host_norms: Vec::new(),
            row_capacity: 0,
            scratch,
            num_dim,
            plan: Plan::Split,
            request,
            profile: None,
        };
        index.rebuild(train)?;
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.host_rows.len() / self.num_dim
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.row_capacity
            .saturating_mul(self.num_dim)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub(crate) fn rebuild(&mut self, train: &ArrayView2<f64>) -> Result<(), IndexError> {
        check_rows(train, self.num_dim)?;
        self.host_rows = flatten_f32(train);
        self.host_norms = norms(&self.host_rows, self.num_dim);
        self.upload_rows();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        check_rows(rows, self.num_dim)?;
        let start = self.len();
        let values = flatten_f32(rows);
        let norms = norms(&values, self.num_dim);
        self.host_rows.extend_from_slice(&values);
        self.host_norms.extend_from_slice(&norms);
        if self.len() > self.row_capacity {
            self.upload_rows();
        } else {
            write_buffer(&self.rows, start * self.num_dim, &values);
            write_buffer(&self.row_norms, start, &norms);
        }
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        check_rows(queries, self.num_dim)?;
        if !Self::supports_neighbors(k_eff) {
            return Err(IndexError::InvalidParameter(format!(
                "Metal index supports 1..={MAX_K} neighbors, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok(pad_k(
                Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                Array2::zeros((queries.nrows(), 0)),
                search_k,
            ));
        }

        let query_values = flatten_f32(queries);
        self.scratch
            .ensure(&self.runtime, self.num_dim, queries.nrows(), k_eff);
        write_buffer(&self.scratch.query, 0, &query_values);
        write_buffer(
            &self.scratch.query_norms,
            0,
            &norms(&query_values, self.num_dim),
        );
        let plan = {
            self.calibrate_distance(queries.nrows())?;
            self.tile_plan(queries.nrows(), k_eff)?
        };
        self.plan = plan;

        let command_buffer = self.runtime.queue.new_command_buffer();
        let start = Instant::now();
        if matches!(
            plan,
            Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
        ) {
            self.encode_tree(plan, command_buffer, queries.nrows(), k_eff)
                .map_err(IndexError::InvalidParameter)?;
        } else {
            self.encode_serial(command_buffer, plan, queries.nrows(), k_eff)
                .map_err(IndexError::InvalidParameter)?;
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        self.profile = Some(KnnProfile {
            rows: self.len(),
            queries: queries.nrows(),
            dims: self.num_dim,
            k: k_eff,
            plan: plan.name(),
            gpu: start.elapsed(),
            scan: Duration::ZERO,
            select: Duration::ZERO,
            reduce: Duration::ZERO,
        });

        Ok(unpack_results(
            &self.scratch,
            queries.nrows(),
            k_eff,
            search_k,
        ))
    }

    fn encode_serial(
        &self,
        command: &metal::CommandBufferRef,
        plan: Plan,
        queries: usize,
        k: usize,
    ) -> Result<(), String> {
        let init = Params {
            rows: to_u32(self.len(), "row count").map_err(|e| e.to_string())?,
            dim: to_u32(self.num_dim, "dimension").map_err(|e| e.to_string())?,
            queries: to_u32(queries, "query count").map_err(|e| e.to_string())?,
            tile_start: 0,
            tile_rows: 0,
            k: to_u32(k, "neighbor count").map_err(|e| e.to_string())?,
            lanes: 1,
            tiles: 0,
            lists: 0,
            groups: 0,
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.init_results);
        encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
        encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
        set_params(&encoder, 2, &init);
        dispatch(&encoder, queries * k);
        encoder.end_encoding();

        for tile_start in (0..self.len()).step_by(TILE_ROWS) {
            let params = Params {
                rows: to_u32(self.len(), "row count").map_err(|e| e.to_string())?,
                dim: to_u32(self.num_dim, "dimension").map_err(|e| e.to_string())?,
                queries: to_u32(queries, "query count").map_err(|e| e.to_string())?,
                tile_start: to_u32(tile_start, "tile start").map_err(|e| e.to_string())?,
                tile_rows: to_u32((self.len() - tile_start).min(TILE_ROWS), "tile rows")
                    .map_err(|e| e.to_string())?,
                k: to_u32(k, "neighbor count").map_err(|e| e.to_string())?,
                lanes: self.distance_lanes(),
                tiles: 0,
                lists: 0,
                groups: 0,
            };
            self.encode_step(command, plan, &params, queries, k);
        }
        Ok(())
    }

    fn encode_step(
        &self,
        command: &metal::CommandBufferRef,
        plan: Plan,
        params: &Params,
        queries: usize,
        k: usize,
    ) {
        match plan {
            Plan::Split => {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(
                    &self.distance[self.distance_choice.expect("distance schedule calibrated")],
                );
                encoder.set_buffer(0, Some(&self.rows), 0);
                encoder.set_buffer(1, Some(&self.scratch.query), 0);
                encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
                set_params(&encoder, 3, params);
                self.default_distance(&encoder, queries);
                encoder.end_encoding();

                let encoder = command.new_compute_command_encoder();
                let topk = if k <= 16 {
                    &self.topk16
                } else {
                    &self.local_topk
                };
                encoder.set_compute_pipeline_state(topk);
                encoder.set_buffer(0, Some(&self.scratch.tile_distances), 0);
                encoder.set_buffer(1, Some(&self.scratch.local_distances), 0);
                encoder.set_buffer(2, Some(&self.scratch.local_indices), 0);
                set_params(&encoder, 3, params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: queries as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
            }
            Plan::Fused => {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.fused16);
                encoder.set_buffer(0, Some(&self.rows), 0);
                encoder.set_buffer(1, Some(&self.scratch.query), 0);
                encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
                encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
                set_params(&encoder, 4, params);
                encoder.dispatch_thread_groups(
                    MTLSize {
                        width: queries as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.end_encoding();
            }
            Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide => {
                unreachable!("tree plan encoded above")
            }
        }

        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.merge);
        encoder.set_buffer(0, Some(&self.scratch.result_distances), 0);
        encoder.set_buffer(1, Some(&self.scratch.result_indices), 0);
        encoder.set_buffer(2, Some(&self.scratch.local_distances), 0);
        encoder.set_buffer(3, Some(&self.scratch.local_indices), 0);
        set_params(&encoder, 4, params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: queries as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
    }

    pub(crate) fn plan(&self) -> &'static str {
        self.plan.name()
    }

    pub(crate) fn profile(&self) -> Option<KnnProfile> {
        self.profile.clone()
    }

    fn encode_tree(
        &self,
        plan: Plan,
        command: &metal::CommandBufferRef,
        queries: usize,
        k: usize,
    ) -> Result<(), String> {
        let scan_lists = plan_lists(plan, self.len());
        let params = Params {
            rows: to_u32(self.len(), "row count").map_err(|error| error.to_string())?,
            dim: to_u32(self.num_dim, "dimension").map_err(|error| error.to_string())?,
            queries: to_u32(queries, "query count").map_err(|error| error.to_string())?,
            tile_start: 0,
            tile_rows: 0,
            k: to_u32(k, "neighbor count").map_err(|error| error.to_string())?,
            lanes: if plan == Plan::Wide {
                1
            } else {
                self.distance_lanes()
            },
            tiles: to_u32(scan_lists, "list count").map_err(|error| error.to_string())?,
            lists: 0,
            groups: 0,
        };
        let (scan_distances, scan_indices) = if scan_lists == 1 {
            (&self.scratch.result_distances, &self.scratch.result_indices)
        } else {
            (&self.scratch.tree_distances, &self.scratch.tree_indices)
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(match plan {
            Plan::Tree => &self.tree16,
            Plan::Tiled => &self.tiled16,
            Plan::Simd => &self.simd16,
            Plan::Gram => &self.gram16,
            Plan::Wide => &self.wide,
            Plan::Split | Plan::Fused => unreachable!("serial plan cannot encode a tree"),
        });
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.query), 0);
        encoder.set_buffer(2, Some(scan_distances), 0);
        encoder.set_buffer(3, Some(scan_indices), 0);
        if plan == Plan::Gram {
            encoder.set_buffer(4, Some(&self.row_norms), 0);
            encoder.set_buffer(5, Some(&self.scratch.query_norms), 0);
            set_params(&encoder, 6, &params);
        } else {
            set_params(&encoder, 4, &params);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: if plan == Plan::Gram {
                    (queries.div_ceil(GRAM_QUERIES) * scan_lists) as u64
                } else {
                    (queries * scan_lists) as u64
                },
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let mut lists = scan_lists;
        let mut tree_input = true;
        while lists > 1 {
            let fan = if plan == Plan::Wide { 2 } else { plan_fan(k) };
            let groups = lists.div_ceil(fan);
            self.reduce_pass(plan, command, &params, queries, lists, groups, tree_input)?;
            lists = groups;
            tree_input = !tree_input;
        }
        Ok(())
    }

    fn reduce_pass(
        &self,
        plan: Plan,
        command: &metal::CommandBufferRef,
        params: &Params,
        queries: usize,
        lists: usize,
        groups: usize,
        tree_input: bool,
    ) -> Result<(), String> {
        let (input_distances, input_indices) = if tree_input {
            (&self.scratch.tree_distances, &self.scratch.tree_indices)
        } else {
            (
                &self.scratch.tree_alt_distances,
                &self.scratch.tree_alt_indices,
            )
        };
        let (output_distances, output_indices) = if groups == 1 {
            (&self.scratch.result_distances, &self.scratch.result_indices)
        } else if tree_input {
            (
                &self.scratch.tree_alt_distances,
                &self.scratch.tree_alt_indices,
            )
        } else {
            (&self.scratch.tree_distances, &self.scratch.tree_indices)
        };
        let params = Params {
            lists: to_u32(lists, "list count").map_err(|error| error.to_string())?,
            groups: to_u32(groups, "merge groups").map_err(|error| error.to_string())?,
            ..*params
        };
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(if plan == Plan::Wide {
            &self.fold
        } else {
            &self.reduce16
        });
        encoder.set_buffer(0, Some(input_distances), 0);
        encoder.set_buffer(1, Some(input_indices), 0);
        encoder.set_buffer(2, Some(output_distances), 0);
        encoder.set_buffer(3, Some(output_indices), 0);
        set_params(&encoder, 4, &params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (queries * groups) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        Ok(())
    }

    fn calibrate_distance(&mut self, queries: usize) -> Result<(), IndexError> {
        if self.distance_choice.is_some() {
            return Ok(());
        }
        let query_count = queries;
        let params = Params {
            rows: to_u32(self.len(), "row count")?,
            dim: to_u32(self.num_dim, "dimension")?,
            queries: to_u32(query_count, "query count")?,
            tile_start: 0,
            tile_rows: to_u32(self.len().min(TILE_ROWS), "tile rows")?,
            k: 1,
            lanes: 1,
            tiles: 0,
            lists: 0,
            groups: 0,
        };
        self.run_distance(0, &params);
        let reference = self.distance_values(query_count);
        let runtime = Arc::clone(&self.runtime);
        let choice = runtime
            .race(
                "knn.distance.v2",
                &[self.num_dim, query_count],
                self.distance.len(),
                5,
                |schedule| {
                    let start = Instant::now();
                    self.run_distance(schedule, &params);
                    let elapsed = start.elapsed();
                    if !same_distances(&reference, &self.distance_values(query_count)) {
                        return Ok(None);
                    }
                    Ok(Some(elapsed))
                },
            )
            .map_err(IndexError::InvalidParameter)?;
        self.distance_choice = Some(choice);
        Ok(())
    }

    fn tile_plan(&mut self, queries: usize, k: usize) -> Result<Plan, IndexError> {
        if let Some(plan) = direct_plan(self.request, k)? {
            if matches!(
                plan,
                Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
            ) {
                self.scratch.ensure_tree(
                    &self.runtime,
                    queries,
                    k,
                    plan_lists(plan, self.len()),
                    if plan == Plan::Wide { 2 } else { plan_fan(k) },
                );
            }
            return Ok(plan);
        }
        let gram = k <= 16 && queries >= GRAM_QUERIES;
        let tree_lists = if gram {
            self.len().div_ceil(GRAM_ROWS)
        } else {
            self.len().div_ceil(TILE_ROWS)
        };
        self.scratch.ensure_tree(
            &self.runtime,
            queries,
            k,
            tree_lists,
            if k <= 16 { plan_fan(k) } else { 2 },
        );
        let candidates = plan_options(gram, k);
        let query_count = queries;
        let mut reference = None;
        let runtime = Arc::clone(&self.runtime);
        let choice = runtime
            .race(
                "knn.plan.metal.v5",
                &[self.num_dim, self.len(), query_count, k],
                candidates.len(),
                5,
                |candidate| {
                    let plan = candidates[candidate];
                    if !plan.graph().valid() {
                        return Ok(None);
                    }
                    let elapsed = self.run_plan(plan, query_count, k)?;
                    let values = self.result_values(query_count, k);
                    if let Some(reference) = &reference {
                        if !same_topk(reference, &values, query_count, k, self.len()) {
                            return Ok(None);
                        }
                    } else {
                        reference = Some(values);
                    }
                    Ok(Some(elapsed))
                },
            )
            .map_err(IndexError::InvalidParameter)?;
        Ok(candidates[choice])
    }

    fn run_plan(&self, plan: Plan, queries: usize, k: usize) -> Result<Duration, String> {
        let start = Instant::now();
        let command_buffer = self.runtime.queue.new_command_buffer();

        if matches!(
            plan,
            Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
        ) {
            self.encode_tree(plan, command_buffer, queries, k)?;
        } else {
            self.encode_serial(command_buffer, plan, queries, k)?;
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(start.elapsed())
    }

    fn result_values(&self, queries: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
        unsafe {
            (
                std::slice::from_raw_parts(
                    self.scratch.result_distances.contents().cast::<f32>(),
                    queries * k,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    self.scratch.result_indices.contents().cast::<u32>(),
                    queries * k,
                )
                .to_vec(),
            )
        }
    }

    fn run_distance(&self, schedule: usize, params: &Params) {
        let command_buffer = self.runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.distance[schedule]);
        encoder.set_buffer(0, Some(&self.rows), 0);
        encoder.set_buffer(1, Some(&self.scratch.query), 0);
        encoder.set_buffer(2, Some(&self.scratch.tile_distances), 0);
        set_params(encoder, 3, params);
        self.scheduled_distance(encoder, schedule, params.queries as usize);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn distance_values(&self, queries: usize) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.scratch.tile_distances.contents().cast::<f32>(),
                queries * TILE_ROWS,
            )
            .to_vec()
        }
    }

    fn distance_lanes(&self) -> u32 {
        [1, 2, 4, 4][self.distance_choice.expect("distance schedule calibrated")]
    }

    fn default_distance(&self, encoder: &metal::ComputeCommandEncoderRef, queries: usize) {
        self.scheduled_distance(
            encoder,
            self.distance_choice.expect("distance schedule calibrated"),
            queries,
        );
    }

    fn scheduled_distance(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        schedule: usize,
        queries: usize,
    ) {
        if schedule + 1 == self.distance.len() {
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: (queries * TILE_ROWS).div_ceil(8) as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
        } else {
            dispatch(encoder, queries * TILE_ROWS);
        }
    }

    fn upload_rows(&mut self) {
        self.row_capacity = next_capacity(self.len());
        self.rows = self.runtime.buffer::<f32>(self.row_capacity * self.num_dim);
        self.row_norms = self.runtime.buffer::<f32>(self.row_capacity);
        write_buffer(&self.rows, 0, &self.host_rows);
        write_buffer(&self.row_norms, 0, &self.host_norms);
    }
}
