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
mod scratch;
pub(crate) use self::scratch::MetalPosterior;
use self::scratch::*;

#[derive(Clone)]
pub(crate) struct MetalIndex {
    runtime: Arc<Runtime>,
    distance: Vec<ComputePipelineState>,
    distance_choice: usize,
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
    rows: Arc<metal::Buffer>,
    row_norms: Arc<metal::Buffer>,
    host_rows: Arc<Vec<f32>>,
    host_norms: Arc<Vec<f32>>,
    row_capacity: usize,
    num_dim: usize,
    plan: Plan,
    request: KnnPlan,
}

impl MetalIndex {
    pub(super) fn supports_neighbors(k: usize) -> bool {
        (1..=MAX_K).contains(&k)
    }

    #[inline]
    pub fn lock(&self) -> Result<&Self, std::convert::Infallible> {
        Ok(self)
    }

    pub(crate) fn new(num_dim: usize, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::new_plan(num_dim, train, KnnPlan::Measured)
    }

    pub(crate) fn new_plan(
        num_dim: usize,
        train: &ArrayView2<f64>,
        request: KnnPlan,
    ) -> Result<Self, IndexError> {
        check_rows(train, num_dim)?;
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

        let host_rows = flatten_f32(train);
        let host_norms = norms(&host_rows, num_dim);
        let row_count = train.nrows();
        let row_capacity = next_capacity(row_count);
        let rows = runtime.buffer::<f32>(row_capacity * num_dim);
        let row_norms = runtime.buffer::<f32>(row_capacity);
        write_buffer(&rows, 0, &host_rows);
        write_buffer(&row_norms, 0, &host_norms);

        let plan = if let Some(plan) = direct_plan(request, 16)? {
            plan
        } else {
            Plan::Tree
        };

        Ok(Self {
            runtime,
            distance,
            distance_choice: 0,
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
            rows: Arc::new(rows),
            row_norms: Arc::new(row_norms),
            host_rows: Arc::new(host_rows),
            host_norms: Arc::new(host_norms),
            row_capacity,
            num_dim,
            plan,
            request,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.host_rows.len() / self.num_dim
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.row_capacity
            .saturating_mul(self.num_dim)
            .saturating_mul(std::mem::size_of::<f32>())
    }

    pub fn rebuild(self, train: &ArrayView2<f64>) -> Result<Self, IndexError> {
        Self::new_plan(self.num_dim, train, self.request)
    }

    pub fn add(mut self, rows: &ArrayView2<f64>, _start_key: u64) -> Result<Self, IndexError> {
        check_rows(rows, self.num_dim)?;
        if rows.nrows() == 0 {
            return Ok(self);
        }
        let old_len = self.len();
        let values = flatten_f32(rows);
        let row_norms = norms(&values, self.num_dim);

        Arc::make_mut(&mut self.host_rows).extend_from_slice(&values);
        Arc::make_mut(&mut self.host_norms).extend_from_slice(&row_norms);

        let new_len = old_len + rows.nrows();
        if new_len > self.row_capacity {
            let new_capacity = next_capacity(new_len);
            let new_rows = self.runtime.buffer::<f32>(new_capacity * self.num_dim);
            let new_norms_buf = self.runtime.buffer::<f32>(new_capacity);

            if old_len > 0 {
                let command_buffer = self.runtime.queue.new_command_buffer();
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &self.rows,
                    0,
                    &new_rows,
                    0,
                    (old_len * self.num_dim * std::mem::size_of::<f32>()) as u64,
                );
                blit.copy_from_buffer(
                    &self.row_norms,
                    0,
                    &new_norms_buf,
                    0,
                    (old_len * std::mem::size_of::<f32>()) as u64,
                );
                blit.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            write_buffer(&new_rows, old_len * self.num_dim, &values);
            write_buffer(&new_norms_buf, old_len, &row_norms);

            self.rows = Arc::new(new_rows);
            self.row_norms = Arc::new(new_norms_buf);
            self.row_capacity = new_capacity;
        } else if Arc::strong_count(&self.rows) == 1 && Arc::strong_count(&self.row_norms) == 1 {
            write_buffer(&self.rows, old_len * self.num_dim, &values);
            write_buffer(&self.row_norms, old_len, &row_norms);
        } else {
            let new_rows = self.runtime.buffer::<f32>(self.row_capacity * self.num_dim);
            let new_norms_buf = self.runtime.buffer::<f32>(self.row_capacity);

            if old_len > 0 {
                let command_buffer = self.runtime.queue.new_command_buffer();
                let blit = command_buffer.new_blit_command_encoder();
                blit.copy_from_buffer(
                    &self.rows,
                    0,
                    &new_rows,
                    0,
                    (old_len * self.num_dim * std::mem::size_of::<f32>()) as u64,
                );
                blit.copy_from_buffer(
                    &self.row_norms,
                    0,
                    &new_norms_buf,
                    0,
                    (old_len * std::mem::size_of::<f32>()) as u64,
                );
                blit.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }

            write_buffer(&new_rows, old_len * self.num_dim, &values);
            write_buffer(&new_norms_buf, old_len, &row_norms);

            self.rows = Arc::new(new_rows);
            self.row_norms = Arc::new(new_norms_buf);
        }

        Ok(self)
    }

    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        let (output, _profile) = self.search_profile(queries, k_eff, search_k)?;
        Ok(output)
    }

    pub fn search_profile(
        &self,
        queries: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<((Array2<f64>, Array2<i64>), Option<KnnProfile>), IndexError> {
        check_rows(queries, self.num_dim)?;
        if !Self::supports_neighbors(k_eff) {
            return Err(IndexError::InvalidParameter(format!(
                "Metal index supports 1..={MAX_K} neighbors, got {k_eff}"
            )));
        }
        if queries.nrows() == 0 || self.len() == 0 {
            return Ok((
                pad_k(
                    Array2::from_elem((queries.nrows(), 0), f64::INFINITY),
                    Array2::zeros((queries.nrows(), 0)),
                    search_k,
                ),
                None,
            ));
        }

        let mut scratch = Scratch::new(&self.runtime);
        let query_values = flatten_f32(queries);
        scratch.ensure(&self.runtime, self.num_dim, queries.nrows(), k_eff);
        write_buffer(&scratch.query, 0, &query_values);
        write_buffer(&scratch.query_norms, 0, &norms(&query_values, self.num_dim));

        let plan = if let Some(plan) = direct_plan(self.request, k_eff)? {
            plan
        } else if k_eff <= 16 {
            Plan::Tree
        } else {
            Plan::Wide
        };

        let command_buffer = self.runtime.queue.new_command_buffer();
        let start = Instant::now();
        if matches!(
            plan,
            Plan::Tree | Plan::Tiled | Plan::Simd | Plan::Gram | Plan::Wide
        ) {
            scratch.ensure_tree(
                &self.runtime,
                queries.nrows(),
                k_eff,
                plan_lists(plan, self.len()),
                if plan == Plan::Wide {
                    2
                } else {
                    plan_fan(k_eff)
                },
            );
            self.encode_tree(&scratch, plan, command_buffer, queries.nrows(), k_eff)
                .map_err(IndexError::InvalidParameter)?;
        } else {
            self.encode_serial(&scratch, command_buffer, plan, queries.nrows(), k_eff)
                .map_err(IndexError::InvalidParameter)?;
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let profile = KnnProfile {
            rows: self.len(),
            queries: queries.nrows(),
            dims: self.num_dim,
            k: k_eff,
            plan: plan.name(),
            gpu: start.elapsed(),
            scan: Duration::ZERO,
            select: Duration::ZERO,
            reduce: Duration::ZERO,
        };

        Ok((
            unpack_results(&scratch, queries.nrows(), k_eff, search_k),
            Some(profile),
        ))
    }

    fn encode_serial(
        &self,
        scratch: &Scratch,
        command: &metal::CommandBufferRef,
        plan: Plan,
        queries: usize,
        k: usize,
    ) -> Result<(), String> {
        scratch.encode_init(
            command,
            &self.init_results,
            self.len(),
            self.num_dim,
            queries,
            k,
        )?;
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
            self.encode_step(scratch, command, plan, &params, queries, k);
        }
        Ok(())
    }

    fn encode_step(
        &self,
        scratch: &Scratch,
        command: &metal::CommandBufferRef,
        plan: Plan,
        params: &Params,
        queries: usize,
        k: usize,
    ) {
        match plan {
            Plan::Split => {
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.distance[self.distance_choice]);
                encoder.set_buffer(0, Some(&self.rows), 0);
                encoder.set_buffer(1, Some(&scratch.query), 0);
                encoder.set_buffer(2, Some(&scratch.tile_distances), 0);
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
                encoder.set_buffer(0, Some(&scratch.tile_distances), 0);
                encoder.set_buffer(1, Some(&scratch.local_distances), 0);
                encoder.set_buffer(2, Some(&scratch.local_indices), 0);
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
                encoder.set_buffer(1, Some(&scratch.query), 0);
                encoder.set_buffer(2, Some(&scratch.local_distances), 0);
                encoder.set_buffer(3, Some(&scratch.local_indices), 0);
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
        encoder.set_buffer(0, Some(&scratch.result_distances), 0);
        encoder.set_buffer(1, Some(&scratch.result_indices), 0);
        encoder.set_buffer(2, Some(&scratch.local_distances), 0);
        encoder.set_buffer(3, Some(&scratch.local_indices), 0);
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
        None
    }

    fn encode_tree(
        &self,
        scratch: &Scratch,
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
            (&scratch.result_distances, &scratch.result_indices)
        } else {
            (&scratch.tree_distances, &scratch.tree_indices)
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
        encoder.set_buffer(1, Some(&scratch.query), 0);
        encoder.set_buffer(2, Some(scan_distances), 0);
        encoder.set_buffer(3, Some(scan_indices), 0);
        if plan == Plan::Gram {
            encoder.set_buffer(4, Some(&self.row_norms), 0);
            encoder.set_buffer(5, Some(&scratch.query_norms), 0);
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
            let reduce = if plan == Plan::Wide {
                &self.fold
            } else {
                &self.reduce16
            };
            scratch.reduce_pass(command, reduce, &params, queries, lists, groups, tree_input)?;
            lists = groups;
            tree_input = !tree_input;
        }
        Ok(())
    }

    fn distance_lanes(&self) -> u32 {
        [1, 2, 4, 4][self.distance_choice]
    }

    fn default_distance(&self, encoder: &metal::ComputeCommandEncoderRef, queries: usize) {
        dispatch_distance(
            encoder,
            self.distance_choice + 1 == self.distance.len(),
            queries,
        );
    }
}
