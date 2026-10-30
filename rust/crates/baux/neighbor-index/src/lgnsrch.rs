//! Large-N BPANN search: indexed fragments + pending mmap brute-force leg.

use ndarray::Array2;
use rayon::prelude::*;

use baux::neighbor_index::bpann_store::BpannStore;
use baux::neighbor_index::distance::bpann_f32;
use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::index::{MmapSearchStore, bpann_mmap};
use baux::neighbor_index::merge::{merge_candidates, merge_dist};
use baux::neighbor_index::parallel::use_rayon;

pub struct SearchPendingArgs<'a> {
    pub total: usize,
    pub k_eff: usize,
    pub pool_k: usize,
    pub exclude_nearest: bool,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
    pub num_dim: usize,
}

pub fn search_pending(
    bpann_store: &BpannStore,
    query_rows: &[Vec<f64>],
    dist2s: &mut Array2<f64>,
    indices: &mut Array2<i64>,
    args: SearchPendingArgs<'_>,
) -> Result<(), BpannError> {
    let SearchPendingArgs {
        total,
        k_eff,
        pool_k,
        exclude_nearest,
        scale_x,
        x_scale: x_scale_vec,
        num_dim,
    } = args;
    let indexed = bpann_store.index.indexed_rows;
    let pending_start = indexed;
    let has_pending = pending_start < total;
    let index_k = if exclude_nearest {
        pool_k.max(k_eff * 2)
    } else {
        k_eff
    };
    let train_x = &bpann_store.train_x;
    let score_query = |query_buf: &Vec<f64>| -> Result<(Vec<f64>, Vec<i64>), BpannError> {
        let mut query_f32 = Vec::with_capacity(num_dim);
        bpann_f32(query_buf, scale_x, x_scale_vec, &mut query_f32);
        let search_store = MmapSearchStore {
            train_x,
            scale_x,
            x_scale: x_scale_vec,
        };

        let leg_a = if indexed > 0 && !bpann_store.index.indices.is_empty() {
            bpann_store
                .index
                .search_candidates(&query_f32, index_k.max(1), Some(&search_store))?
        } else {
            Vec::new()
        };

        let leg_b = if has_pending {
            bpann_mmap(
                train_x,
                pending_start,
                total,
                query_buf,
                pool_k,
                scale_x,
                x_scale_vec,
            )?
        } else {
            Vec::new()
        };

        let merged = if scale_x {
            merge_candidates(
                train_x,
                query_buf,
                &leg_a,
                &leg_b,
                k_eff,
                pool_k,
                exclude_nearest,
                scale_x,
                x_scale_vec,
            )?
        } else {
            merge_dist(&leg_a, &leg_b, k_eff, pool_k, exclude_nearest)
        };
        let mut dist_row = vec![0.0; k_eff];
        let mut idx_row = vec![0; k_eff];
        for (j, (id, dist)) in merged.into_iter().enumerate() {
            dist_row[j] = dist;
            idx_row[j] = id as i64;
        }
        Ok((dist_row, idx_row))
    };

    let per_query: Vec<(Vec<f64>, Vec<i64>)> = if use_rayon() {
        query_rows
            .par_iter()
            .map(score_query)
            .collect::<Result<Vec<_>, BpannError>>()?
    } else {
        query_rows
            .iter()
            .map(score_query)
            .collect::<Result<Vec<_>, BpannError>>()?
    };
    for (q, (dist_row, idx_row)) in per_query.into_iter().enumerate() {
        for j in 0..k_eff {
            dist2s[[q, j]] = dist_row[j];
            indices[[q, j]] = idx_row[j];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn propagates_error() {
        let dir = TempDir::new().unwrap();
        let mut store = BpannStore::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        store
            .append_rows(&array![[1.0, 2.0]].view(), &array![[0.0]].view(), None)
            .unwrap();
        let queries = vec![vec![1.0, 2.0]];
        let mut dist2s = Array2::zeros((1, 1));
        let mut indices = Array2::zeros((1, 1));
        let err = search_pending(
            &store,
            &queries,
            &mut dist2s,
            &mut indices,
            SearchPendingArgs {
                total: 5,
                k_eff: 1,
                pool_k: 1,
                exclude_nearest: false,
                scale_x: false,
                x_scale: &[1.0, 1.0],
                num_dim: 2,
            },
        )
        .unwrap_err();
        assert!(matches!(err, BpannError::InvalidParameter(_)));
    }
}
