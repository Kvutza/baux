use std::sync::Arc;

use baux::semantics::neighbor::simd::{simd_l2, simd_scaled};
use baux::semantics::neighbor::{
    DistanceMetric, MetricBound, PoolBound, certify_metric, is_certified, local_scales,
    metric_delta,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::tie_break::topk_dists;
use crate::error::ENNError;
use crate::model::ENN;

pub type ScaleInfo = (Arc<[f64]>, usize, f64, f64);

pub fn squared_l2(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    scale_x: bool,
    x_scale: ArrayView1<f64>,
) -> f64 {
    if let (Some(xs), Some(ys)) = (x.as_slice(), y.as_slice()) {
        if !scale_x {
            return simd_l2(xs, ys);
        }
        if let Some(ss) = x_scale.as_slice() {
            return simd_scaled(xs, ys, ss);
        }
    }
    let mut acc = 0.0;
    if scale_x {
        for i in 0..x.len() {
            let sc = x_scale[i];
            let d = x[i] / sc - y[i] / sc;
            acc += d * d;
        }
    } else {
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let d = xi - yi;
            acc += d * d;
        }
    }
    acc.max(0.0)
}

pub(crate) fn row_query(model: &ENN, x_row: ArrayView1<f64>) -> Vec<f64> {
    let n_train = model.num_obs();
    (0..n_train)
        .map(|j| {
            let t = model.rows().row_x(j).expect("row_x");
            squared_l2(x_row, t.view(), model.scale_x, model.x_scale.view())
        })
        .collect()
}

pub(crate) fn distance_matrix(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    scale: bool,
    x_scale: &ArrayView1<f64>,
) -> Array2<f64> {
    let (x_s, y_s) = if scale {
        (
            x / &x_scale.view().insert_axis(Axis(0)),
            y / &x_scale.view().insert_axis(Axis(0)),
        )
    } else {
        (x.to_owned(), y.to_owned())
    };
    let xx: Array1<f64> = x_s.map_axis(Axis(1), |r| r.dot(&r));
    let yy: Array1<f64> = y_s.map_axis(Axis(1), |r| r.dot(&r));
    let xy = x_s.dot(&y_s.t());
    let d2 = xx.view().insert_axis(Axis(1)).to_owned() + yy.view().insert_axis(Axis(0)).to_owned()
        - 2.0 * xy;
    d2.mapv(|v| v.max(0.0))
}

pub(crate) fn train_scales(
    model: &ENN,
    metric: DistanceMetric,
) -> Result<Option<ScaleInfo>, ENNError> {
    let local_k = match metric {
        DistanceMetric::Euclidean => return Ok(None),
        DistanceMetric::Conformal { local_k }
        | DistanceMetric::Hyperbolic { local_k }
        | DistanceMetric::Wasserstein { local_k } => local_k.max(1),
    };
    if metric == model.dist_metric {
        if let Some(cached) = model.scale_cache.get() {
            return Ok(Some(cached.clone()));
        }
    }
    let n = model.num_obs();
    let d2 = if let Some(tx) = model.x_opt() {
        distance_matrix(&tx, &tx, model.scale_x, &model.x_scale.view())
    } else {
        let ids: Vec<usize> = (0..n).collect();
        let (tx, _, _) = model.rows().train_rows(&ids)?;
        distance_matrix(&tx.view(), &tx.view(), model.scale_x, &model.x_scale.view())
    };
    let sc: Arc<[f64]> = local_scales(n, local_k, |r, c| d2[[r, c]])
        .into_iter()
        .map(|s2| s2.max(1.0e-12).sqrt())
        .collect::<Vec<_>>()
        .into();
    let s_min = sc.iter().copied().fold(f64::INFINITY, f64::min).max(1.0e-6);
    let s_max = sc.iter().copied().fold(1.0e-6_f64, f64::max).max(s_min);
    let info = (sc, local_k, s_min, s_max);
    if metric == model.dist_metric {
        let _ = model.scale_cache.set(info.clone());
    }
    Ok(Some(info))
}

pub(crate) fn metric_row(
    model: &ENN,
    q_row: ArrayView1<'_, f64>,
    pairs: &[(f64, i64)],
    (k, skip, pool, tie): (usize, usize, PoolBound, bool),
    (metric, sc, local_k, s_min, s_max): (DistanceMetric, &[f64], usize, f64, f64),
) -> Vec<(f64, i64)> {
    let mut pos_d2: Vec<f64> = pairs
        .iter()
        .map(|&(d, _)| d)
        .filter(|&d| d.is_finite() && d > 0.0)
        .collect();
    pos_d2.sort_by(f64::total_cmp);
    let scale_certified = (pos_d2.len() >= local_k
        && is_certified(pos_d2[local_k.saturating_sub(1)], pool))
        || !pool.b32.is_finite();
    if !tie && scale_certified {
        let s_query = pos_d2
            .get(local_k.saturating_sub(1))
            .or_else(|| pos_d2.last())
            .map(|&d2| d2.max(1.0e-12).sqrt())
            .unwrap_or(s_min)
            .max(1.0e-6);
        let cands: Vec<(f64, i64)> = pairs
            .iter()
            .map(|&(d2, t)| (metric_delta(metric, d2, s_query, sc[t as usize]), t))
            .collect();
        let mb = MetricBound {
            pool,
            s_query,
            s_min,
            s_max,
        };
        if let Some(top) = certify_metric(metric, &cands, k, mb) {
            return top;
        }
    }
    let d2_all = row_query(model, q_row);
    let mut all_pos: Vec<f64> = d2_all
        .iter()
        .copied()
        .filter(|&d| d.is_finite() && d > 0.0)
        .collect();
    all_pos.sort_by(f64::total_cmp);
    let sq = all_pos
        .get(local_k.min(all_pos.len()).saturating_sub(1))
        .map(|&d2| d2.max(1.0e-12).sqrt())
        .unwrap_or(s_min)
        .max(1.0e-6);
    let dm: Vec<f64> = d2_all
        .iter()
        .enumerate()
        .map(|(t, &d2)| metric_delta(metric, d2, sq, sc[t]))
        .collect();
    topk_dists(&dm, (k + skip).min(sc.len()), true)
        .into_iter()
        .skip(skip)
        .take(k)
        .map(|t| (dm[t], t as i64))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use baux::semantics::neighbor::DistanceMetric;
    use ndarray::array;

    use crate::IndexDriver;
    use crate::model::ENN;

    #[test]
    fn distances_match() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
        let dists = crate::posterior::neighbor_dist::row_query(&model, array![0.5, 0.0].view());
        assert_eq!(dists.len(), 2);
        let d = crate::posterior::neighbor_dist::squared_l2(
            array![0.0, 0.0, 1.0, 2.0, 3.0].view(),
            array![1.0, 0.0, 1.0, 2.0, 1.0].view(),
            false,
            array![1.0, 1.0, 1.0, 1.0, 1.0].view(),
        );
        assert!((d - 5.0).abs() < 1e-12);
    }

    #[test]
    fn scale_cache() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let m = DistanceMetric::Conformal { local_k: 1 };
        let model = ENN::new(train_x, train_y, None, false, IndexDriver::Exact)
            .unwrap()
            .with_geometry(m);
        let s1 = super::train_scales(&model, m).unwrap().unwrap();
        let cloned = model.clone();
        let s2 = super::train_scales(&cloned, m).unwrap().unwrap();
        assert!(Arc::ptr_eq(&s1.0, &s2.0));
        let grown = model
            .append_cow(&array![[6.0, 0.0]].view(), &array![[3.0]].view(), None)
            .unwrap();
        let s3 = super::train_scales(&grown, m).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&s1.0, &s3.0));
        assert_eq!(s3.0.len(), 4);
    }
}
