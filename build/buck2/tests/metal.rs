use baux::{ENN, ENNParams, IndexDriver, PosteriorFlags, compute_internals};
use ndarray::Array2;

#[test]
fn knn() {
    let rows = Array2::from_shape_fn((137, 7), |(i, j)| {
        ((i * 37 + j * 19 + 5) % 509) as f64 / 509.0
    });
    let values = Array2::from_shape_fn((137, 3), |(i, j)| {
        ((i * 13 + j * 23 + 7) % 521) as f64 / 97.0
    });
    let queries = Array2::from_shape_fn((29, 7), |(i, j)| {
        ((i * 31 + j * 11 + 3) % 503) as f64 / 503.0
    });
    let flags = PosteriorFlags::new().tie_neighbors(false);
    let exact_model = ENN::new(
        rows.clone(),
        values.clone(),
        None,
        false,
        IndexDriver::Exact,
    )
    .unwrap();

    for neighbors in [1, 3, 8, 10, 16, 17, 31, 64] {
        let params = ENNParams::new(neighbors, 0.7, 0.13).unwrap();
        let expected = compute_internals(&exact_model, &queries.view(), &params, &flags).unwrap();
        let driver = IndexDriver::Metal;
        let model = ENN::new(rows.clone(), values.clone(), None, false, driver).unwrap();
        let actual = compute_internals(&model, &queries.view(), &params, &flags).unwrap();
        assert_eq!(actual.idx, expected.idx, "{driver:?}, k={neighbors}");
        assert!(
            actual
                .mu
                .iter()
                .zip(expected.mu.iter())
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-5)
        );
        assert!(
            actual
                .se
                .iter()
                .zip(expected.se.iter())
                .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-5)
        );
    }
}
