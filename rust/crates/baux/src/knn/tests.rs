use super::*;
use ndarray::array;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_unavailable(error: &str) -> bool {
    error.contains("no default Metal device found")
}

#[test]
fn exact_backend() {
    let train = array![[0.0, 0.0], [1.0, 1.0]];
    let backend = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
    assert_eq!(backend.len(), 2);
    let backend = backend.add(&array![[2.0, 2.0]].view(), 2).unwrap();
    assert_eq!(backend.len(), 3);
    let (_d, i) = backend.search(&array![[0.0, 0.0]].view(), 2, 2).unwrap();
    assert_eq!(i[[0, 0]], 0);
    let _backend = backend.rebuild(&train.view()).unwrap();
}

#[test]
fn knn_auto() {
    let train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let auto = match AutoIndex::new(2, &train.view()) {
            Ok(auto) => auto,
            Err(error) if metal_unavailable(&error.to_string()) => return,
            Err(error) => panic!("{error}"),
        };
        let (_distances, indices) = auto.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
        assert_eq!(indices.row(0).to_vec(), vec![0, 1]);
    }
    let backend = KnnBackend::new(2, IndexDriver::Auto, &train.view()).unwrap();
    assert!(backend.memory_bytes() > 0);
    assert_eq!(flatten_f32(&train.view()).len(), 6);
    let (_distances, indices) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
    assert_eq!(indices.row(0).to_vec(), vec![0, 1]);
    let backend = backend.add(&array![[0.05, 0.05]].view(), 3).unwrap();
    let (_distances, indices) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
    assert_eq!(indices[[0, 0]], 3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn cpu_search() {
    let train = array![[0.0], [1.0], [2.0]];
    let auto = match AutoIndex::new(1, &train.view()) {
        Ok(auto) => auto,
        Err(error) if metal_unavailable(&error.to_string()) => return,
        Err(error) => panic!("{error}"),
    };
    // A mismatched accelerator mirror forces a real dimension-check error.
    *auto.gpu.write().unwrap() = Some(match MetalIndex::new(2, &array![[0.0, 0.0]].view()) {
        Ok(index) => index,
        Err(error) if metal_unavailable(&error.to_string()) => return,
        Err(error) => panic!("{error}"),
    });
    let query = array![[0.1]];
    let expected = auto.cpu.search(&query.view(), 2, 2).unwrap();
    let actual = auto.search(&query.view(), 2, 2).unwrap();
    assert_eq!(actual, expected);
    assert!(auto.gpu.read().unwrap().is_none());
    assert_eq!(auto.search(&query.view(), 2, 2).unwrap(), expected);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn cached_search() {
    let train = array![[0.0], [1.0], [2.0]];
    let auto = match AutoIndex::new(1, &train.view()) {
        Ok(auto) => auto,
        Err(error) if metal_unavailable(&error.to_string()) => return,
        Err(error) => panic!("{error}"),
    };
    *auto.gpu.write().unwrap() = Some(match MetalIndex::new(2, &array![[0.0, 0.0]].view()) {
        Ok(index) => index,
        Err(error) if metal_unavailable(&error.to_string()) => return,
        Err(error) => panic!("{error}"),
    });
    auto.decisions
        .write()
        .unwrap()
        .insert((auto.bucket(1), 2, 2), true);
    let query = array![[0.1]];
    assert_eq!(
        auto.search(&query.view(), 2, 2).unwrap(),
        auto.cpu.search(&query.view(), 2, 2).unwrap()
    );
    assert!(auto.gpu.read().unwrap().is_none());
    assert!(auto.decisions.read().unwrap().is_empty());
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn knn_metal() {
    let train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.5, 0.5]];
    let query = array![[0.1, 0.1], [1.8, 1.8]];
    let exact = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
    let metal = match KnnBackend::new(2, IndexDriver::Metal, &train.view()) {
        Ok(index) => index,
        Err(error) if metal_unavailable(&error.to_string()) => return,
        Err(error) => panic!("{error}"),
    };
    let expected = exact.search(&query.view(), 3, 3).unwrap();
    let actual = metal.search(&query.view(), 3, 3).unwrap();
    assert_eq!(actual.1, expected.1);
    for (left, right) in actual.0.iter().zip(expected.0.iter()) {
        assert!((left - right).abs() <= 1e-5);
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn knn_tails() {
    for dimensions in [3, 5, 32, 33] {
        let train = Array2::from_shape_fn((67, dimensions), |(row, col)| {
            ((row * 17 + col * 11) % 101) as f64 / 101.0
        });
        let query = Array2::from_shape_fn((7, dimensions), |(row, col)| {
            ((row * 13 + col * 19 + 3) % 103) as f64 / 103.0
        });
        let exact = KnnBackend::new(dimensions, IndexDriver::Exact, &train.view()).unwrap();
        let metal = match KnnBackend::new(dimensions, IndexDriver::Metal, &train.view()) {
            Ok(index) => index,
            Err(error) if metal_unavailable(&error.to_string()) => return,
            Err(error) => panic!("{error}"),
        };
        let expected = exact.search(&query.view(), 8, 8).unwrap();
        let actual = metal.search(&query.view(), 8, 8).unwrap();
        assert_eq!(actual.1, expected.1);
        for (left, right) in actual.0.iter().zip(expected.0.iter()) {
            assert!((left - right).abs() <= 1e-4);
        }
    }
}

#[test]
fn knn_errors() {
    let train = array![[0.0, 0.0], [1.0, 0.0]];
    match KnnBackend::new(2, IndexDriver::BpAnnDisk, &train.view()) {
        Err(e) => assert!(e.to_string().contains("disk-only")),
        Ok(_) => panic!("expected BpAnnDisk on KnnBackend to error"),
    }
}

#[cfg(any(feature = "usearch", feature = "usearch-native"))]
#[test]
fn knn_usearch() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let backend = KnnBackend::new(2, IndexDriver::USearch, &train.view()).unwrap();
    assert_eq!(backend.len(), 3);
    assert!(matches!(backend, KnnBackend::USearch(_)));
}

#[cfg(not(any(feature = "usearch", feature = "usearch-native")))]
#[test]
fn knn_feature() {
    let train = array![[0.0, 0.0]];
    let error = match KnnBackend::new(2, IndexDriver::USearch, &train.view()) {
        Ok(_) => panic!("expected USearch feature error"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("usearch or usearch-native feature")
    );
}

#[test]
fn pad_unpack() {
    let (d, i) = pad_k(array![[1.0]], array![[0i64]], 3);
    assert_eq!(d.ncols(), 3);
    assert_eq!(i.ncols(), 3);
    let (d2, i2) = unpack_search(1, 2, &[0.5, 1.5], &[0, 1]);
    assert_eq!(d2[[0, 1]], 1.5);
    assert_eq!(i2[[0, 1]], 1);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn check_backend(device: KnnBackend) {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
    let query = array![[0.2, 0.1], [1.1, 0.2]];
    let exact = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
    let expected = exact.search(&query.view(), 3, 3).unwrap();
    let actual = device.search(&query.view(), 3, 3).unwrap();
    assert_eq!(actual.1, expected.1);
    for (actual, expected) in actual.0.iter().zip(expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }

    let large_train = Array2::from_shape_fn((1050, 2), |(row, col)| {
        if col == 0 {
            row as f64 * 0.01
        } else {
            (row % 17) as f64 * 0.1
        }
    });
    let large_query = array![[2.345, 0.7], [7.891, 1.1]];
    let large_exact = KnnBackend::new(2, IndexDriver::Exact, &large_train.view()).unwrap();
    let large_expected = large_exact.search(&large_query.view(), 10, 10).unwrap();
    let device = device.rebuild(&large_train.view()).unwrap();
    let large_actual = device.search(&large_query.view(), 10, 10).unwrap();
    assert_eq!(large_actual.1, large_expected.1);
    for (actual, expected) in large_actual.0.iter().zip(large_expected.0.iter()) {
        assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
    }

    let device = device.add(&array![[0.2, 0.2]].view(), 4).unwrap();
    assert_eq!(device.len(), 1051);
    let device = device.rebuild(&train.view()).unwrap();
    assert_eq!(device.len(), 4);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_index() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
    let device = match KnnBackend::new(2, IndexDriver::Metal, &train.view()) {
        Ok(index) => index,
        Err(error) => {
            eprintln!("Metal runtime unavailable: {error}");
            return;
        }
    };
    check_backend(device);
}

#[test]
fn cuvs_driver() {
    assert!(IndexDriver::Cuda.is_cuda());
    assert!(IndexDriver::Cuvs.is_cuda());
    assert!(IndexDriver::CuvsCagra.is_cuda());
    assert!(!IndexDriver::Exact.is_cuda());
    assert!(!IndexDriver::Metal.is_cuda());

    let cuvs_json = deser_json::to_string(&IndexDriver::Cuvs).unwrap();
    assert_eq!(cuvs_json, "\"cuvs\"");
    let decoded_cuvs: IndexDriver = deser_json::from_str(&cuvs_json).unwrap();
    assert_eq!(decoded_cuvs, IndexDriver::Cuvs);

    let cagra_json = deser_json::to_string(&IndexDriver::CuvsCagra).unwrap();
    assert_eq!(cagra_json, "\"cuvs_cagra\"");
    let decoded_cagra: IndexDriver = deser_json::from_str(&cagra_json).unwrap();
    assert_eq!(decoded_cagra, IndexDriver::CuvsCagra);

    let wgpu_json = deser_json::to_string(&IndexDriver::WebGpu).unwrap();
    assert_eq!(wgpu_json, "\"webgpu\"");
    let decoded_wgpu: IndexDriver = deser_json::from_str(&wgpu_json).unwrap();
    assert_eq!(decoded_wgpu, IndexDriver::WebGpu);
    assert_eq!(IndexDriver::WebGpu.device(), crate::Device::WebGpu);
}

#[test]
fn cow_isolation() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let base = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
    assert_eq!(base.len(), 3);

    // Branch 1 and Branch 2 via Clone (O(1) Arc COW)
    let branch1 = base.clone();
    let branch2 = base.clone();

    // Adding to branch1 must not mutate base or branch2
    let branch1 = branch1.add(&array![[2.0, 2.0]].view(), 3).unwrap();
    assert_eq!(branch1.len(), 4);
    assert_eq!(base.len(), 3);
    assert_eq!(branch2.len(), 3);

    // Adding to branch2 must not mutate base or branch1
    let branch2 = branch2
        .add(&array![[3.0, 3.0], [4.0, 4.0]].view(), 3)
        .unwrap();
    assert_eq!(branch2.len(), 5);
    assert_eq!(base.len(), 3);
    assert_eq!(branch1.len(), 4);

    // Search query on branch1 finds the new point
    let query = array![[2.0, 2.0]];
    let (_, i1) = branch1.search(&query.view(), 1, 1).unwrap();
    assert_eq!(i1[[0, 0]], 3);

    // Search query on base does NOT find key 3 (it only has keys 0, 1, 2)
    let (_, i_base) = base.search(&query.view(), 1, 1).unwrap();
    assert_ne!(i_base[[0, 0]], 3);
}
