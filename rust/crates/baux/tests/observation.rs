use baux::neighbor_index::mmap_store::MmapColumnStore;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn check_dims() {
    baux::neighbor_index::observation::check_dims(4).unwrap();
    assert!(
        baux::neighbor_index::observation::check_dims(
            baux::neighbor_index::observation::MAX_DIM + 1
        )
        .is_err()
    );
}

#[test]
fn check_rows() {
    const MAX_OBS: usize = u32::MAX as usize - 1;
    baux::neighbor_index::observation::check_rows(10).unwrap();
    assert!(baux::neighbor_index::observation::check_rows(MAX_OBS + 1).is_err());
}

#[test]
fn check_index_format() {
    let dir = TempDir::new().unwrap();
    baux::neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    baux::neighbor_index::observation::check_index_format(
        dir.path(),
        baux::neighbor_index::observation::INDEX_FORMAT,
    )
    .unwrap();
    assert!(
        baux::neighbor_index::observation::check_index_format(dir.path(), "wrong_index").is_err()
    );
}

#[test]
fn load_rows() {
    let dir = TempDir::new().unwrap();
    baux::neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 2).unwrap();
    assert_eq!(
        baux::neighbor_index::observation::load_rows(dir.path()),
        Some(2)
    );
}

#[test]
fn load_index_format() {
    let dir = TempDir::new().unwrap();
    baux::neighbor_index::observation::write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    assert_eq!(
        baux::neighbor_index::observation::load_index_format(dir.path()).as_deref(),
        Some(baux::neighbor_index::observation::INDEX_FORMAT)
    );
}

#[test]
fn write_metadata() {
    let dir = TempDir::new().unwrap();
    baux::neighbor_index::observation::write_metadata(dir.path(), 1, 2, 1, false, 3).unwrap();
    assert_eq!(
        baux::neighbor_index::observation::bpann_obs(dir.path()),
        Some(1)
    );
    assert_eq!(
        baux::neighbor_index::observation::load_rows(dir.path()),
        Some(3)
    );
    assert_eq!(
        baux::neighbor_index::observation::load_index_format(dir.path()).as_deref(),
        Some(baux::neighbor_index::observation::INDEX_FORMAT)
    );
}

#[test]
fn open_yvar() {
    let dir = TempDir::new().unwrap();
    let store = baux::neighbor_index::observation::open_yvar(dir.path(), 1, Some(&array![[0.1]]))
        .unwrap()
        .unwrap();
    assert_eq!(store.nrows(), 1);
    assert_eq!(store.ncols(), 1);
    assert_eq!(store.row_slice(0).unwrap(), &[0.1]);
}

#[test]
fn append_yvar() {
    let dir = TempDir::new().unwrap();
    let mut slot = None;
    baux::neighbor_index::observation::append_yvar(
        dir.path(),
        1,
        &mut slot,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
    let store = slot.unwrap();
    assert_eq!(store.nrows(), 1);
    assert_eq!(store.ncols(), 1);
    assert_eq!(store.row_slice(0).unwrap(), &[0.2]);
}

#[test]
fn train_rows() {
    let dir = TempDir::new().unwrap();
    let mut x = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 2, None).unwrap();
    let mut y = MmapColumnStore::open_mmap(dir.path().join("y.bin"), 1, None).unwrap();
    let mut yvar = MmapColumnStore::open_mmap(dir.path().join("yvar.bin"), 1, None).unwrap();
    x.mmap_append(&array![[1.5, -2.5], [3.0, 4.0]].view())
        .unwrap();
    y.mmap_append(&array![[0.5], [1.25]].view()).unwrap();
    yvar.mmap_append(&array![[0.1], [0.2]].view()).unwrap();
    let (train_x, train_y, train_yvar) =
        baux::neighbor_index::observation::train_rows(2, &x, &y, Some(&yvar), &[0, 1]).unwrap();
    assert_eq!(train_x.shape(), &[2, 2]);
    assert_eq!(train_y.shape(), &[2, 1]);
    assert_eq!(train_x, array![[1.5, -2.5], [3.0, 4.0]]);
    assert_eq!(train_y, array![[0.5], [1.25]]);
    let train_yvar = train_yvar.unwrap();
    assert_eq!(train_yvar.shape(), &[2, 1]);
    assert_eq!(train_yvar, array![[0.1], [0.2]]);
}

#[test]
fn mark_dirty() {
    let dirty = Mutex::new(false);
    baux::neighbor_index::observation::mark_dirty(&dirty);
    assert!(*dirty.lock().unwrap());
}

#[test]
fn num_sidecars() {
    let dir = TempDir::new().unwrap();
    let mut counter = baux::neighbor_index::observation::NumObsCounter::open(dir.path()).unwrap();
    counter.set(42);
    assert_eq!(
        baux::neighbor_index::observation::bpann_obs(dir.path()),
        Some(42)
    );
    baux::neighbor_index::observation::write_obs(dir.path(), 7).unwrap();
    assert_eq!(
        baux::neighbor_index::observation::bpann_obs(dir.path()),
        Some(7)
    );
    baux::neighbor_index::observation::write_rows(dir.path(), 5).unwrap();
    assert_eq!(
        baux::neighbor_index::observation::load_rows(dir.path()),
        Some(5)
    );
}

#[test]
fn parse_string() {
    assert_eq!(
        baux::neighbor_index::observation::parse_string(
            "{\"index_backend\":\"bpann_disk\"}",
            "index_backend"
        )
        .as_deref(),
        Some("bpann_disk")
    );
}
