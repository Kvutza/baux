use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array2, ArrayView2};

use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::mmap_store::MmapColumnStore;

pub const FORMAT_VERSION: u32 = 1;
pub const MAX_DIM: usize = 2_097_152;
pub const MAX_STRIDE: usize = 32 * 1024 * 1024;
pub const INDEX_BACKEND: &str = "bpann_disk";

const STORE_FILES: &[&str] = &[
    "train_x.bin",
    "train_y.bin",
    "train_yvar.bin",
    "num_obs.bin",
    "indexed_rows.bin",
    "metadata.json",
    "metric.bin",
    "metric.pending",
    "metric.tmp",
];
const STORE_DIRS: &[&str] = &["index"];

/// Delete the persisted BPANN store while preserving unrelated files.
/// No backend may be open on `work_dir` when this is called.
pub fn remove_store(work_dir: &Path) -> Result<(), BpannError> {
    let gone = |result: std::io::Result<()>| match result {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
            Err(BpannError::InvalidParameter(error.to_string()))
        }
        _ => Ok(()),
    };
    for name in STORE_FILES {
        gone(fs::remove_file(work_dir.join(name)))?;
    }
    for name in STORE_DIRS {
        gone(fs::remove_dir_all(work_dir.join(name)))?;
    }
    Ok(())
}

pub type TrainRows = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub fn check_dims(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if num_dim > MAX_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_DIM}"
        )));
    }
    if record_stride > MAX_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_STRIDE}"
        )));
    }
    Ok(())
}

pub fn check_rows(new_n: usize) -> Result<(), BpannError> {
    if new_n >= u32::MAX as usize {
        return Err(BpannError::InvalidParameter(
            "row count exceeds u32::MAX".to_string(),
        ));
    }
    Ok(())
}

pub fn check_backend(work_dir: &Path, expected: &str) -> Result<(), BpannError> {
    if let Some(backend) = load_backend(work_dir) {
        if backend != expected {
            return Err(BpannError::InvalidParameter(format!(
                "work_dir index_backend is {backend}, expected {expected}"
            )));
        }
    }
    Ok(())
}

pub fn bpann_obs(work_dir: &Path) -> Option<usize> {
    let sidecar = work_dir.join("num_obs.bin");
    let from_sidecar = if let Ok(data) = fs::read(&sidecar) {
        if data.len() == 8 {
            Some(u64::from_le_bytes(data.try_into().ok()?) as usize)
        } else {
            None
        }
    } else {
        None
    };
    let from_meta = fs::read_to_string(work_dir.join("metadata.json"))
        .ok()
        .and_then(|text| parse_number(&text, "num_obs"));
    match (from_sidecar, from_meta) {
        (Some(sidecar), Some(meta)) => Some(sidecar.max(meta)),
        (Some(sidecar), None) => Some(sidecar),
        (None, Some(meta)) => Some(meta),
        (None, None) => None,
    }
}

pub fn write_obs(work_dir: &Path, num_obs: usize) -> Result<(), BpannError> {
    fs::write(work_dir.join("num_obs.bin"), (num_obs as u64).to_le_bytes())
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

#[derive(Clone, Debug)]
pub struct NumObsCounter {
    path: PathBuf,
    count: Arc<AtomicUsize>,
}

impl NumObsCounter {
    pub fn open(work_dir: &Path) -> Result<Self, BpannError> {
        let path = work_dir.join("num_obs.bin");
        let initial = if path.exists() {
            if let Ok(data) = fs::read(&path) {
                if data.len() == 8 {
                    u64::from_le_bytes(data.try_into().unwrap_or([0; 8])) as usize
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            fs::write(&path, 0u64.to_le_bytes())
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            0
        };
        Ok(Self {
            path,
            count: Arc::new(AtomicUsize::new(initial)),
        })
    }

    pub fn set(&self, num_obs: usize) {
        self.count.store(num_obs, Ordering::Relaxed);
        let _ = fs::write(&self.path, (num_obs as u64).to_le_bytes());
    }

    pub fn get(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub fn with_count(self, num_obs: usize) -> Self {
        self.set(num_obs);
        self
    }
}

pub fn write_rows(work_dir: &Path, indexed_rows: usize) -> Result<(), BpannError> {
    fs::write(
        work_dir.join("indexed_rows.bin"),
        (indexed_rows as u64).to_le_bytes(),
    )
    .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub fn load_rows(work_dir: &Path) -> Option<usize> {
    let sidecar = work_dir.join("indexed_rows.bin");
    if let Ok(data) = fs::read(&sidecar) {
        if data.len() == 8 {
            return Some(u64::from_le_bytes(data.try_into().ok()?) as usize);
        }
    }
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_number(&text, "indexed_rows")
}

pub fn load_backend(work_dir: &Path) -> Option<String> {
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_string(&text, "index_backend")
}

pub fn write_metadata(
    work_dir: &Path,
    num_obs: usize,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    indexed_rows: usize,
) -> Result<(), BpannError> {
    let json = format!(
        "{{\"format_version\":{FORMAT_VERSION},\"num_obs\":{num_obs},\"num_dim\":{num_dim},\
         \"num_metrics\":{num_metrics},\"scale_x\":{scale_x},\"index_backend\":\"{INDEX_BACKEND}\",\
         \"indexed_rows\":{indexed_rows}}}"
    );
    fs::write(work_dir.join("metadata.json"), json)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub fn open_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: Option<&Array2<f64>>,
) -> Result<Option<MmapColumnStore>, BpannError> {
    if let Some(yv) = train_yvar {
        let yv_path = work_dir.join("train_yvar.bin");
        let known_nrows = bpann_obs(work_dir);
        let mut store = MmapColumnStore::open_mmap(yv_path, num_metrics, known_nrows)?;
        if store.nrows == 0 && yv.nrows() > 0 {
            store = store.append(&yv.view())?;
        }
        Ok(Some(store))
    } else {
        Ok(None)
    }
}

pub fn append_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: &mut Option<MmapColumnStore>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), BpannError> {
    match (train_yvar.take(), yvar) {
        (Some(store), Some(yv)) => {
            *train_yvar = Some(store.append(yv)?);
        }
        (None, Some(yv)) => {
            let yv_path = work_dir.join("train_yvar.bin");
            let known_nrows = bpann_obs(work_dir);
            let store = MmapColumnStore::open_mmap(yv_path, num_metrics, known_nrows)?;
            *train_yvar = Some(store.append(yv)?);
        }
        (other, _) => {
            *train_yvar = other;
        }
    }
    Ok(())
}

pub fn train_rows(
    n: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<TrainRows, BpannError> {
    for &i in indices {
        if i >= n {
            return Err(BpannError::InvalidParameter(format!(
                "train_rows index {i} out of range [0, {n})"
            )));
        }
    }
    let x = train_x.mmap_gather(indices)?;
    let y = train_y.mmap_gather(indices)?;
    let yvar = train_yvar.map(|s| s.mmap_gather(indices)).transpose()?;
    Ok((x, y, yvar))
}

pub fn mark_dirty(index_dirty: &Mutex<bool>) {
    *index_dirty.lock().expect("index_dirty mutex poisoned") = true;
}

pub(crate) fn parse_number(text: &str, field: &str) -> Option<usize> {
    let needle = format!("\"{field}\":");
    text.split_once(&needle).and_then(|(_, tail)| {
        tail.trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|digits| digits.parse().ok())
    })
}

pub fn parse_string(text: &str, field: &str) -> Option<String> {
    let marker = format!("\"{field}\":\"");
    text.split_once(&marker)
        .and_then(|(_, tail)| tail.split_once('"').map(|(value, _)| value.to_string()))
}

#[derive(Clone)]
pub struct ObservationStore {
    pub work_dir: PathBuf,
    pub train_x: MmapColumnStore,
    pub train_y: MmapColumnStore,
    pub train_yvar: Option<MmapColumnStore>,
    pub num_dim: usize,
    pub num_metrics: usize,
}

pub type ObservationLog = ObservationStore;

const _: () = {
    fn assert_sync<T: Send + Sync + Clone>() {}
    fn check() {
        assert_sync::<ObservationStore>();
        assert_sync::<NumObsCounter>();
    }
};

impl ObservationStore {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
    ) -> Result<Self, BpannError> {
        check_dims(train_x.ncols())?;
        fs::create_dir_all(&work_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        check_backend(&work_dir, INDEX_BACKEND)?;

        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let known_nrows = bpann_obs(&work_dir);

        let mut train_x_store =
            MmapColumnStore::open_mmap(work_dir.join("train_x.bin"), num_dim, known_nrows)?;
        let mut train_y_store =
            MmapColumnStore::open_mmap(work_dir.join("train_y.bin"), num_metrics, known_nrows)?;

        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store = train_x_store.append(&train_x.view())?;
            train_y_store = train_y_store.append(&train_y.view())?;
        }

        let train_yvar_store = open_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;

        Ok(Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
        })
    }

    pub fn open(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, BpannError> {
        check_dims(num_dim)?;
        fs::create_dir_all(&work_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let known_nrows = bpann_obs(&work_dir);
        let train_x_store =
            MmapColumnStore::open_mmap(work_dir.join("train_x.bin"), num_dim, known_nrows)?;
        let train_y_store =
            MmapColumnStore::open_mmap(work_dir.join("train_y.bin"), num_metrics, known_nrows)?;
        let yvar_path = work_dir.join("train_yvar.bin");
        let train_yvar_store = if yvar_path.exists() {
            Some(MmapColumnStore::open_mmap(
                yvar_path,
                num_metrics,
                known_nrows,
            )?)
        } else {
            None
        };
        Ok(Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
        })
    }

    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn row_x(&self, i: usize) -> Result<&[f64], BpannError> {
        self.train_x.row_slice(i)
    }

    pub fn row_y(&self, i: usize) -> Result<&[f64], BpannError> {
        self.train_y.row_slice(i)
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<&[f64]>, BpannError> {
        self.train_yvar.as_ref().map(|s| s.row_slice(i)).transpose()
    }

    pub fn train_rows(&self, indices: &[usize]) -> Result<TrainRows, BpannError> {
        train_rows(
            self.len(),
            &self.train_x,
            &self.train_y,
            self.train_yvar.as_ref(),
            indices,
        )
    }

    pub fn append(
        mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<Self, BpannError> {
        if x.nrows() == 0 {
            return Ok(self);
        }
        if x.ncols() != self.num_dim || y.ncols() != self.num_metrics || x.nrows() != y.nrows() {
            return Err(BpannError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        check_rows(self.len() + x.nrows())?;
        self.train_x = self.train_x.append(x)?;
        self.train_y = self.train_y.append(y)?;
        append_yvar(&self.work_dir, self.num_metrics, &mut self.train_yvar, yvar)?;
        Ok(self)
    }

    pub fn persist(mut self) -> Result<Self, BpannError> {
        self.train_x = self.train_x.persist()?;
        self.train_y = self.train_y.persist()?;
        if let Some(store) = self.train_yvar {
            self.train_yvar = Some(store.persist()?);
        }
        write_obs(&self.work_dir, self.len())?;
        Ok(self)
    }

    pub fn release_pages(mut self) -> Result<Self, BpannError> {
        self.train_x = self.train_x.release_pages()?;
        self.train_y = self.train_y.release_pages()?;
        if let Some(store) = self.train_yvar {
            self.train_yvar = Some(store.release_pages()?);
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::neighbor_index::mmap_store::MmapColumnStore;
    use ndarray::array;
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[test]
    fn bpann_above() {
        check_dims(MAX_DIM).unwrap();
        let err = check_dims(MAX_DIM + 1).unwrap_err();
        assert!(err.to_string().contains(&MAX_DIM.to_string()));
    }

    #[test]
    fn observation_called2() {
        let dir = TempDir::new().unwrap();
        check_dims(4).unwrap();
        assert!(check_dims(MAX_DIM + 1).is_err());
        check_rows(10).unwrap();
        write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
        write_obs(dir.path(), 0).unwrap();
        write_rows(dir.path(), 0).unwrap();
        let counter = NumObsCounter::open(dir.path()).unwrap();
        counter.set(0);
        assert_eq!(bpann_obs(dir.path()), Some(0));
        assert_eq!(load_rows(dir.path()), Some(0));
        assert_eq!(load_backend(dir.path()).as_deref(), Some(INDEX_BACKEND));
        check_backend(dir.path(), INDEX_BACKEND).unwrap();

        let mut yvar = open_yvar(dir.path(), 1, Some(&array![[0.1]])).unwrap();
        append_yvar(dir.path(), 1, &mut yvar, Some(&array![[0.2]].view())).unwrap();

        let x = MmapColumnStore::open_mmap(dir.path().join("x.bin"), 2, None).unwrap();
        let y = MmapColumnStore::open_mmap(dir.path().join("y.bin"), 1, None).unwrap();
        let x = x.append(&array![[0.0, 0.0]].view()).unwrap();
        let y = y.append(&array![[0.0]].view()).unwrap();
        train_rows(1, &x, &y, None, &[0]).unwrap();
        let dirty = Mutex::new(false);
        mark_dirty(&dirty);
        assert_eq!(
            parse_string(r#"{"index_backend":"bpann_disk"}"#, "index_backend").as_deref(),
            Some("bpann_disk")
        );
        assert_eq!(parse_number(r#"{"num_obs":42}"#, "num_obs"), Some(42));
    }

    #[test]
    fn fork_isolation() {
        let dir = TempDir::new().unwrap();
        let store = ObservationStore::new(
            dir.path().to_path_buf(),
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[10.0], [20.0]],
            Some(array![[0.1], [0.2]]),
        )
        .unwrap()
        .persist()
        .unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.row_x(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.row_y(0).unwrap(), &[10.0]);
        assert_eq!(store.row_yvar(0).unwrap().unwrap(), &[0.1]);

        // Fork two branches
        let branch_a = store.clone();
        let branch_b = store.clone();

        // Mutate branch A
        let branch_a = branch_a
            .append(
                &array![[5.0, 6.0]].view(),
                &array![[30.0]].view(),
                Some(&array![[0.3]].view()),
            )
            .unwrap();

        // Mutate branch B with different data
        let branch_b = branch_b
            .append(
                &array![[7.0, 8.0], [9.0, 10.0]].view(),
                &array![[40.0], [50.0]].view(),
                Some(&array![[0.4], [0.5]].view()),
            )
            .unwrap();

        // Check fork isolation in-memory
        assert_eq!(store.len(), 2);
        assert_eq!(branch_a.len(), 3);
        assert_eq!(branch_b.len(), 4);

        assert_eq!(branch_a.row_x(2).unwrap(), &[5.0, 6.0]);
        assert_eq!(branch_a.row_y(2).unwrap(), &[30.0]);
        assert_eq!(branch_a.row_yvar(2).unwrap().unwrap(), &[0.3]);

        assert_eq!(branch_b.row_x(2).unwrap(), &[7.0, 8.0]);
        assert_eq!(branch_b.row_y(2).unwrap(), &[40.0]);
        assert_eq!(branch_b.row_yvar(2).unwrap().unwrap(), &[0.4]);

        assert_eq!(branch_b.row_x(3).unwrap(), &[9.0, 10.0]);
        assert_eq!(branch_b.row_y(3).unwrap(), &[50.0]);
        assert_eq!(branch_b.row_yvar(3).unwrap().unwrap(), &[0.5]);

        // Train rows check
        let (xa, ya, yvara) = branch_a.train_rows(&[2]).unwrap();
        assert_eq!(xa.shape(), &[1, 2]);
        assert_eq!(xa, array![[5.0, 6.0]]);
        assert_eq!(ya, array![[30.0]]);
        assert_eq!(yvara.unwrap(), array![[0.3]]);

        let (xb, yb, yvarb) = branch_b.train_rows(&[2, 3]).unwrap();
        assert_eq!(xb.shape(), &[2, 2]);
        assert_eq!(xb, array![[7.0, 8.0], [9.0, 10.0]]);
        assert_eq!(yb, array![[40.0], [50.0]]);
        assert_eq!(yvarb.unwrap(), array![[0.4], [0.5]]);

        // Persist branch A
        let branch_a = branch_a.persist().unwrap();
        assert_eq!(branch_a.row_x(2).unwrap(), &[5.0, 6.0]);
        assert_eq!(branch_b.row_x(2).unwrap(), &[7.0, 8.0]);

        // Persist branch B
        let branch_b = branch_b.persist().unwrap();
        assert_eq!(branch_b.row_x(2).unwrap(), &[7.0, 8.0]);
        assert_eq!(branch_b.row_x(3).unwrap(), &[9.0, 10.0]);
        assert_eq!(branch_a.row_x(2).unwrap(), &[5.0, 6.0]);
    }
}
