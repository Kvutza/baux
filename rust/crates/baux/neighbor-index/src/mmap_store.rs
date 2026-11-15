use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use ndarray::{Array2, ArrayView2, Axis};

use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::residency::{Residency, ResidencyPolicy};

pub const GROW_ROWS: usize = 64;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn temp_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let cnt = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("store");
    let tmp_name = format!("{file_name}.flush.tmp.{pid}.{cnt}");
    if let Some(parent) = path.parent() {
        parent.join(tmp_name)
    } else {
        PathBuf::from(tmp_name)
    }
}

struct TempFileGuard(Option<PathBuf>);

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.0 {
            let _ = fs::remove_file(path);
        }
    }
}

fn slice_bytes(slice: &[f64]) -> &[u8] {
    // SAFETY: f64 has no uninitialized bytes, and casting &[f64] to &[u8] with
    // length = slice.len() * 8 is safe and valid for the lifetime of slice.
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<f64>(),
        )
    }
}

struct MmapBase {
    file: File,
    mmap: Mmap,
    nrows: usize,
}

#[derive(Clone)]
pub struct MmapColumnStore {
    pub path: PathBuf,
    pub ncols: usize,
    pub nrows: usize,
    base: Option<Arc<MmapBase>>,
    delta: Arc<Vec<f64>>,
    residency: Arc<Residency>,
    lineage: Arc<()>,
}

pub type MmapStore = MmapColumnStore;

const _: () = {
    fn assert_sync<T: Send + Sync + Clone>() {}
    fn check() {
        assert_sync::<MmapColumnStore>();
    }
};

impl MmapColumnStore {
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn len(&self) -> usize {
        self.nrows
    }

    pub fn is_empty(&self) -> bool {
        self.nrows == 0
    }

    pub fn row_bytes(&self) -> usize {
        self.ncols * std::mem::size_of::<f64>()
    }

    pub fn capacity_bytes(&self, nrows: usize) -> usize {
        nrows.saturating_mul(self.row_bytes())
    }

    pub fn base_rows(&self) -> usize {
        self.base.as_ref().map(|b| b.nrows).unwrap_or(0)
    }

    pub fn ensure_capacity(mut self, need_rows: usize) -> Result<Self, BpannError> {
        if need_rows > self.nrows {
            let delta_needed = need_rows.saturating_sub(self.base_rows());
            let delta_elements = delta_needed.saturating_mul(self.ncols);
            let delta = Arc::make_mut(&mut self.delta);
            if delta_elements > delta.len() {
                delta.reserve(delta_elements - delta.len());
            }
        }
        Ok(self)
    }

    pub fn open_mmap(
        path: PathBuf,
        ncols: usize,
        known_nrows: Option<usize>,
    ) -> Result<Self, BpannError> {
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            }
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            drop(file);
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let len = file
            .metadata()
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?
            .len();
        let row_bytes = ncols * std::mem::size_of::<f64>();
        let nrows = known_nrows.unwrap_or_else(|| {
            if row_bytes > 0 {
                (len as usize) / row_bytes
            } else {
                0
            }
        });
        if known_nrows.is_some() && nrows * row_bytes > len as usize {
            return Err(BpannError::InvalidParameter(format!(
                "known_nrows {nrows} exceeds train file bytes {len}"
            )));
        }

        let (base, residency) = if nrows > 0 && len > 0 {
            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
            };
            let residency = Arc::new(Residency::new(mmap.len(), None));
            (Some(Arc::new(MmapBase { file, mmap, nrows })), residency)
        } else {
            (None, Arc::new(Residency::new(0, None)))
        };

        Ok(Self {
            path,
            ncols,
            nrows,
            base,
            delta: Arc::new(Vec::new()),
            residency,
            lineage: Arc::new(()),
        })
    }

    pub fn residency(&self) -> &Residency {
        &self.residency
    }

    pub fn set_budget(&self, bytes: Option<usize>) {
        self.residency.set_budget(bytes);
    }

    pub fn with_budget(self, bytes: Option<usize>) -> Self {
        self.set_budget(bytes);
        self
    }

    /// Pretouch every page of the current mapping so reads pay no
    /// page-fault or block-allocation cost.
    pub fn pretouch(self) -> Self {
        if let Some(ref base) = self.base {
            const PAGE: usize = 4096;
            let len = base.mmap.len();
            let mut i = 0;
            let mut sum = 0u8;
            while i < len {
                sum = sum.wrapping_add(base.mmap[i]);
                i += PAGE;
            }
            std::hint::black_box(sum);
        }
        self
    }

    pub fn append(mut self, rows: &ArrayView2<f64>) -> Result<Self, BpannError> {
        if rows.nrows() == 0 {
            return Ok(self);
        }
        if rows.ncols() != self.ncols {
            return Err(BpannError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![rows.nrows(), rows.ncols()],
            });
        }
        let new_nrows = self.nrows + rows.nrows();
        let expected_prefix_bytes = self.base_rows() * self.row_bytes();
        let file_exists = self.path.exists();
        let file_len = fs::metadata(&self.path)
            .map(|m| m.len() as usize)
            .unwrap_or(0);

        if self.delta.is_empty() && file_exists && file_len == expected_prefix_bytes {
            let offset = expected_prefix_bytes as u64;
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            }
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&self.path)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;

            let flat_vec: Vec<f64>;
            let slice: &[f64] = if let Some(src) = rows.as_slice() {
                src
            } else {
                let mut flat = Vec::with_capacity(rows.nrows() * self.ncols);
                for row in rows.axis_iter(Axis(0)) {
                    flat.extend(row.iter().copied());
                }
                flat_vec = flat;
                &flat_vec
            };
            let bytes = slice_bytes(slice);

            file.seek(SeekFrom::Start(offset))
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            file.write_all(bytes)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            file.flush()
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;

            drop(self.base.take());

            let mmap = unsafe {
                Mmap::map(&file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
            };
            let budget = self.residency.budget();
            self.residency = Arc::new(Residency::new(mmap.len(), budget));
            self.base = Some(Arc::new(MmapBase {
                file,
                mmap,
                nrows: new_nrows,
            }));
            self.nrows = new_nrows;
            Ok(self)
        } else {
            let n = rows.nrows() * self.ncols;
            let delta = Arc::make_mut(&mut self.delta);
            delta.reserve(n);
            if let Some(src) = rows.as_slice() {
                delta.extend_from_slice(src);
            } else {
                for row in rows.axis_iter(Axis(0)) {
                    delta.extend(row.iter().copied());
                }
            }
            self.nrows = new_nrows;
            Ok(self)
        }
    }

    pub fn mmap_append(self, rows: &ArrayView2<f64>) -> Result<Self, BpannError> {
        self.append(rows)
    }

    /// Drop faulted/dirty pages from process RSS while keeping the file
    /// mapping.
    pub fn release_pages(mut self) -> Result<Self, BpannError> {
        if let Some(base) = self.base.take() {
            let is_unique = Arc::strong_count(&base) == 1;
            if is_unique {
                let mmap = unsafe {
                    Mmap::map(&base.file)
                        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?
                };
                let budget = self.residency.budget();
                self.residency = Arc::new(Residency::new(mmap.len(), budget));
                let file = base
                    .file
                    .try_clone()
                    .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
                self.base = Some(Arc::new(MmapBase {
                    file,
                    mmap,
                    nrows: base.nrows,
                }));
            } else {
                self.residency.forget(&base.mmap);
                self.base = Some(base);
            }
        }
        Ok(self)
    }

    pub fn row_slice(&self, i: usize) -> Result<&[f64], BpannError> {
        if i >= self.nrows {
            return Err(BpannError::InvalidParameter(format!(
                "row {i} out of range [0, {})",
                self.nrows
            )));
        }
        let base_rows = self.base_rows();
        if i < base_rows {
            let base = self
                .base
                .as_ref()
                .expect("row index within base_rows but base is None");
            let byte_start = i * self.row_bytes();
            let byte_end = byte_start + self.row_bytes();
            assert!(byte_end <= base.mmap.len());
            self.residency
                .touch(&base.mmap, byte_start, self.row_bytes());
            let bytes = &base.mmap[byte_start..byte_end];
            assert!(bytes.as_ptr().align_offset(std::mem::align_of::<f64>()) == 0);
            let slice: &[f64] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.ncols) };
            Ok(slice)
        } else {
            let delta_i = i - base_rows;
            let start = delta_i * self.ncols;
            let end = start + self.ncols;
            Ok(&self.delta[start..end])
        }
    }

    pub fn mmap_gather(&self, indices: &[usize]) -> Result<Array2<f64>, BpannError> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
            let row = self.row_slice(old_i)?;
            for j in 0..self.ncols {
                out[[new_i, j]] = row[j];
            }
        }
        Ok(out)
    }

    pub fn persist(self) -> Result<Self, BpannError> {
        self.flush_disk()
    }

    pub fn flush_disk(mut self) -> Result<Self, BpannError> {
        if self.delta.is_empty() {
            return Ok(self);
        }
        let base_rows = self.base_rows();
        let total_rows = self.nrows;
        let row_bytes = self.row_bytes();

        let tmp_path = temp_path(&self.path);
        if let Some(parent) = tmp_path.parent() {
            fs::create_dir_all(parent).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }
        let mut guard = TempFileGuard(Some(tmp_path.clone()));

        {
            let mut tmp_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp_path)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;

            if let Some(ref base) = self.base {
                if base_rows > 0 {
                    let base_bytes = base_rows * row_bytes;
                    assert!(base_bytes <= base.mmap.len());
                    let bytes = &base.mmap[..base_bytes];
                    tmp_file
                        .write_all(bytes)
                        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
                }
            }

            let delta_bytes = slice_bytes(&self.delta);
            tmp_file
                .write_all(delta_bytes)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;

            tmp_file
                .flush()
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            tmp_file
                .sync_all()
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }

        fs::rename(&tmp_path, &self.path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        guard.0 = None;

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;

        let mmap =
            unsafe { Mmap::map(&file).map_err(|e| BpannError::InvalidParameter(e.to_string()))? };

        let budget = self.residency.budget();
        self.residency = Arc::new(Residency::new(mmap.len(), budget));
        self.base = Some(Arc::new(MmapBase {
            file,
            mmap,
            nrows: total_rows,
        }));
        self.delta = Arc::new(Vec::new());
        self.lineage = Arc::new(());
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn row_bytes() {
        let dir = TempDir::new().unwrap();
        let store = MmapColumnStore::open_mmap(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.row_bytes(), 16);
    }

    #[test]
    fn capacity_bytes() {
        let dir = TempDir::new().unwrap();
        let store = MmapColumnStore::open_mmap(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.capacity_bytes(5), 80);
    }

    #[test]
    fn ensure_capacity() {
        let dir = TempDir::new().unwrap();
        let store = MmapColumnStore::open_mmap(dir.path().join("c.bin"), 2, None).unwrap();
        let store = store.ensure_capacity(8).unwrap();
        let store = store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.nrows(), 2);
        assert_eq!(store.ncols(), 2);
    }

    #[test]
    fn release_rows() {
        let dir = TempDir::new().unwrap();
        let store = MmapColumnStore::open_mmap(dir.path().join("c.bin"), 2, None).unwrap();
        let store = store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].view())
            .unwrap();
        let store = store.release_pages().unwrap();
        assert_eq!(store.row_slice(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.row_slice(2).unwrap(), &[5.0, 6.0]);
        let store = store.mmap_append(&array![[7.0, 8.0]].view()).unwrap();
        let store = store.release_pages().unwrap();
        assert_eq!(store.row_slice(3).unwrap(), &[7.0, 8.0]);
    }

    #[test]
    fn fork_isolation() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c.bin");
        let store = MmapColumnStore::open_mmap(path, 2, None).unwrap();
        let store = store
            .append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap()
            .persist()
            .unwrap();
        assert_eq!(store.nrows(), 2);

        // Fork two branches
        let branch_a = store.clone();
        let branch_b = store.clone();

        // Mutate branch A
        let branch_a = branch_a.append(&array![[5.0, 6.0]].view()).unwrap();

        // Mutate branch B
        let branch_b = branch_b
            .append(&array![[7.0, 8.0], [9.0, 10.0]].view())
            .unwrap();

        // Verify isolation before persist
        assert_eq!(store.nrows(), 2);
        assert_eq!(branch_a.nrows(), 3);
        assert_eq!(branch_b.nrows(), 4);
        assert_eq!(branch_a.row_slice(2).unwrap(), &[5.0, 6.0]);
        assert_eq!(branch_b.row_slice(2).unwrap(), &[7.0, 8.0]);
        assert_eq!(branch_b.row_slice(3).unwrap(), &[9.0, 10.0]);

        // Persist branch A
        let branch_a = branch_a.persist().unwrap();
        assert_eq!(branch_a.row_slice(2).unwrap(), &[5.0, 6.0]);
        assert_eq!(branch_b.row_slice(2).unwrap(), &[7.0, 8.0]);
        assert_eq!(branch_b.nrows(), 4);

        // Persist branch B
        let branch_b = branch_b.persist().unwrap();
        assert_eq!(branch_b.row_slice(2).unwrap(), &[7.0, 8.0]);
        assert_eq!(branch_b.row_slice(3).unwrap(), &[9.0, 10.0]);
        assert_eq!(branch_a.row_slice(2).unwrap(), &[5.0, 6.0]);
    }
}
