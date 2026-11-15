use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use baux::neighbor_index::error::BpannError;
use baux::neighbor_index::index::build::IndexHeader;
use baux::neighbor_index::index::page::{Page, write_index};

pub(crate) fn persist_files(
    index_dir: &Path,
    header: &IndexHeader,
    pages: &[Page],
    skip_edges: &HashMap<u32, Vec<u32>>,
) -> Result<(), BpannError> {
    persist_inner(index_dir, header, pages, skip_edges, false)
}

#[cfg(test)]
pub(crate) fn persist_test(
    index_dir: &Path,
    header: &IndexHeader,
    pages: &[Page],
    skip_edges: &HashMap<u32, Vec<u32>>,
    fail_after_rename: bool,
) -> Result<(), BpannError> {
    persist_inner(index_dir, header, pages, skip_edges, fail_after_rename)
}

fn persist_inner(
    index_dir: &Path,
    header: &IndexHeader,
    pages: &[Page],
    skip_edges: &HashMap<u32, Vec<u32>>,
    fail_after_rename: bool,
) -> Result<(), BpannError> {
    fs::create_dir_all(index_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let pages_path = index_dir.join("pages.bin");
    let pages_tmp = index_dir.join("pages.bin.tmp");
    let skip_path = index_dir.join("skip_edges.bin");
    let skip_tmp = index_dir.join("skip_edges.bin.tmp");
    let pages_backup = read_existing(&pages_path)?;
    let skip_backup = read_existing(&skip_path)?;
    let persist_result = (|| {
        {
            let file = File::create(&pages_tmp)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            let mut writer = BufWriter::new(file);
            write_index(pages, header.num_dim, &mut writer)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            writer
                .flush()
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }
        write_file(&skip_tmp, skip_edges)?;
        fs::rename(&pages_tmp, &pages_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::rename(&skip_tmp, &skip_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        if fail_after_rename {
            return Err(BpannError::InvalidParameter(
                "test-injected persist failure after pages rename".to_string(),
            ));
        }
        let header_json = baux::artifact::json::pretty_string(header)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::write(index_dir.join("header.json"), header_json)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        Ok(())
    })();
    if persist_result.is_err() {
        if let Some(bytes) = pages_backup {
            let _ = fs::write(&pages_path, bytes);
        }
        if let Some(bytes) = skip_backup {
            let _ = fs::write(&skip_path, bytes);
        }
        let _ = fs::remove_file(&pages_tmp);
        let _ = fs::remove_file(&skip_tmp);
    }
    persist_result
}

fn read_existing(path: &Path) -> Result<Option<Vec<u8>>, BpannError> {
    path.exists()
        .then(|| fs::read(path))
        .transpose()
        .map_err(|error| BpannError::InvalidParameter(error.to_string()))
}

pub(crate) fn skip_bytes(skip_edges: &HashMap<u32, Vec<u32>>) -> Vec<u8> {
    let mut entries: Vec<(&u32, &Vec<u32>)> = skip_edges.iter().collect();
    entries.sort_by_key(|&(&from, _)| from);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (&from, tos) in entries {
        bytes.extend_from_slice(&from.to_le_bytes());
        bytes.extend_from_slice(&(tos.len() as u32).to_le_bytes());
        for &to in tos {
            bytes.extend_from_slice(&to.to_le_bytes());
        }
    }
    bytes
}

pub(crate) fn write_file(
    path: &Path,
    skip_edges: &HashMap<u32, Vec<u32>>,
) -> Result<(), BpannError> {
    let bytes = skip_bytes(skip_edges);
    fs::write(path, bytes).map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::neighbor_index::index::LEAF_CAPACITY;
    use baux::neighbor_index::index::build::BpannIndex;
    use tempfile::TempDir;

    #[test]
    fn skip_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skip_edges.bin");
        let mut edges = HashMap::new();
        edges.insert(1u32, vec![2, 3]);
        write_file(&path, &edges).unwrap();
        let data = fs::read(&path).unwrap();
        assert!(!data.is_empty());
    }

    #[test]
    fn persist_rollback() {
        let dir = TempDir::new().unwrap();
        let index_dir = dir.path().join("index");
        let index = BpannIndex::build_persist(
            &[0u32, 1],
            &[vec![0.0f32, 0.0], vec![1.0, 0.0]],
            2,
            index_dir.clone(),
            false,
        )
        .unwrap();
        persist_files(&index_dir, &index.header, &index.pages, &index.skip_edges).unwrap();
        let pages_before = fs::read(index_dir.join("pages.bin")).unwrap();
        assert!(
            persist_test(
                &index_dir,
                &index.header,
                &index.pages,
                &index.skip_edges,
                true,
            )
            .is_err()
        );
        assert_eq!(pages_before, fs::read(index_dir.join("pages.bin")).unwrap());
        BpannIndex::open(index_dir).expect("index must open after rollback");
        let _ = LEAF_CAPACITY;
    }

    #[test]
    fn persist_write() {
        let dir = TempDir::new().unwrap();
        let index_dir = dir.path().join("index");
        let index = BpannIndex::build_persist(
            &[0u32, 1, 2],
            &[vec![0.0f32, 0.0], vec![1.0, 0.0], vec![0.0f32, 1.0]],
            2,
            index_dir.clone(),
            false,
        )
        .unwrap();
        persist_files(&index_dir, &index.header, &index.pages, &index.skip_edges).unwrap();
        let opened = BpannIndex::open(index_dir.clone()).expect("initial open");
        assert_eq!(opened.header.indexed_rows, 3);

        assert!(
            persist_test(
                &index_dir,
                &index.header,
                &index.pages,
                &index.skip_edges,
                true,
            )
            .is_err()
        );

        let reopened = BpannIndex::open(index_dir).expect("index must open after partial write");
        assert_eq!(reopened.header.indexed_rows, 3);
    }
}
