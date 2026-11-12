use super::types::{EvalOutcome, StepRecord, StoreStats, TrajHead, TrajSlice};
use redb::{ReadableTable, ReadableTableMetadata, TableDefinition};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const EVAL_MEMO: TableDefinition<&str, &[u8]> = TableDefinition::new("eval-memo-v1");
const TRAJ_HEAD: TableDefinition<&str, &[u8]> = TableDefinition::new("traj-head-v1");
const STEP_LOG: TableDefinition<(&str, u32), &[u8]> = TableDefinition::new("step-log-v1");

pub fn default_dir(root: &Path) -> PathBuf {
    root.join(".cache/baux/incremental")
}

pub fn eval_key(env_digest: &str, x_blake3: &str) -> String {
    format!("{env_digest}:{x_blake3}")
}

pub fn traj_key(env_digest: &str, opt_digest: &str, rep: u32) -> String {
    crate::artifact::blake3::hash_hex(format!("traj:v1:{env_digest}:{opt_digest}:{rep}").as_bytes())
}

fn lock(directory: &Path) -> Result<File, String> {
    fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("inc.lock"))
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if start.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => return Err(format!("incremental store lock: {error}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IncStore {
    dir: PathBuf,
}

impl IncStore {
    pub fn open(dir: &Path) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let _lock = lock(dir)?;
        let db = redb::Database::create(dir.join("inc_v1.redb")).map_err(|e| e.to_string())?;
        let tx = db.begin_write().map_err(|e| e.to_string())?;
        {
            let _ = tx.open_table(EVAL_MEMO).map_err(|e| e.to_string())?;
            let _ = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            let _ = tx.open_table(STEP_LOG).map_err(|e| e.to_string())?;
        }
        tx.commit().map_err(|e| e.to_string())?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    fn with_read<T>(
        &self,
        f: impl FnOnce(&redb::ReadTransaction) -> Result<T, String>,
    ) -> Result<T, String> {
        let _lock = lock(&self.dir)?;
        let db = redb::Database::open(self.dir.join("inc_v1.redb")).map_err(|e| e.to_string())?;
        let tx = db.begin_read().map_err(|e| e.to_string())?;
        f(&tx)
    }

    fn with_write<T>(
        &self,
        f: impl FnOnce(&redb::WriteTransaction) -> Result<T, String>,
    ) -> Result<T, String> {
        let _lock = lock(&self.dir)?;
        let db = redb::Database::create(self.dir.join("inc_v1.redb")).map_err(|e| e.to_string())?;
        let tx = db.begin_write().map_err(|e| e.to_string())?;
        let res = f(&tx)?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(res)
    }

    pub fn get_eval(
        &self,
        env_digest: &str,
        x_blake3: &str,
    ) -> Result<Option<EvalOutcome>, String> {
        let mut batch = self.get_evals(env_digest, &[x_blake3.to_string()])?;
        Ok(batch.pop().flatten())
    }

    pub fn get_evals(
        &self,
        env_digest: &str,
        x_hashes: &[String],
    ) -> Result<Vec<Option<EvalOutcome>>, String> {
        self.with_read(|tx| {
            let table = tx.open_table(EVAL_MEMO).map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity(x_hashes.len());
            for h in x_hashes {
                let key = eval_key(env_digest, h);
                let opt = match table.get(key.as_str()).map_err(|e| e.to_string())? {
                    Some(guard) => {
                        let outcome: EvalOutcome = crate::artifact::cbor::from_slice(guard.value())
                            .map_err(|e| e.to_string())?;
                        Some(outcome)
                    }
                    None => None,
                };
                out.push(opt);
            }
            Ok(out)
        })
    }

    pub fn put_evals(
        &self,
        env_digest: &str,
        entries: &[(String, EvalOutcome)],
    ) -> Result<(), String> {
        self.with_write(|tx| {
            let mut table = tx.open_table(EVAL_MEMO).map_err(|e| e.to_string())?;
            for (x_hash, outcome) in entries {
                let key = eval_key(env_digest, x_hash);
                let bytes = crate::artifact::cbor::to_vec(outcome).map_err(|e| e.to_string())?;
                table
                    .insert(key.as_str(), bytes.as_slice())
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        })
    }

    pub fn get_head(&self, traj_digest: &str) -> Result<Option<TrajHead>, String> {
        self.with_read(|tx| {
            let table = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            match table.get(traj_digest).map_err(|e| e.to_string())? {
                Some(guard) => {
                    let head: TrajHead = crate::artifact::cbor::from_slice(guard.value())
                        .map_err(|e| e.to_string())?;
                    Ok(Some(head))
                }
                None => Ok(None),
            }
        })
    }

    pub fn slice_trajectory(
        &self,
        traj_digest: &str,
        horizon: u32,
    ) -> Result<Option<TrajSlice>, String> {
        if horizon == 0 {
            return Ok(None);
        }
        self.with_read(|tx| {
            let head_table = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            let head_guard = match head_table.get(traj_digest).map_err(|e| e.to_string())? {
                Some(g) => g,
                None => return Ok(None),
            };
            let head: TrajHead =
                crate::artifact::cbor::from_slice(head_guard.value()).map_err(|e| e.to_string())?;
            if head.completed_round < horizon {
                return Ok(None);
            }
            let step_table = tx.open_table(STEP_LOG).map_err(|e| e.to_string())?;
            let mut best_scores = Vec::with_capacity(horizon as usize);
            let mut best_viols = Vec::with_capacity(horizon as usize);
            let mut accepts = 0u32;
            for entry in step_table
                .range((traj_digest, 1)..=(traj_digest, horizon))
                .map_err(|e| e.to_string())?
            {
                let (_, val) = entry.map_err(|e| e.to_string())?;
                let step: StepRecord =
                    crate::artifact::cbor::from_slice(val.value()).map_err(|e| e.to_string())?;
                best_scores.push(step.best_score);
                best_viols.push(step.best_viol);
                if step.accepted {
                    accepts += 1;
                }
            }
            let final_score = *best_scores.last().unwrap_or(&head.best_score);
            let initial_error = head.initial_error;
            Ok(Some(TrajSlice {
                head,
                horizon,
                initial_error,
                best_scores,
                best_viols,
                final_score,
                accepts_at_horizon: accepts,
            }))
        })
    }

    pub fn load_prefix(
        &self,
        traj_digest: &str,
    ) -> Result<Option<(TrajHead, Vec<StepRecord>)>, String> {
        self.with_read(|tx| {
            let head_table = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            let head_guard = match head_table.get(traj_digest).map_err(|e| e.to_string())? {
                Some(g) => g,
                None => return Ok(None),
            };
            let head: TrajHead =
                crate::artifact::cbor::from_slice(head_guard.value()).map_err(|e| e.to_string())?;
            let step_table = tx.open_table(STEP_LOG).map_err(|e| e.to_string())?;
            let mut steps = Vec::with_capacity(head.completed_round as usize);
            if head.completed_round > 0 {
                for entry in step_table
                    .range((traj_digest, 1)..=(traj_digest, head.completed_round))
                    .map_err(|e| e.to_string())?
                {
                    let (_, val) = entry.map_err(|e| e.to_string())?;
                    let step: StepRecord = crate::artifact::cbor::from_slice(val.value())
                        .map_err(|e| e.to_string())?;
                    steps.push(step);
                }
            }
            Ok(Some((head, steps)))
        })
    }

    pub fn append_steps(
        &self,
        traj_digest: &str,
        head: &TrajHead,
        start_round: u32,
        steps: &[StepRecord],
    ) -> Result<(), String> {
        self.with_write(|tx| {
            let mut step_table = tx.open_table(STEP_LOG).map_err(|e| e.to_string())?;
            for (idx, step) in steps.iter().enumerate() {
                let round = start_round + idx as u32;
                let step_bytes = crate::artifact::cbor::to_vec(step).map_err(|e| e.to_string())?;
                step_table
                    .insert((traj_digest, round), step_bytes.as_slice())
                    .map_err(|e| e.to_string())?;
            }
            let mut head_table = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            let existing: Option<TrajHead> =
                match head_table.get(traj_digest).map_err(|e| e.to_string())? {
                    Some(g) => Some(
                        crate::artifact::cbor::from_slice(g.value()).map_err(|e| e.to_string())?,
                    ),
                    None => None,
                };
            let should_update = match existing {
                Some(ref old) => head.completed_round >= old.completed_round,
                None => true,
            };
            if should_update {
                let head_bytes = crate::artifact::cbor::to_vec(head).map_err(|e| e.to_string())?;
                head_table
                    .insert(traj_digest, head_bytes.as_slice())
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        })
    }

    pub fn stats(&self) -> Result<StoreStats, String> {
        self.with_read(|tx| {
            let eval_table = tx.open_table(EVAL_MEMO).map_err(|e| e.to_string())?;
            let traj_table = tx.open_table(TRAJ_HEAD).map_err(|e| e.to_string())?;
            let step_table = tx.open_table(STEP_LOG).map_err(|e| e.to_string())?;
            Ok(StoreStats {
                eval_memos: eval_table.len().map_err(|e| e.to_string())?,
                trajectories: traj_table.len().map_err(|e| e.to_string())?,
                step_rows: step_table.len().map_err(|e| e.to_string())?,
            })
        })
    }
}
