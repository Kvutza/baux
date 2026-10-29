use super::*;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Query {
    pub since: Option<i64>,
    pub until: Option<i64>,
    pub task: Option<String>,
    pub optimizer: Option<String>,
    pub protocol: Option<String>,
    pub limit: usize,
}

// A process releases this OS lock even after a crash. Hold it only around a
// catalogue transaction, never for the duration of an experiment.
fn lock(directory: &Path) -> Result<File, String> {
    fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(directory.join("catalog.lock"))
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) if start.elapsed() < Duration::from_secs(5) => {
                std::thread::sleep(Duration::from_millis(20))
            }
            Err(error) => return Err(format!("run catalogue lock: {error}")),
        }
    }
}

const RUNS: redb::TableDefinition<&str, &[u8]> = redb::TableDefinition::new("runs-v1");
const TIMES: redb::MultimapTableDefinition<i64, &str> =
    redb::MultimapTableDefinition::new("started-v1");
const TASKS: redb::MultimapTableDefinition<(&str, i64), &str> =
    redb::MultimapTableDefinition::new("tasks-v1");
const PROTOCOLS: redb::MultimapTableDefinition<(&str, i64), &str> =
    redb::MultimapTableDefinition::new("protocols-v1");
use redb::ReadableTable;

fn insert(db: &redb::Database, record: &Record) -> Result<(), String> {
    let tx = db.begin_write().map_err(|e| e.to_string())?;
    {
        let mut runs = tx.open_table(RUNS).map_err(|e| e.to_string())?;
        let mut times = tx.open_multimap_table(TIMES).map_err(|e| e.to_string())?;
        let mut tasks = tx.open_multimap_table(TASKS).map_err(|e| e.to_string())?;
        let mut protocols = tx
            .open_multimap_table(PROTOCOLS)
            .map_err(|e| e.to_string())?;
        let old = runs
            .get(record.id.as_str())
            .map_err(|e| e.to_string())?
            .map(|bytes| codec::decode(bytes.value()))
            .transpose()
            .map_err(|e| e.to_string())?;
        if let Some(old) = old {
            times
                .remove(old.started, old.id.as_str())
                .map_err(|e| e.to_string())?;
            tasks
                .remove((old.task.as_str(), old.started), old.id.as_str())
                .map_err(|e| e.to_string())?;
            protocols
                .remove((old.protocol.as_str(), old.started), old.id.as_str())
                .map_err(|e| e.to_string())?;
        }
        let bytes = codec::encode(record)?;
        runs.insert(record.id.as_str(), bytes.as_slice())
            .map_err(|e| e.to_string())?;
        times
            .insert(record.started, record.id.as_str())
            .map_err(|e| e.to_string())?;
        tasks
            .insert((record.task.as_str(), record.started), record.id.as_str())
            .map_err(|e| e.to_string())?;
        protocols
            .insert(
                (record.protocol.as_str(), record.started),
                record.id.as_str(),
            )
            .map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())
}

pub(super) fn upsert(directory: &Path, record: &Record) -> Result<(), String> {
    let _lock = lock(directory)?;
    let db = redb::Database::create(directory.join("catalog.db")).map_err(|e| e.to_string())?;
    insert(&db, record)
}

/// Query indexed time ranges, or the active task's index, without reading logs.
pub fn history(root: &Path, query: &Query) -> Result<Vec<Record>, String> {
    let directory = directory(root)?;
    if !directory.join("catalog.db").exists() {
        return Ok(Vec::new());
    }
    let _lock = lock(&directory)?;
    let db = redb::Database::open(directory.join("catalog.db")).map_err(|e| e.to_string())?;
    let tx = db.begin_read().map_err(|e| e.to_string())?;
    lookup(&tx, query)
}

fn read_record(runs: &redb::ReadOnlyTable<&str, &[u8]>, id: &str) -> Result<Record, String> {
    let bytes = runs
        .get(id)
        .map_err(|e| e.to_string())?
        .ok_or("catalogue references a missing run; use tune --history --reindex")?;
    codec::decode(bytes.value())
}

impl Query {
    fn matches(&self, r: &Record) -> bool {
        self.since.is_none_or(|t| r.started >= t)
            && self.until.is_none_or(|t| r.started < t)
            && self.task.as_ref().is_none_or(|t| &r.task == t)
            && self.optimizer.as_ref().is_none_or(|o| &r.optimizer == o)
            && self.protocol.as_ref().is_none_or(|p| &r.protocol == p)
    }
}

fn lookup(tx: &redb::ReadTransaction, query: &Query) -> Result<Vec<Record>, String> {
    let runs = match tx.open_table(RUNS) {
        Ok(runs) => runs,
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
        Err(e) => return Err(e.to_string()),
    };
    let mut records = Vec::new();
    let limit = query.limit.clamp(1, 100);
    if let Some(key) = query.task.as_ref().or(query.protocol.as_ref()) {
        let table = if query.task.is_some() {
            TASKS
        } else {
            PROTOCOLS
        };
        let index = tx.open_multimap_table(table).map_err(|e| e.to_string())?;
        let start = query.since.unwrap_or(i64::MIN);
        let end = query.until.unwrap_or(i64::MAX);
        if start >= end {
            return Ok(records);
        }
        'scoped: for entry in index
            .range((key.as_str(), start)..(key.as_str(), end))
            .map_err(|e| e.to_string())?
            .rev()
        {
            let (_, ids) = entry.map_err(|e| e.to_string())?;
            for id in ids.rev() {
                let row = read_record(&runs, id.map_err(|e| e.to_string())?.value())?;
                if query.matches(&row) {
                    records.push(row);
                }
                if records.len() == limit {
                    break 'scoped;
                }
            }
        }
    } else {
        let times = tx.open_multimap_table(TIMES).map_err(|e| e.to_string())?;
        let range = query.since.unwrap_or(i64::MIN)..query.until.unwrap_or(i64::MAX);
        if range.is_empty() {
            return Ok(records);
        }
        'times: for entry in times.range(range).map_err(|e| e.to_string())?.rev() {
            let (_, ids) = entry.map_err(|e| e.to_string())?;
            for id in ids.rev() {
                let row = read_record(&runs, id.map_err(|e| e.to_string())?.value())?;
                if query.matches(&row) {
                    records.push(row);
                }
                if records.len() == limit {
                    break 'times;
                }
            }
        }
    }
    Ok(records)
}

/// Rebuild only the derived index. Saved manifests and measurements are
/// untouched.
pub fn reindex(root: &Path) -> Result<usize, String> {
    let directory = directory(root)?;
    let _lock = lock(&directory)?;
    backfill::discover(root, &directory)?;
    let records = codec::manifests(&directory.join("manifests"))?;
    let temporary = directory.join("rebuild.db");
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    }
    {
        let db = redb::Database::create(&temporary).map_err(|e| e.to_string())?;
        for record in &records {
            insert(&db, record)?;
        }
    }
    fs::rename(temporary, directory.join("catalog.db")).map_err(|e| e.to_string())?;
    Ok(records.len())
}

/// Calendar dates are local; RFC3339 inputs specify their own offset.
pub fn since(input: &str) -> Result<i64, String> {
    let now = jiff::Zoned::now();
    let date = match input {
        "today" => Some(now.date()),
        "yesterday" => Some(
            now.date()
                .checked_sub(jiff::Span::new().days(1))
                .map_err(|e| e.to_string())?,
        ),
        _ => input.parse::<jiff::civil::Date>().ok(),
    };
    let timestamp = if let Some(date) = date {
        date.at(0, 0, 0, 0)
            .to_zoned(now.time_zone().clone())
            .map_err(|e| e.to_string())?
            .timestamp()
    } else {
        input
            .parse::<jiff::Timestamp>()
            .map_err(|_| "use today, yesterday, YYYY-MM-DD or an RFC3339 timestamp".to_string())?
    };
    Ok(timestamp.as_millisecond())
}

pub fn local_time(millis: i64) -> Result<String, String> {
    Ok(jiff::Timestamp::from_millisecond(millis)
        .map_err(|e| e.to_string())?
        .to_zoned(jiff::tz::TimeZone::system())
        .strftime("%Y-%m-%d %H:%M:%S %Z")
        .to_string())
}
