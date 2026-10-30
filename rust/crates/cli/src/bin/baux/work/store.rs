use crate::work::{Value, json, text};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Beads {
    pub root: PathBuf,
    pub actor: String,
}

impl Beads {
    pub fn bound(binding: &Value) -> Result<Self, String> {
        Ok(Self {
            root: PathBuf::from(text(binding, "store")?),
            actor: text(binding, "actor")?.to_owned(),
        })
    }

    pub fn read(&self, args: &[&str]) -> Result<Value, String> {
        self.run(args, false)
    }
    pub fn write(&self, args: &[&str]) -> Result<Value, String> {
        self.run(args, true)
    }

    fn run(&self, args: &[&str], write: bool) -> Result<Value, String> {
        let mut command = Command::new("bd");
        command
            .current_dir(&self.root)
            .arg("-C")
            .arg(&self.root)
            .args(["--sandbox", "--actor", &self.actor, "--json"]);
        if !write {
            command.arg("--readonly");
        }
        command.args(args);
        let output = output(&mut command)?;
        json::from_str(&output).map_err(|error| format!("invalid Beads JSON: {error}"))
    }

    pub fn issues(&self, ids: &[&str]) -> Result<Vec<Value>, String> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut args = vec!["show", "--brief-deps"];
        for id in ids {
            args.extend(["--id", id]);
        }
        let rows = self.read(&args)?;
        rows.as_seq()
            .map(|r| r.to_vec())
            .ok_or("Beads show returned a non-list".into())
    }

    pub fn issue(&self, id: &str) -> Result<Value, String> {
        let rows = self.read(&["show", "--id", id, "--brief-deps"])?;
        let items = rows
            .as_seq()
            .ok_or("Beads show returned a non-list result")?;
        if items.len() != 1 {
            return Err(format!("expected one Beads task for {id}"));
        }
        Ok(items[0].clone())
    }
}

pub fn output(command: &mut Command) -> Result<String, String> {
    let program = command.get_program().to_string_lossy().into_owned();
    let result = command
        .output()
        .map_err(|error| format!("start {program}: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "{program} failed ({}): {}{}",
            result.status,
            String::from_utf8_lossy(&result.stderr),
            String::from_utf8_lossy(&result.stdout)
        ));
    }
    String::from_utf8(result.stdout)
        .map_err(|error| format!("{program} output is not UTF-8: {error}"))
}

pub fn revision(root: &Path) -> Result<Value, String> {
    let result = output(Command::new("jj").current_dir(root).args([
        "log",
        "-r",
        "@",
        "--no-graph",
        "--no-pager",
        "-T",
        "commit_id ++ \"\\n\" ++ change_id ++ \"\\n\"",
    ]))?;
    let mut lines = result.lines();
    let commit = lines.next().ok_or("JJ did not return a commit")?;
    let change = lines.next().ok_or("JJ did not return a change")?;
    Ok(json!({"commit": commit, "change": change}))
}

pub fn diff(root: &Path, base: &str, candidate: &str) -> Result<String, String> {
    output(Command::new("jj").current_dir(root).args([
        "diff",
        "--from",
        base,
        "--to",
        candidate,
        "--summary",
        "--no-pager",
    ]))
}

fn state_path(root: &Path) -> PathBuf {
    root.join(".cache/baux/task.json")
}

pub fn load(root: &Path) -> Result<Option<Value>, String> {
    let content = match fs::read_to_string(state_path(root)) {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read active task: {error}")),
    };
    let value: Value =
        json::from_str(&content).map_err(|error| format!("invalid active task: {error}"))?;
    if value.get("version").and_then(|item| item.as_u64()) != Some(1) {
        return Err("unsupported active-task format".into());
    }
    if Path::new(text(&value, "workspace")?) != root {
        return Err(
            "active-task state belongs to a different workspace; select the task in this workspace"
                .into(),
        );
    }
    for key in ["id", "store", "actor", "base", "change"] {
        text(&value, key)?;
    }
    Ok(Some(value))
}

pub fn save(root: &Path, binding: &Value) -> Result<(), String> {
    let path = state_path(root);
    let temporary = path.with_extension("json.tmp");
    let content = json::pretty_string(binding).map_err(|error| error.to_string())?;
    fs::write(&temporary, content).map_err(|error| format!("write active task: {error}"))?;
    fs::rename(temporary, path).map_err(|error| format!("publish active task: {error}"))
}

pub fn discover(
    root: &Path,
    explicit: Option<&Path>,
    saved: Option<&Value>,
) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return validate_store(path);
    }
    if let Some(binding) = saved {
        return validate_store(Path::new(text(binding, "store")?));
    }
    let pointer = root.join(".jj/repo");
    let shared = if pointer.is_file() {
        let relative = fs::read_to_string(&pointer).map_err(|error| error.to_string())?;
        root.join(".jj").join(relative.trim())
    } else {
        pointer
    };
    if let Ok(shared) = shared.canonicalize()
        && let Some(owner) = shared.parent().and_then(Path::parent)
        && owner.join(".beads/metadata.json").is_file()
    {
        return validate_store(owner);
    }
    validate_store(root)
}

fn validate_store(path: &Path) -> Result<PathBuf, String> {
    let path = path
        .canonicalize()
        .map_err(|error| format!("resolve Beads store: {error}"))?;
    if !path.join(".beads/metadata.json").is_file() {
        return Err(
            "no existing Beads store found; supply context --store PATH (no database was \
             initialized)"
                .into(),
        );
    }
    Ok(path)
}

pub struct Lock(PathBuf);

impl Lock {
    pub fn acquire(root: &Path) -> Result<Self, String> {
        let directory = root.join(".cache/baux");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let path = directory.join("task.lock");
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                format!(
                    "cannot lock task state at {}: {error}; another task command may be running",
                    path.display()
                )
            })?;
        use std::io::Write;
        if let Err(error) = writeln!(file, "{}", std::process::id()) {
            let _ = fs::remove_file(&path);
            return Err(error.to_string());
        }
        Ok(Self(path))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
