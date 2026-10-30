use baux::config::load_tune;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

mod tune_output;
use tune_output::Terminal;

fn main() -> ExitCode {
    let mut terminal = Terminal::new(anstream::stderr());
    let status = match run(&mut terminal) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            terminal.send(format!("baux tune: {error}\n").as_bytes());
            ExitCode::FAILURE
        }
    };
    terminal.finish();
    status
}

fn run_command(
    mut command: Command,
    mut log: &std::fs::File,
    terminal: &mut Terminal,
) -> Result<(), String> {
    let (reader, writer) = std::io::pipe().map_err(|error| error.to_string())?;
    command
        .stdout(Stdio::from(
            writer.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(writer));
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    drop(command);
    let streamed = (|| -> std::io::Result<()> {
        let mut reader = std::io::BufReader::new(reader);
        let mut line = Vec::new();
        while reader.read_until(b'\n', &mut line)? != 0 {
            baux::artifact::terminal::write_log(&mut log, &line)?;
            terminal.send(&line);
            line.clear();
        }
        log.flush()
    })();
    if streamed.is_err() {
        let _ = child.kill();
    }
    let status = child.wait().map_err(|error| error.to_string())?;
    streamed.map_err(|error| format!("stream worker output: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed ({status})"))
    }
}

struct RunArtifacts {
    output: PathBuf,
    resolved_config: PathBuf,
    log: std::fs::File,
}

impl RunArtifacts {
    fn create(output_root: PathBuf, resolved: &str) -> Result<Self, String> {
        std::fs::create_dir_all(&output_root).map_err(|error| error.to_string())?;
        let output = allocate_output(&output_root)?;
        let resolved_config = output.join("experiment.toml");
        std::fs::write(&resolved_config, resolved).map_err(|error| error.to_string())?;
        write_snapshot(&output)?;
        let log =
            std::fs::File::create(output.join("run.log")).map_err(|error| error.to_string())?;
        Ok(Self {
            output,
            resolved_config,
            log,
        })
    }

    fn record_result(&self, result: &Result<(), String>, summary: &str) -> Result<(), String> {
        baux::artifact::terminal::write_log(&self.log, summary.as_bytes())
            .map_err(|error| error.to_string())?;
        let status = if result.is_ok() {
            "success\n"
        } else {
            "failure\n"
        };
        std::fs::write(self.output.join("exit.txt"), status).map_err(|error| error.to_string())?;
        if let Err(error) = result {
            if !self.output.join("result.toml").exists() {
                let message = error.replace('"', "'");
                std::fs::write(
                    self.output.join("result.toml"),
                    format!("status = \"failure\"\nstage = \"runner\"\nmessage = {message:?}\n"),
                )
                .map_err(|error| error.to_string())?;
            }
        }
        Ok(())
    }
}

fn allocate_output(root: &Path) -> Result<PathBuf, String> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    (0..100)
        .find_map(|suffix| {
            let path = root.join(format!("run-{timestamp}-{}-{suffix}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => Some(Ok(path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                Err(error) => Some(Err(error.to_string())),
            }
        })
        .ok_or_else(|| "could not allocate a unique artifact directory".to_string())?
}

fn write_snapshot(output: &Path) -> Result<(), String> {
    let revision = std::env::var("BAUX_SOURCE_REVISION")
        .map_err(|_| "sealed source revision is missing".to_string())?;
    if !(40..=64).contains(&revision.len())
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("sealed source revision is invalid".into());
    }
    std::fs::write(output.join("source-revision.txt"), format!("{revision}\n"))
        .map_err(|error| error.to_string())?;
    let commands = [
        Command::new("jj")
            .args(["log", "-r", &revision, "--no-graph", "-T", "commit_id"])
            .output(),
        Command::new("jj")
            .args(["diff", "-r", &revision, "--summary", "--no-pager"])
            .output(),
    ];
    let mut source =
        std::fs::File::create(output.join("source.txt")).map_err(|error| error.to_string())?;
    for result in commands {
        match result {
            Ok(result) if result.status.success() => {
                source
                    .write_all(&result.stdout)
                    .map_err(|error| error.to_string())?;
                source.write_all(b"\n").map_err(|error| error.to_string())?;
            }
            _ => source
                .write_all(b"unavailable\n")
                .map_err(|error| error.to_string())?,
        }
    }
    let root = std::env::current_dir().map_err(|error| error.to_string())?;
    let digest =
        baux::experimental::perf_digest(&root, baux::experimental::ExecutionTarget::Metal)?;
    std::fs::write(
        output.join("performance-source.sha256"),
        format!("{digest}\n"),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn worker_command(config: &Path, output: &Path) -> Result<Command, String> {
    let executable = std::env::var_os("BAUX_WORKER_EXECUTABLE")
        .ok_or("worker executable is missing; launch the experiment through ./baux tune")?;
    let mut worker = Command::new(executable);
    worker.arg(config).arg(output);
    // Each worker owns one run directory. Override any inherited capture path
    // so compilation evidence cannot be silently attached to another run.
    worker.env(
        "BAUX_GPU_EVIDENCE",
        output
            .canonicalize()
            .map_err(|error| error.to_string())?
            .join("gpu"),
    );
    Ok(worker)
}

fn run(terminal: &mut Terminal) -> Result<(), String> {
    let input = std::env::args_os()
        .nth(1)
        .ok_or("expected a TuRBO-ENN TOML path")?;
    let (run, resolved) = load_tune(Path::new(&input))?;
    let artifacts = RunArtifacts::create(run.output(), &resolved)?;
    let root = std::env::current_dir().map_err(|error| error.to_string())?;
    let revision = std::env::var("BAUX_SOURCE_REVISION").map_err(|error| error.to_string())?;
    let generated_pretrain =
        run.experiment == Some(baux::ExperimentKind::Pretrain) && run.generation.is_some();
    let title = if run.experiment == Some(baux::ExperimentKind::Generation) {
        "Generation"
    } else if generated_pretrain {
        "Pretraining"
    } else if run.experiment == Some(baux::ExperimentKind::Pretrain) {
        "BAUX pretrain"
    } else {
        "BAUX diagnostics"
    };
    terminal.send(format!("{title}\n").as_bytes());
    if let Some(plan) = plan_record(&run)? {
        terminal.send(plan.as_bytes());
    }
    let mut evidence = baux::artifact::runs::start(&root, &artifacts.output, &resolved, &revision)?;
    let started = std::time::Instant::now();
    let result = worker_command(&artifacts.resolved_config, &artifacts.output)
        .and_then(|command| run_command(command, &artifacts.log, terminal));
    let summary = format!(
        "process {} · {:.1}s\nartifacts {}\n",
        if result.is_ok() { "complete" } else { "failed" },
        started.elapsed().as_secs_f64(),
        artifacts.output.display(),
    );
    let saved_result = artifacts.record_result(&result, &summary);
    // Reporting runs after the measured worker execution. It cannot change the
    // saved training status or discard the original process failure.
    if let Err(error) = baux::artifact::runs::finish(&root, &mut evidence, &result) {
        terminal.send(format!("warning: evidence finalization failed: {error}\n").as_bytes());
    }
    terminal.send(
        format!(
            "report {}{}\n",
            evidence.report_status,
            evidence
                .report_error
                .as_ref()
                .map(|e| format!(": {e}"))
                .unwrap_or_default()
        )
        .as_bytes(),
    );
    if terminal.skipped > 0 {
        let warning = format!(
            "warning: terminal skipped {} low-priority messages; run.log is complete\n",
            terminal.skipped
        );
        baux::artifact::terminal::write_log(&artifacts.log, warning.as_bytes())
            .map_err(|error| error.to_string())?;
        terminal.send(warning.as_bytes());
    }
    terminal.send(summary.as_bytes());
    result.and(saved_result).map_err(|error| {
        format!(
            "experiment failed: {error}; see {}/run.log",
            artifacts.output.display()
        )
    })
}

fn plan_record(run: &baux::config::RunConfig) -> Result<Option<String>, String> {
    let Some(generation) = run.generation.as_ref() else {
        return Ok(None);
    };
    let model = run.model.ok_or("generation model is required")?;
    let optimizer = match run.optimizer.unwrap_or_default() {
        baux::config::TrainingOptimizer::ResidentEnnV1 => "resident-enn-v1",
        baux::config::TrainingOptimizer::OperatorBo => "operator-bo",
        baux::config::TrainingOptimizer::MuonV1 => "muon-v1",
        baux::config::TrainingOptimizer::AdamwV1 => "adamw-v1",
    };
    if matches!(
        run.optimizer(),
        baux::config::TrainingOptimizer::MuonV1 | baux::config::TrainingOptimizer::AdamwV1
    ) {
        return Ok(Some(format!(
            "experiment.plan model={} optimizer={} rounds={} sampling=natural \
             objective=cross-entropy validation_interval={} target_ms={}\n",
            model.id(),
            optimizer,
            run.rounds(),
            run.validation_interval.unwrap_or(run.rounds()),
            run.target_ms(),
        )));
    }
    let prompt = generation
        .corpus_prompt_tokens
        .map_or(generation.corpus_prompt.len(), |tokens| tokens as usize);
    let context = prompt
        .checked_add(generation.max_tokens as usize)
        .ok_or("configured context token count overflow")?;
    let objective = match generation.reward {
        baux::config::GenerationReward::ConditionalCrossEntropy => "conditional-cross-entropy",
        baux::config::GenerationReward::CausalObjectives => "causal-objectives",
        baux::config::GenerationReward::BoundedCrossEntropy => "bounded-cross-entropy",
        baux::config::GenerationReward::SequenceKernel => "sequence-kernel",
        baux::config::GenerationReward::CodeExecution { .. } => "code-execution",
        _ => "generated-reward",
    };
    Ok(Some(format!(
        "experiment.plan model={} optimizer={} rounds={} prompt_budget={} generation_budget={} \
         context_capacity={} objective={} validation_interval={} target_ms={}\n",
        model.id(),
        optimizer,
        run.rounds(),
        prompt,
        generation.max_tokens,
        context,
        objective,
        run.validation_interval.unwrap_or(run.rounds()),
        run.target_ms(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_capture() {
        if let Some(output) = std::env::var_os("BAUX_TEST_CAPTURE") {
            let output = PathBuf::from(output);
            let command = worker_command(Path::new("config"), &output).unwrap();
            let value = command
                .get_envs()
                .find(|(key, _)| *key == "BAUX_GPU_EVIDENCE")
                .and_then(|(_, value)| value)
                .unwrap();
            assert_eq!(Path::new(value), output.canonicalize().unwrap().join("gpu"));
            return;
        }
        let output = tempfile::tempdir().unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "tests::worker_capture"])
            .env("BAUX_TEST_CAPTURE", output.path())
            .env("BAUX_WORKER_EXECUTABLE", "not-executed")
            .env("BAUX_GPU_EVIDENCE", "/wrong/inherited/destination")
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn plain(line: &[u8]) -> String {
        let rendered = tune_output::display_line(line);
        let mut output = anstream::StripStream::new(Vec::new());
        output.write_all(rendered.as_bytes()).unwrap();
        String::from_utf8(output.into_inner()).unwrap()
    }

    #[test]
    fn streams() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'out\\n'; printf 'err\\n' >&2; printf 'tail'; exit 7",
        ]);
        let mut terminal = Terminal::new(std::io::sink());
        assert!(
            run_command(command, &log, &mut terminal)
                .unwrap_err()
                .contains("7")
        );
        terminal.finish();
        let log = std::fs::read_to_string(path).unwrap();
        assert!(log.lines().any(|line| line.ends_with(" out")));
        assert!(log.lines().any(|line| line.ends_with(" err")));
        assert!(log.lines().any(|line| line.ends_with(" tail")));
    }

    #[test]
    fn live_output() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let ack = directory.path().join("ack");
        let log = std::fs::File::create(&path).unwrap();
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "printf 'ready\\n'; while [ ! -f \"$1\" ]; do sleep 0.01; done; printf 'done\\n'",
            "test",
        ]);
        command.arg(&ack);
        let worker = std::thread::spawn(move || {
            let mut terminal = Terminal::new(std::io::sink());
            let result = run_command(command, &log, &mut terminal);
            terminal.finish();
            result
        });
        let started = std::time::Instant::now();
        let mut seen = false;
        while started.elapsed().as_secs() < 5 {
            if std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .any(|line| line.ends_with(" ready"))
            {
                seen = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        std::fs::write(ack, "").unwrap();
        worker.join().unwrap().unwrap();
        assert!(seen, "output must reach the log before the worker exits");
        let log = std::fs::read_to_string(path).unwrap();
        assert!(log.lines().any(|line| line.ends_with(" ready")));
        assert!(log.lines().any(|line| line.ends_with(" done")));
    }

    #[test]
    fn round_display() {
        let line = b"optimizer.round round=2 total=100 wall_seconds=0.88 scorer_gpu_seconds=0.87 reward=-9 variance=0.01 accepted=false radius=0.01 proposal_radius=0.005 changed_weights=95 parameters=100\n";
        assert_eq!(
            plain(line),
            "t| round 2/100 · 880ms wall · 870ms score · nll 9.00000 · 95.00% changed · 0.500% \
             radius · rejected\n"
        );
        assert_eq!(
            plain(b"optimizer.trust round=10 outcome=Inconclusive\n"),
            ""
        );
        assert_eq!(plain(b"diagnostic\n"), "t| diagnostic\n");
        assert_eq!(
            plain(b"generation.sample round=3 tokens=4096 nll=9.2 accepted=false\n"),
            ""
        );
        assert_eq!(plain(b"generation.preview \"generated text\"\n"), "");
        assert_eq!(plain(b"optimizer.round broken\n"), "");
        assert_eq!(plain(b"[2026-09-28] File changed: x\n"), "");
        assert_eq!(
            plain(b"[2026-09-28] error: link failed\n"),
            "e|error: link failed\n"
        );
    }

    #[test]
    fn slow_terminal() {
        struct Blocked(std::sync::mpsc::Receiver<()>);
        impl Write for Blocked {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let _ = self.0.recv();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let (release, blocked) = std::sync::mpsc::channel();
        let (finished, done) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut terminal = Terminal::new(Blocked(blocked));
            let mut command = Command::new("sh");
            command.args([
                "-c",
                "i=0; while [ $i -lt 1000 ]; do printf 'record\\n'; i=$((i+1)); done",
            ]);
            let result = run_command(command, &log, &mut terminal);
            let skipped = terminal.skipped;
            terminal.finish();
            let _ = finished.send((result, skipped));
        });
        let outcome = done.recv_timeout(std::time::Duration::from_secs(5));
        drop(release);
        worker.join().unwrap();
        let (result, skipped) =
            outcome.expect("terminal must not block log collection or shutdown");
        result.unwrap();
        assert!(skipped > 0);
        let log = std::fs::read_to_string(path).unwrap();
        assert_eq!(
            log.lines().filter(|line| line.ends_with(" record")).count(),
            1000
        );
    }

    #[test]
    fn closed_terminal() {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run.log");
        let log = std::fs::File::create(&path).unwrap();
        let mut terminal = Terminal::new(Closed);
        let mut command = Command::new("sh");
        command.args(["-c", "printf 'first\\nlast\\n'"]);
        run_command(command, &log, &mut terminal).unwrap();
        terminal.finish();
        let log = std::fs::read_to_string(path).unwrap();
        assert!(log.lines().any(|line| line.ends_with(" first")));
        assert!(log.lines().any(|line| line.ends_with(" last")));
    }
}
