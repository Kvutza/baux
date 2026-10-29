use std::io::Write;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::time::{Duration, Instant};

use baux::artifact::terminal::{Tag, render, theme};

#[derive(Clone, Copy)]
enum Tone {
    Normal,
    Muted,
    Warning,
    Error,
}

pub(super) struct Terminal {
    sender: SyncSender<Vec<u8>>,
    done: Receiver<()>,
    pub skipped: u64,
}

impl Terminal {
    pub fn new(mut output: impl Write + Send + 'static) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(32);
        let (finished, done) = mpsc::channel();
        std::thread::spawn(move || {
            let mut active = Instant::now();
            let mut pulse_at = active;
            let mut heartbeat = false;
            loop {
                match receiver.recv_timeout(Duration::from_millis(200)) {
                    Ok(line) => {
                        if heartbeat && output.write_all(b"\r\x1b[2K").is_err() {
                            break;
                        }
                        heartbeat = false;
                        active = Instant::now();
                        if output.write_all(display_line(&line).as_bytes()).is_err() {
                            break;
                        }
                    }
                    Err(RecvTimeoutError::Timeout)
                        if active.elapsed() >= Duration::from_secs(2)
                            && pulse_at.elapsed() >= Duration::from_secs(1) =>
                    {
                        let pulse = render(
                            Tag::Heartbeat,
                            &format!("working · {:.0}s", active.elapsed().as_secs_f64()),
                            true,
                            theme(),
                        );
                        let line = format!("\r\x1b[2K{}", pulse.trim_end());
                        if output.write_all(line.as_bytes()).is_err() {
                            break;
                        }
                        let _ = output.flush();
                        heartbeat = true;
                        pulse_at = Instant::now();
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
            if heartbeat {
                let _ = output.write_all(b"\r\x1b[2K");
            }
            let _ = output.flush();
            let _ = finished.send(());
        });
        Self {
            sender,
            done,
            skipped: 0,
        }
    }

    pub fn send(&mut self, line: &[u8]) {
        let mut message = line[..line.len().min(16 * 1024)].to_vec();
        if message.len() < line.len() {
            message.extend_from_slice(b"... [full output in run.log]\n");
        }
        let required = line.starts_with(b"generation.round ")
            || line.starts_with(b"generation.summary ")
            || line.starts_with(b"generation.validation_summary ")
            || line.starts_with(b"experiment.plan ")
            || line.starts_with(b"experiment.phase ")
            || line.starts_with(b"pretrain.initial ")
            || line.starts_with(b"pretrain.round ")
            || line.starts_with(b"pretrain.validation ")
            || line.starts_with(b"optimizer.round ")
            || line.starts_with(b"optimizer.space ")
            || line.starts_with(b"optimizer.initial ")
            || line.starts_with(b"optimizer.loop ")
            || line.starts_with(b"optimizer.summary ")
            || line.starts_with(b"optimizer.setup ")
            || line.starts_with(b"optimizer.complete ")
            || line.starts_with(b"baux tune:")
            || line.starts_with(b"warning:");
        let failed = if required {
            self.sender.send(message).is_err()
        } else {
            self.sender.try_send(message).is_err()
        };
        if failed {
            self.skipped += 1;
        }
    }

    pub fn finish(self) {
        drop(self.sender);
        let _ = self.done.recv_timeout(Duration::from_millis(50));
    }
}

pub(super) fn display_line(line: &[u8]) -> String {
    let text = String::from_utf8_lossy(line);
    if let Some(rendered) = display_record(&text) {
        return rendered;
    }
    if machine_only(&text)
        || text.starts_with("TURBO_ENN_")
        || text.starts_with("generation.")
        || text.starts_with("BAUX_")
        || text.starts_with("FBT_")
    {
        return String::new();
    }
    let message = text
        .strip_prefix('[')
        .and_then(|s| s.split_once("] "))
        .filter(|(stamp, _)| stamp.starts_with("20"))
        .map_or(text.as_ref(), |(_, message)| message);
    if [
        "Build ID:",
        "File changed:",
        "Directory changed:",
        "Network:",
        "Cache hits:",
        "Commands:",
        "BUILD SUCCEEDED - starting your binary",
    ]
    .iter()
    .any(|prefix| message.starts_with(prefix))
        || message
            .trim_end()
            .ends_with("additional file change events")
        || text.starts_with("[weights] per-tensor records:")
        || text.starts_with("[optimization] round records:")
    {
        return String::new();
    }
    if let Some(stage) = message.strip_prefix("[tune] ") {
        return tagged(stage.trim_end(), Tone::Muted);
    }
    if let Some(model) = message.strip_prefix("[model] ") {
        return tagged(model.trim_end(), Tone::Normal);
    }
    if message.starts_with("baux tune:")
        || message.starts_with("TuRBO-ENN worker:")
        || message.starts_with("error:")
        || message.starts_with("Error:")
        || message.starts_with("process failed")
    {
        return tagged(message.trim_end(), Tone::Error);
    }
    if message.starts_with("warning:") || message.starts_with("Warning:") {
        return tagged(message.trim_end(), Tone::Warning);
    }
    tagged(message.trim_end(), Tone::Normal)
}

fn tagged(text: &str, tone: Tone) -> String {
    let tag = match tone {
        Tone::Normal => Tag::Tuning,
        Tone::Muted => Tag::Body,
        Tone::Warning => Tag::Warning,
        Tone::Error => Tag::Error,
    };
    render(tag, text, true, theme())
}

fn tagged_as(tag: Tag, text: &str) -> String {
    render(tag, text, true, theme())
}

fn machine_only(text: &str) -> bool {
    [
        "generation.stage ",
        "generation.repetition ",
        "generation.controls ",
        "generation.reference ",
        "generation.audit ",
        "generation.progress ",
        "pretrain.causal_validation ",
        "pretrain.",
        "optimizer.",
        "scorer.",
        "moe.",
        "pisa1.",
        "diffusion.",
        "block.",
        "kernel.",
        "decode.",
        "reconstruction.",
        "code.",
        "journal.",
        "readout.",
        "TURBO_ENN_TRUST ",
        "TURBO_ENN_CTRL ",
        "TURBO_ENN_FAMILY ",
        "TURBO_ENN_PARAMETERS ",
        "TURBO_ENN_RELIABILITY ",
        "TURBO_ENN round=", // handled by `display_record`; hides malformed records
        "TURBO_ENN complete ",
        "TURBO_ENN setup_",
        "FBT_MODEL ",
        "FBT_PREFILL ",
    ]
    .iter()
    .any(|prefix| text.starts_with(prefix))
}

fn record_field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.split_whitespace()
        .find_map(|part| part.split_once('=').filter(|(key, _)| *key == name))
        .map(|(_, value)| value)
}

fn generated_record(text: &str) -> Option<String> {
    let field = |name| record_field(text, name);
    if field("phase") != Some("initial") {
        return None;
    }
    let (label, value) = field("nll")
        .filter(|value| *value != "unavailable")
        .map(|nll| ("nll", nll))
        .or_else(|| field("bits_per_token").map(|value| ("bits/token", value)))
        .or_else(|| field("reward").map(|reward| ("reward", reward)))?;
    Some(tagged(&format!("initial · {label} {value}"), Tone::Muted))
}

fn display_record(text: &str) -> Option<String> {
    native_record(text).or_else(|| legacy_record(text))
}

fn native_record(text: &str) -> Option<String> {
    experiment_record(text)
        .or_else(|| optimizer_record(text))
        .or_else(|| workload_record(text))
}

fn optimizer_record(text: &str) -> Option<String> {
    if text.starts_with("optimizer.setup ") || text.starts_with("TURBO_ENN setup_seconds=") {
        return setup_line(text);
    }
    if text.starts_with("optimizer.round ") {
        return round_record(text).or_else(|| round_line(text));
    }
    if text.starts_with("TURBO_ENN round=") {
        return round_line(text);
    }
    if text.starts_with("optimizer.complete ") || text.starts_with("TURBO_ENN complete ") {
        return summary_line(text);
    }
    if text.starts_with("optimizer.space ") {
        return space_record(text);
    }
    if text.starts_with("optimizer.initial ") {
        return initial_record(text);
    }
    if text.starts_with("optimizer.loop ") {
        return loop_record(text);
    }
    text.strip_prefix("optimizer.summary ")
        .and_then(summary_record)
}

fn workload_record(text: &str) -> Option<String> {
    if text.starts_with("pretrain.initial ") {
        return pretrain_initial(text);
    }
    if text.starts_with("pretrain.round ") {
        return pretrain_round(text);
    }
    if text.starts_with("pretrain.validation ") {
        return pretrain_validation(text);
    }
    if text.starts_with("generation.summary ") {
        return generation_summary(text);
    }
    if text.starts_with("Pretraining") || text.starts_with("Generation") {
        return Some(tagged_as(
            Tag::Orchestration,
            &text.trim().to_ascii_lowercase(),
        ));
    }
    if text.starts_with("generation.round ") {
        return generation_record(text);
    }
    if text.starts_with("generation.sample ") {
        return generated_record(text);
    }
    None
}

fn generation_summary(text: &str) -> Option<String> {
    let target = if record_field(text, "target_met")? == "true" {
        "target met"
    } else {
        "target missed"
    };
    let summary = format!(
        "complete · {} rounds · {} accepted · {}s loop · {}ms mean · {}ms max · {target}",
        record_field(text, "rounds")?,
        record_field(text, "accepted")?,
        record_field(text, "loop_s")?,
        record_field(text, "mean_round_ms")?,
        record_field(text, "max_round_ms")?,
    );
    Some(format!(
        "{}{}",
        tagged(&summary, Tone::Normal),
        tagged(
            &format!("result {}", record_field(text, "result")?),
            Tone::Muted
        )
    ))
}

fn pretrain_initial(text: &str) -> Option<String> {
    Some(tagged(
        &format!(
            "initial training draw · {} · nll {:.6} · {} tokens · {:.0}ms GPU",
            record_field(text, "source")?,
            finite_field(text, "nll")?,
            record_field(text, "scored_tokens")?,
            finite_field(text, "gpu_ms")?,
        ),
        Tone::Muted,
    ))
}

fn pretrain_round(text: &str) -> Option<String> {
    let round = record_field(text, "round")?.parse::<u64>().ok()?;
    let accepted = record_field(text, "accepted")? == "true";
    let status = if accepted { "accepted" } else { "rejected" };
    let total =
        finite_field(text, "total").map_or_else(String::new, |total| format!("/{total:.0}"));
    let mut row = format!("round {round}{total} · {status}");
    if let Some(gain) = finite_field(text, "gain").or_else(|| finite_field(text, "paired_gain")) {
        row.push_str(&format!(" · gain {gain:+.6}"));
    }
    if let Some(pred_gain) = finite_field(text, "pred_gain") {
        row.push_str(&format!(" · pred {pred_gain:+.6}"));
    }
    if let Some(nll) = finite_field(text, "nll") {
        row.push_str(&format!(" · candidate nll {nll:.6}"));
    }
    if let Some(wall_ms) = finite_field(text, "wall_ms") {
        row.push_str(&format!(" · {wall_ms:.0}ms wall"));
    }
    if let (Some(proposed), Some(next)) = (
        finite_field(text, "proposal_radius"),
        finite_field(text, "next_radius"),
    ) {
        if (proposed - next).abs() > 1e-6 {
            row.push_str(&format!(" · radius {proposed:.6}→{next:.6}"));
        } else {
            row.push_str(&format!(" · radius {proposed:.6}"));
        }
    }
    Some(tagged(&row, Tone::Normal))
}

fn pretrain_validation(text: &str) -> Option<String> {
    Some(tagged_as(
        Tag::Metric,
        &format!(
            "validation {} · nll {:.6} · {} · {} evaluated tokens",
            record_field(text, "round")?,
            finite_field(text, "nll")?,
            record_field(text, "sources")?.replace(',', " · "),
            record_field(text, "evaluated_tokens")?,
        ),
    ))
}

fn experiment_record(text: &str) -> Option<String> {
    if text.starts_with("experiment.plan ") {
        plan_line(text)
    } else if text.starts_with("experiment.phase ") {
        phase_line(text)
    } else if text.starts_with("generation.validation_summary ") {
        validation_line(text)
    } else {
        None
    }
}

fn plan_line(text: &str) -> Option<String> {
    if record_field(text, "sampling") == Some("natural") {
        return Some(format!(
            "{}{}{}",
            tagged_as(
                Tag::Orchestration,
                &format!(
                    "plan · {} · {} · {} rounds · {:.0}ms target",
                    record_field(text, "model")?,
                    record_field(text, "optimizer")?.to_ascii_lowercase(),
                    record_field(text, "rounds")?,
                    finite_field(text, "target_ms")?,
                ),
            ),
            tagged("data · natural corpus units", Tone::Muted),
            tagged(
                &format!(
                    "objective · {} · validation at powers of two and every {} rounds",
                    record_field(text, "objective")?,
                    record_field(text, "validation_interval")?,
                ),
                Tone::Muted,
            ),
        ));
    }
    Some(format!(
        "{}{}{}",
        tagged_as(
            Tag::Orchestration,
            &format!(
                "plan · {} · {} · {} rounds · {:.0}ms target",
                record_field(text, "model")?,
                record_field(text, "optimizer")?.to_ascii_lowercase(),
                record_field(text, "rounds")?,
                finite_field(text, "target_ms")?,
            ),
        ),
        tagged(
            &format!(
                "capacity · {} context · allocation budget {} prompt / {} generation",
                record_field(text, "context_capacity")?,
                record_field(text, "prompt_budget")?,
                record_field(text, "generation_budget")?,
            ),
            Tone::Muted,
        ),
        tagged(
            &format!(
                "objective · {} · validation at powers of two and every {} rounds",
                record_field(text, "objective")?,
                record_field(text, "validation_interval")?,
            ),
            Tone::Muted,
        ),
    ))
}

fn phase_line(text: &str) -> Option<String> {
    let name = record_field(text, "name")?.replace('-', " ");
    match record_field(text, "status")? {
        "start" => Some(tagged_as(Tag::Orchestration, &format!("{name} · running"))),
        "done" => Some(tagged_as(
            Tag::Orchestration,
            &format!(
                "{name} · {:.1}s",
                finite_field(text, "elapsed_ms")? / 1000.0
            ),
        )),
        _ => None,
    }
}

fn validation_line(text: &str) -> Option<String> {
    let metric = record_field(text, "metric")?;
    Some(tagged_as(
        Tag::Metric,
        &format!(
            "validation {} · {} tasks · {metric} {:.6} · {:.1}s",
            record_field(text, "phase")?.replace("validation-", ""),
            record_field(text, "tasks")?,
            finite_field(text, "value")?,
            finite_field(text, "elapsed_ms")? / 1000.0,
        ),
    ))
}

fn legacy_record(text: &str) -> Option<String> {
    if text.starts_with("TURBO_ENN_ACTUAL_ROUND ") {
        return round_record(text);
    }
    if text.starts_with("TURBO_ENN_FULL_SPACE ") {
        return space_record(text);
    }
    if text.starts_with("TURBO_ENN_ACTUAL_INITIAL ") {
        return initial_record(text);
    }
    if text.starts_with("TURBO_ENN_LOOP ") {
        return loop_record(text);
    }
    match text.strip_prefix("TURBO_ENN_SUMMARY ") {
        Some(summary) => summary_record(summary),
        None => None,
    }
}

fn setup_line(text: &str) -> Option<String> {
    Some(tagged(
        &format!(
            "model · {:.3}B parameters · {} context · batch {} · {} candidates · {:.0}ms target",
            finite_field(text, "parameters")? / 1e9,
            record_field(text, "context")?,
            record_field(text, "batch")?,
            record_field(text, "pool")?,
            finite_field(text, "target_round_ms")?,
        ),
        Tone::Normal,
    ))
}

fn round_line(text: &str) -> Option<String> {
    if text.contains("wall_including_result_logging_seconds=") {
        return Some(String::new());
    }
    let accepted = record_field(text, "accepted")? == "true";
    let decision = if accepted { "accepted" } else { "rejected" };
    Some(tagged(
        &format!(
            "round {} · {:.0}ms · nll {:.6} · incumbent {:.6} · {} eval · {:.1} GiB · {decision}",
            record_field(text, "round")?,
            finite_field(text, "compute_seconds")? * 1000.0,
            finite_field(text, "candidate_nll")?,
            finite_field(text, "incumbent_nll")?,
            record_field(text, "objective_calls")?,
            finite_field(text, "allocated_GiB")?,
        ),
        Tone::Normal,
    ))
}

fn summary_line(text: &str) -> Option<String> {
    let met = if finite_field(text, "mean_round_ms")? <= finite_field(text, "target_round_ms")? {
        "target met"
    } else {
        "target missed"
    };
    Some(tagged(
        &format!(
            "complete · {} rounds · {:.3}s loop · {:.0}ms mean · {:.0}ms target · {met}",
            record_field(text, "rounds")?,
            finite_field(text, "loop_seconds")?,
            finite_field(text, "mean_round_ms")?,
            finite_field(text, "target_round_ms")?,
        ),
        Tone::Normal,
    ))
}

fn finite_field(text: &str, name: &str) -> Option<f64> {
    record_field(text, name)?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn generation_record(text: &str) -> Option<String> {
    let round = finite_field(text, "round")? as u64;
    let wall_ms = finite_field(text, "wall_ms").or_else(|| finite_field(text, "elapsed_ms"))?;
    let reward = finite_field(text, "reward")?;
    let accepted = record_field(text, "accepted")
        .map(|a| a == "true")
        .unwrap_or_else(|| {
            record_field(text, "status") == Some("accepted")
                || record_field(text, "status") == Some("applied")
        });
    let status = if accepted {
        "accepted"
    } else if record_field(text, "trust_outcome") == Some("Inconclusive") {
        "inconclusive"
    } else {
        "rejected"
    };
    let (label, value) = finite_field(text, "nll").map_or(("reward", reward), |nll| ("nll", nll));
    let total =
        finite_field(text, "total").map_or_else(String::new, |total| format!("/{total:.0}"));
    let mut row = format!("round {round}{total} · {status}");
    if let Some(gain) = finite_field(text, "gain").or_else(|| finite_field(text, "paired_gain")) {
        row.push_str(&format!(" · gain {gain:+.6}"));
    }
    if let Some(pred_gain) = finite_field(text, "pred_gain") {
        row.push_str(&format!(" · pred {pred_gain:+.6}"));
    }
    row.push_str(&format!(
        " · candidate {label} {value:.6} · {wall_ms:.0}ms wall"
    ));
    if let (Some(proposal), Some(evaluation), Some(tell)) = (
        finite_field(text, "proposal_ms"),
        finite_field(text, "evaluation_ms"),
        finite_field(text, "tell_ms"),
    ) {
        row.push_str(&format!(
            " · {proposal:.0}ms proposal · {evaluation:.0}ms evaluation · {tell:.1}ms tell"
        ));
    }
    if let (Some(proposed), Some(next)) = (
        finite_field(text, "proposal_radius"),
        finite_field(text, "next_radius"),
    ) && (proposed - next).abs() > 1e-6
    {
        row.push_str(&format!(" · radius {proposed:.6}→{next:.6}"));
    }
    Some(tagged(&row, Tone::Normal))
}

fn round_record(text: &str) -> Option<String> {
    let decision = match record_field(text, "accepted")? {
        "true" => "applied",
        "false" => "rejected",
        _ => return None,
    };
    let parameters = finite_field(text, "parameters")?;
    if parameters <= 0.0 {
        return None;
    }
    Some(tagged(
        &format!(
            "round {}/{} · {:.0}ms wall · {:.0}ms score · nll {:.5} · {:.2}% changed · {:.3}% \
             radius · {decision}",
            record_field(text, "round")?,
            record_field(text, "total")?,
            finite_field(text, "wall_seconds")? * 1000.0,
            finite_field(text, "scorer_gpu_seconds")? * 1000.0,
            -finite_field(text, "reward")?,
            finite_field(text, "changed_weights")? / parameters * 100.0,
            finite_field(text, "proposal_radius")? * 100.0,
        ),
        Tone::Normal,
    ))
}

fn space_record(text: &str) -> Option<String> {
    let proposals = match record_field(text, "noise")? {
        "independent_rademacher" => "Rademacher",
        "independent_gaussian" | "independent_gaussian_ziggurat256_v1" => "Gaussian",
        _ => return None,
    };
    let params_b = finite_field(text, "parameters")? / 1e9;
    Some(tagged(
        &format!(
            "model · {params_b:.3}B weights · fp16 · context {} · batch {} · full-weight \
             {proposals} proposals",
            record_field(text, "context")?,
            record_field(text, "batch")?,
        ),
        Tone::Normal,
    ))
}

fn initial_record(text: &str) -> Option<String> {
    Some(tagged(
        &format!("initial · nll {:.5}", -finite_field(text, "reward")?),
        Tone::Normal,
    ))
}

fn loop_record(text: &str) -> Option<String> {
    Some(tagged(
        &format!(
            "loop · {:.3}s · {:.3} rounds/s",
            finite_field(text, "elapsed_seconds")?,
            finite_field(text, "rounds_per_second")?,
        ),
        Tone::Normal,
    ))
}

fn summary_record(text: &str) -> Option<String> {
    let field = |name| record_field(text, name);
    let number = |name| {
        field(name)?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
    };
    let met = match field("target_met")? {
        "true" => "met",
        "false" => "not met",
        _ => return None,
    };
    Some(tagged(
        &format!(
            "complete · {} rounds · {} accepted · {:.0}ms median · {:.0}ms max · {:.0}ms target \
             {met}",
            field("rounds")?,
            field("accepted")?,
            number("median_seconds")? * 1000.0,
            number("max_seconds")? * 1000.0,
            number("target_ms")?,
        ),
        Tone::Normal,
    ))
}

#[cfg(test)]
mod tests {
    use super::display_record;

    #[test]
    fn native_label() {
        assert!(
            display_record("generation.sample round=1 tokens=4096 reward=0.12 accepted=false")
                .is_none()
        );
    }

    #[test]
    fn legacy_records() {
        let record =
            display_record("generation.sample phase=initial tokens=4096 bits_per_token=17.9")
                .unwrap();
        assert!(record.contains("initial"));
        assert!(record.contains("bits/token 17.9"));
        assert!(!record.contains("4096"));
        let record = display_record("generation.sample phase=initial tokens=4096 nll=3.1").unwrap();
        assert!(record.contains("nll 3.1"));
    }

    #[test]
    fn test_generation() {
        let record = display_record(
            "generation.round round=1 total=1000 wall_ms=1850 proposal_ms=70 evaluation_ms=1779 \
             tell_ms=1 proposal_radius=0.4 next_radius=0.8 trust_outcome=Success reward=-12.2 \
             nll=12.2 paired_gain=0.03 paired_se=0.01 accepted=true",
        )
        .unwrap();
        assert!(record.contains("accepted"));
        assert!(record.contains("gain +0.030000"));
        assert!(record.contains("candidate nll 12.200000"));
        assert!(record.contains("1850ms wall"));
        assert!(record.contains("70ms proposal"));
        assert!(record.contains("1779ms evaluation"));
        assert!(record.contains("1.0ms tell"));
        assert!(record.contains("radius 0.400000→0.800000"));
        assert!(record.contains("t| "));
    }

    #[test]
    fn stable_radius() {
        let record = display_record(
            "generation.round round=2 total=1000 wall_ms=700 proposal_ms=80 evaluation_ms=619 \
             tell_ms=0.2 proposal_radius=0.400000006 next_radius=0.4 trust_outcome=Failure \
             reward=-12.3 nll=12.3 paired_gain=-0.1 paired_se=0.02 accepted=false",
        )
        .unwrap();
        assert!(record.contains("rejected"));
        assert!(!record.contains("radius"));
        assert!(!record.contains("generated"));
        assert!(!record.contains("context"));
        assert!(!record.contains("trust"));
    }

    #[test]
    fn worker_records() {
        let setup = display_record(
            "TURBO_ENN setup_seconds=9.0 parameters=1065494016 context=4096 batch=2 pool=4 \
             target_round_ms=1000",
        )
        .unwrap();
        assert!(setup.contains("1.065B parameters"));

        let round = display_record(
            "TURBO_ENN round=1 compute_seconds=30.406615 incumbent_nll=12.013118744 \
             candidate_nll=12.016749717 accepted=false objective_calls=1 allocated_GiB=16.0608",
        )
        .unwrap();
        assert!(round.contains("30407ms"));
        assert!(round.contains("rejected"));

        let complete = display_record(
            "TURBO_ENN complete rounds=1 loop_seconds=30.406703 mean_round_ms=30406.615 \
             target_round_ms=1000 goal_met=false",
        )
        .unwrap();
        assert!(complete.contains("target missed"));
    }

    #[test]
    fn pretrain_records() {
        let initial = display_record(
            "pretrain.initial source=traces window_tokens=1048576 scored_tokens=1048513 \
             corpus_offset=544768 nll=12.464662552 bits_per_token=17.982707227 gpu_ms=1151.191",
        )
        .unwrap();
        assert!(initial.contains("initial training draw"));
        assert!(initial.contains("traces"));

        let accepted = display_record(
            "pretrain.round round=2 total=1000 source=traces offset=2891776 scored_tokens=1048513 \
             nll=12.463944435 accepted=true gpu_ms=1158.165 wall_ms=1302.713",
        )
        .unwrap();
        assert!(accepted.contains("round 2/1000"));
        assert!(accepted.contains("accepted"));
        let rejected = display_record(
            "pretrain.round round=3 total=1000 source=code offset=1 scored_tokens=1048513 \
             nll=12.47 accepted=false gpu_ms=1158.165 wall_ms=1302.713",
        )
        .unwrap();
        assert!(rejected.contains("round 3/1000"));
        assert!(rejected.contains("rejected"));

        let validation = display_record(
            "pretrain.validation round=0 nll=12.466787815 bits_per_token=17.985772830 \
             scored_tokens=1048576 evaluated_tokens=2097026 \
             sources=code:12.477302551,traces:12.456273079 used_for_acceptance=false",
        )
        .unwrap();
        assert!(validation.contains("validation 0"));
        assert!(validation.contains("code:12.477302551"));
        assert!(validation.contains("2097026 evaluated tokens"));
    }
}
