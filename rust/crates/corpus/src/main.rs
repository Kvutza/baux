use std::io::{self, BufReader, BufWriter, Write};

mod encoding;
mod mix;
mod reader;
mod stack;
mod text;
mod tokenizer;
mod traces;
use reader::{Result, collect};

#[tokio::main(worker_threads = 4)]
async fn main() -> Result<()> {
    let job: baux::artifact::json::Value =
        baux::artifact::json::from_reader(BufReader::new(io::stdin().lock()))?;
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("traces") {
        return traces::prepare(job);
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("stack") {
        return stack::prepare(job).await;
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("text") {
        return text::prepare(job).await;
    }
    if job.get("kind").and_then(|v| v.kind().as_str()) == Some("mix") {
        return mix::prepare(job);
    }
    let job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&job)?)?;
    let selected = collect(job).await?;
    let mut output = BufWriter::new(io::stdout().lock());
    baux::artifact::json::to_writer(&mut output, &selected)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::reader::{BUCKETS, Job, Selection, classify};
    use std::collections::BTreeMap;

    #[test]
    fn classification() {
        let languages = vec!["Rust".into(), "Go".into()];
        assert_eq!(
            classify("tests/model.rs", "Rust", &languages, &[]),
            Some("tests")
        );
        assert_eq!(
            classify("/src/model.rs", "Rust", &languages, &[]),
            Some("implementation")
        );
        assert_eq!(
            classify("Readme.custom", "Other", &languages, &[]),
            Some("documentation")
        );
        assert_eq!(
            classify("Cargo.toml", "TOML", &languages, &[]),
            Some("configuration")
        );
        assert_eq!(classify("main.py", "Python", &languages, &[]), None);
        assert_eq!(
            classify("Cargo.lock", "Rust", &languages, &["cargo.lock".into()]),
            None
        );
    }

    #[test]
    fn completion() {
        let targets = BTreeMap::from([(
            "train".into(),
            BUCKETS.into_iter().map(|b| (b.into(), 80)).collect(),
        )]);
        let job = Job {
            root: String::new(),
            paths: Vec::new(),
            targets: targets.clone(),
            target_characters: BTreeMap::new(),
            languages: vec!["Rust".into(), "Go".into()],
            excluded_names: vec!["cargo.lock".into(), "go.sum".into()],
            holdout_partitions: 1,
            max_file: 1024,
            max_repository: 4096,
        };
        let mut selected = Selection::new(&job);
        assert_eq!(selected.active(&job), 15);
        selected.characters = targets.clone();
        assert_eq!(selected.active(&job), 0);
    }
}
