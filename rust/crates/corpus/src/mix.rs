//! Deterministic interleaving of complete tokenized corpus units.
use super::reader::Result;
use baux::artifact::json::{Value, json};
use baux::artifact::pretrain::PretrainDataset;
use deser::Deserialize;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::PathBuf;

const CONTEXT: u32 = 4096;

#[derive(Deserialize)]
struct Job {
    output: PathBuf,
    report: PathBuf,
    bos: u32,
    eos: u32,
    seed: u64,
    parts: Vec<Part>,
}

#[derive(Deserialize)]
struct Part {
    name: String,
    path: PathBuf,
    marker: Option<u32>,
}

struct Source {
    name: String,
    data: PretrainDataset,
    units: Vec<Range<usize>>,
    next: usize,
    used: u64,
}

pub(super) fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    let (vocab, mut sources) = open_sources(&job)?;
    let raw = sources.iter().try_fold(0usize, |total, source| {
        total
            .checked_add(source.data.tokens())
            .ok_or("corpus size overflow")
    })?;
    let block = usize::try_from(CONTEXT)? * 2;
    let quota = raw / block * block;
    if quota == 0 {
        return Err("corpus interleave contains no complete sequence pair".into());
    }
    let mut output = BufWriter::new(File::create(&job.output)?);
    output.write_all(b"BAUXPTN1")?;
    for header in [vocab, CONTEXT, u32::try_from(quota / CONTEXT as usize)?, 0] {
        output.write_all(&header.to_le_bytes())?;
    }
    write_units(&mut output, &mut sources, vocab, quota, job.seed)?;
    output.flush()?;
    write_report(&job, &sources, quota)?;
    eprintln!(
        "corpus | train: {quota} tokens from {} interleaved sources",
        sources.len()
    );
    Ok(())
}

fn open_sources(job: &Job) -> Result<(u32, Vec<Source>)> {
    if job.parts.len() < 2 {
        return Err("corpus interleave requires at least two sources".into());
    }
    let mut vocabulary = None;
    let mut sources = Vec::with_capacity(job.parts.len());
    for part in &job.parts {
        let data = PretrainDataset::load_raw(&part.path)?;
        let vocab = data.vocabulary();
        if vocabulary.is_some_and(|expected| expected != vocab) {
            return Err("corpus interleave sources use different vocabularies".into());
        }
        vocabulary = Some(vocab);
        let slice = data.prefix(data.tokens())?;
        let spans = units(slice, job.bos, job.eos, part.marker)?;
        sources.push(Source {
            name: part.name.clone(),
            data,
            units: spans,
            next: 0,
            used: 0,
        });
    }
    let vocab = vocabulary.ok_or("corpus interleave has no vocabulary")?;
    Ok((vocab, sources))
}

fn write_units(
    output: &mut BufWriter<File>,
    sources: &mut [Source],
    vocab: u32,
    quota: usize,
    seed: u64,
) -> Result<()> {
    let mut written = 0usize;
    while written < quota {
        let index = choose(sources, seed).ok_or("corpus interleave exhausted early")?;
        let source = &mut sources[index];
        let range = source.units[source.next].clone();
        let count = (quota - written).min(range.len());
        let slice = source.data.prefix(source.data.tokens())?;
        write_span(output, &slice[range.start..range.start + count], vocab)?;
        source.next += 1;
        source.used += u64::try_from(count)?;
        written += count;
    }
    Ok(())
}

fn write_span(output: &mut BufWriter<File>, slice: &[u32], vocab: u32) -> Result<()> {
    if vocab > 65_536 && cfg!(target_endian = "little") {
        let bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * 4) };
        output.write_all(bytes)?;
        return Ok(());
    }
    for &token in slice {
        if vocab <= 65_536 {
            output.write_all(&u16::try_from(token)?.to_le_bytes())?;
        } else {
            output.write_all(&token.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_report(job: &Job, sources: &[Source], quota: usize) -> Result<()> {
    let parts = sources
        .iter()
        .map(|source| {
            json!({
                "name":source.name,
                "tokens":source.used,
                "units":source.next,
            })
        })
        .collect::<Vec<_>>();
    fs::write(
        &job.report,
        baux::artifact::json::pretty_string(&json!({
            "construction":"unit-interleave",
            "seed":job.seed,
            "tokens":quota,
            "sequences":quota / CONTEXT as usize,
            "sha256":super::tokenizer::digest(&job.output)?,
            "sources":parts,
        }))?,
    )?;
    Ok(())
}

fn units(tokens: &[u32], bos: u32, eos: u32, marker: Option<u32>) -> Result<Vec<Range<usize>>> {
    let starts = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| {
            (*token == bos
                && (index == 0 || tokens[index - 1] == eos)
                && marker.is_none_or(|marker| tokens.get(index + 1) == Some(&marker)))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    if starts.first() != Some(&0) {
        return Err("corpus does not begin at a declared unit boundary".into());
    }
    Ok(starts
        .iter()
        .copied()
        .zip(starts.iter().copied().skip(1).chain([tokens.len()]))
        .map(|(start, end)| start..end)
        .collect())
}

fn choose(sources: &[Source], seed: u64) -> Option<usize> {
    sources
        .iter()
        .enumerate()
        .filter(|(_, source)| source.next < source.units.len())
        .min_by(|(left_index, left), (right_index, right)| {
            let progress = (u128::from(left.used) * right.data.tokens() as u128)
                .cmp(&(u128::from(right.used) * left.data.tokens() as u128));
            if progress != Ordering::Equal {
                return progress;
            }
            tie(seed, *left_index, left.next).cmp(&tie(seed, *right_index, right.next))
        })
        .map(|(index, _)| index)
}

fn tie(seed: u64, source: usize, unit: usize) -> u64 {
    let mut value = seed
        ^ (source as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (unit as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(path: &PathBuf, vocab: u32, fill: u32) {
        let mut out = BufWriter::new(File::create(path).unwrap());
        out.write_all(b"BAUXPTN1").unwrap();
        for header in [vocab, CONTEXT, 2, 0] {
            out.write_all(&header.to_le_bytes()).unwrap();
        }
        let mut tokens = vec![fill; CONTEXT as usize * 2];
        tokens[0] = 2;
        tokens[CONTEXT as usize - 1] = 1;
        tokens[CONTEXT as usize] = 2;
        tokens[CONTEXT as usize * 2 - 1] = 1;
        write_span(&mut out, &tokens, vocab).unwrap();
        out.flush().unwrap();
    }

    #[test]
    fn preserves_units() {
        assert_eq!(
            units(&[2, 7, 1, 2, 8, 1, 2, 9], 2, 1, None).unwrap(),
            [0..3, 3..6, 6..8]
        );
        assert_eq!(
            units(&[2, 6, 7, 1, 2, 6, 8], 2, 1, Some(6)).unwrap(),
            [0..4, 4..7]
        );
    }

    #[test]
    fn interleaves_sources() {
        let dir = std::env::temp_dir().join(format!("baux_mix_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for vocab in [8192u32, 100_000u32] {
            let left = dir.join("left.bauxptn");
            let right = dir.join("right.bauxptn");
            let output = dir.join("out.bauxptn");
            let report = dir.join("report.json");
            write_fixture(&left, vocab, 10);
            write_fixture(&right, vocab, 20);
            prepare(json!({
                "output": output.to_string_lossy(),
                "report": report.to_string_lossy(),
                "bos": 2,
                "eos": 1,
                "seed": 42,
                "parts": [
                    {"name": "left", "path": left.to_string_lossy()},
                    {"name": "right", "path": right.to_string_lossy()}
                ]
            }))
            .unwrap();
            let merged = PretrainDataset::load(&output).unwrap();
            assert_eq!(merged.vocabulary(), vocab);
            assert_eq!(merged.tokens(), CONTEXT as usize * 4);
            let text = fs::read_to_string(&report).unwrap();
            let meta: Value = baux::artifact::json::from_str(&text).unwrap();
            assert_eq!(
                meta.get("construction").and_then(|v| v.kind().as_str()),
                Some("unit-interleave")
            );
            assert!(text.contains("16384"));
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
