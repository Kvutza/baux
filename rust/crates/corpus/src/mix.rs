//! Deterministic interleaving of complete tokenized corpus units.
use super::Result;
use baux::artifact::json::{Value, json};
use deser::Deserialize;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::PathBuf;

const HEADER: usize = 24;
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
    tokens: Vec<u32>,
    units: Vec<Range<usize>>,
    next: usize,
    used: u64,
}

pub(super) fn prepare(value: Value) -> Result<()> {
    let job: Job = baux::artifact::json::from_str(&baux::artifact::json::to_string(&value)?)?;
    if job.parts.len() < 2 {
        return Err("corpus interleave requires at least two sources".into());
    }
    let mut vocabulary = None;
    let mut sources = Vec::with_capacity(job.parts.len());
    for part in job.parts {
        let (vocab, tokens) = load(&part.path)?;
        if vocabulary.is_some_and(|expected| expected != vocab) {
            return Err("corpus interleave sources use different vocabularies".into());
        }
        vocabulary = Some(vocab);
        let units = units(&tokens, job.bos, job.eos, part.marker)?;
        sources.push(Source {
            name: part.name,
            tokens,
            units,
            next: 0,
            used: 0,
        });
    }
    let raw = sources.iter().try_fold(0usize, |total, source| {
        total
            .checked_add(source.tokens.len())
            .ok_or("corpus size overflow")
    })?;
    let block = usize::try_from(CONTEXT)? * 2;
    let quota = raw / block * block;
    if quota == 0 {
        return Err("corpus interleave contains no complete sequence pair".into());
    }
    let vocab = vocabulary.ok_or("corpus interleave has no vocabulary")?;
    let mut output = BufWriter::new(File::create(&job.output)?);
    output.write_all(b"BAUXPTN1")?;
    for value in [vocab, CONTEXT, u32::try_from(quota / CONTEXT as usize)?, 0] {
        output.write_all(&value.to_le_bytes())?;
    }
    let mut written = 0usize;
    while written < quota {
        let index = choose(&sources, job.seed).ok_or("corpus interleave exhausted early")?;
        let source = &mut sources[index];
        let range = source.units[source.next].clone();
        let count = (quota - written).min(range.len());
        for token in &source.tokens[range.start..range.start + count] {
            output.write_all(&token.to_le_bytes())?;
        }
        source.next += 1;
        source.used += u64::try_from(count)?;
        written += count;
    }
    output.flush()?;
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
    eprintln!(
        "corpus | train: {quota} tokens from {} interleaved sources",
        sources.len()
    );
    Ok(())
}

fn load(path: &PathBuf) -> Result<(u32, Vec<u32>)> {
    let bytes = fs::read(path)?;
    if bytes.len() < HEADER || &bytes[..8] != b"BAUXPTN1" {
        return Err(format!("invalid corpus artifact {}", path.display()).into());
    }
    let field = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let vocab = field(8);
    let context = field(12);
    let sequences = field(16);
    if context != CONTEXT || field(20) != 0 {
        return Err(format!("invalid corpus header {}", path.display()).into());
    }
    let count = usize::try_from(u64::from(sequences) * u64::from(context))?;
    let width = if vocab <= 65_536 { 2 } else { 4 };
    if bytes.len() != HEADER + count * width {
        return Err(format!("invalid corpus length {}", path.display()).into());
    }
    let tokens = if width == 2 {
        bytes[HEADER..]
            .chunks_exact(2)
            .map(|chunk| u32::from(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()
    } else {
        bytes[HEADER..]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    };
    Ok((vocab, tokens))
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
            let progress = (u128::from(left.used) * right.tokens.len() as u128)
                .cmp(&(u128::from(right.used) * left.tokens.len() as u128));
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
}
