use std::fs::File;
use std::path::Path;

const SPAN_MAGIC: &[u8; 8] = b"BAUXSPN1";
const CACHE_HEADER: usize = 40;
pub(super) const MAX_TOKENS: u32 = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SpanIndex {
    pub(super) episodes: Vec<(usize, usize)>,
    pub(super) origins: Vec<usize>,
    pub(super) spans: Vec<Span>,
    pub(super) pad: u32,
    pub(super) bounded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Span {
    pub(super) group: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

impl SpanIndex {
    pub(super) fn load(
        path: &Path,
        tokens: &[u32],
        key: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let manifest = path
            .parent()
            .ok_or("pretraining dataset has no parent directory")?
            .join("manifest.json");
        if !manifest.is_file() {
            return Ok(None);
        }
        let cache = key.map(|hash| Path::new(".cache/baux/digests").join(format!("{hash}.spans")));
        if let Some(cached) = cache.as_deref().and_then(read_cache) {
            return Ok(Some(cached));
        }
        let doc: baux::artifact::json::Value = baux::artifact::json::from_reader(
            File::open(&manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid corpus manifest {}: {error}", manifest.display()))?;
        let built = match doc["recipe"]["serialization"].as_str() {
            Some("role_tool_channel_v4") => Some(load_trace(&doc, tokens)?),
            Some("repository_file_v1") => Some(load_code(&doc, tokens)?),
            _ => None,
        };
        if let (Some(file), Some(index)) = (cache.as_deref(), built.as_ref()) {
            write_cache(file, index);
        }
        Ok(built)
    }

    pub(super) fn task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        if self.spans.is_empty() {
            return Ok(None);
        }
        if self.bounded {
            self.trace_task(tokens, prompt_cap, output_cap, (seed, round))
        } else {
            self.code_task(tokens, prompt_cap, output_cap, (seed, round))
        }
    }

    fn trace_task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        cursor: (u64, u32),
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let eligible = self
            .spans
            .iter()
            .filter(|span| {
                let episode_start = self.episodes[span.group].0;
                let output = span.end - span.start;
                let prompt = (span.start - episode_start).next_multiple_of(64);
                output >= 32
                    && output <= output_cap
                    && prompt <= prompt_cap
                    && prompt + output <= context
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "trace corpus has no complete assistant action within prompt cap {prompt_cap}, \
                 output cap {output_cap}, and context {context}"
            ));
        }
        let span = eligible[permute_index(eligible.len() as u64, cursor.0, cursor.1)];
        let mut prompt_start = self.episodes[span.group].0;
        for &(start, end) in self.episodes[self.origins[span.group]..=span.group]
            .iter()
            .rev()
        {
            let prompt = (span.start - start).next_multiple_of(64);
            if prompt <= prompt_cap
                && prompt + span.end - span.start <= context
                && (end <= span.start || start == self.episodes[span.group].0)
            {
                prompt_start = start;
            }
        }
        let prompt = pad_prompt(&tokens[prompt_start..span.start], self.pad);
        Ok(Some((
            prompt,
            tokens[span.start..span.end].to_vec(),
            span.start,
        )))
    }

    fn code_task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        cursor: (u64, u32),
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let budget = context.min(prompt_cap.saturating_add(output_cap));
        let eligible = self
            .spans
            .iter()
            .filter(|span| {
                let prompt = (span.start - span.group).next_multiple_of(64);
                let output = output_cap.min(budget.saturating_sub(prompt));
                prompt <= prompt_cap
                    && prompt < budget
                    && output >= 32
                    && span.end > span.start
                    && span.start.saturating_add(output) <= tokens.len()
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "code corpus has no file boundary within the {budget}-token context budget"
            ));
        }
        let span = eligible[permute_index(eligible.len() as u64, cursor.0, cursor.1)];
        let prompt = pad_prompt(&tokens[span.group..span.start], self.pad);
        let output = output_cap.min(budget - prompt.len());
        Ok(Some((
            prompt,
            tokens[span.start..span.start + output].to_vec(),
            span.start,
        )))
    }
}

fn marker(doc: &baux::artifact::json::Value, kind: &str, name: &str) -> Result<u32, String> {
    doc["tokenizer"]["special_tokens"][name]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("{kind} corpus manifest has no {name} token"))
}

fn load_trace(doc: &baux::artifact::json::Value, tokens: &[u32]) -> Result<SpanIndex, String> {
    let episode = marker(doc, "trace", "<|episode|>")?;
    let repository = marker(doc, "trace", "<|repository|>")?;
    let bos = marker(doc, "trace", "<bos>")?;
    let turn = marker(doc, "trace", "<|turn|>")?;
    let assistant = marker(doc, "trace", "<|assistant|>")?;
    let endturn = marker(doc, "trace", "<|endturn|>")?;
    let pad = marker(doc, "trace", "<|pad|>")?;
    let markers = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| (*token == episode).then_some(index))
        .collect::<Vec<_>>();
    // The flat corpus quota may truncate its final episode. Consecutive
    // episode markers prove that every span retained here is complete.
    let episodes = markers
        .windows(2)
        .map(|pair| {
            let start =
                pair[0].saturating_sub(usize::from(pair[0] > 0 && tokens[pair[0] - 1] == bos));
            let end =
                pair[1].saturating_sub(usize::from(pair[1] > 0 && tokens[pair[1] - 1] == bos));
            (start, end)
        })
        .collect::<Vec<_>>();
    let origins = trace_origins(tokens, &episodes, bos, repository);
    let spans = trace_spans(tokens, &episodes, (turn, assistant, endturn));
    if episodes.is_empty() || spans.is_empty() {
        return Err("trace corpus contains no complete assistant actions".into());
    }
    Ok(SpanIndex {
        episodes,
        origins,
        spans,
        pad,
        bounded: true,
    })
}

fn trace_origins(
    tokens: &[u32],
    episodes: &[(usize, usize)],
    bos: u32,
    repository: u32,
) -> Vec<usize> {
    let mut origins = Vec::with_capacity(episodes.len());
    let mut origin = 0;
    for (index, &(start, _)) in episodes.iter().enumerate() {
        let head = start + usize::from(tokens[start] == bos);
        if tokens.get(head + 1) == Some(&repository) {
            origin = index;
        }
        origins.push(origin);
    }
    origins
}

fn trace_spans(tokens: &[u32], episodes: &[(usize, usize)], markers: (u32, u32, u32)) -> Vec<Span> {
    let (turn, assistant, endturn) = markers;
    let mut spans = Vec::new();
    for (group, &(start, end)) in episodes.iter().enumerate() {
        for cursor in start..end.saturating_sub(1) {
            if tokens[cursor] != turn || tokens[cursor + 1] != assistant {
                continue;
            }
            if let Some(offset) = tokens[cursor + 2..end].iter().position(|t| *t == endturn) {
                spans.push(Span {
                    group,
                    start: cursor,
                    end: cursor + 2 + offset + 1,
                });
            }
        }
    }
    spans
}

fn load_code(doc: &baux::artifact::json::Value, tokens: &[u32]) -> Result<SpanIndex, String> {
    let repository = marker(doc, "code", "<|repository|>")?;
    let bos = marker(doc, "code", "<bos>")?;
    let file = marker(doc, "code", "<|file|>")?;
    let content = marker(doc, "code", "<|content|>")?;
    let endfile = marker(doc, "code", "<|endfile|>")?;
    let pad = marker(doc, "code", "<|pad|>")?;
    let mut origin = None;
    let mut spans = Vec::new();
    let mut cursor = 0;
    while cursor < tokens.len() {
        if tokens[cursor] == repository {
            origin =
                Some(cursor.saturating_sub(usize::from(cursor > 0 && tokens[cursor - 1] == bos)));
        } else if tokens[cursor] == file {
            let Some(span) = file_span(tokens, cursor, origin, (content, endfile))? else {
                break;
            };
            cursor = span.end;
            spans.push(span);
            continue;
        }
        cursor += 1;
    }
    if spans.is_empty() {
        return Err("code corpus contains no complete files".into());
    }
    Ok(SpanIndex {
        episodes: Vec::new(),
        origins: Vec::new(),
        spans,
        pad,
        bounded: false,
    })
}

fn file_span(
    tokens: &[u32],
    cursor: usize,
    origin: Option<usize>,
    markers: (u32, u32),
) -> Result<Option<Span>, String> {
    let (content, endfile) = markers;
    let Some(start_offset) = tokens[cursor + 1..].iter().position(|t| *t == content) else {
        return Ok(None);
    };
    let start = cursor + 1 + start_offset + 1;
    let Some(end_offset) = tokens[start..].iter().position(|t| *t == endfile) else {
        return Ok(None);
    };
    let group = origin.ok_or("code corpus file appeared before its repository marker")?;
    Ok(Some(Span {
        group,
        start,
        end: start + end_offset + 1,
    }))
}

fn read_cache(path: &Path) -> Option<SpanIndex> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < CACHE_HEADER || &bytes[..8] != SPAN_MAGIC {
        return None;
    }
    let bounded = bytes[8] != 0;
    let pad = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
    let episodes_len = usize::try_from(u64::from_le_bytes(bytes[16..24].try_into().ok()?)).ok()?;
    let origins_len = usize::try_from(u64::from_le_bytes(bytes[24..32].try_into().ok()?)).ok()?;
    let spans_len = usize::try_from(u64::from_le_bytes(bytes[32..40].try_into().ok()?)).ok()?;
    let expected = CACHE_HEADER
        .checked_add(episodes_len.checked_mul(16)?)?
        .checked_add(origins_len.checked_mul(8)?)?
        .checked_add(spans_len.checked_mul(24)?)?;
    if bytes.len() != expected {
        return None;
    }
    let mut words = bytes[CACHE_HEADER..]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as usize);
    let episodes = (0..episodes_len)
        .map(|_| (words.next().unwrap(), words.next().unwrap()))
        .collect();
    let origins = (0..origins_len).map(|_| words.next().unwrap()).collect();
    let spans = (0..spans_len)
        .map(|_| Span {
            group: words.next().unwrap(),
            start: words.next().unwrap(),
            end: words.next().unwrap(),
        })
        .collect();
    Some(SpanIndex {
        episodes,
        origins,
        spans,
        pad,
        bounded,
    })
}

fn write_cache(path: &Path, index: &SpanIndex) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let size =
        CACHE_HEADER + index.episodes.len() * 16 + index.origins.len() * 8 + index.spans.len() * 24;
    let mut bytes = Vec::with_capacity(size);
    bytes.extend_from_slice(SPAN_MAGIC);
    bytes.extend_from_slice(&[u8::from(index.bounded), 0, 0, 0]);
    bytes.extend_from_slice(&index.pad.to_le_bytes());
    bytes.extend_from_slice(&(index.episodes.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(index.origins.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(index.spans.len() as u64).to_le_bytes());
    for &(start, end) in &index.episodes {
        bytes.extend_from_slice(&(start as u64).to_le_bytes());
        bytes.extend_from_slice(&(end as u64).to_le_bytes());
    }
    for &origin in &index.origins {
        bytes.extend_from_slice(&(origin as u64).to_le_bytes());
    }
    for span in &index.spans {
        bytes.extend_from_slice(&(span.group as u64).to_le_bytes());
        bytes.extend_from_slice(&(span.start as u64).to_le_bytes());
        bytes.extend_from_slice(&(span.end as u64).to_le_bytes());
    }
    let temp = path.with_extension(format!("spans.tmp.{}", std::process::id()));
    if std::fs::write(&temp, &bytes).is_ok() {
        let _ = std::fs::rename(&temp, path);
    }
}

fn pad_prompt(source: &[u32], pad: u32) -> Vec<u32> {
    let padding = source.len().next_multiple_of(64) - source.len();
    let mut prompt = Vec::with_capacity(source.len() + padding);
    prompt.resize(padding, pad);
    prompt.extend_from_slice(source);
    prompt
}

pub(super) fn permute_index(count: u64, seed: u64, round: u32) -> usize {
    if count <= 1 {
        return 0;
    }
    let epoch = u64::from(round) / count;
    let step = u64::from(round) % count;
    let epoch_seed = splitmix64(seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let start = epoch_seed % count;
    let mut stride = (splitmix64(epoch_seed ^ 0x6361_7573_616c_2d73) % count).max(1);
    while gcd(stride, count) != 1 {
        stride = ((stride + 1) % count).max(1);
    }
    ((start + step.wrapping_mul(stride) % count) % count) as usize
}

pub(super) fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

pub(super) fn splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}
