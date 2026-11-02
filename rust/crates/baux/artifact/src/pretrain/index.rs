use std::fs::File;
use std::path::Path;

pub(super) const MAX_TOKENS: u32 = 1_048_576;

#[derive(Debug)]
pub(super) struct TraceIndex {
    episodes: Vec<(usize, usize)>,
    repository_starts: Vec<usize>,
    actions: Vec<TraceAction>,
    pad: u32,
}

#[derive(Debug, Clone, Copy)]
struct TraceAction {
    episode: usize,
    start: usize,
    end: usize,
}

#[derive(Debug)]
pub(super) struct CodeIndex {
    pub(super) actions: Vec<CodeAction>,
    pub(super) pad: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CodeAction {
    pub(super) repository_start: usize,
    pub(super) start: usize,
    pub(super) end: usize,
}

impl TraceIndex {
    pub(super) fn task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let eligible = self
            .actions
            .iter()
            .filter(|action| {
                let episode_start = self.episodes[action.episode].0;
                let output = action.end - action.start;
                let prompt = (action.start - episode_start).next_multiple_of(64);
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
        let draw = splitmix64(seed ^ u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let action = eligible[draw as usize % eligible.len()];
        let mut prompt_start = self.episodes[action.episode].0;
        for &(start, end) in self.episodes[self.repository_starts[action.episode]..=action.episode]
            .iter()
            .rev()
        {
            let prompt = (action.start - start).next_multiple_of(64);
            if prompt <= prompt_cap
                && prompt + action.end - action.start <= context
                && (end <= action.start || start == self.episodes[action.episode].0)
            {
                prompt_start = start;
            }
        }
        let source = &tokens[prompt_start..action.start];
        let padding = source.len().next_multiple_of(64) - source.len();
        let mut prompt = Vec::with_capacity(source.len() + padding);
        prompt.resize(padding, self.pad);
        prompt.extend_from_slice(source);
        Ok(Some((
            prompt,
            tokens[action.start..action.end].to_vec(),
            action.start,
        )))
    }

    pub(super) fn load(path: &Path, tokens: &[u32]) -> Result<Option<Self>, String> {
        let manifest = path
            .parent()
            .ok_or("pretraining dataset has no parent directory")?
            .join("manifest.json");
        if !manifest.is_file() {
            return Ok(None);
        }
        let document: baux::artifact::json::Value = baux::artifact::json::from_reader(
            File::open(&manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid corpus manifest {}: {error}", manifest.display()))?;
        if document["recipe"]["serialization"].as_str() != Some("role_tool_channel_v4") {
            return Ok(None);
        }
        let marker = |name: &str| -> Result<u32, String> {
            document["tokenizer"]["special_tokens"][name]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("trace corpus manifest has no {name} token"))
        };
        let episode = marker("<|episode|>")?;
        let repository = marker("<|repository|>")?;
        let bos = marker("<bos>")?;
        let turn = marker("<|turn|>")?;
        let assistant = marker("<|assistant|>")?;
        let endturn = marker("<|endturn|>")?;
        let pad = marker("<|pad|>")?;
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
        let mut repository_starts = Vec::with_capacity(episodes.len());
        let mut repository_start = 0;
        for (index, &(start, _)) in episodes.iter().enumerate() {
            let episode = start + usize::from(tokens[start] == bos);
            if tokens.get(episode + 1) == Some(&repository) {
                repository_start = index;
            }
            repository_starts.push(repository_start);
        }
        let mut actions = Vec::new();
        for (episode_index, &(start, end)) in episodes.iter().enumerate() {
            let mut cursor = start;
            while cursor + 1 < end {
                if tokens[cursor] == turn && tokens[cursor + 1] == assistant {
                    if let Some(offset) = tokens[cursor + 2..end]
                        .iter()
                        .position(|token| *token == endturn)
                    {
                        actions.push(TraceAction {
                            episode: episode_index,
                            start: cursor,
                            end: cursor + 2 + offset + 1,
                        });
                    }
                }
                cursor += 1;
            }
        }
        if episodes.is_empty() || actions.is_empty() {
            return Err("trace corpus contains no complete assistant actions".into());
        }
        Ok(Some(Self {
            episodes,
            repository_starts,
            actions,
            pad,
        }))
    }
}

impl CodeIndex {
    pub(super) fn load(path: &Path, tokens: &[u32]) -> Result<Option<Self>, String> {
        let manifest = path
            .parent()
            .ok_or("pretraining dataset has no parent directory")?
            .join("manifest.json");
        if !manifest.is_file() {
            return Ok(None);
        }
        let document: baux::artifact::json::Value = baux::artifact::json::from_reader(
            File::open(&manifest).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("invalid corpus manifest {}: {error}", manifest.display()))?;
        if document["recipe"]["serialization"].as_str() != Some("repository_file_v1") {
            return Ok(None);
        }
        let marker = |name: &str| -> Result<u32, String> {
            document["tokenizer"]["special_tokens"][name]
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("code corpus manifest has no {name} token"))
        };
        let repository = marker("<|repository|>")?;
        let bos = marker("<bos>")?;
        let file = marker("<|file|>")?;
        let content = marker("<|content|>")?;
        let endfile = marker("<|endfile|>")?;
        let pad = marker("<|pad|>")?;
        let mut repository_start = None;
        let mut actions = Vec::new();
        let mut cursor = 0;
        while cursor < tokens.len() {
            if tokens[cursor] == repository {
                repository_start = Some(
                    cursor.saturating_sub(usize::from(cursor > 0 && tokens[cursor - 1] == bos)),
                );
            } else if tokens[cursor] == file {
                let Some(start_offset) = tokens[cursor + 1..]
                    .iter()
                    .position(|token| *token == content)
                else {
                    break;
                };
                let start = cursor + 1 + start_offset + 1;
                let Some(end_offset) = tokens[start..].iter().position(|token| *token == endfile)
                else {
                    break;
                };
                let end = start + end_offset + 1;
                let repository_start = repository_start
                    .ok_or("code corpus file appeared before its repository marker")?;
                actions.push(CodeAction {
                    repository_start,
                    start,
                    end,
                });
                cursor = end;
                continue;
            }
            cursor += 1;
        }
        if actions.is_empty() {
            return Err("code corpus contains no complete files".into());
        }
        Ok(Some(Self { actions, pad }))
    }

    pub(super) fn task(
        &self,
        tokens: &[u32],
        prompt_cap: usize,
        output_cap: usize,
        seed: u64,
        round: u32,
    ) -> Result<Option<(Vec<u32>, Vec<u32>, usize)>, String> {
        let context = MAX_TOKENS as usize;
        let budget = context.min(prompt_cap.saturating_add(output_cap));
        let eligible = self
            .actions
            .iter()
            .filter(|action| {
                let prompt = (action.start - action.repository_start).next_multiple_of(64);
                prompt <= prompt_cap
                    && prompt < budget
                    && output_cap.min(budget - prompt) >= 32
                    && action.end > action.start
            })
            .collect::<Vec<_>>();
        if eligible.is_empty() {
            return Err(format!(
                "code corpus has no file boundary within the {budget}-token context budget"
            ));
        }
        let draw = splitmix64(seed ^ u64::from(round).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let action = eligible[draw as usize % eligible.len()];
        let source = &tokens[action.repository_start..action.start];
        let padding = source.len().next_multiple_of(64) - source.len();
        let mut prompt = Vec::with_capacity(source.len() + padding);
        prompt.resize(padding, self.pad);
        prompt.extend_from_slice(source);
        let output = output_cap.min(budget - prompt.len());
        Ok(Some((
            prompt,
            circular_tokens(tokens, action.start, output)?,
            action.start,
        )))
    }
}

fn circular_tokens(tokens: &[u32], start: usize, count: usize) -> Result<Vec<u32>, String> {
    if tokens.is_empty() || count > tokens.len() {
        return Err(format!(
            "packed continuation needs {count} tokens, corpus contains {}",
            tokens.len()
        ));
    }
    let start = start % tokens.len();
    let first = count.min(tokens.len() - start);
    let mut output = Vec::with_capacity(count);
    output.extend_from_slice(&tokens[start..start + first]);
    output.extend_from_slice(&tokens[..count - first]);
    Ok(output)
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
