use super::Result;
use baux::artifact::json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tokenizers::Tokenizer;

/// One marker registry shared by every BAUX pretraining corpus.  A blend may
/// only combine corpora derived from the same tokenizer artifact, so individual
/// serializers must not assign the reserved Gemma IDs independently.
pub(super) const MARKERS: [&str; 15] = [
    "<|repository|>",
    "<|file|>",
    "<|content|>",
    "<|endfile|>",
    "<|endrepository|>",
    "<|episode|>",
    "<|turn|>",
    "<|endturn|>",
    "<|reasoning|>",
    "<|toolcalls|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|pad|>",
];

pub(super) fn derive(path: &Path, expected: &str, markers: &[&str]) -> Result<Tokenizer> {
    if digest(path)? != expected {
        return Err("base tokenizer checksum mismatch".into());
    }
    let mut document: Value = baux::artifact::json::from_reader(File::open(path)?)?;
    let vocab = document
        .get_mut("model")
        .and_then(|value| value.get_mut("vocab"))
        .and_then(|value| value.as_map_mut())
        .ok_or("tokenizer vocabulary missing")?;
    if vocab.len() != 262_144 {
        return Err("base tokenizer requires 262144 vocabulary entries".into());
    }
    for (index, marker) in markers.iter().enumerate() {
        let unused = format!("<unused{index}>");
        let id = vocab
            .remove(unused.as_str())
            .ok_or("reserved tokenizer entry missing")?;
        if id.as_u64() != Some(index as u64 + 6) || vocab.get(*marker).is_some() {
            return Err("reserved tokenizer mapping does not match the recipe".into());
        }
        vocab.insert(*marker, id);
    }
    let added = document
        .get_mut("added_tokens")
        .and_then(|value| value.as_seq_mut())
        .ok_or("tokenizer added_tokens missing")?;
    for (index, marker) in markers.iter().enumerate() {
        let id = index as u32 + 6;
        if added
            .iter()
            .any(|entry| entry["id"].as_u64() == Some(u64::from(id)))
        {
            return Err("reserved tokenizer ID is already registered".into());
        }
        added.push(json!({"id":id,"content":marker,"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":true}));
    }
    let tokenizer = Tokenizer::from_bytes(baux::artifact::json::to_string(&document)?.as_bytes())?;
    if tokenizer.get_vocab_size(true) != 262_144 {
        return Err("derived tokenizer changed the vocabulary size".into());
    }
    for (index, marker) in markers.iter().enumerate() {
        if tokenizer.token_to_id(marker) != Some(index as u32 + 6) {
            return Err("derived tokenizer changed a reserved marker ID".into());
        }
    }
    Ok(tokenizer)
}

pub(super) fn from_recipe(path: &Path, recipe: &Value) -> Result<Tokenizer> {
    let expected = recipe["tokenizer"]["sha256"]
        .as_str()
        .ok_or("tokenizer recipe requires a pinned checksum")?;
    derive(path, expected, &MARKERS)
}

pub(super) fn boundaries(tokenizer: &Tokenizer) -> Result<(u32, u32)> {
    let bos = tokenizer
        .token_to_id("<bos>")
        .ok_or("Gemma tokenizer has no BOS token")?;
    let eos = tokenizer
        .token_to_id("<eos>")
        .ok_or("Gemma tokenizer has no EOS token")?;
    Ok((bos, eos))
}

pub(super) fn digest(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
