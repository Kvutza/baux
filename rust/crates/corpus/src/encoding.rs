//! Text encoding and its persisted identity. Byte corpora have no vocabulary
//! file.

use super::reader::Result;
use baux::artifact::json::{Value, json};
use baux::semantics::hnet::{BOS_BYTE, BYTE_VOCAB, EOS_BYTE};
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

pub(super) enum Encoding {
    Bytes,
    Bpe(Tokenizer),
}

impl Encoding {
    pub(super) fn new(path: Option<&Path>, recipe: &Value) -> Result<Self> {
        match recipe["tokenizer"]["kind"].as_str() {
            Some("utf8_bytes") => {
                if path.is_some()
                    || recipe["tokenizer"]["vocabulary"].as_u64() != Some(BYTE_VOCAB as u64)
                    || recipe["tokenizer"]["bos"].as_u64() != Some(u64::from(BOS_BYTE))
                    || recipe["tokenizer"]["eos"].as_u64() != Some(u64::from(EOS_BYTE))
                {
                    return Err("invalid UTF-8 byte encoding contract".into());
                }
                Ok(Self::Bytes)
            }
            Some("gemma4_bpe") => Ok(Self::Bpe(super::tokenizer::from_recipe(
                path.ok_or("BPE encoding requires a tokenizer file")?,
                recipe,
            )?)),
            _ => Err("unsupported text encoding".into()),
        }
    }

    pub(super) fn vocabulary(&self) -> u32 {
        match self {
            Self::Bytes => BYTE_VOCAB as u32,
            Self::Bpe(_) => 262_144,
        }
    }

    pub(super) fn boundaries(&self) -> Result<(u32, u32)> {
        match self {
            Self::Bytes => Ok((u32::from(BOS_BYTE), u32::from(EOS_BYTE))),
            Self::Bpe(tokenizer) => super::tokenizer::boundaries(tokenizer),
        }
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Bytes => Ok(text
                .as_bytes()
                .iter()
                .map(|&byte| u32::from(byte))
                .collect()),
            Self::Bpe(tokenizer) => Ok(tokenizer.encode(text, false)?.get_ids().to_vec()),
        }
    }

    pub(super) fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match self {
            Self::Bytes => texts.iter().map(|text| self.encode(text)).collect(),
            Self::Bpe(tokenizer) => Ok(tokenizer
                .encode_batch(texts.to_vec(), false)?
                .into_iter()
                .map(|encoded| encoded.get_ids().to_vec())
                .collect()),
        }
    }

    pub(super) fn accepts(&self, text: &str) -> bool {
        matches!(self, Self::Bytes)
            || !super::tokenizer::MARKERS
                .iter()
                .any(|marker| text.contains(marker))
    }

    pub(super) fn persist(&self, directory: &Path, recipe: &Value) -> Result<Value> {
        let Self::Bpe(tokenizer) = self else {
            return Ok(json!({
                "kind":"utf8_bytes", "vocabulary":BYTE_VOCAB,
                "special_tokens":{"<bos>":BOS_BYTE,"<eos>":EOS_BYTE},
                "base":recipe["tokenizer"],
            }));
        };
        let path = directory.join("tokenizer.json");
        tokenizer.save(&path, true)?;
        let special: BTreeMap<_, _> = super::tokenizer::MARKERS
            .iter()
            .copied()
            .chain(["<bos>", "<eos>"])
            .map(|marker| (marker, tokenizer.token_to_id(marker)))
            .collect();
        Ok(json!({
            "sha256":super::tokenizer::digest(&path)?,
            "vocabulary":self.vocabulary(),"base":recipe["tokenizer"],
            "special_tokens":special,
        }))
    }
}
