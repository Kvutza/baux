//! Corpus identity must agree with the model's byte or BPE interface.

use super::{Value, canonical, exact_fields};
use baux::config::{ExperimentSpec, ModelArchitecture};
use std::io::Read;
use std::path::Path;

pub(super) fn is_bytes(recipe: &Value) -> bool {
    recipe["tokenizer"]["kind"].as_str() == Some("utf8_bytes")
}

pub(super) fn check_model(spec: &ExperimentSpec, recipe: &Value) -> Result<(), String> {
    let hnet = spec.model == Some(ModelArchitecture::PisaHnetMhc4V1);
    if hnet != is_bytes(recipe) {
        return Err("H-Net requires a UTF-8 byte corpus; BPE corpora require a BPE model".into());
    }
    Ok(())
}

pub(super) fn validate(recipe: &Value) -> Result<(), String> {
    let value = &recipe["tokenizer"];
    if is_bytes(recipe) {
        exact_fields(
            value,
            &["kind", "vocabulary", "bos", "eos"],
            "byte encoding",
        )?;
        if value["vocabulary"].as_u64() != Some(256)
            || value["bos"].as_u64() != Some(254)
            || value["eos"].as_u64() != Some(255)
        {
            return Err("H-Net byte encoding requires vocabulary=256, bos=254, eos=255".into());
        }
    } else {
        exact_fields(
            value,
            &[
                "kind",
                "vocabulary",
                "package",
                "id",
                "revision",
                "sha256",
                "markers",
                "add_special_tokens",
            ],
            "recipe tokenizer",
        )?;
        if value["kind"].as_str() != Some("gemma4_bpe")
            || value["vocabulary"].as_u64() != Some(262_144)
            || value["add_special_tokens"].as_bool() != Some(false)
        {
            return Err("invalid BPE encoding contract".into());
        }
    }
    Ok(())
}

pub(super) fn text_boundaries(recipe: &Value) -> bool {
    if is_bytes(recipe) {
        recipe["boundaries"].as_str() == Some("hnet_bos_eos_per_document_v1")
    } else {
        recipe["boundaries"].as_str() == Some("gemma_bos_eos_per_document_v1")
            && recipe["tokenizer"]["markers"].as_str()
                == Some("unused0..14_to_baux_markers_ids6..20_v1")
    }
}

pub(super) fn verify(directory: &Path, recipe: &Value, manifest: &Value) -> Result<(), String> {
    if !is_bytes(recipe) {
        return super::verify(
            &directory.join("tokenizer.json"),
            manifest["tokenizer"]["sha256"].as_str(),
        );
    }
    let identity = &manifest["tokenizer"];
    if identity["kind"].as_str() != Some("utf8_bytes")
        || identity["vocabulary"].as_u64() != Some(256)
        || identity["special_tokens"]["<bos>"].as_u64() != Some(254)
        || identity["special_tokens"]["<eos>"].as_u64() != Some(255)
        || canonical(&identity["base"])? != canonical(&recipe["tokenizer"])?
        || identity.get("sha256").is_some()
        || directory.join("tokenizer.json").exists()
    {
        return Err("cached corpus does not have the specified byte encoding".into());
    }
    Ok(())
}

pub(super) fn check_header(path: &Path, recipe: &Value) -> Result<(), String> {
    let mut header = [0u8; 24];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| error.to_string())?;
    let vocabulary = u32::from_le_bytes(header[8..12].try_into().map_err(|_| "invalid header")?);
    if !matches!(&header[..8], b"BAUXPTN1" | b"BLCKPTN1")
        || Some(u64::from(vocabulary)) != recipe["tokenizer"]["vocabulary"].as_u64()
    {
        return Err("corpus header vocabulary does not match its encoding".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use baux::artifact::json::json;

    #[test]
    fn identity() {
        let recipe =
            json!({"tokenizer":{"kind":"utf8_bytes","vocabulary":256,"bos":254,"eos":255}});
        assert!(validate(&recipe).is_ok());
        let directory =
            std::env::temp_dir().join(format!("baux-byte-identity-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let manifest = json!({"tokenizer":{
            "kind":"utf8_bytes", "vocabulary":256,"base":recipe["tokenizer"],
            "special_tokens":{"<bos>":254,"<eos>":255},
        }});
        assert!(verify(&directory, &recipe, &manifest).is_ok());
        let invalid =
            json!({"tokenizer":{"kind":"utf8_bytes","vocabulary":262144,"bos":254,"eos":255}});
        assert!(validate(&invalid).is_err());
        let path = directory.join("train.bauxptn");
        let mut header = b"BAUXPTN1".to_vec();
        for word in [262144u32, 4096, 1, 0] {
            header.extend_from_slice(&word.to_le_bytes());
        }
        std::fs::write(&path, header).unwrap();
        assert!(check_header(&path, &recipe).is_err());
        std::fs::write(directory.join("tokenizer.json"), b"{}").unwrap();
        assert!(verify(&directory, &recipe, &manifest).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
