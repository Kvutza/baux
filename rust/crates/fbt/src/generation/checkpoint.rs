use super::generation::*;

pub(super) fn verify_policy(config: &GenerationConfig) -> Result<(), String> {
    if config.purpose != crate::config::GenerationPurpose::CodingOptimization {
        return Ok(());
    }
    crate::BasePolicyManifest::load_verified(
        config
            .qualification_manifest
            .as_deref()
            .ok_or("missing base-policy qualification manifest")?,
        config
            .checkpoint
            .as_deref()
            .ok_or("missing qualified checkpoint")?,
    )?;
    Ok(())
}
