//! Execution mode resolution and hardware capability validation for Julia
//! experiments.

use std::path::Path;

use crate::experiment::julia::config::{Backend, Config, ExecMode, is_ocean, obj_count};
use baux::AnyBackend;
use baux::config::OptimizerSpec;

fn check_manifold(cfg: &Config) -> Result<(), String> {
    let Some(ref mf) = cfg.manifold else {
        return Ok(());
    };
    if mf.stiefel_rank == Some(0) {
        return Err("manifold.stiefel-rank must be positive".into());
    }
    if let Some(alpha) = mf.calibrate_alpha
        && (!alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0)
    {
        return Err("manifold.calibrate-alpha must lie in (0, 1)".into());
    }
    Ok(())
}

fn check_morbo(cfg: &Config) -> Result<(), String> {
    let Some(ref m) = cfg.morbo else {
        return Ok(());
    };
    let n_obj = obj_count(cfg);
    if m.regions == 0 {
        return Err("morbo.regions must be positive".into());
    }
    if !m.alpha.is_finite() || m.alpha < 0.0 {
        return Err("morbo.alpha must be finite and nonnegative".into());
    }
    if !m.preferences.is_empty()
        && (m.preferences.len() != n_obj
            || m.preferences.iter().any(|w| !w.is_finite() || *w <= 0.0))
    {
        return Err(
            "morbo.preferences must supply one positive finite weight per objective".into(),
        );
    }
    if !m.scales.is_empty()
        && (m.scales.len() != n_obj || m.scales.iter().any(|s| !s.is_finite() || *s <= 0.0))
    {
        return Err("morbo.scales must supply one positive finite scale per objective".into());
    }
    Ok(())
}

pub fn metal_ok(cfg: &Config) -> Result<(), String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("resident-metal backend requires macOS with Metal feature".into());
    }
    check_manifold(cfg)?;
    check_morbo(cfg)?;
    baux::UnifiedEngine::check_metal(&cfg.optimizer).map_err(|e| e.to_string())
}

pub fn mode_backend(mode: ExecMode, out_dir: Option<&Path>, rep: u32) -> AnyBackend {
    match mode {
        ExecMode::ResidentMetal => AnyBackend::Metal,
        ExecMode::HostDisk => {
            let base = out_dir.unwrap_or_else(|| Path::new("."));
            AnyBackend::disk(base.join(format!("bpann_rep{rep}")))
        }
        ExecMode::HostRam | ExecMode::MetalPpo => AnyBackend::Cpu,
    }
}

fn special_mode(cfg: &Config) -> Result<Option<ExecMode>, String> {
    match &cfg.optimizer {
        OptimizerSpec::Muon => Err(
            "optimizer = 'muon' requires analytical parameter gradients, which black-box Oceananigans.jl / FUSE.jl workers do not expose".into(),
        ),
        OptimizerSpec::Adamw => Err(
            "optimizer = 'adamw' requires analytical parameter gradients, which black-box Oceananigans.jl / FUSE.jl workers do not expose".into(),
        ),
        OptimizerSpec::Ppo => match cfg.backend {
            Backend::Auto | Backend::ResidentMetal => Ok(Some(ExecMode::MetalPpo)),
            Backend::HostRam | Backend::HostDisk => {
                Err("optimizer = 'ppo' runs on Candle/Metal GPU; use backend = 'auto'".into())
            }
        },
        _ => Ok(None),
    }
}

pub fn resolve_mode(cfg: &Config) -> Result<ExecMode, String> {
    if let Some(mode) = special_mode(cfg)? {
        return Ok(mode);
    }
    if is_ocean(cfg) && cfg.ocean.disk && cfg.backend == Backend::Auto {
        return Ok(ExecMode::HostDisk);
    }
    match cfg.backend {
        Backend::ResidentMetal => {
            metal_ok(cfg)?;
            Ok(ExecMode::ResidentMetal)
        }
        Backend::HostRam => Ok(ExecMode::HostRam),
        Backend::HostDisk => Ok(ExecMode::HostDisk),
        Backend::Auto => {
            if metal_ok(cfg).is_ok() {
                Ok(ExecMode::ResidentMetal)
            } else {
                Ok(ExecMode::HostRam)
            }
        }
    }
}
