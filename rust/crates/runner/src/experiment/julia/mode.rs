//! Execution mode resolution and hardware capability validation for Julia
//! experiments.

use crate::experiment::julia::config::{Backend, Config, ExecMode, is_ocean, obj_count};
use baux::config::OptimizerSpec;

fn check_morbo(cfg: &Config) -> Result<(), String> {
    let Some(ref m) = cfg.morbo else {
        return Ok(());
    };
    let n_obj = obj_count(cfg);
    if m.rescalarize != baux::Rescalarize::OnRestart {
        return Err("resident-metal MORBO requires morbo.rescalarize = 'on-restart'".into());
    }
    if !(2..=8).contains(&m.regions) {
        return Err("resident-metal MORBO requires morbo.regions in 2..=8".into());
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

fn check_optimizer(cfg: &Config) -> Result<(), String> {
    match &cfg.optimizer {
        OptimizerSpec::OperatorBo {
            perturbation: Some(_),
            ..
        } => Err("resident-metal does not accept host [optimizer.perturbation] overrides".into()),
        OptimizerSpec::OperatorBo { .. } => Ok(()),
        OptimizerSpec::BayesianSearch {
            candidates,
            arms,
            perturbation,
            ..
        } => {
            if perturbation.is_some() {
                return Err(
                    "resident-metal does not accept host [optimizer.perturbation] overrides".into(),
                );
            }
            if *arms == 0
                || *candidates == 0
                || *candidates > 4
                || !candidates.is_multiple_of(*arms)
            {
                return Err(
                    "resident-metal requires 1..=4 total candidates divisible by arms".into(),
                );
            }
            Ok(())
        }
        _ => Err("resident-metal ENN backend requires operator-bo or bayesian-search".into()),
    }
}

pub fn metal_ok(cfg: &Config) -> Result<(), String> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Err("resident-metal backend requires macOS with Metal feature".into());
    }
    if cfg.manifold.is_some() {
        return Err("manifold host options are not supported on resident-metal".into());
    }
    check_morbo(cfg)?;
    check_optimizer(cfg)
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
            let prefer = matches!(
                &cfg.optimizer,
                OptimizerSpec::OperatorBo {
                    perturbation: None,
                    ..
                } | OptimizerSpec::BayesianSearch {
                    distribution: Some(_),
                    perturbation: None,
                    ..
                }
            );
            if prefer && metal_ok(cfg).is_ok() {
                Ok(ExecMode::ResidentMetal)
            } else {
                Ok(ExecMode::HostRam)
            }
        }
    }
}
