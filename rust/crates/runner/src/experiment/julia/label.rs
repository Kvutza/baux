//! Experiment labels and model identifiers for Julia experiments.

use crate::experiment::julia::config::{Config, ExecMode, fuse_objs, is_ocean, obj_count};
use baux::config::OptimizerSpec;

pub fn opt_label(cfg: &Config, mode: ExecMode) -> String {
    let n_obj = obj_count(cfg);
    match mode {
        ExecMode::MetalPpo => "candle-ppo-metal".into(),
        ExecMode::ResidentMetal => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "resident-metal-morbo".into()
            } else if n_obj >= 2 {
                "resident-metal-pareto".into()
            } else if matches!(cfg.optimizer, OptimizerSpec::OperatorBo { .. }) {
                "resident-metal-operator-bo".into()
            } else {
                format!(
                    "metal-{}",
                    crate::experiment::tokamak::label::opt_label(&cfg.optimizer)
                )
            }
        }
        ExecMode::HostRam => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "host-ram-morbo".into()
            } else {
                crate::experiment::tokamak::label::opt_label(&cfg.optimizer)
            }
        }
        ExecMode::HostDisk => {
            if cfg.morbo.is_some() && n_obj >= 2 {
                "host-disk-bpann-morbo".into()
            } else {
                "host-disk-bpann-bo".into()
            }
        }
    }
}

pub fn model_label(cfg: &Config) -> String {
    if is_ocean(cfg) {
        let oc = &cfg.ocean;
        let ny_str = if oc.ny > 1 {
            format!("x{}", oc.ny)
        } else {
            String::new()
        };
        format!(
            "oceananigans-1.12-{}-{}{}-{}step-{}act",
            oc.task, oc.nx, ny_str, oc.steps, oc.actuators
        )
    } else {
        let fc = &cfg.fuse;
        format!("fuse-1.2.1-{}-{}-{}", fc.case, fc.workflow, fuse_objs(cfg))
    }
}
