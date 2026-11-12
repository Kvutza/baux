//! Configuration schemas for Julia experiments.

use baux::config::{AcquisitionSpec, OptimizerSpec};
use deser::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[deser(rename_all = "kebab-case")]
pub enum Backend {
    #[default]
    Auto,
    ResidentMetal,
    HostRam,
    HostDisk,
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct MorboCfg {
    #[deser(default = def_mreg())]
    pub regions: usize,
    #[deser(default = def_alpha())]
    pub alpha: f64,
    #[deser(default)]
    pub preferences: Vec<f32>,
    #[deser(default)]
    pub scales: Vec<f32>,
    #[deser(default = def_rescal())]
    pub rescalarize: baux::Rescalarize,
}

impl Default for MorboCfg {
    fn default() -> Self {
        Self {
            regions: def_mreg(),
            alpha: def_alpha(),
            preferences: Vec::new(),
            scales: Vec::new(),
            rescalarize: def_rescal(),
        }
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct ManifoldCfg {
    pub scale_x: bool,
    pub metric_learning: bool,
    pub stiefel_rank: Option<usize>,
    pub calibrate_alpha: Option<f64>,
    pub noise_aware: bool,
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct PerturbCfg {
    pub scale: Option<f64>,
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[deser(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct DistributionCfg {
    pub uniform: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct OceanCfg {
    #[deser(default = def_task())]
    pub task: String,
    #[deser(default = def_64())]
    pub nx: usize,
    #[deser(default = one_usize())]
    pub ny: usize,
    #[deser(default = def_64())]
    pub nz: usize,
    #[deser(default = def_act())]
    pub actuators: usize,
    #[deser(default = def_steps())]
    pub steps: usize,
    #[deser(default = def_dt())]
    pub dt: f64,
    #[deser(default = def_bound())]
    pub bound: f64,
    #[deser(default)]
    pub objectives: Vec<String>,
    #[deser(default)]
    pub disk: bool,
}

impl Default for OceanCfg {
    fn default() -> Self {
        Self {
            task: def_task(),
            nx: def_64(),
            ny: one_usize(),
            nz: def_64(),
            actuators: def_act(),
            steps: def_steps(),
            dt: def_dt(),
            bound: def_bound(),
            objectives: Vec::new(),
            disk: false,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct FuseCfg {
    #[deser(default = def_case())]
    pub case: String,
    #[deser(default = def_workflow())]
    pub workflow: String,
    #[deser(default = def_obj())]
    pub objective: String,
    #[deser(default)]
    pub objectives: Vec<String>,
    #[deser(default)]
    pub constraints: Vec<String>,
    #[deser(default = def_penalty())]
    pub penalty: f64,
}

impl Default for FuseCfg {
    fn default() -> Self {
        Self {
            case: def_case(),
            workflow: def_workflow(),
            objective: def_obj(),
            objectives: Vec::new(),
            constraints: Vec::new(),
            penalty: def_penalty(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[deser(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Config {
    pub version: u32,
    pub experiment: String,
    pub rounds: usize,
    #[deser(default = one_u32())]
    pub repetitions: u32,
    #[deser(default)]
    pub backend: Backend,
    pub optimizer: OptimizerSpec,
    #[deser(default)]
    pub ocean: OceanCfg,
    #[deser(default)]
    pub fuse: FuseCfg,
    pub morbo: Option<MorboCfg>,
    pub manifold: Option<ManifoldCfg>,
    pub perturbation: Option<PerturbCfg>,
    pub distribution: Option<DistributionCfg>,
}

const fn def_mreg() -> usize {
    4
}
const fn def_alpha() -> f64 {
    0.05
}
const fn def_rescal() -> baux::Rescalarize {
    baux::Rescalarize::OnRestart
}
fn def_task() -> String {
    "kelvin-helmholtz".into()
}
const fn def_64() -> usize {
    64
}
const fn one_usize() -> usize {
    1
}
const fn def_act() -> usize {
    2
}
const fn def_steps() -> usize {
    20
}
const fn def_dt() -> f64 {
    0.04
}
const fn def_bound() -> f64 {
    0.25
}
fn def_case() -> String {
    "ARC".into()
}
fn def_workflow() -> String {
    "stationary-plasma".into()
}
fn def_obj() -> String {
    "max_fusion".into()
}
const fn def_penalty() -> f64 {
    25.0
}
const fn one_u32() -> u32 {
    1
}

#[inline]
pub fn is_ocean(cfg: &Config) -> bool {
    cfg.experiment == "oceananigans"
}

pub fn obj_count(cfg: &Config) -> usize {
    let explicit = if is_ocean(cfg) {
        cfg.ocean.objectives.len()
    } else {
        cfg.fuse.objectives.len()
    };
    if explicit > 0 {
        return explicit;
    }
    if let Some(m) = &cfg.morbo {
        if !m.scales.is_empty() {
            return m.scales.len();
        }
        if !m.preferences.is_empty() {
            return m.preferences.len();
        }
    }
    if let OptimizerSpec::BayesianSearch {
        acquisition: AcquisitionSpec::Pareto { scales },
        ..
    } = &cfg.optimizer
    {
        if !scales.is_empty() {
            return scales.len();
        }
    }
    1
}

pub fn fuse_objs(cfg: &Config) -> String {
    if cfg.fuse.objectives.is_empty() {
        cfg.fuse.objective.clone()
    } else {
        cfg.fuse.objectives.join(",")
    }
}

pub fn fuse_key(cfg: &Config) -> String {
    let fc = &cfg.fuse;
    format!(
        "{}|{}|{}|{}|{}",
        fc.case,
        fc.workflow,
        fuse_objs(cfg),
        fc.penalty,
        fc.constraints.join(",")
    )
}

pub fn ocean_key(cfg: &Config) -> String {
    let oc = &cfg.ocean;
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        oc.task,
        oc.nx,
        oc.ny,
        oc.nz,
        oc.actuators,
        oc.steps,
        oc.dt,
        obj_count(cfg)
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecMode {
    ResidentMetal,
    HostRam,
    HostDisk,
    MetalPpo,
}
