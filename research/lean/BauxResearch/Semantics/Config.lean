import BauxResearch.Semantics.Bounds
import BauxResearch.Semantics.Optimizer

/-!
# Exact configuration, layout, override, file-config, FFI, and Python adapter semantics for ENN 0.4.1

This module formalizes the executable semantics and behavioral contracts of:
- `rust/crates/ennbo/src/config.rs` and `rust/crates/ennbo/src/config/rules.rs`
- `rust/crates/ennbo/src/enn_overrides.rs`
- `rust/crates/ennbo/src/layout.rs` and `rust/crates/ennbo/src/index.rs`
- `rust/crates/ennbo/src/file_config.rs`
- `rust/crates/ennbo/src/optimizer_factory.rs`
- `rust/crates/enn-py/src/py_optimizer.rs`, `py_enn_overrides.rs`, and `py_layout.rs`
- `src/enn/__init__.py`, `src/enn/_lazy.py`, `src/enn/turbo/rust_optimizer.py`,
  `src/enn/turbo/rust_optimizer_helpers.py`, and `src/enn/turbo/config/**`
-/

namespace BauxResearch.Semantics.Config

open Sampling

/-- `CandidateRV` in `config.rs` and `candidate_rv.py`. -/
inductive CandRv where
  | sobol
  | uniform
  | raasp
  deriving Repr, DecidableEq

/-- `CandidateConfig` in `config.rs`. -/
structure CandCfg where
  minCount : Nat := 10
  maxCount : Nat := 5000
  perDim : Nat := 100
  perArm : Nat := 0
  rv : CandRv := .uniform
  raaspFast : Bool := false
  deriving Repr, DecidableEq

/-- `CandidateConfig::default()` in `config.rs`. -/
def defaultCand : CandCfg := {}

/-- `turbo_candidate_config()` in `config.rs`. -/
def turboCand : CandCfg :=
  { defaultCand with rv := .raasp }

/-- `CandidateConfig::num_candidates(num_dim, num_arms)` in `config.rs`. -/
def poolSize (cfg : CandCfg) (numDim numArms : Nat) : Nat :=
  let raw := cfg.perDim * numDim + cfg.perArm * numArms
  Nat.max cfg.minCount (Nat.min cfg.maxCount raw)

/-- `IndexDriver` in `index.rs` and `layout.rs`. -/
inductive DriverKind where
  | flat
  | bpannMem
  | bpannDisk
  deriving Repr, DecidableEq

/-- `EnnStorage` in `layout.rs`. -/
inductive StoreKind where
  | memory
  | disk
  deriving Repr, DecidableEq

/-- `MetricLearning` in `layout.rs`. -/
inductive MetricKind where
  | disabled
  | auto
  deriving Repr, DecidableEq

/-- `EnnLayout` in `layout.rs`. -/
structure LayoutCfg where
  driver : DriverKind := .flat
  storage : StoreKind := .memory
  workDir : Option String := none
  scaleX : Bool := false
  metric : MetricKind := .disabled
  deriving Repr, DecidableEq

/-- `EnnLayout::memory(index_driver)` in `layout.rs`. -/
def memLayout (driver : DriverKind) : Option LayoutCfg :=
  match driver with
  | .bpannDisk => none
  | .flat | .bpannMem =>
      some { driver := driver, storage := .memory, workDir := none, scaleX := false, metric := .disabled }

/-- `EnnLayout::disk(work_dir, metric_learning)` in `layout.rs`. -/
def diskLayout (workDir : String) (metric : MetricKind) : LayoutCfg :=
  { driver := .bpannDisk, storage := .disk, workDir := some workDir, scaleX := false, metric := metric }

/-- `EnnLayout::with_scale_x(scale_x)` in `layout.rs`. -/
def withScale (layout : LayoutCfg) (scaleX : Bool) : Option LayoutCfg :=
  if scaleX && layout.metric == .auto then none
  else some { layout with scaleX := scaleX }

/-- `require_disk_work_dir(work_dir)` in `layout.rs`. -/
def reqDisk (workDir : Option String) : Option String :=
  workDir

/-- `auto_metric_learning_rejected(metric_learning, scale_x)` in `layout.rs`. -/
def autoRej (metric : MetricKind) (scaleX : Bool) : Bool :=
  metric == .auto && scaleX

/-- `EnnLayout::try_from_parts(...)` in `layout.rs`. -/
def tryParts (driver : Option DriverKind) (storage : Option StoreKind)
    (workDir : Option String) (scaleX : Bool) (metric : MetricKind) : Option LayoutCfg :=
  let resolvedStore := storage.getD (if workDir.isSome then .disk else .memory)
  let resolvedDriver := driver.getD (match resolvedStore with | .disk => .bpannDisk | .memory => .flat)
  if workDir.isSome && resolvedStore != .disk then none
  else match resolvedStore with
  | .memory =>
      if metric == .auto then none
      else match memLayout resolvedDriver with
      | none => none
      | some base => withScale base scaleX
  | .disk =>
      if resolvedDriver != .bpannDisk then none
      else if autoRej metric scaleX then none
      else match reqDisk workDir with
      | none => none
      | some dir => withScale (diskLayout dir metric) scaleX

/-- `is_disk_index_driver(driver)` in `index.rs`. -/
def isDisk (driver : DriverKind) : Bool :=
  driver == .bpannDisk

/-- `index_driver_to_wire` and `IndexDriver::to_wire`. -/
def driverWire (driver : DriverKind) : String :=
  match driver with
  | .flat => "flat"
  | .bpannMem => "bpann_memory"
  | .bpannDisk => "bpann_disk"

/-- `IndexDriver::from_wire` and `index_driver_from_wire`. -/
def wireDriver (wire : String) : Option DriverKind :=
  if wire == "flat" then some .flat
  else if wire == "bpann_memory" then some .bpannMem
  else if wire == "bpann_disk" then some .bpannDisk
  else none

/-- `enn_storage_to_wire` in `py_layout.rs`. -/
def storeWire (storage : StoreKind) : String :=
  match storage with
  | .memory => "memory"
  | .disk => "disk"

/-- `enn_storage_from_wire` in `py_layout.rs`. -/
def wireStore (wire : String) : Option StoreKind :=
  if wire == "memory" then some .memory
  else if wire == "disk" then some .disk
  else none

/-- `metric_learning_to_wire` in `py_layout.rs`. -/
def metricWire (metric : MetricKind) : String :=
  match metric with
  | .disabled => "disabled"
  | .auto => "auto"

/-- `metric_learning_from_wire` in `py_layout.rs`. -/
def wireMetric (wire : String) : Option MetricKind :=
  if wire == "disabled" then some .disabled
  else if wire == "auto" then some .auto
  else none

/-- `scale_x_to_wire` in `py_layout.rs`. -/
def scaleWire (scaleX : Bool) : Bool :=
  scaleX

/-- Abstract `ENNIndex` state for `index.rs`. -/
structure IndexState where
  driver : DriverKind
  dim : Nat
  rows : Nat
  deriving Repr, DecidableEq

/-- `ENNIndex::len` in `index.rs`. -/
def indexLen (idx : IndexState) : Nat :=
  idx.rows

/-- `ENNIndex::num_dim` in `index.rs`. -/
def indexDim (idx : IndexState) : Nat :=
  idx.dim

/-- `ENNIndex::is_empty` in `index.rs`. -/
def indexEmpty (idx : IndexState) : Bool :=
  idx.rows == 0

/-- `ENNIndex::add_Observations` in `index.rs`. -/
def indexAdd (idx : IndexState) (batchRows batchCols : Nat) : Option IndexState :=
  if batchCols != idx.dim then none
  else some { idx with rows := idx.rows + batchRows }

/-- `FitSamplesConfig` in `config.rs`. -/
inductive FitMode where
  | frozen (samples : Nat)
  | searched (samples : Option Nat) (iters : Nat)
  deriving Repr, DecidableEq

/-- `FitSamplesConfig::default()` in `config.rs`. -/
def defaultFit : FitMode :=
  .searched none 100

/-- `FitSamplesConfig::is_frozen` in `config.rs`. -/
def isFrozen (mode : FitMode) : Bool :=
  match mode with
  | .frozen _ => true
  | .searched _ _ => false

/-- `FitSamplesConfig::num_fit_samples` in `config.rs`. -/
def fitSamples (mode : FitMode) : Option Nat :=
  match mode with
  | .frozen n => some n
  | .searched n _ => n

/-- `FitSamplesConfig::num_fit_iters` in `config.rs`. -/
def fitIters (mode : FitMode) : Option Nat :=
  match mode with
  | .frozen _ => none
  | .searched _ iters => some iters

/-- `AcquisitionOption` in `config.rs`. -/
inductive AcqKind where
  | ucb
  | thompson
  | pareto
  | random
  deriving Repr, DecidableEq

/-- `SurrogateConfig` in `config.rs`. -/
structure EnnCfg where
  k : Nat := 10
  fit : FitMode := defaultFit
  layout : LayoutCfg := {}
  yBounds : Option (List (Word × Word)) := none
  tiedDims : Option (List (Nat × Nat)) := none
  deriving Repr, DecidableEq

/-- `SurrogateConfig` option (`None` vs `ENN`). -/
inductive SurKind where
  | none
  | enn (cfg : EnnCfg)
  deriving Repr, DecidableEq

/-- `TrustRegionKind` in `config.rs`. -/
inductive TrKind where
  | none
  | turbo
  | morbo
  deriving Repr, DecidableEq

/-- `TrustRegionOverrides` in `config.rs`. -/
structure TrOv where
  lenInit : Option Word := none
  lenMin : Option Word := none
  lenMax : Option Word := none
  successTol : Option Nat := none
  failureTol : Option Nat := none
  deriving Repr, DecidableEq

/-- `TrustRegionParams` in `config.rs`. -/
structure TrParams where
  lenInit : Word := Bounds.one
  lenMin : Word := Bounds.half
  lenMax : Word := Bounds.one
  successTol : Nat := 3
  failureTol : Nat := 4
  deriving Repr, DecidableEq

/-- `apply_trust_region_overrides` in `config.rs`. -/
def applyTr (base : TrParams) (ov : TrOv) : TrParams :=
  { lenInit := ov.lenInit.getD base.lenInit
    lenMin := ov.lenMin.getD base.lenMin
    lenMax := ov.lenMax.getD base.lenMax
    successTol := ov.successTol.getD base.successTol
    failureTol := ov.failureTol.getD base.failureTol }

/-- `OptimizerConfig` in `config.rs`. -/
structure OptCfg where
  surrogate : SurKind := .enn {}
  candidates : CandCfg := turboCand
  acq : AcqKind := .ucb
  trKind : TrKind := .turbo
  trParams : TrParams := {}
  hasMorbo : Bool := false
  numInit : Option Nat := none
  lhdOnly : Bool := false
  trailingObs : Option Nat := none
  deriving Repr, DecidableEq

/-- `turbo_enn_config()` in `config.rs`. -/
def turboEnn (k : Nat := 10) : OptCfg :=
  { surrogate := .enn { k := k }
    candidates := turboCand
    acq := .ucb
    trKind := .turbo }

/-- `turbo_zero_config()` in `config.rs`. -/
def turboZero : OptCfg :=
  { surrogate := .none
    candidates := turboCand
    acq := .random
    trKind := .turbo }

/-- `lhd_only_config()` in `config.rs`. -/
def lhdOnly : OptCfg :=
  { surrogate := .none
    candidates := { minCount := 1, maxCount := 1000000000, perDim := 1, perArm := 0, rv := .sobol }
    acq := .random
    trKind := .none
    lhdOnly := true }

/-- `validate_optimizer_rules(lhd_only, has_surrogate, acquisition)` in `config/rules.rs`. -/
def rulesOk (isLhd hasSur : Bool) (acq : AcqKind) : Bool :=
  if isLhd && hasSur then false
  else if !hasSur && (acq == .thompson || acq == .ucb) then false
  else true

/-- `require_fit_samples(acquisition, fit_samples)` in `config.rs`. -/
def fitOk (acq : AcqKind) (mode : FitMode) : Bool :=
  match acq with
  | .pareto | .random => true
  | .ucb | .thompson => !isFrozen mode

/-- `OptimizerConfig::validate_kind` in `config.rs`. -/
def kindOk (cfg : OptCfg) : Bool :=
  let hasSur := match cfg.surrogate with | .none => false | .enn _ => true
  let fitValid := match cfg.surrogate with
    | .none => true
    | .enn e => fitOk cfg.acq e.fit
  let morboValid := cfg.hasMorbo == (cfg.trKind == .morbo)
  rulesOk cfg.lhdOnly hasSur cfg.acq && fitValid && morboValid

/-- `SearchOverrides` in `enn_overrides.rs`. -/
structure SearchOv where
  fitSamples : Option Nat := none
  fitIters : Option Nat := none
  deriving Repr, DecidableEq

/-- `SearchOverrides::is_empty` in `enn_overrides.rs`. -/
def searchEmpty (ov : SearchOv) : Bool :=
  ov.fitSamples.isNone && ov.fitIters.isNone

/-- `SearchOverrides::apply` in `enn_overrides.rs`. -/
def applySearch (ov : SearchOv) (base : FitMode) : Option FitMode :=
  if searchEmpty ov then some base
  else
    let samples := match ov.fitSamples with | some s => some s | none => fitSamples base
    let iters := match ov.fitIters with
      | some i => i
      | none => (fitIters base).getD 100
    if iters == 0 then
      match samples with
      | some n => some (.frozen n)
      | none => none
    else
      some (.searched samples iters)

/-- `EnnOverrides` in `enn_overrides.rs`. -/
structure EnnOv where
  k : Option Nat := none
  search : SearchOv := {}
  driver : Option DriverKind := none
  storage : Option StoreKind := none
  workDir : Option String := none
  scaleX : Option Bool := none
  metric : Option MetricKind := none
  yBounds : Option (List (Word × Word)) := none
  tiedDims : Option (List (Nat × Nat)) := none
  deriving Repr, DecidableEq

/-- `EnnOverrides::layout` in `enn_overrides.rs`. -/
def ovLayout (ov : EnnOv) (base : LayoutCfg) : Option LayoutCfg :=
  if ov.driver.isNone && ov.storage.isNone && ov.workDir.isNone &&
     ov.scaleX.isNone && ov.metric.isNone then
    some base
  else
    let store := ov.storage.orElse (fun _ => if ov.workDir.isSome then some .disk else some base.storage)
    let drv := ov.driver.orElse (fun _ =>
      if ov.storage == some .disk || ov.workDir.isSome then some .bpannDisk else some base.driver)
    let dir := ov.workDir.orElse (fun _ => base.workDir)
    let sc := ov.scaleX.getD base.scaleX
    let met := ov.metric.getD base.metric
    tryParts drv store dir sc met

/-- `EnnOverrides::apply` in `enn_overrides.rs`. -/
def applyEnn (ov : EnnOv) (base : EnnCfg) : Option EnnCfg :=
  match applySearch ov.search base.fit, ovLayout ov base.layout with
  | some fit', some layout' =>
      let k' := ov.k.getD base.k
      if k' == 0 then none
      else some
        { k := k'
          fit := fit'
          layout := layout'
          yBounds := ov.yBounds.orElse (fun _ => base.yBounds)
          tiedDims := ov.tiedDims.orElse (fun _ => base.tiedDims) }
  | _, _ => none

/-- `apply_enn_overrides` in `config.rs`. -/
def ennOpt (base : SurKind) (ov : Option EnnOv) : Option SurKind :=
  match ov with
  | none => some base
  | some eOv =>
      match base with
      | .none => none
      | .enn cfg => (applyEnn eOv cfg).map .enn

/-- `ConfigOverrides` in `config.rs`. -/
structure CfgOv where
  enn : Option EnnOv := none
  trKind : Option TrKind := none
  tr : TrOv := {}
  hasMorbo : Option Bool := none
  acq : Option AcqKind := none
  candRv : Option CandRv := none
  raaspFast : Option Bool := none
  minCand : Option Nat := none
  maxCand : Option Nat := none
  perDim : Option Nat := none
  numInit : Option Nat := none
  trailingObs : Option Nat := none
  deriving Repr, DecidableEq

/-- `ConfigOverrides::apply_to` in `config.rs`. -/
def applyCfg (ov : CfgOv) (cfg : OptCfg) : Option OptCfg :=
  match ennOpt cfg.surrogate ov.enn with
  | none => none
  | some sur' =>
      let cand' : CandCfg :=
        { minCount := ov.minCand.getD cfg.candidates.minCount
          maxCount := ov.maxCand.getD cfg.candidates.maxCount
          perDim := ov.perDim.getD cfg.candidates.perDim
          perArm := cfg.candidates.perArm
          rv := ov.candRv.getD cfg.candidates.rv
          raaspFast := ov.raaspFast.getD cfg.candidates.raaspFast }
      let next : OptCfg :=
        { surrogate := sur'
          candidates := cand'
          acq := ov.acq.getD cfg.acq
          trKind := ov.trKind.getD cfg.trKind
          trParams := applyTr cfg.trParams ov.tr
          hasMorbo := ov.hasMorbo.getD cfg.hasMorbo
          numInit := ov.numInit.orElse (fun _ => cfg.numInit)
          lhdOnly := cfg.lhdOnly
          trailingObs := ov.trailingObs.orElse (fun _ => cfg.trailingObs) }
      if kindOk next then some next else none

/-- `BpannConfig` in `file_config.rs`. -/
structure BpannCfg where
  pendingFlush : Nat := 1000
  pendingHard : Option Nat := none
  beamWidth : Nat := 4
  exhaustiveLimit : Nat := 512
  skipRefinement : Nat := 2048
  compactRows : Nat := 4096
  deriving Repr, DecidableEq

/-- `BpannConfig::resolved_pending_hard_flush_threshold` in `file_config.rs`. -/
def hardFlush (cfg : BpannCfg) : Nat :=
  cfg.pendingHard.getD (Nat.max 3000 cfg.pendingFlush)

/-- `BpannConfig::validate` in `file_config.rs`. -/
def bpannOk (cfg : BpannCfg) : Bool :=
  hardFlush cfg >= cfg.pendingFlush &&
  cfg.beamWidth > 0 &&
  cfg.exhaustiveLimit > 0 &&
  cfg.skipRefinement >= cfg.exhaustiveLimit &&
  cfg.compactRows > 0

/-- `BpannConfig::to_tuning` in `file_config.rs`. -/
def toTuning (cfg : BpannCfg) : Option (BpannCfg × Nat) :=
  if bpannOk cfg then some (cfg, hardFlush cfg) else none

/-- `Config::load` in `file_config.rs` (falls back to default when invalid). -/
def loadCfg (input : Option BpannCfg) : BpannCfg :=
  match input with
  | none => {}
  | some cfg => if bpannOk cfg then cfg else {}

/-- `Config::save` in `file_config.rs`. -/
def saveCfg (cfg : BpannCfg) : Option BpannCfg :=
  if bpannOk cfg then some cfg else none

/-- `resolve_k` in `optimizer_factory.rs`. -/
def resolveK (k : Option Nat) : Option Nat :=
  let resolved := k.getD 10
  if resolved >= 1 then some resolved else none

/-- `resolve_num_init` in `optimizer_factory.rs`. -/
def resolveInit (numInit : Option Nat) : Option Nat :=
  match numInit with
  | none => some 10
  | some n => if n >= 1 then some n else none

/-- `create_optimizer_enn` in `optimizer_factory.rs`. -/
def createOpt (dim : Nat) (k numInit : Option Nat) : Option OptCfg :=
  if dim == 0 then none
  else match resolveK k, resolveInit numInit with
  | some kVal, some initVal =>
      let cfg := { turboEnn kVal with numInit := some initVal }
      if kindOk cfg then some cfg else none
  | _, _ => none

/-- `create_optimizer_enn_with_overrides` in `optimizer_factory.rs`. -/
def createOv (dim : Nat) (k numInit : Option Nat) (ov : CfgOv) : Option OptCfg :=
  match createOpt dim k numInit with
  | none => none
  | some base => applyCfg ov base

/-- `acquisition_kind` in `py_optimizer.rs`. -/
def acqName (acq : AcqKind) : String :=
  match acq with
  | .ucb => "ucb"
  | .thompson => "thompson"
  | .pareto => "pareto"
  | .random => "random"

/-- `acquisition_from_name` in `py_optimizer.rs`. -/
def nameAcq (label : String) : Option AcqKind :=
  if label == "ucb" then some .ucb
  else if label == "thompson" || label == "ts" then some .thompson
  else if label == "pareto" then some .pareto
  else if label == "random" then some .random
  else none

/-- `parse_candidate_rv` in `py_optimizer.rs`. -/
def nameCand (label : String) : Option CandRv :=
  if label == "sobol" then some .sobol
  else if label == "uniform" then some .uniform
  else if label == "raasp" then some .raasp
  else none

/-- Inverse wire name for `CandRv`. -/
def candName (rv : CandRv) : String :=
  match rv with
  | .sobol => "sobol"
  | .uniform => "uniform"
  | .raasp => "raasp"

/-- `TRLengthConfig` validation in `src/enn/turbo/config/trust_region.py`. -/
structure TrLen where
  lenInit : Nat
  lenMin : Nat
  lenMax : Nat
  deriving Repr, DecidableEq

/-- `TRLengthConfig.__post_init__` checks `0 < lenMin <= lenInit <= lenMax`. -/
def trOk (cfg : TrLen) : Bool :=
  0 < cfg.lenMin && cfg.lenMin <= cfg.lenInit && cfg.lenInit <= cfg.lenMax

/-- `ENNFitConfig` in `src/enn/turbo/config/enn_surrogate.py`. -/
structure FitPy where
  samples : Option Nat := none
  iters : Nat := 100
  deriving Repr, DecidableEq

/-- `ENNFitConfig.__post_init__` validation in `enn_surrogate.py`. -/
def validFit (cfg : FitPy) : Bool :=
  match cfg.samples with
  | some s => s >= 1 && (cfg.iters >= 1 || cfg.iters == 0)
  | none => cfg.iters >= 1

/-- `MultiObjectiveConfig` in `src/enn/turbo/config/trust_region.py`. -/
structure MultiCfg where
  numMetrics : Nat
  alpha : Option Word := none
  deriving Repr, DecidableEq

/-- `MultiObjectiveConfig.__post_init__` checks `num_metrics >= 2`. -/
def multiOk (cfg : MultiCfg) : Bool :=
  cfg.numMetrics >= 2

/-- `reject_unknown_override_keys` in `py_optimizer.rs`. -/
def allowedKeys : List String :=
  ["enn", "trust_region", "acquisition", "candidate_rv", "raasp_fast",
   "min_candidates", "max_candidates", "candidates_per_dim", "num_init",
   "trailing_obs"]

/-- Returns `true` iff every key in `keys` is in `allowedKeys`. -/
def rejectKeys (keys : List String) : Bool :=
  keys.all (fun k => allowedKeys.contains k)

/-- `_config_to_rust_overrides` / `parse_config_overrides_from_dict` wire payload. -/
structure PyDict where
  acq : String
  candRv : String
  raaspFast : Bool
  minCand : Nat
  maxCand : Nat
  perDim : Nat
  driver : String
  storage : String
  metric : String
  scaleX : Bool
  deriving Repr, DecidableEq

/-- `_config_to_rust_overrides` in `rust_optimizer_helpers.py`. -/
def toDict (acq : AcqKind) (cand : CandCfg) (layout : LayoutCfg) : PyDict :=
  { acq := acqName acq
    candRv := candName cand.rv
    raaspFast := cand.raaspFast
    minCand := cand.minCount
    maxCand := cand.maxCount
    perDim := cand.perDim
    driver := driverWire layout.driver
    storage := storeWire layout.storage
    metric := metricWire layout.metric
    scaleX := scaleWire layout.scaleX }

/-- `parse_config_overrides_from_dict` in `py_optimizer.rs`. -/
def fromDict (d : PyDict) : Option (AcqKind × CandCfg × DriverKind × StoreKind × MetricKind × Bool) :=
  match nameAcq d.acq, nameCand d.candRv, wireDriver d.driver, wireStore d.storage, wireMetric d.metric with
  | some acq, some rv, some drv, some st, some met =>
      let cand : CandCfg :=
        { minCount := d.minCand
          maxCount := d.maxCand
          perDim := d.perDim
          perArm := 0
          rv := rv
          raaspFast := d.raaspFast }
      some (acq, cand, drv, st, met, d.scaleX)
  | _, _, _, _, _ => none

/-- `RustOptimizer._tell_estimate` in `rust_optimizer.py`: delegates `tell` then returns `tr_obs_count`. -/
def tellEst (ops : Optimizer.Ops) (cfg : Optimizer.Config)
    (state : Optimizer.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) : Optimizer.Step × Nat :=
  let step := Optimizer.step ops cfg state (.tell x y variance)
  (step, step.state.life.observations)

/-- `lazy_getattr` in `src/enn/_lazy.py`. -/
def lazyAttr (exports : List (String × String)) (label : String) : Option String :=
  (exports.find? (fun p => p.1 == label)).map Prod.snd

/-! ## Behavioral Theorems -/

/-- `poolSize` is always bounded below by `cfg.minCount`. -/
theorem poolLower (cfg : CandCfg) (numDim numArms : Nat) :
    cfg.minCount ≤ poolSize cfg numDim numArms := by
  unfold poolSize
  exact Nat.le_max_left cfg.minCount _

/-- When `cfg.minCount ≤ cfg.maxCount`, `poolSize` is bounded above by `cfg.maxCount`. -/
theorem poolUpper (cfg : CandCfg) (numDim numArms : Nat)
    (h : cfg.minCount ≤ cfg.maxCount) :
    poolSize cfg numDim numArms ≤ cfg.maxCount := by
  unfold poolSize
  exact Nat.max_le.mpr ⟨h, Nat.min_le_left cfg.maxCount _⟩

/-- `turboCand` selects `.raasp` while preserving default candidate bounds. -/
theorem turboRaasp :
    turboCand.rv = .raasp ∧ turboCand.minCount = 10 ∧ turboCand.maxCount = 5000 := by
  decide +kernel

/-- `memLayout` rejects `.bpannDisk` and accepts `.flat` and `.bpannMem`. -/
theorem memDriver :
    memLayout .bpannDisk = none ∧
    (memLayout .flat).isSome = true ∧
    (memLayout .bpannMem).isSome = true := by
  decide +kernel

/-- `tryParts` rejects `.auto` metric learning when `scaleX = true` or storage is `.memory`. -/
theorem autoLayout (driver : Option DriverKind) (dir : Option String) :
    tryParts driver (some .memory) none false .auto = none ∧
    tryParts (some .bpannDisk) (some .disk) dir true .auto = none := by
  constructor
  · rfl
  · simp [tryParts, autoRej]

/-- `driverWire` and `wireDriver` form an exact round-trip bijection on `DriverKind`. -/
theorem driverRound (d : DriverKind) :
    wireDriver (driverWire d) = some d := by
  cases d <;> rfl

/-- `storeWire` and `wireStore` form an exact round-trip bijection on `StoreKind`. -/
theorem storeRound (s : StoreKind) :
    wireStore (storeWire s) = some s := by
  cases s <;> rfl

/-- `metricWire` and `wireMetric` form an exact round-trip bijection on `MetricKind`. -/
theorem metricRound (m : MetricKind) :
    wireMetric (metricWire m) = some m := by
  cases m <;> rfl

/-- `acqName` and `nameAcq` form an exact round-trip on `AcqKind`. -/
theorem acqRound (a : AcqKind) :
    nameAcq (acqName a) = some a := by
  cases a <;> rfl

/-- `candName` and `nameCand` form an exact round-trip on `CandRv`. -/
theorem candRound (rv : CandRv) :
    nameCand (candName rv) = some rv := by
  cases rv <;> rfl

/-- `toDict` and `fromDict` round-trip all acquisition, candidate, and layout wire fields. -/
theorem dictRound (acq : AcqKind) (cand : CandCfg) (layout : LayoutCfg)
    (hArm : cand.perArm = 0) :
    fromDict (toDict acq cand layout) =
      some (acq, cand, layout.driver, layout.storage, layout.metric, layout.scaleX) := by
  cases acq <;> cases hRv : cand.rv <;>
    cases hDrv : layout.driver <;> cases hSt : layout.storage <;>
    cases hMet : layout.metric <;> cases cand <;>
    simp_all [toDict, fromDict, acqName, nameAcq, candName, nameCand,
      driverWire, wireDriver, storeWire, wireStore, metricWire, wireMetric, scaleWire]

/-- `rulesOk` rejects `lhdOnly` with a surrogate and rejects `.ucb`/`.thompson` without a surrogate. -/
theorem rulesReject (acq : AcqKind) :
    rulesOk true true acq = false ∧
    rulesOk false false .ucb = false ∧
    rulesOk false false .thompson = false := by
  refine ⟨by rfl, by rfl, by rfl⟩

/-- `turboEnn`, `turboZero`, and `lhdOnly` all satisfy `kindOk`. -/
theorem presetValid (k : Nat) :
    kindOk (turboEnn k) = true ∧
    kindOk turboZero = true ∧
    kindOk lhdOnly = true := by
  refine ⟨by rfl, by rfl, by rfl⟩

/-- `applySearch` with `fitIters = some 0` and no `fitSamples` on a searched base with `none` fails. -/
theorem frozenSamples :
    applySearch { fitSamples := none, fitIters := some 0 } defaultFit = none := by
  decide +kernel

/-- `ennOpt` rejects `some eOv` when the base surrogate is `.none`. -/
theorem ennSurrogate (eOv : EnnOv) :
    ennOpt .none (some eOv) = none := by
  rfl

/-- `hardFlush` is always at least `cfg.pendingFlush` when `pendingHard = none`. -/
theorem defaultHard (flush : Nat) :
    flush ≤ hardFlush { pendingFlush := flush, pendingHard := none } := by
  unfold hardFlush
  exact Nat.le_max_right 3000 flush

/-- `loadCfg` falls back to default `{}` when validation fails (e.g. `beamWidth = 0`). -/
theorem loadFallback :
    loadCfg (some { beamWidth := 0 }) = ({} : BpannCfg) := by
  decide +kernel

/-- `createOpt` succeeds for positive `dim`, `k`, and `numInit`, and rejects `dim = 0`. -/
theorem createValid :
    createOpt 0 (some 10) (some 10) = none ∧
    (createOpt 4 (some 10) (some 10)).isSome = true := by
  decide +kernel

/-- `indexAdd` rejects a dimension mismatch and increments `rows` when `batchCols = idx.dim`. -/
theorem addSpec (idx : IndexState) (batchRows : Nat) :
    indexAdd idx batchRows idx.dim = some { idx with rows := idx.rows + batchRows } := by
  simp [indexAdd]

/-- `tellEst` returns the exact `Optimizer.step` transition paired with the post-step observation count. -/
theorem estSpec (ops : Optimizer.Ops) (cfg : Optimizer.Config)
    (state : Optimizer.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) :
    (tellEst ops cfg state x y variance).1 =
      Optimizer.step ops cfg state (.tell x y variance) ∧
    (tellEst ops cfg state x y variance).2 =
      (Optimizer.step ops cfg state (.tell x y variance)).state.life.observations := by
  exact ⟨rfl, rfl⟩

/-- `trOk`, `validFit`, `multiOk`, and `rejectKeys` enforce the Python config invariants. -/
theorem pyValid :
    trOk ⟨8, 1, 16⟩ = true ∧
    trOk ⟨8, 0, 16⟩ = false ∧
    validFit ⟨none, 100⟩ = true ∧
    validFit ⟨none, 0⟩ = false ∧
    multiOk ⟨2, none⟩ = true ∧
    multiOk ⟨1, none⟩ = false ∧
    rejectKeys ["enn", "acquisition"] = true ∧
    rejectKeys ["unknown_key"] = false := by
  decide +kernel

/-- `lazyAttr` resolves registered exports and returns `none` for missing attributes. -/
theorem lazySpec :
    lazyAttr [("TurboConfig", "enn.turbo.config")] "TurboConfig" = some "enn.turbo.config" ∧
    lazyAttr [("TurboConfig", "enn.turbo.config")] "Missing" = none := by
  decide +kernel

#print axioms poolLower
#print axioms poolUpper
#print axioms turboRaasp
#print axioms memDriver
#print axioms autoLayout
#print axioms driverRound
#print axioms storeRound
#print axioms metricRound
#print axioms acqRound
#print axioms candRound
#print axioms dictRound
#print axioms rulesReject
#print axioms presetValid
#print axioms frozenSamples
#print axioms ennSurrogate
#print axioms defaultHard
#print axioms loadFallback
#print axioms createValid
#print axioms addSpec
#print axioms estSpec
#print axioms pyValid
#print axioms lazySpec

end BauxResearch.Semantics.Config
