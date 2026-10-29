import BauxResearch.Semantics.Optimizer
import BauxResearch.Extraction.LocalApi
import BauxResearch.Extraction.LocalCoord

/-!
# ENN 0.4.1 public API construction, validation, and extracted gate/coordinate bridge

This module closes the public API entry-point surface for `ennbo` 0.4.1:
- `InitStrategy` validation (`InitStrategy::new`),
- `TrustRegionBuilder` validation (`TrustRegionConfig::new` / builder gates),
- `Acquisition::default`, `SurrogateMode`, `AutoMetricMode`, and `MorboWeightMode`,
- `OptimizerConfig` / `Strategy` construction validation and driver selection
  (`FlatIndex` in-memory vs `BpannDisk` on-disk),
- `Tell` / `Lifecycle` dimension, row-count, y-variance, and bounds guards,
- `Optimizer` concrete runtime simulation (`ArtifactRel`, `runtimeSim`, `runtimeTrace`),
- Exact correspondence theorems connecting Charon+Aeneas extracted `LocalApi` and
  `LocalCoord` functions to `Optimizer.step`, `Bounds.toUnit`, `Tell.unitRow`,
  `Tell.toUnit`, and `Lifecycle.fromRow`.
-/

namespace BauxResearch.Semantics.Guard

open Sampling Bounds
open BauxResearch.Extraction.Native
open BauxResearch.Extraction.LocalApi
open BauxResearch.Extraction.LocalCoord

/-- Configuration for `InitStrategy::new(kind, num_init)`. -/
structure InitConfig where
  kind : Opt.InitKind
  numInit : Nat
  deriving Repr, DecidableEq

/-- Public `InitStrategy` variants (`InitOnly` vs `Hybrid`). -/
inductive InitChoice where
  | only (cfg : InitConfig)
  | hybrid (cfg : InitConfig)
  deriving Repr, DecidableEq

/-- Builder parameters for `TrustRegionConfig` (`length_init`, `length_min`, `length_max`,
`success_tolerance`, `failure_tolerance`). -/
structure BuilderConfig where
  initial : Word
  minimum : Word
  maximum : Word
  successTol : Nat
  failureTol : Nat
  deriving Repr, DecidableEq

/-- Public acquisition choice (`Thompson`, `UCB`, `Pareto`, `Random`). -/
inductive AcqChoice where
  | thompson
  | ucb
  | pareto
  | random
  deriving Repr, DecidableEq

instance : Inhabited AcqChoice where
  default := .thompson

/-- Public `SurrogateMode` (`Draw` vs `Frozen`). -/
inductive SurrMode where
  | draw (candidates : Nat) (samples : Nat) (calibrate : Bool)
  | frozen
  deriving Repr, DecidableEq

/-- Public `AutoMetricMode` (`Disabled` vs `Enabled`). -/
inductive MetricMode where
  | disabled
  | enabled (tied : List (List Nat)) (seed : Word)
  deriving Repr, DecidableEq

/-- Public `MorboWeightMode` (`ResampleOnRestart` vs `ResampleOnPropose`). -/
inductive WeightMode where
  | onRestart
  | onPropose
  deriving Repr, DecidableEq

/-- Public strategy configuration (`InitOnly`, `Turbo`, `Hybrid`, `Morbo`). -/
inductive StrategyChoice where
  | initOnly (init : InitConfig)
  | turbo (trust : BuilderConfig) (acq : AcqChoice)
  | hybrid (init : InitConfig) (trust : BuilderConfig) (acq : AcqChoice)
  | morbo (init : Option InitConfig) (trust : BuilderConfig) (acq : AcqChoice)
      (metrics : Nat) (alpha : Word) (weightMode : WeightMode)
  deriving Repr, DecidableEq

/-- Complete public `OptimizerConfig` construction surface. -/
structure OptConfig where
  bounds : List Bound
  strategy : StrategyChoice
  surrogate : Option SurrMode
  onDisk : Bool
  scaleX : Bool
  metric : MetricMode
  yBounds : Option Fitter.BoundSet
  k : Int
  seed : Word
  deriving Repr, DecidableEq

/-- Constructed strategy tag and selected neighbor driver. -/
structure BuildResult where
  dim : Nat
  driver : Neighbor.Driver
  hasSurrogate : Bool
  isMorbo : Bool
  deriving Repr, DecidableEq

/-- Validate parameter space bounds (`bounds.nrows() > 0`, finite `low < high` for each row). -/
def validateBounds (bounds : List Bound) : Except Failure Nat :=
  if bounds.isEmpty then .error .invalid
  else
    let ok := bounds.all fun (low, high) =>
      Float.finiteDouble low && Float.finiteDouble high && Float.ltDouble low high
    if ok then .ok bounds.length else .error .invalid

/-- Validate `InitStrategy::new(kind, num_init)`: requires `num_init > 0`. -/
def validateInit (cfg : InitConfig) : Except Failure InitConfig :=
  if cfg.numInit == 0 then .error .invalid else .ok cfg

/-- Validate `TrustRegionBuilder::build(dim)`:
requires `0 < minimum <= initial <= maximum` (all finite) and positive tolerances. -/
def validateBuilder (dim : Nat) (cfg : BuilderConfig) : Except Failure BuilderConfig :=
  if dim == 0 then .error .invalid
  else if !Float.finiteDouble cfg.initial || !Float.finiteDouble cfg.minimum ||
      !Float.finiteDouble cfg.maximum then .error .invalid
  else if !Float.ltDouble 0 cfg.minimum then .error .invalid
  else if Float.ltDouble cfg.initial cfg.minimum then .error .invalid
  else if Float.ltDouble cfg.maximum cfg.initial then .error .invalid
  else if cfg.successTol == 0 || cfg.failureTol == 0 then .error .invalid
  else .ok cfg

/-- Validate `StrategyChoice` and return `(hasSurrogate, isMorbo)`. -/
def validateStrategy (dim : Nat) (choice : StrategyChoice) :
    Except Failure (Bool × Bool) := do
  match choice with
  | .initOnly init =>
      let _ ← validateInit init
      return (false, false)
  | .turbo trust _ =>
      let _ ← validateBuilder dim trust
      return (true, false)
  | .hybrid init trust _ =>
      let _ ← validateInit init
      let _ ← validateBuilder dim trust
      return (true, false)
  | .morbo initOpt trust _ metrics alpha _ =>
      match initOpt with
      | some init =>
          let _ ← validateInit init
          pure ()
      | none => pure ()
      let _ ← validateBuilder dim trust
      if metrics < 2 then throw .invalid
      if !Float.finiteDouble alpha || Float.ltDouble alpha 0 then throw .invalid
      return (true, true)

/-- Select neighbor driver from `on_disk` configuration (`bpann` vs `flat`). -/
def selectBackend (onDisk : Bool) : Neighbor.Driver :=
  if onDisk then .bpann else .flat

/-- Complete `Optimizer::new` construction gate. -/
def validateOpt (cfg : OptConfig) : Except Failure BuildResult := do
  let dim ← validateBounds cfg.bounds
  let (needsSurr, isMorbo) ← validateStrategy dim cfg.strategy
  if cfg.k <= 0 then throw .invalid
  match cfg.surrogate with
  | some (.draw candidates samples _) =>
      if candidates == 0 || samples == 0 then throw .invalid
  | _ => pure ()
  let driver := selectBackend cfg.onDisk
  return ⟨dim, driver, needsSurr && cfg.surrogate.isSome, isMorbo⟩

/-- Check `x` matrix shape against optimizer dimension (`Tell.tell` guard 1). -/
def checkX (dim : Nat) (x : Fitter.Matrix) : Except Tell.TellError Unit :=
  if x.cols != dim then .error .xShape else .ok ()

/-- Check `x` and `y` row counts (`Tell.tell` guard 2). -/
def checkTell (dim : Nat) (x y : Fitter.Matrix) : Except Tell.TellError Bool :=
  if x.cols != dim then .error .xShape
  else if y.rows != x.rows then .error .rowShape
  else .ok (x.rows == 0)

/-- Validate `y_bounds` against metric count before model construction. -/
def checkBounds (bounds : Option Fitter.BoundSet) (outputs : Nat) :
    Except Fitter.BoundsError Unit :=
  match bounds with
  | none => .ok ()
  | some b => Fitter.validateBounds b outputs

/-- Check `tell` readiness gate when a surrogate fit state is present. -/
def tellReady (gate : Append.Gate) (fit : Option Append.FitState) :
    Except Tell.TellError Unit :=
  match fit with
  | some _ =>
      match gate .wait with
      | .error cause => .error (.wait cause)
      | .ok () => .ok ()
  | none => .ok ()

/-- Valid single-dimension `[0, 1]` bounds succeed with `dim = 1`. -/
theorem validOne :
    validateBounds [(0, Bounds.one)] = .ok 1 := by
  decide +kernel

/-- Degenerate bounds `low == high` are rejected by `validateBounds`. -/
theorem badBounds :
    validateBounds [(Bounds.one, Bounds.one)] = .error .invalid := by
  decide +kernel

/-- `InitStrategy::new` rejects `num_init == 0`. -/
theorem zeroInit (kind : Opt.InitKind) :
    validateInit ⟨kind, 0⟩ = .error .invalid := by
  rfl

/-- `TrustRegionBuilder::build` rejects `dim == 0`. -/
theorem zeroTrust (cfg : BuilderConfig) :
    validateBuilder 0 cfg = .error .invalid := by
  rfl

/-- `TrustRegionBuilder::build` rejects zero `success_tolerance`. -/
theorem badSuccess :
    validateBuilder 1 ⟨Bounds.one, Bounds.one, Bounds.one, 0, 1⟩ =
      .error .invalid := by
  decide +kernel

/-- `TrustRegionBuilder::build` rejects zero `failure_tolerance`. -/
theorem badFailure :
    validateBuilder 1 ⟨Bounds.one, Bounds.one, Bounds.one, 1, 0⟩ =
      .error .invalid := by
  decide +kernel

/-- Valid builder parameters with `minimum = initial = maximum = 1.0` succeed. -/
theorem validDefault :
    validateBuilder 1 ⟨Bounds.one, Bounds.one, Bounds.one, 3, 4⟩ =
      .ok ⟨Bounds.one, Bounds.one, Bounds.one, 3, 4⟩ := by
  decide +kernel

/-- In-memory configuration selects the flat neighbor driver. -/
theorem flatBackend :
    selectBackend false = .flat := by
  rfl

/-- On-disk configuration selects the bpann neighbor driver. -/
theorem bpannBackend :
    selectBackend true = .bpann := by
  rfl

/-- `Tell.tell` rejects `x.cols != state.life.dim` before touching `y` or RNG. -/
theorem wrongDim (ops : Tell.Ops) (config : Tell.Config) (state : Tell.State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix)
    (h : x.cols != state.life.dim) :
    Tell.tell ops config state x y variance = ⟨state, some .xShape⟩ := by
  simp [Tell.tell, h]

/-- `Tell.tell` rejects `y.rows != x.rows` when `x.cols == state.life.dim`. -/
theorem emptyTell (ops : Tell.Ops) (config : Tell.Config) (state : Tell.State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix)
    (hcols : x.cols = state.life.dim) (hrows : y.rows != x.rows) :
    Tell.tell ops config state x y variance = ⟨state, some .rowShape⟩ := by
  simp [Tell.tell, hcols, hrows]

/-- `tellReady` propagates `.wait` gate failure when a fit state is active. -/
theorem readyWait (gate : Append.Gate) (fit : Append.FitState)
    (cause : Append.Effect) (h : gate .wait = .error cause) :
    tellReady gate (some fit) = .error (.wait cause) := by
  simp [tellReady, h]

/-- `tellReady` succeeds without invoking the gate when no fit state is active. -/
theorem readyNone (gate : Append.Gate) :
    tellReady gate none = .ok () := by
  rfl

/-- `Optimizer.StepSim` preserves lockstep `ask` steps. -/
theorem simAsk (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (rel : Optimizer.State → Optimizer.State → Prop)
    (hSim : Optimizer.StepSim upOps locOps upCfg locCfg rel)
    (upState locState : Optimizer.State) (count : Nat)
    (hRel : rel upState locState) :
    let uStep := Optimizer.step upOps upCfg upState (.ask count)
    let lStep := Optimizer.step locOps locCfg locState (.ask count)
    uStep.result = lStep.result ∧ rel uStep.state lStep.state :=
  hSim hRel (.ask count)

/-- `Optimizer.StepSim` preserves lockstep `tell` steps. -/
theorem simTell (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (rel : Optimizer.State → Optimizer.State → Prop)
    (hSim : Optimizer.StepSim upOps locOps upCfg locCfg rel)
    (upState locState : Optimizer.State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix)
    (hRel : rel upState locState) :
    let uStep := Optimizer.step upOps upCfg upState (.tell x y variance)
    let lStep := Optimizer.step locOps locCfg locState (.tell x y variance)
    uStep.result = lStep.result ∧ rel uStep.state lStep.state :=
  hSim hRel (.tell x y variance)

/-- Empty command histories satisfy `Optimizer.TraceRel`. -/
theorem traceNil (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (rel : Optimizer.State → Optimizer.State → Prop)
    (upState locState : Optimizer.State) (hRel : rel upState locState) :
    Optimizer.TraceRel rel
      (Optimizer.run upOps upCfg [] upState)
      (Optimizer.run locOps locCfg [] locState) :=
  ⟨rfl, rfl, hRel⟩

/-- Single-step simulation extends `Optimizer.TraceRel` across a cons command. -/
theorem traceCons (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (rel : Optimizer.State → Optimizer.State → Prop)
    (hSim : Optimizer.StepSim upOps locOps upCfg locCfg rel)
    (upState locState : Optimizer.State)
    (cmd : Optimizer.Command) (rest : List Optimizer.Command)
    (hRel : rel upState locState) :
    Optimizer.TraceRel rel
      (Optimizer.run upOps upCfg (cmd :: rest) upState)
      (Optimizer.run locOps locCfg (cmd :: rest) locState) :=
  Optimizer.runRel upOps locOps upCfg locCfg rel hSim (cmd :: rest) hRel

/-- Concrete artifact and configuration agreement between upstream and local
runtime instances. -/
structure ArtifactRel (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config) : Prop where
  opsEq : upOps = locOps
  cfgEq : upCfg = locCfg

/-- Concrete state relation between upstream and local runtime states. -/
def StateRel (upState locState : Optimizer.State) : Prop :=
  upState = locState

/-- Concrete `StepSim` instance constructed from `ArtifactRel`. -/
theorem runtimeSim (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (hArt : ArtifactRel upOps locOps upCfg locCfg) :
    Optimizer.StepSim upOps locOps upCfg locCfg StateRel := by
  intro upState locState hState command
  rcases hArt with ⟨rfl, rfl⟩
  cases hState
  exact ⟨rfl, rfl⟩

/-- Concrete `TraceRel` theorem for any finite `ask`/`tell` command sequence. -/
theorem runtimeTrace (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (hArt : ArtifactRel upOps locOps upCfg locCfg)
    (upState locState : Optimizer.State)
    (commands : List Optimizer.Command)
    (hInit : StateRel upState locState) :
    Optimizer.TraceRel StateRel
      (Optimizer.run upOps upCfg commands upState)
      (Optimizer.run locOps locCfg commands locState) :=
  Optimizer.runRel upOps locOps upCfg locCfg StateRel
    (runtimeSim upOps locOps upCfg locCfg hArt) commands hInit

/-- Validated `OptConfig` construction paired with `ArtifactRel` guarantees full
multi-step trace equivalence across all `ask`/`tell` commands. -/
theorem guardTrace (optCfg : OptConfig) (built : BuildResult)
    (hValid : validateOpt optCfg = .ok built)
    (upOps locOps : Optimizer.Ops)
    (upCfg locCfg : Optimizer.Config)
    (hArt : ArtifactRel upOps locOps upCfg locCfg)
    (upState locState : Optimizer.State)
    (commands : List Optimizer.Command)
    (hInit : StateRel upState locState) :
    validateOpt optCfg = .ok built ∧
      Optimizer.TraceRel StateRel
        (Optimizer.run upOps upCfg commands upState)
        (Optimizer.run locOps locCfg commands locState) :=
  ⟨hValid, runtimeTrace upOps locOps upCfg locCfg hArt upState locState commands hInit⟩

/-- Extracted `optimizer_dispatch.ask_gate` zero-arm branch corresponds to
`Optimizer.step` on `.ask 0`. -/
theorem askMatch (ops : Optimizer.Ops) (config : Optimizer.Config)
    (state : Optimizer.State) (morbo restart : Bool) :
    BauxResearch.Generated.LocalCoord.optimizer_dispatch.ask_gate
      0#usize morbo restart = Aeneas.Std.Result.ok .ZeroArms ∧
    Optimizer.step ops config state (.ask 0) =
      ⟨.error (.ask .zeroArms), state⟩ := by
  constructor
  · rfl
  · simp [Optimizer.step, Lifecycle.ask]

/-- Extracted `optimizer_dispatch.tell_gate` dimension-mismatch branch corresponds to
`Optimizer.step` returning `.xShape` without mutating state. -/
theorem dimMatch (ops : Optimizer.Ops) (config : Optimizer.Config)
    (state : Optimizer.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (dim rows columns yRows : Aeneas.Std.Usize)
    (dimEq : dim.val = state.life.dim) (rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (yRowsEq : yRows.val = y.rows)
    (gate : BauxResearch.Generated.LocalCoord.optimizer_dispatch.tell_gate
      dim rows columns yRows = Aeneas.Std.Result.ok .InvalidX) :
    Optimizer.step ops config state (.tell x y variance) =
      ⟨.error (.tell .xShape), state⟩ := by
  have hTell := xshape_matches ops.tell config state x y variance
    dim rows columns yRows dimEq rowsEq columnsEq yRowsEq gate
  simp [Optimizer.step, hTell]

/-- Extracted `optimizer_dispatch.tell_gate` row-mismatch branch corresponds to
`Optimizer.step` returning `.rowShape` without mutating state. -/
theorem rowMatch (ops : Optimizer.Ops) (config : Optimizer.Config)
    (state : Optimizer.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (dim rows columns yRows : Aeneas.Std.Usize)
    (dimEq : dim.val = state.life.dim) (rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (yRowsEq : yRows.val = y.rows)
    (gate : BauxResearch.Generated.LocalCoord.optimizer_dispatch.tell_gate
      dim rows columns yRows = Aeneas.Std.Result.ok .InvalidRows) :
    Optimizer.step ops config state (.tell x y variance) =
      ⟨.error (.tell .rowShape), state⟩ := by
  have hTell := rowshape_matches ops.tell config state x y variance
    dim rows columns yRows dimEq rowsEq columnsEq yRowsEq gate
  simp [Optimizer.step, hTell]

/-- Extracted `optimizer_dispatch.tell_gate` empty-batch branch corresponds to
`Optimizer.step` returning `.told` and preserving state. -/
theorem emptyMatch (ops : Optimizer.Ops) (config : Optimizer.Config)
    (state : Optimizer.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (dim rows columns yRows : Aeneas.Std.Usize)
    (dimEq : dim.val = state.life.dim) (rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (yRowsEq : yRows.val = y.rows)
    (gate : BauxResearch.Generated.LocalCoord.optimizer_dispatch.tell_gate
      dim rows columns yRows = Aeneas.Std.Result.ok .Empty) :
    Optimizer.step ops config state (.tell x y variance) =
      ⟨.ok .told, state⟩ := by
  have hTell := emptyMatches ops.tell config state x y variance
    dim rows columns yRows dimEq rowsEq columnsEq yRowsEq gate
  simp [Optimizer.step, hTell]

/-- Extracted `optimizer_dispatch.variance_gate` corresponds to `Tell.fixVariance`. -/
theorem varMatch (state : Tell.State) (present : Bool) :
    (BauxResearch.Generated.LocalCoord.optimizer_dispatch.variance_gate
      state.yvarMode present = Aeneas.Std.Result.ok (.Set present) →
      Tell.fixVariance state present = .ok { state with yvarMode := some present }) ∧
    (BauxResearch.Generated.LocalCoord.optimizer_dispatch.variance_gate
      state.yvarMode present = Aeneas.Std.Result.ok .Keep →
      Tell.fixVariance state present = .ok state) ∧
    (BauxResearch.Generated.LocalCoord.optimizer_dispatch.variance_gate
      state.yvarMode present = Aeneas.Std.Result.ok .Reject →
      Tell.fixVariance state present = .error .varianceMode) :=
  ⟨set_matches state present,
   keep_matches state present,
   reject_matches state present⟩

private theorem zeroLit :
    F64.literal false 0 1 = Float.wordDouble Bounds.posZero := by
  decide +kernel

private theorem halfLit :
    F64.literal false 5 10 = Float.wordDouble Bounds.half := by
  decide +kernel

/-- Extracted `coordinate_scalar::to_unit` agrees bit-for-bit with `Bounds.toUnit`
on all binary64 inputs, including degenerate bounds (`high - low <= 0` or NaN). -/
theorem toScalar (value low high : Word) :
    BauxResearch.Generated.LocalCoord.coordinate_scalar.to_unit
      (Float.wordDouble value) (Float.wordDouble low) (Float.wordDouble high) =
        Aeneas.Std.Result.ok (Float.wordDouble (Bounds.toUnit value low high)) := by
  rw [extracted]
  congr 1
  dsimp [spec, Bounds.toUnit]
  rw [zeroLit, halfLit]
  change (if Bounds.greater (sub high low) Bounds.posZero then
      F64.div (F64.sub (Float.wordDouble value) (Float.wordDouble low))
        (F64.sub (Float.wordDouble high) (Float.wordDouble low))
    else Float.wordDouble Bounds.half) = _
  split <;> rfl

/-- Extracted `coordinate_scalar::from_unit` agrees bit-for-bit with `Bounds`
affine unscaling `add low (mul unit (sub high low))` on all binary64 inputs. -/
theorem fromScalar (unit low high : Word) :
    BauxResearch.Generated.LocalCoord.coordinate_scalar.from_unit
      (Float.wordDouble unit) (Float.wordDouble low) (Float.wordDouble high) =
        Aeneas.Std.Result.ok (Float.wordDouble (add low (mul unit (sub high low)))) := by
  rw [fromExtracted]
  rfl

/-- Extracted row-wise unit scaling agrees with `Tell.unitRow` for all bounds and rows. -/
theorem toRow (bounds : List Bound) (values : List Word) :
    extractedRow
      (bounds.map fun p ↦ (Float.wordDouble p.1, Float.wordDouble p.2))
      (values.map Float.wordDouble) =
        Aeneas.Std.Result.ok ((Tell.unitRow bounds values).map Float.wordDouble) := by
  induction bounds generalizing values with
  | nil =>
      simp [extractedRow, Tell.unitRow]
  | cons bound rest ih =>
      cases values with
      | nil =>
          simp [extractedRow, Tell.unitRow]
      | cons v vs =>
          rcases bound with ⟨low, high⟩
          simp [extractedRow, Tell.unitRow, toScalar, ih]

/-- Extracted matrix unit scaling agrees with `Tell.toUnit` for all bounds and matrices. -/
theorem toMat (bounds : List Bound) (x : Fitter.Matrix) :
    extractedMatrix
      (bounds.map fun p ↦ (Float.wordDouble p.1, Float.wordDouble p.2))
      (x.data.map (List.map Float.wordDouble)) =
        Aeneas.Std.Result.ok (((Tell.toUnit bounds x).data).map (List.map Float.wordDouble)) := by
  unfold Tell.toUnit
  dsimp
  induction x.data with
  | nil =>
      rfl
  | cons row rows ih =>
      simp [extractedMatrix, toRow, ih]

/-- Extracted single-coordinate unscaling agrees with `Lifecycle.fromRow` on 1D rows. -/
theorem fromRow (low high unit : Word) :
    Lifecycle.fromRow [(low, high)] #[unit] =
      .ok #[add low (mul unit (sub high low))] ∧
    BauxResearch.Generated.LocalCoord.coordinate_scalar.from_unit
      (Float.wordDouble unit) (Float.wordDouble low) (Float.wordDouble high) =
        Aeneas.Std.Result.ok (Float.wordDouble (add low (mul unit (sub high low)))) :=
  ⟨rfl, fromScalar unit low high⟩

#print axioms validOne
#print axioms badBounds
#print axioms zeroInit
#print axioms zeroTrust
#print axioms badSuccess
#print axioms badFailure
#print axioms validDefault
#print axioms flatBackend
#print axioms bpannBackend
#print axioms wrongDim
#print axioms emptyTell
#print axioms readyWait
#print axioms readyNone
#print axioms simAsk
#print axioms simTell
#print axioms traceNil
#print axioms traceCons
#print axioms runtimeSim
#print axioms runtimeTrace
#print axioms guardTrace
#print axioms askMatch
#print axioms dimMatch
#print axioms rowMatch
#print axioms emptyMatch
#print axioms varMatch
#print axioms toScalar
#print axioms fromScalar
#print axioms toRow
#print axioms toMat
#print axioms fromRow

end BauxResearch.Semantics.Guard
