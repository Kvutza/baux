import BauxResearch.Semantics.Compose
import BauxResearch.Semantics.Surrogate
import BauxResearch.Semantics.Metric

/-!
# ENN 0.4.1 stateful surrogate append

This module composes model construction and append, fitter `tell`, concrete
scale search, automatic metric learning (`Metric.stepFuel`), and the
sync/search/release plan.  Logical matrix, moment, scale, parameter, calibrator,
automatic-metric, and RNG mutations are computed here.  Fallible storage,
index, and metadata effects are explicit gates, because they belong to the
selected compiled backend rather than to the Rust control logic.
-/

namespace BauxResearch.Semantics.Append

open Sampling Bounds Fitter

def minX : Word := 0x3d719799812dea11
def scaleTol : Word := 0x3f847ae147ae147b
def rebuildTol : Word := 0x3fe62e42fefa39ef

inductive Effect where
  | construct
  | wait
  | append
  | scale
  | persist
  | auto
  | sync
  | release
  deriving Repr, DecidableEq

abbrev Gate := Effect → Except Effect Unit

inductive ModelError where
  | shape
  | xFinite
  | yFinite
  | varFinite
  | varNegative
  | varPresence
  | bounds (cause : BoundsError)
  | warp (cause : BatchError)
  | metric (cause : Metric.Err)
  | effect (cause : Effect)
  deriving Repr, DecidableEq

structure Model where
  batch : Batch.Model
  bounds : BoundSet
  onDisk : Bool
  metricFixed : Bool
  builtScale : List Word
  unscaled : List Nat
  auto : Option Metric.State := none
  deriving Repr, DecidableEq

structure ModelStage where
  model : Model
  error : Option ModelError
  deriving Repr, DecidableEq

def valid (matrix : Fitter.Matrix) : Bool :=
  matrix.data.length == matrix.rows &&
    matrix.data.all (fun row => row.length == matrix.cols)

def sameShape (left right : Fitter.Matrix) : Bool :=
  left.rows == right.rows && left.cols == right.cols

def varPresent (model : Model) : Bool := model.batch.yvar.isSome

/-- Validation order of `EpistemicNearestNeighbors::validate_add`. -/
def validateAdd (model : Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) :
    Option ModelError :=
  if !valid x || !valid y || x.rows != y.rows ||
      x.cols != model.batch.neighbor.trainX.cols ||
      y.cols != model.batch.neighbor.trainY.cols then some .shape
  else if anyNonfinite x.data then some .xFinite
  else if anyNonfinite y.data then some .yFinite
  else match variance with
    | some values =>
        if !valid values || !sameShape values y then some .shape
        else if anyNonfinite values.data then some .varFinite
        else if anyNegative values.data then some .varNegative
        else if model.batch.neighbor.trainX.rows > 0 && !varPresent model then
          some .varPresence
        else none
    | none =>
        if model.batch.neighbor.trainX.rows > 0 && varPresent model then
          some .varPresence
        else none

def appendMatrix (left right : Fitter.Matrix) : Fitter.Matrix :=
  ⟨left.rows + right.rows, left.cols, left.data ++ right.data⟩

def appendVar (left : Option Fitter.Matrix)
    (right : Option Fitter.Matrix) : Option Fitter.Matrix :=
  match left, right with
  | some old, some new => some (appendMatrix old new)
  | none, some new => some new
  | value, none => value

def stdMin (sqrtOp : SqrtOp) (minimum count sum sumsq : Word) : Word :=
  let mean := divide sum count
  let variance := maxZero (sub (divide sumsq count) (mul mean mean))
  let result := sqrtOp variance
  if Float.finiteDouble result && Float.ltDouble minimum result then result
  else Bounds.one

def scales (sqrtOp : SqrtOp) (minimum : Word) (matrix : Fitter.Matrix) : List Word :=
  if matrix.rows < 2 then List.replicate matrix.cols Bounds.one
  else
    let sums := sumRows matrix.cols matrix.data
    let sumsqs := sumSquares matrix.cols matrix.data
    (List.range matrix.cols).map fun column =>
      stdMin sqrtOp minimum (numWord matrix.rows)
        (sums[column]!) (sumsqs[column]!)

def forceOne (values : List Word) (indices : List Nat) : List Word :=
  (List.range values.length).map fun index =>
    if indices.contains index then Bounds.one else values[index]!

def maxNumber (left right : Word) : Word :=
  if Float.nanDouble left then right
  else if Float.nanDouble right then left
  else if Float.ltDouble left right then right else left

def absWord (value : Word) : Word :=
  (Float.absDouble (Float.wordDouble value)).bits

def maxDrift (logOp : LogOp) (left right : List Word) : Word :=
  (List.range (min left.length right.length)).foldl (fun best index =>
    maxNumber best (absWord (logOp (divide left[index]! right[index]!)))) 0

def setScale (model : Model) (scale built : List Word) : Model :=
  { model with
    batch := { model.batch with
      neighbor := { model.batch.neighbor with xScale := scale } }
    builtScale := built }

def updateScale (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Gate)
    (model : Model) : ModelStage :=
  if !model.batch.neighbor.scaleX || model.metricFixed then ⟨model, none⟩
  else
    let fresh := forceOne (scales sqrtOp minX model.batch.neighbor.trainX)
      model.unscaled
    if !model.onDisk then ⟨setScale model fresh model.builtScale, none⟩
    else if Float.leDouble
        (maxDrift logOp fresh model.batch.neighbor.xScale) scaleTol then
      ⟨model, none⟩
    else match gate .scale with
      | .error cause => ⟨model, some (.effect cause)⟩
      | .ok () =>
          let built := if Float.ltDouble rebuildTol
              (maxDrift logOp fresh model.builtScale) then fresh else model.builtScale
          ⟨setScale model fresh built, none⟩

def warpInput (logOp : LogOp) (model : Model) (y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) :
    Except ModelError (Fitter.Matrix × Option Fitter.Matrix) := do
  if identityBounds model.bounds then return (y, variance)
  let warped ← match warpBatch logOp y.data model.bounds.pairs with
    | .ok data => .ok { y with data }
    | .error cause => .error (.warp cause)
  let nextVar ← match variance with
    | none => .ok none
    | some values => match warpVar y.data values.data model.bounds.pairs with
      | .ok data => .ok (some { values with data })
      | .error cause => .error (.warp cause)
  return (warped, nextVar)

def appendData (sqrtOp : SqrtOp) (model : Model)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) : Model :=
  let trainX := appendMatrix model.batch.neighbor.trainX x
  let trainY := appendMatrix model.batch.neighbor.trainY y
  { model with batch :=
      { neighbor := { model.batch.neighbor with trainX, trainY }
        yvar := appendVar model.batch.yvar variance
        yScale := scales sqrtOp 0 trainY } }

def metricFuel : Nat := 64

def stepAuto (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Gate)
    (model : Model) (x y : Fitter.Matrix) : ModelStage :=
  match Metric.stepFuel metricFuel sqrtOp logOp model.auto x y with
  | none => ⟨model, some (.metric .shape)⟩
  | some (.error cause) => ⟨model, some (.metric cause)⟩
  | some (.ok (nextAuto, none)) => ⟨{ model with auto := nextAuto }, none⟩
  | some (.ok (nextAuto, some update)) =>
      let updated := { model with auto := nextAuto, metricFixed := true }
      let fresh := forceOne update.xScale updated.unscaled
      if updated.onDisk then
        match gate .scale with
        | .error cause => ⟨updated, some (.effect cause)⟩
        | .ok () =>
            let built := if update.rebuild then fresh else updated.builtScale
            ⟨setScale updated fresh built, none⟩
      else
        ⟨setScale updated fresh updated.builtScale, none⟩

/-- Stateful `model.add`. A failure after backend append retains the already
appended logical rows and y scale, matching Rust mutation order. -/
def addModel (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Gate)
    (model : Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) : ModelStage :=
  match validateAdd model x y variance with
  | some cause => ⟨model, some cause⟩
  | none => if x.rows = 0 then ⟨model, none⟩ else
      match warpInput logOp model y variance with
      | .error cause => ⟨model, some cause⟩
      | .ok (warped, warpedVar) =>
          match gate .wait with
          | .error cause => ⟨model, some (.effect cause)⟩
          | .ok () => match gate .append with
            | .error cause => ⟨model, some (.effect cause)⟩
            | .ok () =>
                let appended := appendData sqrtOp model x warped warpedVar
                let scaled := updateScale sqrtOp logOp gate appended
                match scaled.error with
                | some _cause => scaled
                | none => match gate .persist with
                  | .error cause => ⟨scaled.model, some (.effect cause)⟩
                  | .ok () => match gate .auto with
                    | .error cause => ⟨scaled.model, some (.effect cause)⟩
                    | .ok () => stepAuto sqrtOp logOp gate scaled.model x y

inductive FitState where
  | unfitted
  | frozen (model : Model) (params : Params)
  | searched (model : Model) (search : FitterAsk.Search)
      (fitter : State) (params : Option Params)
      (calibrator : Option FitterAsk.Calibrator)
  deriving Repr, DecidableEq

structure Config where
  k : Int
  mode : Surrogate.FitMode
  onDisk : Bool
  scaleX : Bool
  metricFixed : Bool
  bounds : BoundSet
  search : FitterAsk.Search
  fitNoise : Bool
  unscaled : List Nat := []
  autoMetric : Option (List (List Nat) × Word) := none
  deriving Repr, DecidableEq

structure Ops where
  pow : PowOp
  sqrt : SqrtOp
  log : LogOp
  powi : Fit.PowiOp
  backend : Neighbor.BackendOp
  gate : Gate
  localWords : Nat

inductive FitError where
  | seed (cause : Failure)
  | model (cause : ModelError)
  | fitter (cause : TellError)
  | search (cause : FitterAsk.AskError)
  | params (cause : ParamError)
  | effect (cause : Effect)
  deriving Repr, DecidableEq

structure Inner where
  state : FitState
  error : Option FitError
  deriving Repr, DecidableEq

def fitParams : FitState → Option Params
  | .unfitted => none
  | .frozen _ params => some params
  | .searched _ _ _ params _ => params

def fitModel : FitState → Option Model
  | .unfitted => none
  | .frozen model _ => some model
  | .searched model _ _ _ _ => some model

def makeModel (ops : Ops) (config : Config) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) : Except ModelError Model := do
  if !valid x || !valid y || x.rows != y.rows then throw .shape
  if anyNonfinite x.data then throw .xFinite
  if anyNonfinite y.data then throw .yFinite
  match variance with
  | some values =>
      if !valid values || !sameShape values y then throw .shape
      if anyNonfinite values.data then throw .varFinite
      if anyNegative values.data then throw .varNegative
  | none => pure ()
  match validateBounds config.bounds y.cols with
  | .error cause => throw (.bounds cause)
  | .ok () => pure ()
  let shell : Model :=
    { batch :=
        { neighbor :=
            { trainX := ⟨0, x.cols, []⟩
              trainY := ⟨0, y.cols, []⟩
              scaleX := config.scaleX
              xScale := List.replicate x.cols Bounds.one
              driver := if config.onDisk then .bpann else .flat }
          yvar := none
          yScale := List.replicate y.cols Bounds.one }
      bounds := config.bounds
      onDisk := config.onDisk
      metricFixed := config.metricFixed
      builtScale := List.replicate x.cols Bounds.one
      unscaled := config.unscaled
      auto := none }
  let built := addModel ops.sqrt ops.log (fun
    | .wait | .append | .persist | .auto | .scale => .ok ()
    | effect => ops.gate effect) shell x y variance
  match built.error with
  | some cause => throw cause
  | none =>
      let initial := if config.scaleX && !config.metricFixed then
        let fresh := forceOne (scales ops.sqrt minX built.model.batch.neighbor.trainX)
          config.unscaled
        setScale built.model fresh fresh
      else built.model
      match ops.gate .construct with
      | .error cause => throw (.effect cause)
      | .ok () => match ops.gate .auto with
        | .error cause => throw (.effect cause)
        | .ok () => match config.autoMetric with
          | none => return initial
          | some (tied, seed) =>
              match Metric.stepEnable x.cols y.cols tied seed with
              | .error cause => throw (.metric cause)
              | .ok autoState =>
                  let withAuto := { initial with auto := some autoState }
                  let stepped := stepAuto ops.sqrt ops.log ops.gate withAuto x y
                  match stepped.error with
                  | some cause => throw cause
                  | none => return stepped.model

def applyPlan (ops : Ops) (config : Config) (plan : Surrogate.AppendPlan)
    (state : FitState) : Draw Inner := do
  if plan.sync then match ops.gate .sync with
    | .error cause => return ⟨state, some (.effect cause)⟩
    | .ok () => pure ()
  let searched ← match state with
    | .searched model search fitter warm _ =>
        if plan.search then
          let outcome ← Compose.ask ops.pow ops.sqrt ops.log ops.powi ops.backend
            fitter model.batch search warm config.fitNoise
          match outcome.result with
          | .error cause =>
              return ⟨.searched model search outcome.state
                outcome.state.params outcome.state.calibrator, some (.search cause)⟩
          | .ok params =>
              pure (.searched model search outcome.state (some params)
                outcome.state.calibrator)
        else pure state
    | _ => pure state
  if plan.release then match ops.gate .release with
    | .error cause => return ⟨searched, some (.effect cause)⟩
    | .ok () => pure ()
  return ⟨searched, none⟩

def startModel (ops : Ops) (config : Config) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (warm : Option Params) : Draw Inner := do
  let model ← match makeModel ops config x y variance with
    | .error cause => return ⟨.unfitted, some (.model cause)⟩
    | .ok model => pure model
  let state ← match config.mode with
    | .frozen => match newParams config.k Bounds.one 0 with
      | .error cause => return ⟨.unfitted, some (.params cause)⟩
      | .ok params => pure (.frozen model params)
    | .draw =>
        let fitter := { blank with k := config.k }
        let told := tell ops.log fitter x y variance (some config.bounds)
        match told.error with
        | some cause => return ⟨.unfitted, some (.fitter cause)⟩
        | none => pure (.searched model config.search told.state warm none)
  let plan := Surrogate.appendPlan
    ⟨config.onDisk, x.rows, config.mode, false, true⟩
  applyPlan ops config plan state

def appendInner (ops : Ops) (config : Config) (state : FitState)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) : Draw Inner :=
  match state with
  | .unfitted => startModel ops config x y variance none
  | .frozen model params => do
      let added := addModel ops.sqrt ops.log ops.gate model x y variance
      match added.error with
      | some cause => return ⟨.frozen added.model params, some (.model cause)⟩
      | none =>
          let next := .frozen added.model params
          applyPlan ops config
            (Surrogate.appendPlan
              ⟨config.onDisk, x.rows, config.mode, true, false⟩) next
  | .searched model search fitter params calibrator => do
      let added := addModel ops.sqrt ops.log ops.gate model x y variance
      match added.error with
      | some cause =>
          return ⟨.searched added.model search fitter params calibrator,
            some (.model cause)⟩
      | none =>
          let told := tell ops.log fitter x y variance (some config.bounds)
          let next := .searched added.model search told.state
            told.state.params told.state.calibrator
          match told.error with
          | some cause => return ⟨next, some (.fitter cause)⟩
          | none =>
              let plan := Surrogate.appendPlan
                ⟨config.onDisk, x.rows, config.mode, params.isSome, false⟩
              applyPlan ops config plan next

structure Step where
  state : FitState
  rng : RngState
  privateRng : Option RngState
  error : Option FitError
  deriving Repr, DecidableEq

/-- Public `fit_append`: four caller words seed a private ChaCha12 stream. The
caller RNG is committed before any model error, while private RNG state is
retained as evidence of scale-search draw order. -/
def append (ops : Ops) (config : Config) (state : FitState) (rng : RngState)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) : Step :=
  let seeded := run Surrogate.localSeed rng
  match seeded.1 with
  | .error cause => ⟨state, seeded.2, none, some (.seed cause)⟩
  | .ok seed =>
      let privateRng := Surrogate.localRng seed ops.localWords
      let result := run (appendInner ops config state x y variance) privateRng
      match result.1 with
      | .error cause => ⟨state, seeded.2, some result.2, some (.seed cause)⟩
      | .ok inner => ⟨inner.state, seeded.2, some result.2, inner.error⟩

def openBound : BoundSet := ⟨1, 2, [(negInf, posInf)]⟩

def pass : Gate := fun _ => .ok ()

def testConfig : Config :=
  ⟨1, .frozen, false, false, false, openBound,
    ⟨1, 1, false, false⟩, false, [], none⟩

def testOps : Ops :=
  ⟨fun _ _ => Bounds.one, fun _ => Bounds.one, fun value => value,
    fun _ _ => Bounds.one, Batch.noBackend, pass, 0⟩

theorem validEmpty (cols : Nat) :
    valid ⟨0, cols, []⟩ = true := by
  simp [valid]

theorem autoDisabled (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Gate)
    (model : Model) (x y : Fitter.Matrix)
    (hauto : model.auto = none) :
    stepAuto sqrtOp logOp gate model x y = ⟨model, none⟩ := by
  cases model
  simp at hauto
  subst hauto
  rfl

#print axioms validEmpty
#print axioms autoDisabled

end BauxResearch.Semantics.Append
