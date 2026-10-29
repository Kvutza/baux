import BauxResearch.Semantics.PrivateRng
import BauxResearch.Semantics.Lifecycle
import BauxResearch.Semantics.Predict

/-!
# ENN 0.4.1 public tell lifecycle

The transition below follows `Optimizer::tell`, `Strategy::tell`, and
`tell_common` in source order.  It retains mutations made before later errors:
the first nonempty call fixes y-variance presence, init progress advances before
fitting, the incumbent tracker advances before output-width and surrogate
checks, model appends may survive fitter/search failures, and RNG consumption is
committed on every strategy return.
-/

namespace BauxResearch.Semantics.Tell

open Sampling Bounds

structure Ops where
  /-- Total private ChaCha12 fitter operations.  The public tell path must not
  choose a finite observation prefix for an otherwise unbounded private
  stream. -/
  append : PrivateRng.Ops
  exp : Bounds.ExpOp

structure Config where
  append : Append.Config
  noiseAware : Bool
  metrics : Nat
  alpha : Word
  deriving Repr, DecidableEq

structure State where
  life : Lifecycle.State
  fit : Option Append.FitState
  tracker : Incumbent.State
  obsX : Fitter.Matrix
  obsY : Fitter.Matrix
  yvarMode : Option Bool
  incumbentX : Option (List Word)
  incumbentY : Option (List Word)
  multi : Option Trust.Multi
  yMin : Option Row
  yMax : Option Row
  deriving Repr, DecidableEq

inductive TellError where
  | xShape
  | rowShape
  | varianceMode
  | wait (cause : Append.Effect)
  | metricWidth
  | tracker (cause : Failure)
  | append (cause : Append.FitError)
  | incumbent (cause : Failure)
  | trust (cause : Trust.Error)
  | restart (cause : Failure)
  deriving Repr, DecidableEq

structure Step where
  state : State
  error : Option TellError
  deriving Repr, DecidableEq

def appendMatrix (left right : Fitter.Matrix) : Fitter.Matrix :=
  ⟨left.rows + right.rows, left.cols, left.data ++ right.data⟩

def unitRow : List Bound → List Word → List Word
  | [], _ => []
  | _, [] => []
  | (low, high) :: bounds, value :: values =>
      Bounds.toUnit value low high :: unitRow bounds values

def toUnit (bounds : List Bound) (x : Fitter.Matrix) : Fitter.Matrix :=
  { x with data := x.data.map (unitRow bounds) }

def modelRows : Append.FitState → Nat
  | .unfitted => 0
  | .frozen model _ => model.batch.neighbor.trainX.rows
  | .searched model _ _ _ _ => model.batch.neighbor.trainX.rows

def modelMetrics : Append.FitState → Option Nat
  | .unfitted => none
  | .frozen model _ => some model.batch.neighbor.trainY.cols
  | .searched model _ _ _ _ => some model.batch.neighbor.trainY.cols

def fixVariance (state : State) (present : Bool) : Except TellError State :=
  match state.yvarMode with
  | none => .ok { state with yvarMode := some present }
  | some expected => if expected = present then .ok state else .error .varianceMode

def bumpInit (state : State) (rows : Nat) : State :=
  match state.life.strategy with
  | .init kind done budget =>
      { state with life := { state.life with
          strategy := .init kind (done + rows) budget } }
  | .hybrid kind done budget active =>
      if active then { state with life := { state.life with
        strategy := .hybrid kind (done + rows) budget active } }
      else state
  | .turbo => state

def finishInit (state : State) : State :=
  match state.life.strategy with
  | .hybrid kind done budget true =>
      if budget ≤ done then { state with life := { state.life with
        strategy := .hybrid kind done budget false } }
      else state
  | _ => state

def consumeReinit (state : State) (rows : Nat) : State :=
  match state.life.reinitLeft with
  | none => state
  | some left => { state with life := { state.life with
      reinitLeft := if left ≤ rows then none else some (left - rows) } }

def tellTracker (state : State) (y : Fitter.Matrix) : Except TellError State :=
  match Incumbent.tellRows state.tracker state.obsX.rows y.data with
  | .error cause => .error (.tracker cause)
  | .ok tracker => .ok { state with tracker }

def rangeRow (low high : Row) (row : List Word) : Row × Row :=
  let count := min row.length (min low.size high.size)
  ((List.range count).map (fun i =>
      if Float.ltDouble row[i]! low[i]! then row[i]! else low[i]!) |>.toArray,
   (List.range count).map (fun i =>
      if Float.ltDouble high[i]! row[i]! then row[i]! else high[i]!) |>.toArray)

def addRanges (state : State) (y : Fitter.Matrix) : State :=
  if !Lifecycle.isMorbo state.life || y.rows = 0 then state
  else
    let initialLow := state.yMin.getD (Array.replicate y.cols posInf)
    let initialHigh := state.yMax.getD (Array.replicate y.cols negInf)
    let ranges := y.data.foldl (fun current row =>
      rangeRow current.1 current.2 row) (initialLow, initialHigh)
    { state with yMin := some ranges.1, yMax := some ranges.2 }

def syncObs (before : Nat) (state : State) (x y : Fitter.Matrix) : State :=
  match state.fit with
  | none => { state with
      obsX := appendMatrix state.obsX x
      obsY := appendMatrix state.obsY y }
  | some fit =>
      if before < modelRows fit then { state with
        obsX := appendMatrix state.obsX x
        obsY := appendMatrix state.obsY y }
      else state

def choose (ops : Ops) (config : Config) (state : State) :
    Except TellError (State × RngState) := do
  let candidates := Incumbent.ask state.tracker
  if candidates.isEmpty then
    return ({ state with incumbentX := none, incumbentY := none }, state.life.rng)
  else if Lifecycle.isMorbo state.life then
    let predicted ← if config.noiseAware then
      match state.fit with
      | none => pure none
      | some fit =>
          let rows ← match Incumbent.rowsAt state.obsX.data candidates with
            | .error cause => throw (.incumbent cause)
            | .ok rows => pure rows
          match Predict.mu ops.append.sqrt ops.exp ops.append.backend
              config.append.k fit ⟨rows.length, state.obsX.cols, rows⟩ with
          | .error _ => throw (.incumbent .invalid)
          | .ok matrix => pure (some matrix.data)
    else pure none
    let weights := state.life.morboWeight.map (·.weights) |>.getD #[]
    let selected := Incumbent.chooseMorbo state.obsX.data state.obsY.data
      candidates predicted weights config.alpha config.metrics state.life.rng
    match selected.1 with
    | .error cause => throw (.incumbent cause)
    | .ok choice => return ({ state with
        incumbentX := some choice.x, incumbentY := some choice.y }, selected.2)
  else
    match Incumbent.chooseRaw state.obsX.data state.obsY.data candidates with
    | .error cause => throw (.incumbent cause)
    | .ok choice => return ({ state with
        incumbentX := some choice.x, incumbentY := some choice.y }, state.life.rng)

def firstColumn : List (List Word) → Except TellError (List Word)
  | [] => .ok []
  | row :: rows => do
      let some value := row[0]? | throw (.incumbent .panic)
      let rest ← firstColumn rows
      return value :: rest

def morboTrust (config : Config) (state : State) : Except TellError State := do
  let some multi := state.multi | return state
  let some incumbent := state.incumbentY | throw (.incumbent .invalid)
  let some low := state.yMin | throw (.incumbent .invalid)
  let some high := state.yMax | throw (.incumbent .invalid)
  let weights := state.life.morboWeight.map (·.weights) |>.getD #[]
  let dense : Morbo.DenseMatrix :=
    ⟨#[incumbent.toArray], config.metrics⟩
  let scores ← match Morbo.scalarize dense low high weights
      config.alpha config.metrics true with
    | .error cause => throw (.incumbent cause)
    | .ok scores => pure scores
  let some score := scores[0]? | throw (.incumbent .panic)
  let updated := Trust.pushScore multi state.obsX.rows score
  match updated.error with
  | some cause => throw (.trust cause)
  | none =>
      let staged := { state with multi := some updated.state }
      return { staged with life := { state.life with trust := updated.state.inner } }

def ordinaryTrust (state : State) (y : Fitter.Matrix) : Except TellError State := do
  let some incumbent := state.incumbentY | throw (.incumbent .invalid)
  let some value := incumbent[0]? | throw (.incumbent .panic)
  let values ← firstColumn y.data
  let updated := Trust.updateBatch state.life.trust values state.obsX.rows value
  match updated.error with
  | some cause => throw (.trust cause)
  | none => return { state with life := { state.life with trust := updated.state } }

def tellCommon (ops : Ops) (config : Config) (state : State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) : Step :=
  let tracked := tellTracker state y
  match tracked with
  | .error cause => ⟨state, some cause⟩
  | .ok tracked =>
      match tracked.fit with
      | some fit => match modelMetrics fit with
        | some metrics => if metrics != y.cols then ⟨tracked, some .metricWidth⟩ else
            let before := modelRows fit
            let fitted := PrivateRng.append ops.append config.append fit
              tracked.life.rng x y variance
            let staged := { tracked with fit := some fitted.state }
            let next := { staged with life := { tracked.life with rng := fitted.rng } }
            let stored := syncObs before next x y
            match fitted.error with
            | some cause => ⟨stored, some (.append cause)⟩
            | none =>
                let ranged := addRanges stored y
                match choose ops config ranged with
                | .error cause => ⟨ranged, some cause⟩
                | .ok (chosen, rng) =>
                    let chosen := { chosen with life := { chosen.life with rng } }
                    if Lifecycle.isMorbo chosen.life then
                      match morboTrust config chosen with
                      | .error cause => ⟨chosen, some cause⟩
                      | .ok final => ⟨final, none⟩
                    else ⟨chosen, none⟩
        | none => ⟨tracked, some .metricWidth⟩
      | none =>
          let stored := addRanges (syncObs tracked.obsX.rows tracked x y) y
          match choose ops config stored with
          | .error cause => ⟨stored, some cause⟩
          | .ok (chosen, rng) =>
              let chosen := { chosen with life := { chosen.life with rng } }
              if Lifecycle.isMorbo chosen.life then
                match morboTrust config chosen with
                | .error cause => ⟨chosen, some cause⟩
                | .ok final => ⟨final, none⟩
              else ⟨chosen, none⟩

def inTurbo (state : State) : Bool :=
  match state.life.reinitLeft, state.life.strategy with
  | some _, _ => false
  | none, .turbo => true
  | none, .hybrid _ _ _ false => true
  | _, _ => false

def restartMorbo (ops : Ops) (state : State) : Step :=
  let resetMulti := state.multi.map Trust.restartMulti
  let resetTrust := resetMulti.map (·.inner) |>.getD state.life.trust
  let cleared := { state with
    multi := resetMulti
    yMin := none
    yMax := none
    life := { state.life with
      trust := resetTrust
      restartGen := state.life.restartGen + 1 } }
  match cleared.life.morboWeight with
  | some weights => match weights.mode with
    | .onPropose => ⟨addRanges cleared cleared.obsY, none⟩
    | .onRestart =>
        let sampled := run
          (Morbo.sampleWeights ops.append.log weights.count) cleared.life.rng
        match sampled.1 with
        | .error cause =>
            let failed := { cleared with life := { cleared.life with rng := sampled.2 } }
            ⟨failed, some (.restart cause)⟩
        | .ok nextWeights =>
            let next := { weights with weights := nextWeights }
            let synced := Lifecycle.syncWeights cleared.life next
            let ready := { cleared with life := { synced with rng := sampled.2 } }
            ⟨addRanges ready ready.obsY, none⟩
  | none => ⟨cleared, none⟩

/-- Public `Optimizer::tell` through the complete successful strategy path and
all source-visible validation and state-retaining failures represented above. -/
def tell (ops : Ops) (config : Config) (state : State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) : Step :=
  if x.cols != state.life.dim then ⟨state, some .xShape⟩
  else if y.rows != x.rows then ⟨state, some .rowShape⟩
  else if x.rows = 0 then ⟨state, none⟩
  else match fixVariance state variance.isSome with
    | .error cause => ⟨state, some cause⟩
    | .ok fixed =>
        let unit := toUnit fixed.life.bounds x
        match fixed.fit with
        | some _ => match ops.append.gate .wait with
          | .error cause => ⟨fixed, some (.wait cause)⟩
          | .ok () => dispatch unit fixed
        | none => dispatch unit fixed
where
  dispatch (unit : Fitter.Matrix) (fixed : State) : Step :=
    let isLocal := fixed.life.reinitLeft.isSome
    let turbo := inTurbo fixed
    let staged := if isLocal then fixed else bumpInit fixed x.rows
    let common := tellCommon ops config staged unit y variance
    match common.error with
    | some _cause => common
    | none =>
        let progressed := if isLocal then consumeReinit common.state x.rows
          else finishInit common.state
        if turbo then
          if Lifecycle.isMorbo progressed.life then
            if Float.ltDouble progressed.life.trust.length
                progressed.life.trust.config.minimum then
              restartMorbo ops progressed
            else ⟨progressed, none⟩
          else match ordinaryTrust progressed y with
            | .error cause => ⟨progressed, some cause⟩
            | .ok final => ⟨final, none⟩
        else ⟨progressed, none⟩

theorem emptyInert (ops : Ops) (config : Config) (state : State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix)
    (hcols : x.cols = state.life.dim) (hrows : y.rows = x.rows)
    (hempty : x.rows = 0) :
    tell ops config state x y variance = ⟨state, none⟩ := by
  simp [tell, hcols, hrows, hempty]

#print axioms emptyInert

end BauxResearch.Semantics.Tell
