import BauxResearch.Semantics.Fit
import BauxResearch.Semantics.Index

/-!
# ENN 0.4.1 fitter ask semantics

This module composes the two rand index-sampling sites, backend row gathering,
posterior scoring, Rust total-order selection, and failure-visible commit order
of `ENNFitter::ask`. The posterior and affine-calibration calls are explicit
operations because their concrete model/index semantics are separate modules;
their inputs, outputs, errors, and RNG effects are not assumed away.

The outer fitter sample gathers exactly `min(p, model.len)` rows. Therefore the
inner `subsample_loglik` site sees `p_actual = n` and takes its all-indices branch
without consuming another random value on this call path.
-/

namespace BauxResearch.Semantics.FitterAsk

open Sampling Fitter

structure Search where
  samples : Nat
  candidates : Nat
  inferAleatoric : Bool
  affine : Bool
  deriving Repr, DecidableEq

structure Model where
  obs : Nat
  dim : Nat
  x : Fitter.Matrix
  y : Fitter.Matrix
  deriving Repr, DecidableEq

/-- One posterior batch per parameter candidate. -/
structure Post where
  mu : List Bounds.Batch
  se : List Bounds.Batch
  deriving Repr, DecidableEq

abbrev Calibrator := List (List Word)

inductive AskError where
  | dimension
  | param (cause : ParamError)
  | needTell
  | rows
  | fitInput
  | postShape
  | scaleShape
  | fit (cause : Fit.FitError)
  | calibrate
  deriving Repr, DecidableEq

/-- The model is captured by the operation, as it is by the Rust method call. -/
abbrev PostOp := Fitter.Matrix → List Params → Except AskError Post

/-- Calibration may consume the same mutable RNG used by the preceding fit. -/
abbrev CalOp := Params → Nat → Draw (Except AskError Calibrator)

structure Outcome where
  state : State
  result : Except AskError Params
  deriving Repr, DecidableEq

def pickData : Bounds.Batch → List Nat → Except AskError Bounds.Batch
  | _, [] => .ok []
  | rows, index :: rest =>
      match rows[index]?, pickData rows rest with
      | some row, .ok tail => .ok (row :: tail)
      | _, _ => .error .rows

def pickMatrix (matrix : Fitter.Matrix) (indices : List Nat) :
    Except AskError Fitter.Matrix := do
  let data ← pickData matrix.data indices
  return ⟨indices.length, matrix.cols, data⟩

def trainRows (model : Model) (indices : List Nat) :
    Except AskError (Fitter.Matrix × Fitter.Matrix) := do
  let x ← pickMatrix model.x indices
  let y ← pickMatrix model.y indices
  return (x, y)

def allIndices (count : Nat) : List Nat := List.range count

/-- Both fitter sampling sites use this branch: no draw when all rows are used. -/
def chooseRows (length amount : Nat) : Draw (List Nat) :=
  if amount = length then pure (allIndices length)
  else Index.sample length amount

def safeStd (value : Word) : Word :=
  if Float.finiteDouble value && Float.ltDouble 0 value then value
  else Bounds.one

def safeStds : List Word → List Word
  | [] => []
  | value :: rest => safeStd value :: safeStds rest

def scaleRow : List Word → List Word → Except AskError (List Word)
  | [], _ => .ok []
  | value :: values, std :: stds => do
      let rest ← scaleRow values stds
      return divide value std :: rest
  | _, _ => .error .scaleShape

def scaleBatch : Bounds.Batch → List Word → Except AskError Bounds.Batch
  | [], _ => .ok []
  | row :: rows, stds => do
      let first ← scaleRow row stds
      let rest ← scaleBatch rows stds
      return first :: rest

def shapeBatch (rows cols : Nat) (values : Bounds.Batch) : Bool :=
  values.length == rows && values.all (fun row => row.length == cols)

def shapePost (params rows cols : Nat) (values : List Bounds.Batch) : Bool :=
  values.length == params && values.all (shapeBatch rows cols)

def scorePosts (logOp : Bounds.LogOp) (powiOp : Fit.PowiOp)
    (y : Bounds.Batch) : List Bounds.Batch → List Bounds.Batch →
      Except AskError (List Word)
  | [], [] => .ok []
  | mu :: mus, se :: ses =>
      match Fit.singleLoglik logOp powiOp y mu se,
          scorePosts logOp powiOp y mus ses with
      | .ok value, .ok rest => .ok (value :: rest)
      | .error cause, _ => .error (.fit cause)
      | _, .error cause => .error cause
  | _, _ => .error .postShape

/-- Executable `subsample_loglik`, with the posterior computation supplied as
an explicit model operation. Its random row sample is the second fitter draw. -/
def subsample (logOp : Bounds.LogOp) (powiOp : Fit.PowiOp) (postOp : PostOp)
    (modelObs : Nat) (x y : Fitter.Matrix) (params : List Params) (p : Nat)
    (stds : List Word) : Draw (Except AskError (List Word)) := do
  if x.rows != y.rows || p = 0 || params.isEmpty then return .error .fitInput
  if x.rows = 0 || modelObs ≤ 1 then return .ok (List.replicate params.length 0)
  if anyNonfinite y.data then
    return .ok (List.replicate params.length Bounds.negInf)
  let amount := min p x.rows
  let indices ← chooseRows x.rows amount
  match pickMatrix x indices, pickMatrix y indices with
  | .ok xSel, .ok ySel =>
      match postOp xSel params with
      | .error cause => return .error cause
      | .ok post =>
          if !shapePost params.length amount y.cols post.mu ||
              !shapePost params.length amount y.cols post.se then
            return .error .postShape
          let safe := safeStds stds
          match scaleBatch ySel.data safe with
          | .error cause => return .error cause
          | .ok yScaled =>
              let rec scaleMany : List Bounds.Batch →
                  Except AskError (List Bounds.Batch)
                | [] => .ok []
                | batch :: rest => do
                    let first ← scaleBatch batch safe
                    let tail ← scaleMany rest
                    return first :: tail
              match scaleMany post.mu, scaleMany post.se with
              | .ok mu, .ok se => return scorePosts logOp powiOp yScaled mu se
              | .error cause, _ => return .error cause
              | _, .error cause => return .error cause
  | .error cause, _ => return .error cause
  | _, .error cause => return .error cause

def commit (calOp : CalOp) (state : State) (best : Params)
    (samples : Nat) (affine : Bool) : Draw Outcome := do
  let stored := { state with params := some best }
  if affine then
    match ← calOp best samples with
    | .ok cal => return ⟨{ stored with calibrator := some cal }, .ok best⟩
    | .error cause => return ⟨stored, .error cause⟩
  else return ⟨{ stored with calibrator := none }, .ok best⟩

def addWarm (state : State) (search : Search) (warm : Option Params)
    (params : List Params) : Except AskError (List Params) :=
  match warm with
  | none =>
      match state.params with
      | none => .ok params
      | some value =>
          match newParams state.k value.epistemic
              (if search.inferAleatoric then value.aleatoric else 0) with
          | .ok next => .ok (params ++ [next])
          | .error cause => .error (.param cause)
  | some value =>
      match newParams state.k value.epistemic
          (if search.inferAleatoric then value.aleatoric else 0) with
      | .ok next => .ok (params ++ [next])
      | .error cause => .error (.param cause)

def fail (state : State) (cause : AskError) : Outcome :=
  ⟨state, .error cause⟩

/-- Complete source-order control semantics of `ENNFitter::ask` around the
explicit posterior and calibrator operations. -/
def ask (powOp : PowOp) (sqrtOp : SqrtOp) (logOp : Bounds.LogOp)
    (powiOp : Fit.PowiOp) (postOp : PostOp) (calOp : CalOp)
    (state : State) (model : Model) (search : Search)
    (warm : Option Params) : Draw Outcome := do
  match state.numDim with
  | some dim => if dim != model.dim then return fail state .dimension
  | none => pure ()
  if model.obs < 2 then
    match newParams state.k Bounds.one 0 with
    | .error cause => return fail state (.param cause)
    | .ok best => return (← commit calOp state best search.samples search.affine)
  if state.yCount = 0 then return fail state .needTell
  match ← randomParams powOp state.k search.candidates search.inferAleatoric with
  | .error cause => return fail state (.param cause)
  | .ok random =>
      match addWarm state search warm random with
      | .error cause => return fail state cause
      | .ok params =>
          let amount := min search.samples model.obs
          let indices ← chooseRows model.obs amount
          match trainRows model indices with
          | .error cause => return fail state cause
          | .ok (trainX, trainY) =>
              let stds := yStd sqrtOp state
              match ← subsample logOp powiOp postOp model.obs trainX trainY
                  params search.samples stds with
              | .error cause => return fail state cause
              | .ok scores =>
                  match params[Fit.bestIndex scores]? with
                  | none => return fail state .fitInput
                  | some best => return (← commit calOp state best search.samples search.affine)

def noCal : CalOp := fun _ _ => pure (.ok [])

def unitPost : PostOp := fun x params =>
  let row := List.replicate x.cols Bounds.one
  let batch := List.replicate x.rows row
  .ok ⟨List.replicate params.length batch, List.replicate params.length batch⟩

def testState : State :=
  { blank with
    numDim := some 1
    yCount := 2
    numMetrics := 1
    ySum := [0]
    ySumsq := [0] }

def testModel : Model :=
  ⟨2, 1, ⟨2, 1, [[0], [Bounds.one]]⟩,
    ⟨2, 1, [[Bounds.one], [Bounds.one]]⟩⟩

theorem allRows (state : RngState) :
    run (chooseRows 3 3) state = (.ok [0, 1, 2], state) := by
  rfl

theorem dimFirst (powOp : PowOp) (sqrtOp : SqrtOp)
    (logOp : Bounds.LogOp)
    (powiOp : Fit.PowiOp) (postOp : PostOp) (calOp : CalOp)
    (state : State) (model : Model) (search : Search) (warm : Option Params)
    (hd : state.numDim = some (model.dim + 1)) :
    (run (ask powOp sqrtOp logOp powiOp postOp calOp state model search warm)
      { words := [] }).1 = .ok (fail state .dimension) := by
  simp [ask, hd, run]
  rfl

theorem earlyDefault :
    (run (ask (fun _ _ => 0) (fun _ => Bounds.one) (fun _ => 0)
      (fun _ _ => 0) unitPost noCal testState { testModel with obs := 1 }
      ⟨1, 0, false, false⟩ none) { words := [] }).1 =
      .ok ⟨{ testState with
        params := some ⟨1, Bounds.one, 0⟩
        calibrator := none }, .ok ⟨1, Bounds.one, 0⟩⟩ := by
  decide +kernel

theorem calStores (best : Params) (state : State) (samples : Nat)
    (calOp : CalOp)
    (h : calOp best samples = pure (.error .calibrate)) :
    commit calOp state best samples true =
      pure ⟨{ state with params := some best }, .error .calibrate⟩ := by
  simp [commit, h]

#print axioms allRows
#print axioms dimFirst
#print axioms earlyDefault
#print axioms calStores

end BauxResearch.Semantics.FitterAsk
