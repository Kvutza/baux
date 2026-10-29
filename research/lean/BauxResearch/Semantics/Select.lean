import BauxResearch.Semantics.Compose
import BauxResearch.Semantics.Acq
import BauxResearch.Semantics.Pareto
import BauxResearch.Semantics.Morbo

/-!
# ENN 0.4.1 surrogate-backed arm selection

This module composes the concrete neighbor posterior, output naturalization,
optional affine calibration, and the Random/UCB/Pareto branches of
`strategy::select_arms`. The compiled neighbor backend and native `sqrt`/`exp`
operations remain explicit typed inputs.
-/

namespace BauxResearch.Semantics.Select

open Sampling Bounds Fitter

inductive SelectError where
  | posterior
  | acquisition
  | rows
  | scalar
  deriving Repr, DecidableEq

structure Context where
  model : Batch.Model
  params : Params
  bounds : List Bound
  calibrator : Option Calibration.Calibrator
  deriving Repr, DecidableEq

structure Prediction where
  mu : Batch
  se : Batch
  deriving Repr, DecidableEq

structure Morbo where
  low : Row
  high : Row
  weights : Row
  alpha : Word
  metrics : Nat
  deriving Repr, DecidableEq

inductive Mode where
  | random
  | ucb (beta : Word) (morbo : Option Morbo)
  | pareto
  | thompson (morbo : Option Morbo)
  deriving Repr, DecidableEq

def matrix (values : Batch) : Fitter.Matrix :=
  let cols := values.head?.map List.length |>.getD 0
  ⟨values.length, cols, values⟩

def queryMatrix (values : Sampling.Matrix) : Fitter.Matrix :=
  let rows := values.toList.map Array.toList
  let cols := values[0]?.map Array.size |>.getD 0
  ⟨values.size, cols, rows⟩

/-- `ENNSurrogate::predict`: posterior in warped storage units, conversion to
natural output units, then optional affine calibration. -/
def predict (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Context)
    (queries : Fitter.Matrix) : Except SelectError Prediction := do
  let stats ← (Batch.one sqrtOp backend ctx.model queries ctx.params false false).mapError
    (fun _ => .posterior)
  let natural ← (Bounds.naturalize expOp ctx.bounds stats.mu stats.se
    stats.seEpi stats.seAle).mapError (fun _ => .posterior)
  match ctx.calibrator with
  | none => return ⟨natural.mu, natural.se⟩
  | some calibrator =>
      let result ← (Calibration.applyPred calibrator
        (matrix natural.mu) (matrix natural.se) ctx.bounds).mapError
          (fun _ => .posterior)
      return ⟨result.1.data, result.2.data⟩

def pick (candidates : Sampling.Matrix) (indices : List Nat) :
    Except SelectError Sampling.Matrix :=
  (Opt.selectRows candidates indices.toArray).mapError (fun _ => .rows)

def firstCol : Batch → Except SelectError (List Word)
  | [] => .ok []
  | row :: rows => match row, firstCol rows with
    | value :: _, .ok rest => .ok (value :: rest)
    | _, _ => .error .posterior

def ucbRow (beta : Word) : List Word → List Word →
    Except SelectError (List Word)
  | [], [] => .ok []
  | mean :: means, std :: stds =>
      match ucbRow beta means stds with
      | .ok rest => .ok (Acq.scoreUcb mean std beta :: rest)
      | .error cause => .error cause
  | _, _ => .error .posterior

def ucbRows (beta : Word) : Batch → Batch → Except SelectError Batch
  | [], [] => .ok []
  | mean :: means, std :: stds =>
      match ucbRow beta mean std, ucbRows beta means stds with
      | .ok row, .ok rows => .ok (row :: rows)
      | .error cause, _ => .error cause
      | _, .error cause => .error cause
  | _, _ => .error .posterior

def dense (values : Batch) (cols : Nat) : Morbo.DenseMatrix :=
  ⟨values.map (fun row => row.toArray) |>.toArray, cols⟩

def randomPick (candidates : Sampling.Matrix) (arms : Nat) :
    Draw (Except SelectError Sampling.Matrix) := do
  match ← Acq.randomSelect candidates.size arms with
  | .error _ => return .error .acquisition
  | .ok indices => return pick candidates indices

def plainUcb (candidates : Sampling.Matrix) (arms : Nat) (beta : Word)
    (prediction : Prediction) : Draw (Except SelectError Sampling.Matrix) := do
  match firstCol prediction.mu, firstCol prediction.se with
  | .ok mu, .ok se =>
      match ← Acq.ucbSelect mu se beta arms with
      | .error _ => return .error .acquisition
      | .ok indices => return pick candidates indices
  | _, _ => return .error .posterior

def morboUcb (candidates : Sampling.Matrix) (arms : Nat) (beta : Word)
    (config : Morbo) (prediction : Prediction) :
    Draw (Except SelectError Sampling.Matrix) := do
  match ucbRows beta prediction.mu prediction.se with
  | .error cause => return .error cause
  | .ok values =>
      match Morbo.scalarize (dense values config.metrics) config.low config.high
          config.weights config.alpha config.metrics false with
      | .error _ => return .error .scalar
      | .ok scores =>
          let indices ← Acq.shuffledRank scores.toList
          return pick candidates (indices.take arms)

def ucbPick (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Context)
    (candidates : Sampling.Matrix) (arms : Nat) (beta : Word)
    (morbo : Option Morbo) : Draw (Except SelectError Sampling.Matrix) := do
  match predict sqrtOp expOp backend ctx (queryMatrix candidates) with
  | .error cause => return .error cause
  | .ok prediction => match morbo with
    | none => plainUcb candidates arms beta prediction
    | some config => morboUcb candidates arms beta config prediction

def paretoPick (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Context)
    (candidates : Sampling.Matrix) (arms : Nat) :
    Draw (Except SelectError Sampling.Matrix) := do
  match predict sqrtOp expOp backend ctx (queryMatrix candidates) with
  | .error cause => return .error cause
  | .ok prediction =>
      match ← Pareto.select prediction.mu prediction.se arms with
      | .error _ => return .error .acquisition
      | .ok indices => return pick candidates indices

/-- Source dispatch for every selection branch whose complete posterior path is
currently represented. Missing surrogates take the source random fallback.
Thompson remains a named error until function-draw hashing is composed. -/
def select (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Option Context)
    (mode : Mode) (candidates : Sampling.Matrix) (arms : Nat) :
    Draw (Except SelectError Sampling.Matrix) :=
  match mode with
  | .random => randomPick candidates arms
  | .ucb beta morbo => match ctx with
    | none => randomPick candidates arms
    | some value => ucbPick sqrtOp expOp backend value candidates arms beta morbo
  | .pareto => match ctx with
    | none => randomPick candidates arms
    | some value => paretoPick sqrtOp expOp backend value candidates arms
  | .thompson _ => pure (.error .acquisition)

theorem randomDispatch (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Option Context)
    (candidates : Sampling.Matrix) (arms : Nat) :
    select sqrtOp expOp backend ctx .random candidates arms =
      randomPick candidates arms := by
  rfl

theorem absentUcb (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (candidates : Sampling.Matrix)
    (arms : Nat) (beta : Word) (morbo : Option Morbo) :
    select sqrtOp expOp backend none (.ucb beta morbo) candidates arms =
      randomPick candidates arms := by
  rfl

#print axioms randomDispatch
#print axioms absentUcb

end BauxResearch.Semantics.Select
