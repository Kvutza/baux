import BauxResearch.Semantics.Append

/-!
# ENN 0.4.1 surrogate prediction

This is the concrete `ENNSurrogate::predict` path: obtain the fitted model,
use fitted parameters or the configured default, compute a public posterior
with default flags, naturalize its four statistics, and apply the optional
affine calibrator to the natural-unit mean and standard error.
-/

namespace BauxResearch.Semantics.Predict

open Bounds Fitter

inductive Error where
  | unfitted
  | params (cause : ParamError)
  | posterior (cause : Batch.ApiError)
  | natural (cause : BatchError)
  | calibrator
  deriving Repr, DecidableEq

def readCal : FitterAsk.Calibrator → Except Error Calibration.Calibrator
  | [a, b, c] => .ok ⟨a, b, c⟩
  | _ => .error .calibrator

def chooseParams (k : Int) (state : Append.FitState) :
    Except Error Params :=
  match Append.fitParams state with
  | some params => .ok params
  | none => match newParams k Bounds.one 0 with
    | .ok params => .ok params
    | .error cause => .error (.params cause)

def runPost (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (model : Append.Model)
    (params : Params) (queries : Matrix) :
    Except Error (Matrix × Matrix) := do
  let stats ← match Batch.one sqrtOp backend model.batch queries params
      false false with
    | .ok stats => .ok stats
    | .error cause => .error (.posterior cause)
  let natural ← match naturalize expOp model.bounds.pairs stats.mu stats.se
      stats.seEpi stats.seAle with
    | .ok value => .ok value
    | .error cause => .error (.natural cause)
  return (⟨queries.rows, model.batch.neighbor.trainY.cols, natural.mu⟩,
    ⟨queries.rows, model.batch.neighbor.trainY.cols, natural.se⟩)

/-- Exact successful control path of `ENNSurrogate::predict`. -/
def predict (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (k : Int)
    (state : Append.FitState) (queries : Matrix) :
    Except Error (Matrix × Matrix) := do
  let model ← match Append.fitModel state with
    | some model => .ok model
    | none => .error .unfitted
  let params ← chooseParams k state
  let pair ← runPost sqrtOp expOp backend model params queries
  let calibrator := match state with
    | .searched _ _ _ _ value => value
    | _ => none
  match calibrator with
  | none => return pair
  | some rows =>
      let cal ← readCal rows
      match Calibration.applyPred cal pair.1 pair.2 model.bounds.pairs with
      | .ok value => return value
      | .error _ => throw .calibrator

def mu (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (k : Int)
    (state : Append.FitState) (queries : Matrix) : Except Error Matrix := do
  return (← predict sqrtOp expOp backend k state queries).1

theorem noModel (sqrtOp : Batch.SqrtOp) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (k : Int) (queries : Matrix) :
    predict sqrtOp expOp backend k .unfitted queries = .error .unfitted := by
  rfl

#print axioms noModel

end BauxResearch.Semantics.Predict
