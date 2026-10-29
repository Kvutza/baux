import BauxResearch.Semantics.Batch
import BauxResearch.Semantics.Calibration

/-!
# ENN 0.4.1 fitter composition

This module removes the abstract posterior and calibrator operations from the
fitter ask transition. The remaining parameters are native numerical functions
and the selected compiled neighbor backend.
-/

namespace BauxResearch.Semantics.Compose

open Sampling Bounds Fitter

def fitterModel (model : Batch.Model) : FitterAsk.Model :=
  ⟨model.neighbor.trainX.rows, model.neighbor.trainX.cols,
    model.neighbor.trainX, model.neighbor.trainY⟩

def calPost (sqrtOp : Calibration.SqrtOp)
    (backend : Neighbor.BackendOp) (model : Batch.Model) :
    Calibration.PostOp := fun queries params noise =>
  match Batch.one sqrtOp backend model queries params true noise with
  | .error _ => .error .posterior
  | .ok stats =>
      .ok (⟨queries.rows, model.neighbor.trainY.cols, stats.mu⟩,
        ⟨queries.rows, model.neighbor.trainY.cols, stats.se⟩)

def calOp (sqrtOp : Calibration.SqrtOp)
    (backend : Neighbor.BackendOp) (model : Batch.Model)
    (noise : Bool) : FitterAsk.CalOp :=
  Calibration.askCal sqrtOp (calPost sqrtOp backend model)
    model.neighbor.trainX model.neighbor.trainY noise

/-- Concrete fitter `ask` through candidate construction, neighbor refinement,
weighted posterior arithmetic, likelihood selection, and affine calibration.
The backend operation and target numerical functions are the remaining typed
implementation boundaries. -/
def ask (powOp : PowOp) (sqrtOp : SqrtOp) (logOp : Bounds.LogOp)
    (powiOp : Fit.PowiOp) (backend : Neighbor.BackendOp)
    (state : State) (model : Batch.Model) (search : FitterAsk.Search)
    (warm : Option Params) (fitNoise : Bool) : Draw FitterAsk.Outcome :=
  FitterAsk.ask powOp sqrtOp logOp powiOp
    (Batch.fitPost sqrtOp backend model)
    (calOp sqrtOp backend model fitNoise)
    state (fitterModel model) search warm

theorem askDef (powOp : PowOp) (sqrtOp : SqrtOp)
    (logOp : Bounds.LogOp) (powiOp : Fit.PowiOp)
    (backend : Neighbor.BackendOp) (state : State)
    (model : Batch.Model) (search : FitterAsk.Search)
    (warm : Option Params) (fitNoise : Bool) :
    ask powOp sqrtOp logOp powiOp backend state model search warm fitNoise =
      FitterAsk.ask powOp sqrtOp logOp powiOp
        (Batch.fitPost sqrtOp backend model)
        (calOp sqrtOp backend model fitNoise)
        state (fitterModel model) search warm := by
  rfl

#print axioms askDef

end BauxResearch.Semantics.Compose
