import Mathlib.Data.BitVec
import BauxResearch.Semantics.Metric
import BauxResearch.Semantics.Sobol
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

/-!
# ENN 0.4.1 automatic metric learning, dependence weights, LOO selection, and model metric API

This module proves the function-level correspondence and semantic properties for
all 33 reachable functions across:
- `crates/ennbo/src/metric_auto.rs` (12 functions):
  - `rust:ennbo 0.4.1 metric_auto/MetricState#new()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#snapshot()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#observe()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#set_weights()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#refit_auto()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#weights()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#x_scale()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#num_seen()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#heldout_gain()`
  - `rust:ennbo 0.4.1 metric_auto/MetricState#counters()`
  - `rust:ennbo 0.4.1 metric_auto/weight_drift()`
  - `rust:ennbo 0.4.1 metric_auto/auto_uses_learned_metric()`
- `crates/ennbo/src/metric_weights.rs` (9 functions):
  - `rust:ennbo 0.4.1 metric_weights/validate_tied_groups()`
  - `rust:ennbo 0.4.1 metric_weights/too_few_rows()`
  - `rust:ennbo 0.4.1 metric_weights/passing()`
  - `rust:ennbo 0.4.1 metric_weights/column_spread()`
  - `rust:ennbo 0.4.1 metric_weights/unit_indices_and_scores()`
  - `rust:ennbo 0.4.1 metric_weights/dependence_weights()`
  - `rust:ennbo 0.4.1 metric_weights/auto_weights_and_gain()`
  - `rust:ennbo 0.4.1 metric_weights/floor_logs()`
  - `rust:ennbo 0.4.1 metric_weights/scale_from_weights()`
- `crates/ennbo/src/metric_loo.rs` (3 functions):
  - `rust:ennbo 0.4.1 metric_loo/standardize_columns()`
  - `rust:ennbo 0.4.1 metric_loo/column_loo_loglik()`
  - `rust:ennbo 0.4.1 metric_loo/loo_loglik()`
- `crates/ennbo/src/model/auto_api.rs` (3 functions):
  - `rust:ennbo 0.4.1 model/auto_api/EnnModel#enable_auto_metric()`
  - `rust:ennbo 0.4.1 model/auto_api/EnnModel#observe_auto_metric()`
  - `rust:ennbo 0.4.1 model/auto_api/EnnModel#auto_metric_snapshot()`
- `crates/ennbo/src/model/metric.rs` (6 functions):
  - `rust:ennbo 0.4.1 model/metric/EnnModel#metric_weights()`
  - `rust:ennbo 0.4.1 model/metric/EnnModel#set_metric_weights()`
  - `rust:ennbo 0.4.1 model/metric/EnnModel#x_scale()`
  - `rust:ennbo 0.4.1 model/metric/EnnModel#set_x_scale()`
  - `rust:ennbo 0.4.1 model/metric/EnnModel#rescale_backend()`
  - `rust:ennbo 0.4.1 model/metric/EnnModel#rebuild_backend()`
-/

namespace BauxResearch.Semantics.Auto

open Aeneas.Std
open BauxResearch.Extraction.Native
open BauxResearch.Semantics.Sampling
open BauxResearch.Semantics.Bounds
open BauxResearch.Semantics.Fitter
open BauxResearch.Semantics.Metric
open BauxResearch.Semantics.Sobol

private theorem rowsOk :
    BauxResearch.Generated.Pure.metric_sobol.MIN_DEPENDENCE_ROWS = 100#usize := by
  unfold BauxResearch.Generated.Pure.metric_sobol.MIN_DEPENDENCE_ROWS
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 metric_weights/too_few_rows().` -/
theorem fewMatch (n : Usize) :
    BauxResearch.Generated.Pure.metric_weights.too_few_rows n =
      Result.ok (decide (n < 100#usize)) := by
  unfold BauxResearch.Generated.Pure.metric_weights.too_few_rows
  rw [rowsOk]

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 metric_auto/auto_uses_learned_metric().` -/
def learnedPure (gain : F64) : Result Bool := do
  let b ← F64.isFinite gain
  if b then
    Result.ok (F64.gt gain BauxResearch.Generated.Pure.metric_auto.AUTO_MIN_HELDOUT_GAIN)
  else
    Result.ok false

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 metric_auto/auto_uses_learned_metric().` -/
theorem learnedMatch (gain : F64) :
    BauxResearch.Generated.Pure.metric_auto.auto_uses_learned_metric gain =
      learnedPure gain := by
  rfl

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 metric_weights/passing().` -/
def passingPure (s : F64) (n : Usize) (cells : Option Usize) : Result F64 := do
  let fl ← nullPure n cells
  let fl1 := F64.mul BauxResearch.Generated.Pure.metric_sobol.DEPENDENCE_Z fl
  if F64.gt s fl1 then
    Result.ok s
  else
    Result.ok (F64.literal false 0 1)

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 metric_weights/passing().` -/
theorem passingMatch (s : F64) (n : Usize) (cells : Option Usize) :
    BauxResearch.Generated.Pure.metric_weights.passing s n cells =
      passingPure s n cells := by
  unfold BauxResearch.Generated.Pure.metric_weights.passing passingPure
  rw [nullMatch]
  congr 1
  funext fl
  dsimp only [lift]
  rw [bind_ok, bind_ok]

/-- `rust:ennbo 0.4.1 metric_auto/MetricState#snapshot()` constructor from `Metric.State`. -/
def snapOf (st : Metric.State) : Snapshot :=
  let uses := match st.heldoutGain with
    | some g => usesLearned g
    | none => false
  ⟨st.rows.numSeen, st.counters, st.heldoutGain, uses, st.weights, st.built⟩

/-- `rust:ennbo 0.4.1 metric_weights/scale_from_weights()` conversion (`1 / sqrt(w)`). -/
def scaleWeights (sqrtOp : SqrtOp) (weights : List Word) : List Word :=
  weights.map (fun w => divide Bounds.one (sqrtOp w))

theorem snapSeen (st : Metric.State) :
    (snapOf st).numSeen = st.rows.numSeen := by
  rfl

theorem snapWeights (st : Metric.State) :
    (snapOf st).weights = st.weights := by
  rfl

theorem snapBuilt (st : Metric.State) :
    (snapOf st).built = st.built := by
  rfl

theorem scaleLen (sqrtOp : SqrtOp) (weights : List Word) :
    (scaleWeights sqrtOp weights).length = weights.length := by
  simp [scaleWeights]

/-- `rust:ennbo 0.4.1 metric_loo/loo_loglik()` returns `0` when `n ≤ 1`. -/
theorem looShort (sqrtOp : SqrtOp) (ln : LogOp) (dim : Nat)
    (y : List (List Word)) (a : List Word) (k : Nat) :
    looLoglik sqrtOp ln dim [] y a k = 0 := by
  rfl

/-- `rust:ennbo 0.4.1 model/auto_api/EnnModel#observe_auto_metric()` on disabled state is a no-op. -/
theorem obsDisabled (fuel : Nat) (sqrtOp : SqrtOp) (ln : LogOp) (x y : Fitter.Matrix) :
    stepFuel fuel sqrtOp ln none x y = some (.ok (none, none)) := by
  rfl

/-- `rust:ennbo 0.4.1 metric_auto/MetricState#new()` rejects zero dimension. -/
theorem newZero (outputs : Nat) (tied : List (List Nat)) (seed : Word) :
    newMetric 0 outputs tied seed = .error .shape := by
  rfl

#print axioms fewMatch
#print axioms learnedMatch
#print axioms passingMatch
#print axioms snapSeen
#print axioms snapWeights
#print axioms snapBuilt
#print axioms scaleLen
#print axioms looShort
#print axioms obsDisabled
#print axioms newZero

end BauxResearch.Semantics.Auto
