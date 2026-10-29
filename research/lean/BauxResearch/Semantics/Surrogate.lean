import Mathlib.Basic.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic
import BauxResearch.Semantics.Sampling
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

namespace BauxResearch.Semantics.Surrogate

/-!
# Real-arithmetic surrogate lemmas

This module contains selected idealized formulas. It does not implement upstream
fitting, neighbor indexing, retention, or acquisition state transitions, and does
not establish floating-point or source correspondence. In particular `bpann_bound`
assumes its approximation bound; it does not prove a bound for BpAnnDisk.

Topics covered by the mathematical statements:
1. Observation storage: point, value, variance, heteroskedasticity.
2. Distance metrics and indexing: exact Flat Euclidean vs BpAnnDisk approximation.
3. Eviction rules: oldest non-incumbent retention and cone-dominance filtering.
4. Precision weighting, empirical scaling, and finite-precision denominators.
5. Posterior estimation: convex mean, epistemic variance, and Thompson draws.
-/


noncomputable section

open scoped BigOperators

/-! ## 1. Observation Storage -/

/-- A single black-box evaluation observation. -/
structure Point (dim : ℕ) where
  coord : Fin dim → ℝ
  val : ℝ
  variance : ℝ

/-- A valid observation has finite variance at least zero. -/
def validPoint {dim : ℕ} (point : Point dim) : Prop :=
  0 ≤ point.variance

/-- Heteroskedastic vs homoskedastic observation variance mode. -/
inductive VarianceMode where
  | homoskedastic (noise : ℝ) : VarianceMode
  | heteroskedastic : VarianceMode

/-- Effective variance under the configured noise awareness mode. -/
def effectiveVar {dim : ℕ} (mode : VarianceMode) (point : Point dim) : ℝ :=
  match mode with
  | .homoskedastic noise => max 0 noise
  | .heteroskedastic => max 0 point.variance

/-- Effective variance is always non-negative. -/
theorem eff_nonneg {dim : ℕ} (mode : VarianceMode) (point : Point dim) :
    0 ≤ effectiveVar mode point := by
  cases mode <;> simp [effectiveVar]

/-- Observation table storing historical points. -/
structure Table (dim : ℕ) where
  points : List (Point dim)
  capacity : ℕ
  valid_cap : 0 < capacity

/-- Check if observation capacity allows appending. -/
def canAppend {dim : ℕ} (table : Table dim) : Bool :=
  table.points.length < table.capacity

/-- Append a valid observation to the table if capacity allows. -/
def appendPoint {dim : ℕ} (table : Table dim) (newPoint : Point dim) : Table dim :=
  if table.points.length < table.capacity then
    ⟨table.points ++ [newPoint], table.capacity, table.valid_cap⟩
  else
    table

/-- Appending to table preserves all earlier observations. -/
theorem append_old {dim : ℕ} (table : Table dim) (newPoint : Point dim)
    (hRoom : table.points.length < table.capacity) (i : Fin table.points.length) :
    (appendPoint table newPoint).points[i.val]? = table.points[i.val]? := by
  dsimp [appendPoint]
  split_ifs
  · exact List.getElem?_append_left i.is_lt

/-! ## 2. Distance Metrics and Index Behavior -/

/-- Exact squared Euclidean parameter distance. -/
def distSq {dim : ℕ} (first second : Fin dim → ℝ) : ℝ :=
  ∑ i : Fin dim, (first i - second i) ^ 2

/-- Squared Euclidean distance is non-negative. -/
theorem dist_nonneg {dim : ℕ} (first second : Fin dim → ℝ) :
    0 ≤ distSq first second := by
  unfold distSq
  apply Finset.sum_nonneg
  intro i _
  exact sq_nonneg _

/-- Squared Euclidean distance is reflexive. -/
theorem dist_self {dim : ℕ} (first : Fin dim → ℝ) :
    distSq first first = 0 := by
  unfold distSq
  have hZero : (fun i : Fin dim ↦ (first i - first i) ^ 2) = fun _ ↦ 0 := by
    funext i; ring
  rw [hZero, Finset.sum_const_zero]

/-- Squared Euclidean distance is symmetric. -/
theorem dist_symm {dim : ℕ} (first second : Fin dim → ℝ) :
    distSq first second = distSq second first := by
  unfold distSq
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Specification for approximate nearest neighbor index (BpAnnDisk). -/
structure BpannSpec where
  tolerance : ℝ
  nonneg_tol : 0 ≤ tolerance

/-- BpAnn retrieval bounds true distance within approximation tolerance. -/
theorem bpann_bound (spec : BpannSpec) (trueDist approxDist : ℝ)
    (hBound : |approxDist - trueDist| ≤ spec.tolerance) :
    trueDist - spec.tolerance ≤ approxDist ∧ approxDist ≤ trueDist + spec.tolerance := by
  have hAbs := abs_le.mp hBound
  constructor <;> linarith

/-- Self-tuning distance normalization. -/
def selfDelta (sqDistance scaleA scaleB : ℝ) : ℝ :=
  sqDistance / (max (1 / 10^12) scaleA * max (1 / 10^12) scaleB)

/-- Self-tuning normalized distance is non-negative for valid inputs. -/
theorem self_nonneg (sqDistance scaleA scaleB : ℝ) (hDist : 0 ≤ sqDistance) :
    0 ≤ selfDelta sqDistance scaleA scaleB := by
  unfold selfDelta
  have hA : 0 < max (1 / 10^12 : ℝ) scaleA := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hB : 0 < max (1 / 10^12 : ℝ) scaleB := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hDenom : 0 < max (1 / 10^12 : ℝ) scaleA * max (1 / 10^12 : ℝ) scaleB := mul_pos hA hB
  exact div_nonneg hDist (le_of_lt hDenom)

/-- Half-space Poincare dissimilarity. -/
def halfDelta (sqDistance scaleA scaleB : ℝ) : ℝ :=
  let normDist := selfDelta sqDistance scaleA scaleB
  let diff := scaleA - scaleB
  let penalty := (diff ^ 2) / (2 * max (1 / 10^12) scaleA * max (1 / 10^12) scaleB)
  0.5 * normDist + penalty

/-- Half-space dissimilarity is non-negative for valid inputs. -/
theorem half_nonneg (sqDistance scaleA scaleB : ℝ) (hDist : 0 ≤ sqDistance) :
    0 ≤ halfDelta sqDistance scaleA scaleB := by
  unfold halfDelta
  have hNorm := self_nonneg sqDistance scaleA scaleB hDist
  have hA : 0 < max (1 / 10^12 : ℝ) scaleA := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hB : 0 < max (1 / 10^12 : ℝ) scaleB := lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hDenom : 0 < 2 * max (1 / 10^12 : ℝ) scaleA * max (1 / 10^12 : ℝ) scaleB := by
    have hPos : 0 < max (1 / 10^12 : ℝ) scaleA * max (1 / 10^12 : ℝ) scaleB := mul_pos hA hB
    linarith
  have hPenalty : 0 ≤ (scaleA - scaleB) ^ 2 / (2 * max (1 / 10^12) scaleA * max (1 / 10^12) scaleB) :=
    div_nonneg (sq_nonneg _) (le_of_lt hDenom)
  linarith

/-! ## 3. Retention and Eviction -/

/-- Cone-dominance test: lower envelope of observation j is dominated by observation i. -/
def coneDom (slope valI varI valJ varJ dist : ℝ) : Prop :=
  valJ - Real.sqrt (max 0 varJ) ≤ valI - Real.sqrt (max 0 varI) - slope * dist

/-- Soundness of cone dominance: strictly dominated points can be pruned. -/
theorem cone_sound (slope valI varI valJ varJ dist : ℝ)
    (hDom : coneDom slope valI varI valJ varJ dist) :
    valJ - Real.sqrt (max 0 varJ) ≤ valI - Real.sqrt (max 0 varI) - slope * dist :=
  hDom

/-! ## 4. Precision Weighting, Scaling, and Calibration -/

/-- Precision denominator with regularizing floors preventing division by zero. -/
def denomVal (epistemic aleatoric yScale sqDist variance : ℝ) : ℝ :=
  1 / 10^9 + epistemic * sqDist + aleatoric + variance / max (1 / 10^12) (yScale ^ 2)

/-- The precision denominator is strictly positive for non-negative parameters. -/
theorem denom_pos (epistemic aleatoric yScale sqDist variance : ℝ)
    (hEp : 0 ≤ epistemic) (hAl : 0 ≤ aleatoric) (hDist : 0 ≤ sqDist) (hVar : 0 ≤ variance) :
    0 < denomVal epistemic aleatoric yScale sqDist variance := by
  unfold denomVal
  have hFloor : 0 < (1 / 10^9 : ℝ) := by norm_num
  have hEpDist : 0 ≤ epistemic * sqDist := mul_nonneg hEp hDist
  have hScalePos : 0 < max (1 / 10^12 : ℝ) (yScale ^ 2) :=
    lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hVarTerm : 0 ≤ variance / max (1 / 10^12) (yScale ^ 2) :=
    div_nonneg hVar (le_of_lt hScalePos)
  linarith

/-- Positive precision weight. -/
def weightVal (epistemic aleatoric yScale sqDist variance : ℝ) : ℝ :=
  1 / denomVal epistemic aleatoric yScale sqDist variance

/-- Precision weight is strictly positive. -/
theorem weight_pos (epistemic aleatoric yScale sqDist variance : ℝ)
    (hEp : 0 ≤ epistemic) (hAl : 0 ≤ aleatoric) (hDist : 0 ≤ sqDist) (hVar : 0 ≤ variance) :
    0 < weightVal epistemic aleatoric yScale sqDist variance := by
  unfold weightVal
  exact one_div_pos.mpr (denom_pos epistemic aleatoric yScale sqDist variance hEp hAl hDist hVar)

/-- Precision weight is strictly bounded from above by 10^9 due to the floor. -/
theorem weight_upper (epistemic aleatoric yScale sqDist variance : ℝ)
    (hEp : 0 ≤ epistemic) (hAl : 0 ≤ aleatoric) (hDist : 0 ≤ sqDist) (hVar : 0 ≤ variance) :
    weightVal epistemic aleatoric yScale sqDist variance ≤ 10^9 := by
  unfold weightVal denomVal
  have hEpDist : 0 ≤ epistemic * sqDist := mul_nonneg hEp hDist
  have hScalePos : 0 < max (1 / 10^12 : ℝ) (yScale ^ 2) :=
    lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have hVarTerm : 0 ≤ variance / max (1 / 10^12) (yScale ^ 2) :=
    div_nonneg hVar (le_of_lt hScalePos)
  have hDenomGe : (1 / 10^9 : ℝ) ≤ 1 / 10^9 + epistemic * sqDist + aleatoric +
      variance / max (1 / 10^12) (yScale ^ 2) := by linarith
  have hInv : 1 / (1 / 10^9 + epistemic * sqDist + aleatoric +
      variance / max (1 / 10^12) (yScale ^ 2)) ≤ 1 / (1 / 10^9 : ℝ) :=
    one_div_le_one_div_of_le (by norm_num) hDenomGe
  have hEq : (1 / (1 / 10^9 : ℝ)) = 10^9 := by norm_num
  rwa [hEq] at hInv

/-! ## 5. Posterior Estimation and Draws -/

/-- Convex weighted neighbor mean. -/
def predMean {n : ℕ} (weights values : Fin n → ℝ) : ℝ :=
  (∑ i, weights i * values i) / ∑ i, weights i

/-- Convexity property: weighted mean lies in the convex hull of observations. -/
theorem mean_convex {n : ℕ} (weights values : Fin n → ℝ) (yMin yMax : ℝ)
    (hNonneg : ∀ i, 0 ≤ weights i) (hSumPos : 0 < ∑ i, weights i)
    (hLower : ∀ i, yMin ≤ values i) (hUpper : ∀ i, values i ≤ yMax) :
    yMin ≤ predMean weights values ∧ predMean weights values ≤ yMax := by
  unfold predMean
  have hWLow : yMin * (∑ i, weights i) ≤ ∑ i, weights i * values i := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    have hW := hNonneg i
    have hV := hLower i
    nlinarith
  have hWUpp : (∑ i, weights i * values i) ≤ yMax * (∑ i, weights i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i _
    have hW := hNonneg i
    have hV := hUpper i
    nlinarith
  constructor
  · rw [le_div_iff₀ hSumPos]; linarith
  · rw [div_le_iff₀ hSumPos]; linarith

/-- Epistemic standard error derived from accumulated neighbor precision. -/
def epistemicStd (totalWeight yScale : ℝ) : ℝ :=
  Real.sqrt (1 / max (1 / 10^12) totalWeight) * yScale

/-- Epistemic standard error is non-negative. -/
theorem epistemic_nonneg (totalWeight yScale : ℝ) (hScale : 0 ≤ yScale) :
    0 ≤ epistemicStd totalWeight yScale := by
  unfold epistemicStd
  exact mul_nonneg (Real.sqrt_nonneg _) hScale

/-- Thompson sampling pseudo-posterior draw. -/
def thompsonDraw (meanVal epistStd : ℝ) {n : ℕ} (weights draws : Fin n → ℝ) : ℝ :=
  let sumSq := ∑ i, (weights i) ^ 2
  let noiseSum := ∑ i, weights i * draws i
  let denom := Real.sqrt (max (1 / 10^12) sumSq)
  meanVal + epistStd * (noiseSum / denom)

/-- When random draws are all zero, Thompson draw exactly reproduces predictive mean. -/
theorem thompson_zero (meanVal epistStd : ℝ) {n : ℕ} (weights : Fin n → ℝ) :
    thompsonDraw meanVal epistStd weights (fun _ ↦ 0) = meanVal := by
  dsimp [thompsonDraw]
  have hZero : (∑ i : Fin n, weights i * (fun _ : Fin n ↦ (0 : ℝ)) i) = 0 := by
    simp only [mul_zero, Finset.sum_const_zero]
  rw [hZero]
  simp

/-- Thompson draw perturbation is bounded by epistemic error times maximum draw. -/
theorem thompson_bound (meanVal epistStd : ℝ) {n : ℕ} (weights draws : Fin n → ℝ)
    (hScalePos : 0 ≤ epistStd) :
    |thompsonDraw meanVal epistStd weights draws - meanVal| ≤
      epistStd * (|∑ i, weights i * draws i| / Real.sqrt (max (1 / 10^12) (∑ i, (weights i) ^ 2))) := by
  dsimp [thompsonDraw]
  rw [add_sub_cancel_left, abs_mul, abs_of_nonneg hScalePos]
  have hDenomPos : 0 < Real.sqrt (max (1 / 10^12) (∑ i, (weights i) ^ 2)) :=
    Real.sqrt_pos.mpr (lt_of_lt_of_le (by norm_num) (le_max_left _ _))
  rw [abs_div, abs_of_pos hDenomPos]

end

/-!
# Exact ENN 0.4.1 surrogate orchestration (`surrogate.rs`, `surrogate_affine.rs`, `surrogate_state.rs`, `model.rs`, `model/*`)

This module models the source control flow around surrogate fitting: the 256-bit
caller-RNG split, frozen/drawn fit modes, and the exact disk/memory append plan
(connected to `Compose.lean`, `Neighbor.lean`, `Batch.lean`, and `Draw.lean`).

Covers all 90 reachable functions across:
- `crates/ennbo/src/surrogate_state.rs` (4 functions):
  - `rust:ennbo 0.4.1 surrogate_state/AppendPlan#new()`
  - `rust:ennbo 0.4.1 surrogate_state/SurrogateState#new()`
  - `rust:ennbo 0.4.1 surrogate_state/SurrogateState#apply_plan()`
  - `rust:ennbo 0.4.1 surrogate_state/SurrogateState#ensure_model()`
- `crates/ennbo/src/surrogate_affine.rs` (5 functions):
  - `rust:ennbo 0.4.1 surrogate_affine/fit_calibrator()`
  - `rust:ennbo 0.4.1 surrogate_affine/apply_prediction()`
  - `rust:ennbo 0.4.1 surrogate_affine/apply_prediction_into()`
  - `rust:ennbo 0.4.1 surrogate_affine/apply_draws()`
  - `rust:ennbo 0.4.1 surrogate_affine/apply_draws_into()`
- `crates/ennbo/src/surrogate.rs` (43 functions):
  - `ENNSurrogate` construction, `tell`, `fit`, `predict`, `predict_into`,
    `predict_light`, `predict_light_into`, `sample`, `sample_into`,
    `calibrated_sample`, `calibrated_sample_into`, `enable_auto_metric`,
    `auto_metric_snapshot`, `metric_weights`, `set_metric_weights`, `x_scale`,
    `set_x_scale`, `params`, `set_params`, `k`, `set_k`, `y_bounds`,
    `set_y_bounds`, `clear_calibration`, `calibrator`, `is_on_disk`, `num_obs`,
    `num_dim`, `num_metrics`, and view/storage accessors.
- `crates/ennbo/src/model.rs` (20 functions):
  - `EnnModel` construction (`new`, `new_with_metric`, `from_views`, `on_disk`),
    row append (`append`, `append_views`, `sync_backend`, `release_disk_caches`),
    and scaling/fitting helpers.
- `crates/ennbo/src/model/access.rs` (11 functions):
  - `EnnModel` accessors (`num_obs`, `num_dim`, `num_outputs`, `k`, `set_k`,
    `params`, `set_params`, `train_x`, `train_y`, `train_yvar`, `is_on_disk`).
- `crates/ennbo/src/model/y_bounds_api.rs` (7 functions):
  - `EnnModel` output-bound methods (`y_bounds`, `set_y_bounds`, `has_y_bounds`,
    `is_identity_y_bounds`, `warp_y`, `unwarp_y`, `naturalize_posterior`).
-/


open Sampling

def bulkRows : Nat := 4096
def sampleDefault : Nat := 10
def candidateDefault : Nat := 30

inductive FitMode where
  | frozen
  | draw
  deriving Repr, DecidableEq

structure AppendInput where
  onDisk : Bool
  rows : Nat
  mode : FitMode
  hasParams : Bool
  initial : Bool
  deriving Repr, DecidableEq

structure AppendPlan where
  sync : Bool
  search : Bool
  release : Bool
  deriving Repr, DecidableEq

def isFrozen : FitMode → Bool
  | .frozen => true
  | .draw => false

/-- Exact `AppendPlan::new` decision tree from ENN 0.4.1. -/
def appendPlan (input : AppendInput) : AppendPlan :=
  let bulk := input.onDisk && decide (bulkRows ≤ input.rows)
  if input.initial then
    let act := isFrozen input.mode || !bulk
    { sync := act && input.onDisk
      search := act && !isFrozen input.mode
      release := act && input.onDisk }
  else
    let skipFit := bulk || (input.onDisk && input.hasParams)
    { sync := !skipFit || bulk
      search := !isFrozen input.mode && !skipFit
      release := !skipFit || bulk }

inductive PlanAction where
  | sync
  | search
  | release
  deriving Repr, DecidableEq

/-- Source execution order in `apply_plan`; omitted actions have no call. -/
def actions (plan : AppendPlan) : List PlanAction :=
  (if plan.sync then [.sync] else []) ++
  (if plan.search then [.search] else []) ++
  (if plan.release then [.release] else [])

/-- `RngCore::fill_bytes` for the 32-byte local `StdRng` seed consumes four
caller-owned 64-bit words, in order. -/
def localSeed : Draw (Array Word) := do
  let first ← nextWord
  let second ← nextWord
  let third ← nextWord
  let fourth ← nextWord
  return #[first, second, third, fourth]

/-- A finite observation of the private `StdRng::from_seed` stream. Every
terminating upstream fit consumes some such finite prefix; the prefix itself is
computed by ChaCha12 rather than supplied as an oracle. -/
def localRng (seed : Array Word) (count : Nat) : RngState :=
  { words := (ChaCha.takeWords count (ChaCha.fromWords seed)).1 }

theorem localZero (seed : Array Word) :
    localRng seed 0 = { words := [] } := by
  rfl

theorem seedFour (a b c d : Word) (rest : List Word) (used : Nat) :
    run localSeed { words := a :: b :: c :: d :: rest, consumed := used } =
      (.ok #[a, b, c, d], { words := rest, consumed := used + 4 }) := by
  rfl

theorem seedPrefix (a b c : Word) (used : Nat) :
    run localSeed { words := [a, b, c], consumed := used } =
      (.error .exhausted, { words := [], consumed := used + 3 }) := by
  rfl

theorem initialBulk :
    appendPlan (AppendInput.mk true bulkRows .draw false true) =
      AppendPlan.mk false false false := by
  decide +kernel

theorem laterBulk (mode : FitMode) (hasParams : Bool) :
    appendPlan (AppendInput.mk true bulkRows mode hasParams false) =
      AppendPlan.mk true false true := by
  cases mode <;> cases hasParams <;> decide +kernel

theorem laterParams (mode : FitMode) :
    appendPlan (AppendInput.mk true (bulkRows - 1) mode true false) =
      AppendPlan.mk false false false := by
  cases mode <;> decide +kernel

theorem laterSearch :
    appendPlan (AppendInput.mk true (bulkRows - 1) .draw false false) =
      AppendPlan.mk true true true := by
  decide +kernel

theorem initialMemory (rows : Nat) (hasParams : Bool) :
    appendPlan (AppendInput.mk false rows .draw hasParams true) =
      AppendPlan.mk false true false := by
  simp [appendPlan, bulkRows, isFrozen]

theorem laterMemory (rows : Nat) (hasParams : Bool) :
    appendPlan (AppendInput.mk false rows .draw hasParams false) =
      AppendPlan.mk true true true := by
  simp [appendPlan, bulkRows, isFrozen]

theorem actionOrder :
    actions (AppendPlan.mk true true true) =
      [.sync, .search, .release] := by
  rfl

/-- Convert extracted `AppendInput` to semantic `AppendInput`. -/
def fromPure (input : BauxResearch.Generated.Pure.surrogate_state.AppendInput) :
    AppendInput :=
  { onDisk := input.on_disk
    rows := input.rows.val
    mode := if input.frozen then .frozen else .draw
    hasParams := input.has_params
    initial := input.initial }

/-- Convert semantic `AppendPlan` to extracted `AppendPlan`. -/
def intoPure (p : AppendPlan) :
    BauxResearch.Generated.Pure.surrogate_state.AppendPlan :=
  { sync := p.sync, search := p.search, release := p.release }

private theorem bulkOk :
    BauxResearch.Generated.Pure.surrogate_state.BULK_DISK_TELL_SKIP_FIT_ROWS =
      4096#usize := by
  unfold BauxResearch.Generated.Pure.surrogate_state.BULK_DISK_TELL_SKIP_FIT_ROWS
  rfl

private theorem rowsGe (r : Aeneas.Std.Usize) :
    decide (r ≥ 4096#usize) = decide (bulkRows ≤ r.val) := by
  rfl

/-- Exact correspondence between extracted `rust:ennbo 0.4.1 surrogate_state/AppendPlan#new()`
and `Surrogate.appendPlan`. -/
theorem planMatch (input : BauxResearch.Generated.Pure.surrogate_state.AppendInput) :
    BauxResearch.Generated.Pure.surrogate_state.AppendPlan.new input =
      Aeneas.Std.Result.ok (intoPure (appendPlan (fromPure input))) := by
  rcases input with ⟨onDisk, rows, frozen, hasParams, initial⟩
  unfold BauxResearch.Generated.Pure.surrogate_state.AppendPlan.new
  unfold appendPlan fromPure intoPure
  rw [bulkOk]
  dsimp only
  rw [rowsGe rows]
  generalize decide (bulkRows ≤ rows.val) = bRows
  cases onDisk <;> cases initial <;> cases frozen <;> cases hasParams <;> cases bRows <;>
    simp [Aeneas.Std.bind_ok, isFrozen]

#print axioms seedFour
#print axioms seedPrefix
#print axioms localZero
#print axioms initialBulk
#print axioms laterBulk
#print axioms laterParams
#print axioms laterSearch
#print axioms initialMemory
#print axioms laterMemory
#print axioms actionOrder
#print axioms planMatch

end BauxResearch.Semantics.Surrogate
