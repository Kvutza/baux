import BauxResearch.Forward.Algebra
import BauxResearch.Objective.Response
import Mathlib.Data.List.OfFn

/-! Real-arithmetic semantics of `optimizer/src/reliability/neighbor.rs`.
Distances are supplied dissimilarities, not assumed to be a Mathlib metric.
Reported variances are statistics, not assumed calibrated population variances.
Fit proposals and held-out indices are explicit inputs; their RNG law is open. -/
/-!
For each selected neighbor `i`, the source weight is the reciprocal of its
declared denominator: epistemic distance, aleatoric term, stored variance, and
the fixed positive floor all remain visible.  The source mean is
`(∑ᵢ wᵢ yᵢ) / max(10^-12, ∑ᵢ wᵢ)`, not an automatically unbiased posterior
mean.  The dissimilarities and reported variances are data, not metric or
calibration theorems.
-/
namespace BauxResearch.Optimization.Neighbor
noncomputable section
open scoped BigOperators
open BauxResearch.Forward

structure Parameters where
  neighbors : ℕ
  epistemic : ℝ
  aleatoric : ℝ

structure Data (capacity : ℕ) where
  values : Vector capacity
  variances : Vector capacity
  distances : Fin capacity → Vector capacity
  count : ℕ
  base : Fin capacity

def defaults (neighbors : ℕ) : Parameters := ⟨neighbors, 1 / 20, 1 / 20⟩

def scale {capacity : ℕ} (data : Data capacity) : ℝ :=
  let active := Finset.univ.filter fun index : Fin capacity => index.val < data.count
  let mean := (∑ index ∈ active, data.values index) / data.count
  let variance := max 0 ((∑ index ∈ active, data.values index ^ 2) / data.count - mean ^ 2)
  if 0 < Real.sqrt variance then Real.sqrt variance else 1

def noise {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (index : Fin capacity) : ℝ :=
  params.aleatoric + data.variances index / max (1 / 10^12) (yScale^2)

def denominator {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (index : Fin capacity) : ℝ :=
  1 / 10^9 + params.epistemic * distances index + noise data params yScale index

def weight {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (index : Fin capacity) : ℝ :=
  1 / denominator data params yScale distances index

def ordered {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (distances : Vector capacity) (excluded : Option (Fin capacity)) : List (Fin capacity) :=
  (((List.ofFn fun index : Fin capacity => index).filter fun index =>
    decide (index.val < data.count ∧ some index ≠ excluded)).mergeSort fun left right =>
      decide (distances left < distances right ∨
        (distances left = distances right ∧ left.val ≤ right.val))).take params.neighbors

structure Prefix where
  total : ℝ
  noise : ℝ
  seen : ℕ
  chosen : ℕ
  best : Option ℝ

def prefixCount {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℕ :=
  let final := nearest.foldl (fun state index =>
    let precision := 1 / max (1 / 10^12) (denominator data params yScale distances index)
    let total := state.total + precision
    let noiseSum := state.noise + precision^2 * noise data params yScale index
    let risk := params.epistemic * distances index + Real.sqrt (noiseSum / max (1 / 10^24) (total^2))
    let better := state.best.isNone || state.best.any (fun previous => decide (risk < previous))
    { total := total, noise := noiseSum, seen := state.seen + 1
      chosen := if better then state.seen + 1 else state.chosen
      best := if better then some risk else state.best })
    ({ total := 0, noise := 0, seen := 0, chosen := 1, best := none } : Prefix)
  final.chosen

def mean {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℝ :=
  let total := (nearest.map (weight data params yScale distances)).sum
  (nearest.map fun index => weight data params yScale distances index * data.values index).sum /
    max (1 / 10^12) total

structure Prediction where
  mean : ℝ
  standardError : ℝ

def posterior {capacity : ℕ} (positive : 0 < capacity) (data : Data capacity)
    (params : Parameters) (yScale : ℝ) (distances : Vector capacity)
    (excluded : Option (Fin capacity)) (adaptive : Bool) : Prediction :=
  let nearest := ordered data params distances excluded
  if nearest.isEmpty then
    { mean := data.values (excluded.getD ⟨0, positive⟩)
      standardError := Real.sqrt (max (1 / 10^12) params.aleatoric) * yScale }
  else
    let chosen := if adaptive then nearest.take (prefixCount data params yScale distances nearest)
      else nearest
    let total := max (1 / 10^12) (chosen.map (weight data params yScale distances)).sum
    let aleatoric := (chosen.map fun index =>
      weight data params yScale distances index / total * noise data params yScale index).sum
    { mean := mean data params yScale distances chosen
      standardError := Real.sqrt (max (1 / 10^12) (1 / total + aleatoric)) * yScale }

def likelihood {capacity : ℕ} (positive : 0 < capacity) (data : Data capacity)
    (params : Parameters) (yScale : ℝ) (held : List (Fin capacity)) : ℝ :=
  (held.map (fun index =>
    let prediction := posterior positive data params yScale (data.distances index) (some index) false
    let deviation := prediction.standardError / yScale
    let residual := (data.values index / yScale - prediction.mean / yScale) / deviation
    (-(1 / 2) * Real.log (2 * Real.pi) - Real.log deviation - (1 / 2) * residual^2))).sum

/-- Source ties use >=, so the last maximizing parameter proposal wins. -/
def fit {capacity : ℕ} (positive : 0 < capacity) (data : Data capacity)
    (previous : Parameters) (proposals : List (ℝ × ℝ)) (held : List (Fin capacity)) : Parameters :=
  let candidates := proposals.map (fun pair =>
    ({ neighbors := previous.neighbors, epistemic := pair.1, aleatoric := pair.2 } : Parameters))
  let yScale := scale data
  (candidates ++ [previous]).foldl (fun best candidate =>
    if likelihood positive data best yScale held ≤ likelihood positive data candidate yScale held
      then candidate else best) previous

def compact {capacity : ℕ} (positive : 0 < capacity) (data : Data capacity) : Data capacity where
  values := fun index => if index.val = 0 then data.values data.base else 0
  variances := fun index => if index.val = 0 then data.variances data.base else 0
  distances := fun _ _ => 0
  count := 1
  base := ⟨0, positive⟩

def append {capacity : ℕ} (data : Data capacity) (row : Vector capacity)
    (value variance : ℝ) (accepted : Bool) : Option (Data capacity) :=
  if room : data.count < capacity then
    let next : Fin capacity := ⟨data.count, room⟩
    some {
      values := Function.update data.values next value
      variances := Function.update data.variances next variance
      distances := fun left right => if left = next ∧ right.val < data.count then row right else
        if right = next ∧ left.val < data.count then row left else data.distances left right
      count := data.count + 1
      base := if accepted then next else data.base }
  else none

theorem Weight.positive {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (index : Fin capacity)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variance : 0 ≤ data.variances index) (distance : 0 ≤ distances index) :
    0 < weight data params yScale distances index := by
  have scalePositive : 0 < max (1 / 10^12 : ℝ) (yScale^2) :=
    lt_of_lt_of_le (by norm_num) (le_max_left _ _)
  have noisePositive : 0 ≤ noise data params yScale index :=
    add_nonneg aleatoric (div_nonneg variance scalePositive.le)
  unfold weight denominator
  apply one_div_pos.mpr
  nlinarith [mul_nonneg epistemic distance]

theorem Compact.baseline {capacity : ℕ} (positive : 0 < capacity) (data : Data capacity) :
    (compact positive data).values (compact positive data).base = data.values data.base ∧
    (compact positive data).variances (compact positive data).base = data.variances data.base ∧
    (compact positive data).count = 1 := by
  simp [compact]

/-- The mathematical neighbor mean is Mathlib's finite center of mass. Position
indices preserve the operational list's multiplicities without a nodup assumption. -/
def libraryMean {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℝ :=
  Finset.univ.centerMass
    (fun position : Fin nearest.length => weight data params yScale distances nearest[position.val])
    (fun position : Fin nearest.length => data.values nearest[position.val])

/-- The denominator floor has a barycenter interpretation too: its extra mass
belongs to a zero-valued anchor. This is a view, not a new stored observation. -/
def anchoredMean {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℝ :=
  let total := (nearest.map (weight data params yScale distances)).sum
  Finset.univ.centerMass
    (fun index : Option (Fin nearest.length) => match index with
      | none => max (1 / 10^12) total - total
      | some position => weight data params yScale distances nearest[position.val])
    (fun index : Option (Fin nearest.length) => match index with
      | none => 0
      | some position => data.values nearest[position.val])

theorem Mean.anchored {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) :
    mean data params yScale distances nearest = anchoredMean data params yScale distances nearest := by
  dsimp only [mean, anchoredMean]
  simp only [Finset.centerMass, smul_eq_mul, Fintype.sum_option, mul_zero, zero_add,
    ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  rw [List.ofFn_getElem_eq_map nearest (fun index =>
    weight data params yScale distances index * data.values index)]
  rw [sub_add_cancel, div_eq_mul_inv, mul_comm]

/-- The denominator floor must be inactive to inherit ordinary barycenter laws. -/
theorem Mean.library {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (inactive : (1 / 10^12 : ℝ) ≤ (nearest.map (weight data params yScale distances)).sum) :
    mean data params yScale distances nearest = libraryMean data params yScale distances nearest := by
  dsimp only [mean, libraryMean]
  rw [max_eq_right inactive]
  simp only [Finset.centerMass, smul_eq_mul, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  rw [div_eq_mul_inv, mul_comm]
  rw [List.ofFn_getElem_eq_map nearest (fun index =>
    weight data params yScale distances index * data.values index)]

/-- Membership in any convex set is inherited from Mathlib, not re-proved for ENN. -/
theorem Mean.convex {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (set : Set ℝ) (convex : Convex ℝ set)
    (nonnegative : ∀ position : Fin nearest.length,
      0 ≤ weight data params yScale distances nearest[position.val])
    (positive : 0 < ∑ position : Fin nearest.length,
      weight data params yScale distances nearest[position.val])
    (members : ∀ position : Fin nearest.length, data.values nearest[position.val] ∈ set) :
    libraryMean data params yScale distances nearest ∈ set :=
  convex.centerMass_mem (fun position _ => nonnegative position) positive
    (fun position _ => members position)

/-- Every convex response functional inherits Jensen's bound from Mathlib. -/
theorem Mean.jensen {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (functional : ℝ → ℝ) (convex : ConvexOn ℝ Set.univ functional)
    (nonnegative : ∀ position : Fin nearest.length,
      0 ≤ weight data params yScale distances nearest[position.val])
    (positive : 0 < ∑ position : Fin nearest.length,
      weight data params yScale distances nearest[position.val]) :
    functional (libraryMean data params yScale distances nearest) ≤
      Finset.univ.centerMass
        (fun position : Fin nearest.length => weight data params yScale distances nearest[position.val])
        (fun position : Fin nearest.length => functional (data.values nearest[position.val])) :=
  convex.map_centerMass_le (fun position _ => nonnegative position) positive (by simp)

/-- Composable ENN uncertainty laws (Section 6.4 of
`TYPED_TENSOR_INFERENCE_AND_OPTIMIZER_ARCHITECTURE.txt`, Obligation P1). -/
inductive UncertaintyLaw
  | deployedLegacy
  | exactDiagonalPlusDistance
  | deterministicDistanceOnly
  deriving DecidableEq

/-- Exact independent-noise variance contribution `∑ (w_i / W)^2 * ν_i` for
normalized neighbor weights `α_i = w_i / W`. -/
def exactNoise {k : ℕ} (alpha nu : Fin k → ℝ) : ℝ :=
  ∑ i : Fin k, (alpha i) ^ 2 * nu i

/-- Legacy linear noise contribution `∑ (w_i / W) * ν_i` used in the deployed
`NeighborValueEstimator` standard-error formula. -/
def legacyNoise {k : ℕ} (alpha nu : Fin k → ℝ) : ℝ :=
  ∑ i : Fin k, alpha i * nu i

/-- Epistemic distance-weighted variance contribution `σ_epi * ∑ α_i * d_i`. -/
def epistemicTerm {k : ℕ} (epistemic : ℝ) (alpha dist : Fin k → ℝ) : ℝ :=
  epistemic * ∑ i : Fin k, alpha i * dist i

/-- Variance decomposition under each `UncertaintyLaw`. -/
def lawVariance {k : ℕ} (law : UncertaintyLaw) (total epistemic : ℝ)
    (alpha nu dist : Fin k → ℝ) : ℝ :=
  match law with
  | .deployedLegacy => 1 / total + legacyNoise alpha nu
  | .exactDiagonalPlusDistance =>
      exactNoise alpha nu + epistemicTerm epistemic alpha dist
  | .deterministicDistanceOnly =>
      epistemicTerm epistemic alpha dist

/-- Obligation P1(a): On a deterministic corpus (`ν_i = 0`), the exact
diagonal-plus-distance uncertainty law reduces identically to the pure
epistemic distance law. -/
theorem zeroNoise {k : ℕ} (total epistemic : ℝ)
    (alpha dist : Fin k → ℝ) :
    lawVariance .exactDiagonalPlusDistance total epistemic alpha (fun _ => 0) dist =
      lawVariance .deterministicDistanceOnly total epistemic alpha (fun _ => 0) dist := by
  simp [lawVariance, exactNoise]

/-- Obligation P1(b): Whenever normalized weights satisfy `0 ≤ α_i ≤ 1` and
noise variances satisfy `0 ≤ ν_i`, the exact independent-noise variance is
bounded above by the legacy linear noise term `∑ α_i^2 ν_i ≤ ∑ α_i ν_i`. -/
theorem exactLe {k : ℕ} (alpha nu : Fin k → ℝ)
    (alphaNonneg : ∀ i, 0 ≤ alpha i)
    (alphaLeOne : ∀ i, alpha i ≤ 1)
    (nuNonneg : ∀ i, 0 ≤ nu i) :
    exactNoise alpha nu ≤ legacyNoise alpha nu := by
  unfold exactNoise legacyNoise
  apply Finset.sum_le_sum
  intro i _
  have sqLe : (alpha i) ^ 2 ≤ alpha i := by
    nlinarith [alphaNonneg i, alphaLeOne i]
  exact mul_le_mul_of_nonneg_right sqLe (nuNonneg i)

/-- Obligation P1(c): Under uniform neighbor weights `α_i = 1 / k` (`k > 0`) and
constant observation noise variance `ν`, the exact diagonal noise variance
shrinks as `ν / k` (`k * exactNoise = ν`), whereas the legacy linear noise
term remains `ν` regardless of neighbor count `k`. -/
theorem uniformNoise {k : ℕ} (hk : 0 < k) (nu : ℝ) :
    exactNoise (fun _ : Fin k => (1 : ℝ) / k) (fun _ => nu) = nu / k ∧
    legacyNoise (fun _ : Fin k => (1 : ℝ) / k) (fun _ => nu) = nu := by
  have hkReal : (k : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (ne_of_gt hk)
  constructor
  · simp only [exactNoise, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    field_simp [hkReal]
  · simp only [legacyNoise, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    field_simp [hkReal]

/-- Upper half-space Poincaré dissimilarity `δ_ℍ((x, z), (x', z')) = (‖x - x'‖² + (z - z')²) / (2 z z')`
from Kisfaludi-Bak & van Wordragen (`arXiv:2305.01356`), unifying horizontal
Euclidean distance `sqDist = ‖x - x'‖²` and vertical scale `z, z' > 0`. -/
def halfDelta (sqDist z z' : ℝ) : ℝ :=
  (sqDist + (z - z') ^ 2) / (2 * z * z')

theorem HalfDelta.nonneg (sqDist z z' : ℝ) (hDist : 0 ≤ sqDist)
    (hz : 0 < z) (hz' : 0 < z') :
    0 ≤ halfDelta sqDist z z' := by
  unfold halfDelta
  apply div_nonneg
  · positivity
  · positivity

theorem HalfDelta.self (z : ℝ) :
    halfDelta 0 z z = 0 := by
  unfold halfDelta
  ring

theorem HalfDelta.horosphere (sqDist z : ℝ) :
    halfDelta sqDist z z = sqDist / (2 * z ^ 2) := by
  unfold halfDelta
  ring

theorem HalfDelta.isometry (sqDist z z' c : ℝ) (hc : c ≠ 0) :
    halfDelta (c ^ 2 * sqDist) (c * z) (c * z') = halfDelta sqDist z z' := by
  unfold halfDelta
  have hc2 : c ^ 2 ≠ 0 := pow_ne_zero 2 hc
  calc
    (c ^ 2 * sqDist + (c * z - c * z') ^ 2) / (2 * (c * z) * (c * z'))
      = (c ^ 2 * (sqDist + (z - z') ^ 2)) / (c ^ 2 * (2 * z * z')) := by ring
    _ = (sqDist + (z - z') ^ 2) / (2 * z * z') :=
        mul_div_mul_left (sqDist + (z - z') ^ 2) (2 * z * z') hc2

/-- Böröczky binary-tiling cell (`arXiv:2305.01356`, Section 2): a cell at vertical
scale `s = 2^ℓ > 0` has horizontal width `s` across `d` coordinates and vertical
span `[s, 2s]`, so its corner-to-corner squared Euclidean displacement is `d * s^2`
and its hyperbolic dissimilarity `halfDelta (d * s^2) s (2 * s)` is `(d + 1) / 4`,
strictly independent of scale `s`. -/
theorem HalfDelta.tile (d s : ℝ) (hs : s ≠ 0) :
    halfDelta (d * s ^ 2) s (2 * s) = (d + 1) / 4 := by
  unfold halfDelta
  have hs2 : s ^ 2 ≠ 0 := pow_ne_zero 2 hs
  calc
    (d * s ^ 2 + (s - 2 * s) ^ 2) / (2 * s * (2 * s))
      = (s ^ 2 * (d + 1)) / (s ^ 2 * 4) := by ring
    _ = (d + 1) / 4 := mul_div_mul_left (d + 1) 4 hs2

/-- Zelnik-Manor & Perona self-tuning distance (`resident_search.metal` `DistanceScaling::SelfTuning`):
normalizes squared Euclidean distance `sqDist` by the geometric mean of local k-NN scales `z * z'`. -/
def selfTuning (sqDist z z' : ℝ) : ℝ :=
  sqDist / (z * z')

/-- Nonnegative hyperbolic vertical scale-mismatch penalty `(z - z')^2 / (2 * z * z')`. -/
def scalePenalty (z z' : ℝ) : ℝ :=
  (z - z') ^ 2 / (2 * z * z')

/-- Exact decomposition linking `DistanceScaling::SelfTuning` to upper-half-space hyperbolic ENN:
`halfDelta` equals half the self-tuning distance plus the vertical scale-mismatch penalty. -/
theorem HalfDelta.decomp (sqDist z z' : ℝ) :
    halfDelta sqDist z z' = selfTuning sqDist z z' / 2 + scalePenalty z z' := by
  unfold halfDelta selfTuning scalePenalty
  ring

theorem HalfDelta.dominates (sqDist z z' : ℝ) (hz : 0 < z) (hz' : 0 < z') :
    selfTuning sqDist z z' / 2 ≤ halfDelta sqDist z z' := by
  rw [HalfDelta.decomp]
  have hPen : 0 ≤ scalePenalty z z' := by
    unfold scalePenalty
    positivity
  linarith

/-- In the ENN posterior, replacing `selfTuning / 2` with the hyperbolic half-space dissimilarity
`halfDelta` strictly downweights cross-scale neighbors (`z ≠ scales index`) while preserving
horosphere neighbors (`z = scales index`). -/
theorem HalfDelta.weightLe {capacity : ℕ} (data : Data capacity) (params : Parameters)
    (yScale : ℝ) (sqDist scales : Vector capacity) (z : ℝ) (index : Fin capacity)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variance : 0 ≤ data.variances index) (distNonneg : 0 ≤ sqDist index)
    (hz : 0 < z) (hScale : 0 < scales index) :
    weight data params yScale (fun i => halfDelta (sqDist i) z (scales i)) index ≤
      weight data params yScale (fun i => selfTuning (sqDist i) z (scales i) / 2) index := by
  have hStNonneg : 0 ≤ (fun i => selfTuning (sqDist i) z (scales i) / 2) index := by
    dsimp
    unfold selfTuning
    positivity
  have hDenomPos : 0 < denominator data params yScale
      (fun i => selfTuning (sqDist i) z (scales i) / 2) index := by
    have hW := Weight.positive data params yScale
      (fun i => selfTuning (sqDist i) z (scales i) / 2) index
      epistemic aleatoric variance hStNonneg
    unfold weight at hW
    exact one_div_pos.mp hW
  have hDom : selfTuning (sqDist index) z (scales index) / 2 ≤
      halfDelta (sqDist index) z (scales index) :=
    HalfDelta.dominates (sqDist index) z (scales index) hz hScale
  have hDenomLe : denominator data params yScale
      (fun i => selfTuning (sqDist i) z (scales i) / 2) index ≤
      denominator data params yScale
      (fun i => halfDelta (sqDist i) z (scales i)) index := by
    unfold denominator
    dsimp
    nlinarith [mul_le_mul_of_nonneg_left hDom epistemic]
  unfold weight
  exact one_div_le_one_div_of_le hDenomPos hDenomLe

#print axioms Weight.positive
#print axioms Compact.baseline
#print axioms Mean.library
#print axioms Mean.anchored
#print axioms Mean.convex
#print axioms Mean.jensen
#print axioms zeroNoise
#print axioms exactLe
#print axioms uniformNoise
#print axioms HalfDelta.nonneg
#print axioms HalfDelta.self
#print axioms HalfDelta.horosphere
#print axioms HalfDelta.isometry
#print axioms HalfDelta.tile
#print axioms HalfDelta.decomp
#print axioms HalfDelta.dominates
#print axioms HalfDelta.weightLe
end
end BauxResearch.Optimization.Neighbor
