import BauxResearch.Optimization.Neighbor
import BauxResearch.Optimization.Step
import Mathlib.Analysis.Normed.Module.Convex

/-!
# Error of the source neighbor mean, including its denominator floor

Mathlib's convex distance function and Jensen theorem act on the actual
zero-anchor center-of-mass view. This is a deterministic, pathwise result:
neither independence nor a posterior law is inferred. Turning measured noise,
sampled dissimilarities, and LOOCV fitting into its premises is separate work.
-/
/-!
The source estimate's error separates into geometry discrepancy, signed
observation error, and the bias created when the normalization denominator
is clamped.  Mathlib's convex distance/Jensen bound controls a center of mass;
it does not certify the ENN inputs or remove this floor term.  Any notation
for an error radius must include all three contributions.
-/
namespace BauxResearch.Optimization.Accuracy
noncomputable section
open scoped BigOperators
open BauxResearch.Forward

theorem centerBound {Index : Type*} [Fintype Index]
    (weights values : Index → ℝ) (truth : ℝ)
    (nonnegative : ∀ index, 0 ≤ weights index) (positive : 0 < ∑ index, weights index) :
    |Finset.univ.centerMass weights values - truth| ≤
      Finset.univ.centerMass weights (fun index => |values index - truth|) := by
  have bound := (convexOn_univ_dist truth).map_centerMass_le
    (fun index _ => nonnegative index) positive (by simp : ∀ index ∈ Finset.univ, values index ∈ Set.univ)
  simpa only [Real.dist_eq, Function.comp_def] using bound

def weightTotal {capacity : ℕ} (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℝ :=
  (nearest.map (Neighbor.weight data params scale distances)).sum

def mass {capacity : ℕ} (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) : ℝ :=
  max (1 / 10^12) (weightTotal data params scale distances nearest)

/-- Signed differences under the actual clamped source weights.  The supplied
parameters, distances, and selected neighbor list may depend on observed data.
No independence or calibration property is asserted by this definition. -/
def weightedDelta {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (left right : Vector capacity) : ℝ :=
  (∑ position : Fin nearest.length,
    Neighbor.weight data params scale distances nearest[position.val] *
      (left nearest[position.val] - right nearest[position.val])) /
    mass data params scale distances nearest

/-- Exact separation of geometric discrepancy, signed observation error, and
the zero-anchor bias of the denominator floor.  Unlike a sum of absolute
observation errors, the signed term retains possible noise cancellation. -/
theorem errorSplit {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (truth : Vector capacity) (query : ℝ) :
    Neighbor.mean data params scale distances nearest - query =
      weightedDelta data params scale distances nearest truth (fun _ ↦ query) +
        weightedDelta data params scale distances nearest data.values truth -
        (mass data params scale distances nearest -
          weightTotal data params scale distances nearest) * query /
          mass data params scale distances nearest := by
  have denominator : mass data params scale distances nearest ≠ 0 :=
    ne_of_gt (lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12)
      (le_max_left _ _))
  have sumWeights : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val]) =
      weightTotal data params scale distances nearest := by
    simp only [weightTotal, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  have meanForm : Neighbor.mean data params scale distances nearest =
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          data.values nearest[position.val]) /
        mass data params scale distances nearest := by
    have sumValues : (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          data.values nearest[position.val]) =
        (nearest.map fun index ↦
          Neighbor.weight data params scale distances index * data.values index).sum := by
      rw [← List.sum_ofFn]
      rw [List.ofFn_getElem_eq_map nearest (fun index ↦
        Neighbor.weight data params scale distances index * data.values index)]
    rw [sumValues]
    rfl
  have sumIdentity :
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          (truth nearest[position.val] - query)) +
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          (data.values nearest[position.val] - truth nearest[position.val])) =
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          data.values nearest[position.val]) -
        weightTotal data params scale distances nearest * query := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, sumWeights]
    ring
  rw [meanForm]
  unfold weightedDelta
  rw [← add_div, sumIdentity]
  field_simp [denominator]
  ring

/-- Geometric and signed-noise certificates suffice for the source mean.
Statistical cancellation must be proved for the fitted weights and selected
query; a fixed-weight variance identity alone cannot provide that certificate. -/
theorem splitBound {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (truth : Vector capacity) (query bias noise : ℝ)
    (geometry : |weightedDelta data params scale distances nearest truth
      (fun _ ↦ query)| ≤ bias)
    (observed : |weightedDelta data params scale distances nearest
      data.values truth| ≤ noise) :
    |Neighbor.mean data params scale distances nearest - query| ≤
      bias + noise +
        (mass data params scale distances nearest -
          weightTotal data params scale distances nearest) * |query| /
          mass data params scale distances nearest := by
  have positive : 0 < mass data params scale distances nearest :=
    lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) (le_max_left _ _)
  have extra : 0 ≤ mass data params scale distances nearest -
      weightTotal data params scale distances nearest :=
    sub_nonneg.mpr (le_max_right _ _)
  rw [errorSplit data params scale distances nearest truth query]
  have triangle (first second third : ℝ) :
      |first + second - third| ≤ |first| + |second| + |third| :=
    (abs_sub _ _).trans (add_le_add (abs_add_le first second) (le_refl |third|))
  refine (triangle _ _ _).trans ?_
  rw [abs_div, abs_mul, abs_of_nonneg extra, abs_of_pos positive]
  exact add_le_add (add_le_add geometry observed) (le_refl _)

/-- The floor contributes an explicit anchor-bias term, even without noise.
List-position indices retain the operational list's multiplicities. -/
theorem clamped {capacity : ℕ} (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) (truth : ℝ)
    (nonnegative : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val]) :
    |Neighbor.mean data params scale distances nearest - truth| ≤
      ((∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          |data.values nearest[position.val] - truth|) +
        (mass data params scale distances nearest - weightTotal data params scale distances nearest) * |truth|) /
      mass data params scale distances nearest := by
  let weights : Option (Fin nearest.length) → ℝ := fun index => match index with
    | none => mass data params scale distances nearest - weightTotal data params scale distances nearest
    | some position => Neighbor.weight data params scale distances nearest[position.val]
  let values : Option (Fin nearest.length) → ℝ := fun index => match index with
    | none => 0
    | some position => data.values nearest[position.val]
  have weightsPositive : ∀ index, 0 ≤ weights index := by
    intro index
    cases index with
    | none => exact sub_nonneg.mpr (le_max_right _ _)
    | some position => exact nonnegative position
  have sumWeights : (∑ index, weights index) = mass data params scale distances nearest := by
    simp only [weights, Fintype.sum_option, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
    unfold weightTotal
    ring
  have sumPositive : 0 < ∑ index, weights index := by
    rw [sumWeights]
    exact lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) (le_max_left _ _)
  have bound := centerBound weights values truth weightsPositive sumPositive
  rw [Neighbor.Mean.anchored]
  change |Finset.univ.centerMass weights values - truth| ≤ _
  have rightEq : Finset.univ.centerMass weights (fun index => |values index - truth|) =
      ((∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          |data.values nearest[position.val] - truth|) +
        (mass data params scale distances nearest - weightTotal data params scale distances nearest) * |truth|) /
      mass data params scale distances nearest := by
    simp only [Finset.centerMass, sumWeights, smul_eq_mul, Fintype.sum_option, weights, values,
      zero_sub, abs_neg, div_eq_mul_inv]
    ring
  rwa [rightEq] at bound

/-- Deterministic geometry and observation error compose with the actual clamped
mean. Supplied source dissimilarities are not assumed to be a metric. -/
theorem geometric {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (objective : State → ℝ) (points : Fin capacity → State) (query : State)
    (constant : NNReal) (noise : Vector capacity)
    (regular : LipschitzWith constant objective)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (nonnegative : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val]) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      ((∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          (noise nearest[position.val] + constant * dist (points nearest[position.val]) query)) +
        (mass data params scale distances nearest - weightTotal data params scale distances nearest) * |objective query|) /
      mass data params scale distances nearest := by
  have bound := clamped data params scale distances nearest (objective query) nonnegative
  have errors : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective query| ≤
        noise nearest[position.val] + constant * dist (points nearest[position.val]) query := by
    intro position
    have continuity := regular.dist_le_mul (points nearest[position.val]) query
    rw [Real.dist_eq] at continuity
    exact (abs_sub_le _ _ _).trans (add_le_add (observed position) continuity)
  apply bound.trans
  apply div_le_div_of_nonneg_right _ (le_of_lt
    (lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) (le_max_left _ _)))
  apply add_le_add
  · apply Finset.sum_le_sum
    intro position _
    exact mul_le_mul_of_nonneg_left (errors position) (nonnegative position)
  · exact le_refl _

/-- With an inactive floor, the actual neighbor mean inherits a common pointwise
error bound. No posterior or statistical confidence is inferred. -/
theorem inactive {capacity : ℕ} (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity)) (truth error : ℝ)
    (nonnegative : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (floorBound : (1 / 10^12 : ℝ) ≤ weightTotal data params scale distances nearest)
    (errors : ∀ position : Fin nearest.length, |data.values nearest[position.val] - truth| ≤ error) :
    |Neighbor.mean data params scale distances nearest - truth| ≤ error := by
  have bound := clamped data params scale distances nearest truth nonnegative
  have sumWeights : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val]) =
      weightTotal data params scale distances nearest := by
    simp only [weightTotal, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  have weighted : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val] *
        |data.values nearest[position.val] - truth|) ≤
      weightTotal data params scale distances nearest * error := by
    calc
      _ ≤ ∑ position : Fin nearest.length,
          Neighbor.weight data params scale distances nearest[position.val] * error :=
        Finset.sum_le_sum (fun position _ => mul_le_mul_of_nonneg_left (errors position) (nonnegative position))
      _ = _ := by rw [← Finset.sum_mul, sumWeights]
  have totalPositive : 0 < weightTotal data params scale distances nearest :=
    lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) floorBound
  simp only [mass, max_eq_right floorBound, sub_self, zero_mul, add_zero] at bound
  exact bound.trans ((div_le_iff₀ totalPositive).2
    (by simpa only [mul_comm (weightTotal data params scale distances nearest) error] using weighted))

/-- This closes the normalization-to-selector connection for mean-based scores.
The bounded observation/geometry premises must hold across the entire pool. -/
theorem poolBound {Candidate : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Candidate → Vector capacity) (nearest : Candidate → List (Fin capacity))
    (truth : Candidate → ℝ) (error : ℝ)
    (nonnegative : ∀ candidate (position : Fin (nearest candidate).length),
      0 ≤ Neighbor.weight data params scale (distances candidate) (nearest candidate)[position.val])
    (floorBound : ∀ candidate,
      (1 / 10^12 : ℝ) ≤ weightTotal data params scale (distances candidate) (nearest candidate))
    (errors : ∀ candidate (position : Fin (nearest candidate).length),
      |data.values (nearest candidate)[position.val] - truth candidate| ≤ error) :
    UniformError truth (fun candidate => Neighbor.mean data params scale (distances candidate)
      (nearest candidate)) error := by
  intro candidate
  exact inactive data params scale (distances candidate) (nearest candidate) (truth candidate) error
    (nonnegative candidate) (floorBound candidate) (errors candidate)

/-- Candidate-specific bounds for the actual weighted neighbor mean. This is
the adaptive counterpart of `poolBound`: every candidate retains its own local
geometry and observation envelope. -/
theorem pointPool {Candidate : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Candidate → Vector capacity) (nearest : Candidate → List (Fin capacity))
    (truth error : Candidate → ℝ)
    (nonnegative : ∀ candidate (position : Fin (nearest candidate).length),
      0 ≤ Neighbor.weight data params scale (distances candidate) (nearest candidate)[position.val])
    (floorBound : ∀ candidate,
      (1 / 10^12 : ℝ) ≤ weightTotal data params scale (distances candidate) (nearest candidate))
    (errors : ∀ candidate (position : Fin (nearest candidate).length),
      |data.values (nearest candidate)[position.val] - truth candidate| ≤ error candidate) :
    PointError truth (fun candidate => Neighbor.mean data params scale (distances candidate)
      (nearest candidate)) error := by
  intro candidate
  exact inactive data params scale (distances candidate) (nearest candidate)
    (truth candidate) (error candidate) (nonnegative candidate)
    (floorBound candidate) (errors candidate)

/-- Approximate mean selection pays only the local envelopes of the selected
and comparison candidates, rather than twice a global worst-case envelope. -/
theorem pointSelect {Candidate : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Candidate → Vector capacity) (nearest : Candidate → List (Fin capacity))
    (truth error : Candidate → ℝ) (selected comparison : Candidate) (slack : ℝ)
    (nonnegative : ∀ candidate (position : Fin (nearest candidate).length),
      0 ≤ Neighbor.weight data params scale (distances candidate) (nearest candidate)[position.val])
    (floorBound : ∀ candidate,
      (1 / 10^12 : ℝ) ≤ weightTotal data params scale (distances candidate) (nearest candidate))
    (errors : ∀ candidate (position : Fin (nearest candidate).length),
      |data.values (nearest candidate)[position.val] - truth candidate| ≤ error candidate)
    (chosen : ApproxMin (fun candidate => Neighbor.mean data params scale (distances candidate)
      (nearest candidate)) selected slack) :
    truth selected - truth comparison ≤
      error selected + error comparison + slack :=
  Selection.pointBound truth _ error selected comparison slack
    (pointPool data params scale distances nearest truth error
      nonnegative floorBound errors) chosen

theorem selectionBound {Candidate : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Candidate → Vector capacity) (nearest : Candidate → List (Fin capacity))
    (truth : Candidate → ℝ) (selected comparison : Candidate) (error slack : ℝ)
    (nonnegative : ∀ candidate (position : Fin (nearest candidate).length),
      0 ≤ Neighbor.weight data params scale (distances candidate) (nearest candidate)[position.val])
    (floorBound : ∀ candidate,
      (1 / 10^12 : ℝ) ≤ weightTotal data params scale (distances candidate) (nearest candidate))
    (errors : ∀ candidate (position : Fin (nearest candidate).length),
      |data.values (nearest candidate)[position.val] - truth candidate| ≤ error)
    (chosen : ApproxMin (fun candidate => Neighbor.mean data params scale (distances candidate)
      (nearest candidate)) selected slack) :
    truth selected - truth comparison ≤ 2 * error + slack :=
  Selection.regretBound truth _ selected comparison error slack
    (poolBound data params scale distances nearest truth error nonnegative floorBound errors) chosen

#print axioms centerBound
#print axioms errorSplit
#print axioms splitBound
#print axioms clamped
#print axioms geometric
#print axioms poolBound
#print axioms pointPool
#print axioms pointSelect
#print axioms selectionBound
end
end BauxResearch.Optimization.Accuracy
