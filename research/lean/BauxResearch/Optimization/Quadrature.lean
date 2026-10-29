import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

/-!
# Deterministic stratified parameter-distance error

The source uses one fixed parameter index per stratum and weights its squared
displacement by stratum cardinality and tensor weight. Fibers of a supplied
stratum map partition all parameter indices. Mathlib finite sums give an exact
error decomposition and a deterministic bound charging every unsampled index.

No random-sampling law, independent hash bits, or adaptation restriction is
assumed. Source index arithmetic, proposal/sample agreement, and floating-point
reductions remain separate refinements. The bound can be large: a small sampled
distance is not itself evidence of a small full distance.
-/

namespace BauxResearch.Optimization.Quadrature

open scoped BigOperators
noncomputable section

variable {Index Cell : Type*} [Fintype Index] [DecidableEq Index]
  [Fintype Cell] [DecidableEq Cell]

def bucket (stratum : Index → Cell) (center : Cell) : Finset Index :=
  Finset.univ.filter fun index => stratum index = center

def full (stratum : Index → Cell) (weight : Cell → ℝ) (delta : Index → ℝ) : ℝ :=
  ∑ index, weight (stratum index) * delta index ^ 2

def sampled (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (delta : Index → ℝ) : ℝ :=
  ∑ center, weight center * (bucket stratum center).card * delta (sample center)^2

def remainder (stratum : Index → Cell) (sample : Cell → Index)
    (weight oscillation : Cell → ℝ) : ℝ :=
  ∑ center, weight center * ((bucket stratum center).erase (sample center)).card * oscillation center

/-- Weighted Euclidean coordinates use every parameter, not just the sampled
indices. Strictly positive weights make this an injective change of units. -/
def embed (stratum : Index → Cell) (weight : Cell → ℝ) (state : Index → ℝ) :
    EuclideanSpace ℝ Index :=
  WithLp.toLp 2 (fun index => Real.sqrt (weight (stratum index)) * state index)

/-- Return from weighted Euclidean units to all physical parameter values. -/
def unembed (stratum : Index → Cell) (weight : Cell → ℝ)
    (state : EuclideanSpace ℝ Index) : Index → ℝ :=
  fun index => state index / Real.sqrt (weight (stratum index))

omit [Fintype Index] [DecidableEq Index] [Fintype Cell] [DecidableEq Cell] in
theorem unembedEmbed (stratum : Index → Cell) (weight : Cell → ℝ) (state : Index → ℝ)
    (positive : ∀ center, 0 < weight center) :
    unembed stratum weight (embed stratum weight state) = state := by
  funext index
  change Real.sqrt (weight (stratum index)) * state index / Real.sqrt (weight (stratum index)) = _
  exact mul_div_cancel_left₀ _ (Real.sqrt_pos.mpr (positive _)).ne'

omit [Fintype Index] [DecidableEq Index] [Fintype Cell] [DecidableEq Cell] in
theorem embedInjective (stratum : Index → Cell) (weight : Cell → ℝ)
    (positive : ∀ center, 0 < weight center) : Function.Injective (embed stratum weight) := by
  intro first second same
  funext index
  have equal := congrArg (fun state : EuclideanSpace ℝ Index => state index) same
  change Real.sqrt (weight (stratum index)) * first index =
    Real.sqrt (weight (stratum index)) * second index at equal
  exact mul_left_cancel₀ (Real.sqrt_pos.mpr (positive _)).ne' equal

omit [DecidableEq Index] [Fintype Cell] [DecidableEq Cell] in
/-- Mathlib's Euclidean squared norm is exactly the full weighted energy.
This identifies an actual metric quantity rather than calling an arbitrary
supplied dissimilarity a metric. Zero weights permit a pseudometric only. -/
theorem distanceSquared (stratum : Index → Cell) (weight : Cell → ℝ)
    (first second : Index → ℝ) (nonnegative : ∀ center, 0 ≤ weight center) :
    dist (embed stratum weight first) (embed stratum weight second)^2 =
      full stratum weight (fun index => first index - second index) := by
  rw [dist_eq_norm, EuclideanSpace.real_norm_sq_eq]
  unfold full
  apply Finset.sum_congr rfl
  intro index _
  change (Real.sqrt (weight (stratum index)) * first index -
    Real.sqrt (weight (stratum index)) * second index)^2 = _
  rw [← mul_sub, mul_pow, Real.sq_sqrt (nonnegative _)]

omit [DecidableEq Index] in
theorem sampledNonneg (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (delta : Index → ℝ) (nonnegative : ∀ center, 0 ≤ weight center) :
    0 ≤ sampled stratum sample weight delta := by
  apply Finset.sum_nonneg
  intro center _
  exact mul_nonneg (mul_nonneg (nonnegative _) (Nat.cast_nonneg _)) (sq_nonneg _)

theorem remainderNonneg (stratum : Index → Cell) (sample : Cell → Index)
    (weight oscillation : Cell → ℝ) (nonnegative : ∀ center, 0 ≤ weight center)
    (bounds : ∀ center, 0 ≤ oscillation center) :
    0 ≤ remainder stratum sample weight oscillation := by
  apply Finset.sum_nonneg
  intro center _
  exact mul_nonneg (mul_nonneg (nonnegative _) (Nat.cast_nonneg _)) (bounds _)

omit [DecidableEq Index] [Fintype Cell] in
/-- The same supplied representative belongs to its source stratum. -/
theorem sampleMember (stratum : Index → Cell) (sample : Cell → Index)
    (coherent : ∀ center, stratum (sample center) = center) (center : Cell) :
    sample center ∈ bucket stratum center := by simp [bucket, coherent]

omit [Fintype Cell] in
theorem massSplit (stratum : Index → Cell) (sample : Cell → Index)
    (coherent : ∀ center, stratum (sample center) = center) (center : Cell) :
    ((bucket stratum center).erase (sample center)).card + 1 = (bucket stratum center).card :=
  Finset.card_erase_add_one (sampleMember stratum sample coherent center)

omit [DecidableEq Index] in
/-- All full-state parameters are counted exactly once, including unsampled
parameters. The weights are constant within a stratum as in the source. -/
theorem fullSplit (stratum : Index → Cell) (weight : Cell → ℝ) (delta : Index → ℝ) :
    full stratum weight delta =
      ∑ center, weight center * ∑ index ∈ bucket stratum center, delta index ^ 2 := by
  classical
  unfold full
  calc
    _ = ∑ index : Index, ∑ center : Cell,
        if stratum index = center then weight center * delta index^2 else 0 := by simp
    _ = ∑ center : Cell, ∑ index : Index,
        if stratum index = center then weight center * delta index^2 else 0 := Finset.sum_comm
    _ = _ := by simp [bucket, Finset.mul_sum, Finset.sum_filter]

omit [Fintype Index] in
/-- A stratum's sampled contribution cancels exactly. Only its other
parameters contribute to quadrature error. -/
theorem localBound (indices : Finset Index) (sample : Index) (member : sample ∈ indices)
    (delta : Index → ℝ) (oscillation : ℝ)
    (bounded : ∀ index ∈ indices, |delta index^2 - delta sample^2| ≤ oscillation) :
    |(∑ index ∈ indices, delta index^2) - indices.card * delta sample^2| ≤
      (indices.erase sample).card * oscillation := by
  have same : (∑ index ∈ indices, (delta index^2 - delta sample^2)) =
      (∑ index ∈ indices, delta index^2) - indices.card * delta sample^2 := by
    simp [Finset.sum_sub_distrib]
  rw [← same]
  have erased := Finset.sum_erase_add indices (fun index => delta index^2 - delta sample^2) member
  simp only [sub_self, add_zero] at erased
  rw [← erased]
  calc
    _ ≤ ∑ index ∈ indices.erase sample, |delta index^2 - delta sample^2| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _index ∈ indices.erase sample, oscillation :=
      Finset.sum_le_sum fun index inside => bounded index (Finset.mem_of_mem_erase inside)
    _ = _ := by simp

/-- A pathwise quadrature certificate, valid for adaptively selected states
without treating fixed source hash indices as fresh independent samples. -/
theorem errorBound (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (delta : Index → ℝ) (oscillation : Cell → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (nonnegative : ∀ center, 0 ≤ weight center)
    (bounded : ∀ center index, stratum index = center →
      |delta index^2 - delta (sample center)^2| ≤ oscillation center) :
    |full stratum weight delta - sampled stratum sample weight delta| ≤
      remainder stratum sample weight oscillation := by
  rw [fullSplit]
  unfold sampled remainder
  rw [← Finset.sum_sub_distrib]
  calc
    _ ≤ ∑ center, |weight center * (∑ index ∈ bucket stratum center, delta index^2) -
        weight center * (bucket stratum center).card * delta (sample center)^2| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro center _
      have localError := localBound (bucket stratum center) (sample center)
        (sampleMember stratum sample coherent center) delta (oscillation center)
        (fun index inside => bounded center index (Finset.mem_filter.mp inside).2)
      have factor : weight center * (∑ index ∈ bucket stratum center, delta index^2) -
          weight center * (bucket stratum center).card * delta (sample center)^2 =
          weight center * ((∑ index ∈ bucket stratum center, delta index^2) -
            (bucket stratum center).card * delta (sample center)^2) := by ring
      rw [factor, abs_mul, abs_of_nonneg (nonnegative center)]
      simpa [mul_assoc] using mul_le_mul_of_nonneg_left localError (nonnegative center)

theorem distanceUpper (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (delta : Index → ℝ) (oscillation : Cell → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (nonnegative : ∀ center, 0 ≤ weight center)
    (bounded : ∀ center index, stratum index = center →
      |delta index^2 - delta (sample center)^2| ≤ oscillation center) :
    full stratum weight delta ≤ sampled stratum sample weight delta +
      remainder stratum sample weight oscillation := by
  have bound := (abs_le.mp (errorBound stratum sample weight delta oscillation coherent nonnegative bounded)).2
  linarith

omit [Fintype Index] [DecidableEq Index] [Fintype Cell] [DecidableEq Cell] in
/-- A proved bound on executed-minus-ideal displacement yields the exact
quadratic error penalty. No BF16 conversion bound is assumed from its name. -/
theorem energyPerturbation (executed ideal error amplitude : ℝ)
    (rounded : |executed - ideal| ≤ error) (bounded : |ideal| ≤ amplitude) :
    |executed^2 - ideal^2| ≤ error * (2 * amplitude + error) := by
  have errorNonneg := (abs_nonneg (executed - ideal)).trans rounded
  have magnitude : |executed| ≤ amplitude + error := by
    have triangle : |executed| ≤ |executed - ideal| + |ideal| := by
      simpa only [sub_add_cancel] using abs_add_le (executed - ideal) ideal
    linarith
  have sumBound : |executed + ideal| ≤ 2 * amplitude + error :=
    (abs_add_le executed ideal).trans (by linarith)
  have factor : executed^2 - ideal^2 = (executed - ideal) * (executed + ideal) := by ring
  rw [factor, abs_mul]
  exact mul_le_mul rounded sumBound (abs_nonneg _) errorNonneg

omit [Fintype Index] [DecidableEq Index] [Fintype Cell] [DecidableEq Cell] in
/-- Ideal field structure and an explicit execution-error cap control actual
within-stratum oscillation. Both representative and unsampled errors count. -/
theorem oscillationBound (stratum : Index → Cell) (sample : Cell → Index)
    (actual ideal : Index → ℝ) (amplitude error oscillation : Cell → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (rounded : ∀ index, |actual index - ideal index| ≤ error (stratum index))
    (bounded : ∀ index, |ideal index| ≤ amplitude (stratum index))
    (structured : ∀ center index, stratum index = center →
      |ideal index^2 - ideal (sample center)^2| ≤ oscillation center)
    (center : Cell) (index : Index) (inside : stratum index = center) :
    |actual index^2 - actual (sample center)^2| ≤
      oscillation center + 2 * (error center * (2 * amplitude center + error center)) := by
  have first := energyPerturbation (actual index) (ideal index) (error (stratum index))
    (amplitude (stratum index)) (rounded index) (bounded index)
  rw [inside] at first
  have last := energyPerturbation (actual (sample center)) (ideal (sample center))
    (error center) (amplitude center) (by simpa only [coherent] using rounded (sample center))
    (by simpa only [coherent] using bounded (sample center))
  rw [abs_sub_comm] at last
  have middle := structured center index inside
  have triangle := (abs_sub_le (actual index^2) (ideal index^2) (actual (sample center)^2)).trans
    (add_le_add first (abs_sub_le (ideal index^2) (ideal (sample center)^2) (actual (sample center)^2)))
  linarith

/-- The same original quadrature on executed parameters receives an explicit
rounding/error residual, rather than inheriting exactness of an ideal field. -/
theorem roundedError (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (actual ideal : Index → ℝ) (amplitude error oscillation : Cell → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (nonnegative : ∀ center, 0 ≤ weight center)
    (rounded : ∀ index, |actual index - ideal index| ≤ error (stratum index))
    (bounded : ∀ index, |ideal index| ≤ amplitude (stratum index))
    (structured : ∀ center index, stratum index = center →
      |ideal index^2 - ideal (sample center)^2| ≤ oscillation center) :
    |full stratum weight actual - sampled stratum sample weight actual| ≤
      remainder stratum sample weight (fun center => oscillation center +
        2 * (error center * (2 * amplitude center + error center))) :=
  errorBound stratum sample weight actual _ coherent nonnegative
    (oscillationBound stratum sample actual ideal amplitude error oscillation coherent rounded bounded structured)

/-- Constant squared displacement per stratum makes the source quadrature
exact. This is stronger than an expectation identity and needs no sign law. -/
theorem constantExact (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (delta : Index → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (nonnegative : ∀ center, 0 ≤ weight center)
    (constant : ∀ center index, stratum index = center →
      delta index^2 = delta (sample center)^2) :
    full stratum weight delta = sampled stratum sample weight delta := by
  have bound := errorBound stratum sample weight delta (fun _ => 0) coherent nonnegative
    (fun center index inside => by rw [constant center index inside]; simp)
  simpa [remainder, abs_nonpos_iff, sub_eq_zero] using bound

/-- Independent real Rademacher updates relative to their own base have this
exactness when amplitude is constant per stratum. Storage rounding and distances
to earlier, different model states are deliberately not covered. -/
theorem signExact (stratum : Index → Cell) (sample : Cell → Index)
    (weight amplitude : Cell → ℝ) (sign : Index → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (nonnegative : ∀ center, 0 ≤ weight center) (unit : ∀ index, sign index^2 = 1) :
    full stratum weight (fun index => amplitude (stratum index) * sign index) =
      sampled stratum sample weight (fun index => amplitude (stratum index) * sign index) := by
  apply constantExact stratum sample weight _ coherent nonnegative
  intro center index inside
  simp [mul_pow, unit, coherent, inside]

/-- An omitted parameter can carry positive full distance while every sampled
coordinate agrees. No unqualified domination by sampled distance is possible. -/
theorem missedEnergy (stratum : Index → Cell) (sample : Cell → Index)
    (weight : Cell → ℝ) (missing : Index)
    (omitted : ∀ center, sample center ≠ missing) (positive : 0 < weight (stratum missing)) :
    ∃ delta : Index → ℝ, sampled stratum sample weight delta = 0 ∧ 0 < full stratum weight delta := by
  refine ⟨fun index => if index = missing then 1 else 0, ?_, ?_⟩
  · simp [sampled, omitted]
  · simpa [full] using positive

#print axioms fullSplit
#print axioms embedInjective
#print axioms distanceSquared
#print axioms errorBound
#print axioms distanceUpper
#print axioms energyPerturbation
#print axioms oscillationBound
#print axioms roundedError
#print axioms constantExact
#print axioms signExact
#print axioms missedEnergy

end
end BauxResearch.Optimization.Quadrature
