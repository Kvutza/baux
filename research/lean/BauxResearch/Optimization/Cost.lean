import BauxResearch.Forward.Work
import Mathlib.Analysis.Asymptotics.Theta

/-! Source-grounded storage and loop counts, not a complete machine cost model.
Asymptotic variables are explicit. Fixed history is not the same regime as
growing history. Sorting, PRNG retries, and forward evaluation are separate. -/
namespace BauxResearch.Optimization.Cost
noncomputable section
open Filter Asymptotics

/-- Allocated f32 matrix plus outcome and variance arrays in neighbor_value.rs. -/
def historyBytes (capacity : ℕ) : ℕ := 4 * (capacity^2 + 2 * capacity)

/-- Two response coefficient arrays and one observation counter per patch. -/
def responseBytes (patches groups : ℕ) : ℕ := 4 * (2 * patches * groups + patches)

/-- One full stored u16 candidate write; excludes reads, backups, and padding. -/
def candidateBytes (parameters : ℕ) : ℕ := 2 * parameters

/-- Source scans history, then has prefix/weight/mean/noise loops over chosen neighbors.
The history sort is deliberately not included in this count. -/
def neighborVisits (history neighbors : ℕ) : ℕ := history + 4 * min neighbors history

def responseVisits (patches groups candidates : ℕ) : ℕ :=
  patches * groups + candidates * groups

theorem History.quadratic :
    (fun capacity : ℕ => (historyBytes capacity : ℝ)) =Θ[atTop]
      (fun capacity : ℕ => (capacity : ℝ)^2) := by
  constructor
  · apply IsBigO.of_bound 12
    filter_upwards [eventually_ge_atTop 1] with capacity positive
    have realPositive : (1 : ℝ) ≤ capacity := by exact_mod_cast positive
    have squareBound : (capacity : ℝ) ≤ (capacity : ℝ)^2 := by nlinarith
    simp only [historyBytes, Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat]
    rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (sq_nonneg _)]
    nlinarith
  · apply IsBigO.of_bound 1
    filter_upwards [] with capacity
    simp only [historyBytes, Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat]
    rw [Real.norm_of_nonneg (sq_nonneg _), Real.norm_of_nonneg (by positivity)]
    nlinarith [sq_nonneg (capacity : ℝ), Nat.cast_nonneg (α := ℝ) capacity]

theorem Response.linear (groups : ℕ) :
    (fun patches : ℕ => (responseBytes patches groups : ℝ)) =Θ[atTop]
      (fun patches : ℕ => (patches : ℝ)) := by
  have equality : (fun patches : ℕ => (responseBytes patches groups : ℝ)) =
      (fun patches : ℕ => (4 * (2 * (groups : ℝ) + 1)) * (patches : ℝ)) := by
    funext patches
    simp only [responseBytes, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat]
    ring
  rw [equality]
  exact (isTheta_refl _ _).const_mul_left (by positivity)

theorem Candidate.linear :
    (fun parameters : ℕ => (candidateBytes parameters : ℝ)) =Θ[atTop]
      (fun parameters : ℕ => (parameters : ℝ)) := by
  simp only [candidateBytes, Nat.cast_mul, Nat.cast_ofNat]
  exact (isTheta_refl _ _).const_mul_left (by norm_num)

theorem Neighbor.scan (neighbors : ℕ) :
    (fun history : ℕ => (neighborVisits history neighbors : ℝ)) =O[atTop]
      (fun history : ℕ => (history : ℝ)) := by
  apply IsBigO.of_bound 5
  filter_upwards [] with history
  have bound : min neighbors history ≤ history := min_le_right _ _
  have realBound : ((min neighbors history : ℕ) : ℝ) ≤ history := by exact_mod_cast bound
  simp only [neighborVisits, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat]
  rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
  linarith

#print axioms History.quadratic
#print axioms Response.linear
#print axioms Candidate.linear
end
end BauxResearch.Optimization.Cost
