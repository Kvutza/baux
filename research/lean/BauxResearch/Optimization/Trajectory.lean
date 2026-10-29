import BauxResearch.Optimization.Step
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Topology.MetricSpace.Basic

/-!
# Trajectory distribution reduction and total-variation reward bounds

This module formalizes the trajectory-distribution reduction:
"A generation-level trust region has a precise meaning."

For a causal model generating token trajectories, raw parameter distance does
not describe model behavior.  The observable reward is the expectation of a
bounded task score `r(y) ∈ [0, 1]` under the trajectory distribution `P_θ`.

By the total variation bound, the expected reward difference between any two
models is bounded by the total variation distance between their trajectory
distributions:
`|E_P[r] - E_Q[r]| ≤ TV(P, Q)`.

Under a trust-region divergence bound `TV(P_θ, P_θ') ≤ K * length`, the reward
satisfies linear Lipschitz stability without assuming Euclidean parameter-space
smoothness.
-/

namespace BauxResearch.Optimization

noncomputable section

open scoped BigOperators

variable {Path : Type*} [Fintype Path]

/-- A probability distribution over a finite trajectory space. -/
structure TrajectoryDist (Path : Type*) [Fintype Path] where
  prob : Path → ℝ
  nonneg : ∀ y, 0 ≤ prob y
  sumOne : ∑ y, prob y = 1

/-- Total variation distance between two trajectory distributions. -/
def totalVariation
    (P Q : TrajectoryDist Path) : ℝ :=
  (1 / 2) * ∑ y, |P.prob y - Q.prob y|

/-- The expected task reward under a trajectory distribution. -/
def expectedReward
    (P : TrajectoryDist Path)
    (reward : Path → ℝ) : ℝ :=
  ∑ y, P.prob y * reward y

/-- Total variation is always nonnegative. -/
theorem tv_nonneg
    (P Q : TrajectoryDist Path) :
    0 ≤ totalVariation P Q := by
  unfold totalVariation
  have hSumNonneg : 0 ≤ ∑ y, |P.prob y - Q.prob y| := by
    apply Finset.sum_nonneg
    intro y _
    exact abs_nonneg _
  nlinarith

/-- Centering a bounded value in `[0, 1]` around `1/2` bounds its absolute value by `1/2`. -/
theorem centerBound
    (r : ℝ)
    (hLower : 0 ≤ r)
    (hUpper : r ≤ 1) :
    |r - 1 / 2| ≤ 1 / 2 := by
  rw [abs_le]
  constructor <;> linarith

/--
Reduction 3: For any bounded task score `reward(y) ∈ [0, 1]`, the difference in
expected reward is bounded by the total variation distance `TV(P, Q)`.
-/
theorem tv_bound
    (P Q : TrajectoryDist Path)
    (reward : Path → ℝ)
    (hLower : ∀ y, 0 ≤ reward y)
    (hUpper : ∀ y, reward y ≤ 1) :
    |expectedReward P reward - expectedReward Q reward| ≤ totalVariation P Q := by
  have hDiffSum : ∑ y, (P.prob y - Q.prob y) = 0 := by
    rw [Finset.sum_sub_distrib, P.sumOne, Q.sumOne, sub_self]
  have hCenterIdentity :
      expectedReward P reward - expectedReward Q reward =
      ∑ y, (P.prob y - Q.prob y) * (reward y - 1 / 2) := by
    unfold expectedReward
    rw [← Finset.sum_sub_distrib]
    have hTermEq : ∀ y, P.prob y * reward y - Q.prob y * reward y =
        (P.prob y - Q.prob y) * (reward y - 1 / 2) + (P.prob y - Q.prob y) * (1 / 2) := by
      intro y
      ring
    simp_rw [hTermEq]
    rw [Finset.sum_add_distrib]
    have hZero : ∑ y, (P.prob y - Q.prob y) * (1 / 2) = 0 := by
      rw [← Finset.sum_mul, hDiffSum, MulZeroClass.zero_mul]
    rw [hZero, add_zero]
  rw [hCenterIdentity]
  have hAbsSum : |∑ y, (P.prob y - Q.prob y) * (reward y - 1 / 2)| ≤
      ∑ y, |(P.prob y - Q.prob y) * (reward y - 1 / 2)| :=
    Finset.abs_sum_le_sum_abs _ _
  have hTermLe : ∀ y, |(P.prob y - Q.prob y) * (reward y - 1 / 2)| ≤
      (1 / 2) * |P.prob y - Q.prob y| := by
    intro y
    rw [abs_mul, mul_comm]
    have hC := centerBound (reward y) (hLower y) (hUpper y)
    exact mul_le_mul_of_nonneg_right hC (abs_nonneg _)
  have hSumLe : (∑ y, |(P.prob y - Q.prob y) * (reward y - 1 / 2)|) ≤
      ∑ y, (1 / 2) * |P.prob y - Q.prob y| :=
    Finset.sum_le_sum fun y _ ↦ hTermLe y
  have hReorder : (∑ y, (1 / 2) * |P.prob y - Q.prob y|) = totalVariation P Q := by
    unfold totalVariation
    rw [← Finset.mul_sum]
  linarith

/--
Trust-region stability: If a candidate step changes the trajectory distribution
by at most `K * length` in total variation, the reward shift is at most `K * length`.
-/
theorem trust_bound
    (P Q : TrajectoryDist Path)
    (reward : Path → ℝ)
    (length K : ℝ)
    (hLower : ∀ y, 0 ≤ reward y)
    (hUpper : ∀ y, reward y ≤ 1)
    (hTV : totalVariation P Q ≤ K * length) :
    |expectedReward P reward - expectedReward Q reward| ≤ K * length := by
  have hBound := tv_bound P Q reward hLower hUpper
  linarith

end

section Drift

variable {State : Type*} [PseudoMetricSpace State]

/--
Cumulative metric displacement bound: Over `T` steps, if each step displacement
is bounded by `radii t`, total displacement from the starting center is bounded by `∑ t, radii t`.
-/
theorem Trajectory.distDrift
    (seq : ℕ → State)
    (radii : ℕ → ℝ)
    (T : ℕ)
    (hStep : ∀ t < T, dist (seq (t + 1)) (seq t) ≤ radii t) :
    dist (seq T) (seq 0) ≤ ∑ t ∈ Finset.range T, radii t := by
  induction T with
  | zero =>
    simp
  | succ n ih =>
    have hSteps : ∀ t < n, dist (seq (t + 1)) (seq t) ≤ radii t := by
      intro t ht
      exact hStep t (Nat.lt_trans ht (Nat.lt_succ_self n))
    have hIH := ih hSteps
    rw [Finset.sum_range_succ]
    have hTri : dist (seq (n + 1)) (seq 0) ≤ dist (seq (n + 1)) (seq n) + dist (seq n) (seq 0) :=
      dist_triangle (seq (n + 1)) (seq n) (seq 0)
    have hLast : dist (seq (n + 1)) (seq n) ≤ radii n :=
      hStep n (Nat.lt_succ_self n)
    linarith

/--
Adaptive center drift bound: Under Lipschitz continuity of the population objective with constant `K`,
the drift of the incumbent objective over `T` trust-region steps is bounded by `K * ∑ t, radii t`.
-/
theorem Trajectory.centerDrift
    (f : State → ℝ)
    (seq : ℕ → State)
    (radii : ℕ → ℝ)
    (K : ℝ)
    (T : ℕ)
    (hK : 0 ≤ K)
    (hLip : ∀ x y, |f x - f y| ≤ K * dist x y)
    (hStep : ∀ t < T, dist (seq (t + 1)) (seq t) ≤ radii t) :
    |f (seq T) - f (seq 0)| ≤ K * ∑ t ∈ Finset.range T, radii t := by
  have hDist := Trajectory.distDrift seq radii T hStep
  have hBound := hLip (seq T) (seq 0)
  have hKMul : K * dist (seq T) (seq 0) ≤ K * ∑ t ∈ Finset.range T, radii t :=
    mul_le_mul_of_nonneg_left hDist hK
  linarith

/--
Cumulative population descent: If each accepted step achieves calibrated decrease `progress t`,
the population objective at round `T` strictly improves by at least `∑ t, progress t`.
-/
theorem Trajectory.lossDecr
    (f : ℕ → ℝ)
    (progress : ℕ → ℝ)
    (T : ℕ)
    (hStep : ∀ t < T, f (t + 1) ≤ f t - progress t) :
    f T ≤ f 0 - ∑ t ∈ Finset.range T, progress t := by
  induction T with
  | zero =>
    simp
  | succ n ih =>
    have hSteps : ∀ t < n, f (t + 1) ≤ f t - progress t := by
      intro t ht
      exact hStep t (Nat.lt_trans ht (Nat.lt_succ_self n))
    have hIH := ih hSteps
    rw [Finset.sum_range_succ]
    have hLast := hStep n (Nat.lt_succ_self n)
    linarith

end Drift

end BauxResearch.Optimization
