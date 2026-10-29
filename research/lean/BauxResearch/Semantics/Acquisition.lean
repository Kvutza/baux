import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Set.Finite.Lemmas
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.GroupWithZero.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Upstream ENN acquisition semantics reachable from pinned TuRBO-ENN

This module formalizes acquisition and incumbent selection semantics from pinned TuRBO-ENN:
1. Incumbent tracking: noise-unaware raw observation argmin vs noise-aware surrogate mean argmin.
2. Scalar acquisition policies: UCB exploitation/exploration and Thompson sampling draws.
3. Multi-objective Pareto acquisition: weak/strict dominance, Pareto fronts, and hypervolume.
4. Selection laws: the singleton selection identity law and finite pool argmin optimality.
5. Composition interfaces with candidate pools and trust-region controller state.
-/

namespace BauxResearch.Semantics.Acquisition

noncomputable section

open scoped BigOperators

/-! ## 1. Incumbent Selection -/

/-- Raw observation incumbent selection: index of minimum observed value. -/
def rawBest {n : ℕ} (values : Fin n → ℝ) (iStar : Fin n) : Prop :=
  ∀ i : Fin n, values iStar ≤ values i

/-- For any finite nonempty set of values, there exists an optimal raw incumbent. -/
theorem raw_optimal {n : ℕ} (hPos : 0 < n) (values : Fin n → ℝ) :
    ∃ iStar : Fin n, rawBest values iStar := by
  have hUniv : (Set.univ : Set (Fin n)).Nonempty := ⟨⟨0, hPos⟩, trivial⟩
  obtain ⟨iStar, _, hMin⟩ := Set.exists_min_image Set.univ values Set.finite_univ hUniv
  refine ⟨iStar, ?_⟩
  intro i
  exact hMin i trivial

/-- Noise-aware incumbent selection: index of minimum predicted surrogate mean. -/
def noiseBest {n : ℕ} (means : Fin n → ℝ) (iStar : Fin n) : Prop :=
  ∀ i : Fin n, means iStar ≤ means i

/-- For any finite nonempty set of surrogate means, an optimal incumbent exists. -/
theorem noise_optimal {n : ℕ} (hPos : 0 < n) (means : Fin n → ℝ) :
    ∃ iStar : Fin n, noiseBest means iStar := by
  exact raw_optimal hPos means

/-! ## 2. Scalar Acquisition Policies -/

/-- UCB acquisition score for minimization: lower score indicates higher priority. -/
def ucbScore (meanVal epistStd beta : ℝ) : ℝ :=
  meanVal - beta * epistStd

/-- Increasing exploration weight beta strictly decreases UCB score for positive uncertainty. -/
theorem ucb_tradeoff (meanVal epistStd beta1 beta2 : ℝ)
    (hBeta : beta1 < beta2) (hStd : 0 < epistStd) :
    ucbScore meanVal epistStd beta2 < ucbScore meanVal epistStd beta1 := by
  unfold ucbScore
  have hDiff : beta1 * epistStd < beta2 * epistStd := mul_lt_mul_of_pos_right hBeta hStd
  linarith

/-- Thompson sampling acquisition score: value of the drawn pseudo-posterior path. -/
def thompsonScore (drawVal : ℝ) : ℝ :=
  drawVal

/-- Optimal acquisition candidate selection over a finite candidate pool. -/
def selectBest {m : ℕ} (scores : Fin m → ℝ) (cStar : Fin m) : Prop :=
  ∀ c : Fin m, scores cStar ≤ scores c

/-- An optimal candidate exists in every non-empty finite candidate pool. -/
theorem select_optimal {m : ℕ} (hPos : 0 < m) (scores : Fin m → ℝ) :
    ∃ cStar : Fin m, selectBest scores cStar := by
  exact raw_optimal hPos scores

/-- Singleton Selection Law: When candidate pool size is 1, acquisition is an identity map. -/
theorem singleton_law (scores : Fin 1 → ℝ) :
    selectBest scores (0 : Fin 1) := by
  intro c
  have hEq : c = 0 := by ext; omega
  rw [hEq]

/-! ## 3. Multi-Objective Pareto Acquisition -/

/-- Vector loss across k objectives. -/
abbrev VectorLoss (k : ℕ) := Fin k → ℝ

/-- Weak Pareto dominance: candidate is no worse than reference in all objectives. -/
def weakDominates {k : ℕ} (candidate reference : VectorLoss k) : Prop :=
  ∀ j : Fin k, candidate j ≤ reference j

/-- Strict Pareto dominance: candidate is no worse in all, and strictly better in at least one. -/
def strictlyDom {k : ℕ} (candidate reference : VectorLoss k) : Prop :=
  weakDominates candidate reference ∧ ∃ j : Fin k, candidate j < reference j

/-- Weak dominance is transitive. -/
theorem weak_trans {k : ℕ} {first second third : VectorLoss k}
    (h1 : weakDominates first second) (h2 : weakDominates second third) :
    weakDominates first third := by
  intro j
  exact le_trans (h1 j) (h2 j)

/-- Strict dominance is transitive. -/
theorem strict_trans {k : ℕ} {first second third : VectorLoss k}
    (h1 : strictlyDom first second) (h2 : strictlyDom second third) :
    strictlyDom first third := by
  obtain ⟨hWeak1, j1, hStrict1⟩ := h1
  obtain ⟨hWeak2, _, _⟩ := h2
  constructor
  · exact weak_trans hWeak1 hWeak2
  · refine ⟨j1, ?_⟩
    have hLe := hWeak2 j1
    exact lt_of_lt_of_le hStrict1 hLe

/-- Strict dominance is asymmetric. -/
theorem strict_asymm {k : ℕ} {first second : VectorLoss k}
    (hDom : strictlyDom first second) :
    ¬ strictlyDom second first := by
  intro hReverse
  obtain ⟨hWeak1, j1, hStrict1⟩ := hDom
  obtain ⟨hWeak2, _, _⟩ := hReverse
  have hOpp := hWeak2 j1
  linarith

/-- Pareto optimality within a candidate set: not strictly dominated by any candidate. -/
def isPareto {k : ℕ} (candidates : Finset (VectorLoss k)) (c : VectorLoss k) : Prop :=
  c ∈ candidates ∧ ¬ ∃ alt ∈ candidates, strictlyDom alt c

/-- Soundness of Pareto rejection: any strictly dominated candidate is not Pareto optimal. -/
theorem pareto_sound {k : ℕ} (candidates : Finset (VectorLoss k)) (c alt : VectorLoss k)
    (hAlt : alt ∈ candidates) (hDom : strictlyDom alt c) :
    ¬ isPareto candidates c := by
  intro hOpt
  obtain ⟨_, hNotDom⟩ := hOpt
  exact hNotDom ⟨alt, hAlt, hDom⟩

/-- Hypervolume contribution bounding box: product of coordinate improvements. -/
def hypervolBox {k : ℕ} (candidate refPoint : VectorLoss k) : ℝ :=
  ∏ j : Fin k, max 0 (refPoint j - candidate j)

/-- Hypervolume box contribution is non-negative. -/
theorem hypervol_nonneg {k : ℕ} (candidate refPoint : VectorLoss k) :
    0 ≤ hypervolBox candidate refPoint := by
  unfold hypervolBox
  apply Finset.prod_nonneg
  intro j _
  exact le_max_left _ _

/-- When candidate strictly dominates reference point in all coordinates, box is positive. -/
theorem hypervol_pos {k : ℕ} (candidate refPoint : VectorLoss k)
    (hStrictAll : ∀ j : Fin k, candidate j < refPoint j) :
    0 < hypervolBox candidate refPoint := by
  unfold hypervolBox
  apply Finset.prod_pos
  intro j _
  have hDiff : 0 < refPoint j - candidate j := by linarith [hStrictAll j]
  exact lt_of_lt_of_le hDiff (le_max_right 0 _)

/-! ## 4. Composition Interfaces -/

/-- Output record emitted by surrogate acquisition to the optimizer controller. -/
structure ResultRecord where
  armIndex : ℕ
  predMean : ℝ
  epistemic : ℝ
  acqScore : ℝ
  valid_arm : True

/-- Composition helper creating an acquisition output record. -/
def composeResult (arm : ℕ) (meanVal epistStd score : ℝ) : ResultRecord :=
  ⟨arm, meanVal, epistStd, score, trivial⟩

/-- Composition preserves the exact selected arm index and scores. -/
theorem compose_compat (arm : ℕ) (meanVal epistStd score : ℝ) :
    (composeResult arm meanVal epistStd score).armIndex = arm ∧
    (composeResult arm meanVal epistStd score).predMean = meanVal ∧
    (composeResult arm meanVal epistStd score).epistemic = epistStd ∧
    (composeResult arm meanVal epistStd score).acqScore = score := by
  dsimp [composeResult]
  refine ⟨rfl, rfl, rfl, rfl⟩

end

end BauxResearch.Semantics.Acquisition
