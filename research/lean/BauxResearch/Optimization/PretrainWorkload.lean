import BauxResearch.Foundation
import Mathlib.Basic.Real.Basic

/-!
# Sample and budget complexity of pretraining workloads

A pretraining or optimization horizon is expressed by the total evaluation budget
consumed (e.g. evaluated sample units, sequences, or tokens). Optimizer
transactions partition this evaluation stream into batches or windows of size `w`.
This module formalizes budget accounting and quality threshold achievement in a
domain-agnostic manner.
-/

/-!
Let `B(n) = ∑_{t<n} costAt(t)` be the cumulative evaluation budget consumed after
`n` transactions.  Under a fixed window protocol of size `w`, `B(n) = n * w`.
This is accounting, not a claim that `n` transactions suffice for convergence;
the latter depends on the optimization policy and population loss path.
-/

namespace BauxResearch.Optimization.PretrainWorkload

open scoped BigOperators

/-- Cumulative evaluated budget units after an arbitrary number of transactions. -/
def budgetClock (costAt : ℕ → ℕ) (steps : ℕ) : ℕ :=
  ∑ step ∈ Finset.range steps, costAt step

theorem budgetSucc (costAt : ℕ → ℕ) (steps : ℕ) :
    budgetClock costAt (steps + 1) = budgetClock costAt steps + costAt steps := by
  simp [budgetClock, Finset.sum_range_succ]

/-- Under a fixed window protocol of size `windowSize`, `steps` transactions consume
exactly `steps * windowSize` evaluation budget units. -/
theorem budgetFixed (windowSize steps : ℕ) :
    budgetClock (fun _ => windowSize) steps = steps * windowSize := by
  simp [budgetClock]

/-- A declared quality criterion stated as a loss threshold reached within a budget. -/
structure BudgetGoal where
  lossAfter : ℕ → ℝ
  threshold : ℝ

/-- The quality criterion has been reached after consuming this evaluation budget. -/
def BudgetGoal.reached (goal : BudgetGoal) (budget : ℕ) : Prop :=
  goal.lossAfter budget ≤ goal.threshold

/-- A budget is sufficient when the criterion is reached within it. -/
def BudgetGoal.sufficient (goal : BudgetGoal) (budget : ℕ) : Prop :=
  ∃ consumed ≤ budget, goal.reached consumed

/-- A budget requirement is the minimal sufficient budget for the declared criterion. -/
def BudgetGoal.required (goal : BudgetGoal) (budget : ℕ) : Prop :=
  goal.sufficient budget ∧ ∀ smaller < budget, ¬goal.sufficient smaller

theorem BudgetGoal.sufficientMono
    {goal : BudgetGoal} {smaller larger : ℕ}
    (enough : goal.sufficient smaller) (order : smaller ≤ larger) :
    goal.sufficient larger := by
  obtain ⟨consumed, within, reached⟩ := enough
  exact ⟨consumed, within.trans order, reached⟩

theorem BudgetGoal.requiredUnique
    {goal : BudgetGoal} {left right : ℕ}
    (leftReq : goal.required left) (rightReq : goal.required right) :
    left = right := by
  rcases leftReq with ⟨leftEnough, leftLeast⟩
  rcases rightReq with ⟨rightEnough, rightLeast⟩
  apply Nat.le_antisymm
  · by_contra notOrder
    exact leftLeast right (Nat.lt_of_not_ge notOrder) rightEnough
  · by_contra notOrder
    exact rightLeast left (Nat.lt_of_not_ge notOrder) leftEnough

/-- Backward-compatible structure for token-based workloads. -/
structure LossGoal where
  lossAfter : ℕ → ℝ
  threshold : ℝ

def LossGoal.toBudget (goal : LossGoal) : BudgetGoal where
  lossAfter := goal.lossAfter
  threshold := goal.threshold

def LossGoal.reached (goal : LossGoal) (tokens : ℕ) : Prop :=
  goal.lossAfter tokens ≤ goal.threshold

def LossGoal.sufficient (goal : LossGoal) (budget : ℕ) : Prop :=
  ∃ tokens ≤ budget, goal.reached tokens

def LossGoal.required (goal : LossGoal) (budget : ℕ) : Prop :=
  goal.sufficient budget ∧ ∀ smaller < budget, ¬goal.sufficient smaller

theorem LossGoal.sufficientMono
    {goal : LossGoal} {smaller larger : ℕ}
    (enough : goal.sufficient smaller) (order : smaller ≤ larger) :
    goal.sufficient larger := by
  obtain ⟨tokens, within, reached⟩ := enough
  exact ⟨tokens, within.trans order, reached⟩

theorem LossGoal.requiredUnique
    {goal : LossGoal} {left right : ℕ}
    (leftReq : goal.required left) (rightReq : goal.required right) :
    left = right := by
  rcases leftReq with ⟨leftEnough, leftLeast⟩
  rcases rightReq with ⟨rightEnough, rightLeast⟩
  apply Nat.le_antisymm
  · by_contra notOrder
    exact leftLeast right (Nat.lt_of_not_ge notOrder) rightEnough
  · by_contra notOrder
    exact rightLeast left (Nat.lt_of_not_ge notOrder) leftEnough

/-- Backward-compatible alias for token-based workloads. -/
abbrev tokenClock := budgetClock

theorem clockSucc (tokensAt : ℕ → ℕ) (steps : ℕ) :
    tokenClock tokensAt (steps + 1) = tokenClock tokensAt steps + tokensAt steps :=
  budgetSucc tokensAt steps

/-- Evaluation window size specialized to the benchmark configuration. -/
def windowTokens : ℕ := currentTarget.contextTokens

theorem clockFixed (steps : ℕ) :
    tokenClock (fun _ => windowTokens) steps = steps * windowTokens :=
  budgetFixed windowTokens steps

/-- Fixed-corpus data-efficiency protocol (`qlabs-eng/slowrun` regime):
`uniqueTokens` is the fixed corpus size (e.g., `100,000,000` FineWeb tokens),
`valLoss` maps `(uniqueTokens, evaluatedTokens)` to held-out validation NLL. -/
structure CorpusGoal where
  uniqueTokens : ℕ
  valLoss : ℕ → ℕ → ℝ
  threshold : ℝ

/-- In the `slowrun` regime, an optimizer reaches `threshold` on the fixed
`uniqueTokens` corpus within `evalBudget` evaluated tokens. -/
def CorpusGoal.reached (goal : CorpusGoal) (evalBudget : ℕ) : Prop :=
  goal.valLoss goal.uniqueTokens evalBudget ≤ goal.threshold

/-- `left` dominates `right` on a fixed `uniqueTokens` corpus when it achieves
strictly lower validation loss at `evalBudget` (`slowrun` leaderboard order). -/
def CorpusGoal.dominates (uniqueTokens evalBudget : ℕ)
    (leftVal rightVal : ℕ → ℕ → ℝ) : Prop :=
  leftVal uniqueTokens evalBudget < rightVal uniqueTokens evalBudget

/-- Under `epochs` passes over a fixed corpus of `uniqueTokens` partitioned into
`windowsPerEpoch` windows of size `windowSize` (`uniqueTokens = windowsPerEpoch * windowSize`),
the evaluated-token clock equals `epochs * uniqueTokens` while the unique-corpus
sample budget remains `uniqueTokens`. -/
theorem CorpusGoal.epochEval (uniqueTokens windows windowSize epochs : ℕ)
    (hPart : uniqueTokens = windows * windowSize) :
    budgetClock (fun _ => windowSize) (epochs * windows) = epochs * uniqueTokens := by
  rw [budgetFixed, hPart, Nat.mul_assoc]

/-- Complete transaction target latency in milliseconds. -/
def targetMs : ℕ := currentTarget.targetMs

/-- Model parameter count for the `fbt-pisa1-mhc4-v1` benchmark target. -/
def paramCount : ℕ := currentTarget.parameterCount

/-- Complete candidate states evaluated per transaction. -/
def candidateCount : ℕ := currentTarget.candidates

/-- Nearest neighbors queried by the ENN surrogate. -/
def neighborK : ℕ := currentTarget.neighbors

theorem activeContract :
    windowTokens = currentTarget.contextTokens ∧
    targetMs = currentTarget.targetMs ∧
    paramCount = currentTarget.parameterCount ∧
    candidateCount = 1 ∧ neighborK = 10 := by
  decide

end BauxResearch.Optimization.PretrainWorkload
