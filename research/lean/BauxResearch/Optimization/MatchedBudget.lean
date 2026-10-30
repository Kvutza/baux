import BauxResearch.Optimization.PretrainWorkload

/-!
# Matched-budget optimizer comparison

Optimizer rounds are not automatically comparable: one Baux transaction and one
Muon update can consume different evaluated-token budgets. This module therefore
compares complete stateful runs only at checkpoints where their cumulative work
is equal. Lower loss is better.

The local composition theorems expose the empirical obligation directly. A Baux
win at a matched checkpoint follows from equal starting loss and no-smaller loss
reduction on every preceding matched-work interval. The theorem does not assume
that Baux dominates Muon, and it does not identify an instantaneous, momentum-free
Muon diagnostic with a Muon trajectory.
-/

namespace BauxResearch.Optimization.MatchedBudget

open PretrainWorkload

/-- The common state space and loss used by every optimizer in a comparison. -/
structure ObjectiveWorld where
  State : Type
  loss : State → ℝ

/-- A complete optimizer trajectory, with cumulative work and state after each round. -/
structure Run (world : ObjectiveWorld) where
  workAfter : ℕ → ℕ
  stateAfter : ℕ → world.State

def Run.lossAfter {world : ObjectiveWorld} (run : Run world) (round : ℕ) : ℝ :=
  world.loss (run.stateAfter round)

/-- Checkpoints pairing possibly different round counts at exactly equal cumulative work. -/
structure Schedule {world : ObjectiveWorld} (left right : Run world) where
  budget : ℕ → ℕ
  leftRound : ℕ → ℕ
  rightRound : ℕ → ℕ
  zeroBudget : budget 0 = 0
  strictlyIncreasing : StrictMono budget
  leftStart : leftRound 0 = 0
  rightStart : rightRound 0 = 0
  sameInitial : left.stateAfter 0 = right.stateAfter 0
  leftWork : ∀ checkpoint, left.workAfter (leftRound checkpoint) = budget checkpoint
  rightWork : ∀ checkpoint, right.workAfter (rightRound checkpoint) = budget checkpoint

def leftLoss {world : ObjectiveWorld} {left right : Run world}
    (schedule : Schedule left right) (checkpoint : ℕ) : ℝ :=
  left.lossAfter (schedule.leftRound checkpoint)

def rightLoss {world : ObjectiveWorld} {left right : Run world}
    (schedule : Schedule left right) (checkpoint : ℕ) : ℝ :=
  right.lossAfter (schedule.rightRound checkpoint)

/-- Loss reduction across one interval of equal-work checkpoints. -/
def gain (loss : ℕ → ℝ) (checkpoint : ℕ) : ℝ :=
  loss checkpoint - loss (checkpoint + 1)

/-- Baux is no worse than Muon at this common-work checkpoint. -/
def DominatesAt {world : ObjectiveWorld} {left right : Run world} (schedule : Schedule left right)
    (checkpoint : ℕ) : Prop :=
  leftLoss schedule checkpoint ≤ rightLoss schedule checkpoint

/-- Baux beats Muon at this common-work checkpoint. -/
def StrictlyDominatesAt {world : ObjectiveWorld} {left right : Run world}
    (schedule : Schedule left right)
    (checkpoint : ℕ) : Prop :=
  leftLoss schedule checkpoint < rightLoss schedule checkpoint

theorem initialLoss_eq
    {world : ObjectiveWorld} {left right : Run world} (schedule : Schedule left right) :
    leftLoss schedule 0 = rightLoss schedule 0 := by
  change world.loss (left.stateAfter (schedule.leftRound 0)) =
    world.loss (right.stateAfter (schedule.rightRound 0))
  rw [schedule.leftStart, schedule.rightStart, schedule.sameInitial]

/-- Matched-interval gain dominance composes to matched-budget loss dominance. -/
theorem dominates_of_local_gain
    {world : ObjectiveWorld} {left right : Run world}
    (schedule : Schedule left right) (checkpoint : ℕ)
    (intervalGain : ∀ step < checkpoint,
      gain (rightLoss schedule) step ≤ gain (leftLoss schedule) step) :
    DominatesAt schedule checkpoint := by
  induction checkpoint with
  | zero => exact (initialLoss_eq schedule).le
  | succ checkpoint induction =>
      have previous : DominatesAt schedule checkpoint :=
        induction fun step before ↦ intervalGain step (before.trans (Nat.lt_succ_self checkpoint))
      have last := intervalGain checkpoint (Nat.lt_succ_self checkpoint)
      change leftLoss schedule checkpoint ≤ rightLoss schedule checkpoint at previous
      change rightLoss schedule checkpoint - rightLoss schedule (checkpoint + 1) ≤
        leftLoss schedule checkpoint - leftLoss schedule (checkpoint + 1) at last
      change leftLoss schedule (checkpoint + 1) ≤ rightLoss schedule (checkpoint + 1)
      linarith

/-- A strict local gain advantage on every nonempty matched interval gives a strict win. -/
theorem strictlyDominates_of_local_gain
    {world : ObjectiveWorld} {left right : Run world}
    (schedule : Schedule left right) (checkpoint : ℕ)
    (positive : 0 < checkpoint)
    (intervalGain : ∀ step < checkpoint,
      gain (rightLoss schedule) step < gain (leftLoss schedule) step) :
    StrictlyDominatesAt schedule checkpoint := by
  obtain ⟨previousCheckpoint, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt positive)
  have previous : DominatesAt schedule previousCheckpoint :=
    dominates_of_local_gain schedule previousCheckpoint fun step before ↦
      (intervalGain step (before.trans (Nat.lt_succ_self previousCheckpoint))).le
  have last := intervalGain previousCheckpoint (Nat.lt_succ_self previousCheckpoint)
  change leftLoss schedule previousCheckpoint ≤ rightLoss schedule previousCheckpoint at previous
  change rightLoss schedule previousCheckpoint - rightLoss schedule (previousCheckpoint + 1) <
    leftLoss schedule previousCheckpoint - leftLoss schedule (previousCheckpoint + 1) at last
  change leftLoss schedule (previousCheckpoint + 1) <
    rightLoss schedule (previousCheckpoint + 1)
  linarith

/-- At an epoch checkpoint, the shared budget is exactly evaluated corpus tokens. -/
theorem strictlyDominates_at_epoch
    {world : ObjectiveWorld} {left right : Run world} (schedule : Schedule left right)
    (uniqueTokens windows windowSize epochs checkpoint : ℕ)
    (partition : uniqueTokens = windows * windowSize)
    (checkpointCost : schedule.budget checkpoint =
      budgetClock (fun _ ↦ windowSize) (epochs * windows))
    (positive : 0 < checkpoint)
    (intervalGain : ∀ step < checkpoint,
      gain (rightLoss schedule) step < gain (leftLoss schedule) step) :
    schedule.budget checkpoint = epochs * uniqueTokens ∧
      StrictlyDominatesAt schedule checkpoint := by
  constructor
  · rw [checkpointCost, CorpusGoal.epochEval uniqueTokens windows windowSize epochs partition]
  · exact strictlyDominates_of_local_gain schedule checkpoint positive intervalGain

/-- A linear toy run used to separate round count from cumulative work. -/
def scalarLoss : ObjectiveWorld where
  State := ℝ
  loss := id

def linearRun (initial gainPerRound : ℝ) (workPerRound : ℕ) : Run scalarLoss where
  workAfter rounds := rounds * workPerRound
  stateAfter rounds := initial - (rounds : ℝ) * gainPerRound

/-- Winning at equal round index can reverse when the compared work is made equal. -/
theorem raw_round_win_does_not_imply_matched_budget_win :
    let baux := linearRun 10 2 2
    let muon := linearRun 10 (3 / 2) 1
    baux.lossAfter 1 < muon.lossAfter 1 ∧
      baux.workAfter 1 = muon.workAfter 2 ∧
      muon.lossAfter 2 < baux.lossAfter 1 := by
  dsimp [linearRun, Run.lossAfter, scalarLoss]
  norm_num

/-- Equal work and a shared initial state do not supply the missing gain comparison. -/
def matchedButLosing :
    Schedule (linearRun 10 1 1) (linearRun 10 2 1) where
  budget := id
  leftRound := id
  rightRound := id
  zeroBudget := rfl
  strictlyIncreasing := strictMono_id
  leftStart := rfl
  rightStart := rfl
  sameInitial := by
    dsimp [linearRun, scalarLoss]
    norm_num
  leftWork checkpoint := by simp [linearRun]
  rightWork checkpoint := by simp [linearRun]

theorem matched_cost_does_not_imply_dominance :
    ¬DominatesAt matchedButLosing 1 := by
  dsimp [DominatesAt, leftLoss, rightLoss, matchedButLosing, linearRun, Run.lossAfter,
    scalarLoss]
  norm_num

#print axioms initialLoss_eq
#print axioms dominates_of_local_gain
#print axioms strictlyDominates_of_local_gain
#print axioms strictlyDominates_at_epoch
#print axioms raw_round_win_does_not_imply_matched_budget_win
#print axioms matched_cost_does_not_imply_dominance

end BauxResearch.Optimization.MatchedBudget
