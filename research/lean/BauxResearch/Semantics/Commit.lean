import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Optimization.EnnCommit

/-!
# Candidate identity at the executed commit

The production backend uses the extracted transition to choose its retained
observation identity. This correspondence does not prove GPU buffer contents,
score accuracy, calibration, or the real interpretation of floating-point gain.
-/

namespace BauxResearch.Semantics.Commit

open BauxResearch.Optimization

abbrev RoundId := BauxResearch.Generated.Optimizer.optimizer.transaction.RoundId

abbrev Valid (active evaluated : RoundId) (candidate : Aeneas.Std.I64) : Prop :=
  active.owner = evaluated.owner ∧ active.proposal = evaluated.proposal ∧
    active.incumbent = evaluated.incumbent ∧ active.incumbent < candidate

/-- The extracted transition checks both identities and a fresh observation ID
before retaining exactly the selected identity. -/
theorem commitMatch (active evaluated : RoundId) (candidate : Aeneas.Std.I64)
    (accepted : Bool) :
    BauxResearch.Generated.Optimizer.optimizer.transaction.commit_round
        active evaluated candidate accepted =
      Aeneas.Std.Result.ok
        (if Valid active evaluated candidate then
          some (if accepted then candidate else active.incumbent) else none) := by
  simp only [BauxResearch.Generated.Optimizer.optimizer.transaction.commit_round, Valid]
  by_cases ho : active.owner = evaluated.owner <;>
    by_cases hp : active.proposal = evaluated.proposal <;>
    by_cases hi : active.incumbent = evaluated.incumbent <;>
    by_cases hf : active.incumbent < candidate <;>
    cases accepted <;> simp_all [not_le]

/-- An incompatible receipt cannot produce a successful retained identity. -/
theorem invalid (active evaluated : RoundId) (candidate : Aeneas.Std.I64)
    (accepted : Bool) (mismatch : ¬ Valid active evaluated candidate) :
    BauxResearch.Generated.Optimizer.optimizer.transaction.commit_round
        active evaluated candidate accepted = Aeneas.Std.Result.ok none := by
  simp [commitMatch, mismatch]

/-- Once the acceptance Boolean represents the mathematical comparison, the
extracted commit realizes `scalarCommit` on logical state identities. Supplying
that Boolean from rounded measurements is a separate numerical obligation. -/
theorem realizes (active : RoundId) (candidate : Aeneas.Std.I64)
    (fresh : active.incumbent < candidate) (gain margin : ℝ) :
    BauxResearch.Generated.Optimizer.optimizer.transaction.commit_round
        active active candidate (decide (EnnCommit.accepts gain margin)) =
      Aeneas.Std.Result.ok
        (some (EnnCommit.scalarCommit active.incumbent candidate 0 gain margin)) := by
  simp [commitMatch, Valid, fresh, EnnCommit.scalarCommit]

#print axioms commitMatch
#print axioms invalid
#print axioms realizes

end BauxResearch.Semantics.Commit
