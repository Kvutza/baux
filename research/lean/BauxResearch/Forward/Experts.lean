import BauxResearch.Forward.Algebra
import BauxResearch.Semantics.FbtProgram

/-!
At an input `x`, the router selects a finite expert list.  The branch output
is the shared expert output plus a weighted sum over that list; each weight is
a sigmoid score divided by the sum of sigmoid scores on the selected list.
This is not a softmax over all experts.  The supplied arithmetic's store
operation is part of each evaluated expert and of the branch output.
-/

namespace BauxResearch.Forward.Experts
noncomputable section
open scoped BigOperators

structure Expert (width hidden : ℕ) where
  gate : Fin width → Fin hidden → ℝ
  up : Fin width → Fin hidden → ℝ
  down : Fin hidden → Fin width → ℝ

def evaluate {width hidden : ℕ} (math : Arithmetic)
    (weights : Expert width hidden) (input : Vector width) : Vector width :=
  let gate := project math weights.gate input
  let up := project math weights.up input
  project math weights.down (fun index => math.store (silu (gate index) * up index))

structure Weights (width hidden experts : ℕ) where
  router : Fin width → Fin experts → ℝ
  routed : Fin experts → Expert width hidden
  shared : Expert width hidden

def selected {experts : ℕ} (count : ℕ) (scores : Vector experts) : List (Fin experts) :=
  (rank scores).take count

/-- Source uses sigmoid normalization, not softmax over router logits. -/
def routeWeight {experts : ℕ} (math : Arithmetic) (routes : List (Fin experts))
    (scores : Vector experts) (index : Fin experts) : ℝ :=
  math.store (sigmoid (scores index) / (routes.map (fun other => sigmoid (scores other))).sum)

def branch {width hidden experts : ℕ} (math : Arithmetic) (count : ℕ)
    (weights : Weights width hidden experts) (input : Vector width) : Vector width :=
  let scores := project math weights.router input
  let routes := selected count scores
  let shared := evaluate math weights.shared input
  fun column => math.store (shared column + (routes.map fun index =>
    routeWeight math routes scores index * evaluate math (weights.routed index) input column).sum)

/-- At a fixed input, changing only routed experts outside the selected list
does not change this branch's output. This is a local forward-semantic fact,
not a statement about other tokens or later routing decisions. -/
theorem routeOnly {width hidden experts : ℕ} (math : Arithmetic) (count : ℕ)
    (left right : Weights width hidden experts) (input : Vector width)
    (routerEq : left.router = right.router)
    (sharedEq : left.shared = right.shared)
    (routedEq : ∀ index ∈ selected count (project math left.router input),
      left.routed index = right.routed index) :
    branch math count left input = branch math count right input := by
  rw [routerEq] at routedEq
  let scores := project math right.router input
  let routes := selected count scores
  have mapEq (column : Fin width) :
      (routes.map fun index => routeWeight math routes scores index *
        evaluate math (left.routed index) input column) =
      (routes.map fun index => routeWeight math routes scores index *
        evaluate math (right.routed index) input column) := by
    apply List.map_congr_left
    intro index member
    rw [routedEq index member]
  funext column
  simpa only [branch, routerEq, sharedEq] using
    congrArg (fun values : List ℝ =>
      math.store (evaluate math right.shared input column + values.sum)) (mapEq column)

/-- Selecting fewer routes than experts leaves at least one expert unselected
for this input. No assumption about how `rank` orders scores is needed. -/
theorem unusedExpert {experts : ℕ} (count : ℕ) (scores : Vector experts)
    (fewer : count < experts) :
    ∃ index : Fin experts, index ∉ selected count scores := by
  classical
  have smaller : (selected count scores).toFinset.card <
      (Finset.univ : Finset (Fin experts)).card := by
    calc
      (selected count scores).toFinset.card ≤ (selected count scores).length :=
        List.toFinset_card_le _
      _ ≤ count := by simp [selected]
      _ < experts := fewer
      _ = (Finset.univ : Finset (Fin experts)).card := by simp
  obtain ⟨index, _, outside⟩ := Finset.exists_mem_notMem_of_card_lt_card smaller
  exact ⟨index, by simpa using outside⟩

/-- A single input cannot observe arbitrary replacement of an unselected
routed expert, provided the router and shared expert are unchanged. -/
theorem unusedSwap {width hidden experts : ℕ} (math : Arithmetic) (count : ℕ)
    (weights : Weights width hidden experts) (input : Vector width)
    (replacement : Expert width hidden) (fewer : count < experts) :
    ∃ index : Fin experts,
      branch math count
        { weights with routed := fun other =>
            if other = index then replacement else weights.routed other } input =
      branch math count weights input := by
  obtain ⟨index, outside⟩ := unusedExpert count (project math weights.router input) fewer
  refine ⟨index, ?_⟩
  apply routeOnly math count _ _ input
  · rfl
  · rfl
  · intro other member
    have different : other ≠ index := by
      intro equal
      exact outside (equal ▸ member)
    simp [different]

#print axioms routeOnly
#print axioms unusedExpert
#print axioms unusedSwap

/-- Scalar score-order stability; whole top-k set invariance remains a separate proof. -/
theorem margin (chosen other margin shiftChosen shiftOther : ℝ)
    (separated : other + margin ≤ chosen)
    (chosenBound : |shiftChosen| < margin / 2)
    (otherBound : |shiftOther| < margin / 2) :
    other + shiftOther < chosen + shiftChosen :=
  BauxResearch.Semantics.FbtProgram.cellInvar chosen other margin shiftChosen shiftOther
    separated chosenBound otherBound

end
end BauxResearch.Forward.Experts
