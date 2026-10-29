import BauxResearch.Optimization.Neighbor
import BauxResearch.Optimization.Step

/-!
# The finite source fitter and Appendix C's maximization premise

The existing Appendix C development in `formalization/enn-lean` supplies
argmax consistency and fitted-prediction transfer. Those are not re-proved here.
This module checks the real-arithmetic finite fold in `Neighbor.fit`: it
maximizes the supplied sampled LOO score over its trial list, not over the
whole variance-parameter box. Source floating-point validity and nonfinite
score rejection are separate refinement obligations.

The score-gap result keeps criterion error and finite-trial coverage error
explicit. Neither is assumed to vanish, and neither is an optimizer regret
bound. The source RNG is not assigned a distribution by these theorems.
-/

namespace BauxResearch.Optimization.Fitting

noncomputable section

/-- The actual trial list: proposed scales at fixed neighbor count, followed
by the previous parameters. Later trials win score ties in `Neighbor.fit`. -/
def trials (previous : Neighbor.Parameters) (proposals : List (ℝ × ℝ)) :
    List Neighbor.Parameters :=
  proposals.map (fun pair =>
    { neighbors := previous.neighbors, epistemic := pair.1, aleatoric := pair.2 }) ++
      [previous]

/-- Appendix C uses an average criterion; the source accumulates a sum.
For a nonempty fixed held-row list, dividing by its length preserves ranking.
This definition alone does not identify the average with a population score. -/
def meanScore {capacity : ℕ} (positive : 0 < capacity)
    (data : Neighbor.Data capacity) (parameters : Neighbor.Parameters)
    (held : List (Fin capacity)) : ℝ :=
  Neighbor.likelihood positive data parameters (Neighbor.scale data) held / held.length

private theorem foldMax {Choice : Type*} (score : Choice → ℝ)
    (choices : List Choice) (initial : Choice) :
    let result := choices.foldl (fun best candidate =>
      if score best ≤ score candidate then candidate else best) initial
    score initial ≤ score result ∧
      ∀ candidate ∈ choices, score candidate ≤ score result := by
  induction choices generalizing initial with
  | nil => simp
  | cons candidate remaining induction =>
      simp only [List.foldl_cons]
      split_ifs with ordered
      · obtain ⟨initialBound, trialBound⟩ := induction candidate
        refine ⟨ordered.trans initialBound, ?_⟩
        intro choice member
        rcases List.mem_cons.mp member with same | rest
        · simpa only [same] using initialBound
        · exact trialBound choice rest
      · obtain ⟨initialBound, trialBound⟩ := induction initial
        refine ⟨initialBound, ?_⟩
        intro choice member
        rcases List.mem_cons.mp member with same | rest
        · exact same ▸ (le_of_not_ge ordered).trans initialBound
        · exact trialBound choice rest

/-- The source-shaped fold supplies empirical maximality over the finite
trial set for any supplied held-out rows. It supplies no whole-box maximum. -/
theorem trialMax {capacity : ℕ} (positive : 0 < capacity)
    (data : Neighbor.Data capacity) (previous : Neighbor.Parameters)
    (proposals : List (ℝ × ℝ)) (held : List (Fin capacity))
    (comparison : Neighbor.Parameters) (member : comparison ∈ trials previous proposals) :
    Neighbor.likelihood positive data comparison (Neighbor.scale data) held ≤
      Neighbor.likelihood positive data
        (Neighbor.fit positive data previous proposals held) (Neighbor.scale data) held := by
  exact (foldMax
    (fun parameters => Neighbor.likelihood positive data parameters (Neighbor.scale data) held)
    (trials previous proposals) previous).2 comparison member

/-- The finite fitter's target-criterion deficit consists of score error and
trial coverage error. `target` may be the population LOO criterion in Appendix
C.4, but that identification and both error bounds must be justified. -/
theorem scoreGap {capacity : ℕ} (positive : 0 < capacity)
    (data : Neighbor.Data capacity) (previous : Neighbor.Parameters)
    (proposals : List (ℝ × ℝ)) (held : List (Fin capacity))
    (heldPositive : 0 < held.length)
    (target : Neighbor.Parameters → ℝ) (reference covered : Neighbor.Parameters)
    (error coverage : ℝ)
    (accurate : UniformError target (fun parameters => meanScore positive data parameters held) error)
    (member : covered ∈ trials previous proposals)
    (cover : target reference ≤ target covered + coverage) :
    target reference - target (Neighbor.fit positive data previous proposals held) ≤
      coverage + 2 * error := by
  have maximal : meanScore positive data covered held ≤
      meanScore positive data (Neighbor.fit positive data previous proposals held) held :=
    div_le_div_of_nonneg_right
      (trialMax positive data previous proposals held covered member)
      (le_of_lt (Nat.cast_pos.mpr heldPositive))
  have comparisonError := (abs_le.mp (accurate covered)).1
  have fittedError := (abs_le.mp
    (accurate (Neighbor.fit positive data previous proposals held))).2
  linarith

#print axioms trialMax
#print axioms scoreGap

end
end BauxResearch.Optimization.Fitting
