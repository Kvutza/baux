import BauxResearch.Forward.SelectionCell
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Normed.Module.Convex

/-!
# Local forward sensitivity

A fixed discrete-selection cell is not yet a numerical forward bound. On a
convex trust ball, a uniform bound on the Frechet derivative of the fixed-cell
forward map supplies that bound by Mathlib's mean-value theorem.
-/

namespace BauxResearch.Forward.LocalSensitivity

noncomputable section
open Metric
open scoped NNReal

variable {State Output Index : Type*}
  [NormedAddCommGroup State] [NormedSpace ℝ State]
  [NormedAddCommGroup Output] [NormedSpace ℝ Output]

/-- A uniform Frechet-derivative bound on a trust ball makes the forward map
Lipschitz on that ball. The constant is derived from the derivative rather than
inserted directly as an objective calibration parameter. -/
theorem ballLipschitz
    (forward : State → Output)
    (center : State)
    (radius : ℝ)
    (constant : ℝ≥0)
    (smooth : ∀ state ∈ closedBall center radius,
      DifferentiableAt ℝ forward state)
    (bounded : ∀ state ∈ closedBall center radius,
      ‖fderiv ℝ forward state‖₊ ≤ constant) :
    LipschitzOnWith constant forward (closedBall center radius) := by
  exact (convex_closedBall center radius).lipschitzOnWith_of_nnnorm_fderiv_le
    smooth bounded

/-- The complete local certificate: the score margin keeps the candidate in
the incumbent discrete-selection cell, while a derivative bound on the trust
ball controls the forward-output displacement. -/
theorem cellBound
    (chosen : Set Index)
    (score : State → Index → ℝ)
    (scoreConstant : ℝ≥0)
    (scoreRegular : ∀ index,
      LipschitzWith scoreConstant (fun state => score state index))
    (forward : State → Output)
    (forwardConstant : ℝ≥0)
    (incumbent candidate : State)
    (radius margin : ℝ)
    (separated : SelectionCell.Margin chosen (score incumbent) margin)
    (candidateMember : candidate ∈ closedBall incumbent radius)
    (cellSmall : scoreConstant * radius < margin / 2)
    (smooth : ∀ state ∈ closedBall incumbent radius,
      DifferentiableAt ℝ forward state)
    (bounded : ∀ state ∈ closedBall incumbent radius,
      ‖fderiv ℝ forward state‖₊ ≤ forwardConstant) :
    SelectionCell.Cell chosen (score candidate) ∧
      dist (forward candidate) (forward incumbent) ≤
        forwardConstant * dist candidate incumbent := by
  have radiusBound : dist candidate incumbent ≤ radius := by
    simpa [mem_closedBall, dist_comm] using candidateMember
  have radiusNonnegative : 0 ≤ radius := (dist_nonneg.trans radiusBound)
  have incumbentMember : incumbent ∈ closedBall incumbent radius :=
    mem_closedBall_self radiusNonnegative
  have stable := SelectionCell.ballPreserve chosen score scoreConstant scoreRegular
    incumbent radius margin separated cellSmall
  have selected : SelectionCell.Cell chosen (score candidate) :=
    stable candidate candidateMember
  have regular := ballLipschitz forward incumbent radius forwardConstant smooth bounded
  exact ⟨selected, regular.dist_le_mul candidate candidateMember incumbent incumbentMember⟩

#print axioms ballLipschitz
#print axioms cellBound

end

end BauxResearch.Forward.LocalSensitivity
