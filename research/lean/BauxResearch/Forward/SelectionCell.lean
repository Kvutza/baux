import Mathlib.Topology.MetricSpace.Lipschitz

/-!
# Stable discrete-selection cells

Sparse attention and routed experts are not globally smooth at selection ties.
This file describes the local cells on which a chosen set is strictly separated
from its complement. A score margin and a Lipschitz trust-radius condition keep
the next parameter state inside the same cell.
-/

namespace BauxResearch.Forward.SelectionCell

noncomputable section
open scoped NNReal

variable {Index State : Type*}

/-- Every chosen index scores strictly above every unchosen index. -/
def Cell (chosen : Set Index) (score : Index → ℝ) : Prop :=
  ∀ selected, selected ∈ chosen → ∀ other, other ∉ chosen →
    score other < score selected

/-- A non-strict positive margin between the chosen and unchosen scores. -/
def Margin (chosen : Set Index) (score : Index → ℝ) (margin : ℝ) : Prop :=
  ∀ selected, selected ∈ chosen → ∀ other, other ∉ chosen →
    score other + margin ≤ score selected

/-- Perturbing every score by less than half the separating margin preserves
the strict chosen-versus-unchosen order. -/
theorem marginPreserve
    (chosen : Set Index)
    (before after : Index → ℝ)
    (margin : ℝ)
    (separated : Margin chosen before margin)
    (shift : ∀ index, |after index - before index| < margin / 2) :
    Cell chosen after := by
  intro selected selectedMember other otherMember
  have selectedShift := abs_lt.mp (shift selected)
  have otherShift := abs_lt.mp (shift other)
  have base := separated selected selectedMember other otherMember
  linarith

/-- Coordinatewise Lipschitz score maps stay in the same selection cell when
the trust-region displacement is smaller than half the incumbent margin after
scaling by the score sensitivity. -/
theorem radiusPreserve
    [PseudoMetricSpace State]
    (chosen : Set Index)
    (score : State → Index → ℝ)
    (constant : ℝ≥0)
    (regular : ∀ index, LipschitzWith constant (fun state => score state index))
    (incumbent candidate : State)
    (margin : ℝ)
    (separated : Margin chosen (score incumbent) margin)
    (small : constant * dist candidate incumbent < margin / 2) :
    Cell chosen (score candidate) := by
  apply marginPreserve chosen (score incumbent) (score candidate) margin separated
  intro index
  calc
    |score candidate index - score incumbent index| =
        dist (score candidate index) (score incumbent index) := by
      rw [Real.dist_eq]
    _ ≤ constant * dist candidate incumbent :=
      (regular index).dist_le_mul candidate incumbent
    _ < margin / 2 := small

/-- One incumbent margin and one score-sensitivity bound certify that the
entire closed trust ball remains inside the same abstract selection cell. -/
theorem ballPreserve
    [PseudoMetricSpace State]
    (chosen : Set Index)
    (score : State → Index → ℝ)
    (constant : ℝ≥0)
    (regular : ∀ index, LipschitzWith constant (fun state => score state index))
    (incumbent : State)
    (radius margin : ℝ)
    (separated : Margin chosen (score incumbent) margin)
    (small : constant * radius < margin / 2) :
    ∀ candidate ∈ Metric.closedBall incumbent radius, Cell chosen (score candidate) := by
  intro candidate candidateMember
  apply radiusPreserve chosen score constant regular incumbent candidate margin separated
  have radiusBound : dist candidate incumbent ≤ radius := by
    simpa [Metric.mem_closedBall, dist_comm] using candidateMember
  exact lt_of_le_of_lt
    (mul_le_mul_of_nonneg_left radiusBound constant.coe_nonneg) small

#print axioms marginPreserve
#print axioms radiusPreserve
#print axioms ballPreserve

end

end BauxResearch.Forward.SelectionCell
