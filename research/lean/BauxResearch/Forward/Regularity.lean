import BauxResearch.Forward.Algebra
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Smoothness derived for actual real forward primitives

The Mathlib umbrella is ContDiff and its finite-sum, product, exp, log, sqrt,
and division composition theorems. Denominator nonvanishing is proved below,
not assumed as a numeric cap. These results concern Arithmetic.exact. Routing,
clipping, storage rounding, and the whole forward objective remain separate.
-/
namespace BauxResearch.Forward.Regularity
noncomputable section
open scoped BigOperators

theorem sigmoid : ContDiff ℝ ⊤ Forward.sigmoid := by
  unfold Forward.sigmoid
  apply contDiff_const.div
  · fun_prop
  · intro value
    have positive := Real.exp_pos (-value)
    linarith

theorem silu : ContDiff ℝ ⊤ Forward.silu := by
  exact contDiff_id.mul sigmoid

theorem probability {width : ℕ} (index : Fin width) :
    ContDiff ℝ ⊤ (fun scores : Vector width => Forward.probability scores index) := by
  unfold Forward.probability
  apply ContDiff.div
  · fun_prop
  · fun_prop
  · intro scores
    exact ne_of_gt (Probability.positive scores index)

theorem logLoss {width : ℕ} (target : Fin width) :
    ContDiff ℝ ⊤ (fun scores : Vector width => Forward.logLoss scores target) := by
  unfold Forward.logLoss
  apply ContDiff.sub
  · apply ContDiff.log
    · fun_prop
    · intro scores
      exact ne_of_gt (Probability.positive scores target)
  · fun_prop

variable {Space : Type*} [NormedAddCommGroup Space] [NormedSpace ℝ Space]

/-- RMS smoothness jointly in any smooth weight and activation paths. Positive
epsilon is sufficient for the square-root denominator to be nonzero. -/
theorem rms {width : ℕ} (epsilon : ℝ) (positive : 0 < epsilon)
    (weights values : Fin width → Space → ℝ)
    (weightSmooth : ∀ index, ContDiff ℝ ⊤ (weights index))
    (valueSmooth : ∀ index, ContDiff ℝ ⊤ (values index)) (index : Fin width) :
    ContDiff ℝ ⊤ (fun state => Forward.rms Arithmetic.exact epsilon
      (fun column => weights column state) (fun column => values column state) index) := by
  have denominator : ContDiff ℝ ⊤ (fun state =>
      (∑ column, values column state ^ 2) / width + epsilon) := by fun_prop
  have positiveDen : ∀ state,
      0 < (∑ column, values column state ^ 2) / (width : ℝ) + epsilon := by
    intro state
    have total := Finset.sum_nonneg (fun column (_ : column ∈ Finset.univ) =>
      sq_nonneg (values column state))
    exact add_pos_of_nonneg_of_pos (div_nonneg total (Nat.cast_nonneg width)) positive
  have root := denominator.sqrt (fun state => ne_of_gt (positiveDen state))
  have numerator : ContDiff ℝ ⊤ (fun state => values index state * weights index state) :=
    (valueSmooth index).mul (weightSmooth index)
  simp only [Forward.rms, Arithmetic.exact, id_eq]
  apply ContDiff.div
  · exact numerator
  · exact root
  · intro state
    exact ne_of_gt (Real.sqrt_pos.2 (positiveDen state))

/-- A finite projection is smooth jointly in its weights and inputs. -/
theorem projection {input output : ℕ}
    (weights : Fin input → Fin output → Space → ℝ) (values : Fin input → Space → ℝ)
    (weightSmooth : ∀ row column, ContDiff ℝ ⊤ (weights row column))
    (valueSmooth : ∀ row, ContDiff ℝ ⊤ (values row)) (column : Fin output) :
    ContDiff ℝ ⊤ (fun state => Forward.project Arithmetic.exact
      (fun row column => weights row column state) (fun row => values row state) column) := by
  simp only [Forward.project, Arithmetic.exact, id_eq]
  fun_prop

#print axioms sigmoid
#print axioms silu
#print axioms probability
#print axioms logLoss
#print axioms rms
#print axioms projection
end
end BauxResearch.Forward.Regularity
