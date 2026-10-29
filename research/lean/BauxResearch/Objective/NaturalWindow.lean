import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic

/-!
# Deterministic natural-window error

The deployed pretraining scorer chooses one aligned, contiguous corpus window
from a fixed seed and transaction index. The candidate state may change after
every transaction, so scores from different transactions are values of
different loss functions.

This module gives the pathwise statement supported by that implementation. A
single selected window estimates the finite-corpus mean only when loss
oscillation over window origins is controlled for the current candidate. No
probability law is inferred from the source hash or from the window length.
-/

namespace BauxResearch.Objective.NaturalWindow

open scoped BigOperators
noncomputable section

/-- Uniform mean over all aligned corpus-window origins. -/
def corpusMean {count : ℕ} (loss : Fin count → ℝ) : ℝ :=
  (count : ℝ)⁻¹ * ∑ origin, loss origin

/-- A fixed selected origin is accurate whenever every aligned-origin loss is
within `radius` of its loss. The result is deterministic and remains valid
when the candidate and selected origin depend on all earlier transactions. -/
theorem biasBound {count : ℕ}
    (countPositive : 0 < count)
    (loss : Fin count → ℝ)
    (selected : Fin count)
    (radius : ℝ)
    (oscillation : ∀ origin, |loss origin - loss selected| ≤ radius) :
    |loss selected - corpusMean loss| ≤ radius := by
  have countRealPositive : 0 < (count : ℝ) := by
    exact_mod_cast countPositive
  have sumBound :
      |∑ origin, (loss selected - loss origin)| ≤ (count : ℝ) * radius := by
    calc
      _ ≤ ∑ origin, |loss selected - loss origin| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _origin : Fin count, radius := by
        apply Finset.sum_le_sum
        intro origin _
        simpa only [abs_sub_comm] using oscillation origin
      _ = (count : ℝ) * radius := by simp
  have identity :
      loss selected - corpusMean loss =
        (count : ℝ)⁻¹ * ∑ origin, (loss selected - loss origin) := by
    unfold corpusMean
    rw [Finset.sum_sub_distrib]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    field_simp
  rw [identity, abs_mul, abs_of_pos (inv_pos.mpr countRealPositive)]
  have scaled := mul_le_mul_of_nonneg_left sumBound (inv_nonneg.mpr countRealPositive.le)
  simpa [countRealPositive.ne'] using scaled

/-- Source-shaped form: every transaction may use a different candidate loss
and a history-dependent selected origin. -/
theorem scheduleBound {count : ℕ}
    (countPositive : 0 < count)
    (loss : ℕ → Fin count → ℝ)
    (selected : ℕ → Fin count)
    (radius : ℕ → ℝ)
    (oscillation : ∀ step origin,
      |loss step origin - loss step (selected step)| ≤ radius step) :
    ∀ step,
      |loss step (selected step) - corpusMean (loss step)| ≤ radius step := by
  intro step
  exact biasBound countPositive (loss step) (selected step) (radius step)
    (oscillation step)

/-- With only one observed origin, population error is unbounded absent a
cross-origin premise: on two origins, the observed loss can be zero while the
corpus mean is any prescribed half-gap. -/
theorem gapWitness (gap : ℝ) :
    ∃ loss : Fin 2 → ℝ,
      loss 0 = 0 ∧ corpusMean loss = gap / 2 := by
  let loss : Fin 2 → ℝ := fun origin => if origin = 0 then 0 else gap
  refine ⟨loss, by simp [loss], ?_⟩
  simp [corpusMean, loss, Fin.sum_univ_two]
  ring

#print axioms biasBound
#print axioms scheduleBound
#print axioms gapWitness

end

end BauxResearch.Objective.NaturalWindow
