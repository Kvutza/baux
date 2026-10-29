import BauxResearch.Objective.NaturalWindow
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.Moments.SubGaussian
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# Uniform aligned-window marginal

This is the positive probabilistic construction for a parameter-only surrogate.
One complete natural window is the random observation. Its origin is uniform on
the finite set of aligned corpus origins. The expectation of its loss is exactly
the finite-corpus mean, and bounded whole-window loss gives a centered
sub-Gaussian observation through Mathlib's Hoeffding lemma.

The sample unit is one window, not its individual tokens. A reduction of the
whole-window range with window length needs a separate corpus-dependence result.
-/

/-!
With a uniform origin `O` among complete aligned windows and whole-window
loss `ℓ(O)`, Mathlib gives `E[ℓ(O)] = corpusMean ℓ`.  If `ℓ(O) ∈ [a,b]`, the
centered sub-Gaussian scale is `((b-a)/2)^2`.  A window's token count does not
appear in that scale without a further dependence theorem.
-/

namespace BauxResearch.Objective.MarginalWindow

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory
open BauxResearch.Objective.NaturalWindow

noncomputable section

/-- Probability measure induced by the uniform aligned-origin distribution. -/
def originMeasure (count : ℕ) [NeZero count] : Measure (Fin count) :=
  Measure.uniformOfFinset Finset.univ

instance originProbability (count : ℕ) [NeZero count] :
    IsProbabilityMeasure (originMeasure count) := by
  constructor
  simp [originMeasure, Measure.uniformOfFinset_apply]
  exact ENNReal.div_self (by exact_mod_cast (NeZero.ne count)) (by simp)

/-- Uniform-origin expected loss is exactly the finite-corpus objective. -/
theorem meanEq (count : ℕ) [NeZero count] (loss : Fin count → ℝ) :
    ∫ origin, loss origin ∂originMeasure count = corpusMean loss := by
  unfold originMeasure Measure.uniformOfFinset
  rw [integral_finsetSum_measure (fun origin _ =>
    (integrable_dirac (by simp)).smul_measure (by finiteness))]
  simp [integral_smul_measure, integral_dirac, corpusMean,
    ENNReal.toReal_inv, Finset.mul_sum]

/-- A bounded natural-window loss is sub-Gaussian around the corpus-marginal
objective. The parameter is the squared half-range of whole-window loss. -/
theorem centeredSubg
    (count : ℕ)
    [NeZero count]
    (loss : Fin count → ℝ)
    (lower upper : ℝ)
    (bounded : ∀ origin, loss origin ∈ Set.Icc lower upper) :
    HasSubgaussianMGF
      (fun origin => loss origin - corpusMean loss)
      ((‖upper - lower‖₊ / 2) ^ 2)
      (originMeasure count) := by
  have measurable : AEMeasurable loss (originMeasure count) :=
    Measurable.aemeasurable (by fun_prop)
  have bounds : ∀ᵐ origin ∂originMeasure count,
      loss origin ∈ Set.Icc lower upper := Filter.Eventually.of_forall bounded
  have result := hasSubgaussianMGF_of_mem_Icc measurable bounds
  rwa [meanEq count loss] at result

/-- Upper tail for one complete natural-window score. There is deliberately no
token-count multiplier: the random sample is the selected window. -/
theorem upperTail
    (count : ℕ)
    [NeZero count]
    (loss : Fin count → ℝ)
    (lower upper epsilon : ℝ)
    (bounded : ∀ origin, loss origin ∈ Set.Icc lower upper)
    (epsilonNonnegative : 0 ≤ epsilon) :
    (originMeasure count).real {origin |
      epsilon ≤ loss origin - corpusMean loss} ≤
      Real.exp (-epsilon ^ 2 /
        (2 * ((‖upper - lower‖₊ / 2) ^ 2))) := by
  exact (centeredSubg count loss lower upper bounded).measure_ge_le
    epsilonNonnegative

/-- Lower tail for one complete natural-window score. This is the unsafe
direction when a sampled window makes a candidate appear better than its
corpus-marginal loss. -/
theorem lowerTail
    (count : ℕ)
    [NeZero count]
    (loss : Fin count → ℝ)
    (lower upper epsilon : ℝ)
    (bounded : ∀ origin, loss origin ∈ Set.Icc lower upper)
    (epsilonNonnegative : 0 ≤ epsilon) :
    (originMeasure count).real {origin |
      epsilon ≤ corpusMean loss - loss origin} ≤
      Real.exp (-epsilon ^ 2 /
        (2 * ((‖upper - lower‖₊ / 2) ^ 2))) := by
  have tail := (centeredSubg count loss lower upper bounded).neg.measure_ge_le
    epsilonNonnegative
  simpa [Pi.neg_apply, neg_sub] using tail

/-- Adaptivity of the candidate does not change the one-step law: after any
prior history, a fresh uniform origin is centered on that history-selected
candidate's corpus marginal. -/
theorem adaptiveSubg
    {History : Type*}
    (count : ℕ)
    [NeZero count]
    (loss : History → Fin count → ℝ)
    (lower upper : ℝ)
    (bounded : ∀ history origin, loss history origin ∈ Set.Icc lower upper) :
    ∀ history,
      HasSubgaussianMGF
        (fun origin => loss history origin - corpusMean (loss history))
        ((‖upper - lower‖₊ / 2) ^ 2)
        (originMeasure count) := by
  intro history
  exact centeredSubg count (loss history) lower upper (bounded history)

/-- Pointwise conditional tail contract for a history-selected candidate and a
fresh uniform origin. No restriction is placed on how the history selected the
candidate. -/
theorem adaptiveTail
    {History : Type*}
    (count : ℕ)
    [NeZero count]
    (loss : History → Fin count → ℝ)
    (lower upper epsilon : ℝ)
    (bounded : ∀ history origin, loss history origin ∈ Set.Icc lower upper)
    (epsilonNonnegative : 0 ≤ epsilon) :
    ∀ history,
      (originMeasure count).real {origin |
        epsilon ≤ loss history origin - corpusMean (loss history)} ≤
        Real.exp (-epsilon ^ 2 /
          (2 * ((‖upper - lower‖₊ / 2) ^ 2))) := by
  intro history
  exact upperTail count (loss history) lower upper epsilon
    (bounded history) epsilonNonnegative

#print axioms meanEq
#print axioms centeredSubg
#print axioms upperTail
#print axioms lowerTail
#print axioms adaptiveSubg
#print axioms adaptiveTail

end

end BauxResearch.Objective.MarginalWindow
