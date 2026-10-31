import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# One-Observation Function-Valued Kernel Loss Estimation (Bubeck, Eldan, Lee, arXiv:1607.03084)

Formalizes the gradient-free one-observation loss-function estimator from
Bubeck, Eldan, and Lee (2016):

* Given a maintained distribution `p`, smoothing kernel `K`, and sampling law
  `q = K p`, observing a single scalar value `Y_t = loss(X_t)` at `X_t ~ q`
  produces the function-valued estimator
  `lossEstimator K q loss X_t x = (loss X_t * K x X_t) / q X_t`.
* **Pointwise Unbiasedness (`lossEstimator_unbiased`)**: The expectation of
  `lossEstimator` under `X_t ~ q` equals the kernel-smoothed loss
  `smoothedLoss K loss x = (K* loss)(x)` at every point `x` simultaneously,
  without forming any gradient estimate of `loss`.
* **Adjoint Duality (`smoothedLoss_duality`, `expected_linear_lossEstimator`)**:
  The expected linear functional `𝔼[⟨p', lossEstimator⟩]` under `X_t ~ q`
  equals `⟨K p', loss⟩`.
* **Second-Moment Identity (`lossEstimator_secondMoment`)**: Exact importance
  sampling second moment `∑ u, q u * (lossEstimator K q loss u x)²`.
-/

namespace BauxResearch.Optimization.KernelBandit

noncomputable section
open scoped BigOperators

variable {Ω : Type*} [Fintype Ω]

/-- Kernel-smoothed sampling distribution `q(u) = (K p)(u) = ∑ x, p x * K x u`. -/
def samplingLaw (K : Ω → Ω → ℝ) (p : Ω → ℝ) (u : Ω) : ℝ :=
  ∑ x : Ω, p x * K x u

/-- Adjoint kernel-smoothed loss `(K* loss)(x) = ∑ u, loss u * K x u`. -/
def smoothedLoss (K : Ω → Ω → ℝ) (loss : Ω → ℝ) (x : Ω) : ℝ :=
  ∑ u : Ω, loss u * K x u

/-- One-observation function-valued loss estimator
`ℓ̂(u, x) = (loss u * K x u) / q u` evaluated from a single draw `u ~ q`. -/
def lossEstimator (K : Ω → Ω → ℝ) (q : Ω → ℝ) (loss : Ω → ℝ) (u x : Ω) : ℝ :=
  (loss u * K x u) / q u

/-- Pointwise unbiasedness: for every query `x`, the expectation of the
one-observation estimator under `u ~ q` equals the smoothed loss `(K* loss)(x)`. -/
theorem lossEstimator_unbiased (K : Ω → Ω → ℝ) (q : Ω → ℝ) (loss : Ω → ℝ) (x : Ω)
    (hq : ∀ u : Ω, q u ≠ 0) :
    ∑ u : Ω, q u * lossEstimator K q loss u x = smoothedLoss K loss x := by
  unfold lossEstimator smoothedLoss
  apply Finset.sum_congr rfl
  intro u _
  exact mul_div_cancel₀ (loss u * K x u) (hq u)

/-- Kernel adjoint duality: `⟨p, K* loss⟩ = ⟨K p, loss⟩`. -/
theorem smoothedLoss_duality (K : Ω → Ω → ℝ) (p loss : Ω → ℝ) :
    ∑ x : Ω, p x * smoothedLoss K loss x =
      ∑ u : Ω, samplingLaw K p u * loss u := by
  unfold smoothedLoss samplingLaw
  calc
    ∑ x : Ω, p x * (∑ u : Ω, loss u * K x u)
      = ∑ x : Ω, ∑ u : Ω, p x * (loss u * K x u) := by
          apply Finset.sum_congr rfl
          intro x _
          rw [Finset.mul_sum]
    _ = ∑ u : Ω, ∑ x : Ω, (p x * K x u) * loss u := by
          rw [Finset.sum_comm]
          apply Finset.sum_congr rfl
          intro u _
          apply Finset.sum_congr rfl
          intro x _
          ring
    _ = ∑ u : Ω, (∑ x : Ω, p x * K x u) * loss u := by
          apply Finset.sum_congr rfl
          intro u _
          rw [← Finset.sum_mul]

/-- Distributional unbiasedness: for any candidate law `p'`, the expected inner
product `𝔼_{u ~ q}[⟨p', ℓ̂(u, ·)⟩]` equals `⟨K p', loss⟩`. -/
theorem expected_linear_lossEstimator (K : Ω → Ω → ℝ) (q p' loss : Ω → ℝ)
    (hq : ∀ u : Ω, q u ≠ 0) :
    ∑ u : Ω, q u * (∑ x : Ω, p' x * lossEstimator K q loss u x) =
      ∑ u : Ω, samplingLaw K p' u * loss u := by
  calc
    ∑ u : Ω, q u * (∑ x : Ω, p' x * lossEstimator K q loss u x)
      = ∑ x : Ω, p' x * (∑ u : Ω, q u * lossEstimator K q loss u x) := by
          have h1 : ∀ u : Ω, q u * (∑ x : Ω, p' x * lossEstimator K q loss u x) =
              ∑ x : Ω, p' x * (q u * lossEstimator K q loss u x) := by
            intro u
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro x _
            ring
          simp_rw [h1]
          rw [Finset.sum_comm]
          apply Finset.sum_congr rfl
          intro x _
          rw [← Finset.mul_sum]
    _ = ∑ x : Ω, p' x * smoothedLoss K loss x := by
          apply Finset.sum_congr rfl
          intro x _
          rw [lossEstimator_unbiased K q loss x hq]
    _ = ∑ u : Ω, samplingLaw K p' u * loss u :=
          smoothedLoss_duality K p' loss

/-- Exact second moment of the one-observation function-valued estimator under `q`. -/
theorem lossEstimator_secondMoment (K : Ω → Ω → ℝ) (q loss : Ω → ℝ) (x : Ω)
    (hq : ∀ u : Ω, q u ≠ 0) :
    ∑ u : Ω, q u * (lossEstimator K q loss u x) ^ 2 =
      ∑ u : Ω, (loss u * K x u) ^ 2 / q u := by
  unfold lossEstimator
  apply Finset.sum_congr rfl
  intro u _
  have hNe := hq u
  field_simp [hNe]

#print axioms lossEstimator_unbiased
#print axioms smoothedLoss_duality
#print axioms expected_linear_lossEstimator
#print axioms lossEstimator_secondMoment

end
end BauxResearch.Optimization.KernelBandit
