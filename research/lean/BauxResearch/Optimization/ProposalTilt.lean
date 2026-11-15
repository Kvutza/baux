module

public import Mathlib.InformationTheory.KullbackLeibler.Basic

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Learned perturbation laws and actual objective loss

The underlying measurable space can be the complete parameter space or a space
of full perturbations. A fixed retained history determines a reference law `μ`,
a learned score `g`, and an actual proposal law `ν` before the next evaluation.
For perturbations, the true loss `f` means the population loss after applying
the perturbation to the current parameters.

`law` is Mathlib's exponentially tilted reference measure. It is a proof-side
comparison, not an executable sampler or an additional objective evaluation.
`risk_bound` retains errors averaged under both actual and reference laws;
`drift_bound` also retains the reference law's loss relative to the current
model. No calibration, favorable drift, or dimension-independent complexity
is assumed. Finite log-likelihood integrability is explicit: a real-valued KL
divergence must never silently stand for an infinite divergence.

The proof uses Mathlib's `integral_llr_tilted_right` and Gibbs inequality.
`flat_law`, `law_null`, and `reverse_strict` characterize uninformative scores,
preservation of null sets, and strictly harmful reversed scores respectively.
-/

public section

namespace BauxResearch.Optimization.ProposalTilt

noncomputable section

open MeasureTheory InformationTheory

variable {X : Type*} [MeasurableSpace X]
variable {μ ν : Measure X} [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
variable {f g : X → ℝ} {eta : ℝ}

/-- Ideal learned-score tilt; exponential integrability makes it a probability law. -/
def law (μ : Measure X) (g : X → ℝ) (eta : ℝ) : Measure X :=
  μ.tilted (fun x ↦ -eta * g x)

/-- The ideal tilt is a probability measure under its explicit integrability condition. -/
theorem law_prob (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ) :
    IsProbabilityMeasure (law μ g eta) :=
  isProbabilityMeasure_tilted hExp

omit [IsProbabilityMeasure μ] in
/-- A finite tilt preserves exactly the reference law's null sets. -/
theorem law_null (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ) (s : Set X) :
    law μ g eta s = 0 ↔ μ s = 0 :=
  ⟨fun hs ↦ absolutelyContinuous_tilted hExp hs,
    fun hs ↦ tilted_absolutelyContinuous μ _ hs⟩

/-- A constant learned score leaves every reference probability law unchanged. -/
theorem flat_law (c eta : ℝ) : law μ (fun _ ↦ c) eta = μ :=
  tilted_const μ (-eta * c)

/-- Adding a common offset to predictions leaves the learned law unchanged. -/
theorem shift_law (c : ℝ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ) :
    law μ (fun x ↦ g x + c) eta = law μ g eta := by
  have := law_prob hExp
  have hShift : law μ (fun x ↦ g x + c) eta =
      (law μ g eta).tilted (fun _ ↦ -eta * c) := by
    unfold law
    rw [tilted_tilted hExp]
    congr 1
    ext x
    simp only [Pi.add_apply]
    ring
  rw [hShift, tilted_const]

/-- Integrability of the score under its tilt guarantees finite KL displacement. -/
theorem log_finite (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hg : Integrable g (law μ g eta)) :
    Integrable (llr (law μ g eta) μ) (law μ g eta) := by
  have := law_prob hExp
  have hEq : llr (law μ g eta) μ =ᵐ[law μ g eta]
      fun x ↦ -eta * g x - Real.log (∫ x, Real.exp (-eta * g x) ∂μ) :=
    (tilted_absolutelyContinuous μ _).ae_le (log_rnDeriv_tilted_left_self hExp)
  rw [integrable_congr hEq]
  exact (hg.const_mul _).sub (integrable_const _)

/-- Exact KL accounting for an arbitrary actual law relative to the ideal learned tilt. -/
theorem kl_shift (hAc : ν ≪ μ) (hg : Integrable g ν)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hLog : Integrable (llr ν μ) ν) :
    (klDiv ν (law μ g eta)).toReal = (klDiv ν μ).toReal +
      eta * (∫ x, g x ∂ν) + Real.log (∫ x, Real.exp (-eta * g x) ∂μ) := by
  have := law_prob hExp
  have hTilt : ν ≪ law μ g eta := hAc.trans (absolutelyContinuous_tilted hExp)
  have hScore : Integrable (fun x ↦ -eta * g x) ν := hg.const_mul _
  have hTiltLog : Integrable (llr ν (law μ g eta)) ν :=
    integrable_llr_tilted_right hAc hScore hLog hExp
  rw [toReal_klDiv hTilt hTiltLog, toReal_klDiv hAc hLog]
  simp only [probReal_univ, add_sub_cancel_right]
  unfold law
  rw [integral_llr_tilted_right hAc hScore hExp hLog, integral_const_mul]
  ring

/-- Gibbs inequality controls the log normalizer without evaluating it algorithmically. -/
theorem partition_bound (hg : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ) :
    0 ≤ eta * (∫ x, g x ∂μ) + Real.log (∫ x, Real.exp (-eta * g x) ∂μ) := by
  have hSelf : Integrable (llr μ μ) μ := by
    rw [integrable_congr (llr_self μ)]
    exact integrable_zero _ _ _
  have hShift := kl_shift (ν := μ) Measure.AbsolutelyContinuous.rfl hg hExp hSelf
  rw [klDiv_self, ENNReal.toReal_zero, zero_add] at hShift
  rw [← hShift]
  exact ENNReal.toReal_nonneg

/-- Learned-score comparison, retaining the actual law's KL sampling defect. -/
theorem score_bound (hAc : ν ≪ μ) (hgν : Integrable g ν) (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hLog : Integrable (llr ν μ) ν) :
    eta * ((∫ x, g x ∂ν) - ∫ x, g x ∂μ) ≤
      (klDiv ν (law μ g eta)).toReal - (klDiv ν μ).toReal := by
  have hShift := kl_shift hAc hgν hExp hLog
  have hPart := partition_bound hgμ hExp
  linarith

/-- Exact learned-score sampling gains at least its finite KL displacement over the reference. -/
theorem tilt_bound (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hgν : Integrable g (law μ g eta)) :
    eta * ((∫ x, g x ∂(law μ g eta)) - ∫ x, g x ∂μ) ≤
      -(klDiv (law μ g eta) μ).toReal := by
  have := law_prob hExp
  have hScore := score_bound (ν := law μ g eta) (g := g)
    (tilted_absolutelyContinuous μ _) hgν hgμ hExp (log_finite hExp hgν)
  simpa only [klDiv_self, ENNReal.toReal_zero, zero_sub] using hScore

/-- Nontrivial exact tilting strictly lowers expected learned score for positive temperature. -/
theorem tilt_strict (hEta : 0 < eta) (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hgν : Integrable g (law μ g eta)) (hNe : law μ g eta ≠ μ) :
    (∫ x, g x ∂(law μ g eta)) < ∫ x, g x ∂μ := by
  have := law_prob hExp
  have hZero : klDiv (law μ g eta) μ ≠ 0 := fun h ↦ hNe (klDiv_eq_zero_iff.mp h)
  have hPos := ENNReal.toReal_pos hZero
    (klDiv_ne_top (tilted_absolutelyContinuous μ _) (log_finite hExp hgν))
  have hBound := tilt_bound hgμ hExp hgν
  nlinarith

/-- Exact signed surrogate residuals transfer the score comparison to the true objective. -/
theorem residual_bound (hAc : ν ≪ μ)
    (hfν : Integrable f ν) (hfμ : Integrable f μ)
    (hgν : Integrable g ν) (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hLog : Integrable (llr ν μ) ν) :
    eta * ((∫ x, f x ∂ν) - ∫ x, f x ∂μ) ≤
      (klDiv ν (law μ g eta)).toReal - (klDiv ν μ).toReal +
        eta * ((∫ x, (f x - g x) ∂ν) - ∫ x, (f x - g x) ∂μ) := by
  have hScore := score_bound hAc hgν hgμ hExp hLog
  rw [integral_sub hfν hgν, integral_sub hfμ hgμ]
  linarith

/-- Expected-loss bound with heterogeneous errors under actual and reference proposals. -/
theorem risk_bound (hEta : 0 ≤ eta) (hAc : ν ≪ μ)
    (hfν : Integrable f ν) (hfμ : Integrable f μ)
    (hgν : Integrable g ν) (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hLog : Integrable (llr ν μ) ν) :
    eta * ((∫ x, f x ∂ν) - ∫ x, f x ∂μ) ≤
      (klDiv ν (law μ g eta)).toReal - (klDiv ν μ).toReal +
        eta * ((∫ x, |f x - g x| ∂ν) + ∫ x, |f x - g x| ∂μ) := by
  have hResidual := residual_bound hAc hfν hfμ hgν hgμ hExp hLog
  have hν : (∫ x, (f x - g x) ∂ν) ≤ ∫ x, |f x - g x| ∂ν :=
    integral_mono (hfν.sub hgν) (hfν.sub hgν).abs (fun x ↦ le_abs_self _)
  have hμ : -(∫ x, (f x - g x) ∂μ) ≤ ∫ x, |f x - g x| ∂μ := by
    rw [← integral_neg]
    exact integral_mono (hfμ.sub hgμ).neg (hfμ.sub hgμ).abs
      (fun x ↦ neg_le_abs _)
  have hError := mul_le_mul_of_nonneg_left (add_le_add hν hμ) hEta
  linarith

/-- Total drift retains the base proposal's loss relative to the current model's loss. -/
theorem drift_bound (current : ℝ) (hEta : 0 ≤ eta) (hAc : ν ≪ μ)
    (hfν : Integrable f ν) (hfμ : Integrable f μ)
    (hgν : Integrable g ν) (hgμ : Integrable g μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * g x)) μ)
    (hLog : Integrable (llr ν μ) ν) :
    eta * ((∫ x, f x ∂ν) - current) ≤
      eta * ((∫ x, f x ∂μ) - current) +
        (klDiv ν (law μ g eta)).toReal - (klDiv ν μ).toReal +
          eta * ((∫ x, |f x - g x| ∂ν) + ∫ x, |f x - g x| ∂μ) := by
  have := risk_bound hEta hAc hfν hfμ hgν hgμ hExp hLog
  linarith

/-- A reversed objective score favors higher true loss by at least the KL displacement. -/
theorem reverse_bound (hfμ : Integrable f μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * (-f x))) μ)
    (hfν : Integrable f (law μ (fun x ↦ -f x) eta)) :
    (klDiv (law μ (fun x ↦ -f x) eta) μ).toReal ≤
      eta * ((∫ x, f x ∂(law μ (fun x ↦ -f x) eta)) - ∫ x, f x ∂μ) := by
  have hScore := tilt_bound (g := fun x ↦ -f x) hfμ.neg hExp hfν.neg
  simp only [integral_neg] at hScore
  linarith

/-- Any nontrivial integrable reversed tilt strictly worsens expected true loss. -/
theorem reverse_strict (hEta : 0 < eta) (hfμ : Integrable f μ)
    (hExp : Integrable (fun x ↦ Real.exp (-eta * (-f x))) μ)
    (hfν : Integrable f (law μ (fun x ↦ -f x) eta))
    (hNe : law μ (fun x ↦ -f x) eta ≠ μ) :
    (∫ x, f x ∂μ) < ∫ x, f x ∂(law μ (fun x ↦ -f x) eta) := by
  have hStrict := tilt_strict (g := fun x ↦ -f x) hEta hfμ.neg hExp hfν.neg hNe
  simpa only [integral_neg, neg_lt_neg_iff] using hStrict

#print axioms law_null
#print axioms kl_shift
#print axioms tilt_strict
#print axioms risk_bound
#print axioms drift_bound
#print axioms reverse_strict

end
end BauxResearch.Optimization.ProposalTilt
