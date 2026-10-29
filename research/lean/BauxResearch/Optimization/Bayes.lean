import Mathlib.Probability.Kernel.Posterior

/-! A posterior research view using Mathlib's existing disintegration construction.
This is not a claim that neighbor_value.rs, its LOOCV scales, or its reported
uncertainty is a posterior for the supplied prior/observation law. That requires
a separate correspondence theorem. The construction is analysis, not a Rust
implementation or a computable algorithm extracted from existence alone. -/
namespace BauxResearch.Optimization.Bayes
noncomputable section
open MeasureTheory ProbabilityTheory

variable {Parameter Feedback : Type*}
  [MeasurableSpace Parameter] [MeasurableSpace Feedback]
  [StandardBorelSpace Parameter] [Nonempty Parameter]

/-- A supplied observation law and prior induce Mathlib's posterior kernel. -/
def infer (prior : Measure Parameter) [IsFiniteMeasure prior]
    (law : Kernel Parameter Feedback) [IsFiniteKernel law] : Kernel Feedback Parameter :=
  ProbabilityTheory.posterior law prior

theorem normalized (prior : Measure Parameter) [IsFiniteMeasure prior]
    (law : Kernel Parameter Feedback) [IsFiniteKernel law] : IsMarkovKernel (infer prior law) := by
  unfold infer
  infer_instance

/-- The posterior is specified by recovering the joint law, not merely two moments. -/
theorem jointLaw (prior : Measure Parameter) [IsFiniteMeasure prior]
    (law : Kernel Parameter Feedback) [IsFiniteKernel law] :
    (law ∘ₘ prior) ⊗ₘ infer prior law = (prior ⊗ₘ law).map Prod.swap :=
  compProd_posterior_eq_map_swap

#print axioms jointLaw
end
end BauxResearch.Optimization.Bayes
