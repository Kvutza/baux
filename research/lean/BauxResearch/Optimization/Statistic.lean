import Mathlib.Probability.Decision.Risk.Basic

/-!
# Statistics as decision-preserving compression

The mathematical object called a statistic is a measurable map from the full
observation into a smaller code.  This file does not call an arbitrary vector a
sufficient statistic.  It uses Mathlib's Bayes-risk data-processing inequality:
discarding information through a statistic cannot improve Bayes risk.

A statistic is lossless for an experiment when a Markov kernel can reconstruct
the original experiment from the compressed one.  Applying data processing in
both directions then proves equality of Bayes risk for every loss and prior.
This is the Blackwell/decision-theoretic notion needed to state what an BAUX
history representation preserves.

Computational cost is deliberately separate.  The identity statistic is
lossless and generally expensive; a small statistic is useful only after its
encoding, storage, and update costs are also bounded.

The experiment is fixed in all comparisons. For an adaptive optimizer its
observation law depends on the policy that collected the observations. The
unrestricted library Bayes risk is an information benchmark; feasibility of a
decision or continuation still requires its own operational correspondence.
-/
namespace BauxResearch.Optimization

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Parameter Data Code Action : Type*}
variable [MeasurableSpace Parameter] [MeasurableSpace Data]
variable [MeasurableSpace Code] [MeasurableSpace Action]

/-- A statistic is a measurable encoding of the complete observed data. -/
structure Statistic (Data Code : Type*) [MeasurableSpace Data] [MeasurableSpace Code] where
  encode : Data → Code
  measurable : Measurable encode

namespace Statistic

/-- The experiment observed only through a statistic. -/
noncomputable def compress (stat : Statistic Data Code) (experiment : Kernel Parameter Data) :
    Kernel Parameter Code :=
  experiment.map stat.encode

/-- Data processing: compression cannot lower the optimal Bayes risk. -/
theorem risk_le (stat : Statistic Data Code) (loss : Parameter → Action → ℝ≥0∞)
    (experiment : Kernel Parameter Data) (prior : Measure Parameter) :
    bayesRisk loss experiment prior ≤ bayesRisk loss (stat.compress experiment) prior :=
  bayesRisk_le_bayesRisk_map loss experiment prior stat.measurable

/-- A lossless statistic admits a stochastic reconstruction of the original experiment. -/
structure Lossless (stat : Statistic Data Code) (experiment : Kernel Parameter Data) where
  decode : Kernel Code Data
  markov : IsMarkovKernel decode
  law : decode ∘ₖ stat.compress experiment = experiment

/-- A lossless statistic preserves Bayes risk for every decision problem and prior. -/
theorem Lossless.risk_eq (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (exact : Lossless stat experiment) (loss : Parameter → Action → ℝ≥0∞)
    (prior : Measure Parameter) :
    bayesRisk loss (stat.compress experiment) prior = bayesRisk loss experiment prior := by
  letI : IsMarkovKernel exact.decode := exact.markov
  apply le_antisymm
  · calc
      bayesRisk loss (stat.compress experiment) prior ≤
          bayesRisk loss (exact.decode ∘ₖ stat.compress experiment) prior :=
        bayesRisk_le_bayesRisk_comp loss (stat.compress experiment) prior exact.decode
      _ = bayesRisk loss experiment prior := by rw [exact.law]
  · exact stat.risk_le loss experiment prior

/-- Decision sufficiency for one declared loss and prior is equality of Bayes risk. -/
def Preserves (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (loss : Parameter → Action → ℝ≥0∞) (prior : Measure Parameter) : Prop :=
  bayesRisk loss (stat.compress experiment) prior = bayesRisk loss experiment prior

theorem Lossless.preserves (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (exact : Lossless stat experiment) (loss : Parameter → Action → ℝ≥0∞)
    (prior : Measure Parameter) : stat.Preserves experiment loss prior :=
  exact.risk_eq stat experiment loss prior

/-- Approximate decision sufficiency permits at most `error` additional Bayes risk. -/
def PreservesWithin (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (loss : Parameter → Action → ℝ≥0∞) (prior : Measure Parameter) (error : ℝ≥0∞) : Prop :=
  bayesRisk loss (stat.compress experiment) prior ≤ bayesRisk loss experiment prior + error

theorem Preserves.within (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (loss : Parameter → Action → ℝ≥0∞) (prior : Measure Parameter)
    (keeps : stat.Preserves experiment loss prior) :
    stat.PreservesWithin experiment loss prior 0 := by
  simpa [PreservesWithin] using keeps.le

/-- A decision after compression is exactly the same rule composed with the
encoding on the original experiment. -/
theorem decisionRisk (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (decision : Kernel Code Action) (loss : Parameter → Action → ℝ≥0∞)
    (prior : Measure Parameter) :
    avgRisk loss (stat.compress experiment) decision prior =
      avgRisk loss experiment
        (decision ∘ₖ Kernel.deterministic stat.encode stat.measurable) prior := by
  rw [compress, ← Kernel.deterministic_comp_eq_map stat.measurable]
  simp only [avgRisk, Kernel.comp_assoc]

/-- Excess risk separates decision error from information discarded by the
statistic. Subtraction is Mathlib's nonnegative extended-real subtraction. -/
theorem riskSplit (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (decision : Kernel Code Action) [IsMarkovKernel decision]
    (loss : Parameter → Action → ℝ≥0∞) (prior : Measure Parameter) :
    avgRisk loss (stat.compress experiment) decision prior - bayesRisk loss experiment prior =
      (avgRisk loss (stat.compress experiment) decision prior -
        bayesRisk loss (stat.compress experiment) prior) +
      (bayesRisk loss (stat.compress experiment) prior - bayesRisk loss experiment prior) :=
  (tsub_add_tsub_cancel (bayesRisk_le_avgRisk loss (stat.compress experiment) decision prior)
    (stat.risk_le loss experiment prior)).symm

/-- Approximate information preservation and decision accuracy compose for
the same fixed experiment, loss, prior, and action space. -/
theorem decisionBound (stat : Statistic Data Code) (experiment : Kernel Parameter Data)
    (decision : Kernel Code Action) (loss : Parameter → Action → ℝ≥0∞)
    (prior : Measure Parameter) (memoryError decisionError : ℝ≥0∞)
    (preserves : stat.PreservesWithin experiment loss prior memoryError)
    (accurate : avgRisk loss (stat.compress experiment) decision prior ≤
      bayesRisk loss (stat.compress experiment) prior + decisionError) :
    avgRisk loss (stat.compress experiment) decision prior ≤
      bayesRisk loss experiment prior + (memoryError + decisionError) := by
  calc
    _ ≤ bayesRisk loss (stat.compress experiment) prior + decisionError := accurate
    _ ≤ (bayesRisk loss experiment prior + memoryError) + decisionError :=
      add_le_add preserves le_rfl
    _ = _ := add_assoc _ _ _

/-- Keeping the complete observation is the baseline lossless statistic. -/
def identity (Data : Type*) [MeasurableSpace Data] : Statistic Data Data where
  encode := id
  measurable := measurable_id

noncomputable def identity_lossless (experiment : Kernel Parameter Data) :
    Lossless (identity Data) experiment where
  decode := Kernel.id
  markov := inferInstance
  law := by simp [compress, identity]

#print axioms Statistic.risk_le
#print axioms Lossless.risk_eq
#print axioms Lossless.preserves
#print axioms identity_lossless
#print axioms decisionRisk
#print axioms riskSplit
#print axioms decisionBound

end Statistic
end BauxResearch.Optimization
