import BauxResearch.Optimization.Synthesis
import Mathlib.Probability.Kernel.IonescuTulcea.Traj
import Mathlib.Probability.Kernel.Composition.IntegralCompProd

/-!
# The existing policy as a Mathlib stochastic process

The finite conditional oracle in `Synthesis` induces a transition kernel on
the policy's memory. Mathlib supplies the law of the complete trajectory.
Its finite marginals and terminal integrals agree with the existing recursion.

The policy is fixed throughout. Changing a policy changes its observation law;
an infimum over terminal decision kernels for one experiment is consequently
not an infimum over adaptive training policies. These results do not identify
the deployed BAUX observation law or establish a learning rate.
-/

namespace BauxResearch.Optimization.Process

open MeasureTheory ProbabilityTheory Synthesis Function Preorder
open scoped BigOperators ENNReal

noncomputable section

variable {Memory State Feedback Noise : Type*} [MeasurableSpace Memory]

/-- The last memory in a finite trajectory prefix. -/
def last (time : ℕ) (history : (i : Finset.Iic time) → Memory) : Memory :=
  history ⟨time, Finset.mem_Iic.mpr le_rfl⟩

theorem lastMeas (time : ℕ) : Measurable (last (Memory := Memory) time) :=
  measurable_pi_apply _

/-- A transition may depend on time and on every item retained in memory. -/
def family (step : ℕ → Kernel Memory Memory) (time : ℕ) (index : ℕ) :
    Kernel ((i : Finset.Iic index) → Memory) Memory :=
  (step (time + index)).comap (last index) (lastMeas index)

/-- Embed the initial memory as the time-zero trajectory prefix. -/
def initial : Kernel Memory ((i : Finset.Iic 0) → Memory) :=
  Kernel.deterministic (fun memory _ => memory)
    (by fun_prop)

instance familyMarkov (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time index : ℕ) :
    IsMarkovKernel (family step time index) := by
  unfold family
  infer_instance

/-- The full memory trajectory law, supplied by Ionescu-Tulcea. -/
def pathLaw (step : ℕ → Kernel Memory Memory) [∀ time, IsMarkovKernel (step time)]
    (time : ℕ) : Kernel Memory (ℕ → Memory) :=
  Kernel.traj (family step time) 0 ∘ₖ initial

instance pathMarkov (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time : ℕ) :
    IsMarkovKernel (pathLaw step time) := by
  unfold pathLaw initial
  infer_instance

/-- The finite terminal law in the same order as `Synthesis.expected`. -/
def terminal (step : ℕ → Kernel Memory Memory) : ℕ → ℕ → Kernel Memory Memory
  | _, 0 => Kernel.id
  | time, count + 1 => terminal step (time + 1) count ∘ₖ step time

instance terminalMarkov (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time count : ℕ) :
    IsMarkovKernel (terminal step time count) := by
  induction count generalizing time with
  | zero => simp only [terminal]; infer_instance
  | succ count induction =>
      simp only [terminal]
      have := induction (time + 1)
      infer_instance

private theorem append (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time count : ℕ) :
    terminal step time (count + 1) = step (time + count) ∘ₖ terminal step time count := by
  induction count generalizing time with
  | zero => simp [terminal]
  | succ count induction =>
      simpa only [terminal, Kernel.comp_assoc, Nat.add_assoc, Nat.add_comm,
        Nat.add_left_comm] using congrArg (fun law => law ∘ₖ step time)
          (induction (time + 1))

/-- Every finite prefix of Mathlib's trajectory gives the same terminal law. -/
theorem prefixLaw (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time count : ℕ) :
    (Kernel.partialTraj (X := fun _ => Memory) (family step time) 0 count ∘ₖ initial).map (last count) =
      terminal step time count := by
  induction count with
  | zero =>
      simp only [Kernel.partialTraj_self, Kernel.id_comp, terminal]
      unfold initial
      rw [Kernel.deterministic_map _ (lastMeas 0)]
      rfl
  | succ count induction =>
      rw [Kernel.partialTraj_succ_eq_comp (Nat.zero_le count), Kernel.comp_assoc,
        Kernel.map_comp]
      have next := Kernel.map_partialTraj_succ_self (X := fun _ => Memory)
        (κ := family step time) count
      change (Kernel.partialTraj (X := fun _ => Memory) (family step time) count (count + 1)).map
        (last (count + 1)) = family step time count at next
      rw [next]
      change (family step time count) ∘ₖ
        (Kernel.partialTraj (X := fun _ => Memory) (family step time) 0 count ∘ₖ initial) = _
      rw [family, ← Kernel.comp_map _ _ (lastMeas count), induction, append]

/-- Evaluation at any finite horizon is a marginal of one infinite run. -/
theorem marginal (step : ℕ → Kernel Memory Memory)
    [∀ time, IsMarkovKernel (step time)] (time count : ℕ) :
    (pathLaw step time).map (fun path => path count) = terminal step time count := by
  have split : (fun path : ℕ → Memory => path count) =
      last count ∘ frestrictLe count := rfl
  rw [pathLaw, Kernel.map_comp, split, Kernel.map_comp_right _ (by fun_prop)
    (lastMeas count), Kernel.traj_map_frestrictLe, ← Kernel.map_comp]
  exact prefixLaw step time count

section Finite

variable [Fintype Noise]

/-- The exact finite oracle law, with its measurability obligations explicit. -/
def oracleKernel (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise)
    (weightMeas : ∀ time noise,
      Measurable (fun memory => oracle.weight time memory (policy.query time memory) noise))
    (moveMeas : ∀ time noise, Measurable (fun memory => advance policy oracle time memory noise))
    (time : ℕ) : Kernel Memory Memory where
  toFun memory := ∑ noise, ENNReal.ofReal
    (oracle.weight time memory (policy.query time memory) noise) •
      Measure.dirac (advance policy oracle time memory noise)
  measurable' := by
    apply Measure.measurable_of_measurable_coe
    intro set measurable
    simp only [Measure.finsetSum_apply, Measure.smul_apply, smul_eq_mul,
      Measure.dirac_apply' _ measurable]
    exact Finset.measurable_sum _ fun noise _ =>
      (weightMeas time noise).ennreal_ofReal.mul
        (measurable_one.indicator ((moveMeas time noise) measurable))

instance oracleMarkov (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas) (time : ℕ) :
    IsMarkovKernel (oracleKernel policy oracle weightMeas moveMeas time) where
  isProbabilityMeasure memory := ⟨by
    change (∑ noise, ENNReal.ofReal
      (oracle.weight time memory (policy.query time memory) noise) •
        Measure.dirac (advance policy oracle time memory noise)) Set.univ = 1
    simp only [Measure.finsetSum_apply, Measure.smul_apply, Measure.dirac_apply_of_mem
      (Set.mem_univ _), smul_eq_mul, mul_one]
    rw [← ENNReal.ofReal_sum_of_nonneg (fun noise _ => oracle.nonnegative time memory _ noise),
      oracle.mass, ENNReal.ofReal_one]⟩

variable [MeasurableSingletonClass Memory]

theorem oracleIntegrable (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (time : ℕ) (memory : Memory) (loss : Memory → ℝ) :
    Integrable loss (oracleKernel policy oracle weightMeas moveMeas time memory) := by
  apply integrable_finsetSum_measure.mpr
  intro noise _
  exact (integrable_dirac (by simp)).smul_measure ENNReal.ofReal_ne_top

/-- The kernel integral is precisely the source oracle's finite weighted sum. -/
theorem oracleIntegral (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (time : ℕ) (memory : Memory) (loss : Memory → ℝ) :
    ∫ next, loss next ∂oracleKernel policy oracle weightMeas moveMeas time memory =
      ∑ noise, oracle.weight time memory (policy.query time memory) noise *
        loss (advance policy oracle time memory noise) := by
  change (∫ next, loss next ∂(∑ noise, ENNReal.ofReal
    (oracle.weight time memory (policy.query time memory) noise) •
      Measure.dirac (advance policy oracle time memory noise))) = _
  rw [integral_finsetSum_measure (fun noise _ =>
    (integrable_dirac (by simp)).smul_measure ENNReal.ofReal_ne_top)]
  simp only [integral_smul_measure, integral_dirac, smul_eq_mul,
    ENNReal.toReal_ofReal (oracle.nonnegative time memory _ _)]

theorem terminalIntegrable (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (time count : ℕ) (memory : Memory) (loss : Memory → ℝ) (measurable : Measurable loss) :
    Integrable loss (terminal (oracleKernel policy oracle weightMeas moveMeas) time count memory) := by
  induction count generalizing time memory with
  | zero => simpa [terminal, Kernel.id_apply] using integrable_dirac (by simp)
  | succ count induction =>
      rw [terminal]
      apply (ProbabilityTheory.integrable_comp_iff measurable.stronglyMeasurable.aestronglyMeasurable).mpr
      exact ⟨ae_of_all _ fun next => induction (time + 1) next,
        oracleIntegrable policy oracle weightMeas moveMeas time memory _⟩

/-- Existing certificates now bound a Mathlib expectation of the same policy. -/
theorem expectedIntegral (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (objective : State → ℝ) (best : State)
    (measurable : Measurable (fun memory =>
      Regret.simpleRegret objective best (policy.recommend memory)))
    (time count : ℕ) (memory : Memory) :
    ∫ next, Regret.simpleRegret objective best (policy.recommend next)
      ∂terminal (oracleKernel policy oracle weightMeas moveMeas) time count memory =
        expected policy oracle objective best time count memory := by
  induction count generalizing time memory with
  | zero => simp [terminal, Kernel.id_apply, expected]
  | succ count induction =>
      rw [terminal, Kernel.integral_comp
        (terminalIntegrable policy oracle weightMeas moveMeas time (count + 1) memory _ measurable),
        oracleIntegral]
      simp only [expected, induction]

/-- The finite recursion is the terminal expectation under one complete run law. -/
theorem pathIntegral (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (objective : State → ℝ) (best : State)
    (measurable : Measurable (fun memory =>
      Regret.simpleRegret objective best (policy.recommend memory)))
    (time count : ℕ) (memory : Memory) :
    ∫ path, Regret.simpleRegret objective best (policy.recommend (path count))
      ∂pathLaw (oracleKernel policy oracle weightMeas moveMeas) time memory =
        expected policy oracle objective best time count memory := by
  rw [← expectedIntegral policy oracle weightMeas moveMeas objective best measurable time count memory,
    ← marginal, Kernel.map_apply _ (by fun_prop),
    integral_map_of_stronglyMeasurable (by fun_prop) measurable.stronglyMeasurable]

/-- A checked finite policy certificate controls the same run under Mathlib's
trajectory measure, with all statistical premises unchanged. -/
theorem pathBound (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (weightMeas moveMeas)
    (objective : State → ℝ) (best : State)
    (measurable : Measurable (fun memory =>
      Regret.simpleRegret objective best (policy.recommend memory)))
    (horizon budget : ℕ) (bound : ℝ)
    (certificate : Certificate policy oracle objective best horizon bound budget) :
    ∫ path, Regret.simpleRegret objective best (policy.recommend (path horizon))
      ∂pathLaw (oracleKernel policy oracle weightMeas moveMeas) 0 policy.initial ≤ bound := by
  rw [pathIntegral policy oracle weightMeas moveMeas objective best measurable]
  exact certificate.regretBound

#print axioms marginal
#print axioms expectedIntegral
#print axioms pathIntegral
#print axioms pathBound

end Finite
end
end BauxResearch.Optimization.Process
