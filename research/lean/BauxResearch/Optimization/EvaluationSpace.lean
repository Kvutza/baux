import BauxResearch.Foundation
import BauxResearch.Optimization.Step
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Mathematical Evaluation Space for Bayesian Optimization

Grounded in:
* David Blackwell (1951, 1953): Comparison of experiments and sufficiency preorders.
* Dennis Lindley (1956): Information gain from experiments.
* Raul Astudillo & Peter Frazier (2019, 2021): Composite Bayesian Optimization `g(f(x))`.
* Javier González et al. (2017): Preferential / relative optimization under Common Random Numbers.
-/

namespace BauxResearch.Optimization.EvaluationSpace

noncomputable section

open scoped BigOperators
open BauxResearch.Optimization

/-- An experimental query space equipped with cost and information sufficiency.
Query `q_2 ⪯ q_1` asserts that experiment `q_1` is Blackwell-sufficient for `q_2`. -/
structure Experiment (Query : Type*) where
  cost : Query → ℝ
  costPos : ∀ q, 0 < cost q
  Suff : Query → Query → Prop

/-- A measurement channel executing experiments across parameter states and environments. -/
structure Channel (State World Query : Type*) (Obs : Query → Type*) where
  evaluate : State → World → (q : Query) → Obs q

/-- Coarsening reduction morphism: query `coarse` is obtained from `fine` via a
deterministic reduction map `reduce`, preserving evaluation across all environments. -/
def Coarsening {State World Query : Type*} {Obs : Query → Type*}
    (channel : Channel State World Query Obs)
    (fine coarse : Query)
    (reduce : Obs fine → Obs coarse) : Prop :=
  ∀ (state : State) (world : World),
    channel.evaluate state world coarse = reduce (channel.evaluate state world fine)

/-- General composite weighted evaluation in dimension m (Astudillo & Frazier 2019/2021). -/
def compMean {m : ℕ} (w : Fin m → ℝ) (v : Fin m → ℝ) : ℝ :=
  ∑ i : Fin m, w i * v i

/-- General composite aggregation is additive. -/
theorem compAdd {m : ℕ} (w : Fin m → ℝ) (u v : Fin m → ℝ) :
    compMean w (u + v) = compMean w u + compMean w v := by
  unfold compMean
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  simp only [Pi.add_apply]
  ring

/-- General composite aggregation scales with scalar multipliers. -/
theorem compScale {m : ℕ} (w : Fin m → ℝ) (c : ℝ) (v : Fin m → ℝ) :
    compMean w (fun i => c * v i) = c * compMean w v := by
  unfold compMean
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- General composite error bound for non-negative weights (Astudillo & Frazier). -/
theorem compError {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i)
    (truth estimate eps : Fin m → ℝ)
    (hEps : ∀ i : Fin m, |estimate i - truth i| ≤ eps i) :
    |compMean w estimate - compMean w truth| ≤ compMean w eps := by
  unfold compMean
  have hDiff : ∑ i, w i * estimate i - ∑ i, w i * truth i =
      ∑ i, w i * (estimate i - truth i) := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hDiff]
  calc
    |∑ i, w i * (estimate i - truth i)| ≤ ∑ i, |w i * (estimate i - truth i)| :=
      Finset.abs_sum_le_sum_abs (fun i => w i * (estimate i - truth i)) Finset.univ
    _ = ∑ i, w i * |estimate i - truth i| := by
      apply Finset.sum_congr rfl
      intro i _
      rw [abs_mul, abs_of_nonneg (hNonneg i)]
    _ ≤ ∑ i, w i * eps i := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_left (hEps i) (hNonneg i)

/-- General composite confidence propagation: componentwise bounds propagate
without dimension distortion through positive linear aggregation. -/
theorem compConf {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i)
    (truth estimate sigma : Fin m → ℝ)
    (beta : ℝ)
    (hConf : ∀ i : Fin m, |estimate i - truth i| ≤ beta * sigma i) :
    |compMean w estimate - compMean w truth| ≤ beta * compMean w sigma := by
  have hBound := compError w hNonneg truth estimate (fun i => beta * sigma i) hConf
  have hScale : compMean w (fun i => beta * sigma i) = beta * compMean w sigma := by
    unfold compMean
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rwa [hScale] at hBound

/-- Paired observation structure recording mean difference and sample variance. -/
structure PairedObs where
  meanDiff : ℝ
  sampleVar : ℝ

/-- Paired consistency: the coupled difference evaluation matches the difference
of scalar evaluations evaluated in the same environment. -/
def PairedConsistent {State World : Type*}
    (scalarEval : State → World → ℝ)
    (pairedEval : State → State → World → PairedObs) : Prop :=
  ∀ cand base w, (pairedEval cand base w).meanDiff = scalarEval cand w - scalarEval base w

/-- Algebraic variance decomposition under environmental covariance. -/
theorem pairedVariance (varBase varCand covar : ℝ) :
    varBase + varCand - 2 * covar = varCand + (varBase - 2 * covar) := by
  ring

/-- Strict variance reduction under Common Random Numbers (CRN): when environmental
covariance exceeds half the baseline variance, paired evaluation has strictly less
variance than evaluating the candidate alone. -/
theorem pairedAdvantage
    (varBase varCand covar : ℝ)
    (hCovar : varBase / 2 < covar) :
    varBase + varCand - 2 * covar < varCand := by
  linarith

/-- Uniform channel weights for arbitrary dimension m (1 / m on m components). -/
def uniformWeights (m : ℕ) : Fin m → ℝ := fun _ => (1 / (m : ℝ))

/-- Uniform channel weights are non-negative. -/
theorem uniformNonneg (m : ℕ) : ∀ i : Fin m, 0 ≤ uniformWeights m i := by
  intro _
  unfold uniformWeights
  positivity

/-- Mean aggregation of m evaluation channels as a composite BO instance. -/
def uniformMean (m : ℕ) (v : Fin m → ℝ) : ℝ :=
  compMean (uniformWeights m) v

/-- Uniform channel mean formula matches scalar average. -/
theorem uniformFormula (m : ℕ) (v : Fin m → ℝ) :
    uniformMean m v = (1 / (m : ℝ)) * ∑ i : Fin m, v i := by
  unfold uniformMean compMean uniformWeights
  rw [← Finset.mul_sum]

/-- Uniform channel mean aggregation is additive. -/
theorem uniformAdd (m : ℕ) (u v : Fin m → ℝ) :
    uniformMean m (u + v) = uniformMean m u + uniformMean m v :=
  compAdd (uniformWeights m) u v

/-- Uniform channel mean scales with scalar multipliers. -/
theorem uniformScale (m : ℕ) (c : ℝ) (v : Fin m → ℝ) :
    uniformMean m (fun i => c * v i) = c * uniformMean m v :=
  compScale (uniformWeights m) c v

/-- Uniform channel error bound specialized from general composite BO. -/
theorem uniformError (m : ℕ)
    (truth estimate eps : Fin m → ℝ)
    (hEps : ∀ i : Fin m, |estimate i - truth i| ≤ eps i) :
    |uniformMean m estimate - uniformMean m truth| ≤ uniformMean m eps :=
  compError (uniformWeights m) (uniformNonneg m) truth estimate eps hEps

/-- Uniform channel confidence propagation specialized from general composite BO. -/
theorem uniformConf (m : ℕ)
    (truth estimate sigma : Fin m → ℝ)
    (beta : ℝ)
    (hConf : ∀ i : Fin m, |estimate i - truth i| ≤ beta * sigma i) :
    |uniformMean m estimate - uniformMean m truth| ≤ beta * uniformMean m sigma :=
  compConf (uniformWeights m) (uniformNonneg m) truth estimate sigma beta hConf

/-- Common-offset invariance of squared residuals under paired additive shifts. -/
theorem residualShift (truth pred shift : ℝ) :
    ((truth + shift) - (pred + shift)) ^ 2 = (truth - pred) ^ 2 := by
  ring

/-- Pointwise quadratic decomposition of a predictably centered residual `obs - center`
around the conditional mean `mean`. -/
theorem centerSquare (obs mean center : ℝ) :
    (obs - center) ^ 2 =
      (obs - mean) ^ 2 + (mean - center) ^ 2 + 2 * (obs - mean) * (mean - center) := by
  ring

/-- Triangle inequality for positive linear composite aggregation: the magnitude
of a weighted sum is bounded by the weighted sum of component magnitudes. -/
theorem vectorBound {m : ℕ} (w : Fin m → ℝ)
    (hNonneg : ∀ i : Fin m, 0 ≤ w i) (g : Fin m → ℝ) :
    |compMean w g| ≤ compMean w (fun i => |g i|) := by
  have hZero : compMean w (fun _ => 0) = 0 := by
    unfold compMean
    simp
  have hBound := compError w hNonneg (fun _ => 0) g (fun i => |g i|)
    (fun i => by simp)
  rwa [hZero, sub_zero] at hBound

/-- Strict componentwise magnitude dominance over scalar aggregation when two
equally weighted channels carry equal and opposite values. -/
theorem vectorStrict (c : ℝ) (posC : 0 < c) :
    let w : Fin 2 → ℝ := fun _ => (1 / 2 : ℝ)
    let g : Fin 2 → ℝ := ![c, -c]
    compMean w g = 0 ∧ compMean w (fun i => |g i|) = c ∧
      |compMean w g| < compMean w (fun i => |g i|) := by
  dsimp only
  have hMean : compMean (fun _ : Fin 2 => (1 / 2 : ℝ)) ![c, -c] = 0 := by
    unfold compMean
    simp [Fin.sum_univ_two]
  have hAbs : compMean (fun _ : Fin 2 => (1 / 2 : ℝ)) (fun i : Fin 2 => |![c, -c] i|) = c := by
    unfold compMean
    simp [Fin.sum_univ_two, abs_of_pos posC]
    ring
  refine ⟨hMean, hAbs, ?_⟩
  rw [hMean, hAbs, abs_zero]
  exact posC

#print axioms residualShift
#print axioms centerSquare
#print axioms vectorBound
#print axioms vectorStrict

end

end BauxResearch.Optimization.EvaluationSpace
