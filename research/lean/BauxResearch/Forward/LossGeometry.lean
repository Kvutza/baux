import BauxResearch.Forward.Algebra
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Topology.MetricSpace.Pseudo.Pi

/-!
# Objective geometry derived from the actual token log loss

Mathlib exp/log order laws and finite sums give a derivative-free bound for the
real `Forward.logLoss` formula. The target-relative logit discrepancy controls
loss with coefficient one; uncentered logit discrepancy gives coefficient two.
No assumed Lipschitz constant for the full parameter-to-logit map is inserted.

Targets and evaluation indices are aligned. These bounds do not equate changing
corpus windows, prove an autonomous rollout coupling, certify sampled parameter
geometry, or establish the cost of obtaining the logits.
-/

/-!
For logits `z` and target `y`, the token loss is
`ℓ(z,y) = log(∑ᵢ exp(zᵢ)) - z_y`.  Its value is unchanged by a common shift
of all logits.  The relative-logit discrepancy therefore gives the sharper
coefficient-one bound; absolute coordinate error gives the coefficient-two
bound.  Neither result supplies a parameter-to-logit sensitivity constant.
-/

namespace BauxResearch.Forward.LossGeometry

open scoped BigOperators NNReal
noncomputable section

/-- Coordinate order bounds log-partition order, with no dimension factor. -/
theorem partitionUpper {width : ℕ} (first second : Vector width) (target : Fin width)
    (shift : ℝ) (bounded : ∀ index, first index ≤ second index + shift) :
    Real.log (∑ index, Real.exp (first index)) ≤
      Real.log (∑ index, Real.exp (second index)) + shift := by
  have sums : (∑ index, Real.exp (first index)) ≤
      (∑ index, Real.exp (second index)) * Real.exp shift := by
    calc
      _ ≤ ∑ index, Real.exp (second index + shift) :=
        Finset.sum_le_sum fun index _ => Real.exp_le_exp.mpr (bounded index)
      _ = _ := by simp_rw [Real.exp_add]; rw [Finset.sum_mul]
  have logs := Real.log_le_log (Probability.positive first target) sums
  rw [Real.log_mul (Probability.positive second target).ne' (Real.exp_pos shift).ne',
    Real.log_exp] at logs
  exact logs

/-- Only relative logits matter. Common logit shifts do not incur a penalty. -/
theorem marginBound {width : ℕ} (first second : Vector width) (target : Fin width)
    (radius : ℝ)
    (bounded : ∀ index,
      |(first index - first target) - (second index - second target)| ≤ radius) :
    |logLoss first target - logLoss second target| ≤ radius := by
  have forward := partitionUpper first second target (radius + first target - second target)
    (fun index => by have bound := (abs_le.mp (bounded index)).2; linarith)
  have backward := partitionUpper second first target (radius + second target - first target)
    (fun index => by have bound := (abs_le.mp (bounded index)).1; linarith)
  unfold logLoss
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- The actual token objective is at most twice the maximum coordinate logit
discrepancy, independently of vocabulary size or parameter dimension. -/
theorem logitBound {width : ℕ} (first second : Vector width) (target : Fin width)
    (radius : ℝ) (bounded : ∀ index, |first index - second index| ≤ radius) :
    |logLoss first target - logLoss second target| ≤ 2 * radius := by
  apply marginBound first second target (2 * radius)
  intro index
  calc
    _ = |(first index - second index) - (first target - second target)| := by congr 1; ring
    _ ≤ |first index - second index| + |first target - second target| := abs_sub _ _
    _ ≤ 2 * radius := by linarith [bounded index, bounded target]

/-- This is a Mathlib Lipschitz object, ready for ordinary composition and
the estimator's regularity hypotheses. The Pi metric is the sup metric. -/
theorem logitLipschitz {width : ℕ} (target : Fin width) :
    LipschitzWith 2 (fun scores : Vector width => logLoss scores target) := by
  apply LipschitzWith.of_dist_le_mul
  intro first second
  rw [Real.dist_eq]
  exact logitBound first second target (dist first second) fun index => by
    simpa only [Real.dist_eq] using dist_le_pi_dist first second index

/-- Only the parameter-to-logit geometry remains to be supplied in this
composition. That hypothesis is not inferred from smooth primitive formulas. -/
theorem parameterBound {State : Type*} [PseudoMetricSpace State] {width : ℕ}
    (logits : State → Vector width) (target : Fin width) (constant : ℝ≥0)
    (regular : LipschitzWith constant logits) :
    LipschitzWith (2 * constant) (fun state => logLoss (logits state) target) :=
  (logitLipschitz target).comp regular

/-- Aligned token averaging preserves the bound instead of multiplying it by
the token count. Each token may have its own logit discrepancy. -/
theorem averageBound {tokens width : ℕ} (first second : Rows tokens width)
    (targets : Fin tokens → Fin width) (radius : Vector tokens)
    (bounded : ∀ token index, |first token index - second token index| ≤ radius token) :
    |average (fun token => logLoss (first token) (targets token)) -
      average (fun token => logLoss (second token) (targets token))| ≤ 2 * average radius := by
  have sums : |∑ token, (logLoss (first token) (targets token) -
      logLoss (second token) (targets token))| ≤ ∑ token, 2 * radius token :=
    (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun token _ =>
      logitBound (first token) (second token) (targets token) (radius token) (bounded token))
  unfold average
  rw [← sub_div, ← Finset.sum_sub_distrib, abs_div]
  have divided := div_le_div_of_nonneg_right sums (Nat.cast_nonneg tokens : (0 : ℝ) ≤ tokens)
  simpa only [abs_of_nonneg (Nat.cast_nonneg tokens : (0 : ℝ) ≤ tokens),
    ← Finset.mul_sum, mul_div_assoc] using divided

/-- A uniform absolute logit bound gives an explicit token-loss interval. -/
theorem rangeBound {width : ℕ}
    (scores : Vector width)
    (target : Fin width)
    (radius : ℝ)
    (bounded : ∀ index, |scores index| ≤ radius) :
    logLoss scores target ∈
      Set.Icc 0 (Real.log width + 2 * radius) := by
  have compared := logitBound scores (fun _ => 0) target radius fun index => by
    simpa using bounded index
  have zeroLoss : logLoss (fun _ : Fin width => 0) target = Real.log width := by
    unfold logLoss
    simp
  constructor
  · exact Loss.nonnegative scores target
  · rw [zeroLoss] at compared
    linarith [le_abs_self (logLoss scores target - Real.log width)]

/-- If every token logit is uniformly bounded, averaging over an arbitrarily
long natural window preserves the same NLL interval. Window length does not
appear in this worst-case range. -/
theorem windowRange {tokens width : ℕ}
    (tokenPositive : 0 < tokens)
    (scores : Rows tokens width)
    (targets : Fin tokens → Fin width)
    (radius : ℝ)
    (bounded : ∀ token index, |scores token index| ≤ radius) :
    average (fun token => logLoss (scores token) (targets token)) ∈
      Set.Icc 0 (Real.log width + 2 * radius) := by
  have countPositive : 0 < (tokens : ℝ) := by exact_mod_cast tokenPositive
  have tokenBound : ∀ token,
      logLoss (scores token) (targets token) ∈
        Set.Icc 0 (Real.log width + 2 * radius) :=
    fun token => rangeBound (scores token) (targets token) radius
      (bounded token)
  constructor
  · unfold average
    exact div_nonneg
      (Finset.sum_nonneg fun token _ => (tokenBound token).1)
      countPositive.le
  · unfold average
    apply (div_le_iff₀ countPositive).2
    calc
      ∑ token, logLoss (scores token) (targets token) ≤
          ∑ _token : Fin tokens, (Real.log width + 2 * radius) :=
        Finset.sum_le_sum fun token _ => (tokenBound token).2
      _ = (Real.log width + 2 * radius) * (tokens : ℝ) := by
        simp [mul_add, mul_comm]

/-- A trust-region candidate inherits an explicit whole-window NLL interval
from an incumbent logit envelope and a parameter-to-logit Lipschitz bound. -/
theorem trustRange
    {State : Type*}
    [PseudoMetricSpace State]
    {tokens width : ℕ}
    (tokenPositive : 0 < tokens)
    (logits : State → Rows tokens width)
    (targets : Fin tokens → Fin width)
    (constant : ℝ≥0)
    (regular : LipschitzWith constant logits)
    (incumbent candidate : State)
    (baseRadius trustRadius : ℝ)
    (baseBound : ∀ token index,
      |logits incumbent token index| ≤ baseRadius)
    (near : dist candidate incumbent ≤ trustRadius) :
    average (fun token => logLoss (logits candidate token) (targets token)) ∈
      Set.Icc 0
        (Real.log width + 2 * (baseRadius + constant * trustRadius)) := by
  apply windowRange tokenPositive (logits candidate) targets
    (baseRadius + constant * trustRadius)
  intro token index
  have coordinate :
      |logits candidate token index - logits incumbent token index| ≤
        dist (logits candidate) (logits incumbent) := by
    calc
      _ = dist (logits candidate token index) (logits incumbent token index) := by
        rw [Real.dist_eq]
      _ ≤ dist (logits candidate token) (logits incumbent token) :=
        dist_le_pi_dist _ _ index
      _ ≤ dist (logits candidate) (logits incumbent) :=
        dist_le_pi_dist _ _ token
  have moved :
      dist (logits candidate) (logits incumbent) ≤ constant * trustRadius :=
    regular.dist_le_mul_of_le near
  calc
    |logits candidate token index| =
        |(logits candidate token index - logits incumbent token index) +
          logits incumbent token index| := by ring_nf
    _ ≤ |logits candidate token index - logits incumbent token index| +
        |logits incumbent token index| := abs_add_le _ _
    _ ≤ baseRadius + constant * trustRadius := by
      linarith [coordinate, baseBound token index, moved]

#print axioms partitionUpper
#print axioms marginBound
#print axioms logitBound
#print axioms logitLipschitz
#print axioms parameterBound
#print axioms averageBound
#print axioms rangeBound
#print axioms windowRange
#print axioms trustRange

end
end BauxResearch.Forward.LossGeometry
