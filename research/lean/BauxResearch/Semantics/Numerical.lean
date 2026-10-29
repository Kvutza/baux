import BauxResearch.Semantics.NumericalRuntime
import BauxResearch.Semantics.Optimizer
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SqrtSemantics
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
import FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Proof

/-!
# Checking the actual scalar arithmetic at the acceptance boundary

The checker consumes the observed binary64 words. It recomputes the threshold
with integer-backed IEEE reference operations and checks the actual decision
against exact dyadic comparison. Its success supplies rounding and comparison
facts; it does not turn empirical calibration into a bound on objective error.
The compiled checker still depends on Lean compilation and runtime correctness.
-/

namespace BauxResearch.Semantics.Numerical

open FloatLib.Floats.Formats.BinaryInterchange

/-- Success derives the finite-domain and actual-result obligations from bits. -/
theorem thresholdParts (multiplier variance actual : Word)
    (checked : checkThreshold multiplier variance actual = true) :
    Model.isFinite multiplier = true ∧ Model.isFinite variance = true ∧
    Model.isFinite actual = true ∧
    (Model.isZero variance = true ∨ Model.signBit variance = false) ∧
    actual = threshold multiplier variance := by
  simpa [checkThreshold, domain, Bool.and_eq_true, Bool.or_eq_true, and_assoc] using checked

/-- Checked output has the exact twice-rounded real semantics, including subnormals. -/
theorem thresholdReal (multiplier variance actual : Word)
    (checked : checkThreshold multiplier variance actual = true) :
    Model.toReal actual = Model.roundAt FloatFormat.binary64
      (Model.toReal multiplier * Model.roundAt FloatFormat.binary64
        (Real.sqrt (Model.toReal variance))) := by
  obtain ⟨hm, hv, ha, hd, he⟩ := thresholdParts multiplier variance actual checked
  have hs := Model.isFinite_sqrt_of_isFinite variance (by decide) hv hd
  have ht : Model.isFinite (threshold multiplier variance) = true := he ▸ ha
  rw [he]
  unfold threshold
  rw [Model.toReal_mul_eq_roundAt multiplier (Model.sqrt variance) (by decide) hm hs ht]
  rw [Model.toReal_sqrt_eq_roundAt variance (by decide) hv hd]

/-- This error budget is derived from both IEEE nearest-even rounding operations. -/
noncomputable def roundBudget (multiplier variance : Word) : ℝ :=
  let exact := Real.sqrt (Model.toReal variance)
  let rounded := Model.roundAt FloatFormat.binary64 exact
  Model.epsilonAt FloatFormat.binary64 (Model.toReal multiplier * rounded) +
    |Model.toReal multiplier| * Model.epsilonAt FloatFormat.binary64 exact

/-- No caller-supplied rounding cap: the checked words determine the half-ULP budget. -/
theorem thresholdError (multiplier variance actual : Word)
    (checked : checkThreshold multiplier variance actual = true) :
    |Model.toReal actual - Model.toReal multiplier * Real.sqrt (Model.toReal variance)| ≤
      roundBudget multiplier variance := by
  rw [thresholdReal multiplier variance actual checked]
  let exact := Real.sqrt (Model.toReal variance)
  let rounded := Model.roundAt FloatFormat.binary64 exact
  have hs := Model.abs_roundAt_sub_le FloatFormat.binary64 exact
  have hp := Model.abs_roundAt_sub_le FloatFormat.binary64 (Model.toReal multiplier * rounded)
  calc
    _ ≤ |Model.roundAt FloatFormat.binary64 (Model.toReal multiplier * rounded) -
          Model.toReal multiplier * rounded| +
        |Model.toReal multiplier * rounded - Model.toReal multiplier * exact| :=
      abs_sub_le _ _ _
    _ ≤ Model.epsilonAt FloatFormat.binary64 (Model.toReal multiplier * rounded) +
        |Model.toReal multiplier| * Model.epsilonAt FloatFormat.binary64 exact := by
      apply add_le_add hp
      rw [← mul_sub, abs_mul]
      exact mul_le_mul_of_nonneg_left hs (abs_nonneg _)
    _ = roundBudget multiplier variance := rfl

/-- The observed Boolean agrees with strict real comparison when the checker succeeds. -/
theorem decisionReal (gain multiplier variance actual : Word) (accepted : Bool)
    (checked : checkDecision gain multiplier variance actual accepted = true) :
    (accepted = true ↔ Model.toReal actual < Model.toReal gain) := by
  have parts : checkThreshold multiplier variance actual = true ∧
      Model.isFinite gain = true ∧
      accepted = decide (Model.compare actual gain = some .lt) := by
    simpa [checkDecision, Bool.and_eq_true, and_assoc] using checked
  have finite := (thresholdParts multiplier variance actual parts.1).2.2.1
  rw [parts.2.2, decide_eq_true_eq]
  exact Model.compare_eq_some_lt_iff_toReal_lt_of_isFinite actual gain finite parts.2.1

/-- A checked machine decision composes with separately justified objective errors.
The numerical premise is checked; the two observation bounds and margin coverage
remain mathematical obligations of the measurement and statistical model. -/
theorem acceptedSafe (gain multiplier variance actual : Word)
    (base candidate baseError candidateError : ℝ)
    (checked : checkDecision gain multiplier variance actual true = true)
    (measured : |Model.toReal gain - (candidate - base)| ≤ baseError + candidateError)
    (covered : baseError + candidateError ≤ Model.toReal actual) :
    base < candidate := by
  have strict := (decisionReal gain multiplier variance actual true checked).mp rfl
  have lower := (abs_le.mp measured).2
  linarith

/-- Exact real expression retaining every source rounding operation. -/
noncomputable def sourceRounded (kind : UInt8) (multiplier left right : Word) : ℝ :=
  let r := Model.roundAt FloatFormat.binary64
  let scale := if kind = 0 then r (Real.sqrt (Model.toReal left))
    else if kind = 1 then r (Real.sqrt (r (Model.toReal left + Model.toReal right)))
    else r (r (Real.sqrt (Model.toReal left)) + r (Real.sqrt (Model.toReal right)))
  r (Model.toReal multiplier * scale)

theorem sourceParts (multiplier left right actual : Word)
    (checked : sourceDomain multiplier left right actual = true) :
    Model.isFinite multiplier = true ∧ Model.isFinite left = true ∧
    Model.isFinite actual = true ∧
    (Model.isZero left = true ∨ Model.signBit left = false) ∧
    Model.isFinite right = true ∧
    (Model.isZero right = true ∨ Model.signBit right = false) := by
  simpa [sourceDomain, domain, Bool.and_eq_true, Bool.or_eq_true, and_assoc] using checked

/-- The real calibration threshold needs rounding headroom before it covers
observation error. The headroom here comes from checked IEEE operations. -/
theorem budgetSafe (gain multiplier variance actual : Word)
    (base candidate baseError candidateError : ℝ)
    (checked : checkDecision gain multiplier variance actual true = true)
    (measured : |Model.toReal gain - (candidate - base)| ≤ baseError + candidateError)
    (covered : baseError + candidateError + roundBudget multiplier variance ≤
      Model.toReal multiplier * Real.sqrt (Model.toReal variance)) :
    base < candidate := by
  have parts : checkThreshold multiplier variance actual = true ∧
      Model.isFinite gain = true ∧
      true = decide (Model.compare actual gain = some .lt) := by
    simpa [checkDecision, Bool.and_eq_true, and_assoc] using checked
  apply acceptedSafe gain multiplier variance actual base candidate baseError candidateError
    checked measured
  have lower := (abs_le.mp (thresholdError multiplier variance actual parts.1)).1
  linarith

/-- Square root is uniformly Hölder, including across zero. -/
theorem sqrtPerturb (x y : ℝ) :
    |Real.sqrt x - Real.sqrt y| ≤ Real.sqrt |x - y| := by
  apply Real.abs_le_sqrt
  have ax := le_abs_self (x - y)
  have ay := neg_le_abs (x - y)
  have sx := Real.sqrt_nonneg x
  have sy := Real.sqrt_nonneg y
  by_cases hx : 0 ≤ x
  · have ex := Real.sq_sqrt hx
    by_cases hy : 0 ≤ y
    · have ey := Real.sq_sqrt hy
      rcases le_total (Real.sqrt x) (Real.sqrt y) with h | h
      · have product := mul_nonneg sx (sub_nonneg.mpr h)
        nlinarith
      · have product := mul_nonneg sy (sub_nonneg.mpr h)
        nlinarith
    · rw [Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hy)]
      nlinarith
  · rw [Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hx)]
    by_cases hy : 0 ≤ y
    · have ey := Real.sq_sqrt hy
      nlinarith
    · rw [Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hy)]
      simpa using abs_nonneg (x - y)

/-- Propagate a proved scale error through the actual nearest-even product. -/
theorem productError (multiplier scale exact bound : ℝ)
    (hscale : |scale - exact| ≤ bound) :
    |Model.roundAt FloatFormat.binary64 (multiplier * scale) - multiplier * exact| ≤
      Model.epsilonAt FloatFormat.binary64 (multiplier * scale) +
        |multiplier| * bound := by
  calc
    _ ≤ |Model.roundAt FloatFormat.binary64 (multiplier * scale) - multiplier * scale| +
        |multiplier * scale - multiplier * exact| := abs_sub_le _ _ _
    _ ≤ _ := by
      apply add_le_add (Model.abs_roundAt_sub_le _ _)
      rw [← mul_sub, abs_mul]
      exact mul_le_mul_of_nonneg_left hscale (abs_nonneg _)

/-- All three formulas have checked rounded semantics, without a native-operation oracle. -/
theorem sourceReal (kind : UInt8) (multiplier left right actual : Word)
    (checked : sourceCheck kind multiplier left right actual = true) :
    Model.toReal actual = sourceRounded kind multiplier left right := by
  have parts : sourceDomain multiplier left right actual = true ∧
      sourceCore kind multiplier left right actual = true := by
    simpa [sourceCheck, Bool.and_eq_true] using checked
  obtain ⟨hc, hs⟩ := parts
  obtain ⟨hm, hl, ha, hdl, hr, hdr⟩ := sourceParts multiplier left right actual hc
  by_cases h0 : kind = 0
  · have ht : checkThreshold multiplier left actual = true := by
      simpa [sourceCore, h0] using hs
    simpa [sourceRounded, h0] using thresholdReal multiplier left actual ht
  by_cases h1 : kind = 1
  · have ht : checkThreshold multiplier (Model.add left right) actual = true := by
      simpa [sourceCore, h0, h1] using hs
    have hf := (thresholdParts multiplier (Model.add left right) actual ht).2.1
    have result := thresholdReal multiplier (Model.add left right) actual ht
    rw [Model.toReal_add_eq_roundAt left right (by decide) hl hr hf] at result
    simpa [sourceRounded, h0, h1] using result
  by_cases h2 : kind = 2
  · have parts : Model.isFinite (sourceScale kind left right) = true ∧
        actual = Model.mul multiplier (sourceScale kind left right) := by
      simpa [sourceCore, h0, h1, h2, Bool.and_eq_true] using hs
    have hsl := Model.isFinite_sqrt_of_isFinite left (by decide) hl hdl
    have hsr := Model.isFinite_sqrt_of_isFinite right (by decide) hr hdr
    have hsum : Model.isFinite (Model.add (Model.sqrt left) (Model.sqrt right)) = true := by
      simpa [sourceScale, h0, h1] using parts.1
    rw [parts.2] at ha ⊢
    rw [Model.toReal_mul_eq_roundAt multiplier _ (by decide) hm parts.1 ha]
    simp only [sourceScale, h0, h1, ite_false]
    rw [Model.toReal_add_eq_roundAt _ _ (by decide) hsl hsr hsum,
      Model.toReal_sqrt_eq_roundAt left (by decide) hl hdl,
      Model.toReal_sqrt_eq_roundAt right (by decide) hr hdr]
    simp [sourceRounded, h0, h1]
  · simp [sourceCore, h0, h1, h2] at hs

/-- The unrounded expression corresponding to the selected source formula. -/
noncomputable def sourceIdeal (kind : UInt8) (multiplier left right : Word) : ℝ :=
  Model.toReal multiplier *
    (if kind = 0 then Real.sqrt (Model.toReal left)
     else if kind = 1 then Real.sqrt (Model.toReal left + Model.toReal right)
     else Real.sqrt (Model.toReal left) + Real.sqrt (Model.toReal right))

/-- Every rounding contributes separately. The combined-variance path also pays
for square-root sensitivity to rounding its input sum, including at zero. -/
noncomputable def sourceBudget (kind : UInt8) (multiplier left right : Word) : ℝ :=
  let r := Model.roundAt FloatFormat.binary64
  let e := Model.epsilonAt FloatFormat.binary64
  let k := Model.toReal multiplier
  let l := Model.toReal left
  let v := Model.toReal right
  if kind = 0 then e (k * r (Real.sqrt l)) + |k| * e (Real.sqrt l)
  else if kind = 1 then
    let sum := r (l + v)
    e (k * r (Real.sqrt sum)) + |k| * (e (Real.sqrt sum) + Real.sqrt (e (l + v)))
  else
    let roots := r (Real.sqrt l) + r (Real.sqrt v)
    e (k * r roots) + |k| * (e roots + (e (Real.sqrt l) + e (Real.sqrt v)))

/-- Derive the full source rounding budget from the actual checked output words. -/
theorem sourceError (kind : UInt8) (multiplier left right actual : Word)
    (checked : sourceCheck kind multiplier left right actual = true) :
    |Model.toReal actual - sourceIdeal kind multiplier left right| ≤
      sourceBudget kind multiplier left right := by
  rw [sourceReal kind multiplier left right actual checked]
  let r := Model.roundAt FloatFormat.binary64
  let e := Model.epsilonAt FloatFormat.binary64
  let l := Model.toReal left
  let v := Model.toReal right
  by_cases h0 : kind = 0
  · simpa only [sourceRounded, sourceIdeal, sourceBudget, if_pos h0] using
      productError (Model.toReal multiplier) (r (Real.sqrt l)) (Real.sqrt l)
        (e (Real.sqrt l)) (Model.abs_roundAt_sub_le _ _)
  by_cases h1 : kind = 1
  · have perturb : |Real.sqrt (r (l + v)) - Real.sqrt (l + v)| ≤
        Real.sqrt (e (l + v)) :=
      (sqrtPerturb (r (l + v)) (l + v)).trans
        (Real.sqrt_le_sqrt (Model.abs_roundAt_sub_le _ _))
    have scale : |r (Real.sqrt (r (l + v))) - Real.sqrt (l + v)| ≤
        e (Real.sqrt (r (l + v))) + Real.sqrt (e (l + v)) := by
      exact (abs_sub_le _ _ _).trans
        (add_le_add (Model.abs_roundAt_sub_le _ _) perturb)
    simpa only [sourceRounded, sourceIdeal, sourceBudget, if_neg h0, if_pos h1] using
      productError (Model.toReal multiplier) _ _ _ scale
  · have roots : |(r (Real.sqrt l) + r (Real.sqrt v)) -
          (Real.sqrt l + Real.sqrt v)| ≤ e (Real.sqrt l) + e (Real.sqrt v) := by
      calc
        _ = |(r (Real.sqrt l) - Real.sqrt l) + (r (Real.sqrt v) - Real.sqrt v)| := by
          congr 1
          ring
        _ ≤ _ := (abs_add_le _ _).trans
          (add_le_add (Model.abs_roundAt_sub_le _ _) (Model.abs_roundAt_sub_le _ _))
    have scale : |r (r (Real.sqrt l) + r (Real.sqrt v)) -
          (Real.sqrt l + Real.sqrt v)| ≤
        e (r (Real.sqrt l) + r (Real.sqrt v)) + (e (Real.sqrt l) + e (Real.sqrt v)) :=
      (abs_sub_le _ _ _).trans (add_le_add (Model.abs_roundAt_sub_le _ _) roots)
    simpa only [sourceRounded, sourceIdeal, sourceBudget, if_neg h0, if_neg h1] using
      productError (Model.toReal multiplier) _ _ _ scale

theorem sourceOrder (kind : UInt8) (gain multiplier left right actual : Word)
    (accepted : Bool)
    (checked : sourceDecision kind gain multiplier left right actual accepted = true) :
    (accepted = true ↔ Model.toReal actual < Model.toReal gain) := by
  have parts : sourceCheck kind multiplier left right actual = true ∧
      Model.isFinite gain = true ∧
      accepted = decide (Model.compare actual gain = some .lt) := by
    simpa [sourceDecision, Bool.and_eq_true, and_assoc] using checked
  have hc : sourceDomain multiplier left right actual = true := by
    have split : sourceDomain multiplier left right actual = true ∧
        sourceCore kind multiplier left right actual = true := by
      simpa [sourceCheck, Bool.and_eq_true] using parts.1
    exact split.1
  have finite := (sourceParts multiplier left right actual hc).2.2.1
  rw [parts.2.2, decide_eq_true_eq]
  exact Model.compare_eq_some_lt_iff_toReal_lt_of_isFinite actual gain finite parts.2.1

/-- Compose the check with the extracted production comparison and the existing
Mathlib-facing acceptance law. This is a proposition-level link, not a file association. -/
theorem rustAccept (kind : UInt8) (gain multiplier left right actual : Word)
    (checked : sourceDecision kind gain multiplier left right actual
      (Optimizer.acceptBits gain.bits actual.bits) = true) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.accept_bits
      ⟨gain.bits⟩ ⟨actual.bits⟩ = Aeneas.Std.Result.ok
        (decide (BauxResearch.Optimization.EnnCommit.accepts
          (Model.toReal gain) (Model.toReal actual))) := by
  rw [Optimizer.bitsMatch]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simpa only [BauxResearch.Optimization.EnnCommit.accepts, decide_eq_true_eq] using
    sourceOrder kind gain multiplier left right actual
      (Optimizer.acceptBits gain.bits actual.bits) checked

/-- The actual source decision is safe under separately justified measurement
accuracy and coverage of that error by the checked, rounded threshold. -/
theorem sourceSafe (kind : UInt8) (gain multiplier left right actual : Word)
    (base candidate error : ℝ)
    (checked : sourceDecision kind gain multiplier left right actual true = true)
    (measured : |Model.toReal gain - (candidate - base)| ≤ error)
    (covered : error ≤ Model.toReal actual) : base < candidate := by
  have strict := (sourceOrder kind gain multiplier left right actual true checked).mp rfl
  have lower := (abs_le.mp measured).2
  linarith

/-- All source formulas compose with a real error margin after paying their
proved rounding budget. Observation accuracy remains a separate obligation. -/
theorem sourceMargin (kind : UInt8) (gain multiplier left right actual : Word)
    (base candidate error : ℝ)
    (checked : sourceDecision kind gain multiplier left right actual true = true)
    (measured : |Model.toReal gain - (candidate - base)| ≤ error)
    (covered : error + sourceBudget kind multiplier left right ≤
      sourceIdeal kind multiplier left right) : base < candidate := by
  have parts : sourceCheck kind multiplier left right actual = true ∧
      Model.isFinite gain = true ∧
      true = decide (Model.compare actual gain = some .lt) := by
    simpa [sourceDecision, Bool.and_eq_true, and_assoc] using checked
  apply sourceSafe kind gain multiplier left right actual base candidate error checked measured
  have lower := (abs_le.mp (sourceError kind multiplier left right actual parts.1)).1
  linarith

/-- A fully checked machine trace can still accept a worse true reward.
The warm-start multiplier is two, reported variance zero, measured gain one,
and true gain minus one. The missing two-unit observation-error coverage is
statistical/model error; correct rounding cannot supply it. -/
theorem calibrationGap :
    sourceDecision 0 (word 0x3ff0000000000000) (word 0x4000000000000000)
      (word 0) (word 0) (word 0) true = true ∧
    |(1 : ℝ) - (-1)| > 0 ∧ (-1 : ℝ) < 0 := by
  constructor
  · decide
  · norm_num

#print axioms thresholdReal
#print axioms thresholdError
#print axioms decisionReal
#print axioms acceptedSafe
#print axioms budgetSafe
#print axioms sourceMargin
#print axioms sourceError
#print axioms sourceReal
#print axioms sourceOrder
#print axioms sourceSafe
#print axioms rustAccept
#print axioms calibrationGap

end BauxResearch.Semantics.Numerical
