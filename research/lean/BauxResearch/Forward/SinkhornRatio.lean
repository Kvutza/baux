import BauxResearch.Forward.Residual
import BauxResearch.Forward.SeedSensitivity
import BauxResearch.Forward.MixingSensitivity
import BauxResearch.Forward.NormSensitivity

/-!
# Multiplicative Sinkhorn stability and transport difference

Absolute-error sensitivity through floored row or column normalization charges
the reciprocal of the `1e-20` floor whenever a column is weak. In projective or
multiplicative coordinates, the floor is identical in both matrices and acts as
a contraction toward one rather than an expansion. This file proves that
multiplicative ratio bounds propagate through floored row normalization, column
normalization, and multi-step Sinkhorn without any reciprocal-floor factor, and
then converts that ratio plus the incumbent residual into a squared Euclidean
operator-difference bound.
-/

namespace BauxResearch.Forward.SinkhornRatio

noncomputable section
open scoped BigOperators
open Matrix Transport MixingSensitivity

/-- Entrywise multiplicative equivalence for nonnegative square matrices. -/
def Ratio {streams : ℕ} (factor : ℝ) (first second : Square streams) : Prop :=
  (∀ row column, 0 ≤ first row column) ∧
    (∀ row column, 0 ≤ second row column) ∧
    (∀ row column, first row column ≤ factor * second row column) ∧
    (∀ row column, second row column ≤ factor * first row column)

theorem Ratio.refl {streams : ℕ} (matrix : Square streams)
    (entries : ∀ row column, 0 ≤ matrix row column) :
    Ratio 1 matrix matrix := by
  refine ⟨entries, entries, ?_, ?_⟩ <;>
    (intro row column; simp)

theorem Ratio.symm {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (ratio : Ratio factor first second) :
    Ratio factor second first :=
  ⟨ratio.2.1, ratio.1, ratio.2.2.2, ratio.2.2.1⟩

theorem rowBound {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (ratio : Ratio factor first second) (row : Fin streams) :
    rowSum first row ≤ factor * rowSum second row := by
  unfold rowSum
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun column _ => ratio.2.2.1 row column

theorem columnBound {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (ratio : Ratio factor first second) (column : Fin streams) :
    columnSum first column ≤ factor * columnSum second column := by
  unfold columnSum
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun row _ => ratio.2.2.1 row column

/-- Floored row denominators preserve the multiplicative ratio without charging
a reciprocal floor factor, even when the floor is active. -/
theorem floorRow {streams : ℕ} (floor factor : ℝ)
    (first second : Square streams)
    (floorNonnegative : 0 ≤ floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) (row : Fin streams) :
    max floor (rowSum first row) ≤ factor * max floor (rowSum second row) := by
  have factorNonnegative : 0 ≤ factor := by linarith
  apply max_le
  · have base : floor ≤ max floor (rowSum second row) := le_max_left _ _
    have maxNonnegative : 0 ≤ max floor (rowSum second row) :=
      floorNonnegative.trans base
    nlinarith
  · exact (rowBound ratio row).trans
      (mul_le_mul_of_nonneg_left (le_max_right _ _) factorNonnegative)

/-- Floored column denominators preserve the multiplicative ratio even on weak
columns where the `1e-20` floor is active. -/
theorem floorColumn {streams : ℕ} (floor factor : ℝ)
    (first second : Square streams)
    (floorNonnegative : 0 ≤ floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) (column : Fin streams) :
    max floor (columnSum first column) ≤
      factor * max floor (columnSum second column) := by
  have factorNonnegative : 0 ≤ factor := by linarith
  apply max_le
  · have base : floor ≤ max floor (columnSum second column) := le_max_left _ _
    have maxNonnegative : 0 ≤ max floor (columnSum second column) :=
      floorNonnegative.trans base
    nlinarith
  · exact (columnBound ratio column).trans
      (mul_le_mul_of_nonneg_left (le_max_right _ _) factorNonnegative)

private theorem normStep {a b d1 d2 factor : ℝ}
    (bNonneg : 0 ≤ b) (d1Pos : 0 < d1) (d2Pos : 0 < d2) (factorOne : 1 ≤ factor)
    (numBound : a ≤ factor * b) (denBound : d2 ≤ factor * d1) :
    a / d1 ≤ factor ^ 2 * (b / d2) := by
  have factorPos : 0 < factor := by linarith
  rw [div_le_iff₀ d1Pos]
  have rewritten : factor ^ 2 * (b / d2) * d1 =
      (factor * b) * (factor * d1) / d2 := by ring
  rw [rewritten, le_div_iff₀ d2Pos]
  exact mul_le_mul numBound denBound d2Pos.le (mul_nonneg factorPos.le bNonneg)

/-- One floored row normalization squares the multiplicative ratio bound. -/
theorem rowRatio {streams : ℕ} (floor factor : ℝ)
    (first second : Square streams)
    (floorPositive : 0 < floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) :
    Ratio (factor ^ 2) (rowNormalize floor first)
      (rowNormalize floor second) := by
  refine ⟨Row.nonnegative floor first floorPositive.le ratio.1,
    Row.nonnegative floor second floorPositive.le ratio.2.1, ?_, ?_⟩
  · intro row column
    exact normStep (ratio.2.1 row column)
      (floorPositive.trans_le (le_max_left _ _))
      (floorPositive.trans_le (le_max_left _ _))
      factorOne (ratio.2.2.1 row column)
      (floorRow floor factor second first floorPositive.le factorOne
        ratio.symm row)
  · intro row column
    exact normStep (ratio.1 row column)
      (floorPositive.trans_le (le_max_left _ _))
      (floorPositive.trans_le (le_max_left _ _))
      factorOne (ratio.2.2.2 row column)
      (floorRow floor factor first second floorPositive.le factorOne ratio row)

/-- One floored column normalization squares the multiplicative ratio bound. -/
theorem columnRatio {streams : ℕ} (floor factor : ℝ)
    (first second : Square streams)
    (floorPositive : 0 < floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) :
    Ratio (factor ^ 2) (columnNormalize floor first)
      (columnNormalize floor second) := by
  refine ⟨Column.nonnegative floor first floorPositive.le ratio.1,
    Column.nonnegative floor second floorPositive.le ratio.2.1, ?_, ?_⟩
  · intro row column
    exact normStep (ratio.2.1 row column)
      (floorPositive.trans_le (le_max_left _ _))
      (floorPositive.trans_le (le_max_left _ _))
      factorOne (ratio.2.2.1 row column)
      (floorColumn floor factor second first floorPositive.le factorOne
        ratio.symm column)
  · intro row column
    exact normStep (ratio.1 row column)
      (floorPositive.trans_le (le_max_left _ _))
      (floorPositive.trans_le (le_max_left _ _))
      factorOne (ratio.2.2.2 row column)
      (floorColumn floor factor first second floorPositive.le factorOne
        ratio column)

/-- One complete row-then-column Sinkhorn pass raises the ratio to the fourth
power. -/
theorem stepRatio {streams : ℕ} (floor factor : ℝ)
    (first second : Square streams)
    (floorPositive : 0 < floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) :
    Ratio (factor ^ 4) (step floor first) (step floor second) := by
  have rows := rowRatio floor factor first second floorPositive factorOne ratio
  have squareOne : 1 ≤ factor ^ 2 := by nlinarith
  have cols := columnRatio floor (factor ^ 2)
    (rowNormalize floor first) (rowNormalize floor second)
    floorPositive squareOne rows
  have power : (factor ^ 2) ^ 2 = factor ^ 4 := by ring
  rwa [power] at cols

/-- Finite Sinkhorn iteration propagates the multiplicative ratio without any
dependence on the reciprocal of the normalization floor. -/
theorem sinkhornRatio {streams : ℕ} (iterations : ℕ) (floor factor : ℝ)
    (first second : Square streams)
    (floorPositive : 0 < floor) (factorOne : 1 ≤ factor)
    (ratio : Ratio factor first second) :
    Ratio (factor ^ (4 ^ iterations))
      (sinkhorn iterations floor first) (sinkhorn iterations floor second) := by
  induction iterations with
  | zero =>
      simpa [sinkhorn] using ratio
  | succ iterations hypothesis =>
      have powerOne : 1 ≤ factor ^ (4 ^ iterations) :=
        one_le_pow₀ factorOne
      have next := stepRatio floor (factor ^ (4 ^ iterations))
        (sinkhorn iterations floor first) (sinkhorn iterations floor second)
        floorPositive powerOne hypothesis
      have exponent : (factor ^ (4 ^ iterations)) ^ 4 =
          factor ^ (4 ^ (iterations + 1)) := by
        rw [← pow_mul, pow_succ]
      simpa [sinkhorn, Function.iterate_succ_apply', exponent] using next

/-- Raw-logit sup error `delta` produces a `exp (2 * delta)` multiplicative
seed ratio. -/
theorem seedRatio {streams : ℕ} (positive : 0 < streams)
    (first second : Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta) :
    Ratio (Real.exp (2 * delta))
      (seed (Mixing.rawMax positive first) first)
      (seed (Mixing.rawMax positive second) second) := by
  refine ⟨fun _ _ => (Real.exp_pos _).le, fun _ _ => (Real.exp_pos _).le, ?_, ?_⟩
  · intro row column
    have clipped := abs_le.mp
      (SeedSensitivity.clipBound positive first second delta shift row column)
    have upper :
        max (first row column - Mixing.rawMax positive first) (-80) ≤
          2 * delta +
            max (second row column - Mixing.rawMax positive second) (-80) := by
      linarith
    simpa [seed, Real.exp_add] using Real.exp_le_exp.mpr upper
  · intro row column
    have clipped := abs_le.mp
      (SeedSensitivity.clipBound positive first second delta shift row column)
    have upper :
        max (second row column - Mixing.rawMax positive second) (-80) ≤
          2 * delta +
            max (first row column - Mixing.rawMax positive first) (-80) := by
      linarith
    simpa [seed, Real.exp_add] using Real.exp_le_exp.mpr upper

/-- Entrywise absolute difference is controlled by `(factor - 1)` times the
incumbent matrix entry alone. -/
theorem entryDiff {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (factorOne : 1 ≤ factor) (ratio : Ratio factor first second)
    (row column : Fin streams) :
    |second row column - first row column| ≤
      (factor - 1) * first row column := by
  have firstNonneg := ratio.1 row column
  have firstUpper := ratio.2.2.1 row column
  have secondUpper := ratio.2.2.2 row column
  apply abs_le.mpr
  constructor
  · have sqNonneg : 0 ≤ (factor - 1) ^ 2 * first row column :=
      mul_nonneg (sq_nonneg _) firstNonneg
    have gapNonneg : 0 ≤ factor * second row column - first row column := by
      linarith
    have algebraic :
        factor * (second row column - first row column +
            (factor - 1) * first row column) =
          (factor - 1) ^ 2 * first row column +
            (factor * second row column - first row column) := by
      ring
    nlinarith
  · linarith

/-- Matrix-vector difference is bounded rowwise by the incumbent matrix acting
on the entrywise absolute value of the vector. -/
theorem vectorDiff {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (factorOne : 1 ≤ factor) (ratio : Ratio factor first second)
    (value : Vector streams) (row : Fin streams) :
    |((second - first) *ᵥ value) row| ≤
      (factor - 1) * (first *ᵥ fun column => |value column|) row := by
  change |∑ column, (second row column - first row column) * value column| ≤
    (factor - 1) * ∑ column, first row column * |value column|
  rw [Finset.mul_sum]
  calc
    |∑ column, (second row column - first row column) * value column| ≤
        ∑ column, |(second row column - first row column) * value column| :=
      Finset.abs_sum_le_sum_abs _ _
    _ = ∑ column, |second row column - first row column| * |value column| := by
      refine Finset.sum_congr rfl fun column _ => abs_mul _ _
    _ ≤ ∑ column, (factor - 1) * first row column * |value column| := by
      apply Finset.sum_le_sum
      intro column _
      exact mul_le_mul_of_nonneg_right
        (entryDiff factorOne ratio row column) (abs_nonneg _)
    _ = ∑ column, (factor - 1) * (first row column * |value column|) := by
      refine Finset.sum_congr rfl fun column _ => by ring

/-- Squared Euclidean energy of the transport difference is controlled by the
multiplicative ratio and the incumbent matrix's own row/column residual. -/
theorem energyDiff {streams : ℕ} {factor : ℝ} {first second : Square streams}
    (factorOne : 1 ≤ factor) (ratio : Ratio factor first second)
    (value : Vector streams) :
    ∑ row, ((second - first) *ᵥ value) row ^ 2 ≤
      (factor - 1) ^ 2 * (1 + residual first) ^ 2 *
        ∑ column, value column ^ 2 := by
  have factorSubNonneg : 0 ≤ factor - 1 := sub_nonneg.mpr factorOne
  have absNonneg : ∀ row, 0 ≤ (first *ᵥ fun column => |value column|) row := by
    intro row
    simp only [Matrix.mulVec, dotProduct]
    exact Finset.sum_nonneg fun column _ =>
      mul_nonneg (ratio.1 row column) (abs_nonneg _)
  have rowSquare : ∀ row,
      ((second - first) *ᵥ value) row ^ 2 ≤
        (factor - 1) ^ 2 *
          ((first *ᵥ fun column => |value column|) row) ^ 2 := by
    intro row
    have rowBound := vectorDiff factorOne ratio value row
    have rhsNonneg : 0 ≤ (factor - 1) *
        (first *ᵥ fun column => |value column|) row :=
      mul_nonneg factorSubNonneg (absNonneg row)
    have diffProd : 0 ≤
        ((factor - 1) * (first *ᵥ fun column => |value column|) row -
          |((second - first) *ᵥ value) row|) *
        ((factor - 1) * (first *ᵥ fun column => |value column|) row +
          |((second - first) *ᵥ value) row|) :=
      mul_nonneg (by linarith) (by positivity)
    nlinarith [sq_abs (((second - first) *ᵥ value) row)]
  have baseEnergy := Residual.energy first ratio.1 (fun column => |value column|)
  simp only [sq_abs] at baseEnergy
  calc
    ∑ row, ((second - first) *ᵥ value) row ^ 2 ≤
        ∑ row, (factor - 1) ^ 2 *
          ((first *ᵥ fun column => |value column|) row) ^ 2 :=
      Finset.sum_le_sum fun row _ => rowSquare row
    _ = (factor - 1) ^ 2 *
        ∑ row, ((first *ᵥ fun column => |value column|) row) ^ 2 := by
      rw [← Finset.mul_sum]
    _ ≤ (factor - 1) ^ 2 *
        ((1 + residual first) ^ 2 * ∑ column, value column ^ 2) :=
      mul_le_mul_of_nonneg_left baseEnergy (sq_nonneg _)
    _ = (factor - 1) ^ 2 * (1 + residual first) ^ 2 *
        ∑ column, value column ^ 2 := by ring

/-- Complete finite-Sinkhorn operator-difference energy bound from raw-logit
sup error `delta` and the incumbent Sinkhorn residual. -/
theorem sourceDiff {streams : ℕ} (positive : 0 < streams)
    (iterations : ℕ) (floor : ℝ) (floorPositive : 0 < floor)
    (first second : Square streams) (delta : ℝ) (deltaNonnegative : 0 ≤ delta)
    (shift : ∀ row column, |second row column - first row column| ≤ delta)
    (value : Vector streams) :
    let firstSeed := seed (Mixing.rawMax positive first) first
    let secondSeed := seed (Mixing.rawMax positive second) second
    ∑ row, ((sinkhorn iterations floor secondSeed -
          sinkhorn iterations floor firstSeed) *ᵥ value) row ^ 2 ≤
      (Real.exp (2 * delta) ^ (4 ^ iterations) - 1) ^ 2 *
        (1 + residual (sinkhorn iterations floor firstSeed)) ^ 2 *
        ∑ column, value column ^ 2 := by
  intro firstSeed secondSeed
  have expOne : 1 ≤ Real.exp (2 * delta) :=
    Real.one_le_exp (by positivity)
  have powOne : 1 ≤ Real.exp (2 * delta) ^ (4 ^ iterations) :=
    one_le_pow₀ expOne
  have ratio := sinkhornRatio iterations floor (Real.exp (2 * delta))
    firstSeed secondSeed floorPositive expOne
    (seedRatio positive first second delta shift)
  exact energyDiff powOne ratio value

/-- Incumbent-conditioned raw residual-logit sensitivity: normalized-state
movement is discharged directly via `NormSensitivity.incumbentBound`. -/
theorem incumbentRaw {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (weightBound : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source) ≤ weightCap)
    (weightMoved : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source -
          firstWeights.residual stream column destination source) ≤ weightShift) :
    ∀ destination source,
      |Mixing.residualRaw secondWeights secondState destination source -
          Mixing.residualRaw firstWeights firstState destination source| ≤
        biasShift + controlShift *
          Real.sqrt ((streams * width : ℝ) * weightCap) +
        controlCap * (Real.sqrt ((6 * Mixing.stateScale firstState ^ 2 *
            gridEnergy (fun stream column =>
              secondState stream column - firstState stream column)) * weightCap) +
          Real.sqrt ((streams * width : ℝ) * weightShift)) := by
  have moveNonneg : 0 ≤ 6 * Mixing.stateScale firstState ^ 2 *
      gridEnergy (fun stream column =>
        secondState stream column - firstState stream column) := by
    have diffNonneg := energyNonneg (fun stream column =>
      secondState stream column - firstState stream column)
    positivity
  exact uniformRaw positiveStreams positiveWidth firstWeights secondWeights
    firstState secondState biasShift controlShift controlCap
    (6 * Mixing.stateScale firstState ^ 2 *
      gridEnergy (fun stream column =>
        secondState stream column - firstState stream column))
    weightCap weightShift moveNonneg biasMoved controlMoved controlBound
    (NormSensitivity.incumbentBound positiveStreams positiveWidth firstState secondState)
    weightBound weightMoved

/-- Incumbent-conditioned logarithmic seed sensitivity for the mHC residual
predictor. -/
theorem incumbentLog {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (weightBound : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source) ≤ weightCap)
    (weightMoved : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source -
          firstWeights.residual stream column destination source) ≤ weightShift)
    (row column : Fin streams) :
    |Real.log (seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw secondWeights secondState))
        (Mixing.residualRaw secondWeights secondState) row column) -
      Real.log (seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw firstWeights firstState))
        (Mixing.residualRaw firstWeights firstState) row column)| ≤
      2 * (biasShift + controlShift *
        Real.sqrt ((streams * width : ℝ) * weightCap) +
        controlCap * (Real.sqrt ((6 * Mixing.stateScale firstState ^ 2 *
            gridEnergy (fun stream column =>
              secondState stream column - firstState stream column)) * weightCap) +
          Real.sqrt ((streams * width : ℝ) * weightShift))) := by
  apply SeedSensitivity.logBound positiveStreams
    (Mixing.residualRaw firstWeights firstState)
    (Mixing.residualRaw secondWeights secondState)
  exact incumbentRaw positiveStreams positiveWidth firstWeights secondWeights
    firstState secondState biasShift controlShift controlCap weightCap weightShift
    biasMoved controlMoved controlBound weightBound weightMoved

/-- Complete incumbent-conditioned mHC predictor and finite-Sinkhorn
operator-difference energy bound. -/
theorem incumbentDiff {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (iterations : ℕ) (floor : ℝ)
    (floorPositive : 0 < floor)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (weightBound : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source) ≤ weightCap)
    (weightMoved : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source -
          firstWeights.residual stream column destination source) ≤ weightShift)
    (value : Vector streams) :
    let firstRaw := Mixing.residualRaw firstWeights firstState
    let secondRaw := Mixing.residualRaw secondWeights secondState
    let firstSeed := seed (Mixing.rawMax positiveStreams firstRaw) firstRaw
    let secondSeed := seed (Mixing.rawMax positiveStreams secondRaw) secondRaw
    let stateShift := 6 * Mixing.stateScale firstState ^ 2 *
      gridEnergy (fun stream column =>
        secondState stream column - firstState stream column)
    let delta := biasShift + controlShift *
        Real.sqrt ((streams * width : ℝ) * weightCap) +
      controlCap * (Real.sqrt (stateShift * weightCap) +
        Real.sqrt ((streams * width : ℝ) * weightShift))
    ∑ row, ((sinkhorn iterations floor secondSeed -
          sinkhorn iterations floor firstSeed) *ᵥ value) row ^ 2 ≤
      (Real.exp (2 * delta) ^ (4 ^ iterations) - 1) ^ 2 *
        (1 + residual (sinkhorn iterations floor firstSeed)) ^ 2 *
        ∑ column, value column ^ 2 := by
  intro firstRaw secondRaw firstSeed secondSeed stateShift delta
  have rawShift := incumbentRaw positiveStreams positiveWidth
    firstWeights secondWeights firstState secondState
    biasShift controlShift controlCap weightCap weightShift
    biasMoved controlMoved controlBound weightBound weightMoved
  let zeroStream : Fin streams := ⟨0, positiveStreams⟩
  have deltaNonnegative : 0 ≤ delta :=
    (abs_nonneg _).trans (rawShift zeroStream zeroStream)
  exact sourceDiff positiveStreams iterations floor floorPositive
    firstRaw secondRaw delta deltaNonnegative rawShift value

#print axioms Ratio.refl
#print axioms Ratio.symm
#print axioms rowBound
#print axioms columnBound
#print axioms floorRow
#print axioms floorColumn
#print axioms rowRatio
#print axioms columnRatio
#print axioms stepRatio
#print axioms sinkhornRatio
#print axioms seedRatio
#print axioms entryDiff
#print axioms vectorDiff
#print axioms energyDiff
#print axioms sourceDiff
#print axioms incumbentRaw
#print axioms incumbentLog
#print axioms incumbentDiff

end
end BauxResearch.Forward.SinkhornRatio
