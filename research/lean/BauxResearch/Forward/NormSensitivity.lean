import BauxResearch.Forward.MixingSensitivity

/-!
# Source RMS-state sensitivity

The mHC predictor certificate consumes movement of the RMS-normalized state.
This file decomposes that quantity into incoming activation movement and
movement of the exact scalar normalization used by the source.
-/

namespace BauxResearch.Forward.NormSensitivity

noncomputable section
open scoped BigOperators
open MixingSensitivity

/-- Exact change in reciprocal square roots. The denominator records the
conditioning that is lost in a uniform epsilon bound. -/
theorem rootSplit (first second : ℝ) (firstPositive : 0 < first)
    (secondPositive : 0 < second) :
    1 / Real.sqrt second - 1 / Real.sqrt first =
      (first - second) /
        (Real.sqrt first * Real.sqrt second *
          (Real.sqrt first + Real.sqrt second)) := by
  have firstRoot : 0 < Real.sqrt first := Real.sqrt_pos.2 firstPositive
  have secondRoot : 0 < Real.sqrt second := Real.sqrt_pos.2 secondPositive
  have firstSquare : (Real.sqrt first) ^ 2 = first := Real.sq_sqrt firstPositive.le
  have secondSquare : (Real.sqrt second) ^ 2 = second := Real.sq_sqrt secondPositive.le
  have sumRoot : 0 < Real.sqrt first + Real.sqrt second := by positivity
  field_simp
  nlinarith

/-- The exact source RMS-scale movement, expressed using incoming activation
energies rather than an assumed activation range. -/
theorem scaleSplit {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    Mixing.stateScale second - Mixing.stateScale first =
      (gridEnergy first - gridEnergy second) /
        ((streams * width : ℝ) *
          Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) *
          Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000) *
          (Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) +
            Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000))) := by
  have countPositive : 0 < (streams * width : ℝ) := by positivity
  have firstPositive : 0 < gridEnergy first / (streams * width : ℝ) + 1 / 100000 := by
    have := energyNonneg first
    positivity
  have secondPositive : 0 < gridEnergy second / (streams * width : ℝ) + 1 / 100000 := by
    have := energyNonneg second
    positivity
  rw [Mixing.stateScale, Mixing.stateScale]
  change 1 / Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000) -
      1 / Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) = _
  rw [rootSplit _ _ firstPositive secondPositive]
  have countNe : (streams * width : ℝ) ≠ 0 := countPositive.ne'
  field_simp
  ring

/-- Activation-energy movement is the inner product of activation movement
with the sum of the two states. -/
theorem energySplit {streams width : ℕ} (first second : Rows streams width) :
    gridEnergy first - gridEnergy second =
      ∑ stream, ∑ column,
        (first stream column - second stream column) *
          (first stream column + second stream column) := by
  simp only [gridEnergy, ← Finset.sum_sub_distrib]
  congr 1
  ext stream
  congr 1
  ext column
  ring

/-- Source activation energy is locally controlled by incoming Euclidean
movement and the two actual activation energies. -/
theorem energyShift {streams width : ℕ} (first second : Rows streams width) :
    |gridEnergy first - gridEnergy second| ≤
      Real.sqrt (gridEnergy (fun stream column =>
          first stream column - second stream column) *
        gridEnergy (fun stream column =>
          first stream column + second stream column)) := by
  rw [energySplit]
  exact gridCauchy _ _

/-- Explicit real expression for the local RMS-scale sensitivity. -/
def scaleRadius {streams width : ℕ} (first second : Rows streams width) : ℝ :=
  Real.sqrt (gridEnergy (fun stream column =>
      first stream column - second stream column) *
    gridEnergy (fun stream column =>
      first stream column + second stream column)) /
    ((streams * width : ℝ) *
      Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) *
      Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000) *
      (Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) +
        Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000)))

/-- Local sensitivity of the exact source RMS scale. All quantities on the
right are activation energies or the source epsilon and grid dimensions. -/
theorem scaleShift {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    |Mixing.stateScale second - Mixing.stateScale first| ≤
      scaleRadius first second := by
  unfold scaleRadius
  have countPositive : 0 < (streams * width : ℝ) := by positivity
  have firstPositive : 0 < gridEnergy first / (streams * width : ℝ) + 1 / 100000 := by
    have := energyNonneg first
    positivity
  have secondPositive : 0 < gridEnergy second / (streams * width : ℝ) + 1 / 100000 := by
    have := energyNonneg second
    positivity
  have denominatorPositive : 0 <
      (streams * width : ℝ) *
        Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) *
        Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000) *
        (Real.sqrt (gridEnergy first / (streams * width : ℝ) + 1 / 100000) +
          Real.sqrt (gridEnergy second / (streams * width : ℝ) + 1 / 100000)) := by
    have firstRoot := Real.sqrt_pos.2 firstPositive
    have secondRoot := Real.sqrt_pos.2 secondPositive
    positivity
  rw [scaleSplit positiveStreams positiveWidth first second, abs_div,
    abs_of_pos denominatorPositive]
  exact (div_le_div_iff_of_pos_right denominatorPositive).2 (energyShift first second)

/-- The squared source RMS scale is globally bounded by the reciprocal of its
fixed epsilon. No activation-range assumption is needed. -/
theorem scaleBound {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (state : Rows streams width) :
    Mixing.stateScale state ^ 2 ≤ 100000 := by
  let total : ℝ := ∑ stream, ∑ column, state stream column ^ 2
  let count : ℝ := streams * width
  have countPositive : 0 < count := by
    dsimp [count]
    positivity
  have totalNonnegative : 0 ≤ total := by
    dsimp [total]
    positivity
  have denominatorPositive : 0 < total / count + 1 / 100000 := by
    positivity
  have denominatorLower : (1 : ℝ) / 100000 ≤ total / count + 1 / 100000 := by
    have quotientNonnegative : 0 ≤ total / count :=
      div_nonneg totalNonnegative countPositive.le
    linarith
  rw [Mixing.stateScale, div_pow, Real.sq_sqrt denominatorPositive.le, one_pow]
  change 1 / (total / count + 1 / 100000) ≤ 100000
  apply (div_le_iff₀ denominatorPositive).2
  nlinarith

/-- Exact decomposition of one normalized-state entry. -/
theorem stateSplit {streams width : ℕ}
    (first second : Rows streams width) (stream : Fin streams) (column : Fin width) :
    Mixing.normState second stream column - Mixing.normState first stream column =
      Mixing.stateScale second * (second stream column - first stream column) +
      (Mixing.stateScale second - Mixing.stateScale first) * first stream column := by
  simp only [Mixing.normState]
  ring

/-- Normalized-state movement is bounded by incoming state movement and scalar
RMS-scale movement. This is an exact-real source bound, not a generic declared
forward constant. -/
theorem energyBound {streams width : ℕ}
    (first second : Rows streams width) :
    gridEnergy (fun stream column =>
      Mixing.normState second stream column - Mixing.normState first stream column) ≤
      2 * Mixing.stateScale second ^ 2 *
          gridEnergy (fun stream column => second stream column - first stream column) +
        2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
          gridEnergy first := by
  unfold gridEnergy
  simp_rw [stateSplit]
  calc
    ∑ stream, ∑ column,
        (Mixing.stateScale second * (second stream column - first stream column) +
          (Mixing.stateScale second - Mixing.stateScale first) * first stream column) ^ 2 ≤
      ∑ stream, ∑ column,
        (2 * (Mixing.stateScale second *
          (second stream column - first stream column)) ^ 2 +
        2 * ((Mixing.stateScale second - Mixing.stateScale first) *
          first stream column) ^ 2) := by
      apply Finset.sum_le_sum
      intro stream _
      apply Finset.sum_le_sum
      intro column _
      nlinarith [sq_nonneg
        (Mixing.stateScale second * (second stream column - first stream column) -
          (Mixing.stateScale second - Mixing.stateScale first) * first stream column)]
    _ = _ := by
      simp only [mul_pow, Finset.sum_add_distrib, ← Finset.mul_sum]
      ring

/-- The source-wide movement bound after eliminating the candidate state's RMS
scale with the fixed source epsilon. Only scalar RMS-scale movement remains to
be related to incoming activation energy. -/
theorem sourceBound {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    gridEnergy (fun stream column =>
      Mixing.normState second stream column - Mixing.normState first stream column) ≤
      200000 * gridEnergy (fun stream column =>
        second stream column - first stream column) +
      2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
        gridEnergy first := by
  have movementNonnegative : 0 ≤ gridEnergy (fun stream column =>
      second stream column - first stream column) := energyNonneg _
  have scale := scaleBound positiveStreams positiveWidth second
  calc
    _ ≤ 2 * Mixing.stateScale second ^ 2 *
          gridEnergy (fun stream column => second stream column - first stream column) +
        2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
          gridEnergy first := energyBound first second
    _ ≤ _ := by nlinarith

/-- The source RMS-normalized state movement is bounded entirely by the
incoming states and their displacement. -/
theorem stateBound {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    gridEnergy (fun stream column =>
      Mixing.normState second stream column - Mixing.normState first stream column) ≤
      200000 * gridEnergy (fun stream column =>
        second stream column - first stream column) +
      2 * scaleRadius first second ^ 2 * gridEnergy first := by
  have radiusBound := scaleShift positiveStreams positiveWidth first second
  have radiusNonnegative : 0 ≤ scaleRadius first second :=
    (abs_nonneg _).trans radiusBound
  have radiusSquare :
      (Mixing.stateScale second - Mixing.stateScale first) ^ 2 ≤
        scaleRadius first second ^ 2 := by
    have product : 0 ≤
        (scaleRadius first second -
          |Mixing.stateScale second - Mixing.stateScale first|) *
        (scaleRadius first second +
          |Mixing.stateScale second - Mixing.stateScale first|) := by
      apply mul_nonneg
      · linarith
      · positivity
    nlinarith [sq_abs (Mixing.stateScale second - Mixing.stateScale first)]
  have firstNonnegative := energyNonneg first
  have secondTerm :
      2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
          gridEnergy first ≤
        2 * scaleRadius first second ^ 2 * gridEnergy first := by
    nlinarith [mul_nonneg (by linarith :
      0 ≤ scaleRadius first second ^ 2 -
        (Mixing.stateScale second - Mixing.stateScale first) ^ 2) firstNonnegative]
  calc
    _ ≤ 200000 * gridEnergy (fun stream column =>
          second stream column - first stream column) +
        2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
          gridEnergy first := sourceBound positiveStreams positiveWidth first second
    _ ≤ _ := by linarith

/-- Exact decomposition of one normalized-state entry around the incumbent
scale rather than the candidate scale. -/
theorem incumbentSplit {streams width : ℕ}
    (first second : Rows streams width) (stream : Fin streams) (column : Fin width) :
    Mixing.normState second stream column - Mixing.normState first stream column =
      Mixing.stateScale first * (second stream column - first stream column) +
      (Mixing.stateScale second - Mixing.stateScale first) * second stream column := by
  simp only [Mixing.normState]
  ring

/-- Incumbent-conditioned normalized-state movement bound. -/
theorem incumbentEnergy {streams width : ℕ}
    (first second : Rows streams width) :
    gridEnergy (fun stream column =>
      Mixing.normState second stream column - Mixing.normState first stream column) ≤
      2 * Mixing.stateScale first ^ 2 *
          gridEnergy (fun stream column => second stream column - first stream column) +
        2 * (Mixing.stateScale second - Mixing.stateScale first) ^ 2 *
          gridEnergy second := by
  unfold gridEnergy
  simp_rw [incumbentSplit]
  calc
    ∑ stream, ∑ column,
        (Mixing.stateScale first * (second stream column - first stream column) +
          (Mixing.stateScale second - Mixing.stateScale first) * second stream column) ^ 2 ≤
      ∑ stream, ∑ column,
        (2 * (Mixing.stateScale first *
          (second stream column - first stream column)) ^ 2 +
        2 * ((Mixing.stateScale second - Mixing.stateScale first) *
          second stream column) ^ 2) := by
      apply Finset.sum_le_sum
      intro stream _
      apply Finset.sum_le_sum
      intro column _
      nlinarith [sq_nonneg
        (Mixing.stateScale first * (second stream column - first stream column) -
          (Mixing.stateScale second - Mixing.stateScale first) * second stream column)]
    _ = _ := by
      simp only [mul_pow, Finset.sum_add_distrib, ← Finset.mul_sum]
      ring

/-- Parallelogram bound for the sum of two activation grids. -/
theorem sumEnergy {streams width : ℕ}
    (first second : Rows streams width) :
    gridEnergy (fun stream column => first stream column + second stream column) ≤
      2 * (gridEnergy first + gridEnergy second) := by
  unfold gridEnergy
  calc
    ∑ stream, ∑ column, (first stream column + second stream column) ^ 2 ≤
      ∑ stream, ∑ column, (2 * first stream column ^ 2 + 2 * second stream column ^ 2) := by
      apply Finset.sum_le_sum
      intro stream _
      apply Finset.sum_le_sum
      intro column _
      nlinarith [sq_nonneg (first stream column - second stream column)]
    _ = 2 * (∑ stream, ∑ column, first stream column ^ 2 +
          ∑ stream, ∑ column, second stream column ^ 2) := by
      simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
      ring

/-- Candidate activation energy cancels against the candidate RMS denominator in
`scaleRadius`, bounding the scalar RMS-scale shift term purely by the incumbent
RMS scale and incoming state displacement. -/
theorem scaleTerm {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    (Mixing.stateScale second - Mixing.stateScale first) ^ 2 * gridEnergy second ≤
      2 * Mixing.stateScale first ^ 2 *
        gridEnergy (fun stream column => second stream column - first stream column) := by
  let count : ℝ := streams * width
  let firstRad := gridEnergy first / count + 1 / 100000
  let secondRad := gridEnergy second / count + 1 / 100000
  let firstRoot := Real.sqrt firstRad
  let secondRoot := Real.sqrt secondRad
  let diffEnergy := gridEnergy (fun stream column => first stream column - second stream column)
  let sumGrid := gridEnergy (fun stream column => first stream column + second stream column)
  let factor := count * secondRoot * (firstRoot + secondRoot)
  have diffSymm : diffEnergy =
      gridEnergy (fun stream column => second stream column - first stream column) := by
    unfold diffEnergy gridEnergy
    congr 1
    ext stream
    congr 1
    ext column
    ring
  have countPositive : 0 < count := by
    dsimp [count]
    positivity
  have firstNonneg := energyNonneg first
  have secondNonneg := energyNonneg second
  have diffNonneg : 0 ≤ diffEnergy := energyNonneg _
  have sumNonneg : 0 ≤ sumGrid := energyNonneg _
  have firstRadPos : 0 < firstRad := by
    dsimp [firstRad]
    positivity
  have secondRadPos : 0 < secondRad := by
    dsimp [secondRad]
    positivity
  have firstRootPos : 0 < firstRoot := Real.sqrt_pos.2 firstRadPos
  have secondRootPos : 0 < secondRoot := Real.sqrt_pos.2 secondRadPos
  have factorPos : 0 < factor := by
    dsimp [factor]
    positivity
  have firstRootSq : firstRoot ^ 2 = firstRad := Real.sq_sqrt firstRadPos.le
  have secondRootSq : secondRoot ^ 2 = secondRad := Real.sq_sqrt secondRadPos.le
  have firstUpper : gridEnergy first ≤ count * firstRoot ^ 2 := by
    rw [firstRootSq]
    dsimp [firstRad]
    have cancel : count * (gridEnergy first / count) = gridEnergy first :=
      mul_div_cancel₀ _ countPositive.ne'
    linarith
  have secondUpper : gridEnergy second ≤ count * secondRoot ^ 2 := by
    rw [secondRootSq]
    dsimp [secondRad]
    have cancel : count * (gridEnergy second / count) = gridEnergy second :=
      mul_div_cancel₀ _ countPositive.ne'
    linarith
  have sumUpper : sumGrid ≤ 2 * count * (firstRoot + secondRoot) ^ 2 := by
    have step1 : sumGrid ≤ 2 * (gridEnergy first + gridEnergy second) :=
      sumEnergy first second
    have step2 : firstRoot ^ 2 + secondRoot ^ 2 ≤ (firstRoot + secondRoot) ^ 2 := by
      nlinarith [mul_pos firstRootPos secondRootPos]
    nlinarith
  have productUpper : sumGrid * gridEnergy second ≤ 2 * factor ^ 2 := by
    have bound := mul_le_mul sumUpper secondUpper secondNonneg (by positivity)
    calc
      sumGrid * gridEnergy second ≤
          (2 * count * (firstRoot + secondRoot) ^ 2) * (count * secondRoot ^ 2) := bound
      _ = 2 * factor ^ 2 := by
        dsimp [factor]
        ring
  have radiusBound := scaleShift positiveStreams positiveWidth first second
  have radiusNonnegative : 0 ≤ scaleRadius first second :=
    (abs_nonneg _).trans radiusBound
  have radiusSquare :
      (Mixing.stateScale second - Mixing.stateScale first) ^ 2 ≤
        scaleRadius first second ^ 2 := by
    have product : 0 ≤
        (scaleRadius first second -
          |Mixing.stateScale second - Mixing.stateScale first|) *
        (scaleRadius first second +
          |Mixing.stateScale second - Mixing.stateScale first|) := by
      apply mul_nonneg
      · linarith
      · positivity
    nlinarith [sq_abs (Mixing.stateScale second - Mixing.stateScale first)]
  have radiusExpand :
      scaleRadius first second ^ 2 =
        (diffEnergy * sumGrid) / (firstRoot * factor) ^ 2 := by
    unfold scaleRadius
    change (Real.sqrt (diffEnergy * sumGrid) /
        (count * firstRoot * secondRoot * (firstRoot + secondRoot))) ^ 2 = _
    rw [div_pow, Real.sq_sqrt (mul_nonneg diffNonneg sumNonneg)]
    congr 1
    dsimp [factor]
    ring
  have scaleExpand : Mixing.stateScale first ^ 2 = 1 / firstRoot ^ 2 := by
    unfold Mixing.stateScale
    change (1 / firstRoot) ^ 2 = 1 / firstRoot ^ 2
    ring
  calc
    (Mixing.stateScale second - Mixing.stateScale first) ^ 2 * gridEnergy second ≤
        scaleRadius first second ^ 2 * gridEnergy second :=
      mul_le_mul_of_nonneg_right radiusSquare secondNonneg
    _ = (diffEnergy * (sumGrid * gridEnergy second)) / (firstRoot ^ 2 * factor ^ 2) := by
      rw [radiusExpand]
      ring
    _ ≤ (diffEnergy * (2 * factor ^ 2)) / (firstRoot ^ 2 * factor ^ 2) := by
      apply div_le_div_of_nonneg_right
      · exact mul_le_mul_of_nonneg_left productUpper diffNonneg
      · positivity
    _ = 2 * (1 / firstRoot ^ 2) * diffEnergy := by
      have factorNe : factor ^ 2 ≠ 0 := by positivity
      have rootNe : firstRoot ^ 2 ≠ 0 := by positivity
      field_simp
    _ = 2 * Mixing.stateScale first ^ 2 *
        gridEnergy (fun stream column => second stream column - first stream column) := by
      rw [← scaleExpand, diffSymm]

/-- Tight incumbent-conditioned RMS-state sensitivity bound: the normalized
state movement depends only on the incumbent RMS scale and the incoming state
displacement, avoiding both the worst-case `1 / eps` explosion and the
candidate state energy. -/
theorem incumbentBound {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (first second : Rows streams width) :
    gridEnergy (fun stream column =>
      Mixing.normState second stream column - Mixing.normState first stream column) ≤
      6 * Mixing.stateScale first ^ 2 *
        gridEnergy (fun stream column => second stream column - first stream column) := by
  have step1 := incumbentEnergy first second
  have step2 := scaleTerm positiveStreams positiveWidth first second
  linarith

#print axioms scaleBound
#print axioms rootSplit
#print axioms scaleSplit
#print axioms energySplit
#print axioms energyShift
#print axioms scaleShift
#print axioms stateBound
#print axioms stateSplit
#print axioms energyBound
#print axioms sourceBound
#print axioms incumbentSplit
#print axioms incumbentEnergy
#print axioms sumEnergy
#print axioms scaleTerm
#print axioms incumbentBound

end
end BauxResearch.Forward.NormSensitivity
