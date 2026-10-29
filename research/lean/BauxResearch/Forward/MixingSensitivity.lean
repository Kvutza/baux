import BauxResearch.Forward.SeedSensitivity

/-!
# Source mHC predictor sensitivity

This file opens the residual predictor used by the Rust, Metal, and CUDA
forward programs.  The bound distinguishes direct parameter movement from
the normalized-state movement propagated by preceding layers.  No opaque
global Lipschitz constant is introduced.
-/

namespace BauxResearch.Forward.MixingSensitivity

noncomputable section
open scoped BigOperators

/-- A pointwise bound on a finite stream-by-width grid bounds its sum with the
exact number of scalar terms. -/
theorem gridBound {streams width : ℕ} (value : Fin streams → Fin width → ℝ)
    (bound : ℝ) (entry : ∀ stream column, |value stream column| ≤ bound) :
    |∑ stream, ∑ column, value stream column| ≤
      (streams * width : ℕ) * bound := by
  calc
    |∑ stream, ∑ column, value stream column| ≤
        ∑ stream, |∑ column, value stream column| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ stream, ∑ column, |value stream column| := by
      exact Finset.sum_le_sum fun stream _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _stream : Fin streams, ∑ _column : Fin width, bound := by
      exact Finset.sum_le_sum fun stream _ =>
        Finset.sum_le_sum fun column _ => entry stream column
    _ = (streams * width : ℕ) * bound := by
      simp [Nat.cast_mul]
      ring

/-- Squared Euclidean energy on the predictor's stream-by-width grid. -/
def gridEnergy {streams width : ℕ} (value : Fin streams → Fin width → ℝ) : ℝ :=
  ∑ stream, ∑ column, value stream column ^ 2

theorem energyNonneg {streams width : ℕ}
    (value : Fin streams → Fin width → ℝ) : 0 ≤ gridEnergy value := by
  unfold gridEnergy
  positivity

/-- Mathlib Cauchy-Schwarz on the product index underlying the source grid. -/
theorem gridCauchy {streams width : ℕ}
    (first second : Fin streams → Fin width → ℝ) :
    |∑ stream, ∑ column, first stream column * second stream column| ≤
      Real.sqrt (gridEnergy first * gridEnergy second) := by
  have cauchy := Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul
    (s := Finset.univ)
    (r := fun stream => ∑ column, first stream column * second stream column)
    (f := fun stream => ∑ column, first stream column ^ 2)
    (g := fun stream => ∑ column, second stream column ^ 2)
    (fun stream _ => Finset.sum_nonneg fun column _ => sq_nonneg (first stream column))
    (fun stream _ => Finset.sum_nonneg fun column _ => sq_nonneg (second stream column))
    (fun stream _ => Finset.sum_mul_sq_le_sq_mul_sq Finset.univ
      (first stream) (second stream))
  apply Real.le_sqrt_of_sq_le
  simpa only [sq_abs, gridEnergy] using cauchy

/-- The source's scale-outside-dot expression is the dot product with its
RMS-normalized state. -/
theorem dotRewrite {streams width : ℕ} (weights : Mixing.Weights streams width)
    (state : Rows streams width) (destination source : Fin streams) :
    Mixing.residualDot weights state destination source =
      ∑ stream, ∑ column, Mixing.normState state stream column *
        weights.residual stream column destination source := by
  simp [Mixing.residualDot, Mixing.normState, Finset.mul_sum, mul_assoc]

/-- Source RMS normalization bounds every normalized state entry using only
the number of entries.  It does not require an empirical activation cap. -/
theorem normCap {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (state : Rows streams width)
    (stream : Fin streams) (column : Fin width) :
    |Mixing.normState state stream column| ≤
      Real.sqrt (streams * width : ℝ) := by
  let total : ℝ := ∑ other, ∑ index, state other index ^ 2
  let count : ℝ := streams * width
  have countPositive : 0 < count := by
    dsimp [count]
    positivity
  have denominatorPositive : 0 < total / count + 1 / 100000 := by
    have totalNonnegative : 0 ≤ total := by
      dsimp [total]
      positivity
    positivity
  have columnBound : state stream column ^ 2 ≤
      ∑ index, state stream index ^ 2 :=
    Finset.single_le_sum (fun index _ => sq_nonneg (state stream index))
      (Finset.mem_univ column)
  have totalBound : state stream column ^ 2 ≤ total := by
    apply columnBound.trans
    dsimp [total]
    exact Finset.single_le_sum
      (fun other _ => Finset.sum_nonneg fun index _ => sq_nonneg (state other index))
      (Finset.mem_univ stream)
  apply Real.le_sqrt_of_sq_le
  rw [sq_abs, Mixing.normState, mul_pow, Mixing.stateScale, div_pow,
    Real.sq_sqrt denominatorPositive.le]
  have countNe : count ≠ 0 := countPositive.ne'
  rw [one_pow, div_mul_eq_mul_div, one_mul]
  change state stream column ^ 2 / (total / count + 1 / 100000) ≤ count
  apply (div_le_iff₀ denominatorPositive).2
  have denominatorForm :
      count * (total / count + 1 / 100000) = total + count / 100000 := by
    field_simp
  rw [denominatorForm]
  nlinarith

/-- The complete normalized-state energy is at most the number of source
predictor inputs. This is the sharper global consequence of RMS normalization. -/
theorem normEnergy {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (state : Rows streams width) :
    gridEnergy (Mixing.normState state) ≤ (streams * width : ℝ) := by
  let total : ℝ := ∑ stream, ∑ column, state stream column ^ 2
  let count : ℝ := streams * width
  have countPositive : 0 < count := by
    dsimp [count]
    positivity
  have totalNonnegative : 0 ≤ total := by
    dsimp [total]
    positivity
  have denominatorPositive : 0 < total / count + 1 / 100000 := by positivity
  have countNe : count ≠ 0 := countPositive.ne'
  have energyForm : gridEnergy (Mixing.normState state) =
      Mixing.stateScale state ^ 2 * total := by
    simp only [gridEnergy, Mixing.normState, mul_pow, ← Finset.mul_sum]
    rfl
  rw [energyForm, Mixing.stateScale, div_pow,
    Real.sq_sqrt denominatorPositive.le, one_pow, div_mul_eq_mul_div, one_mul]
  change total / (total / count + 1 / 100000) ≤ count
  apply (div_le_iff₀ denominatorPositive).2
  have denominatorForm :
      count * (total / count + 1 / 100000) = total + count / 100000 := by
    field_simp
  rw [denominatorForm]
  nlinarith

/-- A residual dot is bounded by normalized-state and predictor-weight
Euclidean energies, avoiding the entrywise dimension factor. -/
theorem energyDot {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width) (weights : Mixing.Weights streams width)
    (state : Rows streams width) (destination source : Fin streams) :
    |Mixing.residualDot weights state destination source| ≤
      Real.sqrt ((streams * width : ℝ) *
        gridEnergy (fun stream column =>
          weights.residual stream column destination source)) := by
  rw [dotRewrite]
  calc
    _ ≤ Real.sqrt (gridEnergy (Mixing.normState state) *
        gridEnergy (fun stream column =>
          weights.residual stream column destination source)) :=
      gridCauchy _ _
    _ ≤ Real.sqrt ((streams * width : ℝ) *
        gridEnergy (fun stream column =>
          weights.residual stream column destination source)) := by
      apply Real.sqrt_le_sqrt
      exact mul_le_mul_of_nonneg_right (normEnergy positiveStreams positiveWidth state)
        (energyNonneg _)

/-- Exact product-rule decomposition of residual-dot movement. -/
theorem dotSplit {streams width : ℕ}
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (destination source : Fin streams) :
    Mixing.residualDot secondWeights secondState destination source -
        Mixing.residualDot firstWeights firstState destination source =
      (∑ stream, ∑ column,
        (Mixing.normState secondState stream column -
            Mixing.normState firstState stream column) *
          secondWeights.residual stream column destination source) +
      (∑ stream, ∑ column,
        Mixing.normState firstState stream column *
          (secondWeights.residual stream column destination source -
            firstWeights.residual stream column destination source)) := by
  rw [dotRewrite, dotRewrite]
  calc
    (∑ stream, ∑ column,
        Mixing.normState secondState stream column *
          secondWeights.residual stream column destination source) -
        (∑ stream, ∑ column,
          Mixing.normState firstState stream column *
            firstWeights.residual stream column destination source) =
      ∑ stream, ∑ column,
        (Mixing.normState secondState stream column *
            secondWeights.residual stream column destination source -
          Mixing.normState firstState stream column *
            firstWeights.residual stream column destination source) := by
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro stream _
      rw [← Finset.sum_sub_distrib]
    _ = ∑ stream, ∑ column,
        ((Mixing.normState secondState stream column -
            Mixing.normState firstState stream column) *
          secondWeights.residual stream column destination source +
        Mixing.normState firstState stream column *
          (secondWeights.residual stream column destination source -
            firstWeights.residual stream column destination source)) := by
      apply Finset.sum_congr rfl
      intro stream _
      apply Finset.sum_congr rfl
      intro column _
      ring
    _ = _ := by
      simp_rw [Finset.sum_add_distrib]

/-- A normalized-state cap and predictor-weight cap bound one residual dot. -/
theorem dotSize {streams width : ℕ} (weights : Mixing.Weights streams width)
    (state : Rows streams width) (destination source : Fin streams)
    (stateCap weightCap : ℝ)
    (stateBound : ∀ stream column,
      |Mixing.normState state stream column| ≤ stateCap)
    (weightBound : ∀ stream column,
      |weights.residual stream column destination source| ≤ weightCap) :
    |Mixing.residualDot weights state destination source| ≤
      (streams * width : ℕ) * (stateCap * weightCap) := by
  rw [dotRewrite]
  apply gridBound
  intro stream column
  rw [abs_mul]
  exact mul_le_mul (stateBound stream column) (weightBound stream column)
    (abs_nonneg _) ((abs_nonneg _).trans (stateBound stream column))

/-- Movement of a residual dot is controlled separately by normalized-state
movement and predictor-weight movement. -/
theorem dotBound {streams width : ℕ}
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (destination source : Fin streams)
    (stateCap stateShift weightCap weightShift : ℝ)
    (stateBound : ∀ stream column,
      |Mixing.normState firstState stream column| ≤ stateCap)
    (stateMoved : ∀ stream column,
      |Mixing.normState secondState stream column -
        Mixing.normState firstState stream column| ≤ stateShift)
    (weightBound : ∀ stream column,
      |secondWeights.residual stream column destination source| ≤ weightCap)
    (weightMoved : ∀ stream column,
      |secondWeights.residual stream column destination source -
        firstWeights.residual stream column destination source| ≤ weightShift) :
    |Mixing.residualDot secondWeights secondState destination source -
        Mixing.residualDot firstWeights firstState destination source| ≤
      (streams * width : ℕ) *
        (stateShift * weightCap + stateCap * weightShift) := by
  rw [dotSplit]
  rw [← Finset.sum_add_distrib]
  simp_rw [← Finset.sum_add_distrib]
  apply gridBound
  intro stream column
  calc
    |_ * _ + _ * _| ≤ |_ * _| + |_ * _| := abs_add_le _ _
    _ =
        |Mixing.normState secondState stream column -
            Mixing.normState firstState stream column| *
            |secondWeights.residual stream column destination source| +
          |Mixing.normState firstState stream column| *
            |secondWeights.residual stream column destination source -
              firstWeights.residual stream column destination source| := by
      rw [abs_mul, abs_mul]
    _ ≤ stateShift * weightCap + stateCap * weightShift := by
      exact add_le_add
        (mul_le_mul (stateMoved stream column) (weightBound stream column)
          (abs_nonneg _) ((abs_nonneg _).trans (stateMoved stream column)))
        (mul_le_mul (stateBound stream column) (weightMoved stream column)
          (abs_nonneg _) ((abs_nonneg _).trans (stateBound stream column)))

/-- Euclidean movement bound for the exact product decomposition. The only
state term is the energy of normalized-state movement. -/
theorem energyMove {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (destination source : Fin streams) :
    |Mixing.residualDot secondWeights secondState destination source -
        Mixing.residualDot firstWeights firstState destination source| ≤
      Real.sqrt (
        gridEnergy (fun stream column =>
          Mixing.normState secondState stream column -
            Mixing.normState firstState stream column) *
        gridEnergy (fun stream column =>
          secondWeights.residual stream column destination source)) +
      Real.sqrt ((streams * width : ℝ) *
        gridEnergy (fun stream column =>
          secondWeights.residual stream column destination source -
            firstWeights.residual stream column destination source)) := by
  rw [dotSplit]
  calc
    |_ + _| ≤ |_| + |_| := abs_add_le _ _
    _ ≤ Real.sqrt (
          gridEnergy (fun stream column =>
            Mixing.normState secondState stream column -
              Mixing.normState firstState stream column) *
          gridEnergy (fun stream column =>
            secondWeights.residual stream column destination source)) +
        Real.sqrt (gridEnergy (Mixing.normState firstState) *
          gridEnergy (fun stream column =>
            secondWeights.residual stream column destination source -
              firstWeights.residual stream column destination source)) :=
      add_le_add (gridCauchy _ _) (gridCauchy _ _)
    _ ≤ _ := by
      apply add_le_add le_rfl
      apply Real.sqrt_le_sqrt
      exact mul_le_mul_of_nonneg_right
        (normEnergy positiveStreams positiveWidth firstState) (energyNonneg _)

/-- Exact decomposition of one source residual logit. -/
theorem rawSplit {streams width : ℕ}
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (destination source : Fin streams) :
    Mixing.residualRaw secondWeights secondState destination source -
        Mixing.residualRaw firstWeights firstState destination source =
      (secondWeights.residualBias destination source -
        firstWeights.residualBias destination source) +
      (secondWeights.control 1 - firstWeights.control 1) *
        Mixing.residualDot secondWeights secondState destination source +
      firstWeights.control 1 *
        (Mixing.residualDot secondWeights secondState destination source -
          Mixing.residualDot firstWeights firstState destination source) := by
  simp only [Mixing.residualRaw]
  ring

/-- Bias, control, and residual-dot bounds give a raw-logit bound. -/
theorem rawBound {streams width : ℕ}
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (destination source : Fin streams)
    (biasShift controlShift controlCap dotCap dotShift : ℝ)
    (biasMoved : |secondWeights.residualBias destination source -
      firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (dotBounded : |Mixing.residualDot secondWeights secondState destination source| ≤ dotCap)
    (dotMoved : |Mixing.residualDot secondWeights secondState destination source -
      Mixing.residualDot firstWeights firstState destination source| ≤ dotShift) :
    |Mixing.residualRaw secondWeights secondState destination source -
        Mixing.residualRaw firstWeights firstState destination source| ≤
      biasShift + controlShift * dotCap + controlCap * dotShift := by
  rw [rawSplit]
  have triangle :
      |(secondWeights.residualBias destination source -
          firstWeights.residualBias destination source) +
        (secondWeights.control 1 - firstWeights.control 1) *
          Mixing.residualDot secondWeights secondState destination source +
        firstWeights.control 1 *
          (Mixing.residualDot secondWeights secondState destination source -
            Mixing.residualDot firstWeights firstState destination source)| ≤
      |secondWeights.residualBias destination source -
          firstWeights.residualBias destination source| +
        |secondWeights.control 1 - firstWeights.control 1| *
          |Mixing.residualDot secondWeights secondState destination source| +
        |firstWeights.control 1| *
          |Mixing.residualDot secondWeights secondState destination source -
            Mixing.residualDot firstWeights firstState destination source| := by
    let a := secondWeights.residualBias destination source -
      firstWeights.residualBias destination source
    let b := (secondWeights.control 1 - firstWeights.control 1) *
      Mixing.residualDot secondWeights secondState destination source
    let c := firstWeights.control 1 *
      (Mixing.residualDot secondWeights secondState destination source -
        Mixing.residualDot firstWeights firstState destination source)
    have tri : |a + b + c| ≤ |a| + |b| + |c| := by
      calc
        |a + b + c| ≤ |a + b| + |c| := abs_add_le _ _
        _ ≤ (|a| + |b|) + |c| := add_le_add (abs_add_le _ _) le_rfl
    simpa [a, b, c, abs_mul] using tri
  calc
    _ ≤ _ := triangle
    _ ≤ biasShift + controlShift * dotCap + controlCap * dotShift := by
      exact add_le_add
        (add_le_add biasMoved
          (mul_le_mul controlMoved dotBounded (abs_nonneg _)
            ((abs_nonneg _).trans controlMoved)))
        (mul_le_mul controlBound dotMoved (abs_nonneg _)
          ((abs_nonneg _).trans controlBound))

/-- Source residual-logit sensitivity in Euclidean energy geometry. This
avoids the extra dimension factor of an entrywise predictor-weight cap. -/
theorem energyRaw {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap) :
    ∀ destination source,
      |Mixing.residualRaw secondWeights secondState destination source -
          Mixing.residualRaw firstWeights firstState destination source| ≤
        biasShift + controlShift * Real.sqrt ((streams * width : ℝ) *
          gridEnergy (fun stream column =>
            secondWeights.residual stream column destination source)) +
        controlCap *
          (Real.sqrt (
            gridEnergy (fun stream column =>
              Mixing.normState secondState stream column -
                Mixing.normState firstState stream column) *
            gridEnergy (fun stream column =>
              secondWeights.residual stream column destination source)) +
          Real.sqrt ((streams * width : ℝ) *
            gridEnergy (fun stream column =>
              secondWeights.residual stream column destination source -
                firstWeights.residual stream column destination source))) := by
  intro destination source
  apply rawBound firstWeights secondWeights firstState secondState destination source
    biasShift controlShift controlCap
    (Real.sqrt ((streams * width : ℝ) *
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source)))
    (Real.sqrt (
      gridEnergy (fun stream column =>
        Mixing.normState secondState stream column -
          Mixing.normState firstState stream column) *
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source)) +
      Real.sqrt ((streams * width : ℝ) *
        gridEnergy (fun stream column =>
          secondWeights.residual stream column destination source -
            firstWeights.residual stream column destination source)))
  · exact biasMoved destination source
  · exact controlMoved
  · exact controlBound
  · exact energyDot positiveStreams positiveWidth secondWeights secondState destination source
  · exact energyMove positiveStreams positiveWidth firstWeights secondWeights
      firstState secondState destination source

/-- Uniform Euclidean-energy caps give one raw-logit radius for the complete
residual matrix. -/
theorem uniformRaw {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap stateShift weightCap weightShift : ℝ)
    (stateNonnegative : 0 ≤ stateShift)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (stateMoved : gridEnergy (fun stream column =>
      Mixing.normState secondState stream column -
        Mixing.normState firstState stream column) ≤ stateShift)
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
        controlCap * (Real.sqrt (stateShift * weightCap) +
          Real.sqrt ((streams * width : ℝ) * weightShift)) := by
  intro destination source
  apply rawBound firstWeights secondWeights firstState secondState destination source
    biasShift controlShift controlCap
    (Real.sqrt ((streams * width : ℝ) * weightCap))
    (Real.sqrt (stateShift * weightCap) +
      Real.sqrt ((streams * width : ℝ) * weightShift))
  · exact biasMoved destination source
  · exact controlMoved
  · exact controlBound
  · exact (energyDot positiveStreams positiveWidth secondWeights secondState
      destination source).trans (Real.sqrt_le_sqrt
        (mul_le_mul_of_nonneg_left (weightBound destination source) (by positivity)))
  · apply (energyMove positiveStreams positiveWidth firstWeights secondWeights
      firstState secondState destination source).trans
    apply add_le_add <;> apply Real.sqrt_le_sqrt
    · exact mul_le_mul stateMoved (weightBound destination source)
        (energyNonneg _) stateNonnegative
    · exact mul_le_mul_of_nonneg_left (weightMoved destination source) (by positivity)

/-- The Euclidean source certificate reaches the logarithmic seed geometry
with a single uniform matrix radius. -/
theorem uniformLog {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap stateShift weightCap weightShift : ℝ)
    (stateNonnegative : 0 ≤ stateShift)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (stateMoved : gridEnergy (fun stream column =>
      Mixing.normState secondState stream column -
        Mixing.normState firstState stream column) ≤ stateShift)
    (weightBound : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source) ≤ weightCap)
    (weightMoved : ∀ destination source,
      gridEnergy (fun stream column =>
        secondWeights.residual stream column destination source -
          firstWeights.residual stream column destination source) ≤ weightShift)
    (row column : Fin streams) :
    |Real.log (Transport.seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw secondWeights secondState))
        (Mixing.residualRaw secondWeights secondState) row column) -
      Real.log (Transport.seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw firstWeights firstState))
        (Mixing.residualRaw firstWeights firstState) row column)| ≤
      2 * (biasShift + controlShift *
        Real.sqrt ((streams * width : ℝ) * weightCap) +
        controlCap * (Real.sqrt (stateShift * weightCap) +
          Real.sqrt ((streams * width : ℝ) * weightShift))) := by
  apply SeedSensitivity.logBound positiveStreams
    (Mixing.residualRaw firstWeights firstState)
    (Mixing.residualRaw secondWeights secondState)
  exact uniformRaw positiveStreams positiveWidth firstWeights secondWeights firstState secondState
    biasShift controlShift controlCap stateShift weightCap weightShift
    stateNonnegative biasMoved controlMoved controlBound
    stateMoved weightBound weightMoved

/-- The source-level residual-logit certificate.  Every term is either a
direct parameter cap or normalized-state movement inherited from preceding
layers. -/
theorem scoreBound {streams width : ℕ}
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap stateCap stateShift weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (stateBound : ∀ stream column,
      |Mixing.normState firstState stream column| ≤ stateCap)
    (stateMoved : ∀ stream column,
      |Mixing.normState secondState stream column -
        Mixing.normState firstState stream column| ≤ stateShift)
    (weightBound : ∀ stream column destination source,
      |secondWeights.residual stream column destination source| ≤ weightCap)
    (weightMoved : ∀ stream column destination source,
      |secondWeights.residual stream column destination source -
        firstWeights.residual stream column destination source| ≤ weightShift) :
    ∀ destination source,
      |Mixing.residualRaw secondWeights secondState destination source -
          Mixing.residualRaw firstWeights firstState destination source| ≤
        biasShift +
          controlShift * ((streams * width : ℕ) * ((stateCap + stateShift) * weightCap)) +
          controlCap * ((streams * width : ℕ) *
            (stateShift * weightCap + stateCap * weightShift)) := by
  intro destination source
  have secondBound : ∀ stream column,
      |Mixing.normState secondState stream column| ≤ stateCap + stateShift := by
    intro stream column
    calc
      |Mixing.normState secondState stream column| =
          |(Mixing.normState secondState stream column -
              Mixing.normState firstState stream column) +
            Mixing.normState firstState stream column| := by ring_nf
      _ ≤ |Mixing.normState secondState stream column -
          Mixing.normState firstState stream column| +
          |Mixing.normState firstState stream column| := abs_add_le _ _
      _ ≤ stateShift + stateCap := add_le_add
        (stateMoved stream column) (stateBound stream column)
      _ = stateCap + stateShift := add_comm _ _
  apply rawBound firstWeights secondWeights firstState secondState destination source
    biasShift controlShift controlCap
    ((streams * width : ℕ) * ((stateCap + stateShift) * weightCap))
    ((streams * width : ℕ) * (stateShift * weightCap + stateCap * weightShift))
  · exact biasMoved destination source
  · exact controlMoved
  · exact controlBound
  · exact dotSize secondWeights secondState destination source
      (stateCap + stateShift) weightCap secondBound
      (fun stream column => weightBound stream column destination source)
  · exact dotBound firstWeights secondWeights firstState secondState destination source
      stateCap stateShift weightCap weightShift stateBound stateMoved
      (fun stream column => weightBound stream column destination source)
      (fun stream column => weightMoved stream column destination source)

/-- Source RMS normalization discharges the activation-size premise, leaving
only normalized-state movement as the recursive forward obligation. -/
theorem sourceBound {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap stateShift weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (stateMoved : ∀ stream column,
      |Mixing.normState secondState stream column -
        Mixing.normState firstState stream column| ≤ stateShift)
    (weightBound : ∀ stream column destination source,
      |secondWeights.residual stream column destination source| ≤ weightCap)
    (weightMoved : ∀ stream column destination source,
      |secondWeights.residual stream column destination source -
        firstWeights.residual stream column destination source| ≤ weightShift) :
    ∀ destination source,
      |Mixing.residualRaw secondWeights secondState destination source -
          Mixing.residualRaw firstWeights firstState destination source| ≤
        biasShift + controlShift *
          ((streams * width : ℕ) * (Real.sqrt (streams * width : ℝ) * weightCap)) +
        controlCap * ((streams * width : ℕ) *
          (stateShift * weightCap + Real.sqrt (streams * width : ℝ) * weightShift)) := by
  intro destination source
  apply rawBound firstWeights secondWeights firstState secondState destination source
    biasShift controlShift controlCap
    ((streams * width : ℕ) * (Real.sqrt (streams * width : ℝ) * weightCap))
    ((streams * width : ℕ) *
      (stateShift * weightCap + Real.sqrt (streams * width : ℝ) * weightShift))
  · exact biasMoved destination source
  · exact controlMoved
  · exact controlBound
  · exact dotSize secondWeights secondState destination source
      (Real.sqrt (streams * width : ℝ)) weightCap
      (normCap positiveStreams positiveWidth secondState)
      (fun stream column => weightBound stream column destination source)
  · exact dotBound firstWeights secondWeights firstState secondState destination source
      (Real.sqrt (streams * width : ℝ)) stateShift weightCap weightShift
      (normCap positiveStreams positiveWidth firstState) stateMoved
      (fun stream column => weightBound stream column destination source)
      (fun stream column => weightMoved stream column destination source)

/-- Direct parameter caps plus propagated normalized-state movement control
the logarithmic seed geometry consumed by Sinkhorn. -/
theorem seedLog {streams width : ℕ} (positiveStreams : 0 < streams)
    (positiveWidth : 0 < width)
    (firstWeights secondWeights : Mixing.Weights streams width)
    (firstState secondState : Rows streams width)
    (biasShift controlShift controlCap stateShift weightCap weightShift : ℝ)
    (biasMoved : ∀ destination source,
      |secondWeights.residualBias destination source -
        firstWeights.residualBias destination source| ≤ biasShift)
    (controlMoved : |secondWeights.control 1 - firstWeights.control 1| ≤ controlShift)
    (controlBound : |firstWeights.control 1| ≤ controlCap)
    (stateMoved : ∀ stream column,
      |Mixing.normState secondState stream column -
        Mixing.normState firstState stream column| ≤ stateShift)
    (weightBound : ∀ stream column destination source,
      |secondWeights.residual stream column destination source| ≤ weightCap)
    (weightMoved : ∀ stream column destination source,
      |secondWeights.residual stream column destination source -
        firstWeights.residual stream column destination source| ≤ weightShift)
    (row column : Fin streams) :
    |Real.log (Transport.seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw secondWeights secondState))
        (Mixing.residualRaw secondWeights secondState) row column) -
      Real.log (Transport.seed
        (Mixing.rawMax positiveStreams (Mixing.residualRaw firstWeights firstState))
        (Mixing.residualRaw firstWeights firstState) row column)| ≤
      2 * (biasShift + controlShift *
        ((streams * width : ℕ) * (Real.sqrt (streams * width : ℝ) * weightCap)) +
        controlCap * ((streams * width : ℕ) *
          (stateShift * weightCap + Real.sqrt (streams * width : ℝ) * weightShift))) := by
  apply SeedSensitivity.logBound positiveStreams
    (Mixing.residualRaw firstWeights firstState)
    (Mixing.residualRaw secondWeights secondState)
  exact sourceBound positiveStreams positiveWidth firstWeights secondWeights
    firstState secondState biasShift controlShift controlCap stateShift weightCap weightShift
    biasMoved controlMoved controlBound stateMoved weightBound weightMoved

#print axioms gridBound
#print axioms energyNonneg
#print axioms gridCauchy
#print axioms dotRewrite
#print axioms normCap
#print axioms normEnergy
#print axioms energyDot
#print axioms dotSplit
#print axioms dotSize
#print axioms dotBound
#print axioms energyMove
#print axioms rawSplit
#print axioms rawBound
#print axioms energyRaw
#print axioms uniformRaw
#print axioms uniformLog
#print axioms scoreBound
#print axioms sourceBound
#print axioms seedLog

end

end BauxResearch.Forward.MixingSensitivity
