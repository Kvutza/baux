import BauxResearch.Forward.Algebra
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
Write `Nᵣ` and `N꜀` for the floor-clamped row and column normalizations below.
Then `sinkhorn` applies `(N꜀ ∘ Nᵣ)` the declared number of times; order matters.
The residual is the sum of absolute row- and column-mass errors, not a proof
that this finite iteration has reached a doubly stochastic matrix.
-/

namespace BauxResearch.Forward.Transport
noncomputable section
open scoped BigOperators
open Matrix

abbrev Square (streams : ℕ) := Matrix (Fin streams) (Fin streams) ℝ

def rowSum {streams : ℕ} (matrix : Square streams) (row : Fin streams) : ℝ :=
  ∑ column, matrix row column
def columnSum {streams : ℕ} (matrix : Square streams) (column : Fin streams) : ℝ :=
  ∑ row, matrix row column

def rowNormalize {streams : ℕ} (floor : ℝ) (matrix : Square streams) : Square streams :=
  fun row column => matrix row column / max floor (rowSum matrix row)
def columnNormalize {streams : ℕ} (floor : ℝ) (matrix : Square streams) : Square streams :=
  fun row column => matrix row column / max floor (columnSum matrix column)

def step {streams : ℕ} (floor : ℝ) (matrix : Square streams) : Square streams :=
  columnNormalize floor (rowNormalize floor matrix)

/-- GPU row-then-column order; the Rust test reference uses a different order. -/
def sinkhorn {streams : ℕ} (iterations : ℕ) (floor : ℝ) (matrix : Square streams) :
    Square streams :=
  (step floor)^[iterations] matrix

def seed {streams : ℕ} (maximum : ℝ) (raw : Square streams) : Square streams :=
  fun row column => Real.exp (max (raw row column - maximum) (-80))

def blend {streams : ℕ} (lambda : ℝ) (matrix : Square streams) : Square streams :=
  (1 - lambda) • (1 : Square streams) + lambda • matrix

def residual {streams : ℕ} (matrix : Square streams) : ℝ :=
  (∑ row, |rowSum matrix row - 1|) + ∑ column, |columnSum matrix column - 1|

/-- A correctly supplied finite maximum puts every clipped exponential seed in
the closed interval `[exp (-80), 1]`. -/
theorem seedRange {streams : ℕ} (maximum : ℝ) (raw : Square streams)
    (upper : ∀ row column, raw row column ≤ maximum) (row column : Fin streams) :
    Real.exp (-80) ≤ seed maximum raw row column ∧
      seed maximum raw row column ≤ 1 := by
  unfold seed
  constructor
  · exact Real.exp_le_exp.mpr (le_max_right _ _)
  · rw [← Real.exp_zero]
    exact Real.exp_le_exp.mpr (max_le (sub_nonpos.mpr (upper row column)) (by norm_num))

/-- Any entry attaining the supplied raw maximum becomes exactly one after the
source seed transformation. -/
theorem seedWitness {streams : ℕ} (maximum : ℝ) (raw : Square streams)
    (row column : Fin streams) (witness : raw row column = maximum) :
    seed maximum raw row column = 1 := by
  simp [seed, witness]

/-- Every seed row has a strictly positive sum. This is true independently of
whether the source normalization floor is active. -/
theorem seedRow {streams : ℕ} (positive : 0 < streams)
    (maximum : ℝ) (raw : Square streams) (row : Fin streams) :
    0 < rowSum (seed maximum raw) row := by
  let column : Fin streams := ⟨0, positive⟩
  exact lt_of_lt_of_le (Real.exp_pos _)
    (Finset.single_le_sum (fun other _ => (Real.exp_pos _).le)
      (Finset.mem_univ column))

theorem Row.normalized {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (row : Fin streams) (positive : 0 < rowSum matrix row)
    (inactive : floor ≤ rowSum matrix row) :
    rowSum (rowNormalize floor matrix) row = 1 := by
  simp only [rowSum, rowNormalize]
  rw [← Finset.sum_div]
  change rowSum matrix row / max floor (rowSum matrix row) = 1
  rw [max_eq_right inactive]
  exact div_self positive.ne'

/-- Exact total row mass after floored normalization. When the original mass
is below the floor the result is its floor-relative mass; otherwise it is one. -/
theorem Row.mass {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (row : Fin streams) (floorPositive : 0 < floor)
    (massPositive : 0 < rowSum matrix row) :
    rowSum (rowNormalize floor matrix) row =
      min 1 (rowSum matrix row / floor) := by
  change (∑ column, matrix row column / max floor (rowSum matrix row)) = _
  rw [← Finset.sum_div]
  change rowSum matrix row / max floor (rowSum matrix row) = _
  by_cases below : rowSum matrix row ≤ floor
  · rw [max_eq_left below]
    exact (min_eq_right ((div_le_one floorPositive).2 below)).symm
  · have above : floor ≤ rowSum matrix row := le_of_not_ge below
    rw [max_eq_right above, div_self massPositive.ne']
    have scaled : 1 * floor ≤ rowSum matrix row := by simpa using above
    exact (min_eq_left ((le_div_iff₀ floorPositive).2 scaled)).symm

/-- Column analogue of `Row.mass`. -/
theorem Column.mass {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (column : Fin streams) (floorPositive : 0 < floor)
    (massPositive : 0 < columnSum matrix column) :
    columnSum (columnNormalize floor matrix) column =
      min 1 (columnSum matrix column / floor) := by
  change (∑ row, matrix row column / max floor (columnSum matrix column)) = _
  rw [← Finset.sum_div]
  change columnSum matrix column / max floor (columnSum matrix column) = _
  by_cases below : columnSum matrix column ≤ floor
  · rw [max_eq_left below]
    exact (min_eq_right ((div_le_one floorPositive).2 below)).symm
  · have above : floor ≤ columnSum matrix column := le_of_not_ge below
    rw [max_eq_right above, div_self massPositive.ne']
    have scaled : 1 * floor ≤ columnSum matrix column := by simpa using above
    exact (min_eq_left ((le_div_iff₀ floorPositive).2 scaled)).symm

theorem Row.nonnegative {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (floorNonnegative : 0 ≤ floor)
    (entries : ∀ row column, 0 ≤ matrix row column) (row column : Fin streams) :
    0 ≤ rowNormalize floor matrix row column := by
  exact div_nonneg (entries row column)
    (le_max_of_le_left floorNonnegative)

theorem Row.capped {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (floorPositive : 0 < floor)
    (entries : ∀ row column, 0 ≤ matrix row column) (row column : Fin streams) :
    rowNormalize floor matrix row column ≤ 1 := by
  have entryMass : matrix row column ≤ rowSum matrix row :=
    Finset.single_le_sum (fun other _ => entries row other) (Finset.mem_univ column)
  have denominatorPositive : 0 < max floor (rowSum matrix row) :=
    floorPositive.trans_le (le_max_left _ _)
  apply (div_le_one denominatorPositive).2
  exact entryMass.trans (le_max_right _ _)

theorem Column.nonnegative {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (floorNonnegative : 0 ≤ floor)
    (entries : ∀ row column, 0 ≤ matrix row column) (row column : Fin streams) :
    0 ≤ columnNormalize floor matrix row column := by
  exact div_nonneg (entries row column)
    (le_max_of_le_left floorNonnegative)

theorem Column.capped {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (floorPositive : 0 < floor)
    (entries : ∀ row column, 0 ≤ matrix row column) (row column : Fin streams) :
    columnNormalize floor matrix row column ≤ 1 := by
  have entryMass : matrix row column ≤ columnSum matrix column :=
    Finset.single_le_sum (fun other _ => entries other column) (Finset.mem_univ row)
  have denominatorPositive : 0 < max floor (columnSum matrix column) :=
    floorPositive.trans_le (le_max_left _ _)
  apply (div_le_one denominatorPositive).2
  exact entryMass.trans (le_max_right _ _)

theorem Step.nonnegative {streams : ℕ} (floor : ℝ) (matrix : Square streams)
    (floorNonnegative : 0 ≤ floor)
    (entries : ∀ row column, 0 ≤ matrix row column) :
    ∀ row column, 0 ≤ step floor matrix row column := by
  intro row column
  apply Column.nonnegative floor (rowNormalize floor matrix) floorNonnegative
  exact Row.nonnegative floor matrix floorNonnegative entries

theorem sinkhornNonnegative {streams : ℕ} (iterations : ℕ) (floor : ℝ)
    (matrix : Square streams) (floorNonnegative : 0 ≤ floor)
    (entries : ∀ row column, 0 ≤ matrix row column) :
    ∀ row column, 0 ≤ sinkhorn iterations floor matrix row column := by
  induction iterations with
  | zero => simpa [sinkhorn] using entries
  | succ iterations hypothesis =>
      simpa [sinkhorn, Function.iterate_succ_apply'] using
        Step.nonnegative floor (sinkhorn iterations floor matrix)
          floorNonnegative hypothesis

/-- For the source's four-stream geometry, a positive entry floor stronger
than the square of the normalization floor makes every row floor-inactive after
one complete row-then-column normalization. This avoids iterating a generic
`1 / floor` sensitivity factor through all later Sinkhorn steps. -/
theorem fourActive (floor lower : ℝ) (matrix : Square 4)
    (floorPositive : 0 < floor)
    (floorFour : floor ≤ 4)
    (lowerPositive : 0 < lower)
    (lowerSmall : 4 * lower ≤ floor)
    (squareLower : floor ^ 2 < lower)
    (entries : ∀ row column, lower ≤ matrix row column) :
    ∀ row, floor < rowSum (columnNormalize floor (rowNormalize floor matrix)) row := by
  let rows := rowNormalize floor matrix
  let result := columnNormalize floor rows
  have matrixNonnegative : ∀ row column, 0 ≤ matrix row column := fun row column =>
    lowerPositive.le.trans (entries row column)
  have rowsNonnegative : ∀ row column, 0 ≤ rows row column := fun row column =>
    Row.nonnegative floor matrix floorPositive.le matrixNonnegative row column
  have rowsAtMostOne : ∀ row column, rows row column ≤ 1 := fun row column =>
    Row.capped floor matrix floorPositive matrixNonnegative row column
  intro row
  have originalPositive : 0 < rowSum matrix row := by
    let column : Fin 4 := ⟨0, by decide⟩
    exact lowerPositive.trans_le <|
      (entries row column).trans <|
        Finset.single_le_sum (fun other _ => matrixNonnegative row other)
          (Finset.mem_univ column)
  have originalLower : 4 * lower ≤ rowSum matrix row := by
    calc
      4 * lower = ∑ _ : Fin 4, lower := by simp
      _ ≤ ∑ column, matrix row column :=
        Finset.sum_le_sum fun column _ => entries row column
      _ = rowSum matrix row := rfl
  have ratioOne : 4 * lower / floor ≤ 1 :=
    (div_le_one floorPositive).2 lowerSmall
  have ratioMass : 4 * lower / floor ≤ rowSum matrix row / floor :=
    div_le_div_of_nonneg_right originalLower floorPositive.le
  have rowsLower : 4 * lower / floor ≤ rowSum rows row := by
    rw [show rowSum rows row = min 1 (rowSum matrix row / floor) by
      exact Row.mass floor matrix row floorPositive originalPositive]
    exact le_min ratioOne ratioMass
  have denominatorBound : ∀ column,
      max floor (columnSum rows column) ≤ 4 := by
    intro column
    apply max_le floorFour
    calc
      columnSum rows column = ∑ other, rows other column := rfl
      _ ≤ ∑ _ : Fin 4, (1 : ℝ) :=
        Finset.sum_le_sum fun other _ => rowsAtMostOne other column
      _ = 4 := by simp
  have quarterBound : rowSum rows row / 4 ≤ rowSum result row := by
    calc
      rowSum rows row / 4 = ∑ column, rows row column / 4 := by
        change (∑ column, rows row column) / 4 = _
        rw [Finset.sum_div]
      _ ≤ ∑ column, rows row column / max floor (columnSum rows column) := by
        apply Finset.sum_le_sum
        intro column _
        exact div_le_div_of_nonneg_left (rowsNonnegative row column)
          (floorPositive.trans_le (le_max_left _ _)) (denominatorBound column)
      _ = rowSum result row := rfl
  have floorRatio : floor < lower / floor := by
    apply (lt_div_iff₀ floorPositive).2
    simpa [pow_two] using squareLower
  calc
    floor < lower / floor := floorRatio
    _ = (4 * lower / floor) / 4 := by ring
    _ ≤ rowSum rows row / 4 :=
      div_le_div_of_nonneg_right rowsLower (by norm_num)
    _ ≤ rowSum result row := quarterBound

/-- Once every source row is above the normalization floor, a complete
four-stream row-then-column pass leaves at least one quarter of unit mass in
every row. Thus the row-floor inactive region is invariant whenever the floor
is below one quarter. -/
theorem fourStable (floor : ℝ) (matrix : Square 4)
    (floorPositive : 0 < floor)
    (floorQuarter : floor < 1 / 4)
    (entries : ∀ row column, 0 ≤ matrix row column)
    (active : ∀ row, floor < rowSum matrix row) :
    ∀ row, 1 / 4 ≤ rowSum (columnNormalize floor (rowNormalize floor matrix)) row := by
  let rows := rowNormalize floor matrix
  let result := columnNormalize floor rows
  have rowsNonnegative : ∀ row column, 0 ≤ rows row column := fun row column =>
    Row.nonnegative floor matrix floorPositive.le entries row column
  have rowsAtMostOne : ∀ row column, rows row column ≤ 1 := fun row column =>
    Row.capped floor matrix floorPositive entries row column
  have rowsUnit : ∀ row, rowSum rows row = 1 := fun row =>
    Row.normalized floor matrix row (floorPositive.trans (active row)) (active row).le
  have denominatorBound : ∀ column,
      max floor (columnSum rows column) ≤ 4 := by
    intro column
    apply max_le (floorQuarter.le.trans (by norm_num))
    calc
      columnSum rows column = ∑ other, rows other column := rfl
      _ ≤ ∑ _ : Fin 4, (1 : ℝ) :=
        Finset.sum_le_sum fun other _ => rowsAtMostOne other column
      _ = 4 := by simp
  intro row
  calc
    1 / 4 = rowSum rows row / 4 := by rw [rowsUnit row]
    _ = ∑ column, rows row column / 4 := by
      change (∑ column, rows row column) / 4 = _
      rw [Finset.sum_div]
    _ ≤ ∑ column, rows row column / max floor (columnSum rows column) := by
      apply Finset.sum_le_sum
      intro column _
      exact div_le_div_of_nonneg_left (rowsNonnegative row column)
        (floorPositive.trans_le (le_max_left _ _)) (denominatorBound column)
    _ = rowSum result row := rfl

/-- A rational lower bound for the source's `exp (-80)` seed floor. The proof
uses Mathlib's certified upper bound on `exp 1`, not floating-point evaluation. -/
theorem sourceLower : 1 / (10 : ℝ) ^ 35 < Real.exp (-80) := by
  have upper : Real.exp 80 < (10 : ℝ) ^ 35 := by
    rw [show (80 : ℝ) = (80 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    exact (pow_lt_pow_left₀ Real.exp_one_lt_d9 (Real.exp_pos 1).le (by decide)).trans
      (by norm_num)
  rw [Real.exp_neg]
  simpa [one_div] using one_div_lt_one_div_of_lt (Real.exp_pos 80) upper

/-- In the actual four-stream seed geometry, the first complete Sinkhorn pass
already moves every row above the `1e-20` floor. -/
theorem sourceActive (maximum : ℝ) (raw : Square 4)
    (upper : ∀ row column, raw row column ≤ maximum) :
    ∀ row, (1 / (10 : ℝ) ^ 20) <
      rowSum (columnNormalize (1 / (10 : ℝ) ^ 20)
        (rowNormalize (1 / (10 : ℝ) ^ 20) (seed maximum raw))) row := by
  apply fourActive (1 / (10 : ℝ) ^ 20) (1 / (10 : ℝ) ^ 35) (seed maximum raw)
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · norm_num
  · intro row column
    exact sourceLower.le.trans (seedRange maximum raw upper row column).1

/-- Every positive iteration count in the source's twenty-step construction
stays above the row floor. The first pass uses the seed lower bound; subsequent
passes use the one-quarter invariant. -/
theorem iterActive (maximum : ℝ) (raw : Square 4)
    (upper : ∀ row column, raw row column ≤ maximum) :
    ∀ iterations row, (1 / (10 : ℝ) ^ 20) <
      rowSum (sinkhorn (iterations + 1) (1 / (10 : ℝ) ^ 20) (seed maximum raw)) row := by
  let floor : ℝ := 1 / (10 : ℝ) ^ 20
  let initial := seed maximum raw
  have floorPositive : 0 < floor := by dsimp [floor]; norm_num
  have floorQuarter : floor < 1 / 4 := by dsimp [floor]; norm_num
  have initialNonnegative : ∀ row column, 0 ≤ initial row column := fun row column =>
    (Real.exp_pos _).le
  have invariant : ∀ iterations,
      (∀ row column, 0 ≤ sinkhorn (iterations + 1) floor initial row column) ∧
      (∀ row, floor < rowSum (sinkhorn (iterations + 1) floor initial) row) := by
    intro iterations
    induction iterations with
    | zero =>
        constructor
        · simpa [sinkhorn, step, initial] using
            Step.nonnegative floor initial floorPositive.le initialNonnegative
        · simpa [floor, initial, sinkhorn, step] using sourceActive maximum raw upper
    | succ iterations hypothesis =>
        constructor
        · simpa [sinkhorn, Function.iterate_succ_apply'] using
            Step.nonnegative floor (sinkhorn (iterations + 1) floor initial)
              floorPositive.le hypothesis.1
        · have stable := fourStable floor (sinkhorn (iterations + 1) floor initial)
            floorPositive floorQuarter hypothesis.1 hypothesis.2
          intro row
          have next : 1 / 4 ≤
              rowSum (step floor (sinkhorn (iterations + 1) floor initial)) row :=
            stable row
          have active : floor <
              rowSum (step floor (sinkhorn (iterations + 1) floor initial)) row :=
            floorQuarter.trans_le next
          simpa [sinkhorn, Function.iterate_succ_apply'] using active
  intro iterations row
  exact (invariant iterations).2 row

/-- From the second pass onward, every source row has at least one quarter of
unit mass. In particular this applies to the configured twenty-pass result. -/
theorem sourceQuarter (maximum : ℝ) (raw : Square 4)
    (upper : ∀ row column, raw row column ≤ maximum) :
    ∀ iterations row, 1 / 4 ≤
      rowSum (sinkhorn (iterations + 2) (1 / (10 : ℝ) ^ 20) (seed maximum raw)) row := by
  let floor : ℝ := 1 / (10 : ℝ) ^ 20
  let initial := seed maximum raw
  have floorPositive : 0 < floor := by dsimp [floor]; norm_num
  have floorQuarter : floor < 1 / 4 := by dsimp [floor]; norm_num
  have initialNonnegative : ∀ row column, 0 ≤ initial row column := fun row column =>
    (Real.exp_pos _).le
  intro iterations row
  have previousNonnegative : ∀ other column,
      0 ≤ sinkhorn (iterations + 1) floor initial other column :=
    sinkhornNonnegative (iterations + 1) floor initial floorPositive.le initialNonnegative
  have previousActive : ∀ other,
      floor < rowSum (sinkhorn (iterations + 1) floor initial) other := by
    simpa [floor, initial] using iterActive maximum raw upper iterations
  have stable := fourStable floor (sinkhorn (iterations + 1) floor initial)
    floorPositive floorQuarter previousNonnegative previousActive row
  simpa [floor, initial, sinkhorn, step, Function.iterate_succ_apply'] using stable

theorem Blend.stochastic {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (lower : 0 ≤ lambda) (upper : lambda ≤ 1)
    (stochastic : matrix ∈ doublyStochastic ℝ (Fin streams)) :
    blend lambda matrix ∈ doublyStochastic ℝ (Fin streams) := by
  exact convex_doublyStochastic (Submonoid.one_mem _) stochastic
    (sub_nonneg.mpr upper) lower (by ring)

theorem Stochastic.constant {streams : ℕ} (matrix : Square streams)
    (stochastic : matrix ∈ doublyStochastic ℝ (Fin streams)) (value : ℝ) :
    matrix *ᵥ (fun _ => value) = fun _ => value := by
  funext row
  simp only [mulVec, dotProduct]
  rw [← Finset.sum_mul, sum_row_of_mem_doublyStochastic stochastic, one_mul]

theorem Stochastic.sum {streams : ℕ} (matrix : Square streams)
    (stochastic : matrix ∈ doublyStochastic ℝ (Fin streams)) (value : Vector streams) :
    ∑ row, (matrix *ᵥ value) row = ∑ row, value row :=
  sum_mulVec_of_mem_doublyStochastic stochastic

theorem Stochastic.permutations {streams : ℕ} (matrix : Square streams)
    (stochastic : matrix ∈ doublyStochastic ℝ (Fin streams)) :
    ∃ weights : Equiv.Perm (Fin streams) → ℝ,
      (∀ permutation, 0 ≤ weights permutation) ∧
      ∑ permutation, weights permutation = 1 ∧
      ∑ permutation, weights permutation • permutation.permMatrix ℝ = matrix :=
  exists_eq_sum_perm_of_mem_doublyStochastic stochastic

/-- Row/column mass bounds control squared Euclidean energy, including
approximately stochastic matrices. This concerns a fixed matrix, not the
activation-dependent predictor or the added branch. -/
theorem Approximate.energy {streams : ℕ} (matrix : Square streams)
    (nonnegative : ∀ row column, 0 ≤ matrix row column)
    (rowCap columnCap : ℝ) (rowPositive : 0 ≤ rowCap)
    (rowsBound : ∀ row, rowSum matrix row ≤ rowCap)
    (columnsBound : ∀ column, columnSum matrix column ≤ columnCap)
    (value : Vector streams) :
    ∑ row, (matrix *ᵥ value) row ^ 2 ≤ rowCap * columnCap * ∑ column, value column ^ 2 := by
  have rowBound : ∀ row, (matrix *ᵥ value) row ^ 2 ≤
      rowCap * ∑ column, matrix row column * value column ^ 2 := by
    intro row
    have cauchy := Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul
      (s := Finset.univ) (r := fun column => matrix row column * value column)
      (f := fun column => matrix row column)
      (g := fun column => matrix row column * value column ^ 2)
      (fun column _ => nonnegative row column)
      (fun column _ => mul_nonneg (nonnegative row column) (sq_nonneg _))
      (fun column _ => by nlinarith [sq_nonneg (matrix row column * value column)])
    exact cauchy.trans (mul_le_mul_of_nonneg_right (rowsBound row)
      (Finset.sum_nonneg (fun column _ => mul_nonneg (nonnegative row column) (sq_nonneg _))))
  calc
    _ ≤ ∑ row, rowCap * ∑ column, matrix row column * value column ^ 2 :=
      Finset.sum_le_sum (fun row _ => rowBound row)
    _ = rowCap * ∑ column, columnSum matrix column * value column ^ 2 := by
      rw [← Finset.mul_sum, Finset.sum_comm]
      simp only [columnSum, Finset.sum_mul]
    _ ≤ rowCap * ∑ column, columnCap * value column ^ 2 := by
      apply mul_le_mul_of_nonneg_left _ rowPositive
      exact Finset.sum_le_sum (fun column _ =>
        mul_le_mul_of_nonneg_right (columnsBound column) (sq_nonneg _))
    _ = _ := by rw [← Finset.mul_sum]; ring

theorem Stochastic.energy {streams : ℕ} (matrix : Square streams)
    (stochastic : matrix ∈ doublyStochastic ℝ (Fin streams)) (value : Vector streams) :
    ∑ row, (matrix *ᵥ value) row ^ 2 ≤ ∑ column, value column ^ 2 := by
  have bound := Approximate.energy matrix (fun _ _ => nonneg_of_mem_doublyStochastic stochastic)
    1 1 (by norm_num) (fun row => (sum_row_of_mem_doublyStochastic stochastic row).le)
    (fun column => (sum_col_of_mem_doublyStochastic stochastic column).le) value
  simpa using bound

#print axioms Stochastic.permutations
#print axioms Approximate.energy
#print axioms Stochastic.energy
#print axioms seedRange
#print axioms seedWitness
#print axioms seedRow
#print axioms Row.mass
#print axioms Column.mass
#print axioms Row.capped
#print axioms Column.capped
#print axioms fourActive
#print axioms fourStable
#print axioms sourceLower
#print axioms sourceActive
#print axioms Step.nonnegative
#print axioms sinkhornNonnegative
#print axioms iterActive
#print axioms sourceQuarter
end
end BauxResearch.Forward.Transport
