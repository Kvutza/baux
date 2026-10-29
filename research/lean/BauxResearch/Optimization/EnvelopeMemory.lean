import BauxResearch.Optimization.Envelope
import Mathlib.Data.Finset.Range

/-!+# Streaming observation bounds at fixed reference states

At each reference state, keep the largest observation lower cone and the
smallest upper cone. A fresh observation updates these two scalars by max/min;
old observations and their parameter vectors are not inputs to the update.
The invariant is exact at every reference state for every finite horizon.

Reading at a query within distance r of a reference state widens each endpoint
of the full-history envelope by at most 2*L*r. This is a spatial approximation
error, not an error paid again on every observation. Geometry, regularity, and
the interpretation of observation intervals are fixed throughout the stream.

Two real scalars per reference state is not a finite-bit memory theorem. The
reference states themselves, distance evaluations, arithmetic refinement, and
finding a nearby reference all require separate resource accounting. This
summary does not preserve ENN likelihoods, fitted means, or posterior laws.
-/

namespace BauxResearch.Optimization.EnvelopeMemory

open scoped NNReal

variable {State Index Cell : Type*} [PseudoMetricSpace State]

noncomputable section

/-- Complete-history specification; only used to state the invariant. -/
def summary (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) : Cell → ℝ × ℝ :=
  fun cell => ⟨Envelope.below records nonempty point lower constant (center cell),
    Envelope.above records nonempty point upper constant (center cell)⟩

/-- The bounded-state update. Both fields depend only on the previous pair at
this reference and the fresh observation, not on a replay of earlier records. -/
def update (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (memory : Cell → ℝ × ℝ) (item : Index) : Cell → ℝ × ℝ :=
  fun cell => ⟨max (memory cell).1 (lower item - constant * dist (point item) (center cell)),
    min (memory cell).2 (upper item + constant * dist (point item) (center cell))⟩

/-- Updating from the compressed state agrees exactly with inserting the new
constraint into the complete history, at every fixed reference state. -/
theorem recordLaw [DecidableEq Index] (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (item : Index) :
    update point lower upper constant center
      (summary records nonempty point lower upper constant center) item =
    summary (insert item records) (Finset.insert_nonempty _ _) point lower upper constant center := by
  funext cell
  apply Prod.ext
  · dsimp [summary, update, Envelope.below]
    rw [Finset.sup'_insert nonempty]
    exact max_comm _ _
  · dsimp [summary, update, Envelope.above]
    rw [Finset.inf'_insert nonempty]
    exact min_comm _ _

/-- Time t means that observations 0 through t have arrived. The recursion
stores only a pair per reference, even though the specification mentions all
past observations. Intervals attached to old identifiers do not change. -/
def stream (point : ℕ → State) (lower upper : ℕ → ℝ) (constant : ℝ≥0)
    (center : Cell → State) : ℕ → Cell → ℝ × ℝ
  | 0 => fun cell => ⟨lower 0 - constant * dist (point 0) (center cell),
      upper 0 + constant * dist (point 0) (center cell)⟩
  | time + 1 => update point lower upper constant center
      (stream point lower upper constant center time) (time + 1)

/-- There is no horizon-dependent approximation in the streaming invariant. -/
theorem streamLaw (point : ℕ → State) (lower upper : ℕ → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (time : ℕ) :
    stream point lower upper constant center time =
      summary (Finset.range (time + 1)) ⟨0, Finset.mem_range.mpr (Nat.succ_pos _)⟩
        point lower upper constant center := by
  induction time with
  | zero =>
      funext cell
      simp [stream, summary, Envelope.below, Envelope.above, Finset.range_one]
  | succ time induction =>
      rw [stream, induction]
      have step := recordLaw (Finset.range (time + 1))
        ⟨0, Finset.mem_range.mpr (Nat.succ_pos _)⟩ point lower upper constant center (time + 1)
      convert step using 1
      congr 1
      exact Finset.range_add_one

/-- Transport one stored interval to a query. A nearby reference must be found
by an actual search procedure; its existence is not a free runtime oracle. -/
def read (constant : ℝ≥0) (center : Cell → State) (memory : Cell → ℝ × ℝ)
    (cell : Cell) (query : State) : ℝ × ℝ :=
  ⟨(memory cell).1 - constant * dist (center cell) query,
    (memory cell).2 + constant * dist (center cell) query⟩

/-- At resolution r, both endpoints lose at most 2*L*r relative to retaining
the entire history. This conclusion does not require consistent observations;
consistency is needed when interpreting the bounds as uncertainty about f. -/
theorem readBounds (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State) (radius : ℝ)
    (near : dist (center cell) query ≤ radius) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    Envelope.below records nonempty point lower constant query - 2 * constant * radius ≤ result.1 ∧
      result.1 ≤ Envelope.below records nonempty point lower constant query ∧
      Envelope.above records nonempty point upper constant query ≤ result.2 ∧
      result.2 ≤ Envelope.above records nonempty point upper constant query + 2 * constant * radius := by
  have lowForward := (Envelope.belowLip records nonempty point lower constant).le_add_mul (center cell) query
  have lowBackward := (Envelope.belowLip records nonempty point lower constant).le_add_mul query (center cell)
  have highForward := (Envelope.aboveLip records nonempty point upper constant).le_add_mul (center cell) query
  have highBackward := (Envelope.aboveLip records nonempty point upper constant).le_add_mul query (center cell)
  rw [dist_comm query (center cell)] at lowBackward highBackward
  have distance := mul_le_mul_of_nonneg_left near constant.coe_nonneg
  dsimp [read, summary]
  exact ⟨by linarith, by linarith, by linarith, by linarith⟩

/-- Every compatible objective lies in the transported interval, including
outside the chosen cover. Distance controls sharpness, not basic validity. -/
theorem readValid (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    result.1 ≤ objective query ∧ objective query ≤ result.2 := by
  have bounds := readBounds records nonempty point lower upper constant center cell query
    (dist (center cell) query) le_rfl
  have valid := Envelope.bounds records nonempty point lower upper constant objective regular observed query
  exact ⟨bounds.2.1.trans valid.1, valid.2.trans bounds.2.2.1⟩

/-- The stored interval bounds the error of any predictor, including an ENN
mean fitted separately. The summary is not required to reproduce that mean. -/
theorem estimateBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State) (estimate : ℝ)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    |estimate - objective query| ≤ max |estimate - result.1| |estimate - result.2| := by
  have valid := readValid records nonempty point lower upper constant center cell query objective regular observed
  dsimp only at *
  have low := abs_le.mp (le_max_left
    |estimate - (read constant center (summary records nonempty point lower upper constant center) cell query).1|
    |estimate - (read constant center (summary records nonempty point lower upper constant center) cell query).2|)
  have high := abs_le.mp (le_max_right
    |estimate - (read constant center (summary records nonempty point lower upper constant center) cell query).1|
    |estimate - (read constant center (summary records nonempty point lower upper constant center) cell query).2|)
  apply abs_le.mpr
  constructor <;> linarith [valid.1, valid.2]

/-- For the same predictor, the guaranteed radius from the stored interval
is at most the full-history radius plus 2*L*r. This comparison does not assert
that fitting on compressed versus uncompressed observations gives equal means. -/
theorem radiusBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State) (estimate radius : ℝ)
    (near : dist (center cell) query ≤ radius) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    max |estimate - result.1| |estimate - result.2| ≤
      max |estimate - Envelope.below records nonempty point lower constant query|
        |estimate - Envelope.above records nonempty point upper constant query| + 2 * constant * radius := by
  have approximation := readBounds records nonempty point lower upper constant center cell query radius near
  have low := abs_le.mp (le_max_left
    |estimate - Envelope.below records nonempty point lower constant query|
    |estimate - Envelope.above records nonempty point upper constant query|)
  have high := abs_le.mp (le_max_right
    |estimate - Envelope.below records nonempty point lower constant query|
    |estimate - Envelope.above records nonempty point upper constant query|)
  have positive : 0 ≤ 2 * (constant : ℝ) * radius :=
    mul_nonneg (mul_nonneg (by norm_num) constant.coe_nonneg) (dist_nonneg.trans near)
  dsimp only at *
  apply max_le
  · apply abs_le.mpr
    constructor <;> linarith [approximation.1, approximation.2.1]
  · apply abs_le.mpr
    constructor <;> linarith [approximation.2.2.1, approximation.2.2.2]

/-- The midpoint's guaranteed error exceeds the full-history half-width by
at most 2*L*r, however many observations have been processed. It need not be
small unless the full-history uncertainty is already small. -/
theorem midpointBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State) (radius : ℝ)
    (near : dist (center cell) query ≤ radius)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    |(result.1 + result.2) / 2 - objective query| ≤
      (Envelope.above records nonempty point upper constant query -
        Envelope.below records nonempty point lower constant query) / 2 + 2 * constant * radius := by
  have approximation := readBounds records nonempty point lower upper constant center cell query radius near
  have valid := readValid records nonempty point lower upper constant center cell query objective regular observed
  dsimp only at *
  apply abs_le.mpr
  constructor <;> linarith [approximation.1, approximation.2.2.2, valid.1, valid.2]

/-- A nearby narrow observation supplies an explicit absolute error bound.
More observations do not automatically shrink their declared error intervals. -/
theorem localError (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query : State) (radius noise distance : ℝ)
    (near : dist (center cell) query ≤ radius) (item : Index) (member : item ∈ records)
    (close : dist (point item) query ≤ distance) (narrow : upper item - lower item ≤ 2 * noise)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective) :
    let result := read constant center (summary records nonempty point lower upper constant center) cell query
    |(result.1 + result.2) / 2 - objective query| ≤ noise + constant * distance + 2 * constant * radius := by
  have prediction := midpointBound records nonempty point lower upper constant center cell query radius near
    objective regular observed
  have low := Finset.le_sup' (fun index => lower index - constant * dist (point index) query) member
  have high := Finset.inf'_le (fun index => upper index + constant * dist (point index) query) member
  change _ ≤ Envelope.below records nonempty point lower constant query at low
  change Envelope.above records nonempty point upper constant query ≤ _ at high
  have metric := mul_le_mul_of_nonneg_left close constant.coe_nonneg
  dsimp only at *
  linarith

/-- A lower bound at a reference can rule out an entire ball. The conclusion
holds for every later recommendation whose true loss remains below the same
validated upper bound. No accurate prediction inside that ball is required.
Keeping track of excluded regions is a separate memory cost. -/
theorem excluded (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (center : Cell → State) (cell : Cell) (query final : State) (radius bound : ℝ)
    (near : dist (center cell) query ≤ radius)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective)
    (incumbent : objective final ≤ bound)
    (ruledOut : bound < (summary records nonempty point lower upper constant center cell).1 -
      constant * radius) : objective final < objective query := by
  have valid := readValid records nonempty point lower upper constant center cell query objective regular observed
  have distance := mul_le_mul_of_nonneg_left near constant.coe_nonneg
  dsimp [read] at valid
  linarith [valid.1]

/-- Two valid histories, from f(x)=x and f(x)=-x, give the same summary at
reference 0 but different summaries at reference 1. Thus moving or refining
references cannot in general reconstruct full-history information from the
old scalar summary alone. This does not prohibit conservative transport. -/
theorem refinementLoss :
    let summarized := fun (sign query : ℝ) =>
      summary ({0, 1} : Finset ℕ) (by simp) (fun index => (index : ℝ))
        (fun index => sign * index) (fun index => sign * index) 1 (fun _ : Unit => query) ()
    summarized 1 0 = summarized (-1) 0 ∧ summarized 1 1 ≠ summarized (-1) 1 := by
  norm_num [summary, Envelope.below, Envelope.above,
    Finset.sup'_insert (Finset.singleton_nonempty 1),
    Finset.inf'_insert (Finset.singleton_nonempty 1), Real.dist_eq]

#print axioms recordLaw
#print axioms streamLaw
#print axioms readBounds
#print axioms readValid
#print axioms estimateBound
#print axioms radiusBound
#print axioms midpointBound
#print axioms localError
#print axioms excluded
#print axioms refinementLoss

end
end BauxResearch.Optimization.EnvelopeMemory
