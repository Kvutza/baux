import BauxResearch.Objective.NaturalWindow
import BauxResearch.Optimization.EnvelopeMemory

/-!
# Corpus-loss intervals from contextual observations

An observation measures l(x,o), not the corpus mean F(x). Keep origin o in
the mathematical domain instead of treating its observed value as F(x).
At each fixed reference state and each origin, store an interval for l.
A new observation tightens only its own origin's intervals by Lipschitz
transport. Averaging transported intervals encloses F at every query.

The state has two real fields per reference/origin pair, independent of the
number of observations. This is not a small-corpus or finite-bit result.
The original window scores and origins are available in the inspected runner,
but this construction is NOT implemented there. No extra forward evaluation
is part of this mathematical update. A useful parameter metric, its constant,
initial bounds, floating-point enclosures, and affordable references are still
required. No independence or conditional uniformity of the origin is assumed.
-/

namespace BauxResearch.Objective.CorpusMemory

open scoped BigOperators NNReal
open BauxResearch.Objective.NaturalWindow
open BauxResearch.Optimization

noncomputable section

variable {State Origin Cell : Type*} [PseudoMetricSpace State]

/-- The initial bounds concern each reference/origin pair, not empirical
confidence inferred from zero observations. -/
def initial (bound : ℝ) : Origin → Cell → ℝ × ℝ := fun _ _ => (0, bound)

/-- Reuse the existing two-scalar envelope update, but only at the observed
origin. Other origins are unchanged. No past parameter state is an input. -/
def update [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (memory : Origin → Cell → ℝ × ℝ) (point : State) (selected : Origin)
    (score error : ℝ) : Origin → Cell → ℝ × ℝ :=
  fun origin => if origin = selected then
    EnvelopeMemory.update (fun _ : Unit => point) (fun _ => score - error)
      (fun _ => score + error) constant center (memory origin) ()
    else memory origin

/-- After t observations, identifiers 0,...,t-1 have been consumed. -/
def stream [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Origin)
    (score error : ℕ → ℝ) : ℕ → Origin → Cell → ℝ × ℝ
  | 0 => initial bound
  | time + 1 => update constant center (stream constant center bound point origin score error time)
      (point time) (origin time) (score time) (error time)

def valid (loss : State → Origin → ℝ) (center : Cell → State)
    (memory : Origin → Cell → ℝ × ℝ) : Prop :=
  ∀ origin cell, (memory origin cell).1 ≤ loss (center cell) origin ∧
    loss (center cell) origin ≤ (memory origin cell).2

omit [PseudoMetricSpace State] in
theorem initialValid (loss : State → Origin → ℝ) (center : Cell → State) (bound : ℝ)
    (bounded : ∀ origin cell, loss (center cell) origin ∈ Set.Icc 0 bound) :
    valid loss center (initial bound) := bounded

/-- The observation error is against its OWN window loss. It need not
estimate the corpus mean. Exact real evaluations instantiate error=0. -/
theorem updateValid [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (memory : Origin → Cell → ℝ × ℝ) (point : State) (selected : Origin)
    (score error : ℝ) (loss : State → Origin → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (previous : valid loss center memory)
    (observed : |score - loss point selected| ≤ error) :
    valid loss center (update constant center memory point selected score error) := by
  intro origin cell
  by_cases same : origin = selected
  · subst origin
    have lower := (regular selected).le_add_mul point (center cell)
    have upper := (regular selected).le_add_mul (center cell) point
    rw [dist_comm (center cell) point] at upper
    have interval := abs_le.mp observed
    have low : score - error - constant * dist point (center cell) ≤
        loss (center cell) selected := by linarith
    have high : loss (center cell) selected ≤
        score + error + constant * dist point (center cell) := by linarith
    simpa [update, EnvelopeMemory.update] using
      And.intro (max_le (previous selected cell).1 low) (le_min (previous selected cell).2 high)
  · simpa only [update, ite_eq_right same] using previous origin cell

/-- Simultaneous pathwise validity for every finite horizon. The candidate
and origin streams may be adaptive; there is no union-bound failure budget. -/
theorem streamValid [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Origin) (score error : ℕ → ℝ)
    (loss : State → Origin → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (bounded : ∀ origin cell, loss (center cell) origin ∈ Set.Icc 0 bound)
    (observed : ∀ time, |score time - loss (point time) (origin time)| ≤ error time)
    (time : ℕ) : valid loss center (stream constant center bound point origin score error time) := by
  induction time with
  | zero => exact initialValid loss center bound bounded
  | succ time induction =>
      exact updateValid constant center _ _ _ _ _ loss regular induction (observed time)

/-- Both endpoints tighten monotonically, even if the supplied observations
are inconsistent. Consistency is needed for validity, not this order property. -/
theorem updateOrder [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (memory : Origin → Cell → ℝ × ℝ) (point : State) (selected : Origin)
    (score error : ℝ) (origin : Origin) (cell : Cell) :
    (memory origin cell).1 ≤ (update constant center memory point selected score error origin cell).1 ∧
      (update constant center memory point selected score error origin cell).2 ≤ (memory origin cell).2 := by
  by_cases same : origin = selected
  · simp only [update, ite_eq_left same, EnvelopeMemory.update]
    exact ⟨le_max_left _ _, min_le_left _ _⟩
  · simp [update, same]

theorem streamOrder [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Origin) (score error : ℕ → ℝ)
    (first second : ℕ) (before : first ≤ second) (selected : Origin) (cell : Cell) :
    (stream constant center bound point origin score error first selected cell).1 ≤
        (stream constant center bound point origin score error second selected cell).1 ∧
      (stream constant center bound point origin score error second selected cell).2 ≤
        (stream constant center bound point origin score error first selected cell).2 := by
  induction second, before using Nat.le_induction with
  | base => exact ⟨le_rfl, le_rfl⟩
  | succ second before induction =>
      have ordered := updateOrder constant center
        (stream constant center bound point origin score error second)
        (point second) (origin second) (score second) (error second) selected cell
      exact ⟨induction.1.trans ordered.1, ordered.2.trans induction.2⟩

/-- A read uses one fixed parameter reference and averages over all origins.
This finite mean is arithmetic on stored fields, not fresh model evaluations. -/
def read {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (cell : Cell) (query : State) : ℝ × ℝ :=
  (corpusMean (fun origin => (EnvelopeMemory.read constant center (memory origin) cell query).1),
    corpusMean (fun origin => (EnvelopeMemory.read constant center (memory origin) cell query).2))

theorem meanOrder {count : ℕ} (lower upper : Fin count → ℝ)
    (ordered : ∀ origin, lower origin ≤ upper origin) : corpusMean lower ≤ corpusMean upper :=
  mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun origin _ => ordered origin)
    (inv_nonneg.mpr (Nat.cast_nonneg count))

theorem readValid {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (cell : Cell) (query : State)
    (loss : State → Fin count → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (previous : valid loss center memory) :
    (read constant center memory cell query).1 ≤ corpusMean (loss query) ∧
      corpusMean (loss query) ≤ (read constant center memory cell query).2 := by
  constructor
  · apply meanOrder
    intro origin
    have transport := (regular origin).le_add_mul (center cell) query
    have old := (previous origin cell).1
    dsimp [EnvelopeMemory.read]
    linarith
  · apply meanOrder
    intro origin
    have transport := (regular origin).le_add_mul query (center cell)
    rw [dist_comm query (center cell)] at transport
    have old := (previous origin cell).2
    dsimp [EnvelopeMemory.read]
    linarith

/-- An explicit radius computed from the stored state; no unknown path-bias
sum is an input to the read. Its smallness remains a geometric/data question. -/
theorem midpointBound {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (cell : Cell) (query : State)
    (loss : State → Fin count → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (previous : valid loss center memory) :
    let result := read constant center memory cell query
    |(result.1 + result.2) / 2 - corpusMean (loss query)| ≤ (result.2 - result.1) / 2 := by
  have bounds := readValid constant center memory cell query loss regular previous
  apply abs_le.mpr
  constructor <;> linarith [bounds.1, bounds.2]

/-- Instantiate the observation condition, rather than assuming population
accuracy: every record is its exact own-window loss. The conclusion bounds
corpus loss using the explicit streaming construction at ANY finite horizon.
Source floating-point evaluations require streamValid's nonzero enclosures. -/
theorem exactBound {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Fin count)
    (loss : State → Fin count → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (bounded : ∀ origin cell, loss (center cell) origin ∈ Set.Icc 0 bound)
    (time : ℕ) (cell : Cell) (query : State) :
    let memory := stream constant center bound point origin
      (fun step => loss (point step) (origin step)) (fun _ => 0) time
    let result := read constant center memory cell query
    |(result.1 + result.2) / 2 - corpusMean (loss query)| ≤ (result.2 - result.1) / 2 := by
  apply midpointBound constant center _ cell query loss regular
  apply streamValid constant center bound point origin _ _ loss regular bounded
  intro step
  simp

/-- Corpus-relative reward has an interval too. This matches a zero-anchored
score when base is the initial state. It does not confuse that score with
negative absolute NLL, and it does not require knowing F(base) exactly. -/
def gainInterval {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (baseCell trialCell : Cell)
    (base trial : State) : ℝ × ℝ :=
  let first := read constant center memory baseCell base
  let second := read constant center memory trialCell trial
  (first.1 - second.2, first.2 - second.1)

theorem gainValid {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (baseCell trialCell : Cell)
    (base trial : State) (loss : State → Fin count → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (previous : valid loss center memory) :
    let result := gainInterval constant center memory baseCell trialCell base trial
    result.1 ≤ corpusMean (loss base) - corpusMean (loss trial) ∧
      corpusMean (loss base) - corpusMean (loss trial) ≤ result.2 := by
  have first := readValid constant center memory baseCell base loss regular previous
  have second := readValid constant center memory trialCell trial loss regular previous
  dsimp [gainInterval]
  constructor <;> linarith [first.1, first.2, second.1, second.2]

/-- An independently fitted ENN prediction can keep its own meaning and
formula: its corpus-relative error is bounded using interval endpoints.
This does not claim calibration of the ENN variance or preserve its posterior. -/
theorem relativeError {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (baseCell trialCell : Cell)
    (base trial : State) (estimate : ℝ) (loss : State → Fin count → ℝ)
    (regular : ∀ origin, LipschitzWith constant (fun state => loss state origin))
    (previous : valid loss center memory) :
    let result := gainInterval constant center memory baseCell trialCell base trial
    |estimate - (corpusMean (loss base) - corpusMean (loss trial))| ≤
      max |estimate - result.1| |estimate - result.2| := by
  have bounds := gainValid constant center memory baseCell trialCell base trial loss regular previous
  dsimp only at *
  have low := abs_le.mp (le_max_left
    |estimate - (gainInterval constant center memory baseCell trialCell base trial).1|
    |estimate - (gainInterval constant center memory baseCell trialCell base trial).2|)
  have high := abs_le.mp (le_max_right
    |estimate - (gainInterval constant center memory baseCell trialCell base trial).1|
    |estimate - (gainInterval constant center memory baseCell trialCell base trial).2|)
  apply abs_le.mpr
  constructor <;> linarith [bounds.1, bounds.2]

/-- At a fixed query and reference, additional data never widens the read.
Moving the query may change its radius; that is not hidden in this statement. -/
theorem readOrder {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (first second : Fin count → Cell → ℝ × ℝ) (cell : Cell) (query : State)
    (ordered : ∀ origin,
      (first origin cell).1 ≤ (second origin cell).1 ∧
        (second origin cell).2 ≤ (first origin cell).2) :
    (read constant center first cell query).1 ≤ (read constant center second cell query).1 ∧
      (read constant center second cell query).2 ≤ (read constant center first cell query).2 := by
  constructor
  · exact meanOrder _ _ fun origin => sub_le_sub_right (ordered origin).1 _
  · apply meanOrder
    intro origin
    dsimp [EnvelopeMemory.read]
    linarith [(ordered origin).2]

/-- Every observation remains represented by an interval constraint at its
own origin, however many later updates arrive. Historical vectors are used
only in this proof statement, not accessed by the streaming update. -/
theorem recordBound [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Origin) (score error : ℕ → ℝ)
    (item time : ℕ) (before : item < time) (cell : Cell) :
    score item - error item - constant * dist (point item) (center cell) ≤
        (stream constant center bound point origin score error time (origin item) cell).1 ∧
      (stream constant center bound point origin score error time (origin item) cell).2 ≤
        score item + error item + constant * dist (point item) (center cell) := by
  have later := streamOrder constant center bound point origin score error
    (item + 1) time before (origin item) cell
  have fresh :
      score item - error item - constant * dist (point item) (center cell) ≤
          (stream constant center bound point origin score error (item + 1) (origin item) cell).1 ∧
        (stream constant center bound point origin score error (item + 1) (origin item) cell).2 ≤
          score item + error item + constant * dist (point item) (center cell) := by
    simp [stream, update, EnvelopeMemory.update]
  exact ⟨fresh.1.trans later.1, later.2.trans fresh.2⟩

/-- If each origin has one observation close to the query, the computed
half-width is at most mean(error + L*distance) + 2*L*reference_distance.
No factor proportional to the number of observations appears. -/
theorem coverageBound {count : ℕ} (positive : 0 < count)
    (constant : ℝ≥0) (center : Cell → State) (bound : ℝ)
    (point : ℕ → State) (origin : ℕ → Fin count) (score error : ℕ → ℝ)
    (time : ℕ) (cell : Cell) (query : State) (radius : ℝ)
    (near : dist (center cell) query ≤ radius)
    (witness : Fin count → ℕ) (before : ∀ selected, witness selected < time)
    (originEq : ∀ selected, origin (witness selected) = selected) :
    let result := read constant center
      (stream constant center bound point origin score error time) cell query
    (result.2 - result.1) / 2 ≤
      corpusMean (fun selected => error (witness selected) +
        constant * dist (point (witness selected)) query) + 2 * constant * radius := by
  have widths : ∀ selected,
      (EnvelopeMemory.read constant center
        (stream constant center bound point origin score error time selected) cell query).2 -
      (EnvelopeMemory.read constant center
        (stream constant center bound point origin score error time selected) cell query).1 ≤
      2 * (error (witness selected) + constant * dist (point (witness selected)) query) +
        4 * constant * radius := by
    intro selected
    have record := recordBound constant center bound point origin score error
      (witness selected) time (before selected) cell
    rw [originEq] at record
    have triangle := dist_triangle (point (witness selected)) query (center cell)
    rw [dist_comm query (center cell)] at triangle
    have scaled := mul_le_mul_of_nonneg_left triangle constant.coe_nonneg
    have reference := mul_le_mul_of_nonneg_left near constant.coe_nonneg
    dsimp [EnvelopeMemory.read]
    linarith [record.1, record.2]
  have averaged := meanOrder _ _ widths
  have nonzero : (count : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt positive
  dsimp [read, corpusMean] at averaged ⊢
  rw [Finset.sum_sub_distrib] at averaged
  simp_rw [Finset.sum_add_distrib, ← Finset.mul_sum] at averaged
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at averaged
  have cancel : (count : ℝ)⁻¹ * (4 * constant * ((count : ℝ) * radius)) =
      4 * constant * radius := by field_simp
  rw [mul_add, cancel] at averaged
  linarith

/-- If an origin has never been observed, its reference bounds remain exactly
the prior interval. Repeated visits to other origins do not change them. -/
theorem unseenExact [DecidableEq Origin] (constant : ℝ≥0) (center : Cell → State)
    (bound : ℝ) (point : ℕ → State) (origin : ℕ → Origin) (score error : ℕ → ℝ)
    (selected : Origin) (time : ℕ) (unseen : ∀ item < time, origin item ≠ selected) :
    stream constant center bound point origin score error time selected = initial bound selected := by
  induction time with
  | zero => rfl
  | succ time induction =>
      have last : selected ≠ origin time := Ne.symm (unseen time (Nat.lt_succ_self _))
      simp only [stream, update, ite_eq_right last]
      exact induction (fun item before => unseen item (Nat.lt_succ_of_lt before))

/-- A single changed origin updates the mean by a scalar correction. This
identity permits cached means; it does not require rescanning the corpus. -/
theorem meanStep {count : ℕ} (first second : Fin count → ℝ) (selected : Fin count)
    (unchanged : ∀ other, other ≠ selected → second other = first other) :
    corpusMean second = corpusMean first +
      (count : ℝ)⁻¹ * (second selected - first selected) := by
  have difference : ∑ origin, (second origin - first origin) =
      second selected - first selected := by
    apply Finset.sum_eq_single selected
    · intro other _ different
      rw [unchanged other different, sub_self]
    · simp
  rw [Finset.sum_sub_distrib] at difference
  dsimp [corpusMean]
  rw [← difference]
  ring

/-- Each reference's cached lower/upper means can be maintained with the old
and new pair at just the observed origin. This is exact-real arithmetic;
roundoff in repeated cache corrections needs a separate numerical bound. -/
theorem meanUpdate {count : ℕ} (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (point : State) (selected : Fin count)
    (score error : ℝ) (cell : Cell) :
    let next := update constant center memory point selected score error
    corpusMean (fun origin => (next origin cell).1) =
        corpusMean (fun origin => (memory origin cell).1) +
          (count : ℝ)⁻¹ * ((next selected cell).1 - (memory selected cell).1) ∧
      corpusMean (fun origin => (next origin cell).2) =
        corpusMean (fun origin => (memory origin cell).2) +
          (count : ℝ)⁻¹ * ((next selected cell).2 - (memory selected cell).2) := by
  constructor <;> apply meanStep <;> intro other different <;> simp [update, different]

/-- A read needs only cached reference means and one query/reference distance.
The origin count cancels from the shared transport term. -/
theorem readCached {count : ℕ} (positive : 0 < count)
    (constant : ℝ≥0) (center : Cell → State)
    (memory : Fin count → Cell → ℝ × ℝ) (cell : Cell) (query : State) :
    read constant center memory cell query =
      (corpusMean (fun origin => (memory origin cell).1) - constant * dist (center cell) query,
        corpusMean (fun origin => (memory origin cell).2) + constant * dist (center cell) query) := by
  have nonzero : (count : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt positive
  apply Prod.ext <;>
    simp [read, EnvelopeMemory.read, corpusMean, Finset.sum_sub_distrib,
      Finset.sum_add_distrib, mul_sub, mul_add, ← mul_assoc, nonzero]

/-- Unobserved origins may reverse the ranking of two parameter states.
Observed origins give identical zero loss for both alternative objectives. -/
def blindLoss {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (favored state : Fin 2) (origin : Fin count) : ℝ :=
  if origin ∈ seen then 0 else if state = favored then 0 else scale

theorem blindBound {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (nonnegative : 0 ≤ scale) (favored state : Fin 2) (origin : Fin count) :
    blindLoss seen scale favored state origin ∈ Set.Icc 0 scale := by
  simp only [blindLoss]
  split_ifs <;> constructor <;> linarith

/-- The two objectives agree on every measured origin, at EVERY parameter
state in this two-state decision problem, not just at one selected point. -/
theorem blindAgreement {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (state : Fin 2) (origin : Fin count) (observed : origin ∈ seen) :
    blindLoss seen scale 0 state origin = blindLoss seen scale 1 state origin := by
  simp [blindLoss, observed]

/-- These are regular two-state objectives: loss change is bounded by scale
times the ordinary distance between the encoded parameter values 0 and 1. -/
theorem blindRegular {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (nonnegative : 0 ≤ scale) (favored left right : Fin 2) (origin : Fin count) :
    |blindLoss seen scale favored left origin - blindLoss seen scale favored right origin| ≤
      scale * |(left.val : ℝ) - (right.val : ℝ)| := by
  fin_cases favored <;> fin_cases left <;> fin_cases right <;>
    by_cases observed : origin ∈ seen <;>
    simp [blindLoss, observed, abs_of_nonneg nonnegative, nonnegative]

theorem blindMean {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (favored state : Fin 2) :
    corpusMean (blindLoss seen scale favored state) =
      if state = favored then 0 else
        (count : ℝ)⁻¹ * ((Finset.univ \ seen).card : ℝ) * scale := by
  by_cases same : state = favored
  · simp [blindLoss, same, corpusMean]
  · simp [blindLoss, same, corpusMean, Finset.sum_ite, Finset.sdiff_eq_filter, mul_assoc]

theorem blindOptimal {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (nonnegative : 0 ≤ scale) (favored state : Fin 2) :
    corpusMean (blindLoss seen scale favored favored) ≤
      corpusMean (blindLoss seen scale favored state) := by
  apply meanOrder
  intro origin
  have lower := (blindBound seen scale nonnegative favored state origin).1
  simpa [blindLoss] using lower

/-- For either recommendation among the two states, one indistinguishable
objective gives simple regret scale times the unobserved-origin fraction.
This is a deterministic two-state decision obstruction, not a lower bound
proved for the actual neural-network class or a randomized policy. -/
theorem unseenRegret {count : ℕ} (seen : Finset (Fin count)) (scale : ℝ)
    (nonnegative : 0 ≤ scale) (recommendation : Fin 2) :
    ∃ favored : Fin 2,
      (∀ state, corpusMean (blindLoss seen scale favored favored) ≤
        corpusMean (blindLoss seen scale favored state)) ∧
        corpusMean (blindLoss seen scale favored recommendation) -
        corpusMean (blindLoss seen scale favored favored) =
          (count : ℝ)⁻¹ * ((Finset.univ \ seen).card : ℝ) * scale := by
  fin_cases recommendation
  · exact ⟨1, blindOptimal seen scale nonnegative 1, by simp [blindMean]⟩
  · exact ⟨0, blindOptimal seen scale nonnegative 0, by simp [blindMean]⟩

#print axioms initialValid
#print axioms updateValid
#print axioms streamValid
#print axioms updateOrder
#print axioms streamOrder
#print axioms meanOrder
#print axioms readValid
#print axioms midpointBound
#print axioms exactBound
#print axioms gainValid
#print axioms relativeError
#print axioms readOrder
#print axioms recordBound
#print axioms coverageBound
#print axioms unseenExact
#print axioms meanStep
#print axioms meanUpdate
#print axioms readCached
#print axioms blindBound
#print axioms blindAgreement
#print axioms blindRegular
#print axioms blindMean
#print axioms blindOptimal
#print axioms unseenRegret

end
end BauxResearch.Objective.CorpusMemory
