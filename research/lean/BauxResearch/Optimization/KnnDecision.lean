import BauxResearch.Optimization.KnnLearning
import BauxResearch.Optimization.Calibration
import BauxResearch.Optimization.Localized
import Mathlib.Data.List.MinMax

/-!
# Constructed optimistic KNN decisions

The query selector is a specified finite scan over cached predictions, not an
assumed acquisition minimizer. Mathlib's list argmin proves its optimality; an
instrumented scan counts comparisons. Candidate generation, computing the
predictions, and materializing full parameter states remain separate costs.

The horizon result allows informative queries that do not improve the incumbent.
It derives the optimistic-selection premise from this scan and its prediction
certificates. Local recall and a justified observation/metric class remain
explicit. No efficient global candidate cover or production policy is asserted.
-/

namespace BauxResearch.Optimization.KnnDecision

open BauxResearch.Forward
open scoped NNReal

noncomputable section

/-- Cached prediction for one complete candidate state. The state is not a
compressed parameterization; `error` is an analytical certificate, not a new
objective observation. Real comparisons are semantic, not a numeric backend. -/
structure Scored (State : Type*) where
  state : State
  mean : ℝ
  error : ℝ

def lower {State : Type*} (row : Scored State) : ℝ := row.mean - row.error

/-- Keep the earlier candidate on ties and count one scalar comparison per
additional cached candidate. No objective function is an input. -/
def scan {State : Type*} (first : Scored State) : List (Scored State) → Scored State × ℕ
  | [] => (first, 0)
  | next :: rest =>
      let result := scan (if lower next < lower first then next else first) rest
      (result.1, result.2 + 1)

def query {State : Type*} (first : Scored State) (rest : List (Scored State)) : Scored State :=
  (scan first rest).1

/-- The instrumented scan is exactly Mathlib's stable finite argmin. -/
theorem scanLaw {State : Type*} (first : Scored State) (rest : List (Scored State)) :
    List.argmin lower (first :: rest) = some (query first rest) := by
  induction rest generalizing first with
  | nil => rfl
  | cons next rest induction =>
      by_cases better : lower next < lower first
      · simpa [query, scan, better, List.argmin, List.argAux] using induction next
      · simpa [query, scan, better, List.argmin, List.argAux] using induction first

theorem queryMember {State : Type*} (first : Scored State) (rest : List (Scored State)) :
    query first rest ∈ first :: rest :=
  List.argmin_mem (by rw [scanLaw]; exact Option.mem_def.mpr rfl)

theorem queryOptimal {State : Type*} (first : Scored State) (rest : List (Scored State))
    (row : Scored State) (member : row ∈ first :: rest) :
    lower (query first rest) ≤ lower row :=
  List.le_of_mem_argmin member (by rw [scanLaw]; exact Option.mem_def.mpr rfl)

theorem comparisons {State : Type*} (first : Scored State) (rest : List (Scored State)) :
    (scan first rest).2 = rest.length := by
  induction rest generalizing first with
  | nil => rfl
  | cons next rest induction => simp [scan, induction]

/-- A minimum-observation recommendation can be maintained incrementally.
The old prefix is represented by its winning row, not stored or rescanned. -/
theorem scanAppend {State : Type*} (first : Scored State)
    (past fresh : List (Scored State)) :
    query first (past ++ fresh) = query (query first past) fresh := by
  induction past generalizing first with
  | nil => rfl
  | cons next past induction =>
      simpa [query, scan] using (induction (if lower next < lower first then next else first))

/-- Prediction evaluation is a separate pass: one cached row per supplied
candidate. This count does not price KNN searches or candidate generation. -/
def cache {State : Type*} (predict : State → ℝ × ℝ) (states : List State) : List (Scored State) :=
  states.map fun state =>
    let prediction := predict state
    ⟨state, prediction.1, prediction.2⟩

theorem cacheLength {State : Type*} (predict : State → ℝ × ℝ) (states : List State) :
    (cache predict states).length = states.length := by simp [cache]

/-- Only the chosen row becomes an objective query. Accuracy at a comparison
row is enough for optimism relative to that row; no useful-update premise occurs. -/
theorem optimistic {State : Type*} (objective : State → ℝ)
    (first : Scored State) (rest : List (Scored State)) (comparison : Scored State)
    (member : comparison ∈ first :: rest)
    (accurate : |comparison.mean - objective comparison.state| ≤ comparison.error) :
    lower (query first rest) ≤ objective comparison.state := by
  have bound := queryOptimal first rest comparison member
  have error := (abs_le.mp accurate).2
  dsimp [lower] at *
  linarith

/-- The selected candidate need not improve the incumbent on every round.
Its objective gap is bounded by its own uncertainty, not the worst pool width. -/
theorem selectedGap {State : Type*} (objective : State → ℝ)
    (first : Scored State) (rest : List (Scored State)) (comparison : Scored State)
    (member : comparison ∈ first :: rest)
    (referenceError : |comparison.mean - objective comparison.state| ≤ comparison.error)
    (selectedError : |(query first rest).mean - objective (query first rest).state| ≤
      (query first rest).error) :
    objective (query first rest).state - objective comparison.state ≤
      2 * (query first rest).error := by
  have optimism := optimistic objective first rest comparison member referenceError
  have error := (abs_le.mp selectedError).1
  dsimp [lower] at optimism
  linarith

/-- A finite represented comparison within a metric radius of the desired
reference contributes an explicit coverage penalty. Coverage is not free. -/
theorem coveredScore {State : Type*} [PseudoMetricSpace State]
    (objective : State → ℝ) (constant : ℝ≥0) (regular : LipschitzWith constant objective)
    (first : Scored State) (rest : List (Scored State)) (comparison : Scored State)
    (reference : State) (radius : ℝ)
    (member : comparison ∈ first :: rest)
    (accurate : |comparison.mean - objective comparison.state| ≤ comparison.error)
    (covered : dist comparison.state reference ≤ radius) :
    lower (query first rest) ≤ objective reference + constant * radius := by
  have optimism := optimistic objective first rest comparison member accurate
  have continuity := regular.dist_le_mul comparison.state reference
  rw [Real.dist_eq] at continuity
  have near := mul_le_mul_of_nonneg_left covered constant.coe_nonneg
  have upper := (abs_le.mp continuity).2
  linarith

/-- The finite selector supplies optimism for the localized query count.
Only its selected state is evaluated. A represented accurate comparison near
the reference is still required: neither a small near-optimal cover nor this
theorem constructs an affordable candidate generator. Memory loss is counted
from earlier evaluations and the actual available observation identities. -/
theorem localizedQueries {State : Type*} [PseudoMetricSpace State] {horizon : ℕ}
    (objective : State → ℝ) (constant : ℝ≥0) (regular : LipschitzWith constant objective)
    (first : Fin horizon → Scored State) (rest : Fin horizon → List (Scored State))
    (comparison : Fin horizon → Scored State) (reference : State)
    (queries : Finset (Fin horizon)) (memory : Fin horizon → Finset (Fin horizon))
    (neighbors : ℕ) (width candidateRadius : ℝ) (radius : ℝ≥0)
    (finite : Metric.coveringNumber radius
      (Localized.nearSet objective (objective reference)
        (2 * width + constant * candidateRadius)) ≠ ⊤)
    (represented : ∀ time ∈ queries, comparison time ∈ first time :: rest time)
    (accurate : ∀ time ∈ queries,
      |(comparison time).mean - objective (comparison time).state| ≤ (comparison time).error)
    (covered : ∀ time ∈ queries, dist (comparison time).state reference ≤ candidateRadius)
    (selected : ∀ time ∈ queries,
      |(query (first time) (rest time)).mean - objective (query (first time) (rest time)).state| ≤
        (query (first time) (rest time)).error)
    (unresolved : ∀ time ∈ queries, width / 2 < (query (first time) (rest time)).error)
    (bounded : ∀ time ∈ queries, (query (first time) (rest time)).error ≤ width)
    (retained : ∀ time ∈ queries,
      neighbors ≤ (Localized.nearby (memory time)
        (fun previous => (query (first previous) (rest previous)).state) (2 * radius) time).card →
      (query (first time) (rest time)).error ≤ width / 2) :
    queries.card ≤
      (Metric.coveringNumber radius (Localized.nearSet objective (objective reference)
        (2 * width + constant * candidateRadius))).toNat * neighbors +
      (Localized.recallLoss queries memory
        (fun time => (query (first time) (rest time)).state) neighbors (2 * radius)).card := by
  apply Localized.localizedCount queries memory
    (fun time => (query (first time) (rest time)).state) objective
    (fun time => (query (first time) (rest time)).mean)
    (fun time => (query (first time) (rest time)).error)
    neighbors (objective reference) (constant * candidateRadius) width radius
    finite selected ?_ unresolved bounded retained
  intro time member
  exact coveredScore objective constant regular (first time) (rest time) (comparison time)
    reference candidateRadius (represented time member) (accurate time member) (covered time member)

/-- A checkable stopping certificate compares an upper bound on the final
state with the lowest cached optimistic score. Coverage remains explicit. -/
theorem stopBound {State : Type*} [PseudoMetricSpace State]
    (objective : State → ℝ) (constant : ℝ≥0) (regular : LipschitzWith constant objective)
    (first : Scored State) (rest : List (Scored State)) (comparison : Scored State)
    (reference final : State) (radius upper tolerance : ℝ)
    (member : comparison ∈ first :: rest)
    (accurate : |comparison.mean - objective comparison.state| ≤ comparison.error)
    (covered : dist comparison.state reference ≤ radius)
    (finalBound : objective final ≤ upper)
    (stopped : upper - lower (query first rest) ≤ tolerance) :
    objective final - objective reference ≤ tolerance + constant * radius := by
  have selected := coveredScore objective constant regular first rest comparison reference radius
    member accurate covered
  linarith

/-- The finite-horizon certificate now uses the constructed selector rather
than an assumed `ApproxMin`. The cover and recall premises constrain geometry
and retained data, not a desired per-round objective improvement. -/
theorem horizonBound {State : Type*} [PseudoMetricSpace State] {horizon cells : ℕ}
    (objective : State → ℝ) (constant : ℝ≥0) (regular : LipschitzWith constant objective)
    (first : Fin horizon → Scored State) (rest : Fin horizon → List (Scored State))
    (comparison : Fin horizon → Scored State) (reference : State)
    (cell : State → Fin cells) (neighbors : ℕ)
    (finalLoss noise coverRadius candidateRadius : ℝ)
    (budget : cells * neighbors < horizon)
    (represented : ∀ time, comparison time ∈ first time :: rest time)
    (accurate : ∀ time,
      |(comparison time).mean - objective (comparison time).state| ≤ (comparison time).error)
    (covered : ∀ time, dist (comparison time).state reference ≤ candidateRadius)
    (retained : ∀ time,
      neighbors ≤ (Finset.univ.filter fun previous =>
        previous < time ∧ cell (query (first previous) (rest previous)).state =
          cell (query (first time) (rest time)).state).card →
      (query (first time) (rest time)).error ≤ noise + 2 * constant * coverRadius)
    (recommended : ∀ time, finalLoss - noise ≤ (query (first time) (rest time)).mean) :
    finalLoss - objective reference ≤
      2 * noise + 2 * constant * coverRadius + constant * candidateRadius := by
  obtain ⟨time, repeated⟩ := KnnLearning.repeatedCell
    (fun time => cell (query (first time) (rest time)).state) neighbors budget
  have selected := coveredScore objective constant regular (first time) (rest time)
    (comparison time) reference candidateRadius (represented time) (accurate time) (covered time)
  have width := retained time repeated
  have floor := recommended time
  dsimp [lower] at selected
  linarith

/-- A source-shaped cached row with the proved geometric error envelope. -/
def sourceRow {State : Type*} {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (state : State) (constant radius noise : ℝ) : Scored State :=
  ⟨state, Neighbor.mean data params scale distances nearest, noise + constant * radius⟩

theorem sourceValid {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (state : State)
    (constant : ℝ≥0) (radius noise : ℝ)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ noise)
    (near : ∀ position : Fin nearest.length, dist (points nearest[position.val]) state ≤ radius) :
    |(sourceRow data params scale distances nearest state constant radius noise).mean - objective state| ≤
      (sourceRow data params scale distances nearest state constant radius noise).error :=
  KnnLearning.sourceError data params scale distances nearest points objective state constant radius noise
    regular epistemic aleatoric variances dissimilarities inactive observed near

/-- The horizon theorem's recommendation premise follows from a streaming
minimum of observed scores, source observation provenance, convexity, and the
observation-error contract. It is not an assumed objective decrease. -/
theorem recommendation {State : Type*} {capacity : ℕ}
    (objective : State → ℝ) (first : Scored State) (rest : List (Scored State))
    (noise : ℝ) (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (common : ∀ row ∈ first :: rest, row.error = noise)
    (accurate : |(query first rest).mean - objective (query first rest).state| ≤ noise)
    (provenance : ∀ position : Fin nearest.length,
      ∃ row ∈ first :: rest, row.mean = data.values nearest[position.val])
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest) :
    objective (query first rest).state - noise ≤ Neighbor.mean data params scale distances nearest := by
  apply KnnLearning.sourceFloor data params scale distances nearest _
    epistemic aleatoric variances dissimilarities inactive
  intro position
  obtain ⟨row, member, value⟩ := provenance position
  have selected := queryOptimal first rest row member
  have selectedNoise := common (query first rest) (queryMember first rest)
  have rowNoise := common row member
  have error := (abs_le.mp accurate).1
  dsimp [lower] at selected
  rw [selectedNoise, rowNoise, value] at selected
  linarith

/-- A second cached source row uses the precision coefficient already derived
from fitted scales. It is not a user-chosen constant exploration multiplier. -/
def precisionRow {State : Type*} {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (state : State) (constant noise : ℝ) : Scored State :=
  ⟨state, Neighbor.mean data params scale distances nearest,
    Real.sqrt ((constant^2 / params.epistemic + noise^2 / params.aleatoric) * nearest.length) *
      Real.sqrt (1 / Accuracy.weightTotal data params scale distances nearest)⟩

theorem precisionValid {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (state : State)
    (constant : ℝ≥0) (noise : ℝ) (noiseNonneg : 0 ≤ noise)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 < params.epistemic) (aleatoric : 0 < params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (geometry : ∀ index, dist (points index) state ^ 2 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ noise) :
    |(precisionRow data params scale distances nearest state constant noise).mean - objective state| ≤
      (precisionRow data params scale distances nearest state constant noise).error :=
  Calibration.sourceConfidence data params scale distances nearest points objective state constant noise
    noiseNonneg regular epistemic aleatoric variances geometry inactive observed

#print axioms scanLaw
#print axioms queryOptimal
#print axioms comparisons
#print axioms scanAppend
#print axioms selectedGap
#print axioms localizedQueries
#print axioms stopBound
#print axioms horizonBound
#print axioms sourceValid
#print axioms recommendation
#print axioms precisionValid

end
end BauxResearch.Optimization.KnnDecision
