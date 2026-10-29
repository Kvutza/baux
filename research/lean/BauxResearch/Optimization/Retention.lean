import BauxResearch.Optimization.Neighbor
import BauxResearch.Optimization.Envelope
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Topology.MetricSpace.CoveringNumbers

/-!
# Information retained by an observation archive

Mathlib's distance to a set supplies explicit indistinguishable Lipschitz
objectives. The resulting lower bound applies to every predictor using those
observations, including a fitted ENN predictor. It is an information bound for
the stated metric class, not a claim that pretraining has this worst case.

The final counterexample uses the actual `Neighbor.compact` map: two valid
archives with equal incumbents become identical while their other observations
differ. This rules out unrestricted decision sufficiency of that archive map.
It does not identify the archive with every item of state in the Rust optimizer.
-/

namespace BauxResearch.Optimization.Retention

open Metric
open scoped NNReal

noncomputable section

variable {State : Type*} [PseudoMetricSpace State]

/-- A distance-to-archive objective is zero at every retained state. -/
def unseen (archive : Set State) (constant : ℝ≥0) (query : State) : ℝ :=
  constant * infDist query archive

theorem unseenLip (archive : Set State) (constant : ℝ≥0) :
    LipschitzWith constant (unseen archive constant) := by
  apply LipschitzWith.of_le_add_mul
  intro first second
  have bound := mul_le_mul_of_nonneg_left
    (infDist_le_infDist_add_dist (x := first) (y := second) (s := archive))
    constant.coe_nonneg
  simpa [unseen, mul_add] using bound

theorem unseenZero (archive : Set State) (constant : ℝ≥0) (query : State)
    (member : query ∈ archive) : unseen archive constant query = 0 := by
  simp [unseen, infDist_zero_of_mem member]

/-- Even exact observations leave this much worst-case prediction error.
The two explicit objectives agree on the entire archive. -/
theorem predictionLower (archive : Set State) (constant : ℝ≥0)
    (query : State) (prediction : ℝ) :
    ∃ objective : State → ℝ,
      LipschitzWith constant objective ∧
      (∀ point ∈ archive, objective point = 0) ∧
      constant * infDist query archive ≤ |prediction - objective query| := by
  by_cases positive : 0 ≤ prediction
  · refine ⟨fun point => -unseen archive constant point, (unseenLip archive constant).neg,
      fun point member => by simp [unseenZero archive constant point member], ?_⟩
    have distance := infDist_nonneg (x := query) (s := archive)
    have value : 0 ≤ unseen archive constant query := mul_nonneg constant.coe_nonneg distance
    rw [sub_neg_eq_add, abs_of_nonneg (add_nonneg positive value)]
    dsimp [unseen]
    linarith
  · refine ⟨unseen archive constant, unseenLip archive constant,
      fun point member => unseenZero archive constant point member, ?_⟩
    have distance := infDist_nonneg (x := query) (s := archive)
    have value : 0 ≤ unseen archive constant query := mul_nonneg constant.coe_nonneg distance
    rw [abs_of_nonpos (sub_nonpos.mpr ((le_of_not_ge positive).trans value))]
    dsimp [unseen]
    linarith

/-- A claimed uniform error over this metric class forces a geometric memory
requirement. No sufficient statistic is assumed from the archive's size. -/
theorem necessaryRadius (archive : Set State) (constant : ℝ≥0)
    (query : State) (prediction error : ℝ)
    (accurate : ∀ objective : State → ℝ, LipschitzWith constant objective →
      (∀ point ∈ archive, objective point = 0) → |prediction - objective query| ≤ error) :
    constant * infDist query archive ≤ error := by
  obtain ⟨objective, regular, observed, lower⟩ := predictionLower archive constant query prediction
  exact lower.trans (accurate objective regular observed)

/-- Conversely, agreement within `2*noise` on a finite archive bounds the
disagreement of two compatible objectives at the query. -/
theorem ambiguityBound (archive : Finset State) (nonempty : archive.Nonempty)
    (constant : ℝ≥0) (first second : State → ℝ) (query : State) (noise : ℝ)
    (firstLip : LipschitzWith constant first) (secondLip : LipschitzWith constant second)
    (compatible : ∀ point ∈ archive, |first point - second point| ≤ 2 * noise) :
    |first query - second query| ≤
      2 * noise + 2 * constant * infDist query (archive : Set State) := by
  obtain ⟨point, member, nearest⟩ := archive.finite_toSet.isCompact.exists_infDist_eq_dist
    (by simpa using nonempty) query
  have left := firstLip.dist_le_mul query point
  have right := secondLip.dist_le_mul point query
  rw [Real.dist_eq] at left right
  have middle := compatible point member
  have bound := (abs_sub_le (first query) (first point) (second query)).trans
    (add_le_add le_rfl (abs_sub_le (first point) (second point) (second query)))
  rw [dist_comm point query] at right
  rw [nearest]
  linarith

/-- Retaining a metric cover of previous observations controls the extra
distance introduced by forgetting, without referring to a parameter dimension. -/
theorem radiusTransfer (full retained : Set State) (fullNonempty : full.Nonempty)
    (query : State) (radius : ℝ)
    (covered : ∀ point ∈ full, ∃ anchor ∈ retained, dist point anchor ≤ radius) :
    infDist query retained ≤ infDist query full + radius := by
  apply le_of_forall_pos_le_add
  intro epsilon positive
  obtain ⟨point, member, near⟩ := (infDist_lt_iff fullNonempty).mp
    (lt_add_of_pos_right (infDist query full) positive)
  obtain ⟨anchor, kept, close⟩ := covered point member
  have upper := (infDist_le_dist_of_mem kept (x := query)).trans (dist_triangle query point anchor)
  linarith

/-- The source weighted mean is exactly zero when all stored outcomes are
zero; fitted parameters and the denominator floor cannot reveal an unseen value. -/
theorem sourceZero {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ)
    (distances : BauxResearch.Forward.Vector capacity) (nearest : List (Fin capacity))
    (zero : ∀ index, data.values index = 0) :
    Neighbor.mean data params scale distances nearest = 0 := by
  simp [Neighbor.mean, zero]

/-- Calibrate the source real-valued mean against the exact information in
valid interval observations. Fitted weights and neighbor selection may depend
on responses: this deterministic bound does not require their independence.
It does not identify the fitted ENN standard deviation with this radius. -/
theorem sourceBound {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ)
    (distances : BauxResearch.Forward.Vector capacity) (nearest : List (Fin capacity))
    (records : Finset (Fin capacity)) (nonempty : records.Nonempty)
    (point : Fin capacity → State) (lower upper : Fin capacity → ℝ) (constant : ℝ≥0)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : Envelope.fits records point lower upper objective) (query : State) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      max |Neighbor.mean data params scale distances nearest -
        Envelope.below records nonempty point lower constant query|
        |Neighbor.mean data params scale distances nearest -
        Envelope.above records nonempty point upper constant query| :=
  Envelope.estimateBound records nonempty point lower upper constant objective regular
    observed query (Neighbor.mean data params scale distances nearest)

/-- A source-shaped archive with one incumbent and one rejected observation. -/
def archivePair (value : ℝ) : Neighbor.Data 2 where
  values := fun index => if index.val = 1 then value else 0
  variances := fun _ => 0
  distances := fun left right => if left = right then 0 else 1
  count := 2
  base := 0

theorem compactSame (first second : ℝ) :
    Neighbor.compact (by decide) (archivePair first) =
      Neighbor.compact (by decide) (archivePair second) := by
  simp [Neighbor.compact, archivePair]

/-- Any decision that only sees the compacted archive fails to recover one
of two distinguishable observations. Both forgotten values exceed the incumbent. -/
theorem compactLoss (decide : Neighbor.Data 2 → Bool) :
    ∃ world : Bool,
      decide (Neighbor.compact (by decide) (archivePair (if world then 2 else 1))) ≠ world := by
  have same := congrArg decide (compactSame 2 1)
  cases outcome : decide (Neighbor.compact (by decide) (archivePair 1))
  · exact ⟨true, by simpa [same] using outcome⟩
  · exact ⟨false, by simp [outcome]⟩

/-- Values on four scalar parameter states. Both worlds reject the observation
at 1 relative to incumbent 0, but their best next updates have opposite identities. -/
def finiteObjective (world : Bool) (state : ℝ) : ℝ :=
  if state = 0 then 0 else if state = 1 then (if world then 2 else 1)
  else if state = 2 then (if world then 1 else -1) else (if world then -1 else 1)

/-- Mathlib extends the finite table to an ordinary Lipschitz objective.
The obstruction therefore occurs inside a regular metric objective class. -/
theorem objectiveExists (world : Bool) :
    ∃ objective : ℝ → ℝ, LipschitzWith 3 objective ∧
      objective 0 = 0 ∧ objective 1 = (if world then 2 else 1) ∧
      objective 2 = (if world then 1 else -1) ∧ objective 3 = (if world then -1 else 1) := by
  have regular : LipschitzOnWith 3 (finiteObjective world) ({0, 1, 2, 3} : Set ℝ) := by
    apply LipschitzOnWith.of_dist_le_mul
    intro first included second included'
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at included included'
    rcases included with rfl | rfl | rfl | rfl <;>
      rcases included' with rfl | rfl | rfl | rfl <;>
      cases world <;> norm_num [finiteObjective, Real.dist_eq]
  obtain ⟨objective, extension, agrees⟩ := regular.extend_real
  refine ⟨objective, extension, ?_, ?_, ?_, ?_⟩
  · simpa [finiteObjective] using (agrees (by simp : (0 : ℝ) ∈ ({0, 1, 2, 3} : Set ℝ))).symm
  · simpa [finiteObjective] using (agrees (by simp : (1 : ℝ) ∈ ({0, 1, 2, 3} : Set ℝ))).symm
  · simpa [finiteObjective] using (agrees (by simp : (2 : ℝ) ∈ ({0, 1, 2, 3} : Set ℝ))).symm
  · simpa [finiteObjective] using (agrees (by simp : (3 : ℝ) ∈ ({0, 1, 2, 3} : Set ℝ))).symm

/-- The forgotten observation identifies the better of the two next updates. -/
def decode (data : Neighbor.Data 2) : Bool := decide (data.values 1 = 2)

theorem decoded (world : Bool) : decode (archivePair (if world then 2 else 1)) = world := by
  cases world <;> norm_num [decode, archivePair]

/-- Every rule using only the actual compacted archive incurs a gap of 2
on one of two Lipschitz worlds, measured against those same two feasible updates.
Full-archive decoding distinguishes them. This is a surrogate-memory result,
not a lower bound for a whole optimizer that may retain other information. -/
theorem compactChoice (choose : Neighbor.Data 2 → Bool) :
    ∃ world : Bool, ∃ objective : ℝ → ℝ,
      LipschitzWith 3 objective ∧ objective 0 = 0 ∧
      objective 1 = (if world then 2 else 1) ∧
      objective (if choose (Neighbor.compact (by decide)
        (archivePair (if world then 2 else 1))) then 3 else 2) -
        min (objective 2) (objective 3) = 2 := by
  obtain ⟨world, wrong⟩ := compactLoss choose
  obtain ⟨objective, regular, zero, observed, left, right⟩ := objectiveExists world
  refine ⟨world, objective, regular, zero, observed, ?_⟩
  generalize selected : choose (Neighbor.compact (by decide)
    (archivePair (if world then 2 else 1))) = action at wrong ⊢
  cases world <;> cases action <;> norm_num at * <;> norm_num [left, right]

#print axioms predictionLower
#print axioms necessaryRadius
#print axioms ambiguityBound
#print axioms radiusTransfer
#print axioms sourceBound
#print axioms compactLoss
#print axioms objectiveExists
#print axioms compactChoice

end
end BauxResearch.Optimization.Retention
