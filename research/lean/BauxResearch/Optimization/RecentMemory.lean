import BauxResearch.Optimization.SampledEnn
import Mathlib.Order.Interval.Finset.Nat

/-!
# Recent observations under incumbent-protected oldest-first eviction

The implicit archive in `resident_metal/history.rs::absolute_objectives` replaces
the smallest observation identity other than the incumbent when capacity is
full. This module models that ID-set operation, not a new retention policy.
Sequential natural IDs normalize the source's initial identity offset.

The resulting trace retains the most recent `capacity - 1` observations for
EVERY sequence of protected identities. This discharges a temporal recall
property without a metric cover, random sampling, or an assumed useful archive.
It does not assert that recent states have useful objective values. Slot/sample
coherence, source integer arithmetic, and geometric/observation errors remain
separate implementation and analytical obligations.
The trace contains ordinary insertions only; explicit archive compaction,
reset, or cross-region activation needs its own transition/refinement account.
-/

namespace BauxResearch.Optimization.RecentMemory

open scoped BigOperators NNReal

/-- Remove the oldest ID other than the protected incumbent, if one exists. -/
def evict (history : Finset ℕ) (base : ℕ) : Finset ℕ :=
  if available : (history.erase base).Nonempty then
    history.erase ((history.erase base).min' available)
  else history

/-- Source-shaped logical insertion: evict only at full capacity, then record
the new observation whether or not its parameter update was accepted. -/
def record (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ) : Finset ℕ :=
  insert fresh (if history.card = capacity then evict history base else history)

theorem evictSubset (history : Finset ℕ) (base : ℕ) :
    evict history base ⊆ history := by
  unfold evict
  split
  · exact Finset.erase_subset _ _
  · exact Finset.Subset.refl _

theorem evictBase (history : Finset ℕ) (base : ℕ) (member : base ∈ history) :
    base ∈ evict history base := by
  unfold evict
  split
  next available =>
    exact Finset.mem_erase.mpr
      ⟨Ne.symm (Finset.mem_erase.mp (Finset.min'_mem _ available)).1, member⟩
  next => exact member

private theorem fullNonempty (capacity : ℕ) (history : Finset ℕ) (base : ℕ)
    (two : 2 ≤ capacity) (full : history.card = capacity) :
    (history.erase base).Nonempty := by
  apply Finset.card_pos.mp
  have lower := Finset.pred_card_le_card_erase (s := history) (a := base)
  rw [full] at lower
  omega

/-- Finite counting forces the evicted ID to be outside the newest
`capacity - 2` old observations. No probabilistic premise is involved. -/
theorem oldestAge (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ)
    (full : history.card = capacity) (older : ∀ item ∈ history, item < fresh)
    (available : (history.erase base).Nonempty) :
    (history.erase base).min' available + capacity ≤ fresh + 1 := by
  have subset : history.erase base ⊆
      Finset.Ico ((history.erase base).min' available) fresh := by
    intro item member
    exact Finset.mem_Ico.mpr
      ⟨Finset.min'_le _ item member, older item (Finset.mem_of_mem_erase member)⟩
  have upper := Finset.card_le_card subset
  rw [Nat.card_Ico] at upper
  have lower := Finset.pred_card_le_card_erase (s := history) (a := base)
  rw [full] at lower
  have age := older _ (Finset.mem_of_mem_erase (Finset.min'_mem _ available))
  omega

/-- The logical archive never exceeds its declared capacity, including when
the supplied protected identity is absent. Real source states protect a member. -/
theorem recordCapacity (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ)
    (two : 2 ≤ capacity) (bounded : history.card ≤ capacity) :
    (record capacity history base fresh).card ≤ capacity := by
  unfold record
  split
  next full =>
    have available := fullNonempty capacity history base two full
    have member := Finset.mem_of_mem_erase (Finset.min'_mem _ available)
    have size := Finset.card_erase_add_one member
    have upper := Finset.card_insert_le fresh (evict history base)
    simp only [evict, dif_pos available] at upper
    simp only [evict, dif_pos available]
    omega
  next room =>
    have upper := Finset.card_insert_le fresh history
    omega

theorem recordFresh (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ) :
    fresh ∈ record capacity history base fresh := by
  simp [record]

theorem recordBase (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ)
    (member : base ∈ history) : base ∈ record capacity history base fresh := by
  unfold record
  apply Finset.mem_insert_of_mem
  split
  · exact evictBase history base member
  · exact member

theorem recordAges (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ)
    (older : ∀ item ∈ history, item < fresh) :
    ∀ item ∈ record capacity history base fresh, item < fresh + 1 := by
  intro item member
  obtain same | kept := Finset.mem_insert.mp member
  · omega
  · have old : item ∈ history := by
      split at kept
      · exact evictSubset history base kept
      · exact kept
    have age := older item old
    omega

/-- One source-shaped insertion preserves temporal recall. The premise is an
inductive invariant of the complete trace below, not an assumed metric cover. -/
theorem recordRecent (capacity : ℕ) (history : Finset ℕ) (base fresh : ℕ)
    (two : 2 ≤ capacity) (older : ∀ item ∈ history, item < fresh)
    (recent : ∀ item, item < fresh → fresh + 1 ≤ item + capacity → item ∈ history) :
    ∀ item, item < fresh + 1 → fresh + 2 ≤ item + capacity →
      item ∈ record capacity history base fresh := by
  intro item before close
  unfold record
  by_cases same : item = fresh
  · exact Finset.mem_insert.mpr (Or.inl same)
  · apply Finset.mem_insert_of_mem
    have kept := recent item (by omega) (by omega)
    split
    next full =>
      have available := fullNonempty capacity history base two full
      have age := oldestAge capacity history base fresh full older available
      have different : item ≠ (history.erase base).min' available := by omega
      simpa only [evict, dif_pos available] using Finset.mem_erase.mpr ⟨different, kept⟩
    next => exact kept

/-- IDs enumerate all observed trials, not just accepted updates. The protected
ID can vary arbitrarily with time; accepted/rejected feedback can choose it. -/
def trace (capacity : ℕ) (base : ℕ → ℕ) : ℕ → Finset ℕ
  | 0 => ∅
  | time + 1 => record capacity (trace capacity base time) (base time) time

/-- Derive bounded storage, sequential ages, and recent recall jointly for
every finite horizon, without assuming any of them along the trajectory. -/
theorem traceFacts (capacity : ℕ) (base : ℕ → ℕ) (two : 2 ≤ capacity) (time : ℕ) :
    (trace capacity base time).card ≤ capacity ∧
      (∀ item ∈ trace capacity base time, item < time) ∧
      (∀ item, item < time → time + 1 ≤ item + capacity →
        item ∈ trace capacity base time) := by
  induction time with
  | zero => simp [trace]
  | succ time induction =>
      exact ⟨recordCapacity capacity _ _ _ two induction.1,
        recordAges capacity _ _ _ induction.2.1,
        recordRecent capacity _ _ _ two induction.2.1 induction.2.2⟩

/-- Any k strictly below capacity has all its latest observation IDs retained.
This is stronger than retention of just the accepted incumbent. -/
theorem traceRecall (capacity neighbors : ℕ) (base : ℕ → ℕ)
    (two : 2 ≤ capacity) (room : neighbors < capacity) (time : ℕ) :
    Finset.Ico (time - neighbors) time ⊆ trace capacity base time := by
  intro item member
  obtain ⟨lower, before⟩ := Finset.mem_Ico.mp member
  exact (traceFacts capacity base two time).2.2 item before (by omega)

/-- After k observations, the actual temporal window supplies k distinct IDs;
no independence or distinct-parameter-state assumption is necessary. -/
theorem traceCount (capacity neighbors : ℕ) (base : ℕ → ℕ)
    (two : 2 ≤ capacity) (room : neighbors < capacity) (time : ℕ)
    (enough : neighbors ≤ time) :
    neighbors ≤ ((trace capacity base time).filter fun item => time - neighbors ≤ item).card := by
  have subset : Finset.Ico (time - neighbors) time ⊆
      (trace capacity base time).filter (fun item => time - neighbors ≤ item) := by
    intro item member
    exact Finset.mem_filter.mpr
      ⟨traceRecall capacity neighbors base two room time member, (Finset.mem_Ico.mp member).1⟩
  have count := Finset.card_le_card subset
  rw [Nat.card_Ico] at count
  omega

/-- Persistent old incumbents use one slot without losing the recent window. -/
example : trace 4 (fun _ => 0) 8 = {0, 5, 6, 7} := by decide

/-- Accepting every preceding observation also preserves bounded recent recall. -/
example : trace 4 (fun time => time - 1) 8 = {4, 5, 6, 7} := by decide

example : trace 2 (fun _ => 0) 8 = {0, 7} := by decide

section Metric

variable {State : Type*} [PseudoMetricSpace State]

/-- Recent trials remain geometrically local when the REALIZED trial
displacements are bounded and the incumbent only stays or accepts a trial.
`radius` is a distance bound in one fixed metric, NOT the TOML radius coefficient.
Mathlib's polygon inequality accounts for all intervening accepted movements. -/
theorem recentRadius (incumbent trial : ℕ → State) (time neighbors : ℕ) (radius : ℝ)
    (nonnegative : 0 ≤ radius)
    (commits : ∀ index, time - neighbors ≤ index → index < time →
      incumbent (index + 1) = incumbent index ∨ incumbent (index + 1) = trial index)
    (near : ∀ index, time - neighbors ≤ index → index ≤ time →
      dist (trial index) (incumbent index) ≤ radius)
    (index : ℕ) (member : index ∈ Finset.Ico (time - neighbors) time) :
    dist (trial index) (trial time) ≤ ((neighbors : ℝ) + 2) * radius := by
  obtain ⟨lower, before⟩ := Finset.mem_Ico.mp member
  have steps : ∀ {step}, index ≤ step → step < time →
      dist (incumbent step) (incumbent (step + 1)) ≤ radius := by
    intro step after earlier
    rcases commits step (by omega) earlier with stayed | accepted
    · rw [stayed, dist_self]
      exact nonnegative
    · rw [accepted, dist_comm]
      exact near step (by omega) (by omega)
  have chain := dist_le_Ico_sum_of_dist_le (f := incumbent)
    (d := fun _ => radius) (Nat.le_of_lt before) steps
  simp only [Finset.sum_const, Nat.card_Ico, nsmul_eq_mul] at chain
  have count : ((time - index : ℕ) : ℝ) ≤ (neighbors : ℝ) := by
    exact_mod_cast (show time - index ≤ neighbors by omega)
  have movement := chain.trans (mul_le_mul_of_nonneg_right count nonnegative)
  have first := near index lower (Nat.le_of_lt before)
  have last := near time (by omega) le_rfl
  rw [dist_comm] at last
  calc
    dist (trial index) (trial time) ≤
        dist (trial index) (incumbent index) + dist (incumbent index) (trial time) :=
      dist_triangle _ _ _
    _ ≤ dist (trial index) (incumbent index) +
        (dist (incumbent index) (incumbent time) + dist (incumbent time) (trial time)) :=
      add_le_add le_rfl (dist_triangle _ _ _)
    _ ≤ ((neighbors : ℝ) + 2) * radius := by linarith

/-- Bound the actual clamped source KNN mean using recent retention, realized
step bounds, and approximate squared distances. The k-close-observations premise
is DERIVED from the source-shaped trace rather than assumed from a metric cover.

The remaining hypotheses are explicit: faithful logical slots/parameter states,
one fixed-metric Lipschitz objective, observation error, sampled-distance error,
valid fitted scales, and an inactive denominator floor. This does not establish
those hypotheses for corpus NLL or prove access to a useful perturbation. -/
theorem sourceError {capacity : ℕ} (memory : ℕ) (base : ℕ → ℕ)
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : BauxResearch.Forward.Vector capacity)
    (slot : ℕ → Fin capacity) (points : Fin capacity → State)
    (incumbent trial : ℕ → State) (time : ℕ) (radius geometryError noise : ℝ)
    (objective : State → ℝ) (constant : ℝ≥0)
    (two : 2 ≤ memory) (room : params.neighbors < memory) (enough : params.neighbors ≤ time)
    (unique : Set.InjOn slot (trace memory base time : Set ℕ))
    (resident : ∀ item ∈ trace memory base time, (slot item).val < data.count)
    (coherent : ∀ item ∈ trace memory base time, points (slot item) = trial item)
    (nonnegative : 0 ≤ radius)
    (commits : ∀ index, time - params.neighbors ≤ index → index < time →
      incumbent (index + 1) = incumbent index ∨ incumbent (index + 1) = trial index)
    (near : ∀ index, time - params.neighbors ≤ index → index ≤ time →
      dist (trial index) (incumbent index) ≤ radius)
    (regular : LipschitzWith constant objective)
    (approximation : ∀ index, index.val < data.count →
      |distances index - dist (points index) (trial time)^2| ≤ geometryError)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances
      (Neighbor.ordered data params distances none))
    (observed : ∀ position : Fin (Neighbor.ordered data params distances none).length,
      |data.values (Neighbor.ordered data params distances none)[position.val] -
        objective (points (Neighbor.ordered data params distances none)[position.val])| ≤ noise) :
    |Neighbor.mean data params scale distances (Neighbor.ordered data params distances none) -
      objective (trial time)| ≤ noise + constant *
        Real.sqrt ((((params.neighbors : ℝ) + 2) * radius)^2 + 2 * geometryError) := by
  have retained := traceRecall memory params.neighbors base two room time
  have count : ((Finset.Ico (time - params.neighbors) time).image slot).card =
      params.neighbors := by
    rw [Finset.card_image_of_injOn]
    · rw [Nat.card_Ico]
      omega
    · intro first firstIn second secondIn same
      exact unique (retained firstIn) (retained secondIn) same
  apply SampledEnn.distortedError data params scale distances none points objective
    (trial time) constant (((params.neighbors : ℝ) + 2) * radius) geometryError noise
    (by positivity) regular approximation _ epistemic aleatoric variances dissimilarities
    inactive observed
  apply le_trans count.symm.le
  apply Finset.card_le_card
  intro index member
  obtain ⟨item, included, rfl⟩ := Finset.mem_image.mp member
  refine Finset.mem_filter.mpr
    ⟨Finset.mem_univ _, resident item (retained included), by simp, ?_⟩
  rw [coherent item (retained included)]
  exact recentRadius incumbent trial time params.neighbors radius nonnegative commits near item included

end Metric

#print axioms oldestAge
#print axioms recordCapacity
#print axioms recordBase
#print axioms traceFacts
#print axioms traceRecall
#print axioms traceCount
#print axioms recentRadius
#print axioms sourceError

end BauxResearch.Optimization.RecentMemory
