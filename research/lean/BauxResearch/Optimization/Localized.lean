import Mathlib.Topology.MetricSpace.CoveringNumbers
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Tactic.Linarith

/-!
# Localized counts for optimistic queries

One query is one objective evaluation, not one accepted update. The counting
argument is pathwise: it permits adaptive queries and does not assume that
individual evaluations improve the incumbent. Statistical validity of the ENN
error bounds is a separate obligation.

The finite-count argument charges the first `k` visits to a cell once, then
charges subsequent unresolved visits to explicit recall failures. Localization
must follow from optimism and accuracy; it is not an assumed small search space.
-/

namespace BauxResearch.Optimization.Localized

open scoped BigOperators NNReal

variable {horizon : ℕ} {Cell : Type*} [DecidableEq Cell]

/-- Earlier queries in the selected set and the same cell. The set can be an
uncertainty band; observations outside it are deliberately not needed here. -/
def earlier (queries : Finset (Fin horizon)) (cell : Fin horizon → Cell)
    (time : Fin horizon) : Finset (Fin horizon) :=
  queries.filter fun previous => previous < time ∧ cell previous = cell time

/-- Within a cell, an additional queried time strictly increases its visit
rank. This remains true for arbitrarily chosen subsequences of evaluations. -/
theorem rankStrict (queries : Finset (Fin horizon)) (cell : Fin horizon → Cell)
    (left right : Fin horizon) (member : left ∈ queries) (order : left < right)
    (same : cell left = cell right) :
    (earlier queries cell left).card < (earlier queries cell right).card := by
  have absent : left ∉ earlier queries cell left := by simp [earlier]
  have subset : insert left (earlier queries cell left) ⊆ earlier queries cell right := by
    intro time inside
    rcases Finset.mem_insert.mp inside with rfl | inside
    · exact Finset.mem_filter.mpr ⟨member, order, same⟩
    · obtain ⟨member, before, equal⟩ := Finset.mem_filter.mp inside
      exact Finset.mem_filter.mpr ⟨member, before.trans order, equal.trans same⟩
  have count := Finset.card_le_card subset
  rw [Finset.card_insert_of_notMem absent] at count
  omega

/-- There are at most `k` initial visits per cell. The injection is the pair
consisting of the cell and the number of earlier selected visits to that cell. -/
theorem earlyCount [Fintype Cell] (queries : Finset (Fin horizon))
    (cell : Fin horizon → Cell) (k : ℕ) :
    (queries.filter fun time => (earlier queries cell time).card < k).card ≤
      Fintype.card Cell * k := by
  let initial := queries.filter fun time => (earlier queries cell time).card < k
  let key := fun time => (cell time, (earlier queries cell time).card)
  have image : initial.image key ⊆ Finset.univ ×ˢ Finset.range k := by
    intro pair inside
    obtain ⟨time, member, rfl⟩ := Finset.mem_image.mp inside
    exact Finset.mem_product.mpr ⟨Finset.mem_univ _,
      Finset.mem_range.mpr (Finset.mem_filter.mp member).2⟩
  have injective : Set.InjOn key initial := by
    intro left leftIn right rightIn equal
    have same : cell left = cell right := congrArg Prod.fst equal
    have ranks : (earlier queries cell left).card = (earlier queries cell right).card :=
      congrArg Prod.snd equal
    rcases lt_trichotomy left right with order | equal | order
    · have strict := rankStrict queries cell left right
        (Finset.mem_filter.mp leftIn).1 order same
      omega
    · exact equal
    · have strict := rankStrict queries cell right left
        (Finset.mem_filter.mp rightIn).1 order same.symm
      omega
  have count := Finset.card_le_card image
  rw [Finset.card_image_of_injOn injective] at count
  simpa [initial] using count

/-- Every visit after the initial allowance is counted explicitly. No memory
policy is silently assumed to preserve information gathered in earlier rounds. -/
theorem visitCount [Fintype Cell] (queries : Finset (Fin horizon))
    (cell : Fin horizon → Cell) (k : ℕ) :
    queries.card ≤ Fintype.card Cell * k +
      (queries.filter fun time => k ≤ (earlier queries cell time).card).card := by
  have initial := earlyCount queries cell k
  have split := Finset.card_filter_add_card_filter_not (s := queries)
    (fun time => (earlier queries cell time).card < k)
  simp only [not_lt] at split
  omega

/-- An optimistic query with a valid error bound lies in the corresponding
near-optimal level set. `reference` is a proof-side comparison value; `slack`
accounts for acquisition approximation and comparator coverage. -/
theorem queryNear (value mean error reference slack width : ℝ)
    (accurate : |mean - value| ≤ error)
    (optimistic : mean - error ≤ reference + slack)
    (bounded : error ≤ width) :
    value - reference ≤ 2 * width + slack := by
  have lower := (abs_le.mp accurate).1
  linarith

/-- At a fixed uncertainty scale, a cell can require at most `k` selected
queries unless recall fails there. `forgotten` can also include retrieval or
distance failures; it does not mean that all such failures are literal eviction.
The recall premise concerns actual available information, not improvement. -/
theorem bandCount [Fintype Cell] (queries forgotten : Finset (Fin horizon))
    (cell : Fin horizon → Cell) (k : ℕ) (error : Fin horizon → ℝ) (width : ℝ)
    (unresolved : ∀ time ∈ queries, width / 2 < error time)
    (retained : ∀ time ∈ queries, time ∉ forgotten →
      k ≤ (earlier queries cell time).card → error time ≤ width / 2) :
    queries.card ≤ Fintype.card Cell * k + (queries ∩ forgotten).card := by
  have subset : (queries.filter fun time => k ≤ (earlier queries cell time).card) ⊆
      queries ∩ forgotten := by
    intro time inside
    obtain ⟨member, count⟩ := Finset.mem_filter.mp inside
    have lost : time ∈ forgotten := by
      by_contra absent
      exact (unresolved time member).not_ge (retained time member absent count)
    exact Finset.mem_inter.mpr ⟨member, lost⟩
  exact (visitCount queries cell k).trans
    (Nat.add_le_add_left (Finset.card_le_card subset) _)

section Metric

variable {State : Type*} [PseudoMetricSpace State]

/-- Earlier selected evaluations within a metric radius of this query. This
counts observation identities, so repeated evaluations of one state count. -/
noncomputable def nearby (queries : Finset (Fin horizon))
    (point : Fin horizon → State) (radius : ℝ) (time : Fin horizon) :
    Finset (Fin horizon) := by
  classical
  exact queries.filter fun previous =>
    previous < time ∧ dist (point previous) (point time) ≤ radius

/-- Times at which enough nearby earlier queries exist but fewer than `k`
are available in memory. This is an explicit discrepancy between the complete
selected trace and available observations, not a free error allowance. -/
noncomputable def recallLoss (queries : Finset (Fin horizon))
    (memory : Fin horizon → Finset (Fin horizon)) (point : Fin horizon → State)
    (k : ℕ) (radius : ℝ) : Finset (Fin horizon) := by
  classical
  exact queries.filter fun time =>
    k ≤ (nearby queries point radius time).card ∧
      (nearby (memory time) point radius time).card < k

/-- Boundary test: with no retained observations, repeated evaluations of one
state incur a recall loss on every visit after the first `k`. Even a one-point
query region cannot make forgetting harmless. The count is `horizon - k`, with
natural subtraction, and is not bounded by the number of stored observations. -/
theorem forgottenCount (state : State) (k : ℕ) (positive : 0 < k)
    (radius : ℝ) (nonneg : 0 ≤ radius) :
    (recallLoss (Finset.univ : Finset (Fin horizon)) (fun _ => ∅)
      (fun _ => state) k radius).card = horizon - k := by
  classical
  have initial : ∀ time : Fin horizon,
      nearby Finset.univ (fun _ => state) radius time = Finset.Iio time := by
    intro time
    ext previous
    simp [nearby, nonneg]
  have empty : ∀ time : Fin horizon, nearby ∅ (fun _ => state) radius time = ∅ := by
    intro time
    simp [nearby]
  have losses : recallLoss (Finset.univ : Finset (Fin horizon)) (fun _ => ∅)
      (fun _ => state) k radius = Finset.univ.filter (fun time => k ≤ time.val) := by
    ext time
    simp [recallLoss, initial, empty, positive, Fin.card_Iio]
  rw [losses]
  by_cases enough : k < horizon
  · have interval : (Finset.univ.filter fun time : Fin horizon => k ≤ time.val) =
        Finset.Ici ⟨k, enough⟩ := by
      ext time
      simp [Fin.le_def]
    rw [interval, Fin.card_Ici]
  · have none : (Finset.univ.filter fun time : Fin horizon => k ≤ time.val) = ∅ := by
      apply Finset.eq_empty_iff_forall_notMem.mpr
      intro time member
      have ordered := (Finset.mem_filter.mp member).2
      have before := time.isLt
      omega
    rw [none]
    simp only [Finset.card_empty]
    omega

/-- A finite cover of just the queried region bounds unresolved evaluations.
The available-memory accuracy premise can be supplied by KNN radius/error
results. Forgotten local observations are charged through `recallLoss`.
The centers are proof-side objects; this does not implement a covering oracle. -/
theorem coverCount (queries : Finset (Fin horizon)) (centers : Finset State)
    (memory : Fin horizon → Finset (Fin horizon)) (point : Fin horizon → State)
    (k : ℕ) (error : Fin horizon → ℝ) (width radius : ℝ)
    (unresolved : ∀ time ∈ queries, width / 2 < error time)
    (covered : ∀ time ∈ queries, ∃ center ∈ centers, dist (point time) center ≤ radius)
    (retained : ∀ time ∈ queries,
      k ≤ (nearby (memory time) point (2 * radius) time).card → error time ≤ width / 2) :
    queries.card ≤ centers.card * k +
      (recallLoss queries memory point k (2 * radius)).card := by
  classical
  by_cases empty : queries = ∅
  · simp [empty, recallLoss]
  obtain ⟨first, firstIn⟩ := Finset.nonempty_iff_ne_empty.mpr empty
  obtain ⟨base, baseIn, _⟩ := covered first firstIn
  let choose : ∀ time, time ∈ queries → {center // center ∈ centers} := fun time member =>
    ⟨(covered time member).choose, (covered time member).choose_spec.1⟩
  let cell := fun time => if member : time ∈ queries then choose time member else ⟨base, baseIn⟩
  have close : ∀ time ∈ queries, dist (point time) (cell time).val ≤ radius := by
    intro time member
    simpa [cell, member, choose] using (covered time member).choose_spec.2
  have inclusion : ∀ time ∈ queries,
      earlier queries cell time ⊆ nearby queries point (2 * radius) time := by
    intro time member previous inside
    obtain ⟨previousIn, before, same⟩ := Finset.mem_filter.mp inside
    have firstClose := close previous previousIn
    have nextClose := close time member
    rw [same] at firstClose
    apply Finset.mem_filter.mpr
    refine ⟨previousIn, before, ?_⟩
    have triangle := dist_triangle (point previous) (cell time).val (point time)
    rw [dist_comm (cell time).val (point time)] at triangle
    linarith
  have count := bandCount queries (recallLoss queries memory point k (2 * radius))
    cell k error width unresolved (by
      intro time member preserved enough
      apply retained time member
      by_contra insufficient
      apply preserved
      apply Finset.mem_filter.mpr
      exact ⟨member, enough.trans (Finset.card_le_card (inclusion time member)),
        Nat.lt_of_not_ge insufficient⟩)
  have subset : recallLoss queries memory point k (2 * radius) ⊆ queries :=
    Finset.filter_subset _ _
  simpa only [Fintype.card_coe, Finset.inter_eq_right.mpr subset] using count

/-- A level set relative to the declared comparison value. If that value is
the minimum, this is a near-optimal set; no minimizer is an optimizer input. -/
def nearSet (objective : State → ℝ) (reference tolerance : ℝ) : Set State :=
  {state | objective state - reference ≤ tolerance}

/-- Optimism derives the region to be covered. At uncertainty scale `width`,
only its `2*width + slack` level set is needed, rather than the whole domain.
Mathlib supplies an attaining finite cover and its cardinality. This cover is
used in the proof, not offered as a free executable search primitive.

The result is pathwise on valid source errors and optimism. It supplies neither
those statistical events nor a small covering number for pretraining losses. -/
theorem localizedCount (queries : Finset (Fin horizon))
    (memory : Fin horizon → Finset (Fin horizon)) (point : Fin horizon → State)
    (objective : State → ℝ) (mean error : Fin horizon → ℝ)
    (k : ℕ) (reference slack width : ℝ) (radius : ℝ≥0)
    (finite : Metric.coveringNumber radius
      (nearSet objective reference (2 * width + slack)) ≠ ⊤)
    (accurate : ∀ time ∈ queries, |mean time - objective (point time)| ≤ error time)
    (optimistic : ∀ time ∈ queries, mean time - error time ≤ reference + slack)
    (unresolved : ∀ time ∈ queries, width / 2 < error time)
    (bounded : ∀ time ∈ queries, error time ≤ width)
    (retained : ∀ time ∈ queries,
      k ≤ (nearby (memory time) point (2 * radius) time).card → error time ≤ width / 2) :
    queries.card ≤
      (Metric.coveringNumber radius (nearSet objective reference (2 * width + slack))).toNat * k +
        (recallLoss queries memory point k (2 * radius)).card := by
  classical
  let region := nearSet objective reference (2 * width + slack)
  have finiteCover := Metric.finite_minimalCover (ε := radius) (A := region)
  let centers := finiteCover.toFinset
  have size : centers.card = (Metric.coveringNumber radius region).toNat := by
    have equality : (centers.card : ℕ∞) = Metric.coveringNumber radius region := by
      rw [← finiteCover.encard_eq_coe_toFinset_card]
      exact Metric.encard_minimalCover finite
    simpa using congrArg ENat.toNat equality
  have covered : ∀ time ∈ queries,
      ∃ center ∈ centers, dist (point time) center ≤ radius := by
    intro time member
    have inside : point time ∈ region :=
      queryNear _ _ _ _ _ _ (accurate time member) (optimistic time member) (bounded time member)
    obtain ⟨center, centerIn, close⟩ := (Metric.isCover_minimalCover finite) inside
    refine ⟨center, finiteCover.mem_toFinset.mpr centerIn, ?_⟩
    exact_mod_cast (edist_le_coe.mp close)
  have count := coverCount queries centers memory point k error width radius
    unresolved covered retained
  rwa [size] at count

/-- The evaluations whose uncertainty belongs to one half-open scale interval.
Scales are mathematical analysis parameters, not additional objective queries. -/
noncomputable def band (queries : Finset (Fin horizon))
    (error : Fin horizon → ℝ) (width : ℝ) : Finset (Fin horizon) := by
  classical
  exact queries.filter fun time => width / 2 < error time ∧ error time ≤ width

/-- Sum the localized bounds over any finite family of uncertainty scales that
covers the selected evaluations. Overlapping scales are permitted and only make
the bound looser. The scale-coverage premise concerns uncertainty values, not
objective improvement, and must include the initial high-uncertainty queries. -/
theorem scaleCount {scales : ℕ} (queries : Finset (Fin horizon))
    (memory : Fin horizon → Finset (Fin horizon)) (point : Fin horizon → State)
    (objective : State → ℝ) (mean error : Fin horizon → ℝ)
    (k : Fin scales → ℕ) (width : Fin scales → ℝ) (radius : Fin scales → ℝ≥0)
    (reference slack : ℝ)
    (finite : ∀ scale, Metric.coveringNumber (radius scale)
      (nearSet objective reference (2 * width scale + slack)) ≠ ⊤)
    (accurate : ∀ time ∈ queries, |mean time - objective (point time)| ≤ error time)
    (optimistic : ∀ time ∈ queries, mean time - error time ≤ reference + slack)
    (covered : ∀ time ∈ queries,
      ∃ scale, width scale / 2 < error time ∧ error time ≤ width scale)
    (retained : ∀ scale, ∀ time ∈ band queries error (width scale),
      k scale ≤ (nearby (memory time) point (2 * radius scale) time).card →
        error time ≤ width scale / 2) :
    queries.card ≤ ∑ scale : Fin scales,
      ((Metric.coveringNumber (radius scale)
        (nearSet objective reference (2 * width scale + slack))).toNat * k scale +
          (recallLoss (band queries error (width scale)) memory point
            (k scale) (2 * radius scale)).card) := by
  classical
  have inclusion : queries ⊆ Finset.univ.biUnion (fun scale => band queries error (width scale)) := by
    intro time member
    obtain ⟨scale, within⟩ := covered time member
    exact Finset.mem_biUnion.mpr ⟨scale, Finset.mem_univ _,
      Finset.mem_filter.mpr ⟨member, within⟩⟩
  apply (Finset.card_le_card inclusion).trans
  apply Finset.card_biUnion_le.trans
  apply Finset.sum_le_sum
  intro scale _
  exact localizedCount (band queries error (width scale)) memory point objective mean error
    (k scale) reference slack (width scale) (radius scale) (finite scale)
    (fun time member => accurate time (Finset.mem_filter.mp member).1)
    (fun time member => optimistic time (Finset.mem_filter.mp member).1)
    (fun _ member => (Finset.mem_filter.mp member).2.1)
    (fun _ member => (Finset.mem_filter.mp member).2.2) (retained scale)

/-- A strict bound on the number of high-uncertainty evaluations forces at
least one accurate-scale query. This is a finite-horizon counting conclusion;
it does not assert improvement at each preceding evaluation. -/
theorem lowError (error : Fin horizon → ℝ) (tolerance : ℝ) (budget : ℕ)
    (count : (Finset.univ.filter fun time => tolerance < error time).card ≤ budget)
    (enough : budget < horizon) : ∃ time, error time ≤ tolerance := by
  classical
  by_contra absent
  push Not at absent
  have all : (Finset.univ.filter fun time => tolerance < error time) = Finset.univ := by
    ext time
    simp [absent time]
  rw [all] at count
  simp only [Finset.card_univ, Fintype.card_fin] at count
  omega

omit [PseudoMetricSpace State] in
/-- The high-uncertainty count yields a terminal objective bound when the
recommendation is no worse than each evaluated state up to `penalty`. For a
minimum noisy-observation recommendation, `penalty` must be justified from the
observation errors. A comparison value is a global optimum only when separately
proved to be one. Neither that optimum nor this counting argument is a runtime
oracle. -/
theorem terminalBound (point : Fin horizon → State) (objective : State → ℝ)
    (mean error : Fin horizon → ℝ) (reference slack tolerance finalLoss penalty : ℝ)
    (budget : ℕ)
    (count : (Finset.univ.filter fun time => tolerance < error time).card ≤ budget)
    (enough : budget < horizon)
    (accurate : ∀ time, |mean time - objective (point time)| ≤ error time)
    (optimistic : ∀ time, mean time - error time ≤ reference + slack)
    (recommended : ∀ time, finalLoss ≤ objective (point time) + penalty) :
    finalLoss - reference ≤ 2 * tolerance + slack + penalty := by
  obtain ⟨time, small⟩ := lowError error tolerance budget count enough
  have near := queryNear _ _ _ _ _ _ (accurate time) (optimistic time) small
  have final := recommended time
  linarith

end Metric

#print axioms rankStrict
#print axioms earlyCount
#print axioms visitCount
#print axioms queryNear
#print axioms bandCount
#print axioms forgottenCount
#print axioms coverCount
#print axioms localizedCount
#print axioms scaleCount
#print axioms lowError
#print axioms terminalBound

end BauxResearch.Optimization.Localized
