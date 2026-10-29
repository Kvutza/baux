import BauxResearch.Optimization.Accuracy
import BauxResearch.Optimization.LocalMemory
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Data.Finset.Max
import Mathlib.Data.Prod.Lex

/-!
# Source ENN error and a finite-horizon information bound

The source mean receives its accuracy from observation errors, metric radii,
and positivity of its actual inverse-denominator weights. Fitted parameters
may depend on those observations: no independent-weight variance identity is
used. The denominator floor is explicit.

An optimistic search then either has a small terminal objective gap or still
needs new local observations. Mathlib's pigeonhole principle bounds this need
by k times a finite cover size, provided the archive preserves local recall.
These are conditional results for a stated metric objective class. Neither
the source shadow dissimilarity nor incumbent-only compaction is silently
assigned the metric/recall property, and acquisition minimization is charged
its own selection gap.
-/

namespace BauxResearch.Optimization.KnnLearning

open BauxResearch.Forward
open scoped BigOperators NNReal

noncomputable section

variable {State : Type*} [PseudoMetricSpace State]

/-- A sorted k-prefix cannot contain a distant point if k eligible points
already lie within the radius. Mathlib supplies sortedness and finite counting. -/
theorem prefixRadius {Index : Type*} [DecidableEq Index]
    (indices : List Index) (distance : Index → ℝ) (neighbors : ℕ) (radius : ℝ)
    (sorted : indices.Pairwise (fun left right => distance left ≤ distance right))
    (many : neighbors ≤ (indices.toFinset.filter fun index => distance index ≤ radius).card) :
    ∀ index ∈ indices.take neighbors, distance index ≤ radius := by
  intro index chosen
  by_contra outside
  have far : radius < distance index := lt_of_not_ge outside
  have across := (List.pairwise_append.mp (show
      ((indices.take neighbors) ++ indices.drop neighbors).Pairwise
        (fun left right => distance left ≤ distance right) by
      simpa only [List.take_append_drop] using sorted)).2.2
  have subset : (indices.toFinset.filter fun other => distance other ≤ radius) ⊆
      (indices.take neighbors).toFinset.erase index := by
    intro other inside
    obtain ⟨member, close⟩ := Finset.mem_filter.mp inside
    have different : other ≠ index := by intro same; subst other; exact outside close
    have earlier : other ∈ indices.take neighbors := by
      by_contra absent
      have later : other ∈ indices.drop neighbors := by
        have both : other ∈ indices.take neighbors ++ indices.drop neighbors := by
          simpa only [List.take_append_drop] using (List.mem_toFinset.mp member)
        exact (List.mem_append.mp both).resolve_left absent
      exact (far.trans_le (across index chosen other later)).not_ge close
    exact Finset.mem_erase.mpr ⟨different, List.mem_toFinset.mpr earlier⟩
  have erased := Finset.card_erase_add_one (List.mem_toFinset.mpr chosen)
  have retained := Finset.card_le_card subset
  have prefixSize := (List.toFinset_card_le (indices.take neighbors)).trans
    (List.length_take_le neighbors indices)
  omega

/-- The actual source distance/index tie ordering is a lexicographic order.
Consequently its k-neighbor list inherits the radius bound from retained counts. -/
theorem sourceRadius {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (distances : Vector capacity)
    (excluded : Option (Fin capacity)) (radius : ℝ)
    (many : params.neighbors ≤ (Finset.univ.filter fun index : Fin capacity =>
      index.val < data.count ∧ some index ≠ excluded ∧ distances index ≤ radius).card) :
    ∀ index ∈ Neighbor.ordered data params distances excluded, distances index ≤ radius := by
  let eligible := (List.ofFn fun index : Fin capacity => index).filter
    (fun index => decide (index.val < data.count ∧ some index ≠ excluded))
  let key := fun index : Fin capacity => toLex (distances index, index.val)
  let compare := fun left right => decide (key left ≤ key right)
  have sorted := List.pairwise_mergeSort (le := compare)
    (fun left middle right first second => by
      simpa only [compare, decide_eq_true_eq] using
        (le_trans (of_decide_eq_true first) (of_decide_eq_true second) : key left ≤ key right))
    (fun left right => by simpa [compare] using (le_total (key left) (key right))) eligible
  have distancesSorted : (eligible.mergeSort compare).Pairwise
      (fun left right => distances left ≤ distances right) := by
    apply sorted.imp
    intro left right ordered
    have order : key left ≤ key right := of_decide_eq_true ordered
    exact Prod.Lex.monotone_fst (key left) (key right) order
  have many' : params.neighbors ≤
      ((eligible.mergeSort compare).toFinset.filter fun index => distances index ≤ radius).card := by
    have same : ((eligible.mergeSort compare).toFinset.filter fun index => distances index ≤ radius) =
        (Finset.univ.filter fun index : Fin capacity =>
          index.val < data.count ∧ some index ≠ excluded ∧ distances index ≤ radius) := by
      ext index
      simp [eligible, and_assoc]
    rwa [same]
  have bound := prefixRadius (eligible.mergeSort compare) distances params.neighbors radius
    distancesSorted many'
  simpa [Neighbor.ordered, eligible, compare, key, Prod.Lex.toLex_le_toLex] using bound

/-- Every selected source index is active and passes the exclusion rule. -/
theorem orderedMember {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (distances : Vector capacity)
    (excluded : Option (Fin capacity)) (index : Fin capacity)
    (selected : index ∈ Neighbor.ordered data params distances excluded) :
    index.val < data.count ∧ some index ≠ excluded := by
  dsimp only [Neighbor.ordered] at selected
  have eligible := List.mem_of_mem_take selected
  simpa using eligible

/-- Squared source dissimilarities that dominate metric distance turn the
actual sort's supplied-distance radius into a metric-neighborhood guarantee. -/
theorem metricRadius {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (distances : Vector capacity)
    (excluded : Option (Fin capacity)) (points : Fin capacity → State)
    (query : State) (radius : ℝ) (nonnegative : 0 ≤ radius)
    (geometry : ∀ index, dist (points index) query ^ 2 ≤ distances index)
    (many : params.neighbors ≤ (Finset.univ.filter fun index : Fin capacity =>
      index.val < data.count ∧ some index ≠ excluded ∧ distances index ≤ radius^2).card) :
    ∀ index ∈ Neighbor.ordered data params distances excluded,
      dist (points index) query ≤ radius := by
  intro index chosen
  exact (sq_le_sq₀ dist_nonneg nonnegative).mp
    ((geometry index).trans (sourceRadius data params distances excluded (radius^2) many index chosen))

/-- The constructed bounded statistic supplies the source sort's squared-radius
premise once a cell has k observations. Slot coherence is an implementation
refinement condition; injectivity is required only within the retained bucket. -/
theorem sourceRecall {Index Cell : Type*} [DecidableEq Index] [DecidableEq Cell]
    {capacity : ℕ} (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (distances : Vector capacity) (cell : Index → Cell) (history : List Index)
    (point : Index → State) (center : Cell → State) (query : Index) (radius : ℝ)
    (slot : Index → Fin capacity)
    (unique : history.Nodup)
    (covered : ∀ index, dist (point index) (center (cell index)) ≤ radius)
    (enough : params.neighbors ≤
      (history.filter fun index => decide (cell index = cell query)).length)
    (injective : ∀ first ∈ LocalMemory.kept params.neighbors cell history (cell query),
      ∀ second ∈ LocalMemory.kept params.neighbors cell history (cell query),
        slot first = slot second → first = second)
    (resident : ∀ index ∈ LocalMemory.kept params.neighbors cell history (cell query),
      (slot index).val < data.count)
    (coherent : ∀ index ∈ LocalMemory.kept params.neighbors cell history (cell query),
      distances (slot index) = dist (point index) (point query) ^ 2) :
    ∀ index ∈ Neighbor.ordered data params distances none, distances index ≤ (2 * radius)^2 := by
  classical
  let bucket := (LocalMemory.kept params.neighbors cell history (cell query)).toFinset
  have count : bucket.card = params.neighbors :=
    (List.toFinset_card_of_nodup (LocalMemory.distinct params.neighbors cell history (cell query) unique)).trans
      (LocalMemory.fullLength params.neighbors cell history (cell query) enough)
  have imageSize : (bucket.image slot).card = params.neighbors := by
    rw [Finset.card_image_of_injOn]
    · exact count
    · intro first firstIn second secondIn same
      exact injective first (List.mem_toFinset.mp firstIn) second (List.mem_toFinset.mp secondIn) same
  apply sourceRadius data params distances none ((2 * radius)^2)
  rw [← imageSize]
  apply Finset.card_le_card
  intro index member
  obtain ⟨item, kept, rfl⟩ := Finset.mem_image.mp member
  have kept' := List.mem_toFinset.mp kept
  apply Finset.mem_filter.mpr
  refine ⟨Finset.mem_univ _, resident item kept', by simp, ?_⟩
  rw [coherent item kept']
  have near := LocalMemory.localRadius params.neighbors cell history point center radius covered query item kept'
  exact (sq_le_sq₀ dist_nonneg (dist_nonneg.trans near)).mpr near

/-- The actual source parameters imply positive weights. The accuracy bound
then holds pathwise, including after response-dependent fitting and selection. -/
theorem sourceError {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (points : Fin capacity → State)
    (objective : State → ℝ) (query : State) (constant : ℝ≥0) (radius noise : ℝ)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ noise)
    (near : ∀ position : Fin nearest.length, dist (points nearest[position.val]) query ≤ radius) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      noise + constant * radius := by
  apply Accuracy.inactive data params scale distances nearest _ _
    (fun position => (Neighbor.Weight.positive data params scale distances _
      epistemic aleatoric (variances _) (dissimilarities _)).le) inactive
  intro position
  have continuity := regular.dist_le_mul (points nearest[position.val]) query
  rw [Real.dist_eq] at continuity
  have radiusBound := mul_le_mul_of_nonneg_left (near position) constant.coe_nonneg
  exact (abs_sub_le _ _ _).trans (add_le_add (observed position) (continuity.trans radiusBound))

/-- Mathlib convexity gives a lower bound from all selected observations.
In particular, this can use the minimum observed loss over the whole run. -/
theorem sourceFloor {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (lower : ℝ)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length, lower ≤ data.values nearest[position.val]) :
    lower ≤ Neighbor.mean data params scale distances nearest := by
  have sumWeights : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val]) =
      Accuracy.weightTotal data params scale distances nearest := by
    simp only [Accuracy.weightTotal, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  rw [Neighbor.Mean.library data params scale distances nearest inactive]
  exact Neighbor.Mean.convex data params scale distances nearest (Set.Ici lower) (convex_Ici _)
    (fun position => (Neighbor.Weight.positive data params scale distances _
      epistemic aleatoric (variances _) (dissimilarities _)).le)
    (by rw [sumWeights]; exact lt_of_lt_of_le (by norm_num) inactive) observed

/-- The floor check can be discharged from a single observed denominator cap.
This is a numeric source condition, not a statistical calibration assumption. -/
theorem floorInactive {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (position : Fin nearest.length)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (bounded : Neighbor.denominator data params scale distances nearest[position.val] ≤ 10^12) :
    (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest := by
  have positive := Neighbor.Weight.positive data params scale distances nearest[position.val]
    epistemic aleatoric (variances _) (dissimilarities _)
  have denominator : 0 < Neighbor.denominator data params scale distances nearest[position.val] :=
    one_div_pos.mp positive
  have lower : (1 / 10^12 : ℝ) ≤ Neighbor.weight data params scale distances nearest[position.val] := by
    exact one_div_le_one_div_of_le denominator bounded
  have sumBound := Finset.single_le_sum
    (s := Finset.univ) (f := fun position : Fin nearest.length =>
      Neighbor.weight data params scale distances nearest[position.val])
    (fun position _ => (Neighbor.Weight.positive data params scale distances _
      epistemic aleatoric (variances _) (dissimilarities _)).le) (Finset.mem_univ position)
  have sumWeights : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val]) =
      Accuracy.weightTotal data params scale distances nearest := by
    simp only [Accuracy.weightTotal, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  exact lower.trans (sumWeights ▸ sumBound)

/-- A conservative lower score follows from actual prediction accuracy. -/
def lowerScore (mean noise constant radius : ℝ) : ℝ := mean - noise - constant * radius

theorem optimistic (mean truth noise constant radius : ℝ)
    (accurate : |mean - truth| ≤ noise + constant * radius) :
    lowerScore mean noise constant radius ≤ truth := by
  have upper := (abs_le.mp accurate).2
  dsimp [lowerScore]
  linarith

/-- The acquisition search pays its explicit approximation error. This does
not assume that a global acquisition minimizer is cheap to compute. -/
theorem selectionGap {Candidate : Type*} (truth estimate error : Candidate → ℝ)
    (selected comparison : Candidate) (slack : ℝ)
    (accurate : |estimate comparison - truth comparison| ≤ error comparison)
    (chosen : ApproxMin (fun candidate => estimate candidate - error candidate) selected slack) :
    estimate selected - error selected ≤ truth comparison + slack := by
  have upper := (abs_le.mp accurate).2
  have selection := chosen comparison
  linarith

/-- A small local radius turns an optimistic query into a terminal-quality
certificate. The final recommendation may occur after this query. -/
theorem terminalGap (mean finalLoss reference noise constant radius slack : ℝ)
    (lower : finalLoss - noise ≤ mean)
    (chosen : lowerScore mean noise constant radius ≤ reference + slack) :
    finalLoss - reference ≤ 2 * noise + constant * radius + slack := by
  dsimp [lowerScore] at chosen
  linarith

/-- More than k times the number of cells forces one query with at least k
earlier queries in its own cell. It holds for every adaptive query sequence. -/
theorem repeatedCell {horizon cells : ℕ} (cell : Fin horizon → Fin cells) (neighbors : ℕ)
    (budget : cells * neighbors < horizon) :
    ∃ time : Fin horizon,
      neighbors ≤ (Finset.univ.filter fun previous => previous < time ∧ cell previous = cell time).card := by
  classical
  obtain ⟨center, _, many⟩ := Finset.exists_lt_card_fiber_of_mul_lt_card_of_maps_to
    (s := Finset.univ) (t := Finset.univ) (f := cell) (n := neighbors)
    (fun _ _ => Finset.mem_univ _) (by simpa using budget)
  let bucket := Finset.univ.filter fun time => cell time = center
  have positive : bucket.Nonempty := Finset.card_pos.mp (by dsimp [bucket]; omega)
  let latest := bucket.max' positive
  have member : latest ∈ bucket := Finset.max'_mem _ _
  have centerEq : cell latest = center := (Finset.mem_filter.mp member).2
  have subset : bucket.erase latest ⊆
      Finset.univ.filter (fun previous => previous < latest ∧ cell previous = cell latest) := by
    intro previous included
    obtain ⟨different, inside⟩ := Finset.mem_erase.mp included
    have order : previous < latest := lt_of_le_of_ne (Finset.le_max' bucket previous inside) different
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, order,
      (Finset.mem_filter.mp inside).2.trans centerEq.symm⟩
  refine ⟨latest, le_trans ?_ (Finset.card_le_card subset)⟩
  have count := Finset.card_erase_add_one member
  dsimp [bucket] at count ⊢
  omega

/-- The finite-horizon bound is derived from repeated local information,
rather than a presumed contraction factor or an independent-draw law.
`retained` is the geometric property required of bounded retention and retrieval. -/
theorem queryBudget {horizon cells : ℕ} (cell : Fin horizon → Fin cells)
    (neighbors : ℕ) (mean radius : Fin horizon → ℝ)
    (finalLoss reference noise : ℝ) (constant : ℝ≥0) (coverRadius slack : ℝ)
    (budget : cells * neighbors < horizon)
    (lower : ∀ time, finalLoss - noise ≤ mean time)
    (chosen : ∀ time, lowerScore (mean time) noise constant (radius time) ≤ reference + slack)
    (retained : ∀ time,
      neighbors ≤ (Finset.univ.filter fun previous => previous < time ∧ cell previous = cell time).card →
      radius time ≤ 2 * coverRadius) :
    finalLoss - reference ≤ 2 * noise + 2 * constant * coverRadius + slack := by
  obtain ⟨time, repeated⟩ := repeatedCell cell neighbors budget
  have bound := terminalGap (mean time) finalLoss reference noise constant (radius time) slack
    (lower time) (chosen time)
  have near := mul_le_mul_of_nonneg_left (retained time repeated) constant.coe_nonneg
  linarith

#print axioms sourceError
#print axioms sourceRadius
#print axioms orderedMember
#print axioms metricRadius
#print axioms sourceRecall
#print axioms sourceFloor
#print axioms floorInactive
#print axioms repeatedCell
#print axioms queryBudget

end
end BauxResearch.Optimization.KnnLearning
