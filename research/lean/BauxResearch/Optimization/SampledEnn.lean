import BauxResearch.Optimization.Calibration
import BauxResearch.Optimization.Quadrature
import BauxResearch.Optimization.KnnDecision

/-!
# ENN prediction with stratified parameter-distance error

The full weighted Euclidean view includes every model parameter. Strictly
positive tensor weights make it an injective change of units, not a latent
representation. The physical objective is unchanged when viewing it in those
units. Its regularity and observation accuracy remain explicit hypotheses.

The quadrature remainder is propagated through the unchanged source mean and
precision weights, then exposed as a certified query-selection width. Actual
sample/proposal agreement, floating-point reductions, and affordable tight
oscillation bounds are not established by the composition theorem.
-/

namespace BauxResearch.Optimization.SampledEnn

open BauxResearch.Forward
open scoped BigOperators NNReal
noncomputable section

/-- The actual sampled dissimilarities enter the weights; the residual appears
only in the error certificate, never as a substituted prediction distance. -/
def row {State : Type*} {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances remainder : Vector capacity)
    (nearest : List (Fin capacity)) (state : State) (constant error : ℝ) : KnnDecision.Scored State :=
  ⟨state, Neighbor.mean data params scale distances nearest,
    Real.sqrt ((constant^2 / params.epistemic + error^2 / params.aleatoric) *
      (nearest.length + params.epistemic * ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] * remainder nearest[position.val]) /
          Accuracy.weightTotal data params scale distances nearest)⟩

theorem rowValid {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances remainder : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (query : State)
    (constant : ℝ≥0) (error : ℝ) (errorNonneg : 0 ≤ error)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 < params.epistemic) (aleatoric : 0 < params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (geometry : ∀ index, dist (points index) query^2 ≤ distances index + remainder index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ error) :
    |(row data params scale distances remainder nearest query constant error).mean - objective query| ≤
      (row data params scale distances remainder nearest query constant error).error :=
  Calibration.sourceRemainder data params scale distances remainder nearest points objective query
    constant error errorNonneg regular epistemic aleatoric variances dissimilarities geometry inactive observed

/-- Two-sided squared-distance accuracy controls the neighbors selected by
the original source sort. No corrected distances are substituted into it. -/
theorem radiusDistortion {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (distances : Vector capacity)
    (excluded : Option (Fin capacity)) (points : Fin capacity → State) (query : State)
    (radius error : ℝ) (radiusNonneg : 0 ≤ radius)
    (approximation : ∀ index, index.val < data.count →
      |distances index - dist (points index) query^2| ≤ error)
    (many : params.neighbors ≤ (Finset.univ.filter fun index : Fin capacity =>
      index.val < data.count ∧ some index ≠ excluded ∧ dist (points index) query ≤ radius).card) :
    ∀ index ∈ Neighbor.ordered data params distances excluded,
      dist (points index) query ≤ Real.sqrt (radius^2 + 2 * error) := by
  have supplied : params.neighbors ≤ (Finset.univ.filter fun index : Fin capacity =>
      index.val < data.count ∧ some index ≠ excluded ∧ distances index ≤ radius^2 + error).card := by
    apply many.trans (Finset.card_le_card _)
    intro index included
    obtain ⟨_, active, eligible, near⟩ := Finset.mem_filter.mp included
    apply Finset.mem_filter.mpr
    refine ⟨Finset.mem_univ _, active, eligible, ?_⟩
    have squared := (sq_le_sq₀ dist_nonneg radiusNonneg).mpr near
    have upper := (abs_le.mp (approximation index active)).2
    linarith
  intro index selected
  have near := KnnLearning.sourceRadius data params distances excluded (radius^2 + error) supplied index selected
  have active := (KnnLearning.orderedMember data params distances excluded index selected).1
  have lower := (abs_le.mp (approximation index active)).1
  exact Real.le_sqrt_of_sq_le (by linarith)

/-- Source KNN prediction after approximate-distance retrieval, including
response-dependent fitted weights and the actual normalization-floor check. -/
theorem distortedError {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (excluded : Option (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (query : State)
    (constant : ℝ≥0) (radius geometryError noise : ℝ) (radiusNonneg : 0 ≤ radius)
    (regular : LipschitzWith constant objective)
    (approximation : ∀ index, index.val < data.count →
      |distances index - dist (points index) query^2| ≤ geometryError)
    (many : params.neighbors ≤ (Finset.univ.filter fun index : Fin capacity =>
      index.val < data.count ∧ some index ≠ excluded ∧ dist (points index) query ≤ radius).card)
    (epistemic : 0 ≤ params.epistemic) (aleatoric : 0 ≤ params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances
      (Neighbor.ordered data params distances excluded))
    (observed : ∀ position : Fin (Neighbor.ordered data params distances excluded).length,
      |data.values (Neighbor.ordered data params distances excluded)[position.val] -
        objective (points (Neighbor.ordered data params distances excluded)[position.val])| ≤ noise) :
    |Neighbor.mean data params scale distances (Neighbor.ordered data params distances excluded) - objective query| ≤
      noise + constant * Real.sqrt (radius^2 + 2 * geometryError) := by
  apply KnnLearning.sourceError data params scale distances _ points objective query constant _ noise
    regular epistemic aleatoric variances dissimilarities inactive observed
  intro position
  exact radiusDistortion data params distances excluded points query radius geometryError radiusNonneg
    approximation many _ (List.getElem_mem _)

/-- A checked composition from all physical parameter values to the source
ENN prediction certificate. The view is invertible, so fitting metric weights
does not silently change the physical objective being evaluated. -/
theorem quadratureBound {Index Cell : Type*} [Fintype Index] [DecidableEq Index]
    [Fintype Cell] [DecidableEq Cell] {capacity : ℕ}
    (stratum : Index → Cell) (sample : Cell → Index) (weight : Cell → ℝ)
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → Index → ℝ) (query : Index → ℝ)
    (objective : (Index → ℝ) → ℝ) (constant : ℝ≥0) (error : ℝ)
    (oscillation : Fin capacity → Cell → ℝ)
    (coherent : ∀ center, stratum (sample center) = center)
    (positive : ∀ center, 0 < weight center) (errorNonneg : 0 ≤ error)
    (regular : LipschitzWith constant (fun state : EuclideanSpace ℝ Index =>
      objective (Quadrature.unembed stratum weight state)))
    (epistemic : 0 < params.epistemic) (aleatoric : 0 < params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (supplied : ∀ index, distances index = Quadrature.sampled stratum sample weight
      (fun parameter => points index parameter - query parameter))
    (bounded : ∀ index center parameter, stratum parameter = center →
      |(points index parameter - query parameter)^2 -
        (points index (sample center) - query (sample center))^2| ≤ oscillation index center)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ error) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      (row data params scale distances
        (fun index => Quadrature.remainder stratum sample weight (oscillation index))
        nearest query constant error).error := by
  let view := fun state : EuclideanSpace ℝ Index => objective (Quadrature.unembed stratum weight state)
  have evaluated (state : Index → ℝ) : view (Quadrature.embed stratum weight state) = objective state := by
    dsimp [view]
    rw [Quadrature.unembedEmbed stratum weight state positive]
  have distancesNonneg (index : Fin capacity) : 0 ≤ distances index := by
    rw [supplied index]
    exact Quadrature.sampledNonneg stratum sample weight _ (fun center => (positive center).le)
  have geometry (index : Fin capacity) :
      dist (Quadrature.embed stratum weight (points index)) (Quadrature.embed stratum weight query)^2 ≤
        distances index + Quadrature.remainder stratum sample weight (oscillation index) := by
    rw [Quadrature.distanceSquared stratum weight _ _ (fun center => (positive center).le), supplied index]
    exact Quadrature.distanceUpper stratum sample weight _ (oscillation index) coherent
      (fun center => (positive center).le) (bounded index)
  have certified := Calibration.sourceRemainder data params scale distances
    (fun index => Quadrature.remainder stratum sample weight (oscillation index)) nearest
    (fun index => Quadrature.embed stratum weight (points index)) view (Quadrature.embed stratum weight query)
    constant error errorNonneg regular epistemic aleatoric variances distancesNonneg geometry inactive
    (fun position => by simpa only [evaluated] using observed position)
  simpa only [evaluated, row] using certified

#print axioms rowValid
#print axioms radiusDistortion
#print axioms distortedError
#print axioms quadratureBound

end
end BauxResearch.Optimization.SampledEnn
