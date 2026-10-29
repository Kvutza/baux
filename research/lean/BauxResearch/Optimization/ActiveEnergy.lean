import BauxResearch.Optimization.Moments

/-!
# Energy available to an active parameter set

This is an idealized probability-law calculation, not a refinement theorem for
the seeded Metal generator. It makes the dimension cost of an isotropic law
explicit. No objective improvement follows without a hypothesis connecting
useful gain to movement on the active set.
-/

namespace BauxResearch.Optimization.ActiveEnergy

noncomputable section

open scoped BigOperators
open MeasureTheory ProbabilityTheory

variable {Index World : Type*} [Fintype Index] [DecidableEq Index]
  [MeasurableSpace World]

def energy (active : Finset Index) (delta : Index → ℝ) : ℝ :=
  ∑ index ∈ active, delta index ^ 2

/-- Weighted energy before storage rounding for a block-scaled proposal. -/
def scaledEnergy (active : Finset Index) (weight scale : Index → ℝ)
    (radius : ℝ) (noise : Index → ℝ) : ℝ :=
  ∑ index ∈ active, weight index * (radius * scale index * noise index) ^ 2

omit [Fintype Index] in
theorem squareMean (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (variance : ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (centered : Moments.mean measure delta = 0)
    (isotropic : Moments.covariance measure delta =
      Matrix.diagonal (fun _ => variance)) (index : Index) :
    (∫ world, delta world index ^ 2 ∂measure) = variance := by
  have diagonal := congrArg (fun matrix : Matrix Index Index ℝ => matrix index index) isotropic
  have center : Moments.mean measure delta index = 0 := congrFun centered index
  have identity := covariance_eq_sub (moments index) (moments index)
  simp only [Moments.covariance, Matrix.diagonal_apply_eq] at diagonal
  rw [identity] at diagonal
  change (∫ world, delta world index * delta world index ∂measure) -
    Moments.mean measure delta index * Moments.mean measure delta index = variance at diagonal
  rw [center] at diagonal
  simp only [mul_zero, sub_zero] at diagonal
  simpa only [pow_two] using diagonal

omit [Fintype Index] in
theorem activeMean (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (variance : ℝ) (active : Finset Index)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (centered : Moments.mean measure delta = 0)
    (isotropic : Moments.covariance measure delta =
      Matrix.diagonal (fun _ => variance)) :
    (∫ world, energy active (delta world) ∂measure) = active.card * variance := by
  unfold energy
  rw [integral_finsetSum]
  · simp only [squareMean measure delta variance moments centered isotropic,
      Finset.sum_const, nsmul_eq_mul]
  · exact fun index _ => (moments index).integrable_sq

/-- The active set receives its cardinality share of total expected energy.
This cross-multiplied form also covers an empty index type. -/
theorem energyShare (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (variance : ℝ) (active : Finset Index)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (centered : Moments.mean measure delta = 0)
    (isotropic : Moments.covariance measure delta =
      Matrix.diagonal (fun _ => variance)) :
    (∫ world, energy active (delta world) ∂measure) * Fintype.card Index =
      (∫ world, energy Finset.univ (delta world) ∂measure) * active.card := by
  rw [activeMean measure delta variance active moments centered isotropic,
    activeMean measure delta variance Finset.univ moments centered isotropic]
  simp only [Finset.card_univ]
  ring

omit [Fintype Index] in
/-- The source-style procedural norm is the expected pre-rounding weighted
energy when noise is centered with unit diagonal covariance. The result does
not assert those moments for the indexed generator or survive FP16 rounding. -/
theorem scaledMean (measure : Measure World) [IsProbabilityMeasure measure]
    (noise : World → Index → ℝ) (active : Finset Index)
    (weight scale : Index → ℝ) (radius : ℝ)
    (moments : ∀ index, MemLp (fun world => noise world index) 2 measure)
    (centered : Moments.mean measure noise = 0)
    (isotropic : Moments.covariance measure noise =
      Matrix.diagonal (fun _ => (1 : ℝ))) :
    (∫ world, scaledEnergy active weight scale radius (noise world) ∂measure) =
      radius ^ 2 * ∑ index ∈ active, weight index * scale index ^ 2 := by
  have expression (index : Index) :
      (fun world => weight index * (radius * scale index * noise world index) ^ 2) =
      (fun world => (weight index * (radius * scale index) ^ 2) *
        noise world index ^ 2) := by
    funext world
    ring
  have each (index : Index) :
      (∫ world, weight index * (radius * scale index * noise world index) ^ 2
        ∂measure) = radius ^ 2 * (weight index * scale index ^ 2) := by
    rw [expression, integral_const_mul,
      squareMean measure noise 1 moments centered isotropic index]
    ring
  unfold scaledEnergy
  rw [integral_finsetSum]
  · simp only [each, Finset.mul_sum]
  · intro index _
    rw [expression]
    exact (moments index).integrable_sq.const_mul _

omit [Fintype Index] in
theorem activeTail (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (variance : ℝ) (active : Finset Index)
    (threshold : ℝ) (positive : 0 < threshold)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (centered : Moments.mean measure delta = 0)
    (isotropic : Moments.covariance measure delta =
      Matrix.diagonal (fun _ => variance)) :
    measure.real { world | threshold ≤ energy active (delta world) } ≤
      active.card * variance / threshold := by
  have mean := activeMean measure delta variance active moments centered isotropic
  have integrable : Integrable (fun world => energy active (delta world)) measure := by
    exact integrable_finsetSum _ (fun index _ => (moments index).integrable_sq)
  have nonneg : 0 ≤ᵐ[measure] fun world => energy active (delta world) :=
    Filter.Eventually.of_forall (fun world => Finset.sum_nonneg (fun index _ => sq_nonneg _))
  have markov := mul_meas_ge_le_integral_of_nonneg nonneg integrable threshold
  rw [mean] at markov
  exact (le_div_iff₀ positive).mpr (by simpa only [mul_comm] using markov)

/-- Weighted energy of a realized post-rounding parameter displacement. -/
def roundedEnergy (active : Finset Index) (weight realized : Index → ℝ) : ℝ :=
  ∑ index ∈ active, weight index * realized index ^ 2

omit [Fintype Index] [DecidableEq Index] in
/-- When every coordinate on `active` falls inside the FP16 rounding deadzone
and rounds to zero displacement, realized weighted energy vanishes. -/
theorem deadzoneZero (active : Finset Index)
    (weight scale threshold realized : Index → ℝ)
    (radius : ℝ) (noise : Index → ℝ)
    (deadzone : ∀ index ∈ active,
      |radius * scale index * noise index| < threshold index →
        realized index = 0)
    (below : ∀ index ∈ active,
      |radius * scale index * noise index| < threshold index) :
    roundedEnergy active weight realized = 0 := by
  unfold roundedEnergy
  apply Finset.sum_eq_zero
  intro index member
  rw [deadzone index member (below index member)]
  ring

omit [Fintype Index] [DecidableEq Index] in
/-- Post-rounding weighted energy perturbation bound between realized updates
and pre-rounding block-scaled proposals. -/
theorem roundingGap (active : Finset Index)
    (weight scale realized error : Index → ℝ)
    (radius : ℝ) (noise : Index → ℝ)
    (nonneg : ∀ index ∈ active, 0 ≤ weight index)
    (bounded : ∀ index ∈ active,
      |realized index - radius * scale index * noise index| ≤ error index) :
    |roundedEnergy active weight realized -
      scaledEnergy active weight scale radius noise| ≤
      ∑ index ∈ active, weight index *
        (2 * |radius * scale index * noise index| * error index +
          error index ^ 2) := by
  unfold roundedEnergy scaledEnergy
  rw [← Finset.sum_sub_distrib]
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  apply Finset.sum_le_sum
  intro index member
  let step := radius * scale index * noise index
  let diff := realized index - step
  have split : weight index * realized index ^ 2 - weight index * step ^ 2 =
      weight index * (2 * step * diff + diff ^ 2) := by
    dsimp [diff]
    ring
  have stepBound : |2 * step * diff + diff ^ 2| ≤
      2 * |step| * error index + error index ^ 2 := by
    have triangle := abs_add_le (2 * step * diff) (diff ^ 2)
    have absTerm : |2 * step * diff| = 2 * |step| * |diff| := by
      rw [abs_mul, abs_mul, abs_of_pos zero_lt_two]
    have absSq : |diff ^ 2| = |diff| ^ 2 := by
      rw [abs_pow]
    have diffBound : |diff| ≤ error index := bounded index member
    have diffNonneg : 0 ≤ |diff| := abs_nonneg diff
    have stepNonneg : 0 ≤ |step| := abs_nonneg step
    nlinarith [triangle, absTerm, absSq]
  rw [split, abs_mul, abs_of_nonneg (nonneg index member)]
  exact mul_le_mul_of_nonneg_left stepBound (nonneg index member)

/-- Zeroing proposal scales outside `active` concentrates all pre-rounding
weighted energy onto `active`. -/
theorem routedScale (active : Finset Index) (weight scale : Index → ℝ)
    (radius : ℝ) (noise : Index → ℝ)
    (inactive : ∀ index, index ∉ active → scale index = 0) :
    scaledEnergy Finset.univ weight scale radius noise =
      scaledEnergy active weight scale radius noise := by
  unfold scaledEnergy
  symm
  apply Finset.sum_subset (Finset.subset_univ active)
  intro index _ outside
  rw [inactive index outside]
  ring

/-- When `scale index = 0` outside `active`, 100% of expected pre-rounding
weighted energy on `Finset.univ` is concentrated in `scaledEnergy active`,
avoiding the `active.card / Fintype.card Index` dilution of `energyShare`. -/
theorem activeShare (measure : Measure World) [IsProbabilityMeasure measure]
    (noise : World → Index → ℝ) (active : Finset Index)
    (weight scale : Index → ℝ) (radius : ℝ)
    (moments : ∀ index, MemLp (fun world => noise world index) 2 measure)
    (centered : Moments.mean measure noise = 0)
    (isotropic : Moments.covariance measure noise =
      Matrix.diagonal (fun _ => (1 : ℝ)))
    (inactive : ∀ index, index ∉ active → scale index = 0) :
    (∫ world, scaledEnergy Finset.univ weight scale radius (noise world) ∂measure) =
      (∫ world, scaledEnergy active weight scale radius (noise world) ∂measure) ∧
    (∫ world, scaledEnergy Finset.univ weight scale radius (noise world) ∂measure) =
      radius ^ 2 * ∑ index ∈ active, weight index * scale index ^ 2 := by
  have same :
      (fun world => scaledEnergy Finset.univ weight scale radius (noise world)) =
      (fun world => scaledEnergy active weight scale radius (noise world)) := by
    funext world
    exact routedScale active weight scale radius (noise world) inactive
  rw [same]
  exact ⟨rfl, scaledMean measure noise active weight scale radius
    moments centered isotropic⟩

/-- Obligation P4(a): For the `fbt-pisa1-mhc4-v1` routed MoE sublayer (`625`
routed experts with `3` active per token), an isotropic perturbation wastes
`622 / 625` (`99.52%`) of its routed-expert energy on inactive experts. -/
theorem routedDilution :
    (625 - 3 : ℕ) = 622 ∧ (3 : ℝ) / 625 + 622 / 625 = 1 := by
  norm_num

/-- Obligation P4(b): Family-structured and active-routed perturbation scale
`familyScale familyOf scales activeMask`, which assigns `scales (familyOf idx)`
to active coordinates (`activeMask idx = true`) and `0` to inactive coordinates
(such as unrouted MoE experts or frozen RMSNorms). -/
def familyScale {Family : Type*} (familyOf : Index → Family)
    (scales : Family → ℝ) (active : Finset Index) (index : Index) : ℝ :=
  if index ∈ active then scales (familyOf index) else 0

theorem familyInactive {Family : Type*} (familyOf : Index → Family)
    (scales : Family → ℝ) (active : Finset Index) (index : Index)
    (outside : index ∉ active) :
    familyScale familyOf scales active index = 0 := by
  simp [familyScale, outside]

/-- Obligation P5: Under `familyScale`, 100% of total expected weighted
perturbation energy across the full parameter space `Finset.univ` is delivered
to the active parameter subset `active`, with each active coordinate `i ∈ active`
weighted by its family scale `scales (familyOf i)^2`. -/
theorem familyShare {Family : Type*}
    (measure : Measure World) [IsProbabilityMeasure measure]
    (noise : World → Index → ℝ) (active : Finset Index)
    (weight : Index → ℝ) (familyOf : Index → Family) (scales : Family → ℝ)
    (radius : ℝ)
    (moments : ∀ index, MemLp (fun world => noise world index) 2 measure)
    (centered : Moments.mean measure noise = 0)
    (isotropic : Moments.covariance measure noise =
      Matrix.diagonal (fun _ => (1 : ℝ))) :
    (∫ world,
      scaledEnergy Finset.univ weight
        (familyScale familyOf scales active) radius (noise world) ∂measure) =
      radius ^ 2 * ∑ index ∈ active, weight index * (scales (familyOf index)) ^ 2 := by
  have share := (activeShare measure noise active weight
    (familyScale familyOf scales active) radius moments centered isotropic
    (familyInactive familyOf scales active)).2
  rw [share]
  congr 1
  apply Finset.sum_congr rfl
  intro index member
  simp [familyScale, member]

#print axioms squareMean
#print axioms activeMean
#print axioms energyShare
#print axioms scaledMean
#print axioms activeTail
#print axioms deadzoneZero
#print axioms roundingGap
#print axioms routedScale
#print axioms activeShare
#print axioms routedDilution
#print axioms familyShare

end
end BauxResearch.Optimization.ActiveEnergy
