import BauxResearch.Optimization.Moments

/-!
# The source response-field materializer as an affine random field

CUDA gaussian_fp16 uses a frozen rounded basis, fitted group coefficient and
uncertainty, and candidate-specific hashed signs. Its pre-commit real view is
amplitude[i] * (coefficient[i] + spread[i] * sign[world,i]). The amplitude
includes block scale, radius, basis, and the explicitly supplied numeric gain.
No gain^2=1/2 identity, independent signs, or Gaussian law is assumed from names.
F32 expression order, integer hash distribution, FP16/BF16 rounding, and forced
ULP steps remain execution obligations. Results are conditional on fixed history.
-/
namespace BauxResearch.Optimization.Field
noncomputable section
open MeasureTheory ProbabilityTheory

variable {Index World : Type*}

structure Parameters (Index : Type*) where
  basis : Index → ℝ
  scale : Index → ℝ
  radius : ℝ
  gain : ℝ
  coefficient : Index → ℝ
  spread : Index → ℝ

/-- Source coefficient/uncertainty arrays are indexed by operator family;
the generated field remains indexed by every model parameter. -/
def fromGroups {groups : ℕ} (basis scale : Index → ℝ) (radius gain : ℝ)
    (group : Index → Fin groups) (coefficient spread : Fin groups → ℝ) : Parameters Index where
  basis := basis
  scale := scale
  radius := radius
  gain := gain
  coefficient := fun index => coefficient (group index)
  spread := fun index => spread (group index)

theorem Group.shared {groups : ℕ} (basis scale : Index → ℝ) (radius gain : ℝ)
    (group : Index → Fin groups) (coefficient spread : Fin groups → ℝ)
    (first second : Index) (same : group first = group second) :
    (fromGroups basis scale radius gain group coefficient spread).coefficient first =
      (fromGroups basis scale radius gain group coefficient spread).coefficient second ∧
    (fromGroups basis scale radius gain group coefficient spread).spread first =
      (fromGroups basis scale radius gain group coefficient spread).spread second := by
  simp [fromGroups, same]

def amplitude (params : Parameters Index) (index : Index) : ℝ :=
  params.scale index * params.radius * params.gain * params.basis index

def generate (params : Parameters Index) (signs : World → Index → ℝ) (world : World) (index : Index) : ℝ :=
  amplitude params index * (params.coefficient index + params.spread index * signs world index)

def signValue (bit : Bool) : ℝ := if bit then 1 else -1

theorem signUnit (bit : Bool) : signValue bit ^ 2 = 1 := by cases bit <;> norm_num [signValue]

theorem frozenZero (params : Parameters Index) (signs : World → Index → ℝ) (index : Index)
    (zero : params.basis index = 0) (world : World) : generate params signs world index = 0 := by
  simp [generate, amplitude, zero]

/-- Zero spread removes candidate-specific innovation in this pre-commit view. -/
theorem spreadZero (params : Parameters Index) (signs : World → Index → ℝ) (index : Index)
    (zero : params.spread index = 0) (first second : World) :
    generate params signs first index = generate params signs second index := by
  simp [generate, zero]

variable [MeasurableSpace World]

theorem mean (measure : Measure World) [IsProbabilityMeasure measure]
    (params : Parameters Index) (signs : World → Index → ℝ) (index : Index)
    (moment : MemLp (fun world => signs world index) 2 measure) :
    (∫ world, generate params signs world index ∂measure) =
      amplitude params index * (params.coefficient index +
        params.spread index * ∫ world, signs world index ∂measure) := by
  unfold generate
  rw [integral_const_mul, integral_add (integrable_const _)
    ((moment.integrable (by simp)).const_mul _), integral_const, integral_const_mul]
  simp

/-- Source-shaped covariance retains all cross-parameter sign covariance. -/
theorem covariance (measure : Measure World) [IsProbabilityMeasure measure]
    (params : Parameters Index) (signs : World → Index → ℝ) (first second : Index)
    (firstMoment : MemLp (fun world => signs world first) 2 measure)
    (secondMoment : MemLp (fun world => signs world second) 2 measure) :
    cov[fun world => generate params signs world first,
      fun world => generate params signs world second; measure] =
      (amplitude params first * params.spread first) *
        (amplitude params second * params.spread second) *
          cov[fun world => signs world first, fun world => signs world second; measure] := by
  have firstEq : (fun world => generate params signs world first) =
      fun world => amplitude params first * params.coefficient first +
        (amplitude params first * params.spread first) * signs world first := by funext world; simp [generate]; ring
  have secondEq : (fun world => generate params signs world second) =
      fun world => amplitude params second * params.coefficient second +
        (amplitude params second * params.spread second) * signs world second := by funext world; simp [generate]; ring
  rw [firstEq, secondEq,
    covariance_const_add_left ((firstMoment.integrable (by simp)).const_mul _),
    covariance_const_add_right ((secondMoment.integrable (by simp)).const_mul _),
    covariance_const_mul_left, covariance_const_mul_right]
  ring

theorem balancedMean (measure : Measure World) [IsProbabilityMeasure measure]
    (params : Parameters Index) (signs : World → Index → ℝ) (index : Index)
    (moment : MemLp (fun world => signs world index) 2 measure)
    (balanced : (∫ world, signs world index ∂measure) = 0) :
    (∫ world, generate params signs world index ∂measure) =
      amplitude params index * params.coefficient index := by
  rw [mean measure params signs index moment, balanced]
  ring

theorem signVariance (measure : Measure World) [IsProbabilityMeasure measure]
    (signs : World → Index → ℝ) (index : Index)
    (moment : MemLp (fun world => signs world index) 2 measure)
    (balanced : (∫ world, signs world index ∂measure) = 0)
    (unit : ∀ world, signs world index ^ 2 = 1) :
    cov[fun world => signs world index, fun world => signs world index; measure] = 1 := by
  rw [covariance_eq_sub moment moment, balanced]
  have same : (fun world => signs world index * signs world index) = fun _ => (1 : ℝ) := by
    funext world
    simpa only [pow_two] using unit world
  simp only [Pi.mul_apply] at same ⊢
  rw [same]
  simp

/-- Diagonal covariance is conditional on the actual signs being uncorrelated;
it is generally anisotropic and may be singular when an amplitude/spread is zero. -/
theorem diagonal [Fintype Index] [DecidableEq Index]
    (measure : Measure World) [IsProbabilityMeasure measure]
    (params : Parameters Index) (signs : World → Index → ℝ)
    (moments : ∀ index, MemLp (fun world => signs world index) 2 measure)
    (balanced : ∀ index, (∫ world, signs world index ∂measure) = 0)
    (unit : ∀ world index, signs world index ^ 2 = 1)
    (uncorrelated : ∀ first second, first ≠ second →
      cov[fun world => signs world first, fun world => signs world second; measure] = 0) :
    Moments.covariance measure (generate params signs) =
      Matrix.diagonal (fun index => (amplitude params index * params.spread index)^2) := by
  ext first second
  change cov[fun world => generate params signs world first,
    fun world => generate params signs world second; measure] = _
  rw [covariance measure params signs first second (moments first) (moments second)]
  by_cases same : first = second
  · subst second
    rw [signVariance measure signs first (moments first) (balanced first) (fun world => unit world first)]
    simp [pow_two]
  · rw [uncorrelated first second same]
    simp [Matrix.diagonal, same]

/-- A nondegenerate innovation covariance needs all these conditions. This does
not prove they hold for the source hash or imply objective improvement. -/
theorem positive [Fintype Index] [DecidableEq Index]
    (measure : Measure World) [IsProbabilityMeasure measure]
    (params : Parameters Index) (signs : World → Index → ℝ)
    (moments : ∀ index, MemLp (fun world => signs world index) 2 measure)
    (balanced : ∀ index, (∫ world, signs world index ∂measure) = 0)
    (unit : ∀ world index, signs world index ^ 2 = 1)
    (uncorrelated : ∀ first second, first ≠ second →
      cov[fun world => signs world first, fun world => signs world second; measure] = 0)
    (innovates : ∀ index, amplitude params index * params.spread index ≠ 0) :
    (Moments.covariance measure (generate params signs)).PosDef := by
  rw [diagonal measure params signs moments balanced unit uncorrelated]
  apply Matrix.posDef_diagonal_iff.mpr
  intro index
  exact sq_pos_of_ne_zero (innovates index)

#print axioms mean
#print axioms covariance
#print axioms balancedMean
#print axioms diagonal
#print axioms positive
end
end BauxResearch.Optimization.Field
