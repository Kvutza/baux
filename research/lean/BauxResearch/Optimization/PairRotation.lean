import BauxResearch.Optimization.ObjectiveSearch

/-!
# The implemented additive parameter-pair rotation

The Metal program rotates each pair `(x,y)` toward `(y,-x)`, mixes an
innovation, adds the result, and rounds. It does NOT compute the commutator
or exact orthogonal multiplication studied in `LieOperator.lean`.

These identities concern the real operation BEFORE FP32/FP16 rounding. They
give an exact squared-energy counterexample, not a theorem about pretraining
loss. In particular, expected raw-candidate energy does not determine progress
after accept/reject. A centered innovation law is an explicit extra premise;
the source hash is not assumed to be a proved centered Gaussian.
-/

namespace BauxResearch.Optimization.PairRotation

noncomputable section
open scoped BigOperators

/-- Explicit Euclidean energy; the ordinary norm on a product need not be this. -/
def energy (state : ℝ × ℝ) : ℝ := state.1^2 + state.2^2

/-- The common source pair operation with signed `scale` and supplied mixture
coefficients. Noise generation and finite-precision encoding are separate. -/
def candidate (state noise : ℝ × ℝ) (scale rotation innovation : ℝ) : ℝ × ℝ :=
  (state.1 + scale * (rotation * state.2 + innovation * noise.1),
   state.2 + scale * (-rotation * state.1 + innovation * noise.2))

def blend (first second : ℝ × ℝ) (fraction : ℝ) : ℝ × ℝ :=
  ((1 - fraction) * first.1 + fraction * second.1,
   (1 - fraction) * first.2 + fraction * second.2)

/-- Explicit reverse operation. It uses the complete current pair as well as
the original innovation and coefficients; a seed alone does not encode weights.
The denominator is positive for every real signed scale and rotation. -/
def inverse (next noise : ℝ × ℝ) (scale rotation innovation : ℝ) : ℝ × ℝ :=
  let turn := scale * rotation
  let first := next.1 - scale * innovation * noise.1
  let second := next.2 - scale * innovation * noise.2
  ((first - turn * second) / (1 + turn^2),
   (turn * first + second) / (1 + turn^2))

theorem inverseCandidate (state noise : ℝ × ℝ) (scale rotation innovation : ℝ) :
    inverse (candidate state noise scale rotation innovation) noise scale rotation innovation =
      state := by
  have denominator : (1 : ℝ) + (scale * rotation)^2 ≠ 0 := by positivity
  apply Prod.ext <;> dsimp [inverse, candidate] <;> field_simp [denominator] <;> ring

theorem candidateInverse (next noise : ℝ × ℝ) (scale rotation innovation : ℝ) :
    candidate (inverse next noise scale rotation innovation) noise scale rotation innovation =
      next := by
  have denominator : (1 : ℝ) + (scale * rotation)^2 ≠ 0 := by positivity
  apply Prod.ext <;> dsimp [inverse, candidate] <;> field_simp [denominator] <;> ring

/-- A constructed Mathlib equivalence for FIXED recorded coefficients and
innovation. It is not a bijection theorem for an adaptive accept/reject policy
or the quantized backend. Finite fixed-trace compositions inherit inverses
through `Equiv.trans` rather than requiring a new optimizer assumption. -/
def equivalence (noise : ℝ × ℝ) (scale rotation innovation : ℝ) :
    (ℝ × ℝ) ≃ (ℝ × ℝ) where
  toFun := fun state => candidate state noise scale rotation innovation
  invFun := fun next => inverse next noise scale rotation innovation
  left_inv := fun state => inverseCandidate state noise scale rotation innovation
  right_inv := fun next => candidateInverse next noise scale rotation innovation

/-- Pointwise assembly on every complete parameter pair; no finite matrix
inverse or low-dimensional latent parameterization is introduced. -/
def fullEquivalence {Pair : Type*} (noise : Pair → ℝ × ℝ)
    (scale rotation innovation : Pair → ℝ) : (Pair → ℝ × ℝ) ≃ (Pair → ℝ × ℝ) :=
  Equiv.piCongrRight fun pair =>
    equivalence (noise pair) (scale pair) (rotation pair) (innovation pair)

/-- With the source's zero partner padding, an unpaired final parameter has
only the innovation translation. Mathlib already supplies its equivalence. -/
def tailEquivalence (noise scale innovation : ℝ) : ℝ ≃ ℝ :=
  Equiv.addRight (scale * innovation * noise)

theorem tailError (state noise executed scale innovation : ℝ) :
    ((tailEquivalence noise scale innovation).symm executed - state)^2 =
      (executed - tailEquivalence noise scale innovation state)^2 := by
  change (executed - scale * innovation * noise - state)^2 =
    (executed - (state + scale * innovation * noise))^2
  ring

/-- Stable reconstruction of the previous full pair. All forward FP32/FP16
error can be included in `executed - candidate ...`; the exact reverse operation
contracts its squared Euclidean energy. Reverse-computation rounding is NOT
included and needs its own implementation account. -/
theorem inverseError (state noise executed : ℝ × ℝ) (scale rotation innovation : ℝ) :
    energy (inverse executed noise scale rotation innovation - state) =
      energy (executed - candidate state noise scale rotation innovation) /
        (1 + (scale * rotation)^2) := by
  have denominator : (1 : ℝ) + (scale * rotation)^2 ≠ 0 := by positivity
  dsimp [energy, inverse, candidate]
  field_simp [denominator]
  ring

theorem energyNonneg (state : ℝ × ℝ) : 0 ≤ energy state := by
  exact add_nonneg (sq_nonneg _) (sq_nonneg _)

theorem inverseNonexpansive (state noise executed : ℝ × ℝ) (scale rotation innovation : ℝ) :
    energy (inverse executed noise scale rotation innovation - state) ≤
      energy (executed - candidate state noise scale rotation innovation) := by
  rw [inverseError]
  exact div_le_self (energyNonneg _) (by nlinarith [sq_nonneg (scale * rotation)])

/-- Reconstruct every complete parameter pair with no dense matrix inverse.
This is a mathematical full-state operation, not the deployed archive format
or a constant-time multi-round replay algorithm. -/
theorem fullInverse {Pair : Type*} [Fintype Pair]
    (state noise executed : Pair → ℝ × ℝ) (scale rotation innovation : Pair → ℝ) :
    (∑ pair, energy (inverse (executed pair) (noise pair)
      (scale pair) (rotation pair) (innovation pair) - state pair)) ≤
      ∑ pair, energy (executed pair - candidate (state pair) (noise pair)
        (scale pair) (rotation pair) (innovation pair)) := by
  exact Finset.sum_le_sum fun pair _ => inverseNonexpansive _ _ _ _ _ _

/-- Exact source-shaped energy expansion, retaining both noise cross terms. -/
theorem energyIdentity (state noise : ℝ × ℝ) (scale rotation innovation : ℝ) :
    energy (candidate state noise scale rotation innovation) =
      (1 + (scale * rotation)^2) * energy state +
      2 * scale * innovation *
        ((state.1 + scale * rotation * state.2) * noise.1 +
         (state.2 - scale * rotation * state.1) * noise.2) +
      (scale * innovation)^2 * energy noise := by
  unfold energy candidate
  ring

/-- Additive skew rotation has positive quadratic drift, not exact norm
preservation. This does not assert anything about the neural-network objective. -/
theorem rotationDrift (state : ℝ × ℝ) (scale rotation : ℝ) :
    energy (candidate state (0, 0) scale rotation 0) - energy state =
      (scale * rotation)^2 * energy state := by
  unfold energy candidate
  ring

theorem rotationNonneg (state : ℝ × ℝ) (scale rotation : ℝ) :
    energy state ≤ energy (candidate state (0, 0) scale rotation 0) := by
  have drift := rotationDrift state scale rotation
  have nonnegative := mul_nonneg (sq_nonneg (scale * rotation)) (energyNonneg state)
  linarith

/-- No interpolation coefficient, including extrapolation, can improve squared
energy along the pure rotation chord. A theorem giving useful interpolation
from other endpoint geometries does not remove this obstruction. -/
theorem chordDrift (state : ℝ × ℝ) (scale rotation fraction : ℝ) :
    energy (blend state (candidate state (0, 0) scale rotation 0) fraction) - energy state =
      (fraction * scale * rotation)^2 * energy state := by
  unfold energy blend candidate
  ring

theorem chordNonneg (state : ℝ × ℝ) (scale rotation fraction : ℝ) :
    energy state ≤ energy (blend state (candidate state (0, 0) scale rotation 0) fraction) := by
  have drift := chordDrift state scale rotation fraction
  have nonnegative := mul_nonneg (sq_nonneg (fraction * scale * rotation)) (energyNonneg state)
  linarith

/-- This obstruction sums over all complete parameter pairs, with independently
chosen scales, rotations, and chord coefficients. An unpaired source tail has
zero pure-rotation change and can be accounted for separately. -/
theorem fullNonneg {Pair : Type*} [Fintype Pair]
    (state : Pair → ℝ × ℝ) (scale rotation fraction : Pair → ℝ) :
    (∑ pair, energy (state pair)) ≤
      ∑ pair, energy (blend (state pair)
        (candidate (state pair) (0, 0) (scale pair) (rotation pair) 0) (fraction pair)) := by
  exact Finset.sum_le_sum fun pair _ => chordNonneg _ _ _ _

/-- Only conditional centering is needed for the raw-candidate identity;
isotropy, coordinate independence, and a Gaussian posterior are not premises. -/
theorem expectedEnergy {Noise : Type*} [Fintype Noise]
    (law : ObjectiveSearch.ProposalLaw Noise) (noise : Noise → ℝ × ℝ)
    (state : ℝ × ℝ) (scale rotation innovation : ℝ)
    (firstCentered : (∑ index, law.weight index * (noise index).1) = 0)
    (secondCentered : (∑ index, law.weight index * (noise index).2) = 0) :
    (∑ index, law.weight index * energy (candidate state (noise index) scale rotation innovation)) =
      (1 + (scale * rotation)^2) * energy state +
        (scale * innovation)^2 * ∑ index, law.weight index * energy (noise index) := by
  have expanded : ∀ index,
      law.weight index * energy (candidate state (noise index) scale rotation innovation) =
        ((1 + (scale * rotation)^2) * energy state) * law.weight index +
        (2 * scale * innovation * (state.1 + scale * rotation * state.2)) *
          (law.weight index * (noise index).1) +
        (2 * scale * innovation * (state.2 - scale * rotation * state.1)) *
          (law.weight index * (noise index).2) +
        (scale * innovation)^2 * (law.weight index * energy (noise index)) := by
    intro index
    rw [energyIdentity]
    ring
  simp_rw [expanded, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [law.total, firstCentered, secondCentered]
  ring

theorem expectedNonneg {Noise : Type*} [Fintype Noise]
    (law : ObjectiveSearch.ProposalLaw Noise) (noise : Noise → ℝ × ℝ)
    (state : ℝ × ℝ) (scale rotation innovation : ℝ)
    (firstCentered : (∑ index, law.weight index * (noise index).1) = 0)
    (secondCentered : (∑ index, law.weight index * (noise index).2) = 0) :
    energy state ≤
      ∑ index, law.weight index * energy (candidate state (noise index) scale rotation innovation) := by
  rw [expectedEnergy law noise state scale rotation innovation firstCentered secondCentered]
  have noiseNonneg : 0 ≤ ∑ index, law.weight index * energy (noise index) :=
    Finset.sum_nonneg fun index _ => mul_nonneg (law.nonnegative _) (energyNonneg _)
  have rotationNonneg := mul_nonneg (sq_nonneg (scale * rotation)) (energyNonneg state)
  have innovationNonneg := mul_nonneg (sq_nonneg (scale * innovation)) noiseNonneg
  nlinarith

/-- Unit-tangent orthogonal partner `(y, -x) / √(x² + y²)` used by the Metal
`pair-rotation` shader (`resident_search.metal`, `program == 3u`). -/
def unitTangent (state : ℝ × ℝ) : ℝ × ℝ :=
  let norm := Real.sqrt (energy state)
  (state.2 / norm, -state.1 / norm)

/-- The unit-tangent vector is identically orthogonal to `state`. -/
theorem tangentOrthogonal (state : ℝ × ℝ) :
    state.1 * (unitTangent state).1 + state.2 * (unitTangent state).2 = 0 := by
  unfold unitTangent
  ring

/-- For any nonzero parameter pair, `unitTangent state` has unit Euclidean energy. -/
theorem tangentEnergy (state : ℝ × ℝ) (hPos : 0 < energy state) :
    energy (unitTangent state) = 1 := by
  unfold energy unitTangent
  dsimp
  have hNormSq : Real.sqrt (state.1 ^ 2 + state.2 ^ 2) ^ 2 = state.1 ^ 2 + state.2 ^ 2 :=
    Real.sq_sqrt (le_of_lt hPos)
  have hNe : state.1 ^ 2 + state.2 ^ 2 ≠ 0 := ne_of_gt hPos
  calc
    (state.2 / Real.sqrt (state.1 ^ 2 + state.2 ^ 2)) ^ 2 +
        (-state.1 / Real.sqrt (state.1 ^ 2 + state.2 ^ 2)) ^ 2
      = (state.2 ^ 2 + (-state.1) ^ 2) / (Real.sqrt (state.1 ^ 2 + state.2 ^ 2)) ^ 2 := by
        ring
    _ = (state.1 ^ 2 + state.2 ^ 2) / (state.1 ^ 2 + state.2 ^ 2) := by
        rw [hNormSq]
        ring
    _ = 1 := div_self hNe

/-- The trigonometric slot mixture `cos(α) ξ + sin(α) t(x, y)` in `resident_search.metal` (`program == 3u`). -/
def trigProposal (state noise : ℝ × ℝ) (alpha : ℝ) : ℝ × ℝ :=
  let t := unitTangent state
  (Real.cos alpha * noise.1 + Real.sin alpha * t.1,
   Real.cos alpha * noise.2 + Real.sin alpha * t.2)

/-- The candidate state `(x, y) + scale * trigProposal state noise alpha` executed by `program == 3u`. -/
def trigCandidate (state noise : ℝ × ℝ) (scale alpha : ℝ) : ℝ × ℝ :=
  let u := trigProposal state noise alpha
  (state.1 + scale * u.1, state.2 + scale * u.2)

/-- Exact energy expansion for `trigCandidate`, using orthogonality of `state` and `unitTangent state`. -/
theorem trigEnergy (state noise : ℝ × ℝ) (scale alpha : ℝ) :
    energy (trigCandidate state noise scale alpha) =
      energy state +
        (scale * Real.sin alpha) ^ 2 * energy (unitTangent state) +
        2 * scale * Real.cos alpha *
          ((state.1 + scale * Real.sin alpha * (unitTangent state).1) * noise.1 +
           (state.2 + scale * Real.sin alpha * (unitTangent state).2) * noise.2) +
        (scale * Real.cos alpha) ^ 2 * energy noise := by
  have hOrth := tangentOrthogonal state
  unfold energy trigCandidate trigProposal
  dsimp
  linear_combination 2 * scale * Real.sin alpha * hOrth

/-- Pure unit-tangent stepping (`noise = (0, 0)`) has state-norm-independent quadratic radial drift
`(scale * sin α)²`, contrasting with `(scale * rotation)² * energy state` in `rotationDrift`,
yet still strictly increases Euclidean energy whenever `scale * sin α ≠ 0`. -/
theorem trigDrift (state : ℝ × ℝ) (scale alpha : ℝ) (hPos : 0 < energy state) :
    energy (trigCandidate state (0, 0) scale alpha) - energy state =
      (scale * Real.sin alpha) ^ 2 := by
  rw [trigEnergy, tangentEnergy state hPos]
  unfold energy
  ring

/-- Expected energy of the shader's unit-tangent trigonometric candidate under a centered innovation law. -/
theorem trigMoment {Noise : Type*} [Fintype Noise]
    (law : ObjectiveSearch.ProposalLaw Noise) (noise : Noise → ℝ × ℝ)
    (state : ℝ × ℝ) (scale alpha : ℝ)
    (firstCentered : (∑ index, law.weight index * (noise index).1) = 0)
    (secondCentered : (∑ index, law.weight index * (noise index).2) = 0) :
    (∑ index, law.weight index * energy (trigCandidate state (noise index) scale alpha)) =
      energy state +
        (scale * Real.sin alpha) ^ 2 * energy (unitTangent state) +
        (scale * Real.cos alpha) ^ 2 * ∑ index, law.weight index * energy (noise index) := by
  have expanded : ∀ index,
      law.weight index * energy (trigCandidate state (noise index) scale alpha) =
        (energy state + (scale * Real.sin alpha) ^ 2 * energy (unitTangent state)) * law.weight index +
        (2 * scale * Real.cos alpha * (state.1 + scale * Real.sin alpha * (unitTangent state).1)) *
          (law.weight index * (noise index).1) +
        (2 * scale * Real.cos alpha * (state.2 + scale * Real.sin alpha * (unitTangent state).2)) *
          (law.weight index * (noise index).2) +
        (scale * Real.cos alpha) ^ 2 * (law.weight index * energy (noise index)) := by
    intro index
    rw [trigEnergy]
    ring
  simp_rw [expanded, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [law.total, firstCentered, secondCentered]
  ring

/-- When `energy state > 0` and the centered innovation pair has unit expected energy,
the expected post-step energy is `energy state + scale²`, invariant to the mixing angle `α`. -/
theorem trigInvariant {Noise : Type*} [Fintype Noise]
    (law : ObjectiveSearch.ProposalLaw Noise) (noise : Noise → ℝ × ℝ)
    (state : ℝ × ℝ) (scale alpha : ℝ)
    (hPos : 0 < energy state)
    (firstCentered : (∑ index, law.weight index * (noise index).1) = 0)
    (secondCentered : (∑ index, law.weight index * (noise index).2) = 0)
    (unitNoise : (∑ index, law.weight index * energy (noise index)) = 1) :
    (∑ index, law.weight index * energy (trigCandidate state (noise index) scale alpha)) =
      energy state + scale ^ 2 := by
  rw [trigMoment law noise state scale alpha firstCentered secondCentered,
    tangentEnergy state hPos, unitNoise]
  have hTrig : Real.sin alpha ^ 2 + Real.cos alpha ^ 2 = 1 := Real.sin_sq_add_cos_sq alpha
  linear_combination scale ^ 2 * hTrig

#print axioms energyIdentity
#print axioms inverseCandidate
#print axioms candidateInverse
#print axioms equivalence
#print axioms fullEquivalence
#print axioms tailEquivalence
#print axioms tailError
#print axioms inverseError
#print axioms inverseNonexpansive
#print axioms fullInverse
#print axioms rotationDrift
#print axioms chordNonneg
#print axioms fullNonneg
#print axioms expectedEnergy
#print axioms expectedNonneg
#print axioms tangentOrthogonal
#print axioms tangentEnergy
#print axioms trigEnergy
#print axioms trigDrift
#print axioms trigMoment
#print axioms trigInvariant

end
end BauxResearch.Optimization.PairRotation
