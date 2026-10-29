import BauxResearch.Foundation
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Perturbation definitions, views, and findings

This module is the first multi-view optimizer object.  A `Core` says only how
a source value produces one update for every model parameter.  A `View` is an
exploratory observation of a generated update; views are not required to agree
with one another or to reconstruct the core.

The first geometric view measures total squared update energy.  For a signed
field, keeping one fixed scale per parameter makes that energy grow linearly
with the number of parameters.  Scaling by the square root of the parameter
count instead keeps the energy at a declared radius.  This is a candidate
perturbation mechanism exposed by the view, not a convergence or improvement
claim about an objective.
-/

namespace BauxResearch.Optimization.Perturbation

open scoped BigOperators

/-- The operational definition of a full-parameter perturbation family. -/
structure Core (parameterCount : ℕ) where
  Source : Type*
  generate : Source → Parameter.Proposal parameterCount

/-- One exploratory observation of a full parameter update. -/
structure View (parameterCount : ℕ) (Value : Type*) where
  observe : Parameter.Proposal parameterCount → Value

/-- Total squared update magnitude, viewed independently of an objective. -/
def energyView (parameterCount : ℕ) : View parameterCount ℝ where
  observe proposal := ∑ parameter, proposal parameter ^ 2

/-- Observe a perturbation through one linear response functional. -/
def linearView
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ) :
    View parameterCount ℝ where
  observe := response

/-- A full parameter field whose entries have unit squared magnitude. -/
structure SignField (parameterCount : ℕ) where
  value : ParameterIndex parameterCount → ℝ
  unitSq : ∀ parameter, value parameter ^ 2 = 1

/-- Generate a full signed perturbation with one common per-parameter scale. -/
def signedCore (parameterCount : ℕ) (scale : ℝ) : Core parameterCount where
  Source := SignField parameterCount
  generate signs := WithLp.toLp 2 fun parameter ↦
    scale * signs.value parameter

/-!
Findings are ordinary definitions and theorems, not records asserting that a
view agrees with the core.  A finding may expose a mechanism worth testing, a
conditional result, or a counterexample.
-/
namespace Finding

/-- Add a mirrored sign choice to any perturbation core. -/
def mirrorCore
    {parameterCount : ℕ}
    (core : Core parameterCount) : Core parameterCount where
  Source := core.Source × Bool
  generate source :=
    if source.2 then core.generate source.1 else -core.generate source.1

/-- The energy view exposes the dimension dependence of fixed-scale signs. -/
theorem signEnergy
    {parameterCount : ℕ}
    (scale : ℝ)
    (signs : SignField parameterCount) :
    (energyView parameterCount).observe
        ((signedCore parameterCount scale).generate signs) =
      parameterCount * scale ^ 2 := by
  simp [energyView, signedCore, mul_pow, signs.unitSq]

/-- Per-parameter scale obtained from a declared total-energy radius. -/
noncomputable def radiusScale (parameterCount : ℕ) (radius : ℝ) : ℝ :=
  radius / Real.sqrt parameterCount

/--
For a nonempty model, square-root scaling makes signed perturbation energy
exactly the square of the declared radius.
-/
theorem radiusEnergy
    {parameterCount : ℕ}
    (nonempty : 0 < parameterCount)
    (radius : ℝ)
    (signs : SignField parameterCount) :
    (energyView parameterCount).observe
        ((signedCore parameterCount (radiusScale parameterCount radius)).generate signs) =
      radius ^ 2 := by
  rw [signEnergy]
  unfold radiusScale
  have countNe : (parameterCount : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt nonempty
  have sqrtNe : Real.sqrt (parameterCount : ℝ) ≠ 0 := by
    exact Real.sqrt_ne_zero'.mpr (by exact_mod_cast nonempty)
  rw [div_pow, Real.sq_sqrt (by positivity)]
  field_simp

/-- A mirrored pair exposes the exact best change under a linear view. -/
theorem mirrorBest
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (proposal : Parameter.Proposal parameterCount) :
    min
        ((linearView response).observe proposal)
        ((linearView response).observe (-proposal)) =
      -|((linearView response).observe proposal)| := by
  simp only [linearView, LinearMap.map_neg]
  let change := response proposal
  change min change (-change) = -|change|
  rcases le_total change 0 with nonpos | nonneg
  · rw [min_eq_left (by linarith), abs_of_nonpos nonpos]
    ring
  · rw [min_eq_right (by linarith), abs_of_nonneg nonneg]

/-- The mirrored-core construction realizes both sides of `mirrorBest`. -/
theorem coreMirror
    {parameterCount : ℕ}
    (core : Core parameterCount)
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (source : core.Source) :
    min
        ((linearView response).observe
          ((mirrorCore core).generate (source, true)))
        ((linearView response).observe
          ((mirrorCore core).generate (source, false))) =
      -|((linearView response).observe (core.generate source))| := by
  simpa [mirrorCore] using mirrorBest response (core.generate source)

/-- The energy of a negated proposal equals the energy of the original proposal. -/
theorem energyNeg
    (parameterCount : ℕ)
    (proposal : Parameter.Proposal parameterCount) :
    (energyView parameterCount).observe (-proposal) =
      (energyView parameterCount).observe proposal := by
  simp [energyView]

/-- Mirrored perturbations expose linear alignment but retain the assumed curvature penalty. -/
theorem mirrorDescent
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (proposal : Parameter.Proposal parameterCount)
    (curvature : ℝ)
    (dPos dNeg : ℝ)
    (hPos : dPos ≤ (linearView response).observe proposal +
      curvature * (energyView parameterCount).observe proposal)
    (hNeg : dNeg ≤ (linearView response).observe (-proposal) +
      curvature * (energyView parameterCount).observe proposal) :
    min dPos dNeg ≤
      -|((linearView response).observe proposal)| +
        curvature * (energyView parameterCount).observe proposal := by
  have hMin : min dPos dNeg ≤
      min ((linearView response).observe proposal +
            curvature * (energyView parameterCount).observe proposal)
          ((linearView response).observe (-proposal) +
            curvature * (energyView parameterCount).observe proposal) :=
    min_le_min hPos hNeg
  rw [min_add_add_right] at hMin
  rw [mirrorBest] at hMin
  exact hMin

/-- Mirrored perturbations guarantee that the maximum opportunity of the pair is at least
the linear alignment minus the quadratic curvature penalty. -/
theorem mirrorOpp
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (proposal : Parameter.Proposal parameterCount)
    (curvature : ℝ)
    (oppPos oppNeg : ℝ)
    (hPos : -oppPos ≤ (linearView response).observe proposal +
      curvature * (energyView parameterCount).observe proposal)
    (hNeg : -oppNeg ≤ (linearView response).observe (-proposal) +
      curvature * (energyView parameterCount).observe proposal) :
    |((linearView response).observe proposal)| -
        curvature * (energyView parameterCount).observe proposal ≤
      max oppPos oppNeg := by
  have hDesc := mirrorDescent response proposal curvature (-oppPos) (-oppNeg) hPos hNeg
  rw [min_neg_neg] at hDesc
  linarith

/-- When linear response magnitude strictly exceeds the quadratic curvature penalty,
at least one mirrored candidate has strictly positive opportunity. -/
theorem mirrorPositive
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (proposal : Parameter.Proposal parameterCount)
    (curvature : ℝ)
    (oppPos oppNeg : ℝ)
    (hPos : -oppPos ≤ (linearView response).observe proposal +
      curvature * (energyView parameterCount).observe proposal)
    (hNeg : -oppNeg ≤ (linearView response).observe (-proposal) +
      curvature * (energyView parameterCount).observe proposal)
    (hAlign : curvature * (energyView parameterCount).observe proposal <
      |((linearView response).observe proposal)|) :
    0 < max oppPos oppNeg := by
  have hOpp := mirrorOpp response proposal curvature oppPos oppNeg hPos hNeg
  linarith

/-- Direct perturbation bound: a candidate need not belong to an antithetic pair.
When linear descent strictly exceeds the quadratic curvature penalty, that candidate
has strictly positive opportunity. -/
theorem directDescent
    {parameterCount : ℕ}
    (response : Parameter.Proposal parameterCount →ₗ[ℝ] ℝ)
    (proposal : Parameter.Proposal parameterCount)
    (curvature : ℝ)
    (opp : ℝ)
    (hBound : -opp ≤ (linearView response).observe proposal +
      curvature * (energyView parameterCount).observe proposal)
    (hAlign : curvature * (energyView parameterCount).observe proposal <
      -((linearView response).observe proposal)) :
    0 < opp := by
  linarith

/-- Directional strength scaling: For any positive strength scale, if scaled curvature
remains below linear descent, positive opportunity is guaranteed without mirroring. -/
theorem strengthDescent
    (linearTerm curvature energy scale : ℝ)
    (opp : ℝ)
    (hScalePos : 0 < scale)
    (hBound : -opp ≤ scale * linearTerm + scale ^ 2 * curvature * energy)
    (hAlign : scale * curvature * energy < -linearTerm) :
    0 < opp := by
  have hScaleCurv : scale * (scale * curvature * energy) < scale * (-linearTerm) :=
    mul_lt_mul_of_pos_left hAlign hScalePos
  have hMul : scale ^ 2 * curvature * energy = scale * (scale * curvature * energy) := by
    ring
  rw [hMul] at hBound
  linarith

/-- Full-parameter dense coverage: Every parameter receives a non-zero update
whenever declared radius and parameter count are positive, ensuring no low-rank truncation. -/
theorem fullDense
    {parameterCount : ℕ}
    (nonempty : 0 < parameterCount)
    (radius : ℝ)
    (hradius : 0 < radius)
    (signs : SignField parameterCount)
    (parameter : ParameterIndex parameterCount) :
    ((signedCore parameterCount (radiusScale parameterCount radius)).generate signs) parameter ≠ 0 := by
  unfold signedCore radiusScale
  have hScaleNe : radius / Real.sqrt (parameterCount : ℝ) ≠ 0 := by
    apply div_ne_zero (ne_of_gt hradius)
    exact Real.sqrt_ne_zero'.mpr (by exact_mod_cast nonempty)
  have hSignNe : signs.value parameter ≠ 0 := by
    intro hZero
    have hSq := signs.unitSq parameter
    rw [hZero, zero_pow (by norm_num)] at hSq
    norm_num at hSq
  exact mul_ne_zero hScaleNe hSignNe

/-- Any fixed coordinate support smaller than the full parameter index misses a
coordinate. Consequently every proposal confined to that support is invisible
to the coordinate objective at the missed index. The support may be the union
of every coordinate touched by a whole candidate batch, so increasing candidate
count helps only when it increases joint coverage. -/
theorem sparseBlind
    {parameterCount : ℕ}
    (support : Finset (ParameterIndex parameterCount))
    (sparse : support.card < parameterCount) :
    ∃ missed : ParameterIndex parameterCount,
      missed ∉ support ∧
      ∀ proposal : Parameter.Proposal parameterCount,
        (∀ parameter, parameter ∉ support → proposal parameter = 0) →
        ∀ state : Parameter.State parameterCount,
          (state + proposal) missed = state missed := by
  have smaller : support.card < (Finset.univ : Finset (ParameterIndex parameterCount)).card := by
    simpa using sparse
  obtain ⟨missed, _, outside⟩ := Finset.exists_mem_notMem_of_card_lt_card smaller
  refine ⟨missed, outside, ?_⟩
  intro proposal confined state
  simp [confined missed outside]

/--
Antithetic descent guarantee: For any non-orthogonal linear alignment `a ≠ 0`,
the minimum of the antithetic pair is strictly negative: `min a (-a) = -|a| < 0`.
-/
theorem Antithetic.descentGuarantee
    (alignment : ℝ)
    (hNonzero : alignment ≠ 0) :
    min alignment (-alignment) < 0 := by
  have hMin : min alignment (-alignment) = -|alignment| := by
    cases le_total 0 alignment with
    | inl hPos =>
      rw [min_eq_right (by linarith), abs_of_nonneg hPos]
    | inr hNeg =>
      rw [min_eq_left (by linarith), abs_of_nonpos hNeg]
      ring
  rw [hMin]
  have hAbsPos : 0 < |alignment| := abs_pos.mpr hNonzero
  linarith

/--
Optimal curvature step: For a quadratic directional model `-a * η + (1 / 2) * b * η^2`
with linear descent `a > 0` and curvature `b > 0`, the optimal step `η = a / b`
strictly decreases the objective by `-a^2 / (2 * b) < 0`.
-/
theorem Antithetic.stepOptimal
    (a b : ℝ)
    (_ha : 0 < a)
    (hb : 0 < b) :
    -a * (a / b) + (1 / 2) * b * (a / b) ^ 2 = -a ^ 2 / (2 * b) := by
  have _hbNe : b ≠ 0 := ne_of_gt hb
  field_simp
  ring

end Finding

end BauxResearch.Optimization.Perturbation
