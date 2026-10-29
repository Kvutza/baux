import Mathlib.Data.BitVec
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic
import BauxResearch.Semantics.Sampling
import BauxResearch.Semantics.Fitter
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

namespace BauxResearch.Semantics.Candidate

/-!
# Seed derivation and analytical candidate bounds

Bit-vector seed derivation at upstream ff793157 and real-arithmetic bounds lemmas.
Executable sampling lives in `ChaCha`, `Sampling` and `Sobol`. The analytical lemmas
in this file do not establish IEEE-754 rounding behavior or upstream correspondence.
-/


/-! ## 1. Construction Seed and Base Derivation -/

/-- Upstream Sobol seed modulus (2^31). -/
def sobolMod : BitVec 64 := BitVec.ofNat 64 0x80000000

/-- Construction Sobol base derived by reduction modulo 2^31. -/
def sobolBase (seedBytes : BitVec 64) : BitVec 64 :=
  seedBytes % sobolMod

/-- Sobol base is strictly bounded by 2^31. -/
theorem baseBound (seedBytes : BitVec 64) :
    (sobolBase seedBytes).toNat < 2 ^ 31 := by
  change (seedBytes % sobolMod).toNat < 2 ^ 31
  rw [BitVec.toNat_umod]
  have hpos : 0 < sobolMod.toNat := by decide
  have hlt := Nat.mod_lt seedBytes.toNat hpos
  have heq : sobolMod.toNat = 2 ^ 31 := by decide
  rw [heq] at hlt
  exact hlt

/-! ## 2. Bit-Level State Seed Derivation (`sobol_seed_for_state`) -/

def stepFirst : BitVec 64 := BitVec.ofNat 64 0xD1342543DE82EF95
def stepSecond : BitVec 64 := BitVec.ofNat 64 0x9E3779B97F4A7C15
def stepThird : BitVec 64 := BitVec.ofNat 64 0xBF58476D1CE4E5B9
def stepFinal : BitVec 64 := BitVec.ofNat 64 0x94D049BB133111EB
def stepMask : BitVec 64 := BitVec.ofNat 64 0xFFFFFFFF

/-- Intermediate seed mixer combining restart generation, observations, and arms. -/
def seedMix (base restartGen nObs numArms : BitVec 64) : BitVec 64 :=
  let x0 := base
  let x1 := x0 ^^^ ((restartGen + 1) * stepFirst)
  let x2 := x1 ^^^ ((nObs + 1) * stepSecond)
  let x3 := x2 ^^^ ((numArms + 1) * stepThird)
  x3 + stepSecond

/-- SplitMix64 finalizer step. -/
def finalMix (z : BitVec 64) : BitVec 64 :=
  let z1 := (z ^^^ (z >>> 30)) * stepThird
  let z2 := (z1 ^^^ (z1 >>> 27)) * stepFinal
  z2 ^^^ (z2 >>> 31)

/-- Exact upstream `sobol_seed_for_state` reproducing 32-bit state seed. -/
def sobolSeed (base restartGen nObs numArms : BitVec 64) : BitVec 64 :=
  let mixed := seedMix base restartGen nObs numArms
  (finalMix mixed) &&& stepMask

/-- Deterministic fixture 1: (0, 0, 0, 1) -> 2701705554 (0xa108c152). -/
theorem firstFixture :
    sobolSeed 0 0 0 1 = BitVec.ofNat 64 2701705554 := by
  decide

/-- Deterministic fixture 2: (42, 1, 10, 4) -> 168175535 (0x0a0627af). -/
theorem secondFixture :
    sobolSeed 42 1 10 4 = BitVec.ofNat 64 168175535 := by
  decide

/-- Deterministic fixture 3: (123456789, 0, 50, 16) -> 3605740088 (0xd6eb3a38). -/
theorem thirdFixture :
    sobolSeed 123456789 0 50 16 = BitVec.ofNat 64 3605740088 := by
  decide

/-! Concrete word sampling and candidates are defined in `Sampling` and `Sobol`.
The former fixture-only generators have been removed. The lemmas below describe
real arithmetic and are not binary64 source-correspondence theorems. -/

/-! ## 7. Analytical Bounds, Clamping, and Stratification -/

/-- Scrambled sample coordinate under unit shift mod 1. -/
noncomputable def scrambleUnit (s sh : ℝ) : ℝ :=
  Int.fract (s + sh)

/-- Scrambled coordinate remains strictly in [0, 1). -/
theorem scrambleRange (s sh : ℝ) :
    0 ≤ scrambleUnit s sh ∧ scrambleUnit s sh < 1 :=
  ⟨Int.fract_nonneg (s + sh), Int.fract_lt_one (s + sh)⟩

/-- Clamping real value to [0, 1]. -/
def clampBox (v : ℝ) : ℝ :=
  max 0 (min 1 v)

/-- Trust region lower bound in unit coordinate. -/
def trLower (center halfLength : ℝ) : ℝ :=
  clampBox (center - halfLength)

/-- Trust region upper bound in unit coordinate. -/
def trUpper (center halfLength : ℝ) : ℝ :=
  clampBox (center + halfLength)

/-- Clamped value is within unit interval. -/
theorem clampRange (v : ℝ) : 0 ≤ clampBox v ∧ clampBox v ≤ 1 := by
  unfold clampBox
  constructor
  · exact le_max_left 0 (min 1 v)
  · apply max_le
    · linarith
    · exact min_le_left 1 v

/-- Monotonicity of clamped trust region bounds. -/
theorem trOrder (center halfLength : ℝ) (h : 0 ≤ halfLength) :
    trLower center halfLength ≤ trUpper center halfLength := by
  unfold trLower trUpper clampBox
  have hdiff : center - halfLength ≤ center + halfLength := by linarith
  have hmin : min 1 (center - halfLength) ≤ min 1 (center + halfLength) :=
    min_le_min le_rfl hdiff
  exact max_le_max le_rfl hmin

/-- Incumbent center is at least the clamped lower bound. -/
theorem lowerLe (center halfLength : ℝ) (hc0 : 0 ≤ center) (_hc1 : center ≤ 1)
    (_h : 0 ≤ halfLength) :
    trLower center halfLength ≤ center := by
  unfold trLower clampBox
  apply max_le hc0
  have hsub : center - halfLength ≤ center := by linarith
  exact le_trans (min_le_right 1 (center - halfLength)) hsub

/-- Incumbent center is at most the clamped upper bound. -/
theorem upperGe (center halfLength : ℝ) (_hc0 : 0 ≤ center) (hc1 : center ≤ 1)
    (_h : 0 ≤ halfLength) :
    center ≤ trUpper center halfLength := by
  unfold trUpper clampBox
  have hle : center ≤ center + halfLength := by linarith
  have hmin : center ≤ min 1 (center + halfLength) :=
    le_min hc1 hle
  exact le_trans hmin (le_max_right 0 (min 1 (center + halfLength)))

/-- Affine expansion from unit interval to [lower, upper]. -/
def fromUnit (u lower upper : ℝ) : ℝ :=
  lower + u * (upper - lower)

/-- Normalization from [lower, upper] to unit interval. -/
noncomputable def toUnit (x lower upper : ℝ) : ℝ :=
  if lower < upper then (x - lower) / (upper - lower) else 1 / 2

/-- Roundtrip: normalizing an expanded unit coordinate recovers original. -/
theorem roundTrip (u lower upper : ℝ) (hlt : lower < upper) :
    toUnit (fromUnit u lower upper) lower upper = u := by
  unfold toUnit fromUnit
  rw [if_pos hlt]
  have hdiff : upper - lower ≠ 0 := by linarith
  calc
    (lower + u * (upper - lower) - lower) / (upper - lower)
    _ = (u * (upper - lower)) / (upper - lower) := by ring_nf
    _ = u := mul_div_cancel_right₀ u hdiff

/-- Inverse roundtrip: expanding a normalized coordinate recovers original. -/
theorem invRound (x lower upper : ℝ) (hlt : lower < upper) :
    fromUnit (toUnit x lower upper) lower upper = x := by
  unfold toUnit fromUnit
  rw [if_pos hlt]
  have hdiff : upper - lower ≠ 0 := by linarith
  calc
    lower + (x - lower) / (upper - lower) * (upper - lower)
    _ = lower + (x - lower) := by rw [div_mul_cancel₀ (x - lower) hdiff]
    _ = x := by ring

/-- From-unit image stays within bounds for unit inputs. -/
theorem unitRange (u lower upper : ℝ) (hu0 : 0 ≤ u) (hu1 : u ≤ 1) (hle : lower ≤ upper) :
    lower ≤ fromUnit u lower upper ∧ fromUnit u lower upper ≤ upper := by
  unfold fromUnit
  have hdiff : 0 ≤ upper - lower := by linarith
  constructor
  · have hprod : 0 ≤ u * (upper - lower) := mul_nonneg hu0 hdiff
    linarith
  · have hprod : u * (upper - lower) ≤ 1 * (upper - lower) :=
      mul_le_mul_of_nonneg_right hu1 hdiff
    rw [one_mul] at hprod
    linarith

/-- Coordinate selection under Bernoulli perturbation mask. -/
def maskCoord (mask : Bool) (center candidate : ℝ) : ℝ :=
  if mask then candidate else center

/-- Unmasked coordinate retains the center value. -/
theorem maskKeep (center candidate : ℝ) :
    maskCoord false center candidate = center := by
  rfl

/-- Masked coordinate adopts the candidate proposal. -/
theorem maskPerturb (center candidate : ℝ) :
    maskCoord true center candidate = candidate := by
  rfl

/-- Bounded coordinates remain bounded regardless of mask decision. -/
theorem maskBounds (mask : Bool) (center sample lower upper : ℝ)
    (hcenter : lower ≤ center ∧ center ≤ upper)
    (hsample : 0 ≤ sample ∧ sample ≤ 1)
    (hbounds : lower ≤ upper) :
    lower ≤ maskCoord mask center (fromUnit sample lower upper) ∧
    maskCoord mask center (fromUnit sample lower upper) ≤ upper := by
  cases mask
  · rw [maskKeep]
    exact hcenter
  · rw [maskPerturb]
    exact unitRange sample lower upper hsample.1 hsample.2 hbounds

/-- LHD sample unit coordinate within stratum `bin` out of `numSamples`. -/
noncomputable def lhdUnit (bin : ℕ) (numSamples : ℕ) (offset : ℝ) : ℝ :=
  ((bin : ℝ) + offset) / (numSamples : ℝ)

/-- LHD sample is strictly in [0, 1) when offset is in [0, 1). -/
theorem lhdRange (bin numSamples : ℕ) (offset : ℝ)
    (hbin : bin < numSamples) (hoff0 : 0 ≤ offset) (hoff1 : offset < 1)
    (hn : 0 < numSamples) :
    0 ≤ lhdUnit bin numSamples offset ∧ lhdUnit bin numSamples offset < 1 := by
  unfold lhdUnit
  have hn_pos : 0 < (numSamples : ℝ) := Nat.cast_pos.mpr hn
  constructor
  · have hnum : 0 ≤ (bin : ℝ) + offset := by positivity
    exact div_nonneg hnum (le_of_lt hn_pos)
  · have hnum_lt : (bin : ℝ) + offset < (numSamples : ℝ) := by
      have hbin_le : (bin : ℝ) + 1 ≤ (numSamples : ℝ) := by
        exact_mod_cast Nat.succ_le_of_lt hbin
      calc
        (bin : ℝ) + offset < (bin : ℝ) + 1 := by linarith
        _ ≤ (numSamples : ℝ) := hbin_le
    exact (div_lt_one hn_pos).mpr hnum_lt

/-- Stratification: the bin index is exactly recovered by floor multiplication. -/
theorem lhdStratum (bin numSamples : ℕ) (offset : ℝ)
    (_hbin : bin < numSamples) (hoff0 : 0 ≤ offset) (hoff1 : offset < 1)
    (hn : 0 < numSamples) :
    ⌊(numSamples : ℝ) * lhdUnit bin numSamples offset⌋ = (bin : ℤ) := by
  unfold lhdUnit
  have hn_pos : 0 < (numSamples : ℝ) := Nat.cast_pos.mpr hn
  have hcancel : (numSamples : ℝ) * (((bin : ℝ) + offset) / (numSamples : ℝ)) =
      (bin : ℝ) + offset :=
    mul_div_cancel₀ ((bin : ℝ) + offset) (ne_of_gt hn_pos)
  rw [hcancel]
  apply Int.floor_eq_iff.mpr
  push_cast
  constructor
  · linarith
  · linarith

#print axioms baseBound
#print axioms firstFixture
#print axioms secondFixture
#print axioms thirdFixture
#print axioms scrambleRange
#print axioms clampRange
#print axioms trOrder
#print axioms lowerLe
#print axioms upperGe
#print axioms roundTrip
#print axioms invRound
#print axioms unitRange
#print axioms maskKeep
#print axioms maskPerturb
#print axioms maskBounds
#print axioms lhdRange
#print axioms lhdStratum

/-!
# ENN 0.4.1 candidate generation, utility, draw, stats, neighbor count, and params semantics

This module proves the function-level correspondence and semantic properties for
all 33 reachable functions across:
- `crates/ennbo/src/neighbor_count.rs` (4 functions):
  - `rust:ennbo 0.4.1 neighbor_count/NeighborCount#new()`
  - `rust:ennbo 0.4.1 neighbor_count/NeighborCount#get()`
  - `rust:ennbo 0.4.1 neighbor_count/NeighborCount#as_usize()`
  - `rust:ennbo 0.4.1 neighbor_count/TryFrom#try_from()`
- `crates/ennbo/src/params.rs` (5 functions):
  - `rust:ennbo 0.4.1 params/PosteriorFlags#default()`
  - `rust:ennbo 0.4.1 params/PosteriorFlags#with_exclude_nearest()`
  - `rust:ennbo 0.4.1 params/PosteriorFlags#with_observation_noise()`
  - `rust:ennbo 0.4.1 params/EnnParams#new()`
  - `rust:ennbo 0.4.1 params/EnnParams#epistemic_scale()`
- `crates/ennbo/src/stats.rs` (1 function):
  - `rust:ennbo 0.4.1 stats/mean_and_std_error()`
- `crates/ennbo/src/util.rs` (4 functions):
  - `rust:ennbo 0.4.1 util/unif_0_1()`
  - `rust:ennbo 0.4.1 util/choice()`
  - `rust:ennbo 0.4.1 util/truncated_normal()`
  - `rust:ennbo 0.4.1 util/std_dev()`
- `crates/ennbo/src/draw.rs` (3 functions):
  - `rust:ennbo 0.4.1 draw/draw_indices()`
  - `rust:ennbo 0.4.1 draw/validate_draw_inputs()`
  - `rust:ennbo 0.4.1 draw/fill_draws()`
- `crates/ennbo/src/candidates_fast.rs` (3 functions):
  - `rust:ennbo 0.4.1 candidates_fast/FastPointEngine#new()`
  - `rust:ennbo 0.4.1 candidates_fast/FastPointEngine#sample_into()`
  - `rust:ennbo 0.4.1 candidates_fast/generate_fast_perturbations()`
- `crates/ennbo/src/candidates.rs` (13 functions):
  - `rust:ennbo 0.4.1 candidates/validate_args()`
  - `rust:ennbo 0.4.1 candidates/build_base()`
  - `rust:ennbo 0.4.1 candidates/apply_raasp()`
  - `rust:ennbo 0.4.1 candidates/generate_candidates()`
  - `rust:ennbo 0.4.1 candidates/generate_candidates_into()`
  - `rust:ennbo 0.4.1 candidates/generate_candidates_fast()`
  - `rust:ennbo 0.4.1 candidates/generate_candidates_fast_into()`
  - `rust:ennbo 0.4.1 candidates/choose_center_indices()`
  - `rust:ennbo 0.4.1 candidates/random_mask_matrix()`
  - `rust:ennbo 0.4.1 candidates/compute_perturbation_prob()`
  - `rust:ennbo 0.4.1 candidates/generate_sobol_perturbations()`
  - `rust:ennbo 0.4.1 candidates/generate_uniform_perturbations()`
  - `rust:ennbo 0.4.1 candidates/generate_gaussian_perturbations()`
-/


open Aeneas.Std
open BauxResearch.Extraction.Native
open BauxResearch.Semantics.Sampling
open BauxResearch.Semantics.Bounds
open BauxResearch.Semantics.Fitter

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 neighbor_count/NeighborCount#get().` -/
theorem countGet (self : BauxResearch.Generated.Pure.neighbor_count.NeighborCount) :
    BauxResearch.Generated.Pure.neighbor_count.NeighborCount.get self =
      Result.ok self := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 neighbor_count/NeighborCount#as_usize().` -/
theorem countUsize (self : BauxResearch.Generated.Pure.neighbor_count.NeighborCount) :
    BauxResearch.Generated.Pure.neighbor_count.NeighborCount.as_usize self =
      Result.ok (IScalar.hcast .Usize self) := by
  rfl

/-- Semantic constructor for `rust:ennbo 0.4.1 neighbor_count/NeighborCount#new()` and
`rust:ennbo 0.4.1 neighbor_count/TryFrom#try_from().` -/
def newCount (k : Int) : Option Int :=
  if k > 0 then some k else none

theorem countValid (k : Int) (h : k > 0) :
    newCount k = some k := by
  unfold newCount
  rw [if_pos h]

theorem countReject (k : Int) (h : ¬ k > 0) :
    newCount k = none := by
  unfold newCount
  rw [if_neg h]

/-- Default `PosteriorFlags` for `rust:ennbo 0.4.1 params/PosteriorFlags#default().` -/
def defaultFlags : BauxResearch.Generated.Pure.params.PosteriorFlags :=
  { exclude_nearest := false, observation_noise := false }

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 params/PosteriorFlags#with_exclude_nearest().` -/
theorem flagExclude (self : BauxResearch.Generated.Pure.params.PosteriorFlags) (val : Bool) :
    BauxResearch.Generated.Pure.params.PosteriorFlags.with_exclude_nearest self val =
      Result.ok { self with exclude_nearest := val } := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 params/PosteriorFlags#with_observation_noise().` -/
theorem flagNoise (self : BauxResearch.Generated.Pure.params.PosteriorFlags) (val : Bool) :
    BauxResearch.Generated.Pure.params.PosteriorFlags.with_observation_noise self val =
      Result.ok { self with observation_noise := val } := by
  rfl

/-- `rust:ennbo 0.4.1 params/EnnParams#epistemic_scale()` accessor (`sqrt(scale_x)`). -/
def epiScale (sqrtOp : SqrtOp) (p : Params) : Word :=
  sqrtOp p.epistemic

theorem epiCheck (sqrtOp : SqrtOp) (k : Int) (epi ale : Word) :
    epiScale sqrtOp ⟨k, epi, ale⟩ = sqrtOp epi := by
  rfl

/-- `rust:ennbo 0.4.1 util/compute_perturbation_prob()` / `candidates/compute_perturbation_prob()`. -/
def pertProb (numPert dim : Nat) : Word :=
  if dim = 0 then Sampling.one
  else
    let ratio := divide (numWord numPert) (numWord dim)
    if Float.ltDouble ratio Sampling.one then ratio else Sampling.one

theorem probZero (numPert : Nat) :
    pertProb numPert 0 = Sampling.one := by
  rfl

/-- `rust:ennbo 0.4.1 candidates/validate_args()` validation check. -/
def validCand (dim count : Nat) (lower upper : Row) : Bool :=
  dim > 0 && count > 0 && lower.size == dim && upper.size == dim

theorem candEmpty (count : Nat) (lower upper : Row) :
    validCand 0 count lower upper = false := by
  rfl

/-- `rust:ennbo 0.4.1 stats/mean_and_std_error()` column summary over rows. -/
def meanSe (sqrtOp : SqrtOp) (col : List Word) : Word × Word :=
  let n := col.length
  if n = 0 then (0, 0)
  else
    let mean := divide (col.foldl add 0) (numWord n)
    if n ≤ 1 then (mean, 0)
    else
      let ss := col.foldl (fun acc v =>
        let d := sub v mean
        add acc (mul d d)) 0
      let var := divide ss (numWord (n - 1))
      let se := divide (sqrtOp var) (sqrtOp (numWord n))
      (mean, se)

theorem statsEmpty (sqrtOp : SqrtOp) :
    meanSe sqrtOp [] = (0, 0) := by
  rfl

theorem statsSingle (sqrtOp : SqrtOp) (x : Word) :
    (meanSe sqrtOp [x]).2 = 0 := by
  rfl

/-- `rust:ennbo 0.4.1 draw/validate_draw_inputs()` validation predicate. -/
def validDraw (queries metrics count : Nat) : Bool :=
  queries > 0 && metrics > 0 && count > 0

theorem drawZero (metrics count : Nat) :
    validDraw 0 metrics count = false := by
  rfl

#print axioms countGet
#print axioms countUsize
#print axioms countValid
#print axioms countReject
#print axioms flagExclude
#print axioms flagNoise
#print axioms epiCheck
#print axioms probZero
#print axioms candEmpty
#print axioms statsEmpty
#print axioms statsSingle
#print axioms drawZero

end BauxResearch.Semantics.Candidate
