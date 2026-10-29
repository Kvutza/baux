import BauxResearch.Optimization.Step
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.BigOperators.Intervals

/-!
# Epistemic surrogate error bounds

This module formalizes the nearest-neighbor precision-weighted Epistemic
Nearest Neighbors (ENN) surrogate from first principles through Mathlib.

For observations `(x_i, y_i)` with noise bounds, the surrogate mean is a
convex combination of neighbor values.  Under Lipschitz continuity of the
objective function, surrogate approximation error is bounded by neighbor
distance plus observation noise.

Inside a trust region of radius `length`, all neighbor distances are bounded
by `2 * length`.  This establishes the uniform error bound required by
`Selection.regretBound`.
-/

namespace BauxResearch.Optimization

open scoped BigOperators

variable {State : Type*}

/-- A single black-box function observation. -/
structure Observation (State : Type*) where
  point : State
  value : ℝ
  variance : ℝ

/--
A convex combination of values bounded in `[yMin, yMax]` remains in `[yMin, yMax]`.
-/
theorem Convex.bound
    {ι : Type*} (s : Finset ι)
    (weights : ι → ℝ) (values : ι → ℝ)
    (yMin yMax : ℝ)
    (hNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hSumPos : 0 < ∑ i ∈ s, weights i)
    (hLower : ∀ i ∈ s, yMin ≤ values i)
    (hUpper : ∀ i ∈ s, values i ≤ yMax) :
    yMin ≤ (∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) ∧
    (∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) ≤ yMax := by
  have hSumNonneg : 0 ≤ ∑ i ∈ s, weights i := le_of_lt hSumPos
  have hWeightedLower : yMin * (∑ i ∈ s, weights i) ≤ ∑ i ∈ s, weights i * values i := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have hW := hNonneg i hi
    have hV := hLower i hi
    nlinarith
  have hWeightedUpper : (∑ i ∈ s, weights i * values i) ≤ yMax * (∑ i ∈ s, weights i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have hW := hNonneg i hi
    have hV := hUpper i hi
    nlinarith
  constructor
  · rw [le_div_iff₀ hSumPos]
    linarith
  · rw [div_le_iff₀ hSumPos]
    linarith

/--
Under Lipschitz continuity and bounded noise, every neighbor value is close to `f query`.
-/
theorem Neighbor.valueBound
    [MetricSpace State]
    (objective : Objective State)
    (query neighborPoint : State)
    (neighborValue : ℝ)
    (K radius noise : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ x y, |objective x - objective y| ≤ K * dist x y)
    (hDist : dist query neighborPoint ≤ radius)
    (hNoise : |neighborValue - objective neighborPoint| ≤ noise) :
    |neighborValue - objective query| ≤ K * radius + noise := by
  have hObjDist : |objective neighborPoint - objective query| ≤ K * dist neighborPoint query := by
    have h := hLip neighborPoint query
    exact h
  rw [dist_comm] at hObjDist
  have hKDist : K * dist query neighborPoint ≤ K * radius :=
    mul_le_mul_of_nonneg_left hDist hK
  have hTriangle : |neighborValue - objective query| ≤
      |neighborValue - objective neighborPoint| + |objective neighborPoint - objective query| := by
    have hEq : neighborValue - objective query =
        (neighborValue - objective neighborPoint) + (objective neighborPoint - objective query) := by ring
    rw [hEq]
    exact abs_add_le _ _
  linarith

/--
Uniform surrogate error on a trust region of radius `length`:
If all observations and the query candidate lie within radius `length` of an incumbent,
the surrogate error is bounded by `2 * K * length + noise`.
-/
theorem Surrogate.trustError
    [MetricSpace State]
    (objective : Objective State)
    (incumbent query : State)
    (length K noise : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ x y, |objective x - objective y| ≤ K * dist x y)
    (hQueryInBall : dist query incumbent ≤ length)
    {ι : Type*} (s : Finset ι)
    (points : ι → State)
    (values : ι → ℝ)
    (weights : ι → ℝ)
    (hPointsInBall : ∀ i ∈ s, dist (points i) incumbent ≤ length)
    (hNoiseBound : ∀ i ∈ s, |values i - objective (points i)| ≤ noise)
    (hWeightsNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hWeightsSumPos : 0 < ∑ i ∈ s, weights i) :
    |(∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) - objective query| ≤
      2 * K * length + noise := by
  have hNeighborDist : ∀ i ∈ s, dist query (points i) ≤ 2 * length := by
    intro i hi
    have hTri := dist_triangle query incumbent (points i)
    have hP := hPointsInBall i hi
    rw [dist_comm (points i) incumbent] at hP
    linarith
  have hNeighborValue : ∀ i ∈ s, |values i - objective query| ≤ 2 * K * length + noise := by
    intro i hi
    have hVal := Neighbor.valueBound objective query (points i) (values i) K (2 * length) noise
      hK hLip (hNeighborDist i hi) (hNoiseBound i hi)
    linarith
  have hLower : ∀ i ∈ s, objective query - (2 * K * length + noise) ≤ values i := by
    intro i hi
    have hAbs := (abs_le.mp (hNeighborValue i hi)).1
    linarith
  have hUpper : ∀ i ∈ s, values i ≤ objective query + (2 * K * length + noise) := by
    intro i hi
    have hAbs := (abs_le.mp (hNeighborValue i hi)).2
    linarith
  have hConvex := Convex.bound s weights values
    (objective query - (2 * K * length + noise))
    (objective query + (2 * K * length + noise))
    hWeightsNonneg hWeightsSumPos hLower hUpper
  apply abs_le.mpr
  constructor
  · linarith [hConvex.1]
  · linarith [hConvex.2]

/--
Epistemic uncertainty contraction: Adding an observation with positive precision `w > 0`
strictly contracts the epistemic variance `1 / W`.
-/
theorem Epistemic.varianceContract
    (W w : ℝ)
    (hW : 0 < W)
    (hw : 0 < w) :
    1 / (W + w) < 1 / W := by
  have hSum : W < W + w := by linarith
  exact one_div_lt_one_div_of_lt hW hSum

/--
Epistemic sharpness bound: When accumulated neighbor precision satisfies `W_min ≤ W`,
the epistemic variance is bounded by `1 / W_min`.
-/
theorem Epistemic.sharpnessBound
    (W W_min : ℝ)
    (hWMin : 0 < W_min)
    (hLe : W_min ≤ W) :
    1 / W ≤ 1 / W_min := by
  exact one_div_le_one_div_of_le hWMin hLe

/-!
### ENN Uncertainty Estimation Combination

In ENN, the predictive variance decomposes into an epistemic component
(controlled by proximity to observed states) and an aleatoric component
(controlled by minibatch evaluation noise):
`σ_total^2 = 1 / W + (∑ w_i σ_i^2) / W`.

This separation is critical:
1. Epistemic uncertainty contracts with neighbor precision (`1 / W`), driving exploration.
2. Aleatoric noise reflects irreducible batch sampling variance, downweighting noisy batches.
3. In distribution space, epistemic contrast is preserved because distances do not concentrate.
-/

/--
Decomposition of ENN total predictive variance into epistemic and aleatoric components:
`(1 + noiseSum) / W = 1 / W + noiseSum / W`.
-/
theorem Uncertainty.totalDecomp
    (W noiseSum : ℝ) :
    (1 + noiseSum) / W = 1 / W + noiseSum / W := by
  ring

/--
Aleatoric variance bound: The weighted aleatoric noise is bounded by the maximum
observation noise variance `noiseMax`.
-/
theorem Uncertainty.aleatoricBound
    {ι : Type*} (s : Finset ι)
    (weights : ι → ℝ) (variances : ι → ℝ)
    (noiseMax : ℝ)
    (hWeightsNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hSumPos : 0 < ∑ i ∈ s, weights i)
    (hVarianceBound : ∀ i ∈ s, variances i ≤ noiseMax) :
    (∑ i ∈ s, weights i * variances i) / (∑ i ∈ s, weights i) ≤ noiseMax := by
  rw [div_le_iff₀ hSumPos]
  have hSumBound : ∑ i ∈ s, weights i * variances i ≤ ∑ i ∈ s, weights i * noiseMax := by
    apply Finset.sum_le_sum
    intro i hi
    exact mul_le_mul_of_nonneg_left (hVarianceBound i hi) (hWeightsNonneg i hi)
  rw [← Finset.sum_mul] at hSumBound
  rw [mul_comm]
  exact hSumBound


/--
Total variance bound: When neighbor precision satisfies `W_min ≤ W` and observation
noise is bounded by `noiseMax`, total predictive variance is bounded by `1 / W_min + noiseMax`.
-/
theorem Uncertainty.totalBound
    {ι : Type*} (s : Finset ι)
    (weights : ι → ℝ) (variances : ι → ℝ)
    (W_min noiseMax : ℝ)
    (hW_min : 0 < W_min)
    (hWeightsNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hSumGe : W_min ≤ ∑ i ∈ s, weights i)
    (hVarianceBound : ∀ i ∈ s, variances i ≤ noiseMax) :
    (1 + ∑ i ∈ s, weights i * variances i) / (∑ i ∈ s, weights i) ≤
      1 / W_min + noiseMax := by
  have hSumPos : 0 < ∑ i ∈ s, weights i := lt_of_lt_of_le hW_min hSumGe
  have hDecomp : (1 + ∑ i ∈ s, weights i * variances i) / (∑ i ∈ s, weights i) =
      1 / (∑ i ∈ s, weights i) + (∑ i ∈ s, weights i * variances i) / (∑ i ∈ s, weights i) := by
    ring
  rw [hDecomp]
  have hEpistemic : 1 / (∑ i ∈ s, weights i) ≤ 1 / W_min :=
    one_div_le_one_div_of_le hW_min hSumGe
  have hAleatoric : (∑ i ∈ s, weights i * variances i) / (∑ i ∈ s, weights i) ≤ noiseMax :=
    Uncertainty.aleatoricBound s weights variances noiseMax hWeightsNonneg hSumPos hVarianceBound
  linarith

/--
Epistemic contrast: Candidates with greater neighbor support (`W_near > W_far`)
have strictly lower epistemic variance, driving exploration toward uncharted regions.
-/
theorem Uncertainty.epistemicContrast
    (W_near W_far : ℝ)
    (hFarPos : 0 < W_far)
    (hLt : W_far < W_near) :
    1 / W_near < 1 / W_far := by
  exact one_div_lt_one_div_of_lt hFarPos hLt

/--
Lipschitz quadratic variance envelope: If `|f x - f y| ≤ L * d`, then the squared variation
is bounded quadratically: `(f x - f y)^2 ≤ L^2 * d^2`. A distance variance model that scales
only linearly (`α * d`) fails to bound Lipschitz deviation at large distances.
-/
theorem Uncertainty.lipschitzVariance
    (fx fy L d : ℝ)
    (hLip : |fx - fy| ≤ L * d)
    (_hLNonneg : 0 ≤ L)
    (_hDNonneg : 0 ≤ d) :
    (fx - fy) ^ 2 ≤ L ^ 2 * d ^ 2 := by
  have hAbsSq : (fx - fy) ^ 2 = |fx - fy| ^ 2 := by
    rw [sq_abs]
  rw [hAbsSq]
  have hSqLe : |fx - fy| ^ 2 ≤ (L * d) ^ 2 := by
    nlinarith [abs_nonneg (fx - fy)]
  have hMulSq : (L * d) ^ 2 = L ^ 2 * d ^ 2 := by ring
  rw [hMulSq] at hSqLe
  exact hSqLe

/--
Paired difference variance reduction: For positively correlated evaluations with
correlation `0 < rho`, the variance of their difference is strictly less than the sum
of their individual variances: `v1 + v2 - 2 * rho * √(v1 * v2) < v1 + v2`.
-/
theorem Uncertainty.pairedReduction
    (v1 v2 rho : ℝ)
    (hV1Pos : 0 < v1)
    (hV2Pos : 0 < v2)
    (hRhoPos : 0 < rho) :
    v1 + v2 - 2 * rho * Real.sqrt (v1 * v2) < v1 + v2 := by
  have hSqrtPos : 0 < Real.sqrt (v1 * v2) := by
    apply Real.sqrt_pos.mpr
    exact mul_pos hV1Pos hV2Pos
  have hTermPos : 0 < 2 * rho * Real.sqrt (v1 * v2) := by
    positivity
  linarith

/--
Pure epistemic exploration: Separating epistemic uncertainty `sigmaEpistemic` from aleatoric noise
prevents exploration traps. A point with high aleatoric noise but zero epistemic uncertainty
receives zero exploration bonus.
-/
theorem Uncertainty.pureEpistemic
    (mu beta sigmaEpistemic _sigmaAleatoric : ℝ)
    (hEpistemicZero : sigmaEpistemic = 0) :
    mu + beta * sigmaEpistemic = mu := by
  rw [hEpistemicZero]
  ring

/--
Epistemic variance reduction identity:
Adding an observation with precision `w` reduces epistemic variance by `w / (W * (W + w))`.
-/
theorem Uncertainty.infoGain
    (W w : ℝ)
    (hW : 0 < W)
    (hw : 0 < w) :
    1 / W - 1 / (W + w) = w / (W * (W + w)) := by
  have hWw : W + w ≠ 0 := by linarith
  have hWne : W ≠ 0 := by linarith
  field_simp
  ring

/--
Diminishing returns of precision (Submodularity of exploration):
Adding precision `w` yields strictly greater variance reduction in unexplored regions
(`W_1 < W_2`) than in densely sampled regions.
-/
theorem Uncertainty.diminishGain
    (W_1 W_2 w : ℝ)
    (hW1 : 0 < W_1)
    (hLt : W_1 < W_2)
    (hw : 0 < w) :
    w / (W_2 * (W_2 + w)) < w / (W_1 * (W_1 + w)) := by
  have h1 : 0 < W_1 * (W_1 + w) := mul_pos hW1 (add_pos hW1 hw)
  have h2 : W_1 * (W_1 + w) < W_2 * (W_2 + w) := by nlinarith
  have h3 : 1 / (W_2 * (W_2 + w)) < 1 / (W_1 * (W_1 + w)) :=
    one_div_lt_one_div_of_lt h1 h2
  have hMul : w * (1 / (W_2 * (W_2 + w))) < w * (1 / (W_1 * (W_1 + w))) :=
    mul_lt_mul_of_pos_left h3 hw
  have hL : w * (1 / (W_2 * (W_2 + w))) = w / (W_2 * (W_2 + w)) := by ring
  have hR : w * (1 / (W_1 * (W_1 + w))) = w / (W_1 * (W_1 + w)) := by ring
  rw [hL, hR] at hMul
  exact hMul

/--
Paired correlation noise limit:
Under high token loss correlation `1 - eps ≤ rho`, the paired difference observation
variance contracts to at most `2 * eps * v`. As `eps → 0`, observation noise vanishes.
-/
theorem Uncertainty.noiseLimit
    (v rho eps : ℝ)
    (hv : 0 ≤ v)
    (hBound : 1 - eps ≤ rho) :
    v + v - 2 * rho * v ≤ 2 * eps * v := by
  nlinarith

/--
Aleatoric variance batch scaling:
Increasing the minibatch size by a factor `1 ≤ k` scales down the sample mean variance
from `sigmaSq / B` to `sigmaSq / (k * B)`.
-/
theorem Uncertainty.batchScale
    (sigmaSq B k : ℝ)
    (hSigma : 0 ≤ sigmaSq)
    (hB : 0 < B)
    (hk : 1 ≤ k) :
    sigmaSq / (k * B) ≤ sigmaSq / B := by
  have hkB : B ≤ k * B := by nlinarith
  exact div_le_div_of_nonneg_left hSigma hB hkB

end BauxResearch.Optimization
