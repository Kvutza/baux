import BauxResearch.Foundation
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Intervals

/-!
# Distance geometry and observation maps

Pullback symmetry, nonnegativity, and the triangle inequality follow from
Mathlib's metric laws. Objective regularity transfers only when the objective
factors through the supplied observation map. Conversely, any objective with
a finite Lipschitz bound in this pullback distance is constant on its zero-distance
fibers. A cheap observation map does not itself establish that property.

The sign-field results are real-arithmetic identities for exact binary signs.
The operation comparison uses an explicitly shared cost per coordinate; it
does not establish hardware throughput, bandwidth, or nanosecond latency.
-/

namespace BauxResearch.Optimization.Metric

open scoped BigOperators

variable {State Response : Type*}

/-- The behavioral pullback distance induced by an observation or response map. -/
def pullbackDist
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (x y : State) : ℝ :=
  dist (observe x) (observe y)

/-- Pullback distance is symmetric. -/
theorem pullbackSymm
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (x y : State) :
    pullbackDist observe x y = pullbackDist observe y x :=
  dist_comm (observe x) (observe y)

/-- Pullback distance satisfies the triangle inequality. -/
theorem pullbackTriangle
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (x y z : State) :
    pullbackDist observe x z ≤
      pullbackDist observe x y + pullbackDist observe y z :=
  dist_triangle (observe x) (observe y) (observe z)

/-- Pullback distance is non-negative. -/
theorem pullbackNonneg
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (x y : State) :
    0 ≤ pullbackDist observe x y :=
  dist_nonneg

/-- Downstream Lipschitz continuity under pullback distance: If objective factors through
the response map and is Lipschitz on responses, surrogate error is bounded by pullback distance. -/
theorem pullbackLipschitz
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (evaluator : Response → ℝ)
    (K : ℝ)
    (hLip : ∀ r1 r2, |evaluator r1 - evaluator r2| ≤ K * dist r1 r2)
    (x y : State) :
    |evaluator (observe x) - evaluator (observe y)| ≤
      K * pullbackDist observe x y :=
  hLip (observe x) (observe y)

/-- A regular objective must agree at states that the observation map cannot
distinguish. This is a necessary condition on a proposed cheap geometry. -/
theorem fiberNecessary [PseudoMetricSpace Response]
    (observe : State → Response) (objective : State → ℝ) (constant : ℝ)
    (regular : ∀ first second, |objective first - objective second| ≤
      constant * pullbackDist observe first second)
    (first second : State) (same : observe first = observe second) :
    objective first = objective second := by
  have bound := regular first second
  simpa [pullbackDist, same, abs_nonpos_iff, sub_eq_zero] using bound

/--
Convex combination deviation bound: Any convex combination of values deviating by at most `M`
from a reference `c` deviates by at most `M` from `c`.
-/
theorem convexBound
    {ι : Type*} (s : Finset ι)
    (weights : ι → ℝ) (values : ι → ℝ) (c M : ℝ)
    (hNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hSumPos : 0 < ∑ i ∈ s, weights i)
    (hDev : ∀ i ∈ s, |values i - c| ≤ M) :
    |(∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) - c| ≤ M := by
  have hLower : ∀ i ∈ s, c - M ≤ values i := by
    intro i hi
    have hAbs := (abs_le.mp (hDev i hi)).1
    linarith
  have hUpper : ∀ i ∈ s, values i ≤ c + M := by
    intro i hi
    have hAbs := (abs_le.mp (hDev i hi)).2
    linarith
  have hWeightedLower : (c - M) * (∑ i ∈ s, weights i) ≤ ∑ i ∈ s, weights i * values i := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have hW := hNonneg i hi
    have hV := hLower i hi
    nlinarith
  have hWeightedUpper : (∑ i ∈ s, weights i * values i) ≤ (c + M) * (∑ i ∈ s, weights i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have hW := hNonneg i hi
    have hV := hUpper i hi
    nlinarith
  have h1 : c - M ≤ (∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) := by
    rw [le_div_iff₀ hSumPos]
    linarith
  have h2 : (∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) ≤ c + M := by
    rw [div_le_iff₀ hSumPos]
    linarith
  rw [abs_le]
  constructor <;> linarith

/--
Under Lipschitz regularity of the response evaluator and bounded noise, each neighbor
value is within `K * radius + noise` of the true objective at the query state.
-/
theorem pullbackNeighbor
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (evaluator : Response → ℝ)
    (query neighbor : State)
    (value K radius noise : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ r1 r2, |evaluator r1 - evaluator r2| ≤ K * dist r1 r2)
    (hDist : pullbackDist observe query neighbor ≤ radius)
    (hNoise : |value - evaluator (observe neighbor)| ≤ noise) :
    |value - evaluator (observe query)| ≤ K * radius + noise := by
  have hObjDist : |evaluator (observe neighbor) - evaluator (observe query)| ≤
      K * dist (observe neighbor) (observe query) :=
    hLip (observe neighbor) (observe query)
  rw [dist_comm (observe neighbor) (observe query)] at hObjDist
  have hKDist : K * dist (observe query) (observe neighbor) ≤ K * radius :=
    mul_le_mul_of_nonneg_left hDist hK
  have hTriangle : |value - evaluator (observe query)| ≤
      |value - evaluator (observe neighbor)| +
      |evaluator (observe neighbor) - evaluator (observe query)| := by
    have hEq : value - evaluator (observe query) =
        (value - evaluator (observe neighbor)) +
        (evaluator (observe neighbor) - evaluator (observe query)) := by ring
    rw [hEq]
    exact abs_add_le _ _
  linarith

/--
Dimension-free surrogate trust error bound under behavioral pullback distance:
When the trust region of radius `radius` is defined in the response space, the ENN surrogate
estimation error is bounded by `2 * K * radius + noise`, with zero dependence on the
ambient parameter dimension or orthogonal unobserved coordinate drifts.
-/
theorem pullbackError
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (evaluator : Response → ℝ)
    (base query : State)
    (radius K noise : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ r1 r2, |evaluator r1 - evaluator r2| ≤ K * dist r1 r2)
    (hQuery : pullbackDist observe query base ≤ radius)
    {ι : Type*} (s : Finset ι)
    (points : ι → State)
    (values : ι → ℝ)
    (weights : ι → ℝ)
    (hPoints : ∀ i ∈ s, pullbackDist observe (points i) base ≤ radius)
    (hNoise : ∀ i ∈ s, |values i - evaluator (observe (points i))| ≤ noise)
    (hNonneg : ∀ i ∈ s, 0 ≤ weights i)
    (hSumPos : 0 < ∑ i ∈ s, weights i) :
    |(∑ i ∈ s, weights i * values i) / (∑ i ∈ s, weights i) - evaluator (observe query)| ≤
      2 * K * radius + noise := by
  have hDistTri : ∀ i ∈ s, pullbackDist observe query (points i) ≤ 2 * radius := by
    intro i hi
    have hTri := pullbackTriangle observe query base (points i)
    have hP := hPoints i hi
    rw [pullbackSymm observe (points i) base] at hP
    linarith
  have hNeighborDev : ∀ i ∈ s, |values i - evaluator (observe query)| ≤ 2 * K * radius + noise := by
    intro i hi
    have hVal := pullbackNeighbor observe evaluator query (points i) (values i) K (2 * radius) noise
      hK hLip (hDistTri i hi) (hNoise i hi)
    linarith
  exact convexBound s weights values (evaluator (observe query)) (2 * K * radius + noise)
    hNonneg hSumPos hNeighborDev

/--
Null-space drift invariance: Any perturbation between parameter states that leaves the
functional response invariant has exactly zero pullback distance.
-/
theorem nullDrift
    [PseudoMetricSpace Response]
    (observe : State → Response)
    (x y : State)
    (hNull : observe x = observe y) :
    pullbackDist observe x y = 0 := by
  dsimp [pullbackDist]
  rw [hNull]
  exact dist_self (observe y)

/--
Fiber insensitivity: Functionally invariant parameter changes produce zero objective discrepancy.
-/
theorem fiberZero
    (observe : State → Response)
    (evaluator : Response → ℝ)
    (x y : State)
    (hNull : observe x = observe y) :
    evaluator (observe x) - evaluator (observe y) = 0 := by
  rw [hNull]
  ring

/--
Under the declared linear operation model, fewer represented coordinates cost
less. The observation-map evaluation and source hardware costs are separate.
-/
theorem computeAdvantage
    (dimResponse parameterCount : ℕ)
    (costPerDim : ℝ)
    (hCost : 0 < costPerDim)
    (hDimLt : dimResponse < parameterCount) :
    (dimResponse : ℝ) * costPerDim < (parameterCount : ℝ) * costPerDim := by
  have hCast : (dimResponse : ℝ) < (parameterCount : ℝ) := by exact_mod_cast hDimLt
  exact mul_lt_mul_of_pos_right hCast hCost

/--
Hamming sign distance identity: For any coordinate where signs differ in `{-1, 1}`,
the squared Euclidean difference `(s1 - s2)^2` is exactly `4`.
-/
theorem signDiff
    (s1 s2 : ℝ)
    (h1 : s1 = 1 ∨ s1 = -1)
    (h2 : s2 = 1 ∨ s2 = -1)
    (hDiff : s1 ≠ s2) :
    (s1 - s2) ^ 2 = 4 := by
  rcases h1 with rfl | rfl <;> rcases h2 with rfl | rfl
  · contradiction
  · norm_num
  · norm_num
  · contradiction

/--
Identical signs yield zero squared Euclidean difference.
-/
theorem signSame
    (s1 s2 : ℝ)
    (hEq : s1 = s2) :
    (s1 - s2) ^ 2 = 0 := by
  rw [hEq]
  ring

/-!
### Finite output and paired-loss geometry

Half the L1 distance between normalized probability vectors is total variation.
The algebra below also applies to arbitrary real vectors; normalization is not
silently inferred from their type. A bounded reward has discrepancy at most
twice its bound times this distance. Equal output vectors have zero pullback
distance, but whether this is a useful or affordable task geometry remains open.

The paired-loss quantity below is mean absolute difference on aligned indices.
It is not defined by an infimum over couplings and is not identified here with
Wasserstein distance. The empirical mean bound follows from the finite-sum
triangle inequality. Neither population generalization nor zero memory traffic
is established by these finite-array identities.
-/

variable {Ω : Type*} [Fintype Ω]

noncomputable section

/-- Half-L1 discrepancy; total variation for normalized probability vectors. -/
def tvDist (p q : Ω → ℝ) : ℝ :=
  (1 / 2) * ∑ ω, |p ω - q ω|

/-- Total Variation distance is symmetric. -/
theorem tvSymm (p q : Ω → ℝ) : tvDist p q = tvDist q p := by
  dsimp [tvDist]
  congr 1
  apply Finset.sum_congr rfl
  intro ω _
  rw [abs_sub_comm]

/-- Total Variation distance is non-negative. -/
theorem tvNonneg (p q : Ω → ℝ) : 0 ≤ tvDist p q := by
  dsimp [tvDist]
  apply mul_nonneg (by norm_num)
  exact Finset.sum_nonneg fun ω _ => abs_nonneg _

/-- Total Variation distance satisfies the triangle inequality. -/
theorem tvTriangle (p q r : Ω → ℝ) :
    tvDist p r ≤ tvDist p q + tvDist q r := by
  dsimp [tvDist]
  rw [← mul_add]
  gcongr
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro ω _
  have h : p ω - r ω = (p ω - q ω) + (q ω - r ω) := by ring
  rw [h]
  exact abs_add_le (p ω - q ω) (q ω - r ω)

/--
Universal expectation bound under Total Variation distance:
For any reward bounded by `M`, the discrepancy in expected reward
is bounded by `2 * M * tvDist p q`.
-/
theorem tvBound
    (p q : Ω → ℝ)
    (reward : Ω → ℝ)
    (M : ℝ)
    (hM : ∀ ω, |reward ω| ≤ M) :
    |∑ ω, (p ω - q ω) * reward ω| ≤ 2 * M * tvDist p q := by
  have hLe : |∑ ω, (p ω - q ω) * reward ω| ≤ ∑ ω, |(p ω - q ω) * reward ω| :=
    Finset.abs_sum_le_sum_abs _ _
  have hTerm : ∀ ω, |(p ω - q ω) * reward ω| ≤ |p ω - q ω| * M := by
    intro ω
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (hM ω) (abs_nonneg _)
  have hSum : ∑ ω, |(p ω - q ω) * reward ω| ≤ ∑ ω, (|p ω - q ω| * M) :=
    Finset.sum_le_sum fun ω _ => hTerm ω
  rw [← Finset.sum_mul] at hSum
  have hTv : ∑ ω, |p ω - q ω| = 2 * tvDist p q := by
    dsimp [tvDist]
    ring
  rw [hTv] at hSum
  have hMul : (2 * tvDist p q) * M = 2 * M * tvDist p q := by ring
  rw [hMul] at hSum
  exact le_trans hLe hSum

/-- Pullback distributional distance induced on model parameter states. -/
def distribPullback
    (observe : State → (Ω → ℝ))
    (x y : State) : ℝ :=
  tvDist (observe x) (observe y)

/-- Gauge invariance: Parameter changes that leave output distribution invariant have distance zero. -/
theorem distribGauge
    (observe : State → (Ω → ℝ))
    (x y : State)
    (hEq : observe x = observe y) :
    distribPullback observe x y = 0 := by
  dsimp [distribPullback, tvDist]
  rw [hEq]
  have hZero : (∑ ω : Ω, |observe y ω - observe y ω|) = 0 := by
    simp
  rw [hZero]
  ring

variable {ι : Type*} [Fintype ι]

/--
Mean absolute discrepancy between paired loss vectors on the same finite index
set. This is a specified coupling, not an optimized distributional distance.
-/
def lossDist (l1 l2 : ι → ℝ) : ℝ :=
  (1 / (Fintype.card ι : ℝ)) * ∑ i, |l1 i - l2 i|

/-- Paired mean absolute discrepancy is symmetric. -/
theorem lossSymm (l1 l2 : ι → ℝ) : lossDist l1 l2 = lossDist l2 l1 := by
  dsimp [lossDist]
  congr 1
  apply Finset.sum_congr rfl
  intro i _
  rw [abs_sub_comm]

/-- Paired mean absolute discrepancy is non-negative. -/
theorem lossNonneg (l1 l2 : ι → ℝ) (hCard : 0 < Fintype.card ι) : 0 ≤ lossDist l1 l2 := by
  dsimp [lossDist]
  have hCardPos : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr hCard
  apply mul_nonneg (le_of_lt (one_div_pos.mpr hCardPos))
  exact Finset.sum_nonneg fun i _ => abs_nonneg _

/-- Paired mean absolute discrepancy satisfies the triangle inequality. -/
theorem lossTriangle (l1 l2 l3 : ι → ℝ) :
    lossDist l1 l3 ≤ lossDist l1 l2 + lossDist l2 l3 := by
  dsimp [lossDist]
  rw [← mul_add]
  have hCardNonneg : 0 ≤ 1 / (Fintype.card ι : ℝ) := by
    positivity
  apply mul_le_mul_of_nonneg_left _ hCardNonneg
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro i _
  have h : l1 i - l3 i = (l1 i - l2 i) + (l2 i - l3 i) := by ring
  rw [h]
  exact abs_add_le (l1 i - l2 i) (l2 i - l3 i)

/--
The finite-sum triangle inequality bounds the difference of empirical means by
paired mean absolute discrepancy. No optimal-coupling duality is invoked.
-/
theorem lossBound
    (l1 l2 : ι → ℝ)
    (hCard : 0 < Fintype.card ι) :
    |(1 / (Fintype.card ι : ℝ)) * ∑ i, l1 i - (1 / (Fintype.card ι : ℝ)) * ∑ i, l2 i| ≤
      lossDist l1 l2 := by
  have hCardPos : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr hCard
  have hDiff : (1 / (Fintype.card ι : ℝ)) * ∑ i, l1 i - (1 / (Fintype.card ι : ℝ)) * ∑ i, l2 i =
      (1 / (Fintype.card ι : ℝ)) * ∑ i, (l1 i - l2 i) := by
    rw [← mul_sub, ← Finset.sum_sub_distrib]
  rw [hDiff]
  rw [abs_mul, abs_of_pos (one_div_pos.mpr hCardPos)]
  have hSumLe : |∑ i, (l1 i - l2 i)| ≤ ∑ i, |l1 i - l2 i| :=
    Finset.abs_sum_le_sum_abs _ _
  dsimp [lossDist]
  exact mul_le_mul_of_nonneg_left hSumLe (le_of_lt (one_div_pos.mpr hCardPos))

/--
Finite-array bound on paired squared loss variation:
If the per-token loss discrepancy is bounded by `M`, the mean squared loss variation
is upper-bounded by `M * lossDist l1 l2`. This concerns a finite empirical second
moment, not a population variance or a convergence theorem for the source law.
-/
theorem lossSquare
    (l1 l2 : ι → ℝ)
    (M : ℝ)
    (hCard : 0 < Fintype.card ι)
    (hM : ∀ i, |l1 i - l2 i| ≤ M) :
    (1 / (Fintype.card ι : ℝ)) * ∑ i, (l1 i - l2 i) ^ 2 ≤
      M * lossDist l1 l2 := by
  have hTerm : ∀ i, (l1 i - l2 i) ^ 2 ≤ |l1 i - l2 i| * M := by
    intro i
    have hSq : (l1 i - l2 i) ^ 2 = |l1 i - l2 i| ^ 2 := by rw [sq_abs]
    rw [hSq, sq]
    exact mul_le_mul_of_nonneg_left (hM i) (abs_nonneg _)
  have hSum : ∑ i, (l1 i - l2 i) ^ 2 ≤ ∑ i, (|l1 i - l2 i| * M) :=
    Finset.sum_le_sum fun i _ => hTerm i
  rw [← Finset.sum_mul] at hSum
  have hCardPos : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr hCard
  have hCardNonneg : 0 ≤ 1 / (Fintype.card ι : ℝ) := le_of_lt (one_div_pos.mpr hCardPos)
  have hLe := mul_le_mul_of_nonneg_left hSum hCardNonneg
  have hRing : (1 / (Fintype.card ι : ℝ)) * ((∑ i, |l1 i - l2 i|) * M) =
      M * ((1 / (Fintype.card ι : ℝ)) * ∑ i, |l1 i - l2 i|) := by ring
  rw [hRing] at hLe
  dsimp [lossDist]
  exact hLe

/-- Antipodal projective quotient distance `min (dist u v) (dist u (-v))` on a
normed parameter space, identifying opposite perturbation directions `±v` that
share identical second-order directional curvature along their common line. -/
def projectiveDist {Space : Type*} [NormedAddCommGroup Space]
    (u v : Space) : ℝ :=
  min (dist u v) (dist u (-v))

/-- Antipodal projective distance is non-negative. -/
theorem projectiveNonneg {Space : Type*} [NormedAddCommGroup Space]
    (u v : Space) : 0 ≤ projectiveDist u v := by
  unfold projectiveDist
  exact le_min dist_nonneg dist_nonneg

/-- Antipodal projective distance is invariant under negating the second
argument (`±v` sign-flip symmetry). -/
theorem projectiveFlip {Space : Type*} [NormedAddCommGroup Space]
    (u v : Space) : projectiveDist u (-v) = projectiveDist u v := by
  unfold projectiveDist
  rw [neg_neg, min_comm]

/-- Antipodal projective distance is symmetric. -/
theorem projectiveSymm {Space : Type*} [NormedAddCommGroup Space]
    (u v : Space) : projectiveDist u v = projectiveDist v u := by
  unfold projectiveDist
  have oppEq : dist u (-v) = dist v (-u) := by
    rw [dist_eq_norm, dist_eq_norm]
    have eqSub : u - -v = v - -u := by abel
    rw [eqSub]
  rw [dist_comm u v, oppEq]

/-- Antipodal projective distance is bounded above by ambient distance. -/
theorem projectiveLe {Space : Type*} [NormedAddCommGroup Space]
    (u v : Space) : projectiveDist u v ≤ dist u v :=
  min_le_left _ _

/-- Admissible-coupling transport cost induced by a finite index bijection `σ`
(a vertex of the Birkhoff polytope).  Evaluating any candidate bijection on the
true 1D ground distance yields a valid upper bound on batch mean discrepancy
with zero tree-metric underestimation remainder. -/
def couplingCost (σ : Equiv.Perm ι) (l1 l2 : ι → ℝ) : ℝ :=
  (1 / (Fintype.card ι : ℝ)) * ∑ i, |l1 i - l2 (σ i)|

/-- Every finite index bijection `σ` bounds the difference of empirical means
by `couplingCost σ l1 l2` with zero underestimation remainder. -/
theorem couplingBound
    (σ : Equiv.Perm ι) (l1 l2 : ι → ℝ)
    (hCard : 0 < Fintype.card ι) :
    |(1 / (Fintype.card ι : ℝ)) * ∑ i, l1 i -
      (1 / (Fintype.card ι : ℝ)) * ∑ i, l2 i| ≤
      couplingCost σ l1 l2 := by
  have hCardPos : 0 < (Fintype.card ι : ℝ) := Nat.cast_pos.mpr hCard
  have hPermSum : (∑ i, l2 (σ i)) = ∑ i, l2 i :=
    Equiv.sum_comp σ l2
  rw [← hPermSum]
  have hDiff :
      (1 / (Fintype.card ι : ℝ)) * ∑ i, l1 i -
        (1 / (Fintype.card ι : ℝ)) * ∑ i, l2 (σ i) =
      (1 / (Fintype.card ι : ℝ)) * ∑ i, (l1 i - l2 (σ i)) := by
    rw [← mul_sub, ← Finset.sum_sub_distrib]
  rw [hDiff, abs_mul, abs_of_pos (one_div_pos.mpr hCardPos)]
  have hSumLe : |∑ i, (l1 i - l2 (σ i))| ≤ ∑ i, |l1 i - l2 (σ i)| :=
    Finset.abs_sum_le_sum_abs _ _
  dsimp [couplingCost]
  exact mul_le_mul_of_nonneg_left hSumLe (le_of_lt (one_div_pos.mpr hCardPos))

/-- When an admissible bijection `σ` reduces transport cost relative to the
identity index pairing, it sandwiches empirical mean discrepancy more tightly
than `lossDist` without introducing any underestimation remainder. -/
theorem couplingTighter
    (σ : Equiv.Perm ι) (l1 l2 : ι → ℝ)
    (hCard : 0 < Fintype.card ι)
    (hImproved : couplingCost σ l1 l2 ≤ lossDist l1 l2) :
    |(1 / (Fintype.card ι : ℝ)) * ∑ i, l1 i -
      (1 / (Fintype.card ι : ℝ)) * ∑ i, l2 i| ≤
      couplingCost σ l1 l2 ∧
    couplingCost σ l1 l2 ≤ lossDist l1 l2 :=
  ⟨couplingBound σ l1 l2 hCard, hImproved⟩

end

#print axioms fiberNecessary
#print axioms pullbackError
#print axioms nullDrift
#print axioms projectiveSymm
#print axioms projectiveFlip
#print axioms couplingBound
#print axioms couplingTighter

end BauxResearch.Optimization.Metric
