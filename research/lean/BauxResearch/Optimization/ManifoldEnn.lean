import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# Riemannian Manifold ENN Geometry, Subspace Shrinkage, and Conformal Calibration

Formalizes the four geometric and statistical pillars of the Riemannian Manifold
Epistemic Nearest Neighbors (ENN) architecture in `baux::semantics`:

1. **Poincaré Upper Half-Space Conformal Local Scaling (`neighbor.rs`):**
   Embeds each observation `(x_i, s_i)` with positive local scale `s_i > 0` into
   the Poincaré upper half-space `ℍ^{d+1} = ℝ^d × ℝ_{>0}` with Riemannian metric
   `ds² = (‖dx‖₂² + ds²) / s²`. The isometric invariant
   `δ(u, v) = (b² + (s_q - s)²) / (2 s_q s)` decomposes relative to the radial
   distance `r = √(b² + s_q²)` from the query mirror point `(x_q, -s_q)` as
   `δ - (r - s_q) / s_q = (s - r)² / (2 s_q s) ≥ 0`, yielding the exact
   branch-and-bound lower envelope `(r - s_q) / s_q ≤ δ` achieved when `s = r`.
   - H. Poincaré (1882), *Théorie des groupes fuchsiens*, Acta Math. 1, 1–62.
   - J. W. Cannon, W. J. Floyd, R. Kenyon, and W. R. Parry (1997), *Hyperbolic
     Geometry*, Flavors of Geometry, MSRI Publ. 31, 59–115.
   - L. Zelnik-Manor and P. Perona (2004), *Self-Tuning Spectral Clustering*,
     NeurIPS 17, 1601–1608.
   - M. Nickel and D. Kiela (2017), *Poincaré Embeddings for Learning
     Hierarchical Representations*, NeurIPS 30 (arXiv:1705.08039).
   - Conformal local-scale half-space dissimilarity and branch-and-bound pruning
     (arXiv:2305.01356).

2. **Orthogonal-Subspace James–Stein Shrinkage (`shrinkage.rs`):**
   Decomposes the local surrogate or trajectory displacement into a low-rank
   signal subspace with variance `v_∥` and an orthogonal complement with variance
   `v_⊥ > 0` and squared bias `b_⊥² ≥ 0`. Leaving the signal subspace unshrunk
   while contracting the orthogonal complement by the empirical-Bayes James–Stein
   weight `α* = b_⊥² / (b_⊥² + v_⊥)` reduces risk to
   `v_∥ + (v_⊥ b_⊥²) / (b_⊥² + v_⊥)`, strictly dominating the unshrunk estimator
   (`α = 1`, risk `v_∥ + v_⊥`) whenever `v_⊥ > 0`.
   - C. Stein (1956), *Inadmissibility of the Usual Estimator for the Mean of a
     Multivariate Normal Distribution*, Proc. Third Berkeley Symp. Math. Statist.
     Probab. 1, 197–206.
   - W. James and C. Stein (1961), *Estimation with Quadratic Loss*, Proc. Fourth
     Berkeley Symp. Math. Statist. Probab. 1, 361–379.
   - B. Efron and C. Morris (1973), *Stein's Estimation Rule and Its Competitors—
     An Empirical Bayes Approach*, J. Amer. Statist. Assoc. 68(341), 117–130.

3. **Fisher–Rao GLS & Diagonal Mahalanobis Rademacher Projection (`calibration.rs`):**
   Under a diagonal metric tensor `g : Fin w → ℝ` and constant-amplitude codes
   `e_j² = c²` (such as Rademacher vertices `e_j ∈ {-c, +c}`), the Mahalanobis
   squared distance `∑ j, g_j (v_j - e_j)²` expands into a code-independent base
   energy `∑ j, g_j v_j² + c² ∑ j, g_j` minus `2 ∑ j, g_j v_j e_j`, reversing the
   order between squared distance and the metric inner product.
   - A. C. Aitken (1935), *On Least Squares and Linear Combination of
     Observations*, Proc. R. Soc. Edinb. 55, 42–48.
   - S.-i. Amari (1985), *Differential-Geometrical Methods in Statistics*,
     Lecture Notes in Statistics 28, Springer.
   - V. Vovk, A. Gammerman, and G. Shafer (2005), *Algorithmic Learning in a
     Random World*, Springer.
   - J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman (2018),
     *Distribution-Free Predictive Inference for Regression*, J. Amer. Statist.
     Assoc. 113(523), 1094–1111 (arXiv:1604.04173).

4. **Johnson–Lindenstrauss & Fisher–Rao Hellinger Simplex Embeddings (`categorical.rs`):**
   Maps probability distributions `p, q ∈ Δ^{d-1}` via the square-root Hellinger
   chart `ψ_i = √p_i, φ_i = √q_i` onto the unit sphere `∑ i, ψ_i² = ∑ i, φ_i² = 1`,
   where the Euclidean chord squared distance `∑ i, (ψ_i - φ_i)²` equals
   `2 (1 - ∑ i, ψ_i φ_i)` in terms of the Bhattacharyya coefficient.
   - W. B. Johnson and J. Lindenstrauss (1984), *Extensions of Lipschitz Mappings
     into a Hilbert Space*, Contemp. Math. 26, 189–206.
   - D. Achlioptas (2003), *Database-Friendly Random Projections:
     Johnson-Lindenstrauss with Binary Coins*, J. Comput. System Sci. 66(4),
     671–687.
   - A. Bhattacharyya (1943), *On a Measure of Divergence between Two Statistical
     Populations Defined by Their Probability Distributions*, Bull. Calcutta Math.
     Soc. 35, 99–109.
   - N. N. Čencov (1982), *Statistical Decision Rules and Optimal Inference*,
     Translations of Mathematical Monographs 53, Amer. Math. Soc.
-/

namespace BauxResearch.Optimization.ManifoldEnn

noncomputable section
open scoped BigOperators

/-- Poincaré upper half-space isometric invariant
`δ(u, v) = (b² + (s_q - s)²) / (2 s_q s)` for base Euclidean distance `b` and
positive vertical scales `s_q, s`. -/
def halfspaceDelta (b sq s : ℝ) : ℝ :=
  (b ^ 2 + (sq - s) ^ 2) / (2 * sq * s)

/-- Conformal lower envelope `(r - s_q) / s_q` of `halfspaceDelta` in terms of the
mirror-point Euclidean radius `r = √(b² + s_q²)`. -/
def halfspaceFloor (r sq : ℝ) : ℝ :=
  (r - sq) / sq

/-- Exact algebraic decomposition of the Poincaré half-space invariant `δ` above
its radial lower envelope `(r - s_q) / s_q`. -/
theorem halfspace_decomp (b sq s r : ℝ) (hsq : sq ≠ 0) (hs : s ≠ 0)
    (hr : b ^ 2 + sq ^ 2 = r ^ 2) :
    halfspaceDelta b sq s - halfspaceFloor r sq = (s - r) ^ 2 / (2 * sq * s) := by
  unfold halfspaceDelta halfspaceFloor
  have hRad : b ^ 2 + (sq - s) ^ 2 = r ^ 2 - 2 * sq * s + s ^ 2 := by
    calc
      b ^ 2 + (sq - s) ^ 2 = (b ^ 2 + sq ^ 2) - 2 * sq * s + s ^ 2 := by ring
      _ = r ^ 2 - 2 * sq * s + s ^ 2 := by rw [hr]
  rw [hRad]
  field_simp [hsq, hs]
  ring

/-- Branch-and-bound lower bound on the Poincaré half-space invariant: for any
positive scales `s_q, s > 0` and mirror radius `b² + s_q² = r²`,
`halfspaceFloor r sq ≤ halfspaceDelta b sq s`. -/
theorem halfspace_bound (b sq s r : ℝ) (hsq : 0 < sq) (hs : 0 < s)
    (hr : b ^ 2 + sq ^ 2 = r ^ 2) :
    halfspaceFloor r sq ≤ halfspaceDelta b sq s := by
  have hDecomp :
      halfspaceDelta b sq s - halfspaceFloor r sq = (s - r) ^ 2 / (2 * sq * s) :=
    halfspace_decomp b sq s r (ne_of_gt hsq) (ne_of_gt hs) hr
  have hNonneg : 0 ≤ (s - r) ^ 2 / (2 * sq * s) :=
    div_nonneg (sq_nonneg (s - r)) (by positivity)
  linarith

/-- Quadratic risk `v_∥ + α² v_⊥ + (1 - α)² b_⊥²` of an orthogonal-subspace
estimator that preserves the tangent subspace and scales the normal subspace by
`α`. -/
def steinRisk (vPar vPerp bPerp alpha : ℝ) : ℝ :=
  vPar + alpha ^ 2 * vPerp + (1 - alpha) ^ 2 * bPerp ^ 2

/-- Optimal empirical-Bayes shrinkage weight `α* = b_⊥² / (b_⊥² + v_⊥)` on the
orthogonal complement. -/
def steinWeight (vPerp bPerp : ℝ) : ℝ :=
  bPerp ^ 2 / (bPerp ^ 2 + vPerp)

/-- Exact risk identity at the optimal orthogonal-subspace shrinkage weight
`steinWeight vPerp bPerp`. -/
theorem stein_risk (vPar vPerp bPerp : ℝ) (hDen : bPerp ^ 2 + vPerp ≠ 0) :
    steinRisk vPar vPerp bPerp (steinWeight vPerp bPerp) =
      vPar + (vPerp * bPerp ^ 2) / (bPerp ^ 2 + vPerp) := by
  unfold steinRisk steinWeight
  field_simp [hDen]
  ring

/-- Strict risk domination of orthogonal-subspace shrinkage over the unshrunk
estimator (`α = 1`) whenever the normal-subspace variance `v_⊥` is strictly
positive. -/
theorem stein_lt (vPar vPerp bPerp : ℝ) (hv : 0 < vPerp) (hb : 0 ≤ bPerp ^ 2) :
    steinRisk vPar vPerp bPerp (steinWeight vPerp bPerp) <
      steinRisk vPar vPerp bPerp 1 := by
  have hDenPos : 0 < bPerp ^ 2 + vPerp := add_pos_of_nonneg_of_pos hb hv
  have hDenNe : bPerp ^ 2 + vPerp ≠ 0 := ne_of_gt hDenPos
  rw [stein_risk vPar vPerp bPerp hDenNe]
  unfold steinRisk
  have hGap :
      (vPar + 1 ^ 2 * vPerp + (1 - 1) ^ 2 * bPerp ^ 2) -
        (vPar + (vPerp * bPerp ^ 2) / (bPerp ^ 2 + vPerp)) =
        vPerp ^ 2 / (bPerp ^ 2 + vPerp) := by
    field_simp [hDenNe]
    ring
  have hPos : 0 < vPerp ^ 2 / (bPerp ^ 2 + vPerp) :=
    div_pos (sq_pos_of_pos hv) hDenPos
  linarith

alias stein_opt_lt := stein_lt

/-- Diagonal Mahalanobis squared distance `∑ j, g_j (v_j - e_j)²`. -/
def mahalanobisSq {w : ℕ} (g v e : Fin w → ℝ) : ℝ :=
  ∑ j : Fin w, g j * (v j - e j) ^ 2

/-- Metric-weighted inner product `∑ j, g_j v_j e_j`. -/
def metricDot {w : ℕ} (g v e : Fin w → ℝ) : ℝ :=
  ∑ j : Fin w, g j * v j * e j

/-- Constant-amplitude decomposition of diagonal Mahalanobis squared distance into
weighted vector energy, code energy, and metric inner product. -/
theorem mahalanobis_dot {w : ℕ} (g v e : Fin w → ℝ) (c : ℝ)
    (hc : ∀ j : Fin w, (e j) ^ 2 = c ^ 2) :
    mahalanobisSq g v e =
      (∑ j : Fin w, g j * (v j) ^ 2) + c ^ 2 * (∑ j : Fin w, g j) -
        2 * metricDot g v e := by
  unfold mahalanobisSq metricDot
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  have hj := hc j
  calc
    g j * (v j - e j) ^ 2 =
        g j * (v j) ^ 2 + g j * (e j) ^ 2 - 2 * (g j * v j * e j) := by ring
    _ = g j * (v j) ^ 2 + c ^ 2 * g j - 2 * (g j * v j * e j) := by
        rw [hj]
        ring

/-- Order reversal between diagonal Mahalanobis squared distance and the metric
inner product across constant-amplitude codewords `e₁, e₂`. -/
theorem mahalanobis_order {w : ℕ} (g v e1 e2 : Fin w → ℝ) (c : ℝ)
    (hc1 : ∀ j, (e1 j) ^ 2 = c ^ 2) (hc2 : ∀ j, (e2 j) ^ 2 = c ^ 2) :
    mahalanobisSq g v e1 ≤ mahalanobisSq g v e2 ↔
      metricDot g v e2 ≤ metricDot g v e1 := by
  rw [mahalanobis_dot g v e1 c hc1, mahalanobis_dot g v e2 c hc2]
  constructor <;> intro h <;> linarith

/-- Squared Hellinger chord distance `∑ i, (ψ_i - φ_i)²` on the root-density
sphere. -/
def hellingerSq {d : ℕ} (psi phi : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, (psi i - phi i) ^ 2

/-- Bhattacharyya fidelity inner product `∑ i, ψ_i φ_i` on the root-density
sphere. -/
def bhattacharyya {d : ℕ} (psi phi : Fin d → ℝ) : ℝ :=
  ∑ i : Fin d, psi i * phi i

/-- Exact identity `hellingerSq ψ φ = 2 (1 - bhattacharyya ψ φ)` for unit
root-density embeddings `∑ i, ψ_i² = ∑ i, φ_i² = 1`. -/
theorem hellinger_sq {d : ℕ} (psi phi : Fin d → ℝ)
    (hPsi : ∑ i : Fin d, (psi i) ^ 2 = 1)
    (hPhi : ∑ i : Fin d, (phi i) ^ 2 = 1) :
    hellingerSq psi phi = 2 * (1 - bhattacharyya psi phi) := by
  unfold hellingerSq bhattacharyya
  have hExpand :
      (∑ i : Fin d, (psi i - phi i) ^ 2) =
        (∑ i : Fin d, (psi i) ^ 2) + (∑ i : Fin d, (phi i) ^ 2) -
          2 * (∑ i : Fin d, psi i * phi i) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hExpand, hPsi, hPhi]
  ring

#print axioms halfspace_decomp
#print axioms halfspace_bound
#print axioms stein_risk
#print axioms stein_lt
#print axioms stein_opt_lt
#print axioms mahalanobis_dot
#print axioms mahalanobis_order
#print axioms hellinger_sq

end
end BauxResearch.Optimization.ManifoldEnn
