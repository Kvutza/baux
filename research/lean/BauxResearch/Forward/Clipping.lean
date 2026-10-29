import Mathlib.Topology.MetricSpace.Lipschitz

/-!
# Nonsmooth scalar controls

The source uses maximum floors and interval clipping. They need not be
differentiable at their boundaries, but Mathlib proves the exact global
Lipschitz property needed for compositional forward bounds.
-/

namespace BauxResearch.Forward.Clipping

noncomputable section

/-- Lower flooring used by transport normalization and exponential seeding. -/
def floor (lower value : ℝ) : ℝ := max value lower

/-- Closed-interval-style clipping used by the mHC residual blend. The
Lipschitz result below does not require an ordering premise on the endpoints. -/
def clip (lower upper value : ℝ) : ℝ := min (max value lower) upper

theorem floorLipschitz (lower : ℝ) : LipschitzWith 1 (floor lower) := by
  change LipschitzWith 1 (fun value : ℝ => max value lower)
  exact LipschitzWith.id.max_const lower

theorem clipLipschitz (lower upper : ℝ) : LipschitzWith 1 (clip lower upper) := by
  change LipschitzWith 1 (fun value : ℝ => min (max value lower) upper)
  exact (LipschitzWith.id.max_const lower).min_const upper

theorem sourceClip (value : ℝ) : clip 0 1 value = min 1 (max 0 value) := by
  simp only [clip, min_comm, max_comm]

#print axioms floorLipschitz
#print axioms clipLipschitz
#print axioms sourceClip

end

end BauxResearch.Forward.Clipping
