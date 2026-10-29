import BauxResearch.Optimization.Guide
import Mathlib.Analysis.Convex.Strong

/-!
# Constructing a proposal from two retained parameter states

This is a conditional construction study, not a replacement production optimizer.
Mathlib's `UniformConvexOn` supplies a chord upper envelope. The observed endpoint
values and a proved convexity modulus determine an explicit interpolation step;
neither the optimum nor a derivative is an input. The output need not equal either
input state, so selecting from a fixed candidate pool is a different operation.

The modulus is a mathematical obligation, NOT a quantity certified by fitting two
observations. No convexity property is asserted for the actual pretraining loss.
Retained states must be materializable, observation errors refer to one fixed
objective, and execution error is charged separately. No backend is changed.
-/

namespace BauxResearch.Optimization.Interpolation

noncomputable section

open scoped NNReal BigOperators

variable {Space : Type*} [NormedAddCommGroup Space] [NormedSpace ℝ Space]

/-- A full-state interpolation, not a scalar objective interpolation. -/
def point (incumbent retained : Space) (fraction : ℝ) : Space :=
  (1 - fraction) • incumbent + fraction • retained

/-- The upper-envelope minimizer when `0 ≤ gap < bend`. -/
def step (bend gap : ℝ) : ℝ := (bend - gap) / (2 * bend)

/-- The certified ideal gain at the unconstrained envelope minimizer. -/
def margin (bend gap : ℝ) : ℝ := (bend - gap)^2 / (4 * bend)

/-- The value-only upper envelope of a chord. -/
def envelope (first second bend fraction : ℝ) : ℝ :=
  (1 - fraction) * first + fraction * second - fraction * (1 - fraction) * bend

theorem stepRange (bend gap : ℝ) (positive : 0 < bend)
    (nonnegative : 0 ≤ gap) (useful : gap < bend) :
    0 < step bend gap ∧ step bend gap ≤ 1 / 2 := by
  constructor
  · exact div_pos (sub_pos.mpr useful) (by positivity)
  · unfold step
    apply (div_le_iff₀ (show 0 < 2 * bend by positivity)).2
    linarith

/-- This is the exact best value of the scalar envelope, not a search oracle. -/
theorem envelopeMin (first second bend fraction : ℝ) (positive : 0 < bend) :
    envelope first second bend fraction =
      first - margin bend (second - first) +
        bend * (fraction - step bend (second - first))^2 := by
  unfold envelope margin step
  field_simp
  ring

theorem envelopeBest (first second bend fraction : ℝ) (positive : 0 < bend) :
    first - margin bend (second - first) ≤ envelope first second bend fraction := by
  rw [envelopeMin first second bend fraction positive]
  exact le_add_of_nonneg_right (mul_nonneg positive.le (sq_nonneg _))

theorem stepGain (bend gap : ℝ) (positive : 0 < bend) :
    step bend gap * (1 - step bend gap) * bend - step bend gap * gap =
      margin bend gap := by
  unfold step margin
  field_simp
  ring

theorem marginPositive (bend gap : ℝ) (positive : 0 < bend) (useful : gap < bend) :
    0 < margin bend gap := by
  exact div_pos (sq_pos_of_pos (sub_pos.mpr useful)) (by positivity)

theorem stepHalf (bend : ℝ) (positive : 0 < bend) : step bend 0 = 1 / 2 := by
  unfold step
  field_simp
  ring

/-- Mathlib's existing modulus, evaluated at the endpoint separation, is enough.
The convex region can be just the segment being considered, not the whole model
parameter space. A proof on that region is still required. -/
theorem chordBound (objective : Space → ℝ) (region : Set Space) (modulus : ℝ → ℝ)
    (regular : UniformConvexOn region modulus objective)
    (incumbent retained : Space) (first : incumbent ∈ region) (second : retained ∈ region)
    (fraction : ℝ) (nonnegative : 0 ≤ fraction) (atMost : fraction ≤ 1) :
    objective (point incumbent retained fraction) ≤
      envelope (objective incumbent) (objective retained)
        (modulus ‖incumbent - retained‖) fraction := by
  simpa only [point, envelope, smul_eq_mul, mul_comm] using
    regular.2 first second (sub_nonneg.mpr atMost) nonnegative (by ring)

/-- A bound on the endpoint objective gap supplies an explicit useful candidate.
The gap can come from noisy observations; it need not be an exact objective value.
Unlike an assumed accurate optimum anchor, the state is constructed here. -/
theorem proposalGain (objective : Space → ℝ) (region : Set Space) (modulus : ℝ → ℝ)
    (regular : UniformConvexOn region modulus objective)
    (incumbent retained : Space) (first : incumbent ∈ region) (second : retained ∈ region)
    (gap : ℝ) (nonnegative : 0 ≤ gap)
    (observed : objective retained - objective incumbent ≤ gap)
    (useful : gap < modulus ‖incumbent - retained‖) :
    margin (modulus ‖incumbent - retained‖) gap ≤
      progress objective incumbent
        (point incumbent retained (step (modulus ‖incumbent - retained‖) gap)) := by
  let bend := modulus ‖incumbent - retained‖
  have positive : 0 < bend := nonnegative.trans_lt useful
  have range := stepRange bend gap positive nonnegative useful
  have bound := chordBound objective region modulus regular incumbent retained first second
    (step bend gap) range.1.le (range.2.trans (by norm_num))
  have weighted := mul_le_mul_of_nonneg_left observed range.1.le
  have identity := stepGain bend gap positive
  change _ ≤ objective incumbent - _
  dsimp only [envelope] at bound
  change objective (point incumbent retained (step bend gap)) ≤ _ at bound
  change margin bend gap ≤ _
  nlinarith

omit [NormedAddCommGroup Space] [NormedSpace ℝ Space] in
/-- Two endpoint observation errors bound the true endpoint gap. They do not
justify the convexity modulus or stationarity of changing corpus windows. -/
theorem observedGap (objective : Space → ℝ) (incumbent retained : Space)
    (firstScore secondScore firstError secondError : ℝ)
    (first : |firstScore - objective incumbent| ≤ firstError)
    (second : |secondScore - objective retained| ≤ secondError) :
    objective retained - objective incumbent ≤
      secondScore - firstScore + firstError + secondError := by
  have lower := (abs_le.mp first).2
  have upper := (abs_le.mp second).1
  linarith

omit [NormedSpace ℝ Space] in
/-- Any materialization/rounding error is charged after constructing the ideal
state. This does not prove an error bound for the backend BF16 converter. -/
theorem executionGain (objective : Space → ℝ) (constant : ℝ≥0)
    (regular : LipschitzWith constant objective)
    (incumbent ideal executed : Space) (gain error : ℝ)
    (idealGain : gain ≤ progress objective incumbent ideal)
    (execution : dist executed ideal ≤ error) :
    gain - constant * error ≤ progress objective incumbent executed := by
  have bound := (regular.dist_le_mul executed ideal).trans
    (mul_le_mul_of_nonneg_left execution constant.coe_nonneg)
  rw [Real.dist_eq] at bound
  have upper := (abs_le.mp bound).2
  unfold progress at *
  linarith

omit [NormedSpace ℝ Space] in
/-- Reserve the execution-error budget INSIDE the ideal trust radius if the
executed, rather than merely ideal, state must satisfy that radius. -/
theorem executionTrust (incumbent ideal executed : Space) (radius error : ℝ)
    (execution : dist executed ideal ≤ error)
    (interior : dist ideal incumbent ≤ radius - error) :
    dist executed incumbent ≤ radius := by
  calc
    dist executed incumbent ≤ dist executed ideal + dist ideal incumbent := dist_triangle _ _ _
    _ ≤ error + (radius - error) := add_le_add execution interior
    _ = radius := by ring

/-- A trust radius clips the scalar step without a candidate-search loop. -/
def capped (bend gap radius separation : ℝ) : ℝ :=
  min (step bend gap) (radius / separation)

theorem capRange (bend gap radius separation : ℝ)
    (positive : 0 < bend) (nonnegative : 0 ≤ gap) (useful : gap < bend)
    (radiusPositive : 0 < radius) (separated : 0 < separation) :
    0 < capped bend gap radius separation ∧
      capped bend gap radius separation ≤ step bend gap := by
  exact ⟨lt_min (stepRange bend gap positive nonnegative useful).1
    (div_pos radiusPositive separated), min_le_left _ _⟩

theorem capRadius (bend gap radius separation : ℝ) (separated : 0 < separation) :
    capped bend gap radius separation * separation ≤ radius := by
  exact (le_div_iff₀ separated).mp (min_le_right _ _)

theorem pointDistance (incumbent retained : Space) (fraction : ℝ) :
    dist (point incumbent retained fraction) incumbent =
      |fraction| * dist retained incumbent := by
  rw [dist_eq_norm, dist_eq_norm]
  have identity : point incumbent retained fraction - incumbent =
      fraction • (retained - incumbent) := by unfold point; module
  rw [identity, norm_smul, Real.norm_eq_abs]

/-- Every positive step up to the explicit minimizer keeps a certified gain.
This includes radius-clipped steps; clipping can reduce the gain arbitrarily. -/
theorem clippedGain (bend gap fraction : ℝ) (positive : 0 < bend)
    (nonnegative : 0 ≤ fraction) (bounded : fraction ≤ step bend gap) :
    fraction * (bend - gap) / 2 ≤
      fraction * (1 - fraction) * bend - fraction * gap := by
  have bound := (le_div_iff₀ (show 0 < 2 * bend by positivity)).mp bounded
  have scaled := mul_le_mul_of_nonneg_left bound nonnegative
  nlinarith

/-- With nonnegative bending, an endpoint gap at least that large leaves this
envelope unable to certify any positive improvement inside the chord. This is
not a lower bound on the true objective or a reason to reject another generator. -/
theorem noCertificate (bend gap fraction : ℝ) (nonnegative : 0 ≤ bend)
    (large : bend ≤ gap) (fractionNonneg : 0 ≤ fraction) :
    fraction * (1 - fraction) * bend - fraction * gap ≤ 0 := by
  have bound := mul_le_mul_of_nonneg_left large fractionNonneg
  have square := mul_nonneg nonnegative (sq_nonneg fraction)
  nlinarith

/-- Convexity by itself does not supply a positive modulus: a flat objective
admits no positive bending at any retained endpoint separation. -/
theorem flatModulus (region : Set Space) (modulus : ℝ → ℝ) (value : ℝ)
    (regular : UniformConvexOn region modulus (fun _ : Space => value))
    (incumbent retained : Space) (first : incumbent ∈ region) (second : retained ∈ region) :
    modulus ‖incumbent - retained‖ ≤ 0 := by
  have bound : value ≤ (1 / 2 : ℝ) * value + (1 / 2 : ℝ) * value -
      (1 / 2 : ℝ) * (1 / 2 : ℝ) * modulus ‖incumbent - retained‖ :=
    regular.2 first second (by norm_num) (by norm_num) (by norm_num)
  linarith

/-- A conservative lower bound on the modulus is sufficient for constructing
the step. In particular the algorithm need not compute a full-state norm when
a separately proved sampled-distance certificate supplies this bound. -/
theorem certifiedGain (objective : Space → ℝ) (region : Set Space) (modulus : ℝ → ℝ)
    (regular : UniformConvexOn region modulus objective)
    (incumbent retained : Space) (first : incumbent ∈ region) (second : retained ∈ region)
    (bend gap fraction : ℝ) (bendBound : bend ≤ modulus ‖incumbent - retained‖)
    (observed : objective retained - objective incumbent ≤ gap)
    (fractionNonneg : 0 ≤ fraction) (fractionBound : fraction ≤ 1) :
    fraction * (1 - fraction) * bend - fraction * gap ≤
      progress objective incumbent (point incumbent retained fraction) := by
  have bound := chordBound objective region modulus regular incumbent retained first second
    fraction fractionNonneg fractionBound
  have weighted := mul_le_mul_of_nonneg_left observed fractionNonneg
  have curved := mul_le_mul_of_nonneg_left bendBound
    (mul_nonneg fractionNonneg (sub_nonneg.mpr fractionBound))
  unfold envelope progress at *
  nlinarith

omit [NormedSpace ℝ Space] in
/-- Strong convexity converts a LOWER squared-distance certificate into a
curvature benefit. This is the opposite side of the distance-error interval
from the upper-radius bound used for KNN prediction error. -/
theorem sampledBend (incumbent retained : Space) (curvature estimate error : ℝ)
    (nonnegative : 0 ≤ curvature)
    (geometry : |estimate - dist incumbent retained ^ 2| ≤ error) :
    curvature / 2 * (estimate - error) ≤
      curvature / 2 * ‖incumbent - retained‖^2 := by
  have upper := (abs_le.mp geometry).2
  rw [dist_eq_norm] at upper
  exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)

omit [NormedSpace ℝ Space] in
/-- The same two-sided energy error supplies an upper endpoint separation
for clipping the step to a trust radius. -/
theorem sampledRadius (incumbent retained : Space) (estimate error : ℝ)
    (geometry : |estimate - dist incumbent retained ^ 2| ≤ error) :
    dist incumbent retained ≤ Real.sqrt (estimate + error) := by
  have lower := (abs_le.mp geometry).1
  have square : dist incumbent retained ^ 2 ≤ estimate + error := by linarith
  exact (Real.le_sqrt dist_nonneg (by nlinarith [sq_nonneg (dist incumbent retained)])).2 square

/-- Radius clipping and a modulus certificate together yield an explicit
positive ideal gain. The retained state may itself have higher objective value. -/
theorem cappedGain (objective : Space → ℝ) (region : Set Space) (modulus : ℝ → ℝ)
    (regular : UniformConvexOn region modulus objective)
    (incumbent retained : Space) (first : incumbent ∈ region) (second : retained ∈ region)
    (bend gap radius separation : ℝ) (positive : 0 < bend)
    (gapNonneg : 0 ≤ gap) (useful : gap < bend)
    (radiusPositive : 0 < radius) (separated : 0 < separation)
    (bendBound : bend ≤ modulus ‖incumbent - retained‖)
    (observed : objective retained - objective incumbent ≤ gap) :
    capped bend gap radius separation * (bend - gap) / 2 ≤
      progress objective incumbent
        (point incumbent retained (capped bend gap radius separation)) := by
  have range := capRange bend gap radius separation positive gapNonneg useful radiusPositive separated
  exact (clippedGain bend gap _ positive range.1.le range.2).trans
    (certifiedGain objective region modulus regular incumbent retained first second bend gap _
      bendBound observed range.1.le
      (range.2.trans ((stepRange bend gap positive gapNonneg useful).2.trans (by norm_num))))

theorem cappedTrust (incumbent retained : Space) (bend gap radius separation : ℝ)
    (positive : 0 < bend) (gapNonneg : 0 ≤ gap) (useful : gap < bend)
    (radiusPositive : 0 < radius) (separated : 0 < separation)
    (geometry : dist retained incumbent ≤ separation) :
    dist (point incumbent retained (capped bend gap radius separation)) incumbent ≤ radius := by
  have range := capRange bend gap radius separation positive gapNonneg useful radiusPositive separated
  rw [pointDistance, abs_of_pos range.1]
  exact (mul_le_mul_of_nonneg_left geometry range.1.le).trans
    (capRadius bend gap radius separation separated)

omit [NormedAddCommGroup Space] [NormedSpace ℝ Space] in
/-- Whole-trajectory progress accounting, for any horizon rather than one
diagnostic round. A construction certificate can enter as `gains time`; its
validity must persist or be re-established at every step. -/
theorem cumulativeGain (objective : Space → ℝ) (states : ℕ → Space) (gains : ℕ → ℝ)
    (certified : ∀ time, gains time ≤ progress objective (states time) (states (time + 1)))
    (horizon : ℕ) :
    (∑ time ∈ Finset.range horizon, gains time) ≤
      objective (states 0) - objective (states horizon) := by
  induction horizon with
  | zero => simp
  | succ horizon induction =>
    rw [Finset.sum_range_succ]
    have decrement := certified horizon
    unfold progress at decrement
    linarith

end

noncomputable section Quadratic

variable {Space : Type*} [NormedAddCommGroup Space] [InnerProductSpace ℝ Space]

/-- A nonvacuous objective family on the complete parameter space. The chord
identity holds exactly in every real inner-product dimension, with no derivative. -/
theorem squareChord (best incumbent retained : Space) (fraction : ℝ) :
    ‖point incumbent retained fraction - best‖^2 =
      envelope (‖incumbent - best‖^2) (‖retained - best‖^2)
        (‖incumbent - retained‖^2) fraction := by
  have stateEq : point incumbent retained fraction - best =
      (1 - fraction) • (incumbent - best) + fraction • (retained - best) := by
    unfold point
    module
  have distanceEq : ‖incumbent - retained‖^2 =
      ‖incumbent - best‖^2 + ‖retained - best‖^2 -
        2 * inner ℝ (incumbent - best) (retained - best) := by
    have identity : incumbent - retained = (incumbent - best) - (retained - best) := by abel
    rw [identity, norm_sub_sq_real]
    ring
  rw [stateEq, norm_add_sq_real, norm_smul, norm_smul, real_inner_smul_left,
    real_inner_smul_right]
  simp only [Real.norm_eq_abs, mul_pow, sq_abs]
  unfold envelope
  rw [distanceEq]
  ring

/-- The quadratic family really satisfies the imported Mathlib property;
the successful example does not merely postulate its curvature certificate. -/
theorem squareRegular (best : Space) :
    StrongConvexOn Set.univ 2 (fun state : Space => ‖state - best‖^2) := by
  refine ⟨convex_univ, ?_⟩
  intro incumbent _ retained _ first second _ _ total
  have firstEq : first = 1 - second := by linarith
  subst first
  have identity := squareChord best incumbent retained second
  simpa only [point, envelope, smul_eq_mul, show (2 : ℝ) / 2 = 1 by norm_num,
    one_mul, mul_one, mul_comm] using identity.le

/-- In this family the predicted ideal gain is exact, not just an upper
bound on regret. The optimum is proof-side only and is not used by `step`. -/
theorem squareExact (best incumbent retained : Space)
    (separated : 0 < ‖incumbent - retained‖^2) :
    progress (fun state => ‖state - best‖^2) incumbent
      (point incumbent retained
        (step (‖incumbent - retained‖^2) (‖retained - best‖^2 - ‖incumbent - best‖^2))) =
      margin (‖incumbent - retained‖^2) (‖retained - best‖^2 - ‖incumbent - best‖^2) := by
  dsimp only [progress]
  rw [squareChord, envelopeMin _ _ _ _ separated]
  simp

/-- A whole-space success witness: two equally bad points around a quadratic
optimum yield that optimum as a NEW point. A selector restricted to the two
endpoints cannot realize this state unless the displacement is zero. -/
theorem midpointGain (best displacement : Space) :
    progress (fun state => ‖state - best‖^2) (best + displacement)
      (point (best + displacement) (best - displacement) (1 / 2)) = ‖displacement‖^2 := by
  have identity : point (best + displacement) (best - displacement) (1 / 2) = best := by
    unfold point
    module
  dsimp only [progress]
  rw [identity]
  simp

end Quadratic

#print axioms proposalGain
#print axioms certifiedGain
#print axioms sampledBend
#print axioms sampledRadius
#print axioms cappedGain
#print axioms cappedTrust
#print axioms executionGain
#print axioms executionTrust
#print axioms cumulativeGain
#print axioms squareRegular
#print axioms squareExact
#print axioms midpointGain
#print axioms flatModulus

end BauxResearch.Optimization.Interpolation
