import BauxResearch.Foundation

/-!
# One exact optimizer step

This module separates three quantities without assuming a proposal law,
surrogate, acquisition rule, smooth objective, or noise model.

* `opportunity` is the improvement offered by a comparison candidate.
* `selectionRegret` is how much worse the selected candidate is than that
  comparison candidate.
* `progress` is the incumbent-to-selected objective decrease.

The decomposition theorem is an algebraic identity.  Nonnegativity and positive
progress require the hypotheses stated by the subsequent theorems.
-/

/-!
For an objective `f`, incumbent `θ`, comparison state `c`, and selected state
`θ′`, the exact identity is
`f(θ) - f(θ′) = (f(θ) - f(c)) - (f(θ′) - f(c))`.
Thus `progress = opportunity - selectionRegret` introduces no statistical
guarantee: a useful comparison and controlled selection error are separate
premises.
-/

namespace BauxResearch.Optimization

variable {State : Type*}

abbrev Objective (State : Type*) := State → ℝ

def opportunity
    (objective : Objective State)
    (incumbent comparison : State) : ℝ :=
  objective incumbent - objective comparison

def selectionRegret
    (objective : Objective State)
    (comparison selected : State) : ℝ :=
  objective selected - objective comparison

def progress
    (objective : Objective State)
    (incumbent selected : State) : ℝ :=
  objective incumbent - objective selected

theorem progress_decomposition
    (objective : Objective State)
    (incumbent comparison selected : State) :
    progress objective incumbent selected =
      opportunity objective incumbent comparison -
        selectionRegret objective comparison selected := by
  unfold progress opportunity selectionRegret
  ring

theorem opportunity_nonnegative
    (objective : Objective State)
    (incumbent comparison : State)
    (comparisonNoWorse : objective comparison ≤ objective incumbent) :
    0 ≤ opportunity objective incumbent comparison := by
  exact sub_nonneg.mpr comparisonNoWorse

theorem Selection.nonneg
    (objective : Objective State)
    (comparison selected : State)
    (comparisonNoWorse : objective comparison ≤ objective selected) :
    0 ≤ selectionRegret objective comparison selected := by
  exact sub_nonneg.mpr comparisonNoWorse

theorem Progress.positive
    (objective : Objective State)
    (incumbent comparison selected : State)
    (dominates :
      selectionRegret objective comparison selected <
        opportunity objective incumbent comparison) :
    0 < progress objective incumbent selected := by
  rw [progress_decomposition]
  exact sub_pos.mpr dominates

/-- Uniform error of an estimator on a domain. -/
def UniformError
    {Candidate : Type*}
    (truth estimate : Candidate → ℝ)
    (epsilon : ℝ) : Prop :=
  ∀ candidate, |estimate candidate - truth candidate| ≤ epsilon

/-- An element minimizes a function up to `slack` (used e.g. for Lepski scale selection). -/
def ApproxMin
    {Candidate : Type*}
    (estimate : Candidate → ℝ)
    (selected : Candidate)
    (slack : ℝ) : Prop :=
  ∀ candidate, estimate selected ≤ estimate candidate + slack

/-- Point-specific estimation error envelope. -/
def PointError
    {Candidate : Type*}
    (truth estimate error : Candidate → ℝ) : Prop :=
  ∀ candidate, |estimate candidate - truth candidate| ≤ error candidate

/-- Pointwise error envelopes charge the selected and comparison elements separately. -/
theorem Selection.pointBound
    {Candidate : Type*}
    (truth estimate error : Candidate → ℝ)
    (selected comparison : Candidate)
    (slack : ℝ)
    (errorBound : PointError truth estimate error)
    (selectedWithin : ApproxMin estimate selected slack) :
    truth selected - truth comparison ≤
      error selected + error comparison + slack := by
  have selectedError := (abs_le.mp (errorBound selected)).1
  have comparisonError := (abs_le.mp (errorBound comparison)).2
  have selectionError := selectedWithin comparison
  linarith

/-- Uniform error `epsilon` and slack imply difference at most `2ε + ξ`. -/
theorem Selection.regretBound
    {Candidate : Type*}
    (truth estimate : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError truth estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack) :
    truth selected - truth comparison ≤ 2 * epsilon + slack := by
  have selectedError := (abs_le.mp (errorBound selected)).1
  have comparisonError := (abs_le.mp (errorBound comparison)).2
  have selectionError := selectedWithin comparison
  linarith

/-- Approximate minimization under uniform error loses at most `2ε + ξ` of opportunity. -/
theorem Progress.lowerBound
    {State : Type*}
    {Candidate : Type*}
    (objective : Objective State)
    (incumbent : State)
    (candidate : Candidate → State)
    (estimate : Candidate → ℝ)
    (selected comparison : Candidate)
    (epsilon slack : ℝ)
    (errorBound : UniformError
      (fun index => objective (candidate index)) estimate epsilon)
    (selectedWithin : ApproxMin estimate selected slack) :
    opportunity objective incumbent (candidate comparison) -
        (2 * epsilon + slack) ≤
      progress objective incumbent (candidate selected) := by
  have regretBound := Selection.regretBound
    (fun index => objective (candidate index))
    estimate selected comparison epsilon slack errorBound selectedWithin
  unfold opportunity progress at *
  linarith

end BauxResearch.Optimization
