import BauxResearch.Objective.MultiObjective

/-!
# Objective construction from generated outputs

This module exposes the stages that are often collapsed into the word
"reward": generation, raw measurement, aggregation across tasks and randomness,
and the observation made available to an optimizer.

No expectation, risk measure, scalarization, benchmark, or probability law is
selected here.  Those are alternative constructions whose laws must be stated
and checked separately.
-/

/-!
Let `g_z(t, ω)` be generation at parameter state `z`, task `t`, and randomness
`ω`; let `ℓⱼ(t, g_z(t, ω))` be measurement `j`.  The constructed loss vector
aggregates these raw functions.  Until an `AggregationRule` is fixed, `f(z)`
does not designate a unique population or empirical objective.
-/

namespace BauxResearch.Objective.Construction

open BauxResearch.Objective.MultiObjective

/-- Semantic generation from parameters, a task, and explicit randomness. -/
structure GenerationSemantics (parameterCount : ℕ) where
  Task : Type
  Randomness : Type
  Output : Type
  generate :
    Parameter.State parameterCount → Task → Randomness → Output

/-- A named family of raw loss measurements on generated outputs. -/
structure MeasurementFamily
    {parameterCount objectiveCount : ℕ}
    (generation : GenerationSemantics parameterCount) where
  name : Fin objectiveCount → String
  loss : Fin objectiveCount → generation.Task → generation.Output → ℝ

/-- The unaggregated loss vector produced by one task/randomness pair. -/
def rawLosses
    {parameterCount objectiveCount : ℕ}
    (generation : GenerationSemantics parameterCount)
    (measurements : MeasurementFamily generation (objectiveCount := objectiveCount))
    (state : Parameter.State parameterCount)
    (task : generation.Task)
    (randomness : generation.Randomness) : LossVector objectiveCount :=
  fun objective =>
    measurements.loss objective task
      (generation.generate state task randomness)

/--
An aggregation rule may represent expectation, an empirical mean, a quantile,
CVaR, a worst case, or another explicitly defined risk functional.
-/
structure AggregationRule
    (Task Randomness : Type)
    (objectiveCount : ℕ) where
  name : String
  aggregate :
    (Task → Randomness → LossVector objectiveCount) → LossVector objectiveCount

/-- The population objective induced by generation, measurement, and aggregation. -/
def constructedLosses
    {parameterCount objectiveCount : ℕ}
    (generation : GenerationSemantics parameterCount)
    (measurements : MeasurementFamily generation (objectiveCount := objectiveCount))
    (aggregation :
      AggregationRule generation.Task generation.Randomness objectiveCount)
    (state : Parameter.State parameterCount) : LossVector objectiveCount :=
  aggregation.aggregate (fun task randomness =>
    rawLosses generation measurements state task randomness)

/-- A single fixed task/randomness pair is one valid, degenerate aggregation rule. -/
def pointAggregation
    {Task Randomness : Type}
    {objectiveCount : ℕ}
    (task : Task)
    (randomness : Randomness) :
    AggregationRule Task Randomness objectiveCount where
  name := "point evaluation"
  aggregate := fun losses => losses task randomness

theorem Constructed.point
    {parameterCount objectiveCount : ℕ}
    (generation : GenerationSemantics parameterCount)
    (measurements : MeasurementFamily generation (objectiveCount := objectiveCount))
    (state : Parameter.State parameterCount)
    (task : generation.Task)
    (randomness : generation.Randomness) :
    constructedLosses generation measurements
      (pointAggregation task randomness) state =
        rawLosses generation measurements state task randomness := by
  rfl

/--
The constructed objective can only distinguish two states when some raw
measurement presented to the aggregation rule distinguishes them.
-/
theorem Constructed.congr
    {parameterCount objectiveCount : ℕ}
    (generation : GenerationSemantics parameterCount)
    (measurements : MeasurementFamily generation (objectiveCount := objectiveCount))
    (aggregation :
      AggregationRule generation.Task generation.Randomness objectiveCount)
    (first second : Parameter.State parameterCount)
    (rawEqual : ∀ task randomness,
      rawLosses generation measurements first task randomness =
        rawLosses generation measurements second task randomness) :
    constructedLosses generation measurements aggregation first =
      constructedLosses generation measurements aggregation second := by
  unfold constructedLosses
  congr 1
  funext task randomness
  exact rawEqual task randomness

/--
The optimizer observation is a separate choice from the objective itself.  It
may retain the vector loss or deliberately compress it, and that information
loss must be studied as part of the optimizer.
-/
structure ObservationEncoder (objectiveCount : ℕ) where
  Observation : Type
  encode : LossVector objectiveCount → Observation

end BauxResearch.Objective.Construction
