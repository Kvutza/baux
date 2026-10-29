import BauxResearch.Optimization.Metric

/-!
# Context is part of the surrogate domain

A pretraining observation is indexed by both a parameter state and a corpus
condition. The deployed ENN geometry compares parameter states but does not
represent the corpus-window origin. This module records the exact consequence:
any finite Lipschitz certificate in that geometry forces loss to be identical
across every ignored corpus condition at a fixed parameter state.

There are therefore two coherent mathematical objects. A contextual surrogate
represents both parameter state and corpus condition. A marginal surrogate
represents an expectation over conditions and separately needs a justified
sampling law. A deterministic changing condition cannot silently serve as the
second object while being omitted from the first.
-/

namespace BauxResearch.Optimization.ContextualEnn

open BauxResearch.Optimization.Metric

variable {Parameter Context Response : Type*}

/-- Parameter-only geometry makes all corpus conditions at one parameter state
indistinguishable. Any objective Lipschitz in that geometry must consequently
be condition-invariant. -/
theorem contextConstant
    [PseudoMetricSpace Response]
    (observe : Parameter → Response)
    (objective : Parameter × Context → ℝ)
    (constant : ℝ)
    (regular : ∀ first second,
      |objective first - objective second| ≤
        constant * pullbackDist (fun state => observe state.1) first second)
    (parameter : Parameter)
    (first second : Context) :
    objective (parameter, first) = objective (parameter, second) := by
  exact fiberNecessary (fun state : Parameter × Context => observe state.1)
    objective constant regular (parameter, first) (parameter, second) rfl

/-- If two corpus conditions have different loss at one parameter state, no
finite global Lipschitz certificate exists for a geometry that ignores those
conditions. -/
theorem contextObstruct
    [PseudoMetricSpace Response]
    (observe : Parameter → Response)
    (objective : Parameter × Context → ℝ)
    (constant : ℝ)
    (parameter : Parameter)
    (first second : Context)
    (different : objective (parameter, first) ≠ objective (parameter, second)) :
    ¬ ∀ left right,
      |objective left - objective right| ≤
        constant * pullbackDist (fun state => observe state.1) left right := by
  intro regular
  exact different (contextConstant observe objective constant regular parameter first second)

#print axioms contextConstant
#print axioms contextObstruct

end BauxResearch.Optimization.ContextualEnn
