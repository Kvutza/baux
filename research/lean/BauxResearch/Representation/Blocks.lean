import BauxResearch.Foundation

/-!
# Parameter blocks and proposal representations

This module prevents three different objects from being given one name.

* A `BlockLayout` assigns each learned parameter to an execution or scaling
  block.  Its block count is not the parameter dimension.
* A `BlockBasis` forms proposals from one real coefficient per fixed block
  field.  Such proposals lie in the span of those fields even when every
  materialized proposal contains one value for every parameter.
* A `SeededField` maps a seed directly to a full parameter proposal.  The seed
  size is not called the proposal dimension, and coordinatewise activity is
  not called joint reachability.
-/

/-!
For fixed block fields `φ_b`, a coefficient proposal has the form
`u = ∑_b a_b φ_b`.  Its materialized value has `p` parameter coordinates,
but its reachable linear span has rank at most the number of fixed fields.
A seeded field is a different map `seed → Θₚ`; seed cardinality alone gives
neither that span bound nor full-space reachability.
-/

namespace BauxResearch.Representation

open scoped BigOperators

/-- A nonredundant partition label for every learned parameter. -/
structure BlockLayout (parameterCount blockCount : ℕ) where
  block : ParameterIndex parameterCount → Fin blockCount
  used : Function.Surjective block

/-- A blockwise scalar field over the full parameter index set. -/
structure BlockScale (parameterCount blockCount : ℕ) where
  layout : BlockLayout parameterCount blockCount
  value : Fin blockCount → ℝ

/-- The scale assigned to one learned parameter by its block. -/
def BlockScale.at
    {parameterCount blockCount : ℕ}
    (scale : BlockScale parameterCount blockCount)
    (parameter : ParameterIndex parameterCount) : ℝ :=
  scale.value (scale.layout.block parameter)

/-- Fixed proposal fields localized to the blocks of one layout. -/
structure BlockBasis (parameterCount blockCount : ℕ) where
  layout : BlockLayout parameterCount blockCount
  field : Fin blockCount → Parameter.Proposal parameterCount
  localized : ∀ block parameter,
    layout.block parameter ≠ block → field block parameter = 0

/-- A proposal made from one coefficient per fixed block field. -/
def BlockBasis.proposal
    {parameterCount blockCount : ℕ}
    (basis : BlockBasis parameterCount blockCount)
    (coefficient : Fin blockCount → ℝ) :
    Parameter.Proposal parameterCount :=
  ∑ block, coefficient block • basis.field block

/-- Every fixed-block coefficient proposal lies in the span of its block fields. -/
theorem BlockBasis.inSpan
    {parameterCount blockCount : ℕ}
    (basis : BlockBasis parameterCount blockCount)
    (coefficient : Fin blockCount → ℝ) :
    basis.proposal coefficient ∈
      Submodule.span ℝ (Set.range basis.field) := by
  unfold BlockBasis.proposal
  apply Submodule.sum_mem
  intro block _
  exact Submodule.smul_mem _ _
    (Submodule.subset_span (Set.mem_range_self block))

/-- The reachable linear span of `blockCount` fixed fields has rank at most `blockCount`. -/
theorem BlockBasis.spanRank
    {parameterCount blockCount : ℕ}
    (basis : BlockBasis parameterCount blockCount) :
    Module.finrank ℝ (Submodule.span ℝ (Set.range basis.field)) ≤
      blockCount := by
  change (Set.range basis.field).finrank ℝ ≤ blockCount
  simpa using finrank_range_le_card
    (R := ℝ)
    (M := Parameter.Proposal parameterCount)
    basis.field

/-- A generator that maps a seed directly to a full parameter proposal. -/
structure SeededField (parameterCount : ℕ) (Seed : Type*) where
  sample : Seed → Parameter.Proposal parameterCount

/-- Exact joint reachability of one complete parameter proposal. -/
def SeededField.reaches
    {parameterCount : ℕ}
    {Seed : Type*}
    (field : SeededField parameterCount Seed)
    (proposal : Parameter.Proposal parameterCount) : Prop :=
  ∃ seed, field.sample seed = proposal

/-- Every parameter can be changed by some seed, possibly a different seed each time. -/
def SeededField.coordActive
    {parameterCount : ℕ}
    {Seed : Type*}
    (field : SeededField parameterCount Seed) : Prop :=
  ∀ parameter, ∃ seed, field.sample seed parameter ≠ 0

end BauxResearch.Representation
