import BauxResearch.Foundation
import Mathlib.Data.Real.Basic

set_option linter.unusedVariables false

/-!
# Earlier architecture facts and pairwise routing-margin lemma

This is not a forward-program implementation. Declared constants record architectural
dimensions; they do not prove executed sparsity or stochasticity.
The source-grounded real-arithmetic composition is in `Forward/Program.lean`.
-/

namespace BauxResearch.Semantics.FbtProgram

/-- Number of active causal leaves retained by PISA-1 attention. -/
def pisaLeaves : ℕ := 8

/-- Earlier fixed block-count fixture, not the context-dependent executed leaf count. -/
def pisaBlocks : ℕ := 64

/-- The 4-stream mHC recurrence width. -/
def mhcStreams : ℕ := 4

/-- Total physical layers in the FBT core. -/
def physLayers : ℕ := 5

/-- Routed expert count per physical layer. -/
def routedExperts : ℕ := 625

/-- Selected top experts per MoE evaluation. -/
def topExperts : ℕ := 3

/-- A router margin separates top-k selected experts from unselected candidates. -/
def routerMargin (sTop sOther : ℝ) : ℝ :=
  sTop - sOther

/--
Pairwise order preservation under explicitly bounded score shifts.
Whole top-k set preservation and parameter-to-score bounds are separate obligations.
-/
theorem cellInvar
    (sSelected sUnselected : ℝ)
    (gamma deltaSelected deltaUnselected : ℝ)
    (hMargin : sUnselected + gamma ≤ sSelected)
    (hDeltaSel : |deltaSelected| < gamma / 2)
    (hDeltaUnsel : |deltaUnselected| < gamma / 2) :
    sUnselected + deltaUnselected < sSelected + deltaSelected := by
  have h1 : - (gamma / 2) < deltaSelected := (abs_lt.mp hDeltaSel).1
  have h2 : deltaUnselected < gamma / 2 := (abs_lt.mp hDeltaUnsel).2
  linarith

/--
Predicate recording a proposed loss-jump bound; no jump bound is proved here.
-/
def cellBound (lossCand lossInc maxJump : ℝ) : Prop :=
  |lossCand - lossInc| ≤ maxJump

end BauxResearch.Semantics.FbtProgram
