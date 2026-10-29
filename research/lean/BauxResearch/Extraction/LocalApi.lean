import BauxResearch.Generated.LocalCoord.Funs
import BauxResearch.Semantics.Optimizer

/-!
# Extracted public optimizer guards

The production Rust `BayesOptimizer::ask` and `tell_variance` methods call the
three functions proved below before entering their mutating bodies.  This file
relates their Aeneas translations to the corresponding ENN 0.4.1 lifecycle
branches, including validation order and variance-mode state changes.
-/

namespace BauxResearch.Extraction.LocalApi

open Aeneas Aeneas.Std Result
open BauxResearch.Generated.LocalCoord
open BauxResearch.Semantics

abbrev AskGate := optimizer_dispatch.AskGate
abbrev TellGate := optimizer_dispatch.TellGate
abbrev VarianceGate := optimizer_dispatch.VarianceGate

def askSpec (arms : Usize) (morbo restart : Bool) : AskGate :=
  if arms.val = 0 then .ZeroArms
  else if morbo then .Dispatch
  else if restart then .RestartLocal
  else .Dispatch

def tellSpec (dim rows columns yRows : Usize) : TellGate :=
  if columns.val != dim.val then .InvalidX
  else if yRows.val != rows.val then .InvalidRows
  else if rows.val = 0 then .Empty
  else .Dispatch

def varianceSpec (current : Option Bool) (present : Bool) : VarianceGate :=
  match current with
  | none => .Set present
  | some expected => if expected != present then .Reject else .Keep

theorem askExtracted (arms : Usize) (morbo restart : Bool) :
    optimizer_dispatch.ask_gate arms morbo restart =
      .ok (askSpec arms morbo restart) := by
  by_cases zero : arms.val = 0
  · simp [optimizer_dispatch.ask_gate, askSpec, UScalar.eq_equiv, zero]
  · cases morbo <;> cases restart <;>
      simp [optimizer_dispatch.ask_gate, askSpec, zero]

theorem tellExtracted (dim rows columns yRows : Usize) :
    optimizer_dispatch.tell_gate dim rows columns yRows =
      .ok (tellSpec dim rows columns yRows) := by
  by_cases columnsMatch : columns.val = dim.val
  · by_cases rowsMatch : yRows.val = rows.val
    · by_cases empty : rows.val = 0
      <;> simp [optimizer_dispatch.tell_gate, tellSpec, UScalar.eq_equiv,
        columnsMatch, rowsMatch, empty]
    · simp [optimizer_dispatch.tell_gate, tellSpec, UScalar.eq_equiv,
        columnsMatch, rowsMatch]
  · simp [optimizer_dispatch.tell_gate, tellSpec, columnsMatch]

theorem varianceExtracted (current : Option Bool) (present : Bool) :
    optimizer_dispatch.variance_gate current present =
      .ok (varianceSpec current present) := by
  cases current with
  | none => simp [optimizer_dispatch.variance_gate, varianceSpec]
  | some expected =>
      cases expected <;> cases present <;>
        simp [optimizer_dispatch.variance_gate, varianceSpec]

theorem zero_iff (arms : Usize) (morbo restart : Bool) :
    optimizer_dispatch.ask_gate arms morbo restart = .ok .ZeroArms ↔
      arms.val = 0 := by
  by_cases zero : arms.val = 0
  · simp [optimizer_dispatch.ask_gate, UScalar.eq_equiv, zero]
  · cases morbo <;> cases restart <;>
      simp [optimizer_dispatch.ask_gate, zero]

theorem restart_iff (arms : Usize) (morbo restart : Bool) :
    optimizer_dispatch.ask_gate arms morbo restart = .ok .RestartLocal ↔
      arms.val ≠ 0 ∧ morbo = false ∧ restart = true := by
  by_cases zero : arms.val = 0
  <;> cases morbo <;> cases restart
  <;> simp [optimizer_dispatch.ask_gate, UScalar.eq_equiv, zero]

theorem xshape_iff (dim rows columns yRows : Usize) :
    optimizer_dispatch.tell_gate dim rows columns yRows = .ok .InvalidX ↔
      columns.val ≠ dim.val := by
  by_cases columnsMatch : columns.val = dim.val
  · by_cases rowsMatch : yRows.val = rows.val
    <;> by_cases empty : rows.val = 0
    <;> simp [optimizer_dispatch.tell_gate, UScalar.eq_equiv,
      columnsMatch, rowsMatch, empty]
    <;> split <;> simp
  · simp [optimizer_dispatch.tell_gate, columnsMatch]

theorem rowshape_iff (dim rows columns yRows : Usize) :
    optimizer_dispatch.tell_gate dim rows columns yRows = .ok .InvalidRows ↔
      columns.val = dim.val ∧ yRows.val ≠ rows.val := by
  by_cases columnsMatch : columns.val = dim.val
  · by_cases rowsMatch : yRows.val = rows.val
    <;> by_cases empty : rows.val = 0
    <;> simp [optimizer_dispatch.tell_gate, UScalar.eq_equiv,
      columnsMatch, rowsMatch, empty]
  · simp [optimizer_dispatch.tell_gate, columnsMatch]

theorem empty_iff (dim rows columns yRows : Usize) :
    optimizer_dispatch.tell_gate dim rows columns yRows = .ok .Empty ↔
      columns.val = dim.val ∧ yRows.val = rows.val ∧ rows.val = 0 := by
  by_cases columnsMatch : columns.val = dim.val
  <;> by_cases rowsMatch : yRows.val = rows.val
  <;> by_cases empty : rows.val = 0
  <;> simp [optimizer_dispatch.tell_gate, UScalar.eq_equiv,
    columnsMatch, rowsMatch, empty]

theorem set_iff (current : Option Bool) (present : Bool) :
    optimizer_dispatch.variance_gate current present = .ok (.Set present) ↔
      current = none := by
  cases current with
  | none => simp [optimizer_dispatch.variance_gate]
  | some expected =>
      cases expected <;> cases present <;>
        simp [optimizer_dispatch.variance_gate]

theorem keep_iff (current : Option Bool) (present : Bool) :
    optimizer_dispatch.variance_gate current present = .ok .Keep ↔
      current = some present := by
  cases current <;> cases present <;>
    simp [optimizer_dispatch.variance_gate]

theorem reject_iff (current : Option Bool) (present : Bool) :
    optimizer_dispatch.variance_gate current present = .ok .Reject ↔
      ∃ expected, current = some expected ∧ expected ≠ present := by
  cases current <;> cases present <;>
    simp [optimizer_dispatch.variance_gate]

theorem zeroMatches (ops : Lifecycle.Ops) (state : Lifecycle.State)
    (arms : Usize) (morbo restart : Bool)
    (gate : optimizer_dispatch.ask_gate arms morbo restart = .ok .ZeroArms) :
    Lifecycle.ask ops state arms.val = ⟨.error .zeroArms, state⟩ := by
  have zero := (zero_iff arms morbo restart).mp gate
  simpa [zero] using Lifecycle.zeroInert ops state

theorem xshape_matches (ops : Tell.Ops) (config : Tell.Config)
    (state : Tell.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (dim rows columns yRows : Usize)
    (dimEq : dim.val = state.life.dim) (_rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (_yRowsEq : yRows.val = y.rows)
    (gate : optimizer_dispatch.tell_gate dim rows columns yRows = .ok .InvalidX) :
    Tell.tell ops config state x y variance = ⟨state, some .xShape⟩ := by
  have unequal := (xshape_iff dim rows columns yRows).mp gate
  have shape : x.cols ≠ state.life.dim := by
    simpa [columnsEq, dimEq] using unequal
  simp [Tell.tell, shape]

theorem rowshape_matches (ops : Tell.Ops) (config : Tell.Config)
    (state : Tell.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (dim rows columns yRows : Usize)
    (dimEq : dim.val = state.life.dim) (rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (yRowsEq : yRows.val = y.rows)
    (gate : optimizer_dispatch.tell_gate dim rows columns yRows = .ok .InvalidRows) :
    Tell.tell ops config state x y variance = ⟨state, some .rowShape⟩ := by
  obtain ⟨sameColumns, unequalRows⟩ :=
    (rowshape_iff dim rows columns yRows).mp gate
  have shape : x.cols = state.life.dim := by
    simpa [columnsEq, dimEq] using sameColumns
  have rowsDiffer : y.rows ≠ x.rows := by
    simpa [yRowsEq, rowsEq] using unequalRows
  simp [Tell.tell, shape, rowsDiffer]

theorem emptyMatches (ops : Tell.Ops) (config : Tell.Config)
    (state : Tell.State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (dim rows columns yRows : Usize)
    (dimEq : dim.val = state.life.dim) (rowsEq : rows.val = x.rows)
    (columnsEq : columns.val = x.cols) (yRowsEq : yRows.val = y.rows)
    (gate : optimizer_dispatch.tell_gate dim rows columns yRows = .ok .Empty) :
    Tell.tell ops config state x y variance = ⟨state, none⟩ := by
  obtain ⟨sameColumns, sameRows, zeroRows⟩ :=
    (empty_iff dim rows columns yRows).mp gate
  have shape : x.cols = state.life.dim := by
    simpa [columnsEq, dimEq] using sameColumns
  have rowShape : y.rows = x.rows := by
    simpa [yRowsEq, rowsEq] using sameRows
  have empty : x.rows = 0 := by
    simpa [rowsEq] using zeroRows
  exact Tell.emptyInert ops config state x y variance shape rowShape empty

theorem set_matches (state : Tell.State) (present : Bool)
    (gate : optimizer_dispatch.variance_gate state.yvarMode present =
      .ok (.Set present)) :
    Tell.fixVariance state present =
      .ok { state with yvarMode := some present } := by
  have empty := (set_iff state.yvarMode present).mp gate
  simp [Tell.fixVariance, empty]

theorem keep_matches (state : Tell.State) (present : Bool)
    (gate : optimizer_dispatch.variance_gate state.yvarMode present = .ok .Keep) :
    Tell.fixVariance state present = .ok state := by
  have same := (keep_iff state.yvarMode present).mp gate
  simp [Tell.fixVariance, same]

theorem reject_matches (state : Tell.State) (present : Bool)
    (gate : optimizer_dispatch.variance_gate state.yvarMode present = .ok .Reject) :
    Tell.fixVariance state present = .error .varianceMode := by
  obtain ⟨expected, mode, different⟩ :=
    (reject_iff state.yvarMode present).mp gate
  simp [Tell.fixVariance, mode, different]

#print axioms askExtracted
#print axioms tellExtracted
#print axioms varianceExtracted
#print axioms zeroMatches
#print axioms xshape_matches
#print axioms rowshape_matches
#print axioms emptyMatches
#print axioms set_matches
#print axioms keep_matches
#print axioms reject_matches

end BauxResearch.Extraction.LocalApi
