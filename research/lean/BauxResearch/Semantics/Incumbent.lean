import BauxResearch.Semantics.Acq
import BauxResearch.Semantics.Morbo

/-!
# ENN 0.4.1 incremental incumbent tracker

This module executes `incumbent_tracker.rs` at binary64 precision.  It covers
all four source modes, including Rust `f64::total_cmp` ordering for retained
top-m entries, ordinary IEEE `>`/`==` behavior for the noiseless maximum,
ascending index output, reset, and rebuild.  Invalid constructor dimensions or
short metric rows represent the source assertions/index panics explicitly.
-/

namespace BauxResearch.Semantics.Incumbent

open Sampling Bounds

def allCutoff : Nat := 1000000000

inductive Mode where
  | all
  | noiselessMax
  | scalarTop
  | multiTop
  deriving Repr, DecidableEq

structure Entry where
  index : Nat
  value : Word
  deriving Repr, DecidableEq

structure State where
  m : Nat
  numMetrics : Nat
  mode : Mode
  observationCount : Nat
  allIndices : List Nat
  maxY : Word
  maxIndices : List Nat
  maxInitialized : Bool
  topEntries : List Entry
  perMetricEntries : List (List Entry)
  deriving Repr, DecidableEq

def modeFor (m : Nat) (noiseAware : Bool) (numMetrics : Nat) : Mode :=
  if allCutoff ≤ m then .all
  else if numMetrics = 1 then
    if noiseAware then .scalarTop else .noiselessMax
  else .multiTop

def new (m : Nat) (noiseAware : Bool) (numMetrics : Nat) : Except Failure State :=
  if m = 0 || numMetrics = 0 then .error .panic
  else
    let mode := modeFor m noiseAware numMetrics
    .ok {
      m
      numMetrics
      mode
      observationCount := 0
      allIndices := []
      maxY := negInf
      maxIndices := []
      maxInitialized := false
      topEntries := []
      perMetricEntries := if mode = .multiTop then List.replicate numMetrics [] else [] }

/-- Comparator passed to Rust's stable `sort_by`: descending value under
`f64::total_cmp`, then ascending observation index. -/
def before (left right : Entry) : Bool :=
  let lk := Fit.totalKey left.value
  let rk := Fit.totalKey right.value
  decide (rk < lk || (rk = lk && left.index < right.index))

def insert (entry : Entry) : List Entry → List Entry
  | [] => [entry]
  | head :: tail =>
      if before entry head then entry :: head :: tail
      else head :: insert entry tail

def sortEntries (entries : List Entry) : List Entry :=
  entries.foldr insert []

def pushTop (entries : List Entry) (entry : Entry) (m : Nat) : List Entry :=
  (sortEntries (entries ++ [entry])).take m

def natInsert (value : Nat) : List Nat → List Nat
  | [] => [value]
  | head :: tail =>
      if value ≤ head then value :: head :: tail
      else head :: natInsert value tail

def natSort (values : List Nat) : List Nat :=
  values.foldr natInsert []

def sortedIndices (entries : List Entry) : List Nat :=
  natSort (entries.map Entry.index)

def pushMetrics (index m : Nat) : List Word → List (List Entry) →
    Except Failure (List (List Entry))
  | _, [] => .ok []
  | [], _ :: _ => .error .panic
  | value :: values, entries :: rest => do
      let tail ← pushMetrics index m values rest
      return pushTop entries ⟨index, value⟩ m :: tail

/-- One `IncrementalIncumbentTracker::tell` call. -/
def tell (state : State) (index : Nat) (y : List Word) : Except Failure State := do
  let counted := { state with observationCount := state.observationCount + 1 }
  match state.mode with
  | .all => return { counted with allIndices := counted.allIndices ++ [index] }
  | .noiselessMax =>
      let some value := y[0]? | throw .panic
      if !state.maxInitialized || Float.ltDouble state.maxY value then
        return { counted with maxY := value, maxIndices := [index], maxInitialized := true }
      else if Float.eqDouble value state.maxY then
        return { counted with maxIndices := state.maxIndices ++ [index] }
      else return counted
  | .scalarTop =>
      let some value := y[0]? | throw .panic
      return { counted with
        topEntries := pushTop state.topEntries ⟨index, value⟩ state.m }
  | .multiTop =>
      let entries ← pushMetrics index state.m y state.perMetricEntries
      return { counted with perMetricEntries := entries }

def memberMetric (index : Nat) (metrics : List (List Entry)) : Bool :=
  metrics.any fun entries => entries.any fun entry => entry.index == index

/-- `IncrementalIncumbentTracker::ask`; every returned list is ascending. -/
def ask (state : State) : List Nat :=
  if state.observationCount = 0 then []
  else match state.mode with
  | .all => natSort state.allIndices
  | .noiselessMax => natSort state.maxIndices
  | .scalarTop => sortedIndices state.topEntries
  | .multiTop =>
      (List.range state.observationCount).filter fun index =>
        memberMetric index state.perMetricEntries

def reset (state : State) : State :=
  { state with
    observationCount := 0
    allIndices := []
    maxY := negInf
    maxIndices := []
    maxInitialized := false
    topEntries := []
    perMetricEntries := if state.mode = .multiTop then
      List.replicate state.numMetrics [] else [] }

def tellRows : State → Nat → List (List Word) → Except Failure State
  | state, _, [] => .ok state
  | state, index, row :: rows => do
      let next ← tell state index row
      tellRows next (index + 1) rows

/-- Source rebuild resets the tracker and replays rows with indices `0..n`. -/
def rebuild (state : State) (rows : List (List Word)) : Except Failure State :=
  tellRows (reset state) 0 rows

structure Choice where
  index : Nat
  x : List Word
  y : List Word
  deriving Repr, DecidableEq

def rowsAt (rows : List (List Word)) : List Nat → Except Failure (List (List Word))
  | [] => .ok []
  | index :: indices => do
      let some row := rows[index]? | throw .panic
      let tail ← rowsAt rows indices
      return row :: tail

/-- Rust iterator `max_by(total_cmp)` retains the later candidate when values
compare equal. -/
def rawBest (y : List (List Word)) : Nat → List Nat → Except Failure Nat
  | best, [] => .ok best
  | best, candidate :: rest => do
      let some bestRow := y[best]? | throw .panic
      let some nextRow := y[candidate]? | throw .panic
      let some bestValue := bestRow[0]? | throw .panic
      let some nextValue := nextRow[0]? | throw .panic
      let chosen := if Fit.totalLe bestValue nextValue then candidate else best
      rawBest y chosen rest

/-- Noiseless single-objective `Optimizer::update_incumbent` after the tracker
has supplied its candidate indices. -/
def chooseRaw (x y : List (List Word)) (candidates : List Nat) : Except Failure Choice := do
  let first :: rest := candidates | throw .invalid
  let best ← rawBest y first rest
  let some xRow := x[best]? | throw .panic
  let some yRow := y[best]? | throw .panic
  return ⟨best, xRow, yRow⟩

def resolve (x scoreRows : List (List Word)) (candidates : List Nat)
    (scores : Row) (rng : RngState) : Except Failure Choice × RngState :=
  let selected := run (Acq.argmaxTie scores.toList) rng
  match selected.1 with
  | .error cause => (.error cause, selected.2)
  | .ok position =>
      match candidates[position]? with
      | none => (.error .panic, selected.2)
      | some index =>
          match x[index]?, scoreRows[position]? with
          | some xRow, some yRow => (.ok ⟨index, xRow, yRow⟩, selected.2)
          | _, _ => (.error .panic, selected.2)

/-- MORBO incumbent selection. `predicted = some rows` is the noise-aware
surrogate branch; `none` uses gathered natural observations. Candidate-local
ranges, Chebyshev scalarization, and random tie selection are concrete. -/
def chooseMorbo (x y : List (List Word)) (candidates : List Nat)
    (predicted : Option (List (List Word))) (weights : Row) (alpha : Word)
    (metrics : Nat) (rng : RngState) : Except Failure Choice × RngState :=
  match rowsAt y candidates with
  | .error cause => (.error cause, rng)
  | .ok rawRows =>
      let scoreRows := predicted.getD rawRows
      match Morbo.scalarizeLocal
          { rows := scoreRows.map List.toArray |>.toArray, cols := metrics }
          weights alpha metrics with
      | .error cause => (.error cause, rng)
      | .ok scores => resolve x scoreRows candidates scores rng

theorem modeTable :
    modeFor (allCutoff + 1) false 1 = .all ∧
    modeFor 3 false 1 = .noiselessMax ∧
    modeFor 3 true 1 = .scalarTop ∧
    modeFor 2 false 2 = .multiTop := by
  decide +kernel

theorem tieLower :
    (pushTop [⟨1, Bounds.one⟩] ⟨0, Bounds.one⟩ 2).map Entry.index = [0, 1] := by
  decide +kernel

theorem noiselessTies :
    (do
      let state ← new 3 false 1
      let state ← tell state 0 [Bounds.one]
      let state ← tell state 1 [0x4008000000000000]
      let state ← tell state 3 [0x4008000000000000]
      return ask state) = .ok [1, 3] := by
  decide +kernel

theorem scalarThree :
    (do
      let state ← new 3 true 1
      let state ← tell state 0 [0x3fb999999999999a]
      let state ← tell state 1 [0x3feccccccccccccd]
      let state ← tell state 2 [0x3fd3333333333333]
      let state ← tell state 3 [0x3fe999999999999a]
      let state ← tell state 4 [0x3fc999999999999a]
      return ask state) = .ok [1, 2, 3] := by
  decide +kernel

theorem multiUnion :
    (do
      let state ← new 2 false 2
      let state ← tell state 0 [0x4024000000000000, 0]
      let state ← tell state 1 [0x4022000000000000, Bounds.one]
      let state ← tell state 2 [0, 0x4024000000000000]
      let state ← tell state 3 [Bounds.one, 0x4022000000000000]
      let state ← tell state 4 [0x4014000000000000, 0x4014000000000000]
      return ask state) = .ok [0, 1, 2, 3] := by
  decide +kernel

theorem rebuildResets :
    (do
      let state ← new 2 true 1
      let state ← tell state 9 [Bounds.one]
      let state ← rebuild state [[Bounds.one], [0x4008000000000000],
        [0x4000000000000000]]
      return (state.observationCount, ask state)) = .ok (3, [1, 2]) := by
  decide +kernel

theorem rawLater :
    chooseRaw [[0], [Bounds.one]] [[Bounds.one], [Bounds.one]] [0, 1] =
      .ok ⟨1, [Bounds.one], [Bounds.one]⟩ := by
  decide +kernel

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem morboUnique (rng : RngState) :
    chooseMorbo [[0], [Bounds.one]]
      [[numWord 10, numWord 10], [numWord 9, numWord 9]] [0, 1] none
      #[0x3fe0000000000000, 0x3fe0000000000000]
      0x3fb999999999999a 2 rng =
      (.ok ⟨0, [0], [numWord 10, numWord 10]⟩, rng) := by
  have scores :
      Morbo.scalarizeLocal
        { rows := #[#[numWord 10, numWord 10], #[numWord 9, numWord 9]], cols := 2 }
        #[4602678819172646912#64, 4602678819172646912#64]
        4591870180066957722#64 2 = .ok #[4603579539098121011#64, 0#64] := by
    decide +kernel
  have pick : run (Acq.argmaxTie [4603579539098121011#64, 0#64]) rng =
      (.ok 0, rng) := by
    rfl
  simp [chooseMorbo, rowsAt, resolve, scores, pick]

#print axioms modeTable
#print axioms tieLower
#print axioms noiselessTies
#print axioms scalarThree
#print axioms multiUnion
#print axioms rebuildResets
#print axioms rawLater
#print axioms morboUnique

end BauxResearch.Semantics.Incumbent
