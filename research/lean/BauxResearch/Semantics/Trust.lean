import BauxResearch.Semantics.Bounds
import BauxResearch.Semantics.Morbo

/-!
# ENN 0.4.1 trust-region transitions

This module gives executable binary64 semantics for the single-objective
trust-region update used by `tell`, its local restart, and MORBO's scalar-score
history.  In particular, ordinary TuRBO uses `abs(best_value)` as its
improvement scale, while MORBO supplies the range of its retained scalar scores.
-/

namespace BauxResearch.Semantics.Trust

open Sampling Bounds

def milli : Word := 0x3f50624dd2f1a9fc
def micro : Word := 0x3eb0c6f7a0b5ed8d
def two : Word := 0x4000000000000000

structure Config where
  initial : Word
  minimum : Word
  maximum : Word
  deriving Repr, DecidableEq

structure State where
  config : Config
  length : Word
  failures : Nat
  successes : Nat
  best : Word
  previous : Nat
  successTol : Nat
  failureTol : Nat
  histMin : Word
  histMax : Word
  deriving Repr, DecidableEq

inductive Error where
  | obsCount
  deriving Repr, DecidableEq

structure Stage where
  state : State
  error : Option Error
  deriving Repr, DecidableEq

def improvementScale (state : State) : Word :=
  if Float.finiteDouble state.best then
    (Float.absDouble (Float.wordDouble state.best)).bits
  else 0

def incorporate (state : State) (values : List Word) : State :=
  values.foldl (fun current value =>
    { current with
      histMin := Morbo.minNumber current.histMin value
      histMax := Morbo.maxNumber current.histMax value }) state

def adapt (state : State) (improved : Bool) : State :=
  let successes := if improved then state.successes + 1 else 0
  let failures := if improved then 0 else state.failures + 1
  if state.successTol ≤ successes then
    { state with
      length := Morbo.minNumber (mul state.length two) state.config.maximum
      successes := 0
      failures }
  else if state.failureTol ≤ failures then
    { state with
      length := mul state.length 0x3fe0000000000000
      successes
      failures := 0 }
  else { state with successes, failures }

def updateWith (state : State) (values : List Word) (numObs : Nat)
    (incumbent scale : Word) (keepHistory : Bool) : Stage :=
  if values.isEmpty then ⟨state, none⟩
  else if numObs != state.previous + values.length then ⟨state, some .obsCount⟩
  else if !Float.finiteDouble state.best then
    let next := { state with best := incumbent, previous := numObs }
    ⟨if keepHistory then incorporate next values else next, none⟩
  else
    let threshold := add state.best (mul milli scale)
    let changed := adapt state (Float.ltDouble threshold incumbent)
    let next := {
      changed with
      best := Morbo.maxNumber state.best incumbent
      previous := numObs }
    ⟨if keepHistory then incorporate next values else next, none⟩

/-- `TurboTrustRegion::update_with_incumbent_new_batch`. -/
def updateBatch (state : State) (values : List Word) (numObs : Nat)
    (incumbent : Word) : Stage :=
  updateWith state values numObs incumbent (improvementScale state) true

/-- `update_scalar_incumbent`, with the MORBO-computed scale supplied explicitly. -/
def updateScalar (state : State) (numObs : Nat) (incumbent scale : Word) : Stage :=
  if numObs = 0 || numObs = state.previous then ⟨state, none⟩
  else if numObs < state.previous then ⟨state, some .obsCount⟩
  else updateWith state [incumbent] numObs incumbent scale false

def restart (state : State) : State :=
  { state with
    length := state.config.initial
    failures := 0
    successes := 0
    best := negInf }

def restartLocal (state : State) : State :=
  { restart state with previous := 0, histMin := posInf, histMax := negInf }

structure Multi where
  inner : State
  scoreMin : Option Word
  scoreMax : Option Word
  scoreCount : Nat
  deriving Repr, DecidableEq

structure MultiStage where
  state : Multi
  error : Option Error
  deriving Repr, DecidableEq

def scoreScale (state : Multi) : Word :=
  if 2 ≤ state.scoreCount then
    match state.scoreMin, state.scoreMax with
    | some low, some high => Morbo.maxNumber (sub high low) micro
    | _, _ => 0
  else 0

/-- MORBO `push_scalar_update`, including score-history commit order. -/
def pushScore (state : Multi) (numObs : Nat) (score : Word) : MultiStage :=
  let result := updateScalar state.inner numObs score (scoreScale state)
  match result.error with
  | some cause => ⟨state, some cause⟩
  | none =>
      let low := state.scoreMin.map (Morbo.minNumber · score) |>.getD score
      let high := state.scoreMax.map (Morbo.maxNumber · score) |>.getD score
      ⟨{ inner := result.state
         scoreMin := some low
         scoreMax := some high
         scoreCount := state.scoreCount + 1 }, none⟩

def restartMulti (state : Multi) : Multi :=
  { inner := restart state.inner
    scoreMin := none
    scoreMax := none
    scoreCount := 0 }

def testConfig : Config :=
  ⟨0x3fe999999999999a, 0x3f80000000000000, 0x3ff999999999999a⟩

def blank : State :=
  ⟨testConfig, testConfig.initial, 0, 0, negInf, 0, 3, 4, posInf, negInf⟩

theorem firstBatch :
    updateBatch blank [Sampling.one] 1 Sampling.one =
      ⟨{ blank with
          best := Sampling.one
          previous := 1
          histMin := Sampling.one
          histMax := Sampling.one }, none⟩ := by
  decide +kernel

/-- The upstream absolute-incumbent threshold classifies this as a success;
the historical range `[-1000, 10]` would classify it as a failure. -/
theorem absoluteBar :
    let state := {
      blank with
      best := 0x4024000000000000
      previous := 2
      histMin := 0xc08f400000000000
      histMax := 0x4024000000000000 }
    (updateBatch state [0x40240a3d70a3d70a] 3 0x40240a3d70a3d70a).state.successes = 1 := by
  decide +kernel

theorem failedCount (state : State) (values : List Word) (numObs : Nat)
    (incumbent : Word) (hne : values.isEmpty = false)
    (hcount : numObs ≠ state.previous + values.length) :
    updateBatch state values numObs incumbent = ⟨state, some .obsCount⟩ := by
  simp [updateBatch, updateWith, hne, hcount]

theorem localReset (state : State) :
    (restartLocal state).previous = 0 ∧
      (restartLocal state).histMin = posInf ∧
      (restartLocal state).histMax = negInf := by
  simp [restartLocal]

theorem multiReset (state : Multi) :
    (restartMulti state).scoreCount = 0 ∧
      (restartMulti state).scoreMin = none ∧
      (restartMulti state).scoreMax = none := by
  simp [restartMulti]

#print axioms firstBatch
#print axioms absoluteBar
#print axioms failedCount
#print axioms localReset
#print axioms multiReset

end BauxResearch.Semantics.Trust
