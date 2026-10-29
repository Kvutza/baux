import BauxResearch.Semantics.Trust
import BauxResearch.Semantics.Turbo
import BauxResearch.Semantics.Incumbent

/-!
# ENN 0.4.1 public ask lifecycle

This module composes the public zero-arm guard, local restart, reinitialization,
ordinary initialization, MORBO weight resampling and Sobol reseeding, concrete
TuRBO candidate generation, all acquisition modes, and conversion from unit to
natural coordinates.  The state returned on every represented error retains
the random and Sobol transitions that occurred before that error.

The compiled neighbor backend and native transcendental operations remain the
typed artifact boundaries already exposed by the component modules.
-/

namespace BauxResearch.Semantics.Lifecycle

open Sampling Bounds

inductive Strategy where
  | init (kind : Opt.InitKind) (done budget : Nat)
  | turbo
  | hybrid (kind : Opt.InitKind) (done budget : Nat) (active : Bool)
  deriving Repr, DecidableEq

structure Ops where
  draw : Draw.Ops
  exp : Bounds.ExpOp
  log : Bounds.LogOp
  backend : Neighbor.BackendOp

structure State where
  dim : Nat
  bounds : List Bound
  strategy : Strategy
  trust : Trust.State
  candidates : Opt.CandidateCfg
  selectMode : Select.Mode
  context : Option Select.Context
  center : Option Row
  scales : Option Row
  engine : Option Sobol.Engine
  rng : RngState
  morboWeight : Option Morbo.WeightState
  morboSelect : Option Select.Morbo
  sobolBase : Word
  restartGen : Nat
  observations : Nat
  reinitLeft : Option Nat
  localKind : Opt.InitKind
  localBudget : Nat
  clearReady : Bool
  deriving Repr, DecidableEq

inductive AskError where
  | zeroArms
  | clear
  | weight (cause : Failure)
  | reseed (cause : Failure)
  | init (cause : Opt.AskError)
  | turbo (cause : Turbo.TurboError)
  | bounds
  deriving Repr, DecidableEq

structure AskStep where
  result : Except AskError Matrix
  state : State
  deriving Repr, DecidableEq

structure Prep where
  state : State
  error : Option AskError
  deriving Repr, DecidableEq

def isMorbo (state : State) : Bool :=
  state.morboWeight.isSome

def needsRestart (state : State) : Bool :=
  !isMorbo state && Float.ltDouble state.trust.length state.trust.config.minimum

def restartLocal (state : State) : State :=
  { state with
    trust := Trust.restartLocal state.trust
    context := none
    center := none
    scales := none
    restartGen := state.restartGen + 1
    observations := 0
    reinitLeft := if state.localBudget = 0 then none else some state.localBudget }

def initKind (state : State) : Option Opt.InitKind :=
  match state.reinitLeft with
  | some _ => some state.localKind
  | none => match state.strategy with
    | .init kind _ _ => some kind
    | .hybrid kind _ _ true => some kind
    | _ => none

def fromRow : List Bound → Row → Except AskError Row
  | [], values => if values.isEmpty then .ok #[] else .error .bounds
  | _, #[] => .error .bounds
  | (low, high) :: bounds, values =>
      match fromRow bounds (values.extract 1 values.size) with
      | .error cause => .error cause
      | .ok tail => .ok (#[add low (mul values[0]! (sub high low))] ++ tail)

def fromUnit (bounds : List Bound) (values : Matrix) : Except AskError Matrix :=
  values.toList.mapM (fromRow bounds) |>.map (fun rows => rows.toArray)

def syncWeights (state : State) (weights : Morbo.WeightState) : State :=
  let selected := state.morboSelect.map fun config =>
    { config with weights := weights.weights }
  { state with morboWeight := some weights, morboSelect := selected }

def redraw (ops : Ops) (state : State) : Prep :=
  match state.morboWeight with
  | none => ⟨state, none⟩
  | some weights =>
      let result := run (Morbo.redraw ops.log weights) state.rng
      match result.1 with
      | .error cause => ⟨{ state with rng := result.2 }, some (.weight cause)⟩
      | .ok next => ⟨{ syncWeights state next with rng := result.2 }, none⟩

def reseed (state : State) (arms : Nat) : Except AskError State :=
  match state.morboWeight, state.candidates.mode with
  | some _, .sobol =>
      match Morbo.reseedSobol state.sobolBase (BitVec.ofNat 64 state.restartGen)
          (BitVec.ofNat 64 state.observations) (BitVec.ofNat 64 arms) state.engine with
      | .error cause => .error (.reseed cause)
      | .ok engine => .ok { state with engine }
  | _, _ => .ok state

def midpoint (dim : Nat) : Row :=
  Array.replicate dim 0x3fe0000000000000

def askInit (state : State) (kind : Opt.InitKind) (arms : Nat) : AskStep :=
  let init : Opt.InitState :=
    ⟨state.dim, kind, state.reinitLeft, state.rng⟩
  let step := Opt.askInit init arms
  let next := { state with rng := step.state.rng }
  match step.result with
  | .error cause => ⟨.error (.init cause), next⟩
  | .ok unit =>
      match fromUnit state.bounds unit with
      | .error cause => ⟨.error cause, next⟩
      | .ok natural => ⟨.ok natural, next⟩

def askTurbo (ops : Ops) (state : State) (arms : Nat) : AskStep :=
  let weighted := redraw ops state
  match weighted.error with
  | some cause => ⟨.error cause, weighted.state⟩
  | none =>
      match reseed weighted.state arms with
      | .error cause => ⟨.error cause, weighted.state⟩
      | .ok ready =>
          let center := ready.center.getD (midpoint ready.dim)
          let step := Turbo.ask ops.draw ops.exp ops.backend ready.context
            ready.selectMode ready.candidates (isMorbo ready) center
            ready.trust.length ready.scales arms ready.engine ready.rng
          let next := { ready with engine := step.engine, rng := step.rng }
          match step.result with
          | .error cause => ⟨.error (.turbo cause), next⟩
          | .ok unit =>
              match fromUnit ready.bounds unit with
              | .error cause => ⟨.error cause, next⟩
              | .ok natural => ⟨.ok natural, next⟩

/-- Public `Optimizer::ask`, with restart before strategy dispatch and natural
coordinate conversion after a successful unit-space proposal. -/
def ask (ops : Ops) (state : State) (arms : Nat) : AskStep :=
  if arms = 0 then ⟨.error .zeroArms, state⟩
  else
    let restarted := needsRestart state
    if restarted && !state.clearReady then ⟨.error .clear, state⟩
    else
      let ready := if restarted then restartLocal state else state
      match initKind ready with
      | some kind => askInit ready kind arms
      | none => askTurbo ops ready arms

theorem zeroInert (ops : Ops) (state : State) :
    ask ops state 0 = ⟨.error .zeroArms, state⟩ := by
  simp [ask]

theorem clearInert (ops : Ops) (state : State) (arms : Nat)
    (hpos : arms ≠ 0) (hneed : needsRestart state = true)
    (hclear : state.clearReady = false) :
    ask ops state arms = ⟨.error .clear, state⟩ := by
  simp [ask, hpos, hneed, hclear]

theorem initDispatch (ops : Ops) (state : State) (kind : Opt.InitKind)
    (arms : Nat) (hpos : arms ≠ 0) (hrestart : needsRestart state = false)
    (hkind : initKind state = some kind) :
    ask ops state arms = askInit state kind arms := by
  simp [ask, hpos, hrestart, hkind]

theorem localReset (state : State) :
    (restartLocal state).observations = 0 ∧
      (restartLocal state).center = none ∧
      (restartLocal state).restartGen = state.restartGen + 1 := by
  simp [restartLocal]

theorem redrawFail (ops : Ops) (state : State)
    (weights : Morbo.WeightState) (cause : Failure) (next : RngState)
    (hweight : state.morboWeight = some weights)
    (hdraw : run (Morbo.redraw ops.log weights) state.rng = (.error cause, next)) :
    redraw ops state = ⟨{ state with rng := next }, some (.weight cause)⟩ := by
  simp [redraw, hweight, hdraw]

#print axioms zeroInert
#print axioms clearInert
#print axioms initDispatch
#print axioms localReset
#print axioms redrawFail

end BauxResearch.Semantics.Lifecycle
