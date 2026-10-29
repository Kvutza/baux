import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# Real-arithmetic API sketch and invariant lemmas

This is an incomplete real-arithmetic sketch, not executable semantics of
revision `ff793157`. Its abstract seams, error-state handling, incumbent selection,
and lifecycle do not yet correspond to the source. The invariant lemmas below
prove properties of this sketch only. Concrete candidate computations are in
`Sampling` and `Sobol`; full API correspondence remains open.

The sketch contains:
1. Error conditions (`EnnError`) and outcome types (`EnnResult`).
2. Domain hyperrectangle and coordinate transformations (`toUnit`, `fromUnit`).
3. Trust-region configuration and adaptation dynamics (`TrConfig`, `TrState`).
4. Strategy execution mode and progression (`Strategy`).
5. Observation append deltas (`ObsDelta`).
6. Seams for RNG candidate generation and surrogate fitting/acquisition.
7. Composable optimizer state and transitions (`create`, `ask`, `tell`, `restart`).
8. Telemetry tracking, observation queries, and predictive evaluation.
9. Verified bounds preservation, counter evolution, and invariant gates.
-/

namespace BauxResearch.Semantics.TurboApi

noncomputable section

/-! ## 1. Error Model and Results -/

/-- Error conditions returned by the reference optimizer API. -/
inductive EnnError where
  | invalidShape (expected : List Nat) (got : List Nat)
  | invalidParam (msg : String)
  | indexError (msg : String)
  | lowObs (got : Nat) (need : Nat)
  | shapeError (msg : String)
  deriving Repr, BEq, DecidableEq

/-- Standard result wrapper for optimizer transitions. -/
abbrev EnnResult (α : Type) := Except EnnError α

/-! ## 2. Bounds and Coordinate Systems -/

/-- Dimension-wise real lower and upper bounds with strict ordering. -/
structure BoundPair where
  lb : ℝ
  ub : ℝ
  valid : lb < ub

/-- Map a coordinate from the natural bounding interval to the unit interval. -/
def toUnit (b : BoundPair) (x : ℝ) : ℝ :=
  (x - b.lb) / (b.ub - b.lb)

/-- Map a coordinate from the unit interval to the natural bounding interval. -/
def fromUnit (b : BoundPair) (u : ℝ) : ℝ :=
  b.lb + u * (b.ub - b.lb)

/-- Transforming to unit coordinates and back preserves the unit value. -/
theorem unitRound (b : BoundPair) (u : ℝ) :
    toUnit b (fromUnit b u) = u := by
  have hdiff : b.ub - b.lb ≠ 0 := by linarith [b.valid]
  unfold toUnit fromUnit
  calc
    (b.lb + u * (b.ub - b.lb) - b.lb) / (b.ub - b.lb)
      = (u * (b.ub - b.lb)) / (b.ub - b.lb) := by ring
    _ = u := mul_div_cancel_right₀ u hdiff

/-- Transforming to natural coordinates and back preserves the natural value. -/
theorem natRound (b : BoundPair) (x : ℝ) :
    fromUnit b (toUnit b x) = x := by
  have hdiff : b.ub - b.lb ≠ 0 := by linarith [b.valid]
  unfold toUnit fromUnit
  calc
    b.lb + (x - b.lb) / (b.ub - b.lb) * (b.ub - b.lb)
      = b.lb + (x - b.lb) := by rw [div_mul_cancel₀ (x - b.lb) hdiff]
    _ = x := by ring

/-! ## 3. Trust Region Configuration and State -/

/-- Hyperparameters governing trust region radius expansion and contraction. -/
structure TrConfig where
  initLen : ℝ
  minLen : ℝ
  maxLen : ℝ
  minPos : 0 < minLen
  minOrder : minLen < maxLen
  minInit : minLen ≤ initLen
  initMax : initLen ≤ maxLen

/-- Standard reference parameters: init=0.8, min=0.5^7=1/128, max=1.6. -/
def trDefault : TrConfig where
  initLen := 4 / 5
  minLen := 1 / 128
  maxLen := 8 / 5
  minPos := by norm_num
  minOrder := by norm_num
  minInit := by norm_num
  initMax := by norm_num

/-- Runtime state of a TuRBO trust region. -/
structure TrState where
  dim : Nat
  length : ℝ
  failCount : Nat
  succCount : Nat
  bestValue : Option ℝ
  prevObs : Nat
  succTol : Nat
  failTol : Nat
  numArms : Nat
  cfg : TrConfig

/-- Initialize a fresh trust region state. -/
def trInit (dim : Nat) (cfg : TrConfig) (numArms : Nat) : TrState where
  dim := dim
  length := cfg.initLen
  failCount := 0
  succCount := 0
  bestValue := none
  prevObs := 0
  succTol := 3
  failTol := max 1 ((dim + numArms - 1) / (max 1 numArms))
  numArms := numArms
  cfg := cfg

/-- Clamp real value to unit interval [0, 1]. -/
def clampBox (v : ℝ) : ℝ :=
  max 0 (min 1 v)

theorem clampNonneg (v : ℝ) : 0 ≤ clampBox v :=
  le_max_left 0 (min 1 v)

theorem clampTop (v : ℝ) : clampBox v ≤ 1 := by
  unfold clampBox
  exact max_le (by linarith) (min_le_left 1 v)

/-- Compute clamped coordinate-wise trust region bounds [lb, ub] in unit space. -/
def trBounds (dim : Nat) (length : ℝ) (center : Fin dim → ℝ) :
    (Fin dim → ℝ) × (Fin dim → ℝ) :=
  let lb := fun i ↦ clampBox (center i - length / 2)
  let ub := fun i ↦ clampBox (center i + length / 2)
  (lb, ub)

/-- Trust region lower bound is non-negative. -/
theorem trLower (dim : Nat) (length : ℝ) (center : Fin dim → ℝ) (i : Fin dim) :
    0 ≤ (trBounds dim length center).1 i :=
  clampNonneg (center i - length / 2)

/-- Trust region upper bound is at most one. -/
theorem trUpper (dim : Nat) (length : ℝ) (center : Fin dim → ℝ) (i : Fin dim) :
    (trBounds dim length center).2 i ≤ 1 :=
  clampTop (center i + length / 2)

/-- For non-negative length, the lower bound does not exceed the upper bound. -/
theorem trValid (dim : Nat) {length : ℝ} (hlen : 0 ≤ length)
    (center : Fin dim → ℝ) (i : Fin dim) :
    (trBounds dim length center).1 i ≤ (trBounds dim length center).2 i := by
  unfold trBounds clampBox
  dsimp
  have hdiff : center i - length / 2 ≤ center i + length / 2 := by linarith [hlen]
  have hmin : min 1 (center i - length / 2) ≤ min 1 (center i + length / 2) :=
    min_le_min le_rfl hdiff
  exact max_le_max le_rfl hmin

/-- Predicate determining whether trust region contraction warrants a restart. -/
def trNeeds (tr : TrState) : Bool :=
  decide (tr.length < tr.cfg.minLen)

/-- Reset trust region length, counters, and incumbent value upon restart. -/
def trRestart (tr : TrState) : TrState :=
  { tr with
    length := tr.cfg.initLen
    failCount := 0
    succCount := 0
    bestValue := none }

/-- Reset trust region including historical observation counter. -/
def trReset (tr : TrState) : TrState :=
  { trRestart tr with prevObs := 0 }

/-- Advance trust region counters and length following a batch evaluation. -/
def trUpdate (tr : TrState) (newBest : ℝ) (numObs : Nat) : EnnResult TrState :=
  if numObs < tr.prevObs then
    Except.error (EnnError.invalidParam "num_obs went backwards")
  else if numObs = tr.prevObs then
    Except.ok tr
  else
    match tr.bestValue with
    | none =>
      Except.ok { tr with bestValue := some newBest, prevObs := numObs }
    | some best =>
      let scale := |best|
      let improved := decide (newBest > best + (1 / 1000) * scale)
      let (succ', fail', best') :=
        if improved then
          (tr.succCount + 1, 0, max best newBest)
        else
          (0, tr.failCount + 1, best)
      let (len', succ'', fail'') :=
        if succ' ≥ tr.succTol then
          (min tr.cfg.maxLen (tr.length * 2), 0, fail')
        else if fail' ≥ tr.failTol then
          (tr.length / 2, succ', 0)
        else
          (tr.length, succ', fail')
      Except.ok { tr with
        length := len'
        succCount := succ''
        failCount := fail''
        bestValue := some best'
        prevObs := numObs }

/-! ## 4. Strategy Mode and Progression -/

/-- Strategy lifecycle phase for candidate generation. -/
inductive Strategy where
  | initMode (numInit : Nat) (completed : Nat)
  | turboMode
  | hybridMode (numInit : Nat) (completed : Nat) (inInit : Bool)
  deriving Repr, BEq, DecidableEq

/-- Advance strategy progress and transition from initial phase to trust region. -/
def stratTell (strat : Strategy) (batchSize : Nat) : Strategy :=
  match strat with
  | Strategy.initMode numInit completed =>
    Strategy.initMode numInit (completed + batchSize)
  | Strategy.turboMode =>
    Strategy.turboMode
  | Strategy.hybridMode numInit completed inInit =>
    if inInit then
      let completed' := completed + batchSize
      let inInit' := decide (completed' < numInit)
      Strategy.hybridMode numInit completed' inInit'
    else
      Strategy.hybridMode numInit completed false

/-- Query initialization progress (completed evaluations, total initial budget). -/
def initProgress (strat : Strategy) : Option (Nat × Nat) :=
  match strat with
  | Strategy.initMode numInit completed => some (completed, numInit)
  | Strategy.hybridMode numInit completed inInit =>
    if inInit then some (completed, numInit) else none
  | Strategy.turboMode => none

/-! ## 5. Observation Append Deltas -/

/-- Record of newly appended observation coordinates and objectives. -/
structure ObsDelta (dim : Nat) where
  oldObs : Nat
  newObs : Nat
  xNew : Array (Fin dim → ℝ)
  yNew : Array ℝ

/-- Validate and construct an observation delta record. -/
def makeDelta (dim : Nat) (oldObs : Nat) (xNew : Array (Fin dim → ℝ))
    (yNew : Array ℝ) : EnnResult (ObsDelta dim) :=
  if xNew.size ≠ yNew.size then
    Except.error (EnnError.invalidShape [xNew.size] [yNew.size])
  else if xNew.size = 0 then
    Except.error (EnnError.invalidParam "empty batch")
  else
    Except.ok {
      oldObs := oldObs
      newObs := oldObs + xNew.size
      xNew := xNew
      yNew := yNew
    }

/-! ## 6. Modular Seams for Sibling Ownership -/

/-- Abstract interface for RNG and candidate proposal generators. -/
structure SeamRng (RngState : Type) where
  genInit : RngState → (dim : Nat) → (numArms : Nat) →
    (RngState × Array (Fin dim → ℝ))
  genTurbo : RngState → (dim : Nat) → (numArms : Nat) →
    (lb : Fin dim → ℝ) → (ub : Fin dim → ℝ) → (center : Fin dim → ℝ) →
    (RngState × Array (Fin dim → ℝ))

/-- Stable application boundary for TuRBO generation.  Keeping the field
application behind a definition also gives correspondence proofs a common
syntactic term across independently represented RNG states. -/
def turboGen {RngState : Type} (seam : SeamRng RngState)
    (rng : RngState) (dim numArms : Nat)
    (lb ub center : Fin dim → ℝ) : RngState × Array (Fin dim → ℝ) :=
  seam.genTurbo rng dim numArms lb ub center

/-- Incumbent center used by TuRBO, with the unit-box midpoint before the
first observation. -/
def turboCenter {dim : Nat} (bestX : Option (Fin dim → ℝ)) : Fin dim → ℝ :=
  bestX.getD (fun _ ↦ 1 / 2)

/-- Abstract interface for surrogate model fitting and acquisition selection. -/
structure SeamSurr (SurrState : Type) (dim : Nat) where
  fit : SurrState → (x : Array (Fin dim → ℝ)) → (y : Array ℝ) →
    (yvar : Option (Array ℝ)) → EnnResult SurrState
  fitAppend : SurrState → (xNew : Array (Fin dim → ℝ)) → (yNew : Array ℝ) →
    (yvarNew : Option (Array ℝ)) → EnnResult SurrState
  predict : SurrState → (x : Array (Fin dim → ℝ)) →
    EnnResult (Array ℝ × Array ℝ)
  selectArms : SurrState → (cand : Array (Fin dim → ℝ)) → (numArms : Nat) →
    EnnResult (Array (Fin dim → ℝ))
  clear : SurrState → SurrState

/-! ## 7. Complete Optimizer State and API -/

/-- Telemetry counters recording elapsed times and candidate counts. -/
structure Telemetry where
  dtFit : ℝ
  dtGen : ℝ
  dtSel : ℝ
  dtTell : ℝ
  numCand : Nat

/-- Initial zero telemetry record. -/
def telDefault : Telemetry where
  dtFit := 0
  dtGen := 0
  dtSel := 0
  dtTell := 0
  numCand := 0

/-- Comprehensive transition state of the TuRBO-ENN optimizer. -/
structure OptState (dim : Nat) (RngState : Type) (SurrState : Type) where
  bounds : Fin dim → BoundPair
  tr : TrState
  strat : Strategy
  surrogate : Option SurrState
  rng : RngState
  obsX : Array (Fin dim → ℝ)
  obsY : Array ℝ
  bestIdx : Option Nat
  bestX : Option (Fin dim → ℝ)
  bestY : Option ℝ
  restartGen : Nat
  telemetry : Telemetry
  yvarTell : Option Bool
  reinitLeft : Option Nat
  localBudget : Nat

/-- Local restart clears observations, resets trust region, and increments generation. -/
def optRestart {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) : OptState dim RngState SurrState :=
  { st with
    tr := trReset st.tr
    surrogate := st.surrogate.map seamSurr.clear
    obsX := #[]
    obsY := #[]
    bestIdx := none
    bestX := none
    bestY := none
    restartGen := st.restartGen + 1
    telemetry := telDefault
    reinitLeft := if st.localBudget > 0 then some st.localBudget else none }

/-- Factory creating a verified optimizer instance. -/
def optCreate {RngState SurrState : Type} (dim : Nat)
    (bounds : Fin dim → BoundPair)
    (trCfg : TrConfig)
    (strat : Strategy)
    (rngInit : RngState)
    (surrInit : Option SurrState)
    (budget : Nat)
    (numArms : Nat) : EnnResult (OptState dim RngState SurrState) :=
  if numArms = 0 then
    Except.error (EnnError.invalidParam "numArms must be positive")
  else
    Except.ok {
      bounds := bounds
      tr := trInit dim trCfg numArms
      strat := strat
      surrogate := surrInit
      rng := rngInit
      obsX := #[]
      obsY := #[]
      bestIdx := none
      bestX := none
      bestY := none
      restartGen := 0
      telemetry := telDefault
      yvarTell := none
      reinitLeft := none
      localBudget := budget
    }

/-- Ask transition generating candidate parameter evaluations. -/
def chooseArms {dim : Nat} {SurrState : Type}
    (seamSurr : SeamSurr SurrState dim) (surrogate : Option SurrState)
    (raw : Array (Fin dim → ℝ)) (numArms : Nat) : Array (Fin dim → ℝ) :=
  match surrogate with
  | some surr =>
    match seamSurr.selectArms surr raw numArms with
    | Except.ok selected => selected
    | Except.error _ => raw
  | none => raw

/-- Whether the current strategy requests initialization candidates. -/
def isInit : Strategy → Bool
  | Strategy.initMode _ _ => true
  | Strategy.turboMode => false
  | Strategy.hybridMode _ _ inInit => inInit

/-- Successful ask body after validation and the optional restart have run. -/
def askReady {dim : Nat} {RngState SurrState : Type}
    (seamRng : SeamRng RngState)
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (numArms : Nat) :
    Array (Fin dim → ℝ) × OptState dim RngState SurrState :=
  let (st1, unitCand) :=
    if isInit st.strat || st.reinitLeft.isSome then
      let (rng', raw) := seamRng.genInit st.rng dim numArms
      let reinit' := st.reinitLeft.map (fun left ↦ left - numArms)
      ({ st with rng := rng', reinitLeft := reinit' }, raw)
    else
      let center := turboCenter st.bestX
      let (lb, ub) := trBounds dim st.tr.length center
      let (rng', raw) := turboGen seamRng st.rng dim numArms lb ub center
      let selected := chooseArms seamSurr st.surrogate raw numArms
      ({ st with rng := rng' }, selected)
  let natCand := unitCand.map (fun u ↦ fun i ↦ fromUnit (st1.bounds i) (u i))
  (natCand, st1)

/-- Ask transition generating candidate parameter evaluations. -/
def optAsk {dim : Nat} {RngState SurrState : Type}
    (seamRng : SeamRng RngState)
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (numArms : Nat) :
    EnnResult (Array (Fin dim → ℝ) × OptState dim RngState SurrState) :=
  if numArms = 0 then
    Except.error (EnnError.invalidParam "numArms must be positive")
  else
    let st0 := if trNeeds st.tr then optRestart seamSurr st else st
    Except.ok (askReady seamRng seamSurr st0 numArms)

/-- Compute maximum element in a real array starting from an accumulator. -/
def batchMax (y : Array ℝ) (init : ℝ) : ℝ :=
  y.foldl max init

/-- Proceed with tell transition after validation checks succeed. -/
def appendSurr {dim : Nat} {SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (surrogate : Option SurrState)
    (xUnit : Array (Fin dim → ℝ))
    (y : Array ℝ)
    (yvar : Option (Array ℝ)) : Option SurrState :=
  match surrogate with
  | some s =>
    match seamSurr.fitAppend s xUnit y yvar with
    | Except.ok s' => some s'
    | Except.error _ => some s
  | none => none

/-- Proceed with tell transition after validation checks succeed. -/
def proceedTell {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (xNat : Array (Fin dim → ℝ))
    (y : Array ℝ)
    (yvar : Option (Array ℝ)) :
    EnnResult (OptState dim RngState SurrState) :=
  let xUnit := xNat.map (fun x ↦ fun i ↦ toUnit (st.bounds i) (x i))
  let obsX' := st.obsX ++ xUnit
  let obsY' := st.obsY ++ y
  let surr' := appendSurr seamSurr st.surrogate xUnit y yvar
  let initialBest :=
    match st.bestY with
    | some b => b
    | none => if h : 0 < y.size then y[0] else 0
  let newBatchBest := batchMax y initialBest
  let (bestY', bestIdx', bestX') :=
    match st.bestY with
    | some oldBest =>
      if newBatchBest > oldBest then
        (some newBatchBest, some (obsY'.size - 1),
         if h : 0 < xUnit.size then some xUnit[xUnit.size - 1] else none)
      else
        (st.bestY, st.bestIdx, st.bestX)
    | none =>
      (some newBatchBest, some 0,
       if h : 0 < xUnit.size then some xUnit[0] else none)
  match trUpdate st.tr newBatchBest obsY'.size with
  | Except.error err => Except.error err
  | Except.ok tr' =>
    let strat' := stratTell st.strat y.size
    Except.ok { st with
      tr := tr'
      strat := strat'
      surrogate := surr'
      obsX := obsX'
      obsY := obsY'
      bestIdx := bestIdx'
      bestX := bestX'
      bestY := bestY' }

/-- Tell transition consuming evaluated points and updating state. -/
def optTell {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (xNat : Array (Fin dim → ℝ))
    (y : Array ℝ)
    (yvar : Option (Array ℝ)) :
    EnnResult (OptState dim RngState SurrState) :=
  if xNat.size ≠ y.size then
    Except.error (EnnError.invalidShape [xNat.size] [y.size])
  else if xNat.size = 0 then
    Except.ok st
  else
    match st.yvarTell with
    | some exp =>
      if exp ≠ yvar.isSome then
        Except.error (EnnError.invalidParam "yvar mismatch on tell")
      else
        proceedTell seamSurr st xNat y yvar
    | none =>
      let st' := { st with yvarTell := some yvar.isSome }
      proceedTell seamSurr st' xNat y yvar

/-- Public optimizer commands used to state whole-execution correspondence. -/
inductive OptCommand (dim : Nat)
  | ask (numArms : Nat)
  | tell (x : Array (Fin dim → ℝ)) (y : Array ℝ)
      (yvar : Option (Array ℝ))

/-- Observable result of one public optimizer command. -/
inductive OptOutput (dim : Nat)
  | asked (candidates : Array (Fin dim → ℝ))
  | told

/-- One complete public optimizer transition. -/
def optimizerStep {dim : Nat} {RngState SurrState : Type}
    (seamRng : SeamRng RngState)
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) :
    OptCommand dim → EnnResult (OptOutput dim × OptState dim RngState SurrState)
  | .ask numArms =>
      (optAsk seamRng seamSurr st numArms).map
        (fun result ↦ (.asked result.1, result.2))
  | .tell x y yvar =>
      (optTell seamSurr st x y yvar).map (fun next ↦ (.told, next))

/-- Query total accumulated observation count. -/
def obsCount {dim : Nat} {RngState SurrState : Type}
    (st : OptState dim RngState SurrState) : Nat :=
  st.obsY.size

/-- Query natural coordinate representation of the best incumbent point. -/
def bestNatural {dim : Nat} {RngState SurrState : Type}
    (st : OptState dim RngState SurrState) : Option (Fin dim → ℝ) :=
  st.bestX.map (fun u ↦ fun i ↦ fromUnit (st.bounds i) (u i))

/-- Upstream `posterior_mu` / surrogate predict interface on natural inputs. -/
def optPredict {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (xNat : Array (Fin dim → ℝ)) : EnnResult (Array ℝ × Array ℝ) :=
  match st.surrogate with
  | some surr =>
    let xUnit := xNat.map (fun x ↦ fun i ↦ toUnit (st.bounds i) (x i))
    seamSurr.predict surr xUnit
  | none =>
    Except.error (EnnError.invalidParam "no surrogate configured")

/-- Append observations and return observation delta without trust-region step. -/
def optDelta {dim : Nat} {RngState SurrState : Type}
    (st : OptState dim RngState SurrState)
    (xNat : Array (Fin dim → ℝ))
    (y : Array ℝ) : EnnResult (ObsDelta dim × OptState dim RngState SurrState) :=
  match makeDelta dim st.obsY.size xNat y with
  | Except.error err => Except.error err
  | Except.ok delta =>
    let xUnit := xNat.map (fun x ↦ fun i ↦ toUnit (st.bounds i) (x i))
    let st' := { st with obsX := st.obsX ++ xUnit, obsY := st.obsY ++ y }
    Except.ok (delta, st')

/-- Reference Python `_tell_estimate` extracting surrogate mean predictions. -/
def tellEstimate {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (xNat : Array (Fin dim → ℝ))
    (yIn : Array ℝ) : Array ℝ :=
  match st.surrogate with
  | some surr =>
    let xUnit := xNat.map (fun x ↦ fun i ↦ toUnit (st.bounds i) (x i))
    match seamSurr.predict surr xUnit with
    | Except.ok (mu, _) => mu
    | Except.error _ => yIn
  | none => yIn

/-! ## 8. Soundness and Invariant Theorems -/

/-- Restart strictly increments the restart generation counter. -/
theorem genStep {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) :
    (optRestart seamSurr st).restartGen = st.restartGen + 1 := rfl

/-- Restart resets the trust region length to its initial configuration. -/
theorem lenStep {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) :
    (optRestart seamSurr st).tr.length = st.tr.cfg.initLen := rfl

/-- Restart clears all stored observations. -/
theorem obsReset {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) :
    (optRestart seamSurr st).obsX = #[] := rfl

/-- Requesting zero arms is strictly rejected by ask. -/
theorem askGate {dim : Nat} {RngState SurrState : Type}
    (seamRng : SeamRng RngState)
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState) :
    optAsk seamRng seamSurr st 0 = Except.error (EnnError.invalidParam "numArms must be positive") := by
  unfold optAsk
  split <;> simp_all

/-- Dimension mismatch between X and Y inputs is rejected with invalidShape. -/
theorem shapeGate {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yvar : Option (Array ℝ))
    (hdiff : x.size ≠ y.size) :
    optTell seamSurr st x y yvar = Except.error (EnnError.invalidShape [x.size] [y.size]) := by
  unfold optTell
  split <;> simp_all

/-- Conflicting presence/absence of yvar across tell invocations is rejected. -/
theorem yvarGate {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yvar : Option (Array ℝ))
    (hsize : x.size = y.size) (hpos : 0 < x.size)
    (exp : Bool) (hexp : st.yvarTell = some exp) (hmis : exp ≠ yvar.isSome) :
    optTell seamSurr st x y yvar = Except.error (EnnError.invalidParam "yvar mismatch on tell") := by
  unfold optTell
  have hne : ¬(x.size ≠ y.size) := by simp [hsize]
  have hzero : ¬(x.size = 0) := by exact ne_of_gt hpos
  simp only [hne, ↓reduceIte, hzero, hexp]
  split <;> simp_all

/-- A valid non-empty batch produces an observation delta with incremented count. -/
theorem deltaCount (dim : Nat) (old : Nat) (x : Array (Fin dim → ℝ)) (y : Array ℝ)
    (hsize : x.size = y.size) (hpos : 0 < x.size) :
    ∃ d, makeDelta dim old x y = Except.ok d ∧ d.newObs = old + x.size := by
  unfold makeDelta
  have hne : ¬(x.size ≠ y.size) := by simp [hsize]
  have hzero : ¬(x.size = 0) := by exact ne_of_gt hpos
  simp [hne, hzero]

/-- Prediction without a configured surrogate yields an invalidParam error. -/
theorem emptySurr {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (x : Array (Fin dim → ℝ))
    (hsurr : st.surrogate = none) :
    optPredict seamSurr st x = Except.error (EnnError.invalidParam "no surrogate configured") := by
  unfold optPredict
  simp [hsurr]

/-- When surrogate is absent, tellEstimate passes raw inputs unchanged. -/
theorem rawEst {dim : Nat} {RngState SurrState : Type}
    (seamSurr : SeamSurr SurrState dim)
    (st : OptState dim RngState SurrState)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ)
    (hsurr : st.surrogate = none) :
    tellEstimate seamSurr st x y = y := by
  unfold tellEstimate
  simp [hsurr]

/-- Initialization progress is none when hybrid strategy has entered turbo mode. -/
theorem progDone (numInit completed : Nat) :
    initProgress (Strategy.hybridMode numInit completed false) = none := rfl

/-- Initialization progress returns completed and budget when in init mode. -/
theorem progInit (numInit completed : Nat) :
    initProgress (Strategy.hybridMode numInit completed true) = some (completed, numInit) := rfl

/-- Adding observations increments the accumulated observation count. -/
theorem deltaObs {dim : Nat} {RngState SurrState : Type}
    (st : OptState dim RngState SurrState)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ)
    (hsize : x.size = y.size) (hpos : 0 < x.size) :
    ∃ delta st', optDelta st x y = Except.ok (delta, st') ∧ st'.obsY.size = st.obsY.size + y.size := by
  unfold optDelta makeDelta
  have hne : ¬(x.size ≠ y.size) := by simp [hsize]
  have hzero : ¬(x.size = 0) := by exact ne_of_gt hpos
  simp [hne, hzero]

end

end BauxResearch.Semantics.TurboApi
