import BauxResearch.Semantics.Sobol

/-!
# Exact ENN 0.4.1 initialization transitions

This module gives executable binary64 semantics for the initialization and local-restart
branches of the pinned ENN 0.4.1 optimizer.  Candidate coordinates are IEEE-754 bit patterns,
and the finite random-word state is returned even when evaluation fails.  The latter matters:
Rust's caller-owned RNG is not rolled back when a candidate generator returns an error.

This file also covers the post-resampling TuRBO candidate and random-acquisition path. MORBO
resampling/scalarization, surrogate-backed acquisition, observation, and backend branches live in
separate modules or remain open composition obligations.
-/

namespace BauxResearch.Semantics.Opt

open Sampling

/-- Initialization design selected by `InitStrategy` in ENN 0.4.1. -/
inductive InitKind where
  | lhd
  | random
  deriving Repr, DecidableEq

/-- Errors observable at the public ask boundary represented by this module. -/
inductive AskError where
  | zeroArms
  | sampling (failure : Failure)
  deriving Repr, DecidableEq

/-- State read or mutated by an initialization-phase ask. -/
structure InitState where
  dim : Nat
  kind : InitKind
  reinitLeft : Option Nat
  rng : RngState
  deriving Repr, DecidableEq

/-- An ask result paired with the successor state, including on failure. -/
structure AskStep where
  result : Except AskError Matrix
  state : InitState
  deriving Repr, DecidableEq

/-- Unit-cube lower bounds used by both initialization generators. -/
def unitLower (dim : Nat) : Row :=
  Array.replicate dim 0

/-- Unit-cube upper bounds used by uniform initialization. -/
def unitUpper (dim : Nat) : Row :=
  Array.replicate dim one

/-- Two-column unit bounds used by Latin-hypercube initialization. -/
def unitBounds (dim : Nat) : Array (Word × Word) :=
  Array.replicate dim (0, one)

/-- Execute the source-selected initialization generator. -/
def initDraw (kind : InitKind) (dim count : Nat) : Draw Matrix :=
  match kind with
  | .lhd => genLhd count dim (unitBounds dim)
  | .random => genUniform (unitLower dim) (unitUpper dim) count

/-- ENN's exact local-restart request count, including its one-arm floor. -/
def reinitCount (requested left : Nat) : Nat :=
  max 1 (min requested left)

/-- Preserve the caller-owned RNG successor when lifting a candidate draw. -/
def finishDraw (state : InitState) (draw : Except Failure Matrix × RngState) : AskStep :=
  { result := draw.1.mapError .sampling
    state := { state with rng := draw.2 } }

/-- Public initialization/reinitialization ask semantics from ENN 0.4.1.

The public zero-arm guard runs before restart or strategy dispatch.  A present local-restart
budget takes precedence over the ordinary initialization phase.
-/
def askInit (state : InitState) (requested : Nat) : AskStep :=
  if requested = 0 then
    { result := .error .zeroArms, state }
  else
    let count := match state.reinitLeft with
      | none => requested
      | some left => reinitCount requested left
    finishDraw state (run (initDraw state.kind state.dim count) state.rng)

/-- ENN's saturating local-restart budget update after a tell. -/
def consumeReinit (left : Option Nat) (count : Nat) : Option Nat :=
  match left with
  | none => none
  | some left =>
    let remaining := left - count
    if remaining = 0 then none else some remaining

/-! ## TuRBO candidate construction -/

/-- Candidate-pool configuration after the public configuration has been resolved. -/
structure CandidateCfg where
  minCount : Nat
  maxCount : Nat
  perDim : Nat
  perArm : Nat
  mode : Sobol.Mode
  fast : Bool
  deriving Repr, DecidableEq

/-- Largest `usize` on the pinned 64-bit target. -/
def usizeMax : Nat :=
  2 ^ 64 - 1

/-- Pinned-target `usize::saturating_mul`. -/
def satMul (left right : Nat) : Nat :=
  min usizeMax (left * right)

/-- Pinned-target `usize::saturating_add`. -/
def satAdd (left right : Nat) : Nat :=
  min usizeMax (left + right)

/-- Exact candidate-pool formula from `CandidateConfig::num_candidates`. -/
def poolSize (config : CandidateCfg) (dim arms : Nat) : Nat :=
  max config.minCount
    (min config.maxCount (satAdd (satMul config.perDim dim) (satMul config.perArm arms)))

/-- Rust `f64::clamp(value, 0.0, 1.0)`, retaining NaNs and signed zero. -/
def clampUnit (value : Word) : Word :=
  if Float.ltDouble value 0 then 0
  else if Float.ltDouble one value then one
  else value

/-- Binary64 half-length vector with the same rounding points as ndarray expressions. -/
def halfLengths (center : Row) (length : Word) (scales : Option Row) : Except Failure Row := do
  match scales with
  | none => return Array.replicate center.size (divide length (numWord 2))
  | some scales =>
    let mut result := #[]
    for j in [:center.size] do
      let some scale := scales[j]? | throw .panic
      result := result.push (divide (mul scale length) (numWord 2))
    return result

/-- Exact TuRBO one-dimensional bounds, including every binary64 rounding point. -/
def turboBounds (center : Row) (length : Word) (scales : Option Row) :
    Except Failure (Row × Row) := do
  let half ← halfLengths center length scales
  let mut lower := #[]
  let mut upper := #[]
  for j in [:center.size] do
    let some radius := half[j]? | throw .panic
    lower := lower.push (clampUnit (sub center[j]! radius))
    upper := upper.push (clampUnit (add center[j]! radius))
  return (lower, upper)

/-- Post-resampling TuRBO candidate generation and all of its successor random state. -/
def turboDraw (config : CandidateCfg) (morbo : Bool) (center : Row) (length : Word)
    (scales : Option Row) (arms : Nat) (engine : Option Sobol.Engine) (rng : RngState) :
    Except Failure Matrix × Option Sobol.Engine × RngState :=
  match turboBounds center length scales with
  | .error error => (.error error, engine, rng)
  | .ok (lower, upper) =>
    Sobol.turbo config.fast morbo config.mode center lower upper
      (poolSize config center.size arms) engine rng

/-- Stack selected candidate rows in index order. -/
def selectRows (candidates : Matrix) (indices : Array Nat) : Except Failure Matrix := do
  let mut result := #[]
  for index in indices do
    let some row := candidates[index]? | throw .panic
    result := result.push row
  return result

/-- Exact `RandomAcquisition::select` followed by `select_by_indices`. -/
def randomPick (candidates : Matrix) (arms : Nat) : Draw Matrix := do
  if candidates.isEmpty then throw .invalid
  let indices ← shuffle candidates.size
  match selectRows candidates (indices.extract 0 arms) with
  | .error error => throw error
  | .ok selected => return selected

/-- TuRBO candidate draw followed by the random acquisition/fallback path. -/
def randomTurbo (config : CandidateCfg) (morbo : Bool) (center : Row) (length : Word)
    (scales : Option Row) (arms : Nat) (engine : Option Sobol.Engine) (rng : RngState) :
    Except Failure Matrix × Option Sobol.Engine × RngState :=
  let (candidates, engine, rng) :=
    turboDraw config morbo center length scales arms engine rng
  match candidates with
  | .error error => (.error error, engine, rng)
  | .ok candidates =>
    let (selected, rng) := run (randomPick candidates arms) rng
    (selected, engine, rng)

/-- A rejected zero-arm ask is observationally inert, including its RNG. -/
theorem zeroAsk (state : InitState) :
    askInit state 0 = { result := .error .zeroArms, state } := by
  simp [askInit]

/-- Without a restart budget, the requested initialization batch is passed through exactly. -/
theorem normalAsk (state : InitState) (requested : Nat) (hpos : requested ≠ 0)
    (hleft : state.reinitLeft = none) :
    askInit state requested =
      finishDraw state (run (initDraw state.kind state.dim requested) state.rng) := by
  simp [askInit, hpos, hleft]

/-- A local restart uses exactly the source expression `min(requested, left).max(1)`. -/
theorem localAsk (state : InitState) (requested left : Nat) (hpos : requested ≠ 0)
    (hleft : state.reinitLeft = some left) :
    askInit state requested =
      finishDraw state
        (run (initDraw state.kind state.dim (reinitCount requested left)) state.rng) := by
  simp [askInit, hpos, hleft]

/-- The source's one-arm floor applies even to the otherwise unreachable zero budget state. -/
theorem emptyBudget (requested : Nat) :
    reinitCount requested 0 = 1 := by
  simp [reinitCount]

/-- A positive local budget never asks for more than the requested batch. -/
theorem countBound (requested left : Nat) (hreq : 0 < requested) (_hleft : 0 < left) :
    reinitCount requested left ≤ requested := by
  simp only [reinitCount]
  exact max_le hreq (min_le_left requested left)

/-- Consuming the entire remaining local budget exits reinitialization. -/
theorem consumeAll (left count : Nat) (hle : left ≤ count) :
    consumeReinit (some left) count = none := by
  simp [consumeReinit, Nat.sub_eq_zero_of_le hle]

/-- The default 0.4.1 pool has 200 candidates in two dimensions. -/
theorem defaultPool :
    let config : CandidateCfg :=
      { minCount := 10, maxCount := 5000, perDim := 100, perArm := 0,
        mode := .uniform, fast := false }
    poolSize config 2 1 = 200 := by
  decide +kernel

/-- Saturating arithmetic occurs before the configured cap is applied. -/
theorem saturatedPool :
    let config : CandidateCfg :=
      { minCount := 10, maxCount := 5000, perDim := 2 ^ 64 - 1, perArm := 1,
        mode := .uniform, fast := false }
    poolSize config 2 1 = 5000 := by
  decide +kernel

/-- Default center and length produce the exact rounded 0.1/0.9 bounds. -/
theorem defaultBounds :
    turboBounds #[0x3fe0000000000000, 0x3fe0000000000000]
      0x3fe999999999999a none =
      .ok (#[0x3fb9999999999998, 0x3fb9999999999998],
        #[0x3feccccccccccccd, 0x3feccccccccccccd]) := by
  decide +kernel

/-- A lengthscale shape mismatch is a source-level panic, not a fabricated bound. -/
theorem scaleMismatch :
    turboBounds #[0, 0] one (some #[one]) = .error .panic := by
  decide +kernel

/-- A missing Sobol engine takes the source's uniform fallback with identical RNG state. -/
theorem sobolFallback (config : CandidateCfg) (morbo : Bool) (center : Row)
    (length : Word) (scales : Option Row) (arms : Nat) (rng : RngState)
    (hmode : config.mode = .sobol) (hfast : config.fast = false)
    (hbounds : turboBounds center length scales = .ok (unitLower center.size, unitUpper center.size)) :
    turboDraw config morbo center length scales arms none rng =
      Sobol.turbo false morbo .uniform center (unitLower center.size) (unitUpper center.size)
        (poolSize config center.size arms) none rng := by
  simp [turboDraw, hbounds, hmode, hfast, Sobol.turbo, Sobol.dispatch]

/-- Random selection rejects an empty candidate matrix before drawing RNG words. -/
theorem randomEmpty (arms : Nat) (rng : RngState) :
    run (randomPick #[] arms) rng = (.error .invalid, rng) := by
  rfl

/-- Selecting the only candidate performs no shuffle draw. -/
theorem randomSingle (row : Row) (rng : RngState) :
    run (randomPick #[row] 1) rng = (.ok #[row], rng) := by
  simp [randomPick, shuffle, selectRows, run]
  have hp : (pure #[row] : Except Failure Matrix) = .ok #[row] := rfl
  rw [hp]
  rfl

set_option maxRecDepth 16384 in
/-- The complete public LHD branch, including its successor RNG state. -/
theorem lhdAsk :
    let state : InitState :=
      { dim := 2, kind := .lhd, reinitLeft := none, rng := seeded 42 16 }
    let result := askInit state 4
    result.result = .ok #[
      #[4591976496691633548, 4602373228015199930],
      #[4605007952137404557, 4603818566212700789],
      #[4600044019043131640, 4591701732378506428],
      #[4604339351361603652, 4605247852531002938]] ∧
    result.state.rng.consumed = 14 ∧
    result.state.rng.words.head? = some 9648369018563374588 := by
  decide +kernel

set_option maxRecDepth 16384 in
/-- The complete public uniform branch, including its successor RNG state. -/
theorem randomAsk :
    let state : InitState :=
      { dim := 2, kind := .random, reinitLeft := none, rng := seeded 42 8 }
    let result := askInit state 2
    result.result = .ok #[#[4602918027047224548, 4603063653651445162],
      #[4603907987511953958, 4600983695946374540]] ∧
    result.state.rng.words.head? = some 633513173585076202 := by
  decide +kernel

/-- A failed finite-trace model exposes the consumed RNG successor instead of rolling it back. -/
theorem failedRng :
    let state : InitState :=
      { dim := 1, kind := .random, reinitLeft := none, rng := { words := [] } }
    let result := askInit state 1
    result.result = .error (.sampling .exhausted) ∧ result.state.rng = state.rng := by
  decide +kernel

#print axioms zeroAsk
#print axioms normalAsk
#print axioms localAsk
#print axioms emptyBudget
#print axioms countBound
#print axioms consumeAll
#print axioms defaultPool
#print axioms saturatedPool
#print axioms defaultBounds
#print axioms scaleMismatch
#print axioms sobolFallback
#print axioms randomEmpty
#print axioms randomSingle
#print axioms lhdAsk
#print axioms randomAsk
#print axioms failedRng

end BauxResearch.Semantics.Opt
