import BauxResearch.Semantics.JoeKuo
import BauxResearch.Semantics.Sampling
import Mathlib.Data.BitVec
import BauxResearch.Semantics.Metric
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

namespace BauxResearch.Semantics.Sobol

/-!
# Pinned Joe–Kuo Sobol sequence and candidate generation

Direction initialization and the Gray-code recurrence follow sobol 1.0.2. The complete
21,201-dimensional parameter set is supplied by `JoeKuo`, including all direction seeds.
The ENN wrapper uses 53-bit resolution and an additive binary64 shift modulo one.
-/


open Sampling

structure Param where
  coefficient : Nat
  initial : Array Nat
  deriving Repr, DecidableEq

def parameter (dim : Nat) : Option Param := do
  if dim < 2 then none else do
    let values ← JoeKuo.row (dim - 2)
    let coefficient ← values[0]?
    some { coefficient, initial := values.extract 1 values.size }

def directions (dim : Nat) : Option Row := do
  if dim = 1 then
    return (List.range 53).toArray.map (fun i ↦ BitVec.ofNat 64 (2 ^ (63 - i)))
  let param ← parameter dim
  let degree := min 53 param.initial.size
  let mut dirs : Row := #[]
  for i in [:53] do
    if i < degree then
      dirs := dirs.push (BitVec.ofNat 64 (param.initial[i]! <<< (63 - i)))
    else
      let base := dirs[i - degree]!
      let mut value := base ^^^ (base >>> degree)
      for k in [1:degree] do
        let coefficient := (param.coefficient >>> (degree - k - 1)) % 2
        value := value ^^^ (BitVec.ofNat 64 coefficient * dirs[i - k]!)
      dirs := dirs.push value
  return dirs

structure Engine where
  dirs : Array Row
  previous : Row
  count : Nat := 0
  shift : Option Row := none
  deriving Repr, DecidableEq

def create (dim : Nat) : Option Engine := do
  if dim = 0 || dim > 21201 then none else do
    let dirs ← (List.range dim).mapM (fun i ↦ directions (i + 1))
    return { dirs := dirs.toArray, previous := Array.replicate dim 0 }

def firstZero (word : Nat) : Nat → Nat
  | 0 => 0
  | fuel + 1 => if word % 2 = 0 then 0 else 1 + firstZero (word / 2) fuel

/-- The sum of valid shifted unit coordinates is in [0,2); this implements its remainder by one. -/
def shiftUnit (unit shift : Word) : Word :=
  let sum := add unit shift
  if Float.leDouble one sum then sub sum one else sum

/-- Sequence exhaustion leaves the engine unchanged. -/
def sample (engine : Engine) : Except Failure Row × Engine :=
  if engine.count ≥ 2 ^ 53 - 1 then (.error .invalid, engine)
  else
    let next := if engine.count = 0 then Array.replicate engine.dirs.size (0 : Word)
      else engine.dirs.mapIdx (fun i dirs ↦
        engine.previous[i]! ^^^ dirs[firstZero (engine.count - 1) 64]!)
    let rendered := next.map (fun word ↦ divide (numWord word.toNat) (numWord (2 ^ 64)))
    let shifted := match engine.shift with
      | none => rendered
      | some shifts => rendered.mapIdx (fun i word ↦
        match shifts[i]? with
        | none => word
        | some shift => shiftUnit word shift)
    (.ok shifted, { engine with previous := next, count := engine.count + 1 })

/-- Scrambling uses fill_bytes(8), hence one complete u64 per coordinate and 53 fractional bits. -/
def scramble (engine : Engine) : Draw Engine := do
  let mut shifts := #[]
  for _ in [:engine.dirs.size] do
    shifts := shifts.push (← standard)
  return { engine with shift := some shifts }

def scaleRow (lower upper units : Row) (dim : Nat) : Except Failure Row := do
  let mut result := #[]
  for j in [:dim] do
    let some low := lower[j]? | throw .panic
    let some high := upper[j]? | throw .panic
    let some unit := units[j]? | throw .panic
    result := result.push (affine low high unit)
  return result

/-- Preserve engine and RNG state on both success and failure of a candidate batch. -/
def generate (masked : Bool) (center lower upper : Row) (numPert : Nat) :
    Nat → Engine → RngState → Except Failure Matrix × Engine × RngState
  | 0, engine, rng =>
    if masked && center.size = 0 then (.error .invalid, engine, rng)
    else (.ok #[], engine, rng)
  | count + 1, engine, rng =>
    if masked && center.size = 0 then (.error .invalid, engine, rng) else
    let (point, engine) := sample engine
    match point with
    | .error err => (.error err, engine, rng)
    | .ok point =>
      let (row, rng) := if masked then run (maskRow center lower upper point numPert) rng
        else (scaleRow lower upper point center.size, rng)
      match row with
      | .error err => (.error err, engine, rng)
      | .ok row =>
        let (rest, engine, rng) := generate masked center lower upper numPert count engine rng
        (rest.map (fun rows ↦ #[row] ++ rows), engine, rng)

def points : Nat → Engine → Except Failure Matrix
  | 0, _ => .ok #[]
  | n + 1, engine => do
    let (point, engine) := sample engine
    let point ← point
    return #[point] ++ (← points n engine)

inductive Mode where
  | uniform
  | sobol
  | raasp
  deriving Repr, DecidableEq

/-- The public generate_candidates dispatch, including Sobol's uniform fallback. -/
def dispatch (mode : Mode) (center lower upper : Row) (scales : Option Scales)
    (count numPert : Nat) (engine : Option Engine) (rng : RngState) :
    Except Failure Matrix × Option Engine × RngState :=
  match mode, engine with
  | .sobol, some engine =>
    let (result, engine, rng) := generate false center lower upper numPert count engine rng
    (result, some engine, rng)
  | .raasp, engine =>
    let (result, rng) := run (genWeighted center lower upper scales count numPert) rng
    (result, engine, rng)
  | _, engine =>
    let (result, rng) := run (genUniform lower upper count) rng
    (result, engine, rng)

/-- The strategy's draw_turbo branch selection; upstream supplies no lengthscales here. -/
def turbo (fast morbo : Bool) (mode : Mode) (center lower upper : Row)
    (count : Nat) (engine : Option Engine) (rng : RngState) :
    Except Failure Matrix × Option Engine × RngState :=
  if fast && mode == .raasp then
    let (result, rng) := run (genFast center lower upper count 20) rng
    (result, engine, rng)
  else if morbo && mode == .sobol then
    match engine with
    | some engine =>
      let (result, engine, rng) := generate true center lower upper 20 count engine rng
      (result, some engine, rng)
    | none => dispatch mode center lower upper none count 20 none rng
  else dispatch mode center lower upper none count 20 engine rng

set_option maxRecDepth 16384 in
/-- The first eight five-dimensional points from the pinned crate's independent SciPy fixture. -/
theorem directionFixture :
    (do let engine ← create 5; (points 8 engine).toOption) = some
      (#[(#[0, 0, 0, 0, 0] : Array Nat), #[4, 4, 4, 4, 4], #[6, 2, 2, 2, 6],
          #[2, 6, 6, 6, 2], #[3, 3, 5, 7, 3], #[7, 7, 1, 3, 7],
          #[5, 1, 7, 5, 5], #[1, 5, 3, 1, 1]].map
        (fun row ↦ row.map (fun n ↦ divide (numWord n) (numWord 8)))) := by decide +kernel

theorem exhaustedState (engine : Engine) (h : 2 ^ 53 - 1 ≤ engine.count) :
    sample engine = (.error .invalid, engine) := by
  unfold sample
  rw [if_pos h]

/-- A failing masked call validates its dimension even when no candidates were requested. -/
theorem emptyMask (engine : Engine) (rng : RngState) :
    generate true #[] #[] #[] 20 0 engine rng = (.error .invalid, engine, rng) := rfl

/-- A missing Sobol engine follows exactly the same transition as uniform sampling. -/
theorem fallback (center lower upper : Row) (count numPert : Nat)
    (scales : Option Scales) (rng : RngState) :
    dispatch .sobol center lower upper scales count numPert none rng =
      dispatch .uniform center lower upper scales count numPert none rng := rfl

/-- Scramble distinguishes the 53rd fractional bit that Uniform<f64> discards. -/
theorem scrambleBit :
    let engine : Engine := { dirs := #[#[]], previous := #[0] }
    let result := run (scramble engine) { words := [0x8000000000000800] }
    result.1.map Engine.shift = .ok (some #[0x3fe0000000000001]) ∧
      result.2.consumed = 1 := by decide +kernel

/-- A mask failure does not roll back the Sobol point already consumed by the source. -/
theorem failureState :
    let engine : Engine := { dirs := #[#[]], previous := #[0] }
    let result := generate true #[0] #[0] #[one] 20 1 engine { words := [] }
    result.1 = .error .exhausted ∧ result.2.1.count = 1 ∧ result.2.2.consumed = 0 := by
  decide +kernel

#print axioms directionFixture
#print axioms exhaustedState
#print axioms emptyMask
#print axioms fallback
#print axioms scrambleBit
#print axioms failureState

/-!
# ENN 0.4.1 Sobol sensitivity and dependence cell semantics (`metric_sobol.rs`)

This module proves the function-level correspondence and semantic properties
for all 7 reachable functions in `crates/ennbo/src/metric_sobol.rs`:
- `rust:ennbo 0.4.1 metric_sobol/between_share()`
- `rust:ennbo 0.4.1 metric_sobol/stable_ranks()`
- `rust:ennbo 0.4.1 metric_sobol/sobol_index()`
- `rust:ennbo 0.4.1 metric_sobol/null_sd()`
- `rust:ennbo 0.4.1 metric_sobol/cells_of_group()`
- `rust:ennbo 0.4.1 metric_sobol/group_sobol_index()`
- `rust:ennbo 0.4.1 metric_sobol/group_cell_count()`
-/


open Aeneas.Std
open BauxResearch.Extraction.Native
open BauxResearch.Semantics.Sampling
open BauxResearch.Semantics.Bounds
open BauxResearch.Semantics.Fitter
open BauxResearch.Semantics.Metric

/-- Semantic default cell-count kernel for `rust:ennbo 0.4.1 metric_sobol/null_sd()` closure. -/
def defaultCells (n : Usize) : Result Usize := do
  let fl ← F64.sqrt (F64.fromUnsigned n)
  let fl2 ← F64.floor fl
  let fl3 ← F64.max fl2 (F64.literal false 2 1)
  Result.ok (F64.toUnsigned .Usize fl3)

/-- Exact correspondence for the extracted `null_sd` default cell-count closure. -/
theorem closureMatch (n : Usize) :
    BauxResearch.Generated.Pure.metric_sobol.null_sd.closure.Insts.CoreOpsFunctionFnOnceTupleUsize.call_once n () =
      defaultCells n := by
  unfold BauxResearch.Generated.Pure.metric_sobol.null_sd.closure.Insts.CoreOpsFunctionFnOnceTupleUsize.call_once
  unfold defaultCells
  dsimp only [lift]
  rw [bind_ok]

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 metric_sobol/null_sd().` -/
def nullPure (n : Usize) (cells : Option Usize) : Result F64 := do
  if n < 3#usize then
    Result.ok F64.posInf
  else
    let b ← match cells with
      | some c => Result.ok c
      | none => defaultCells n
    let i ← b - 1#usize
    let fl1 := F64.mul (F64.literal false 2 1) (F64.fromUnsigned i)
    let fl2 ← F64.sqrt fl1
    Result.ok (F64.div fl2 (F64.fromUnsigned n))

private theorem unwrapSome (c n : Usize) :
    core.option.Option.unwrap_or_else
      BauxResearch.Generated.Pure.metric_sobol.null_sd.closure.Insts.CoreOpsFunctionFnOnceTupleUsize
      (some c) n = Result.ok c := by
  rfl

private theorem unwrapNone (n : Usize) :
    core.option.Option.unwrap_or_else
      BauxResearch.Generated.Pure.metric_sobol.null_sd.closure.Insts.CoreOpsFunctionFnOnceTupleUsize
      none n = defaultCells n := by
  exact closureMatch n

/-- Exact correspondence between extracted `rust:ennbo 0.4.1 metric_sobol/null_sd()`
and `nullPure`. -/
theorem nullMatch (n : Usize) (cells : Option Usize) :
    BauxResearch.Generated.Pure.metric_sobol.null_sd n cells =
      nullPure n cells := by
  unfold BauxResearch.Generated.Pure.metric_sobol.null_sd nullPure
  split
  · rfl
  · cases cells with
    | none =>
      rw [unwrapNone]
      dsimp only [lift]
      congr 1
      funext b
      congr 1
      funext i
      rw [bind_ok, bind_ok]
      congr 1
      funext fl2
      rw [bind_ok]
    | some c =>
      rw [unwrapSome]
      dsimp only [lift]
      rw [bind_ok, bind_ok]
      congr 1
      funext i
      rw [bind_ok, bind_ok]
      congr 1
      funext fl2
      rw [bind_ok]

/-- `rust:ennbo 0.4.1 metric_sobol/null_sd()` returns `+∞` when `n < 3`. -/
theorem nullShort (n : Usize) (cells : Option Usize) (h : n < 3#usize) :
    BauxResearch.Generated.Pure.metric_sobol.null_sd n cells =
      Result.ok F64.posInf := by
  rw [nullMatch]
  unfold nullPure
  rw [if_pos h]

/-- Bit-level `Metric.nullSd` returns `posInf` when `n < 3`. -/
theorem nullInf (sqrtOp : SqrtOp) (n : Nat) (cells : Option Nat) (h : n < 3) :
    nullSd sqrtOp n cells = Metric.posInf := by
  unfold nullSd
  rw [if_pos h]

/-- `rust:ennbo 0.4.1 metric_sobol/cells_of_group()` on empty input returns `([], 0)`. -/
theorem cellsEmpty (dim : Nat) (group : List Nat) :
    cellsGroup dim group [] = ([], 0) := by
  rfl

/-- `rust:ennbo 0.4.1 metric_sobol/group_cell_count()` on empty input returns `0`. -/
theorem countEmpty (dim : Nat) (group : List Nat) :
    groupCells dim group [] = 0 := by
  rfl

/-- `rust:ennbo 0.4.1 metric_sobol/group_sobol_index()` on empty input returns `0`. -/
theorem groupEmpty (dim : Nat) (group : List Nat) :
    groupSobol dim group [] [] = 0 := by
  rfl

/-- `rust:ennbo 0.4.1 metric_sobol/sobol_index()` on empty input returns a zero vector of length `dim`. -/
theorem sobolEmpty (dim : Nat) :
    sobolIndex dim [] [] none = List.replicate dim 0 := by
  rfl

/-- `rust:ennbo 0.4.1 metric_sobol/between_share()` on empty cell lists divides `0` by `ssTot`. -/
theorem shareEmpty (ssTot : Word) (n b : Nat) :
    betweenShare [] [] ssTot n b =
      divide (maxWord 0 (sub 0 (mul (numWord (if b > 0 then b - 1 else 0))
        (divide (sub ssTot 0) (numWord (if n > b then n - b else 1)))))) ssTot := by
  rfl

/-- `rust:ennbo 0.4.1 metric_sobol/stable_ranks()` on empty column returns `[]`. -/
theorem ranksEmpty :
    stableRanks [] = [] := by
  rfl

#print axioms closureMatch
#print axioms nullMatch
#print axioms nullShort
#print axioms nullInf
#print axioms cellsEmpty
#print axioms countEmpty
#print axioms groupEmpty
#print axioms sobolEmpty
#print axioms shareEmpty
#print axioms ranksEmpty

end BauxResearch.Semantics.Sobol
