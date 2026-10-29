import BauxResearch.Semantics.ChaCha
import BauxResearch.Semantics.Float

/-!
# Candidate sampling over a finite random-word trace

This evaluator follows the pinned rand 0.8.5 sampling operations on a 64-bit host.
Arithmetic rounds after each operation using FloatLib's binary64 model. Trace exhaustion
is an explicit evaluator outcome, distinct from an upstream invalid-input error or panic.
`seeded` produces the trace by executing ChaCha12 rather than supplying sampled outputs.
Source correspondence and unbounded execution are not established by finite regression tests.
-/

namespace BauxResearch.Semantics.Sampling

open FloatLib.Floats.Formats.BinaryInterchange

abbrev Word := BitVec 64
abbrev Row := Array Word
abbrev Matrix := Array Row

inductive Failure where
  | exhausted
  | invalid
  | panic
  deriving DecidableEq, Repr

structure RngState where
  words : List Word
  /-- A single pending 32-bit half after an odd number of `next_u32` calls.
  Keeping it explicit makes mixed `next_u32`/`next_u64` consumption agree with
  rand_core's `BlockRng<u32>` adapter. -/
  pending : List (BitVec 32) := []
  consumed : Nat := 0
  deriving DecidableEq, Repr

abbrev Draw := ExceptT Failure (StateM RngState)

def seeded (seed : Word) (count : Nat) : RngState :=
  { words := (ChaCha.takeWords count (ChaCha.seeded seed)).1 }

def run (action : Draw α) (state : RngState) : Except Failure α × RngState :=
  action.run state

def nextWord : Draw Word := do
  let state ← get
  match state.pending with
  | first :: second :: pending =>
      set ({ state with pending := pending, consumed := state.consumed + 1 } : RngState)
      return BitVec.ofNat 64 (first.toNat + (second.toNat <<< 32))
  | first :: [] =>
      match state.words with
      | [] => throw .exhausted
      | word :: rest =>
          let low := BitVec.ofNat 32 word.toNat
          let high := BitVec.ofNat 32 (word.toNat >>> 32)
          set ({ words := rest, pending := [high], consumed := state.consumed + 1 } : RngState)
          return BitVec.ofNat 64 (first.toNat + (low.toNat <<< 32))
  | [] =>
      match state.words with
      | [] => throw .exhausted
      | word :: rest =>
          set ({ words := rest, consumed := state.consumed + 1 } : RngState)
          return word

/-- The `next_u32` view of the same finite stream. A following `nextWord`
combines the pending high half with the next stream half, exactly as
rand_core's `BlockRng<u32>::next_u64` does. -/
def nextHalf : Draw (BitVec 32) := do
  let state ← get
  match state.pending with
  | value :: rest =>
      set ({ state with pending := rest, consumed := state.consumed + 1 } : RngState)
      return value
  | [] =>
      match state.words with
      | [] => throw .exhausted
      | word :: rest =>
          let low := BitVec.ofNat 32 word.toNat
          let high := BitVec.ofNat 32 (word.toNat >>> 32)
          set ({ words := rest, pending := [high], consumed := state.consumed + 1 } : RngState)
          return low

def numWord (n : Nat) : Word := (Model.roundRat FloatFormat.binary64 false n 1).bits

def fraction (word : Word) (discard : Nat) : Word :=
  (Model.roundRat FloatFormat.binary64 false (word >>> discard).toNat (2 ^ (64 - discard))).bits

def add (x y : Word) : Word := (Float.addDouble (Float.wordDouble x) (Float.wordDouble y)).bits

def sub (x y : Word) : Word := (Float.subDouble (Float.wordDouble x) (Float.wordDouble y)).bits

def mul (x y : Word) : Word := (Float.mulDouble (Float.wordDouble x) (Float.wordDouble y)).bits

def divide (x y : Word) : Word := (Float.divDouble (Float.wordDouble x) (Float.wordDouble y)).bits

def one : Word := 0x3ff0000000000000

/-- Standard<f64> uses 53 random bits, unlike Uniform<f64>'s 52. -/
def standard : Draw Word := do return fraction (← nextWord) 11

/-- rand 0.8.5's conservative rejection zone for a 64-bit usize range. -/
def rangeZone (range : Nat) : Nat :=
  (range * 2 ^ (63 - range.log2) - 1) % 2 ^ 64

def rangeLoop (range zone : Nat) : Nat → Draw Nat
  | 0 => throw .exhausted
  | fuel + 1 => do
    let word ← nextWord
    let product := word.toNat * range
    if product % 2 ^ 64 ≤ zone then return product / 2 ^ 64
    else rangeLoop range zone fuel

/-- Rejected words are consumed, including for the singleton range. -/
def rangeSample (range : Nat) : Draw Nat := do
  if range = 0 || range > 2 ^ 64 then throw .panic
  if range = 2 ^ 64 then return (← nextWord).toNat
  let state ← get
  rangeLoop range (rangeZone range) (state.words.length + state.pending.length)

/-- Decrease the positive scale by one representable value until the upper bound is excluded. -/
def scaleLoop (lower upper : Word) : Nat → Word
  | 0 => 0
  | bits + 1 =>
    let scale := BitVec.ofNat 64 (bits + 1)
    let greatest : Word := 0x3feffffffffffffe
    if Float.leDouble upper (add (mul scale greatest) lower) then
      scaleLoop lower upper bits
    else scale

def uniform (lower upper : Word) : Draw Word := do
  if !Float.finiteDouble lower || !Float.finiteDouble upper ||
      !Float.ltDouble lower upper then throw .panic
  let scale := sub upper lower
  if !Float.finiteDouble scale then throw .panic
  let scale := scaleLoop lower upper scale.toNat
  return add (mul (fraction (← nextWord) 12) scale) lower

/-- Cast a nonnegative finite binary64 value to a saturating u64. -/
def truncWord (word : Word) : Nat :=
  match (Model.toRat? (Model.ofBits word : Model FloatFormat.binary64)) with
  | none => 0
  | some value => min (2 ^ 64 - 1) (value.num.toNat / value.den)

def bernoulli (numPert dim : Nat) : Draw Bool := do
  if dim = 0 then throw .invalid
  let prob := divide (numWord numPert) (numWord dim)
  if Float.leDouble one prob then return true
  let threshold := truncWord (mul prob (numWord (2 ^ 64)))
  return (← nextWord).toNat < threshold

def affine (lower upper unit : Word) : Word := add lower (mul unit (sub upper lower))

def genUniform (lower upper : Row) (count : Nat) : Draw Matrix := do
  let mut result := #[]
  for _ in [:count] do
    let mut row := #[]
    for j in [:lower.size] do
      let some high := upper[j]? | throw .panic
      row := row.push (← uniform lower[j]! high)
    result := result.push row
  return result

def shuffle (size : Nat) : Draw (Array Nat) := do
  let mut perm := (List.range size).toArray
  for k in [:size - 1] do
    let i := size - 1 - k
    let j ← rangeSample (i + 1)
    let old := perm[i]!
    perm := (perm.set! i perm[j]!).set! j old
  return perm

/-- LHD preserves column-major shuffle/draw order and each binary64 rounding point. -/
def genLhd (count dim : Nat) (bounds : Array (Word × Word)) : Draw Matrix := do
  let mut result := Array.replicate count (Array.replicate dim (0 : Word))
  for j in [:dim] do
    let some (lower, upper) := bounds[j]? | throw .panic
    let width := sub upper lower
    let perm ← shuffle count
    for i in [:count] do
      let bin := perm[i]!
      let start := divide (numWord bin) (numWord count)
      let stop := divide (numWord (bin + 1)) (numWord count)
      let offset ← standard
      let unit := add start (mul offset (sub stop start))
      result := result.set! i (result[i]!.set! j (add lower (mul unit width)))
  return result

/-- Upstream's binary search, including its last-coordinate fallback. -/
def searchCdf (cdf : Row) (word : Word) (lo hi : Nat) : Nat → Nat
  | 0 => lo
  | fuel + 1 =>
    if lo < hi then
      let mid := lo + (hi - lo) / 2
      if Float.leDouble word cdf[mid]! then searchCdf cdf word lo mid fuel
      else searchCdf cdf word (mid + 1) hi fuel
    else lo

def raaspDim (cdf : Row) (word : Word) : Nat :=
  searchCdf cdf word 0 (cdf.size - 1) cdf.size

/-- Sample RAASP using the already constructed CDF; repeated dimensions overwrite earlier draws. -/
def raaspCdf (center lower upper cdf : Row) (count numPert : Nat) : Draw Matrix := do
  let mut result := #[]
  for _ in [:count] do
    let mut row := center
    for _ in [:max 1 numPert] do
      let d := raaspDim cdf (← standard)
      let some low := lower[d]? | throw .panic
      let some high := upper[d]? | throw .panic
      if d ≥ row.size then throw .panic
      row := row.set! d (← uniform low high)
    result := result.push row
  return result

/-- Uniform coordinate probabilities, accumulated with binary64 addition as upstream does. -/
def uniformCdf (dim : Nat) : Row := Id.run do
  let prob := divide one (numWord dim)
  let mut cumulative : Word := 0
  let mut result := #[]
  for _ in [:dim] do
    cumulative := add cumulative prob
    result := result.push cumulative
  return result

def genRaasp (center lower upper : Row) (count numPert : Nat) : Draw Matrix :=
  raaspCdf center lower upper (uniformCdf center.size) count numPert

/-- ndarray 0.16.1 sums contiguous storage with eight independent accumulators.
The input is in physical memory order, which may differ from a reversed view. -/
def arraySum (values : Row) : Word := Id.run do
  let mut lanes := Array.replicate 8 (0 : Word)
  for block in [:values.size / 8] do
    for lane in [:8] do
      lanes := lanes.set! lane (add lanes[lane]! values[block * 8 + lane]!)
  let mut total : Word := 0
  for lane in [:4] do
    total := add total (add lanes[lane]! lanes[lane + 4]!)
  for i in [values.size / 8 * 8:values.size] do
    total := add total values[i]!
  return total

inductive Layout where
  | forward
  | reverse
  | strided
  deriving Repr, DecidableEq

/-- A logical one-dimensional view and the storage order needed by ndarray's sum. -/
structure Scales where
  values : Row
  layout : Layout := .forward
  deriving Repr, DecidableEq

def scaleSum (scales : Scales) : Word :=
  match scales.layout with
  | .forward => arraySum scales.values
  | .reverse => arraySum scales.values.reverse
  | .strided => add 0 (scales.values.foldl add 0)

def weightedCdf (dim : Nat) (scales : Option Scales) : Except Failure Row := do
  let some scales := scales | return uniformCdf dim
  let total := scaleSum scales
  if !Float.ltDouble 0 total then return uniformCdf dim
  let mut cumulative : Word := 0
  let mut result := #[]
  for j in [:dim] do
    let some scale := scales.values[j]? | throw .panic
    cumulative := add cumulative (divide scale total)
    result := result.push cumulative
  return result

def genWeighted (center lower upper : Row) (scales : Option Scales)
    (count numPert : Nat) : Draw Matrix := do
  match weightedCdf center.size scales with
  | .error err => throw err
  | .ok cdf => raaspCdf center lower upper cdf count numPert

/-- Fast RAASP draws all binomial trials, then all indices, then all coordinates.
Unlike the Sobol mask, even probability one consumes one standard draw per trial. -/
def genFast (center lower upper : Row) (count numPert : Nat) : Draw Matrix := do
  if center.size = 0 || count = 0 then
    return Array.replicate count (Array.replicate center.size 0)
  let ratio := divide (numWord numPert) (numWord center.size)
  let prob := if Float.ltDouble ratio one then ratio else one
  let mut result := #[]
  for _ in [:count] do
    let mut hits := 0
    for _ in [:center.size] do
      if Float.ltDouble (← standard) prob then hits := hits + 1
    let selected := max 1 hits
    let mut dims := (List.range center.size).toArray
    for t in [:selected] do
      let j := t + (← rangeSample (center.size - t))
      let old := dims[t]!
      dims := (dims.set! t dims[j]!).set! j old
    let mut row := center
    for t in [:selected] do
      let d := dims[t]!
      let unit ← standard
      let some low := lower[d]? | throw .panic
      let some high := upper[d]? | throw .panic
      row := row.set! d (add low (mul (sub high low) unit))
    result := result.push row
  return result

/-- Apply the upstream binomial mask and partial Fisher-Yates permutation to one Sobol row. -/
def maskRow (center lower upper sample : Row) (numPert : Nat) : Draw Row := do
  if center.size = 0 then throw .invalid
  let mut count := 0
  for _ in [:center.size] do
    if ← bernoulli numPert center.size then count := count + 1
  let selected := max 1 count
  let mut dims := (List.range center.size).toArray
  let mut row := center
  for t in [:selected] do
    let j := t + (← rangeSample (center.size - t))
    let old := dims[t]!
    dims := (dims.set! t dims[j]!).set! j old
    let d := dims[t]!
    let some low := lower[d]? | throw .panic
    let some high := upper[d]? | throw .panic
    let some unit := sample[d]? | throw .panic
    row := row.set! d (affine low high unit)
  return row

/-- At the trace boundary, exhausted input never manufactures a zero word. -/
theorem emptyTrace (used : Nat) :
    run nextWord { words := [], consumed := used } =
      (.error .exhausted, { words := [], consumed := used }) := rfl

/-- A singleton range still consumes a word upstream. -/
theorem singletonDraw :
    run (rangeSample 1) { words := [0], consumed := 0 } =
      (.ok 0, { words := [], consumed := 1 }) := by decide

/-- A rejected high word is skipped; the following accepted word determines the result. -/
theorem rejectionDraw :
    run (rangeSample 3) { words := [0xffffffffffffffff, 0], consumed := 0 } =
      (.ok 0, { words := [], consumed := 2 }) := by decide

set_option maxRecDepth 16384 in
/-- Row-major LHD output captured from the Rust implementation, including the residual RNG word. -/
theorem lhdFixture :
    let result := run (genLhd 4 2 #[(0, one), (0, one)]) (seeded 42 16)
    result.1 = .ok #[
      #[4591976496691633548, 4602373228015199930],
      #[4605007952137404557, 4603818566212700789],
      #[4600044019043131640, 4591701732378506428],
      #[4604339351361603652, 4605247852531002938]] ∧
    result.2.consumed = 14 ∧ result.2.words.head? = some 9648369018563374588 := by
  decide +kernel

set_option maxRecDepth 16384 in
theorem uniformFixture :
    let result := run (genUniform #[0, 0] #[one, one] 2) (seeded 42 8)
    result.1 = .ok #[#[4602918027047224548, 4603063653651445162],
      #[4603907987511953958, 4600983695946374540]] ∧
    result.2.words.head? = some 633513173585076202 := by decide +kernel

/-- Zero sample count remains zero, regardless of the nonzero dimension. -/
theorem lhdEmpty :
    run (genLhd 0 3 #[(0, one), (0, one), (0, one)]) { words := [] } =
      (.ok #[], { words := [] }) := by decide +kernel

/-- A one-point, one-dimensional LHD uses one standard f64 draw and no shuffle. -/
theorem lhdSingle :
    run (genLhd 1 1 #[(0, one)]) { words := [0x8000000000000800] } =
      (.ok #[#[0x3fe0000000000001]], { words := [], consumed := 1 }) := by decide +kernel

/-- At probability one fast RAASP still consumes trial, index, and coordinate words. -/
theorem fastDraw :
    run (genFast #[0] #[0] #[one] 1 20) { words := [0, 0, 0x8000000000000800] } =
      (.ok #[#[0x3fe0000000000001]], { words := [], consumed := 3 }) := by decide +kernel

/-- The Sobol mask at probability one skips the Bernoulli draw but consumes its index draw. -/
theorem maskDraw :
    run (maskRow #[0] #[0] #[one] #[0x3fe0000000000000] 20) { words := [0] } =
      (.ok #[0x3fe0000000000000], { words := [], consumed := 1 }) := by decide +kernel

/-- ndarray's reduction order is observable: a sequential sum of this input gives zero. -/
theorem sumOrder :
    let values : Row := #[0x4340000000000000, 0x3ff0000000000000, 0, 0,
      0xc340000000000000, 0, 0, 0]
    arraySum values = one ∧ values.foldl add 0 = 0 := by decide +kernel

set_option maxRecDepth 16384 in
/-- Independent Rust RAASP trace, with weighted CDF and residual RNG state. -/
theorem raaspFixture :
    let result := run (genWeighted #[0x3fe0000000000000, 0x3fe0000000000000]
      #[0x3fb999999999999a, 0x3fc999999999999a]
      #[0x3feccccccccccccd, 0x3fe999999999999a]
      (some { values := #[one, 0x4000000000000000] }) 2 1) (seeded 42 8)
    result.1 = .ok #[#[0x3fe0000000000000, 4602909719859925862],
      #[0x3fe0000000000000, 4601661745236883489]] ∧
    result.2.words.head? = some 633513173585076202 := by decide +kernel

#print axioms emptyTrace
#print axioms singletonDraw
#print axioms rejectionDraw
#print axioms lhdFixture
#print axioms uniformFixture
#print axioms lhdEmpty
#print axioms lhdSingle
#print axioms fastDraw
#print axioms maskDraw
#print axioms sumOrder
#print axioms raaspFixture

end BauxResearch.Semantics.Sampling
