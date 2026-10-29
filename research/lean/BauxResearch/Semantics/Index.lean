import BauxResearch.Semantics.Sampling

/-!
# rand 0.8.5 index sampling

This module models the complete algorithm dispatch and random-consumption order
of `rand::seq::index::sample` on the 64-bit ENN target. Ranges over `u32` use
`RngCore::next_u32`; ranges over `usize` above `u32::MAX` use `next_u64`.
-/

namespace BauxResearch.Semantics.Index

open FloatLib.Floats.Formats.BinaryInterchange
open Sampling

def max32 : Nat := 2 ^ 32 - 1

def zone32 (range : Nat) : Nat :=
  (range * 2 ^ (31 - range.log2) - 1) % 2 ^ 32

def loop32 (low range zone : Nat) : Nat → Draw Nat
  | 0 => throw .exhausted
  | fuel + 1 => do
      let value := (← nextHalf).toNat
      let product := value * range
      if product % 2 ^ 32 ≤ zone then return low + product / 2 ^ 32
      else loop32 low range zone fuel

/-- `UniformInt<u32>::sample_single_inclusive`. -/
def range32 (low high : Nat) : Draw Nat := do
  if high > max32 || high < low then throw .panic
  let range := high - low + 1
  if range = 2 ^ 32 then return low + (← nextHalf).toNat
  let state ← get
  loop32 low range (zone32 range) (2 * state.words.length + state.pending.length)

def insertAt (values : List Nat) (index value : Nat) : List Nat :=
  values.take index ++ value :: values.drop index

def swapAt (values : Array Nat) (left right : Nat) : Array Nat :=
  let a := values[left]!
  let b := values[right]!
  (values.set! left b).set! right a

def floydLoop (length amount : Nat) : Nat → List Nat → Nat → Draw (List Nat)
  | j, values, 0 => if j < length then throw .exhausted else pure values
  | j, values, fuel + 1 => do
      if j < length then
        let target ← range32 0 j
        let next := if amount < 50 then
          match values.findIdx? (fun value => value == target) with
          | some pos => insertAt values pos j
          | none => values ++ [target]
        else if values.contains target then values ++ [j]
        else values ++ [target]
        floydLoop length amount (j + 1) next fuel
      else pure values

def shuffleLoop : Nat → Array Nat → Draw (Array Nat)
  | 0, values => pure values
  | i + 1, values => do
      let target ← range32 0 (i + 1)
      shuffleLoop i (swapAt values (i + 1) target)

def sampleFloyd (length amount : Nat) : Draw (List Nat) := do
  let values ← floydLoop length amount (length - amount) [] amount
  if amount < 50 then return values
  else return (← shuffleLoop (amount - 1) values.toArray).toList

def inplaceLoop (length amount : Nat) : Nat → Array Nat → Nat → Draw (Array Nat)
  | i, values, 0 => if i < amount then throw .exhausted else pure values
  | i, values, fuel + 1 => do
      if i < amount then
        let target ← range32 i (length - 1)
        inplaceLoop length amount (i + 1) (swapAt values i target) fuel
      else pure values

def sampleInplace (length amount : Nat) : Draw (List Nat) := do
  let values ← inplaceLoop length amount 0 (List.range length).toArray amount
  return (values.extract 0 amount).toList

def reject32 (length amount : Nat) : Nat → List Nat → Nat → Draw (List Nat)
  | count, values, 0 => if count < amount then throw .exhausted else pure values
  | count, values, fuel + 1 => do
      if count < amount then
        let target ← range32 0 (length - 1)
        if values.contains target then reject32 length amount count values fuel
        else reject32 length amount (count + 1) (values ++ [target]) fuel
      else pure values

def reject64 (length amount : Nat) : Nat → List Nat → Nat → Draw (List Nat)
  | count, values, 0 => if count < amount then throw .exhausted else pure values
  | count, values, fuel + 1 => do
      if count < amount then
        let target ← rangeSample length
        if values.contains target then reject64 length amount count values fuel
        else reject64 length amount (count + 1) (values ++ [target]) fuel
      else pure values

def singleNat (value : Nat) : BitVec 32 :=
  (Model.roundRat FloatFormat.binary32 false value 1).bits

def singleRat (num den : Nat) : BitVec 32 :=
  (Model.roundRat FloatFormat.binary32 false num den).bits

def singleAdd (left right : BitVec 32) : BitVec 32 :=
  (Float.addSingle (Float.wordSingle left) (Float.wordSingle right)).bits

def singleMul (left right : BitVec 32) : BitVec 32 :=
  (Float.mulSingle (Float.wordSingle left) (Float.wordSingle right)).bits

def shortInplace (length amount : Nat) : Bool :=
  let column := if length < 500000 then 0 else 1
  let c0 := if column = 0 then singleRat 8 5 else singleRat 8 45
  let c1 := if column = 0 then singleNat 10 else singleRat 70 9
  let amountFloat := singleNat amount
  let threshold := singleMul (singleAdd c1 (singleMul c0 amountFloat)) amountFloat
  amount > 11 && Float.ltSingle (singleNat length) threshold

def largeInplace (length amount : Nat) : Bool :=
  let coefficient := if length < 500000 then singleNat 270 else singleRat 110 3
  Float.ltSingle (singleNat length) (singleMul coefficient (singleNat amount))

/-- Complete 64-bit dispatch of rand 0.8.5 `seq::index::sample`. -/
def sample (length amount : Nat) : Draw (List Nat) := do
  if amount > length then throw .panic
  let state ← get
  if length > max32 then
    return (← reject64 length amount 0 [] (state.words.length + state.pending.length))
  if amount < 163 then
    if shortInplace length amount then sampleInplace length amount
    else sampleFloyd length amount
  else if largeInplace length amount then sampleInplace length amount
  else reject32 length amount 0 [] (2 * state.words.length + state.pending.length)

theorem mixedDraw :
    run (do
      let a ← nextHalf
      let b ← nextWord
      pure (a, b)) { words := [0x0000000200000001, 0x0000000400000003] } =
      (.ok (1, 0x0000000300000002),
        { words := [], pending := [4], consumed := 2 }) := by
  decide +kernel

theorem floydEmpty :
    run (sample 8 0) { words := [] } = (.ok [], { words := [] }) := by
  decide +kernel

theorem floydFull :
    (run (sample 1 1) { words := [0] }).1 = .ok [0] := by
  decide +kernel

#print axioms mixedDraw
#print axioms floydEmpty
#print axioms floydFull

end BauxResearch.Semantics.Index
