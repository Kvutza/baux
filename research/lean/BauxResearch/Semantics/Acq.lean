import BauxResearch.Semantics.Fit

/-!
# ENN 0.4.1 scalar acquisition

This module executes Random, UCB, Thompson, and random-tie argmax selection at
the binary64 and rand 0.8.5 layers. Pareto-front selection is a separate
multi-objective construction.
-/

namespace BauxResearch.Semantics.Acq

open FloatLib.Floats.Formats.BinaryInterchange
open Sampling Bounds

abbrev LnOp := Word → Word
abbrev SqrtOp := Word → Word
abbrev CosOp := Word → Word

inductive AcqError where
  | empty
  | shape
  deriving Repr, DecidableEq

def two : Word := 0x4000000000000000
def negTwo : Word := 0xc000000000000000
def pi : Word := 0x400921fb54442d18

def clipMin : Word :=
  (Model.roundRat FloatFormat.binary64 false 1 1000000000000).bits

def clipMax : Word := sub Bounds.one clipMin

def clampUnit (value : Word) : Word :=
  if Float.ltDouble value clipMin then clipMin
  else if Float.ltDouble clipMax value then clipMax
  else value

def scoreUcb (mu sigma beta : Word) : Word :=
  add mu (mul sigma beta)

def scoreList (mu sigma : List Word) (beta : Word) : Except AcqError (List Word) :=
  if mu.length != sigma.length then .error .shape
  else .ok (mu.zipWith (fun mean std => scoreUcb mean std beta) sigma)

def rankBefore (scores : List Word) (left right : Nat) : Bool :=
  decide (Fit.totalKey scores[right]! < Fit.totalKey scores[left]!)

def rankInsert (scores : List Word) (index : Nat) : List Nat → List Nat
  | [] => [index]
  | head :: tail =>
      if rankBefore scores index head then index :: head :: tail
      else head :: rankInsert scores index tail

def rankScores (scores : List Word) (indices : List Nat) : List Nat :=
  indices.foldl (fun sorted index => rankInsert scores index sorted) []

def shuffledRank (scores : List Word) : Draw (List Nat) := do
  let indices ← shuffle scores.length
  return rankScores scores indices.toList

def randomSelect (candidates arms : Nat) : Draw (Except AcqError (List Nat)) := do
  if candidates = 0 then return .error .empty
  let indices ← shuffle candidates
  return .ok (indices.toList.take arms)

def ucbSelect (mu sigma : List Word) (beta : Word) (arms : Nat) :
    Draw (Except AcqError (List Nat)) := do
  if mu.isEmpty then return .error .empty
  if arms = 0 then return .ok []
  match scoreList mu sigma beta with
  | .error cause => return .error cause
  | .ok scores => return .ok ((← shuffledRank scores).take arms)

def normalOne (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (mean std : Word) : Draw Word := do
  let u1 := clampUnit (← standard)
  let u2 ← standard
  let radius := sqrtOp (mul negTwo (lnOp u1))
  let angle := mul (mul two pi) u2
  let z := mul radius (cosOp angle)
  return add mean (mul std z)

def normalList (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp) :
    List Word → List Word → Draw (Except AcqError (List Word))
  | [], [] => pure (.ok [])
  | mean :: means, std :: stds => do
      let value ← normalOne lnOp sqrtOp cosOp mean std
      match ← normalList lnOp sqrtOp cosOp means stds with
      | .ok rest => return .ok (value :: rest)
      | .error cause => return .error cause
  | _, _ => pure (.error .shape)

def thompson (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (mu sigma : List Word) (arms : Nat) : Draw (Except AcqError (List Nat)) := do
  if mu.isEmpty then return .error .empty
  if arms = 0 then return .ok []
  match ← normalList lnOp sqrtOp cosOp mu sigma with
  | .error cause => return .error cause
  | .ok samples => return .ok ((← shuffledRank samples).take arms)

/-- Rust `f64::max`, which suppresses a NaN on the left or right by returning
the other operand. Equal signed zeros return the right operand. -/
def maxWord (left right : Word) : Word :=
  if Float.nanDouble left then right
  else if Float.nanDouble right then left
  else if Float.ltDouble left right then right
  else if Float.ltDouble right left then left
  else right

def maxValue (values : List Word) : Word := values.foldl maxWord negInf

def ties (values : List Word) : List Nat :=
  let maximum := maxValue values
  (List.range values.length).filter fun index =>
    Float.leDouble maximum values[index]!

/-- `util::argmax_random_tie`, including its no-draw unique-maximum branch. -/
def argmaxTie (values : List Word) : Draw Nat := do
  if values.isEmpty then return 0
  let choices := ties values
  if choices.length = 1 then return choices[0]!
  let selected ← rangeSample choices.length
  return choices[selected]!

theorem ucbBits :
    scoreUcb Bounds.one Bounds.one Bounds.one =
      0x4000000000000000 := by
  decide +kernel

theorem randomSingle (state : RngState) :
    run (randomSelect 1 1) state = (.ok (.ok [0]), state) := by
  simp [randomSelect, shuffle, run]
  rfl

theorem uniqueDraw (state : RngState) :
    run (argmaxTie [0, Bounds.one, 0]) state = (.ok 1, state) := by
  rfl

#print axioms ucbBits
#print axioms randomSingle
#print axioms uniqueDraw

end BauxResearch.Semantics.Acq
