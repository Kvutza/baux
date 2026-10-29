import BauxResearch.Semantics.Acq

/-!
# ENN 0.4.1 Pareto acquisition

This module executes the non-dominated sorting and seeded overflow selection
used by Pareto acquisition. Objectives are maximized exactly as in upstream.
-/

namespace BauxResearch.Semantics.Pareto

open Sampling Bounds

abbrev Matrix := Batch

def rectangular (values : Matrix) : Bool :=
  match values with
  | [] => true
  | row :: rows => rows.all (fun next => next.length == row.length)

def sameShape (left right : Matrix) : Bool :=
  rectangular left && rectangular right && left.length == right.length &&
    left.head?.map List.length == right.head?.map List.length

def dominates (left right : List Word) : Bool :=
  left.length == right.length &&
    (List.range left.length).all (fun index =>
      !Float.ltDouble left[index]! right[index]!) &&
    (List.range left.length).any (fun index =>
      Float.ltDouble right[index]! left[index]!)

def dominated (values : Matrix) (index : Nat) : List Nat :=
  (List.range values.length).filter fun other =>
    index != other && dominates values[index]! values[other]!

def domCount (values : Matrix) (index : Nat) : Nat :=
  ((List.range values.length).filter fun other =>
    index != other && dominates values[other]! values[index]!).length

def reduceOne (lists : List (List Nat)) (index : Nat)
    (state : Array Nat × List Nat) : Array Nat × List Nat :=
  lists[index]!.foldl (fun current target =>
    let count := current.1[target]! - 1
    let counts := current.1.set! target count
    if count = 0 then (counts, current.2 ++ [target])
    else (counts, current.2)) state

def peelLoop (lists : List (List Nat)) : Nat → List Nat →
    Array Nat → List (List Nat) → List (List Nat)
  | 0, _, _, fronts => fronts
  | fuel + 1, [], _, fronts => fronts
  | fuel + 1, current, counts, fronts =>
      let next := current.foldl (fun state index => reduceOne lists index state)
        (counts, [])
      peelLoop lists fuel next.2 next.1 (fronts ++ [current])

/-- General source-order non-dominated sorting for interleaved multi-objective
mean and uncertainty columns. -/
def nonDomin (values : Matrix) : List (List Nat) :=
  let lists := (List.range values.length).map (dominated values)
  let counts := (List.range values.length).map (domCount values) |>.toArray
  let first := (List.range values.length).filter fun index => counts[index]! == 0
  peelLoop lists values.length first counts []

def interleaveRow : List Word → List Word → List Word
  | [], [] => []
  | mean :: means, std :: stds => mean :: std :: interleaveRow means stds
  | _, _ => []

def interleave (mu se : Matrix) : Matrix :=
  mu.zipWith interleaveRow se

def shuffled (values : List Nat) : Draw (List Nat) := do
  let order ← shuffle values.length
  return order.toList.map (fun index => values[index]!)

def takeFronts : Nat → List (List Nat) → List Nat → Draw (List Nat)
  | _, [], selected => pure selected
  | arms, front :: fronts, selected => do
      if arms ≤ selected.length then return selected
      let remaining := arms - selected.length
      if front.length ≤ remaining then
        takeFronts arms fronts (selected ++ front)
      else return selected ++ (← shuffled front).take remaining

def scoreBefore (scores : List Word) (left right : Nat) : Bool :=
  decide (Fit.totalKey scores[right]! < Fit.totalKey scores[left]!)

def scoreInsert (scores : List Word) (index : Nat) : List Nat → List Nat
  | [] => [index]
  | head :: tail =>
      if scoreBefore scores index head then index :: head :: tail
      else head :: scoreInsert scores index tail

def scoreSort (scores : List Word) (indices : List Nat) : List Nat :=
  indices.foldl (fun sorted index => scoreInsert scores index sorted) []

def finiteAt (a b : List Word) (indices : List Nat) : Bool :=
  indices.all fun index =>
    Float.finiteDouble a[index]! && Float.finiteDouble b[index]!

structure Sky where
  best : Word
  lastA : Word
  lastB : Word
  front : List Nat

def front2 (a b : List Word) (indices : List Nat) : List Nat :=
  let byB := indices.foldl (fun sorted index =>
    let rec insertB : List Nat → List Nat
      | [] => [index]
      | head :: tail =>
          let ai := Fit.totalKey a[index]!
          let ah := Fit.totalKey a[head]!
          let bi := Fit.totalKey b[index]!
          let bh := Fit.totalKey b[head]!
          if ah < ai || (ah = ai && bh < bi) then index :: head :: tail
          else head :: insertB tail
    insertB sorted) []
  let step := byB.foldl (fun (state : Sky) index =>
    let av := a[index]!
    let bv := b[index]!
    if Float.ltDouble state.best bv then
      { best := bv, lastA := av, lastB := bv, front := state.front ++ [index] }
    else if Float.eqDouble bv state.best && Float.eqDouble av state.lastA &&
        Float.eqDouble bv state.lastB then
      { state with front := state.front ++ [index] }
    else state)
    (Sky.mk negInf 0x7ff8000000000000 0x7ff8000000000000 [])
  step.front

def removeMany (values removed : List Nat) : List Nat :=
  values.filter fun value => !removed.contains value

def lcgStep (state : Word) : Word :=
  state * 6364136223846793005 + 1442695040888963407

def chooseLoop : Nat → Word → List Nat → List Nat → List Nat
  | 0, _, _, selected => selected
  | count + 1, state, pool, selected =>
      let next := lcgStep state
      let index := next.toNat % pool.length
      chooseLoop count next (pool.eraseIdx index) (selected ++ [pool[index]!])

def deterministic (indices : List Nat) (count : Nat) (seed : Word) : List Nat :=
  if indices.length ≤ count then indices
  else chooseLoop count seed indices []

def armsLoop (mu se : List Word) (arms : Nat) (seed : Word) :
    Nat → List Nat → List Nat → Except Acq.AcqError (List Nat)
  | 0, _, selected => .ok (scoreSort mu selected)
  | fuel + 1, [], selected => .ok (scoreSort mu selected)
  | fuel + 1, remaining, selected =>
      if !finiteAt mu se remaining then .error .shape
      else
        let front := front2 mu se remaining
        if front.isEmpty then .ok (scoreSort mu selected)
        else
          let sorted := scoreSort mu front
          if selected.length + sorted.length ≤ arms then
            armsLoop mu se arms seed fuel (removeMany remaining sorted)
              (selected ++ sorted)
          else .ok (scoreSort mu
            (selected ++ deterministic sorted (arms - selected.length) seed))

/-- Single-objective Pareto path. One caller RNG word seeds every deterministic
overflow choice, matching `arms_from_pareto_fronts_2d`. -/
def selectOne (mu se : List Word) (arms : Nat) :
    Draw (Except Acq.AcqError (List Nat)) := do
  if mu.length != se.length then return .error .shape
  if mu.isEmpty then return .error .empty
  if arms = 0 then return .ok []
  let seed ← nextWord
  return armsLoop mu se (min arms mu.length) seed mu.length
    (List.range mu.length) []

def select (mu se : Matrix) (arms : Nat) :
    Draw (Except Acq.AcqError (List Nat)) := do
  if mu.isEmpty then return .error .empty
  if arms = 0 then return .ok []
  if !sameShape mu se then return .error .shape
  let objectives := mu[0]!.length
  if objectives = 1 then
    selectOne (mu.map fun row => row[0]!) (se.map fun row => row[0]!) arms
  else return .ok (← takeFronts arms (nonDomin (interleave mu se)) [])

theorem dominatesEx :
    dominates [Bounds.one] [0] = true := by
  decide +kernel

theorem zeroArms (mu se : Matrix) (h : mu.isEmpty = false) :
    select mu se 0 = pure (.ok []) := by
  simp [select, h]

#print axioms dominatesEx
#print axioms zeroArms

end BauxResearch.Semantics.Pareto
