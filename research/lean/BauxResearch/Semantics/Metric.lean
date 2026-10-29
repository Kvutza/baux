import Mathlib.Data.BitVec
import BauxResearch.Semantics.Sampling
import BauxResearch.Semantics.Compose
import BauxResearch.Semantics.Fit
import BauxResearch.Semantics.Neighbor
import BauxResearch.Semantics.Bounds
import BauxResearch.Semantics.Pcg

/-!
# ENN 0.4.1 automatic metric learning gate

This module formalizes the exact ENN 0.4.1 automatic metric learning state
transitions, relational Algorithm R reservoir sampling quantified over exact
bounded-draw derivations, dependence weights via Sobol indices, leave-one-out
log-likelihood selection, refit pacing, scale update computation, and model
error order.
-/

namespace BauxResearch.Semantics.Metric

open Sampling Bounds Fitter Fit

def driftTol : Word := 0x3fe62e42fefa39ef
def driftFloor : Word := 0x40226bb1bbb55516
def minGain : Word := 0x0000000000000000
def resCap : Nat := 1000
def autoK : Nat := 10
def refitRate : Word := 0x3ff8000000000000
def rescaleTol : Word := 0x3f847ae147ae147b
def depFloor : Word := 0x3eb0c6f7a0b5ed8d
def minVar : Word := 0x3af357c299a88ea7
def minRows : Nat := 100
def depZ : Word := 0x4008000000000000
def looEps : Word := 0x3e112e0be826d695
def twoPi : Word := 0x401921fb54442d18
def halfWord : Word := 0x3fe0000000000000
def negInf : Word := 0xfff0000000000000
def posInf : Word := 0x7ff0000000000000

def scaleGrid : List Word := [
  0x3f847ae147ae147b, 0x3fa030dc4ea03a72, 0x3fb999999999999a,
  0x3fd43d136248490f, 0x3ff0000000000000, 0x40094c583ada5b53,
  0x4024000000000000, 0x403f9f6e4990f227, 0x4059000000000000,
  0x4073c3a4edfa9759, 0x408f400000000000, 0x40a8b48e29793d2f,
  0x40c3880000000000
]

def noiseGrid : List Word := [
  0x3f50624dd2f1a9fc, 0x3f689374bc6a7efa, 0x3f847ae147ae147b,
  0x3f9eb851eb851eb8, 0x3fb999999999999a, 0x3fd3333333333333,
  0x3ff0000000000000
]

structure Counts where
  numRefits : Nat
  numRescales : Nat
  numRebuilds : Nat
  deriving Repr, DecidableEq

structure Snapshot where
  numSeen : Nat
  counters : Counts
  heldoutGain : Option Word
  usesLearned : Bool
  weights : List Word
  built : List Word
  deriving Repr, DecidableEq

structure Update where
  xScale : List Word
  rebuild : Bool
  deriving Repr, DecidableEq

inductive Err where
  | shape
  | nonfinite
  | nonpositive
  | tied
  deriving Repr, DecidableEq


structure Reservoir where
  capacity : Nat
  dim : Nat
  outputs : Nat
  seed : Word
  rng : Pcg.State
  numSeen : Nat
  xs : List (List Word)
  ys : List (List Word)
  deriving Repr, DecidableEq

/-- Exact Algorithm R reservoir row addition relation quantified over an exact
bounded-draw derivation. -/
inductive PushRel : Reservoir → List Word → List Word → Reservoir → Prop where
  | fill (r : Reservoir) (x y : List Word) :
      r.xs.length < r.capacity →
      PushRel r x y ⟨r.capacity, r.dim, r.outputs, r.seed, r.rng, r.numSeen + 1,
        r.xs ++ [x], r.ys ++ [y]⟩
  | replace (r : Reservoir) (x y : List Word) (slot : BitVec 64) (nextRng : Pcg.State) :
      r.xs.length ≥ r.capacity →
      Pcg.IntHigh (BitVec.ofNat 64 (r.numSeen + 1)) r.rng slot nextRng →
      slot.toNat < r.capacity →
      PushRel r x y ⟨r.capacity, r.dim, r.outputs, r.seed, nextRng, r.numSeen + 1,
        r.xs.set slot.toNat x, r.ys.set slot.toNat y⟩
  | discard (r : Reservoir) (x y : List Word) (slot : BitVec 64) (nextRng : Pcg.State) :
      r.xs.length ≥ r.capacity →
      Pcg.IntHigh (BitVec.ofNat 64 (r.numSeen + 1)) r.rng slot nextRng →
      slot.toNat ≥ r.capacity →
      PushRel r x y ⟨r.capacity, r.dim, r.outputs, r.seed, nextRng, r.numSeen + 1,
        r.xs, r.ys⟩

/-- Exact batch reservoir row addition over multiple rows relation. -/
inductive RowsRel : Reservoir → List (List Word) → List (List Word) → Reservoir → Prop where
  | nil (r : Reservoir) :
      RowsRel r [] [] r
  | cons (r r1 r2 : Reservoir) (x : List Word) (xs : List (List Word))
      (y : List Word) (ys : List (List Word)) :
      PushRel r x y r1 →
      RowsRel r1 xs ys r2 →
      RowsRel r (x :: xs) (y :: ys) r2

structure State where
  rows : Reservoir
  tied : List (List Nat)
  weights : List Word
  built : List Word
  rebuildDrift : Word
  refitGrowth : Word
  nextRefit : Nat
  heldoutGain : Option Word
  counters : Counts
  deriving Repr, DecidableEq

def distinct : List Nat → Bool
  | [] => true
  | x :: xs => !xs.contains x && distinct xs

def tiedValid (tied : List (List Nat)) (dim : Nat) : Bool :=
  tied.all (fun group => !group.isEmpty && group.all (fun j => j < dim)) &&
    distinct (tied.flatten)

def fewRows (n : Nat) : Bool :=
  n < minRows

def resEmpty (dim outputs : Nat) (seed : Word) : Reservoir :=
  ⟨resCap, dim, outputs, seed, Pcg.fromSeed seed, 0, [], []⟩

def newMetric (dim outputs : Nat) (tied : List (List Nat)) (seed : Word) :
    Except Err State :=
  if dim = 0 || outputs = 0 then .error .shape
  else if !tiedValid tied dim then .error .tied
  else
    .ok ⟨resEmpty dim outputs seed, tied, List.replicate dim Bounds.one,
      List.replicate dim Bounds.one, driftTol, refitRate, minRows, none,
      ⟨0, 0, 0⟩⟩

/-! A finite evaluator is used only when its result is available. Rejection
sampling has no fixed maximum draw count, so exhaustion is not an ENN result. -/
def pushFuel (fuel : Nat) (r : Reservoir) (x y : List Word) : Option Reservoir :=
  if r.xs.length < r.capacity then
    some ⟨r.capacity, r.dim, r.outputs, r.seed, r.rng, r.numSeen + 1,
      r.xs ++ [x], r.ys ++ [y]⟩
  else
    match Pcg.evalHigh fuel (BitVec.ofNat 64 (r.numSeen + 1)) r.rng with
    | none => none
    | some (slot, nextRng) =>
      if slot.toNat < r.capacity then
        some ⟨r.capacity, r.dim, r.outputs, r.seed, nextRng, r.numSeen + 1,
          r.xs.set slot.toNat x, r.ys.set slot.toNat y⟩
      else
        some ⟨r.capacity, r.dim, r.outputs, r.seed, nextRng, r.numSeen + 1,
          r.xs, r.ys⟩

theorem pushSound (fuel : Nat) (r next : Reservoir) (x y : List Word)
    (h : pushFuel fuel r x y = some next) : PushRel r x y next := by
  by_cases hfill : r.xs.length < r.capacity
  · simp only [pushFuel, if_pos hfill] at h
    cases h
    exact PushRel.fill r x y hfill
  · have hfull : r.xs.length ≥ r.capacity := by omega
    simp only [pushFuel, if_neg hfill] at h
    cases hdraw : Pcg.evalHigh fuel (BitVec.ofNat 64 (r.numSeen + 1)) r.rng with
    | none => simp [hdraw] at h
    | some pair =>
      rcases pair with ⟨slot, nextRng⟩
      simp only [hdraw] at h
      by_cases hslot : slot.toNat < r.capacity
      · simp only [if_pos hslot, Option.some.injEq] at h
        subst next
        exact PushRel.replace r x y slot nextRng hfull
          (Pcg.soundHigh fuel _ _ _ _ hdraw) hslot
      · have hout : slot.toNat ≥ r.capacity := by omega
        simp only [if_neg hslot, Option.some.injEq] at h
        subst next
        exact PushRel.discard r x y slot nextRng hfull
          (Pcg.soundHigh fuel _ _ _ _ hdraw) hout

def rowsFuel (fuel : Nat) (r : Reservoir) (xs ys : List (List Word)) :
    Option Reservoir :=
  (List.range xs.length).foldlM (fun acc index =>
    pushFuel fuel acc (xs.getD index []) (ys.getD index [])) r

def sumWords (xs : List Word) : Word :=
  xs.foldl add 0

def maxWord (left right : Word) : Word :=
  if Float.nanDouble left then right
  else if Float.nanDouble right then left
  else if Float.ltDouble left right then right else left

def minWord (left right : Word) : Word :=
  if Float.nanDouble left then right
  else if Float.nanDouble right then left
  else if Float.ltDouble left right then left else right

def floorLog (ln : LogOp) (w : List Word) : List Word :=
  let logs := w.map ln
  let maxVal := logs.foldl maxWord negInf
  let floorBound := sub maxVal driftFloor
  logs.map (fun v => maxWord v floorBound)

def weightDrift (ln : LogOp) (weights built : List Word) : Word :=
  let a := floorLog ln weights
  let b := floorLog ln built
  let diffMax := (List.range (min a.length b.length)).foldl (fun best index =>
    let ai := a.getD index 0
    let bi := b.getD index 0
    maxWord best (Float.absDouble (Float.wordDouble (sub ai bi))).bits) 0
  mul halfWord diffMax

def usesLearned (gain : Word) : Bool :=
  Float.finiteDouble gain && Float.ltDouble minGain gain

def rankBefore (left right : Word × Nat) : Bool :=
  let lk := Fit.totalKey left.1
  let rk := Fit.totalKey right.1
  decide (lk < rk || (lk = rk && left.2 ≤ right.2))

def insertRank (entry : Word × Nat) : List (Word × Nat) → List (Word × Nat)
  | [] => [entry]
  | head :: tail =>
      if rankBefore entry head then entry :: head :: tail
      else head :: insertRank entry tail

def sortRanks (entries : List (Word × Nat)) : List (Word × Nat) :=
  entries.foldr insertRank []

def stableRanks (col : List Word) : List Nat :=
  let indexed := (List.range col.length).map fun index => (col.getD index 0, index)
  let sorted := sortRanks indexed
  let ranks := List.replicate col.length 0
  (List.range sorted.length).foldl (fun acc rank =>
    let entry := sorted.getD rank (0, 0)
    acc.set entry.2 rank) ranks

def betweenShare (count total : List Word) (ssTot : Word) (n b : Nat) : Word :=
  let ssBetween := (List.range (min count.length total.length)).foldl (fun acc index =>
    let c := count.getD index 0
    let t := total.getD index 0
    let cEff := maxWord Bounds.one c
    add acc (divide (mul t t) cEff)) 0
  let nSubB := if n > b then n - b else 1
  let msWithin := divide (sub ssTot ssBetween) (numWord nSubB)
  let bMinus1 := if b > 0 then b - 1 else 0
  let penalty := mul (numWord bMinus1) msWithin
  let numerator := maxWord 0 (sub ssBetween penalty)
  divide numerator ssTot

def nullSd (sqrtOp : SqrtOp) (n : Nat) (cells : Option Nat) : Word :=
  if n < 3 then posInf
  else
    let b := match cells with
      | some c => c
      | none => max 2 (Nat.sqrt n)
    let bSub1 := if b > 0 then b - 1 else 0
    let rad := numWord (2 * bSub1)
    divide (sqrtOp rad) (numWord n)

def wordsLt : List Word → List Word → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | x :: xs, y :: ys =>
      let xk := Fit.totalKey x
      let yk := Fit.totalKey y
      if xk < yk then true
      else if yk < xk then false
      else wordsLt xs ys

def rowBefore (left right : List Word × Nat) : Bool :=
  if wordsLt left.1 right.1 then true
  else if wordsLt right.1 left.1 then false
  else left.2 ≤ right.2

def insertRow (entry : List Word × Nat) :
    List (List Word × Nat) → List (List Word × Nat)
  | [] => [entry]
  | head :: tail =>
      if rowBefore entry head then entry :: head :: tail
      else head :: insertRow entry tail

def sortRows (entries : List (List Word × Nat)) :
    List (List Word × Nat) :=
  entries.foldr insertRow []

def cellsGroup (_dim : Nat) (group : List Nat) (x : List (List Word)) :
    List Nat × Nat :=
  let n := x.length
  if n = 0 then ([], 0)
  else
    let rows := (List.range n).map fun index =>
      let row := x.getD index []
      (group.map (fun j => row.getD j 0), index)
    let sorted := sortRows rows
    let step := sorted.foldl (fun (acc : List Nat × Option (List Word) × Nat) entry =>
      match acc.2.1 with
      | none => (acc.1.set entry.2 0, some entry.1, 0)
      | some prev =>
          if entry.1 == prev then
            (acc.1.set entry.2 acc.2.2, some entry.1, acc.2.2)
          else
            let nextCell := acc.2.2 + 1
            (acc.1.set entry.2 nextCell, some entry.1, nextCell)
    ) (List.replicate n 0, none, 0)
    (step.1, step.2.2 + 1)

def groupSobol (dim : Nat) (group : List Nat) (x : List (List Word))
    (y : List Word) : Word :=
  let (cells, b) := cellsGroup dim group x
  let n := x.length
  let mean := divide (sumWords y) (numWord (max 1 n))
  let yc := y.map (fun v => sub v mean)
  let ssTot := sumWords (yc.map (fun v => mul v v))
  if b < 2 || n < 2 * b || Float.leDouble ssTot 0 then 0
  else
    let initCount : List Word := List.replicate b 0
    let initTotal : List Word := List.replicate b 0
    let step := (List.range n).foldl (fun (acc : List Word × List Word) i =>
      let cell := cells.getD i 0
      if cell < b then
        (acc.1.set cell (add (acc.1.getD cell 0) Bounds.one),
         acc.2.set cell (add (acc.2.getD cell 0) (yc.getD i 0)))
      else acc
    ) (initCount, initTotal)
    betweenShare step.1 step.2 ssTot n b

def groupCells (dim : Nat) (group : List Nat) (x : List (List Word)) : Nat :=
  (cellsGroup dim group x).2

def sobolIndex (dim : Nat) (x : List (List Word)) (y : List Word)
    (numBins : Option Nat) : List Word :=
  let n := x.length
  let b := match numBins with
    | some c => c
    | none => max 2 (Nat.sqrt n)
  let mean := divide (sumWords y) (numWord (max 1 n))
  let yc := y.map (fun v => sub v mean)
  let ssTot := sumWords (yc.map (fun v => mul v v))
  if n < 2 * b || Float.leDouble ssTot 0 then List.replicate dim 0
  else
    (List.range dim).map fun j =>
      let col := (List.range n).map fun i =>
        (x.getD i []).getD j 0
      let ranks := stableRanks col
      let initCount : List Word := List.replicate b 0
      let initTotal : List Word := List.replicate b 0
      let step := (List.range n).foldl (fun (acc : List Word × List Word) i =>
        let rankI := ranks.getD i 0
        let cell := (rankI * b) / (max 1 n)
        if cell < b then
          (acc.1.set cell (add (acc.1.getD cell 0) Bounds.one),
           acc.2.set cell (add (acc.2.getD cell 0) (yc.getD i 0)))
        else acc
      ) (initCount, initTotal)
      betweenShare step.1 step.2 ssTot n b

def spreadCol (dim : Nat) (x : List (List Word)) : List Word :=
  let n := x.length
  if n = 0 then List.replicate dim Bounds.one
  else
    (List.range dim).map fun j =>
      let col := (List.range n).map fun i =>
        (x.getD i []).getD j 0
      let sum := sumWords col
      let sumsq := sumWords (col.map (fun v => mul v v))
      let mean := divide sum (numWord n)
      let variance := maxZero (sub (divide sumsq (numWord n)) (mul mean mean))
      if Float.ltDouble minVar variance then variance else Bounds.one

def passCheck (sqrtOp : SqrtOp) (s : Word) (n : Nat) (cells : Option Nat) : Word :=
  let thresh := mul depZ (nullSd sqrtOp n cells)
  if Float.ltDouble thresh s then s else 0

def unitIndices (sqrtOp : SqrtOp) (dim : Nat) (tied : List (List Nat))
    (x : List (List Word)) (y : List (List Word)) : List Word × List Nat :=
  let n := x.length
  let m := if y.isEmpty then 0 else (y.getD 0 []).length
  let meanSobol := (List.range dim).map fun j =>
    let sumCol := (List.range m).foldl (fun acc c =>
      let colY := (List.range n).map fun i =>
        (y.getD i []).getD c 0
      let s := sobolIndex dim x colY none
      add acc (s.getD j 0)) 0
    divide sumCol (numWord (max 1 m))
  let s := meanSobol.map (fun v => passCheck sqrtOp v n none)
  if tied.isEmpty then (s, List.range dim)
  else
    let inGroup := (List.range dim).map fun j =>
      tied.any (fun g => g.contains j)
    let free := (List.range dim).filter (fun j => !inGroup.getD j false)
    let unit := List.replicate dim 0
    let unitAfterFree := (List.range free.length).foldl (fun acc k =>
      acc.set (free.getD k 0) k) unit
    let jointStep := (List.range tied.length).foldl (fun (acc : List Nat × List Word) i =>
      let g := tied.getD i []
      let nextUnit := g.foldl (fun u j => u.set j (free.length + i)) acc.1
      let sumG := (List.range m).foldl (fun sg c =>
        let colY := (List.range n).map fun r =>
          (y.getD r []).getD c 0
        add sg (groupSobol dim g x colY)) 0
      let sgMean := divide sumG (numWord (max 1 m))
      let cells := groupCells dim g x
      let passed := passCheck sqrtOp sgMean n (some cells)
      (nextUnit, acc.2 ++ [passed])
    ) (unitAfterFree, [])
    let out := (free.map (fun j => s.getD j 0)) ++ jointStep.2
    (out, jointStep.1)

def depWeights (sqrtOp : SqrtOp) (dim : Nat) (tied : List (List Nat))
    (x : List (List Word)) (y : List (List Word)) : List Word :=
  let (sRaw, unit) := unitIndices sqrtOp dim tied x y
  let sMax := sRaw.foldl maxWord 0
  let s := if Float.ltDouble 0 sMax then
    sRaw.map (fun v => maxWord v (mul depFloor sMax))
  else List.replicate sRaw.length Bounds.one
  let sprRaw := spreadCol dim x
  let spr := (List.range dim).map fun j =>
    if tied.any (fun g => g.contains j) then Bounds.one else sprRaw.getD j 0
  let sSum := sumWords s
  let u := numWord s.length
  (List.range dim).map fun j =>
    let unitIdx := unit.getD j 0
    let sVal := if unitIdx < s.length then s.getD unitIdx 0 else Bounds.one
    divide (divide (mul u sVal) sSum) (spr.getD j Bounds.one)

def colLoglik (ln : LogOp) (z : List (List Word)) (ycol aScaled : List Word)
    (n dim k : Nat) : Word :=
  let rowA := (List.range n).map fun r =>
    let zr := z.getD r []
    (List.range dim).foldl (fun acc j =>
      let zj := zr.getD j 0
      let azj := mul (aScaled.getD j Bounds.one) zj
      add acc (mul azj zj)) 0
  let nbrStep := (List.range n).map fun i =>
    let rowI := rowA.getD i 0
    let zi := z.getD i []
    let dists := (List.range n).filter (fun r => r != i) |>.map fun r =>
      let zr := z.getD r []
      let dot := (List.range dim).foldl (fun acc j =>
        let zj := zr.getD j 0
        let azj := mul (aScaled.getD j Bounds.one) (zi.getD j 0)
        add acc (mul azj zj)) 0
      let d2 := maxWord 0 (sub (add rowI (rowA.getD r 0)) (mul (numWord 2) dot))
      (d2, r)
    let sorted := sortRanks dists
    sorted.take k
  scaleGrid.foldl (fun bestScale scale =>
    noiseGrid.foldl (fun bestNoise noise =>
      let totalLog := (List.range n).foldl (fun acc i =>
        let nbrs := nbrStep.getD i []
        let wStep := nbrs.foldl (fun wacc pair =>
          let d2 := pair.1
          let r := pair.2
          let denom := add looEps (add (maxWord 0 (mul d2 scale)) noise)
          let w := divide Bounds.one denom
          let yr := ycol.getD r 0
          (add wacc.1 w, add wacc.2 (mul w yr))) (0, 0)
        let wSum := wStep.1
        let mu := divide wStep.2 wSum
        let variance := add (divide Bounds.one wSum) noise
        let err := sub (ycol.getD i 0) mu
        let term := sub (mul (sub 0 halfWord) (ln (mul twoPi variance)))
          (divide (mul halfWord (mul err err)) variance)
        add acc term) 0
      let avg := divide totalLog (numWord (max 1 n))
      maxWord bestNoise avg) bestScale) negInf

def looLoglik (sqrtOp : SqrtOp) (ln : LogOp) (dim : Nat) (x : List (List Word))
    (y : List (List Word)) (a : List Word) (k : Nat) : Word :=
  let n := x.length
  let m := if y.isEmpty then 0 else (y.getD 0 []).length
  let kk := min k (if n > 0 then n - 1 else 0)
  if n <= 1 || m = 0 then 0
  else
    let spr := spreadCol dim x
    let means := (List.range dim).map fun j =>
      let sum := (List.range n).foldl (fun acc i =>
        add acc ((x.getD i []).getD j 0)) 0
      divide sum (numWord n)
    let z := (List.range n).map fun i =>
      (List.range dim).map fun j =>
        let xj := (x.getD i []).getD j 0
        let mj := means.getD j 0
        let sj := sqrtOp (spr.getD j Bounds.one)
        divide (sub xj mj) sj
    let aScaled := (List.range dim).map fun j =>
      mul (a.getD j Bounds.one) (spr.getD j Bounds.one)
    let totalCol := (List.range m).foldl (fun acc c =>
      let raw := (List.range n).map fun i =>
        (y.getD i []).getD c 0
      let ySum := sumWords raw
      let yMean := divide ySum (numWord n)
      let yVar := (spreadCol 1 (raw.map (fun v => [v]))).getD 0 Bounds.one
      let yStd := sqrtOp yVar
      let yz := raw.map (fun v => divide (sub v yMean) yStd)
      add acc (colLoglik ln z yz aScaled n dim kk)) 0
    divide totalCol (numWord m)

def autoWeights (sqrtOp : SqrtOp) (ln : LogOp) (dim : Nat)
    (tied : List (List Nat)) (x : List (List Word))
    (y : List (List Word)) : Except Err (List Word × Word) :=
  if !tiedValid tied dim then .error .tied
  else if fewRows x.length then
    .ok (List.replicate dim Bounds.one, negInf)
  else
    let w := depWeights sqrtOp dim tied x y
    let ones := List.replicate dim Bounds.one
    let logW := looLoglik sqrtOp ln dim x y w autoK
    let logOnes := looLoglik sqrtOp ln dim x y ones autoK
    let gain := sub logW logOnes
    .ok (w, gain)

def setWeights (sqrtOp : SqrtOp) (ln : LogOp) (state : State)
    (weights : List Word) : Except Err (State × Update) :=
  if weights.length != state.rows.dim then .error .shape
  else if !weights.all (fun w => Float.finiteDouble w && Float.ltDouble 0 w) then
    .error .nonpositive
  else
    let rebuild := Float.ltDouble state.rebuildDrift
      (weightDrift ln weights state.built)
    let xScale := weights.map (fun w => divide Bounds.one (sqrtOp w))
    let nextBuilt := if rebuild then weights else state.built
    let nextCounters : Counts :=
      ⟨state.counters.numRefits,
       if rebuild then state.counters.numRescales else state.counters.numRescales + 1,
       if rebuild then state.counters.numRebuilds + 1 else state.counters.numRebuilds⟩
    let nextState : State :=
      { state with weights, built := nextBuilt, counters := nextCounters }
    .ok (nextState, ⟨xScale, rebuild⟩)

def refitAuto (sqrtOp : SqrtOp) (ln : LogOp) (state : State) :
    Except Err (State × Option Update) := do
  let (w, gain) ← autoWeights sqrtOp ln state.rows.dim state.tied state.rows.xs state.rows.ys
  let nextRefits := state.counters.numRefits + 1
  let grown := (3 * state.rows.numSeen + 1) / 2
  let nextRefit := max minRows grown
  let stateRefitted : State :=
    { state with
      heldoutGain := some gain,
      nextRefit,
      counters := { state.counters with numRefits := nextRefits } }
  let target := if usesLearned gain then w
    else List.replicate state.rows.dim Bounds.one
  let change := mul halfWord ((List.range state.rows.dim).foldl (fun best j =>
    let ratio := divide (target.getD j Bounds.one) (state.weights.getD j Bounds.one)
    let absLn := (Float.absDouble (Float.wordDouble (ln ratio))).bits
    maxWord best absLn) 0)
  if Float.ltDouble rescaleTol change then
    let (updated, update) ← setWeights sqrtOp ln stateRefitted target
    return (updated, some update)
  else
    return (stateRefitted, none)

def observeFuel (fuel : Nat) (sqrtOp : SqrtOp) (ln : LogOp) (state : State)
    (x y : Fitter.Matrix) : Option (Except Err (State × Option Update)) :=
  if x.rows != y.rows || x.cols != state.rows.dim || y.cols != state.rows.outputs then
    some (.error .shape)
  else
    match rowsFuel fuel state.rows x.data y.data with
    | none => none
    | some nextRows =>
      let stepped : State := { state with rows := nextRows }
      if stepped.rows.numSeen >= stepped.nextRefit then
        some (refitAuto sqrtOp ln stepped)
      else
        some (.ok (stepped, none))

def stepFuel (fuel : Nat) (sqrtOp : SqrtOp) (ln : LogOp) (state : Option State)
    (x y : Fitter.Matrix) : Option (Except Err (Option State × Option Update)) :=
  match state with
  | none => some (.ok (none, none))
  | some st => match observeFuel fuel sqrtOp ln st x y with
      | none => none
      | some (.ok (nextState, update)) => some (.ok (some nextState, update))
      | some (.error cause) => some (.error cause)

def stepEnable (dim outputs : Nat) (tied : List (List Nat)) (seed : Word) :
    Except Err State :=
  newMetric dim outputs tied seed

theorem emptyDrift (ln : LogOp) :
    weightDrift ln [] [] = mul halfWord 0 := rfl

theorem emptyRes (dim outputs : Nat) (seed : Word) :
    (resEmpty dim outputs seed).numSeen = 0 := rfl

theorem pushOne (dim outputs : Nat) (seed : Word) (x y : List Word) :
    (pushFuel 0 (resEmpty dim outputs seed) x y).map Reservoir.numSeen = some 1 := rfl

theorem defaultK : autoK = 10 := rfl

theorem capEq : resCap = 1000 := rfl

theorem rowMin : minRows = 100 := rfl

theorem learnOk (gain : Word) :
    usesLearned gain = (Float.finiteDouble gain && Float.ltDouble minGain gain) := rfl

/-- Exact observation transition relation on automatic metric state. -/
inductive ObsRel (sqrtOp : SqrtOp) (ln : LogOp) :
    State → Fitter.Matrix → Fitter.Matrix → Except Err (State × Option Update) → Prop where
  | shape (st : State) (x y : Fitter.Matrix) :
      (x.rows ≠ y.rows ∨ x.cols ≠ st.rows.dim ∨ y.cols ≠ st.rows.outputs) →
      ObsRel sqrtOp ln st x y (.error .shape)
  | pass (st : State) (x y : Fitter.Matrix) (nextRows : Reservoir) :
      x.rows = y.rows →
      x.cols = st.rows.dim →
      y.cols = st.rows.outputs →
      RowsRel st.rows x.data y.data nextRows →
      nextRows.numSeen < st.nextRefit →
      ObsRel sqrtOp ln st x y (.ok ({ st with rows := nextRows }, none))
  | refit (st : State) (x y : Fitter.Matrix) (nextRows : Reservoir)
      (out : State × Option Update) :
      x.rows = y.rows →
      x.cols = st.rows.dim →
      y.cols = st.rows.outputs →
      RowsRel st.rows x.data y.data nextRows →
      nextRows.numSeen ≥ st.nextRefit →
      refitAuto sqrtOp ln { st with rows := nextRows } = .ok out →
      ObsRel sqrtOp ln st x y (.ok out)
  | refitErr (st : State) (x y : Fitter.Matrix) (nextRows : Reservoir) (cause : Err) :
      x.rows = y.rows →
      x.cols = st.rows.dim →
      y.cols = st.rows.outputs →
      RowsRel st.rows x.data y.data nextRows →
      nextRows.numSeen ≥ st.nextRefit →
      refitAuto sqrtOp ln { st with rows := nextRows } = .error cause →
      ObsRel sqrtOp ln st x y (.error cause)

/-- Exact step observation relation lifting through optional state. -/
inductive StepRel (sqrtOp : SqrtOp) (ln : LogOp) :
    Option State → Fitter.Matrix → Fitter.Matrix →
    Except Err (Option State × Option Update) → Prop where
  | none (x y : Fitter.Matrix) :
      StepRel sqrtOp ln none x y (.ok (none, none))
  | someOk (st nextSt : State) (upd : Option Update) (x y : Fitter.Matrix) :
      ObsRel sqrtOp ln st x y (.ok (nextSt, upd)) →
      StepRel sqrtOp ln (some st) x y (.ok (some nextSt, upd))
  | someErr (st : State) (cause : Err) (x y : Fitter.Matrix) :
      ObsRel sqrtOp ln st x y (.error cause) →
      StepRel sqrtOp ln (some st) x y (.error cause)

theorem dimMismatch (sqrtOp : SqrtOp) (ln : LogOp) (state : State) :
    observeFuel 0 sqrtOp ln state ⟨1, state.rows.dim + 1, []⟩
      ⟨1, state.rows.outputs, []⟩ = some (.error .shape) := by
  simp [observeFuel]

theorem badWeight (sqrtOp : SqrtOp) (ln : LogOp) (state : State) :
    setWeights sqrtOp ln state (List.replicate (state.rows.dim + 1) Bounds.one) =
      .error .shape := by
  simp [setWeights]

#print axioms emptyDrift
#print axioms emptyRes
#print axioms pushOne
#print axioms defaultK
#print axioms capEq
#print axioms rowMin
#print axioms learnOk
#print axioms dimMismatch
#print axioms badWeight
end BauxResearch.Semantics.Metric
