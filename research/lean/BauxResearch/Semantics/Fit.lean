import BauxResearch.Semantics.Fitter

/-!
# ENN 0.4.1 likelihood semantics

This module gives the binary64 loop used to score one parameter candidate.
Rust `f64::ln` and `f64::powi(_, 2)` remain named implementation boundaries;
the surrounding validation, traversal, arithmetic order, and final finite clamp
are executable.
-/

namespace BauxResearch.Semantics.Fit

open Sampling Bounds

abbrev PowiOp := Word → Int → Word

def two : Word := 0x4000000000000000
def half : Word := 0x3fe0000000000000
def negHalf : Word := 0xbfe0000000000000
def pi : Word := 0x400921fb54442d18

def allFinite : Batch → Bool
  | [] => true
  | row :: rows => row.all Float.finiteDouble && allFinite rows

def anyNonpositive : Batch → Bool
  | [] => false
  | row :: rows => row.any (fun value => Float.leDouble value 0) ||
      anyNonpositive rows

inductive FitError where
  | shape
  deriving Repr, DecidableEq

/-- One source loop iteration, with every binary64 rounding point exposed. -/
def logCell (logOp : LogOp) (powiOp : PowiOp)
    (y mu se : Word) : Word :=
  let variance := mul se se
  let normalizer := mul (mul two pi) variance
  let gaussian := mul negHalf (logOp normalizer)
  let residual := powiOp (sub y mu) 2
  let penalty := divide (mul half residual) variance
  sub gaussian penalty

def logRow (logOp : LogOp) (powiOp : PowiOp) :
    Word → List Word → List Word → List Word → Except FitError Word
  | total, [], [], [] => .ok total
  | total, y :: ys, mu :: mus, se :: ses =>
      logRow logOp powiOp (add total (logCell logOp powiOp y mu se))
        ys mus ses
  | _, _, _, _ => .error .shape

def logRows (logOp : LogOp) (powiOp : PowiOp) :
    Word → Batch → Batch → Batch → Except FitError Word
  | total, [], [], [] => .ok total
  | total, y :: ys, mu :: mus, se :: ses =>
      match logRow logOp powiOp total y mu se with
      | .error cause => .error cause
      | .ok next => logRows logOp powiOp next ys mus ses
  | _, _, _, _ => .error .shape

/-- `compute_single_loglik`, including whole-array validation before traversal
and conversion of a nonfinite accumulated result to negative infinity. -/
def singleLoglik (logOp : LogOp) (powiOp : PowiOp)
    (y mu se : Batch) : Except FitError Word :=
  if !allFinite mu || !allFinite se then .ok negInf
  else if anyNonpositive se then .ok negInf
  else
    match logRows logOp powiOp 0 y mu se with
    | .error cause => .error cause
    | .ok result =>
        if Float.finiteDouble result then .ok result else .ok negInf

def signMask : Word := 0x8000000000000000

/-- Unsigned key with the same ordering as Rust `f64::total_cmp`. -/
def totalKey (value : Word) : Nat :=
  if value.toNat < 2 ^ 63 then (value ^^^ signMask).toNat
  else (~~~value).toNat

def totalLe (left right : Word) : Bool :=
  decide (totalKey left ≤ totalKey right)

def bestLoop : Nat → Nat → Word → List Word → Nat
  | _, best, _, [] => best
  | next, best, value, candidate :: rest =>
      if totalLe value candidate then bestLoop (next + 1) next candidate rest
      else bestLoop (next + 1) best value rest

/-- `.enumerate().max_by(total_cmp).map(index).unwrap_or(0)`. Rust's iterator
maximum keeps the later element on equality. -/
def bestIndex : List Word → Nat
  | [] => 0
  | value :: rest => bestLoop 1 0 value rest

theorem cellOrder (logOp : LogOp) (powiOp : PowiOp) (y mu se : Word) :
    logCell logOp powiOp y mu se =
      let variance := mul se se
      let normalizer := mul (mul two pi) variance
      sub (mul negHalf (logOp normalizer))
        (divide (mul half (powiOp (sub y mu) 2)) variance) := by
  rfl

theorem badMean (logOp : LogOp) (powiOp : PowiOp) :
    singleLoglik logOp powiOp [[0]] [[posInf]] [[Bounds.one]] =
      .ok negInf := by
  rfl

theorem badScale (logOp : LogOp) (powiOp : PowiOp) :
    singleLoglik logOp powiOp [[0]] [[0]] [[0]] = .ok negInf := by
  rfl

theorem oneCell :
    singleLoglik (fun _ => 0) (fun _ _ => Bounds.one)
      [[Bounds.one]] [[0]] [[Bounds.one]] = .ok negHalf := by
  decide +kernel

theorem tieLast : bestIndex [Bounds.one, Bounds.one] = 1 := by
  decide +kernel

theorem nanWins : bestIndex [posInf, 0x7ff8000000000000] = 1 := by
  decide +kernel

theorem lowNan : bestIndex [0xfff8000000000000, negInf] = 1 := by
  decide +kernel

#print axioms cellOrder
#print axioms badMean
#print axioms badScale
#print axioms oneCell
#print axioms tieLast
#print axioms nanWins
#print axioms lowNan

end BauxResearch.Semantics.Fit
