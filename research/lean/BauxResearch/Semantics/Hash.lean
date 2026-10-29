import BauxResearch.Semantics.Acq

/-!
# ENN 0.4.1 hash normals

Exact wrapping-integer and binary64 orchestration for the SplitMix64 and
Box--Muller functions used by correlated posterior function draws. Platform
`ln`, `sqrt`, and `cos` remain explicit operations.
-/

namespace BauxResearch.Semantics.Hash

open Sampling

abbrev LnOp := Word → Word
abbrev SqrtOp := Word → Word
abbrev CosOp := Word → Word

def golden : Word := 0x9E3779B97F4A7C15
def multiplier1 : Word := 0xBF58476D1CE4E5B9
def multiplier2 : Word := 0x94D049BB133111EB
def xorOffset : Word := 0xD2B74407B1CE6E93
def seedPrime : Word := 1000003
def querySalt : Word := 0xC0FFEE00D15CA11E

def splitmix (value : Word) : Word :=
  let z := value + golden
  let z := (z ^^^ (z >>> 30)) * multiplier1
  let z := (z ^^^ (z >>> 27)) * multiplier2
  z ^^^ (z >>> 31)

def uniform (value : Word) : Word :=
  Sampling.fraction value 11

def box (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (u1 u2 : Word) : Word :=
  let radius := sqrtOp (mul Acq.negTwo (lnOp u1))
  let angle := mul (mul Acq.two Acq.pi) u2
  mul radius (cosOp angle)

def normalBase (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (combined : Word) : Word :=
  let first := uniform (splitmix combined)
  let second := uniform (splitmix (combined ^^^ xorOffset))
  box lnOp sqrtOp cosOp (Acq.clampUnit first) second

def indexNormal (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (seed index : Word) (metric : Nat) : Word :=
  let base := (seed * seedPrime + index) * seedPrime
  normalBase lnOp sqrtOp cosOp (base + BitVec.ofNat 64 metric)

def queryNormal (lnOp : LnOp) (sqrtOp : SqrtOp) (cosOp : CosOp)
    (seed : Word) (query metric : Nat) : Word :=
  let base := (seed * seedPrime + BitVec.ofNat 64 query) * seedPrime + querySalt
  normalBase lnOp sqrtOp cosOp (base + BitVec.ofNat 64 metric)

/-- Deduplicate flat row indices while recording the inverse mapping for every input slot.
Matches `rust:ennbo 0.4.1 hash/unique_index_inverse().` -/
def uniqueInverse (indices : List Word) : List Word × List Nat :=
  indices.foldl (fun (acc : List Word × List Nat) idx ↦
    match acc.1.idxOf? idx with
    | some pos => (acc.1, acc.2 ++ [pos])
    | none => (acc.1 ++ [idx], acc.2 ++ [acc.1.length])) ([], [])

private theorem invStep (xs : List Word) (init : List Word × List Nat) :
    (xs.foldl (fun (acc : List Word × List Nat) idx ↦
      match acc.1.idxOf? idx with
      | some pos => (acc.1, acc.2 ++ [pos])
      | none => (acc.1 ++ [idx], acc.2 ++ [acc.1.length])) init).2.length =
      init.2.length + xs.length := by
  induction xs generalizing init with
  | nil => simp
  | cons x rest ih =>
    simp only [List.foldl_cons]
    split <;> simp only [ih, List.length_append, List.length_cons, List.length_nil] <;> omega

/-- Inverse index array has the exact same length as the input index slice
(`rust:ennbo 0.4.1 hash/unique_index_inverse().`). -/
theorem inverseLen (indices : List Word) :
    (uniqueInverse indices).2.length = indices.length := by
  unfold uniqueInverse
  simpa using invStep indices ([], [])

theorem splitZero : splitmix 0 = 0xe220a8397b1dcdaf := by
  decide +kernel

theorem uniformZero : uniform 0 = 0 := by
  decide +kernel

#print axioms inverseLen
#print axioms splitZero
#print axioms uniformZero

end BauxResearch.Semantics.Hash
