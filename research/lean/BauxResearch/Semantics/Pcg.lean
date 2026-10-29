import Mathlib.Data.BitVec
import Mathlib.Tactic

/-!
# Exact semantics of NumPy PCG64 and SeedSequence

Ported from ENN 0.4.1 `numpy_seed.rs` and `numpy_pcg.rs`.
This module formalizes:
1. `SeedSequence` u64 expansion into 128-bit state and increment words.
2. 128-bit LCG state transition with 64-bit XSL-RR output permutation.
3. Cached 32-bit output buffer semantics (`has_u32` / `u32_buf`).
4. IEEE-754 53-bit fraction float extraction (`next_f64`).
5. Exact inductive big-step relation for Lemire rejection sampling (`Lemire32`, `Lemire64`).
6. Source semantics for bounded integers (`IntHigh`, `TakeHigh`) with explicit divergence.
7. Determinism and first-acceptance theorems for rejection sampling.
8. State and draw-order correspondence between half and word draws (`cachedSplit`).
9. Bounded evaluators for test fixtures returning `Option` on fuel exhaustion without fallbacks.
10. Exact regression fixtures for state initialization, sequential draws, and rejection paths.
-/

namespace BauxResearch.Semantics.Pcg

def multHi : BitVec 128 := BitVec.ofNat 128 2549297995355413924
def multLo : BitVec 128 := BitVec.ofNat 128 4865540595714422341
def multVal : BitVec 128 := (multHi <<< 64) ||| multLo

def initA : BitVec 32 := 0x43b0d7e5
def multA : BitVec 32 := 0x931e8875
def initB : BitVec 32 := 0x8b51f9dd
def multB : BitVec 32 := 0x58f38ded
def mixLeft : BitVec 32 := 0xca01f9dd
def mixRight : BitVec 32 := 0x4973f715
def shiftX : Nat := 16

def hashmix (val : BitVec 32) (hc : BitVec 32) : BitVec 32 × BitVec 32 :=
  let val := val ^^^ hc
  let hc := hc * multA
  let val := val * hc
  let val := val ^^^ (val >>> shiftX)
  (val, hc)

def mixWords (x y : BitVec 32) : BitVec 32 :=
  let res := mixLeft * x - mixRight * y
  res ^^^ (res >>> shiftX)

def wordsEntropy (entropy : BitVec 64) : List (BitVec 32) :=
  if entropy = 0 then [0]
  else if entropy.toNat ≤ 0xffffffff then
    [BitVec.setWidth 32 entropy]
  else
    [BitVec.setWidth 32 entropy, BitVec.setWidth 32 (entropy >>> 32)]

def mixEntropy (entropy : List (BitVec 32)) : List (BitVec 32) :=
  let step1 := (List.range 4).foldl (fun (acc : List (BitVec 32) × BitVec 32) i ↦
    let word := if i < entropy.length then entropy.getD i 0 else 0
    let (v, newHc) := hashmix word acc.2
    (acc.1 ++ [v], newHc)) ([], initA)
  let pool := step1.1
  let hc := step1.2
  let pairs := [(0, 1), (0, 2), (0, 3),
                (1, 0), (1, 2), (1, 3),
                (2, 0), (2, 1), (2, 3),
                (3, 0), (3, 1), (3, 2)]
  let step2 := pairs.foldl (fun (acc : List (BitVec 32) × BitVec 32) (src, dst) ↦
    let (mixed, newHc) := hashmix (acc.1.getD src 0) acc.2
    let curDst := acc.1.getD dst 0
    let updated := mixWords curDst mixed
    let newPool := acc.1.set dst updated
    (newPool, newHc)) (pool, hc)
  step2.1

/-- `SeedSequence(entropy).generate_state(n_u64, dtype=uint64)`. -/
def seedWords (entropy : BitVec 64) (nU64 : Nat) : List (BitVec 64) :=
  let words := wordsEntropy entropy
  let pool := mixEntropy words
  let nU32 := nU64 * 2
  let step := (List.range nU32).foldl (fun (acc : List (BitVec 32) × BitVec 32) i ↦
    let data := (pool.getD (i % 4) 0) ^^^ acc.2
    let hc := acc.2 * multB
    let data := data * hc
    let slot := data ^^^ (data >>> shiftX)
    (acc.1 ++ [slot], hc)) ([], initB)
  let state := step.1
  (List.range nU64).map fun i ↦
    let c0 := (state.getD (2 * i) 0).toNat
    let c1 := (state.getD (2 * i + 1) 0).toNat
    BitVec.ofNat 64 (c0 + (c1 <<< 32))

/-- Concrete PCG64 state matching NumPy Generator(PCG64(seed)). -/
structure State where
  state : BitVec 128
  inc : BitVec 128
  hasU32 : Bool := false
  u32Buf : BitVec 32 := 0
  deriving DecidableEq, Repr

def fromParts (state inc : BitVec 128) : State :=
  { state := state, inc := inc, hasU32 := false, u32Buf := 0 }

/-- Initialize PCG64 from a 64-bit entropy seed via 4 SeedSequence u64 words. -/
def fromSeed (seed : BitVec 64) : State :=
  let words := seedWords seed 4
  let initstate := (BitVec.zeroExtend 128 (words.getD 0 0) <<< 64) |||
                   (BitVec.zeroExtend 128 (words.getD 1 0))
  let initseq   := (BitVec.zeroExtend 128 (words.getD 2 0) <<< 64) |||
                   (BitVec.zeroExtend 128 (words.getD 3 0))
  let inc := (initseq <<< 1) ||| 1
  let mutState : BitVec 128 := 0
  let mutState := mutState * multVal + inc
  let mutState := mutState + initstate
  let mutState := mutState * multVal + inc
  { state := mutState, inc := inc, hasU32 := false, u32Buf := 0 }

/-- Advance 128-bit state by one LCG step and emit 64-bit XSL-RR output. -/
def nextWord (s : State) : BitVec 64 × State :=
  let state := s.state * multVal + s.inc
  let high : BitVec 64 := BitVec.setWidth 64 (state >>> 64)
  let low : BitVec 64 := BitVec.setWidth 64 state
  let rot : Nat := (high >>> 58).toNat
  let word := (high ^^^ low).rotateRight rot
  (word, { s with state := state })

/-- Cached 32-bit generator view matching NumPy PCG64 next_u32. -/
def nextHalf (s : State) : BitVec 32 × State :=
  if s.hasU32 then
    (s.u32Buf, { s with hasU32 := false })
  else
    let (next, s) := nextWord s
    let buf : BitVec 32 := BitVec.setWidth 32 (next >>> 32)
    (BitVec.setWidth 32 next, { s with hasU32 := true, u32Buf := buf })

/-- 53-bit mantissa extraction for standard f64: (next_u64 >> 11). -/
def nextFloat (s : State) : BitVec 64 × State :=
  let (word, s) := nextWord s
  (word >>> 11, s)

/-- Exact inductive big-step relation for 32-bit Lemire rejection sampling.
Relates an initial state `s` to an output `v` and next state `s'` after zero or
more rejected candidate draws ending at the first accepted candidate draw. -/
inductive Lemire32 (rng : BitVec 32) : State → BitVec 32 → State → Prop where
  | accept (s s' : State) (u : BitVec 32)
      (hdraw : nextHalf s = (u, s'))
      (hacc : ((2^32 - 1) - rng.toNat) % (rng.toNat + 1) ≤
        (u.toNat * (rng.toNat + 1)) % (2^32)) :
      Lemire32 rng s (BitVec.ofNat 32 ((u.toNat * (rng.toNat + 1)) >>> 32)) s'
  | reject (s s' s'' : State) (u : BitVec 32) (out : BitVec 32)
      (hdraw : nextHalf s = (u, s'))
      (hrej : (u.toNat * (rng.toNat + 1)) % (2^32) <
        ((2^32 - 1) - rng.toNat) % (rng.toNat + 1))
      (hrest : Lemire32 rng s' out s'') :
      Lemire32 rng s out s''

/-- Exact inductive big-step relation for 64-bit Lemire rejection sampling.
Relates an initial state `s` to an output `v` and next state `s'` after zero or
more rejected candidate draws ending at the first accepted candidate draw. -/
inductive Lemire64 (rng : BitVec 64) : State → BitVec 64 → State → Prop where
  | accept (s s' : State) (u : BitVec 64)
      (hdraw : nextWord s = (u, s'))
      (hacc : ((2^64 - 1) - rng.toNat) % (rng.toNat + 1) ≤
        (u.toNat * (rng.toNat + 1)) % (2^64)) :
      Lemire64 rng s (BitVec.ofNat 64 ((u.toNat * (rng.toNat + 1)) >>> 64)) s'
  | reject (s s' s'' : State) (u : BitVec 64) (out : BitVec 64)
      (hdraw : nextWord s = (u, s'))
      (hrej : (u.toNat * (rng.toNat + 1)) % (2^64) <
        ((2^64 - 1) - rng.toNat) % (rng.toNat + 1))
      (hrest : Lemire64 rng s' out s'') :
      Lemire64 rng s out s''

/-- Exact inductive big-step relation for NumPy `Generator.integers(0, high)`. -/
inductive IntHigh : BitVec 64 → State → BitVec 64 → State → Prop where
  | zero (s : State) (high : BitVec 64) (h : high.toNat ≤ 1) :
      IntHigh high s 0 s
  | maxHalf (high : BitVec 64) (s s' : State) (u : BitVec 32)
      (hgt : 1 < high.toNat)
      (heq : high.toNat - 1 = 0xffffffff)
      (hdraw : nextHalf s = (u, s')) :
      IntHigh high s (BitVec.zeroExtend 64 u) s'
  | lemire32 (s s' : State) (high : BitVec 64) (v : BitVec 32)
      (hgt : 1 < high.toNat)
      (hlt : high.toNat - 1 < 0xffffffff)
      (hstep : Lemire32 (BitVec.ofNat 32 (high.toNat - 1)) s v s') :
      IntHigh high s (BitVec.zeroExtend 64 v) s'
  | lemire64 (s s' : State) (high : BitVec 64) (v : BitVec 64)
      (hgt : 0xffffffff < high.toNat - 1)
      (hstep : Lemire64 (BitVec.ofNat 64 (high.toNat - 1)) s v s') :
      IntHigh high s v s'

/-- Exact inductive big-step relation for a sequence of integer draws. -/
inductive TakeHigh : List (BitVec 64) → State → List (BitVec 64) → State → Prop where
  | nil (s : State) : TakeHigh [] s [] s
  | cons (h : BitVec 64) (hs : List (BitVec 64)) (s s' s'' : State)
      (v : BitVec 64) (vs : List (BitVec 64))
      (hstep : IntHigh h s v s')
      (hrest : TakeHigh hs s' vs s'') :
      TakeHigh (h :: hs) s (v :: vs) s''

/-- Explicit divergence: absence of any finite terminating acceptance derivation. -/
def Diverges32 (rng : BitVec 32) (s : State) : Prop :=
  ∀ v s', ¬ Lemire32 rng s v s'

/-- Explicit divergence: absence of any finite terminating acceptance derivation. -/
def Diverges64 (rng : BitVec 64) (s : State) : Prop :=
  ∀ v s', ¬ Lemire64 rng s v s'

/-- Determinism of 32-bit Lemire rejection sampling: big-step outcomes are unique. -/
theorem det32 (rng : BitVec 32) {s s1 s2 : State} {v1 v2 : BitVec 32}
    (h1 : Lemire32 rng s v1 s1) (h2 : Lemire32 rng s v2 s2) : v1 = v2 ∧ s1 = s2 := by
  induction h1 generalizing v2 s2 with
  | accept s s1' u1 hdraw1 hacc1 =>
    cases h2 with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact ⟨rfl, rfl⟩
    | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
  | reject s s_mid s1' u1 out1 hdraw1 hrej1 hrest1 ih =>
    cases h2 with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
    | reject _ s_mid2 s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact ih hrest2

/-- Determinism of 64-bit Lemire rejection sampling: big-step outcomes are unique. -/
theorem det64 (rng : BitVec 64) {s s1 s2 : State} {v1 v2 : BitVec 64}
    (h1 : Lemire64 rng s v1 s1) (h2 : Lemire64 rng s v2 s2) : v1 = v2 ∧ s1 = s2 := by
  induction h1 generalizing v2 s2 with
  | accept s s1' u1 hdraw1 hacc1 =>
    cases h2 with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact ⟨rfl, rfl⟩
    | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
  | reject s s_mid s1' u1 out1 hdraw1 hrej1 hrest1 ih =>
    cases h2 with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
    | reject _ s_mid2 s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact ih hrest2

/-- First-acceptance property for 32-bit: an accepted candidate draw completes immediately
and uniquely determines the output value and state. -/
theorem first32 (rng : BitVec 32) (s s' : State) (u : BitVec 32)
    (hdraw : nextHalf s = (u, s'))
    (hacc : ((2^32 - 1) - rng.toNat) % (rng.toNat + 1) ≤
      (u.toNat * (rng.toNat + 1)) % (2^32)) :
    ∀ v s'', Lemire32 rng s v s'' →
      v = BitVec.ofNat 32 ((u.toNat * (rng.toNat + 1)) >>> 32) ∧ s'' = s' := by
  intro v s'' hstep
  cases hstep with
  | accept _ s2' u2 hdraw2 hacc2 =>
    rw [hdraw] at hdraw2
    injection hdraw2 with hu hs
    subst hu hs
    exact ⟨rfl, rfl⟩
  | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
    rw [hdraw] at hdraw2
    injection hdraw2 with hu _
    subst hu
    omega

/-- First-acceptance step progression for 32-bit: a rejected candidate draw cannot accept
and transitions directly to the remainder of the rejection chain. -/
theorem step32 (rng : BitVec 32) (s s' : State) (u : BitVec 32)
    (hdraw : nextHalf s = (u, s'))
    (hrej : (u.toNat * (rng.toNat + 1)) % (2^32) <
      ((2^32 - 1) - rng.toNat) % (rng.toNat + 1)) :
    ∀ v s'', Lemire32 rng s v s'' ↔ Lemire32 rng s' v s'' := by
  intro v s''
  constructor
  · intro hstep
    cases hstep with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
    | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact hrest2
  · intro hrest
    exact Lemire32.reject s s' s'' u v hdraw hrej hrest

/-- First-acceptance property for 64-bit: an accepted candidate draw completes immediately
and uniquely determines the output value and state. -/
theorem first64 (rng : BitVec 64) (s s' : State) (u : BitVec 64)
    (hdraw : nextWord s = (u, s'))
    (hacc : ((2^64 - 1) - rng.toNat) % (rng.toNat + 1) ≤
      (u.toNat * (rng.toNat + 1)) % (2^64)) :
    ∀ v s'', Lemire64 rng s v s'' →
      v = BitVec.ofNat 64 ((u.toNat * (rng.toNat + 1)) >>> 64) ∧ s'' = s' := by
  intro v s'' hstep
  cases hstep with
  | accept _ s2' u2 hdraw2 hacc2 =>
    rw [hdraw] at hdraw2
    injection hdraw2 with hu hs
    subst hu hs
    exact ⟨rfl, rfl⟩
  | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
    rw [hdraw] at hdraw2
    injection hdraw2 with hu _
    subst hu
    omega

/-- First-acceptance step progression for 64-bit: a rejected candidate draw cannot accept
and transitions directly to the remainder of the rejection chain. -/
theorem step64 (rng : BitVec 64) (s s' : State) (u : BitVec 64)
    (hdraw : nextWord s = (u, s'))
    (hrej : (u.toNat * (rng.toNat + 1)) % (2^64) <
      ((2^64 - 1) - rng.toNat) % (rng.toNat + 1)) :
    ∀ v s'', Lemire64 rng s v s'' ↔ Lemire64 rng s' v s'' := by
  intro v s''
  constructor
  · intro hstep
    cases hstep with
    | accept _ s2' u2 hdraw2 hacc2 =>
      rw [hdraw] at hdraw2
      injection hdraw2 with hu _
      subst hu
      omega
    | reject _ s_mid s2' u2 out2 hdraw2 hrej2 hrest2 =>
      rw [hdraw] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact hrest2
  · intro hrest
    exact Lemire64.reject s s' s'' u v hdraw hrej hrest

/-- Determinism of NumPy `integers(0, high)`. -/
theorem detHigh {high : BitVec 64} {s s1 s2 : State} {v1 v2 : BitVec 64}
    (h1 : IntHigh high s v1 s1) (h2 : IntHigh high s v2 s2) : v1 = v2 ∧ s1 = s2 := by
  cases h1 with
  | zero _ _ hle1 =>
    cases h2 with
    | zero => exact ⟨rfl, rfl⟩
    | maxHalf _ _ _ _ hgt2 _ _ => omega
    | lemire32 _ _ _ _ hgt2 _ _ => omega
    | lemire64 _ _ _ _ hgt2 _ => omega
  | maxHalf _ _ s1' u1 hgt1 heq1 hdraw1 =>
    cases h2 with
    | zero _ _ hle2 => omega
    | maxHalf _ _ s2' u2 _ _ hdraw2 =>
      rw [hdraw1] at hdraw2
      injection hdraw2 with hu hs
      subst hu hs
      exact ⟨rfl, rfl⟩
    | lemire32 _ _ _ _ _ hlt2 _ => omega
    | lemire64 _ _ _ _ hgt2 _ => omega
  | lemire32 _ s1' _ v1' hgt1 hlt1 hstep1 =>
    cases h2 with
    | zero _ _ hle2 => omega
    | maxHalf _ _ _ _ _ heq2 _ => omega
    | lemire32 _ s2' _ v2' _ _ hstep2 =>
      have ⟨hv, hs⟩ := det32 _ hstep1 hstep2
      subst hv hs
      exact ⟨rfl, rfl⟩
    | lemire64 _ _ _ _ hgt2 _ => omega
  | lemire64 _ s1' _ v1' hgt1 hstep1 =>
    cases h2 with
    | zero _ _ hle2 => omega
    | maxHalf _ _ _ _ _ heq2 _ => omega
    | lemire32 _ _ _ _ _ hlt2 _ => omega
    | lemire64 _ s2' _ v2' _ hstep2 =>
      have ⟨hv, hs⟩ := det64 _ hstep1 hstep2
      subst hv hs
      exact ⟨rfl, rfl⟩

/-- Determinism of sequential integer draws. -/
theorem detList {highs : List (BitVec 64)} {s s1 s2 : State} {v1 v2 : List (BitVec 64)}
    (h1 : TakeHigh highs s v1 s1) (h2 : TakeHigh highs s v2 s2) : v1 = v2 ∧ s1 = s2 := by
  induction h1 generalizing v2 s2 with
  | nil =>
    cases h2
    exact ⟨rfl, rfl⟩
  | cons h hs s s_mid1 s1' v1 vs1 hstep1 hrest1 ih =>
    cases h2 with
    | cons _ _ _ s_mid2 s2' v2 vs2 hstep2 hrest2 =>
      have ⟨hv, hs_mid⟩ := detHigh hstep1 hstep2
      subst hv hs_mid
      have ⟨hvs, hs2⟩ := ih hrest2
      subst hvs hs2
      exact ⟨rfl, rfl⟩

/-- Bounded 32-bit Lemire evaluator for test fixtures only (not optimizer semantics).
Returns `none` on fuel exhaustion without fabricating outputs. -/
def eval32 (fuel : Nat) (rng : BitVec 32) (s : State) : Option (BitVec 32 × State) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    let rngExcl := rng.toNat + 1
    let threshold := ((2^32 - 1) - rng.toNat) % rngExcl
    let (u, s') := nextHalf s
    let m := u.toNat * rngExcl
    let leftover := m % (2^32)
    if threshold ≤ leftover then
      some (BitVec.ofNat 32 (m >>> 32), s')
    else
      eval32 fuel rng s'

/-- Bounded 64-bit Lemire evaluator for test fixtures only (not optimizer semantics).
Returns `none` on fuel exhaustion without fabricating outputs. -/
def eval64 (fuel : Nat) (rng : BitVec 64) (s : State) : Option (BitVec 64 × State) :=
  match fuel with
  | 0 => none
  | fuel + 1 =>
    let rngExcl := rng.toNat + 1
    let threshold := ((2^64 - 1) - rng.toNat) % rngExcl
    let (u, s') := nextWord s
    let m := u.toNat * rngExcl
    let leftover := m % (2^64)
    if threshold ≤ leftover then
      some (BitVec.ofNat 64 (m >>> 64), s')
    else
      eval64 fuel rng s'

/-- Bounded evaluator for `integers(0, high)` on test fixtures only. -/
def evalHigh (fuel : Nat) (high : BitVec 64) (s : State) : Option (BitVec 64 × State) :=
  if high.toNat ≤ 1 then
    some (0, s)
  else if high.toNat - 1 = 0xffffffff then
    let pair := nextHalf s
    some (BitVec.zeroExtend 64 pair.1, pair.2)
  else if high.toNat - 1 < 0xffffffff then
    match eval32 fuel (BitVec.ofNat 32 (high.toNat - 1)) s with
    | some (v, s') => some (BitVec.zeroExtend 64 v, s')
    | none => none
  else
    eval64 fuel (BitVec.ofNat 64 (high.toNat - 1)) s

/-- Bounded evaluator for a sequence of integer draws on test fixtures only. -/
def evalList (fuel : Nat) : List (BitVec 64) → State → Option (List (BitVec 64) × State)
  | [], s => some ([], s)
  | h :: hs, s =>
    match evalHigh fuel h s with
    | some (v, s') =>
      match evalList fuel hs s' with
      | some (vs, s'') => some (v :: vs, s'')
      | none => none
    | none => none

/-- Soundness of 32-bit bounded evaluator: successful evaluation implies the big-step
relation holds. -/
theorem sound32 (fuel : Nat) (rng : BitVec 32) (s : State) (v : BitVec 32) (s' : State)
    (h : eval32 fuel rng s = some (v, s')) : Lemire32 rng s v s' := by
  induction fuel generalizing s with
  | zero => contradiction
  | succ fuel ih =>
    unfold eval32 at h
    dsimp only at h
    rcases hdraw : nextHalf s with ⟨u, s1⟩
    rw [hdraw] at h
    dsimp only at h
    split at h
    · rename_i hacc
      injection h with hres
      injection hres with hv hs
      rw [← hv, ← hs]
      exact Lemire32.accept s s1 u hdraw hacc
    · rename_i hnot
      have hrej : (u.toNat * (rng.toNat + 1)) % 2^32 <
          ((2^32 - 1) - rng.toNat) % (rng.toNat + 1) := by
        omega
      exact Lemire32.reject s s1 s' u v hdraw hrej (ih s1 h)

/-- Soundness of 64-bit bounded evaluator: successful evaluation implies the big-step
relation holds. -/
theorem sound64 (fuel : Nat) (rng : BitVec 64) (s : State) (v : BitVec 64) (s' : State)
    (h : eval64 fuel rng s = some (v, s')) : Lemire64 rng s v s' := by
  induction fuel generalizing s with
  | zero => contradiction
  | succ fuel ih =>
    unfold eval64 at h
    dsimp only at h
    rcases hdraw : nextWord s with ⟨u, s1⟩
    rw [hdraw] at h
    dsimp only at h
    split at h
    · rename_i hacc
      injection h with hres
      injection hres with hv hs
      rw [← hv, ← hs]
      exact Lemire64.accept s s1 u hdraw hacc
    · rename_i hnot
      have hrej : (u.toNat * (rng.toNat + 1)) % 2^64 <
          ((2^64 - 1) - rng.toNat) % (rng.toNat + 1) := by
        omega
      exact Lemire64.reject s s1 s' u v hdraw hrej (ih s1 h)

/-- Soundness of bounded `evalHigh` against exact big-step `IntHigh`. -/
theorem soundHigh (fuel : Nat) (high : BitVec 64) (s : State) (v : BitVec 64) (s' : State)
    (h : evalHigh fuel high s = some (v, s')) : IntHigh high s v s' := by
  unfold evalHigh at h
  split at h
  · rename_i hle
    injection h with hres
    injection hres with hv hs
    subst hv hs
    exact IntHigh.zero s high hle
  · rename_i hgt
    have hgt1 : 1 < high.toNat := by omega
    split at h
    · rename_i heq32
      rcases hdraw : nextHalf s with ⟨u, s1⟩
      rw [hdraw] at h
      dsimp only at h
      injection h with hres
      injection hres with hv hs
      subst hv hs
      exact IntHigh.maxHalf high s s1 u hgt1 heq32 hdraw
    · rename_i hne32
      split at h
      · rename_i hlt32
        split at h
        · rename_i v32 s1 heval
          injection h with hres
          injection hres with hv hs
          subst hv hs
          exact IntHigh.lemire32 s s1 high v32 hgt1 hlt32 (sound32 fuel _ s v32 s1 heval)
        · contradiction
      · rename_i hge32
        have hgt_words : 0xffffffff < high.toNat - 1 := by omega
        exact IntHigh.lemire64 s s' high v hgt_words (sound64 fuel _ s v s' h)

/-- Soundness of sequential bounded evaluator against exact big-step `TakeHigh`. -/
theorem soundList (fuel : Nat) (highs : List (BitVec 64)) (s : State)
    (vs : List (BitVec 64)) (s' : State)
    (h : evalList fuel highs s = some (vs, s')) : TakeHigh highs s vs s' := by
  induction highs generalizing s vs with
  | nil =>
    unfold evalList at h
    injection h with hres
    injection hres with hvs hs
    subst hvs hs
    exact TakeHigh.nil s
  | cons head tail ih =>
    unfold evalList at h
    split at h
    · rename_i v1 s1 heval1
      split at h
      · rename_i vtail s2 heval2
        injection h with hres
        injection hres with hvs hs
        subst hvs hs
        exact TakeHigh.cons head tail s s1 s2 v1 vtail
          (soundHigh fuel head s v1 s1 heval1) (ih s1 vtail heval2)
      · contradiction
    · contradiction

/-- Sequential 64-bit output stream prefix. -/
def takeWord : Nat → State → List (BitVec 64) × State
  | 0, s => ([], s)
  | n + 1, s =>
    let (w, s') := nextWord s
    let (rest, s'') := takeWord n s'
    (w :: rest, s'')

/-- Sequential 32-bit output stream prefix. -/
def takeHalf : Nat → State → List (BitVec 32) × State
  | 0, s => ([], s)
  | n + 1, s =>
    let (w, s') := nextHalf s
    let (rest, s'') := takeHalf n s'
    (w :: rest, s'')

/-- State and draw-order correspondence between two 32-bit draws and one 64-bit draw. -/
theorem cachedSplit (s : State) (h : s.hasU32 = false) :
    let (u0, s1) := nextHalf s
    let (u1, s2) := nextHalf s1
    let (w, s64) := nextWord s
    u0 = BitVec.setWidth 32 w ∧
    u1 = BitVec.setWidth 32 (w >>> 32) ∧
    s2.state = s64.state ∧
    s2.inc = s64.inc ∧
    s2.hasU32 = false := by
  unfold nextHalf
  simp [h]

/-- Positive acceptance in 32-bit Lemire: rejection threshold is strictly bounded. -/
theorem bound32 (rng : BitVec 32) (h : rng.toNat < 0xffffffff) :
    let rngExcl := rng.toNat + 1
    let threshold := ((2^32 - 1) - rng.toNat) % rngExcl
    threshold < rngExcl ∧ 0 < 2^32 - threshold := by
  intro rngExcl threshold
  have hExcl : 0 < rngExcl := by omega
  have hMod : threshold < rngExcl := Nat.mod_lt _ hExcl
  have hLe : rngExcl ≤ 2^32 - 1 := by omega
  refine ⟨hMod, ?_⟩
  omega

/-- Positive acceptance in 64-bit Lemire: rejection threshold is strictly bounded. -/
theorem bound64 (rng : BitVec 64) (h : rng.toNat < 0xffffffffffffffff) :
    let rngExcl := rng.toNat + 1
    let threshold := ((2^64 - 1) - rng.toNat) % rngExcl
    threshold < rngExcl ∧ 0 < 2^64 - threshold := by
  intro rngExcl threshold
  have hExcl : 0 < rngExcl := by omega
  have hMod : threshold < rngExcl := Nat.mod_lt _ hExcl
  have hLe : rngExcl ≤ 2^64 - 1 := by omega
  refine ⟨hMod, ?_⟩
  omega

theorem boundZero (s : State) : IntHigh 0 s 0 s :=
  IntHigh.zero s 0 (by decide)

theorem boundOne (s : State) : IntHigh 1 s 0 s :=
  IntHigh.zero s 1 (by decide)

theorem maxHalf (s s' : State) (u : BitVec 32) (hdraw : nextHalf s = (u, s')) :
    IntHigh (1 <<< 32) s (BitVec.zeroExtend 64 u) s' :=
  IntHigh.maxHalf (1 <<< 32) s s' u (by decide) (by decide) hdraw

/-- Independent NumPy SeedSequence seed 123 fixture. -/
theorem seedFixture :
    seedWords 123 2 = [12770025807176811766, 11695957281888622767] := by
  decide +kernel

/-- Independent PCG64 seed 0 state and increment initialization fixture. -/
theorem zeroState :
    (fromSeed 0).state = 35399562948360463058890781895381311971 ∧
    (fromSeed 0).inc = 87136372517582989555478159403783844777 := by
  decide +kernel

set_option maxRecDepth 16384 in
/-- Upstream NumPy PCG64 seed 0 integer draws matching numpy_rng_parity.rs. -/
theorem highFixture :
    ((evalList 16 (List.replicate 8 10) (fromSeed 0)).map Prod.fst) =
      some [8, 6, 5, 2, 3, 0, 0, 0] := by
  decide +kernel

set_option maxRecDepth 16384 in
/-- Upstream metric_checks.rs seed 719 integer draws across 32-bit and 64-bit thresholds. -/
theorem fixture719 :
    ((evalList 16 [5, 6, 7, 1 <<< 32, (1 <<< 32) + 3, 1 <<< 48, 17] (fromSeed 719)).map Prod.fst) =
      some [2, 4, 0, 1822184769, 2849654795, 112494928035177, 13] := by
  decide +kernel

/-- Successive u64 draws from seed 0. -/
theorem wordsFixture :
    (takeWord 4 (fromSeed 0)).1 =
      [11749869230777074271, 4976686463289251617, 755828109848996024, 304881062738325533] := by
  decide +kernel

/-- Successive cached u32 draws from seed 0. -/
theorem halvesFixture :
    (takeHalf 4 (fromSeed 0)).1 =
      [3653403231, 2735729615, 2195314465, 1158725112] := by
  decide +kernel

set_option maxRecDepth 16384 in
/-- Edge path: Lemire 32-bit draw with rejection loop executed from seed 0. -/
theorem rej32 :
    ((evalHigh 16 ((1 <<< 31) + 1) (fromSeed 0)).map Prod.fst) = some 87989972 := by
  decide +kernel

set_option maxRecDepth 16384 in
/-- Edge path: Lemire 64-bit draw with rejection loop executed from seed 0. -/
theorem rej64 :
    ((evalHigh 16 ((1 <<< 63) + 1) (fromSeed 0)).map Prod.fst) = some 2488343231644625808 := by
  decide +kernel

#print axioms cachedSplit
#print axioms bound32
#print axioms bound64
#print axioms boundZero
#print axioms boundOne
#print axioms maxHalf
#print axioms seedFixture
#print axioms zeroState
#print axioms highFixture
#print axioms fixture719
#print axioms wordsFixture
#print axioms halvesFixture
#print axioms rej32
#print axioms rej64
#print axioms det32
#print axioms det64
#print axioms first32
#print axioms step32
#print axioms first64
#print axioms step64
#print axioms detHigh
#print axioms detList
#print axioms sound32
#print axioms sound64
#print axioms soundHigh
#print axioms soundList

end BauxResearch.Semantics.Pcg
