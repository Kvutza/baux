import BauxResearch.Semantics.Candidate
import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Generated.Pure.Funs
import BauxResearch.Extraction.Seed
import BauxResearch.Semantics.Pcg

namespace BauxResearch.Semantics.Seed

/-!
# Extracted candidate seed correspondence

These theorems connect the actual extracted Rust seed helpers to the bit-vector
model for every input. They do not assert source correspondence of the remaining
candidate samplers or the optimizer API.
-/


open Aeneas.Std
open BauxResearch.Generated.Optimizer

private theorem shr30 (value : U64) :
    value >>> 30#i32 = Result.ok (⟨value.bv >>> 30⟩ : U64) := by rfl

private theorem shr27 (value : U64) :
    value >>> 27#i32 = Result.ok (⟨value.bv >>> 27⟩ : U64) := by rfl

private theorem shr31 (value : U64) :
    value >>> 31#i32 = Result.ok (⟨value.bv >>> 31⟩ : U64) := by rfl

theorem baseMatch (value : U64) :
    random.sobol_base value = Result.ok (⟨Candidate.sobolBase value.bv⟩ : U64) := by
  have hmod : random.SOBOL_MOD = Result.ok (2147483648#u64) := by
    unfold random.SOBOL_MOD
    rfl
  simp only [random.sobol_base, hmod, bind_ok]
  rfl

private theorem hostAdd {width : Nat} (word : BitVec width) (host : width = 64) :
    (word + 1#width).setWidth 64 = word.setWidth 64 + 1#64 := by
  subst width
  simp

/-- The model's 64-bit restart addition requires the actual host usize width. -/
theorem seedMatch (base : U64) (restart obs arms : Usize)
    (host : System.Platform.numBits = 64) :
    random.sobol_seed base restart obs arms = Result.ok (⟨Candidate.sobolSeed
      base.bv (restart.bv.setWidth 64) (obs.bv.setWidth 64) (arms.bv.setWidth 64)⟩ : U64) := by
  simp only [random.sobol_seed, shr30, shr27, shr31, lift, bind_ok]
  apply congrArg Result.ok
  apply U64.bv_eq_imp_eq
  simp [Candidate.sobolSeed, Candidate.seedMix, Candidate.finalMix,
    Candidate.stepFirst, Candidate.stepSecond, Candidate.stepThird,
    Candidate.stepFinal, Candidate.stepMask, UScalar.cast]
  rw [hostAdd restart.bv host]

#print axioms baseMatch
#print axioms seedMatch

/-!
# Extracted ENN 0.4.1 `numpy_seed.rs` and `numpy_pcg.rs` Correspondence

Covers all reachable functions in `crates/ennbo/src/numpy_seed.rs` (5 functions)
and bridges `crates/ennbo/src/numpy_pcg.rs` (6 functions) to the extracted
Charon/Aeneas definitions in `BauxResearch.Generated.Pure`:

- `rust:ennbo 0.4.1 numpy_seed/hashmix().` (`hashmixMatch`)
- `rust:ennbo 0.4.1 numpy_seed/mix().` (`mixMatch`)
- `rust:ennbo 0.4.1 numpy_seed/entropy_words().` (`entropyLen`)
- `rust:ennbo 0.4.1 numpy_seed/mix_entropy().` (`poolLen`)
- `rust:ennbo 0.4.1 numpy_seed/seed_sequence_u64().` (`seedLen`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#from_parts().` (`partsMatch`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#from_seed().` (`fromSeed`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_u64().` (`wordMatch`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_u32().` (`halfMatch`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_f64().` (`doubleMatch`)
- `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#standard_normal_pair().` (`normalPair`)
-/


open Aeneas.Std Aeneas.Std.Result
open BauxResearch.Extraction.Native
open BauxResearch.Semantics.Pcg

private theorem shift16 (v : U32) :
    v >>> BauxResearch.Generated.Pure.numpy_seed.XSHIFT =
      Result.ok (⟨v.bv >>> 16⟩ : U32) := by
  unfold BauxResearch.Generated.Pure.numpy_seed.XSHIFT
  rfl

private theorem halfMul (a b : BitVec 32) :
    core.num.U32.wrapping_mul (⟨a⟩ : U32) (⟨b⟩ : U32) = (⟨a * b⟩ : U32) := by
  rfl

private theorem halfSub (a b : BitVec 32) :
    core.num.U32.wrapping_sub (⟨a⟩ : U32) (⟨b⟩ : U32) = (⟨a - b⟩ : U32) := by
  rfl

private theorem halfXor (a b : BitVec 32) :
    (⟨a⟩ : U32) ^^^ (⟨b⟩ : U32) = (⟨a ^^^ b⟩ : U32) := by
  rfl

private theorem seedMult :
    BauxResearch.Generated.Pure.numpy_seed.MULT_A = (⟨multA⟩ : U32) := by
  unfold BauxResearch.Generated.Pure.numpy_seed.MULT_A
  rfl

private theorem leftOk :
    BauxResearch.Generated.Pure.numpy_seed.MIX_MULT_L = (⟨mixLeft⟩ : U32) := by
  unfold BauxResearch.Generated.Pure.numpy_seed.MIX_MULT_L
  rfl

private theorem rightOk :
    BauxResearch.Generated.Pure.numpy_seed.MIX_MULT_R = (⟨mixRight⟩ : U32) := by
  unfold BauxResearch.Generated.Pure.numpy_seed.MIX_MULT_R
  rfl

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_seed/hashmix().` -/
theorem hashmixMatch (val hc : U32) :
    BauxResearch.Generated.Pure.numpy_seed.hashmix val hc =
      Result.ok (⟨(hashmix val.bv hc.bv).1⟩, ⟨(hashmix val.bv hc.bv).2⟩) := by
  cases val with | mk vb =>
  cases hc with | mk hb =>
  simp only [BauxResearch.Generated.Pure.numpy_seed.hashmix,
    seedMult, halfXor, halfMul, shift16, lift, bind_ok, hashmix, shiftX]

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_seed/mix().` -/
theorem mixMatch (x y : U32) :
    BauxResearch.Generated.Pure.numpy_seed.mix x y =
      Result.ok (⟨mixWords x.bv y.bv⟩ : U32) := by
  cases x with | mk xb =>
  cases y with | mk yb =>
  simp only [BauxResearch.Generated.Pure.numpy_seed.mix,
    leftOk, rightOk, halfMul, halfSub, halfXor, shift16, lift, bind_ok, mixWords, shiftX]

/-- Word count invariant for `rust:ennbo 0.4.1 numpy_seed/entropy_words().` -/
theorem entropyLen (entropy : BitVec 64) :
    1 ≤ (wordsEntropy entropy).length ∧ (wordsEntropy entropy).length ≤ 2 := by
  unfold wordsEntropy
  split_ifs <;> simp

private theorem appendFold (xs : List Nat)
    (step : List (BitVec 32) × BitVec 32 → Nat → BitVec 32 × BitVec 32)
    (init : List (BitVec 32) × BitVec 32) :
    ((xs.foldl (fun acc i ↦
      let (v, hc) := step acc i
      (acc.1 ++ [v], hc)) init).1).length = init.1.length + xs.length := by
  induction xs generalizing init with
  | nil => simp
  | cons x rest ih =>
    simp only [List.foldl_cons, ih, List.length_append, List.length_cons, List.length_nil]
    omega

private theorem setFold (pairs : List (Nat × Nat))
    (step : List (BitVec 32) × BitVec 32 → Nat × Nat → List (BitVec 32) × BitVec 32)
    (hstep : ∀ acc p, (step acc p).1.length = acc.1.length)
    (init : List (BitVec 32) × BitVec 32) :
    ((pairs.foldl step init).1).length = init.1.length := by
  induction pairs generalizing init with
  | nil => rfl
  | cons p rest ih =>
    rw [List.foldl_cons, ih, hstep]

/-- Pool width invariant for `rust:ennbo 0.4.1 numpy_seed/mix_entropy().` -/
theorem poolLen (entropy : List (BitVec 32)) :
    (mixEntropy entropy).length = 4 := by
  unfold mixEntropy
  rw [setFold]
  · rw [appendFold]
    simp
  · intro acc p
    rcases p with ⟨src, dst⟩
    simp only [List.length_set]

/-- Output length invariant for `rust:ennbo 0.4.1 numpy_seed/seed_sequence_u64().` -/
theorem seedLen (entropy : BitVec 64) (n : Nat) :
    (seedWords entropy n).length = n := by
  unfold seedWords
  simp only [List.length_map, List.length_range]

/-- Re-export of extracted `sobol_seed` equivalence. -/
theorem sobolMatch (base : U64) (restart obs arms : Usize) :
    BauxResearch.Generated.Seed.sobolSeed base restart obs arms =
      BauxResearch.Generated.Optimizer.random.sobol_seed base restart obs arms :=
  BauxResearch.Extraction.Seed.seedMatch base restart obs arms

/-- Project semantic `Pcg.State` to extracted `NumpyPcg64`. -/
def toPure (s : State) : BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64 :=
  { state := ⟨s.state⟩
    inc := ⟨s.inc⟩
    has_u32 := s.hasU32
    u32_buf := ⟨s.u32Buf⟩ }

/-- Project extracted `NumpyPcg64` to semantic `Pcg.State`. -/
def ofPure (s : BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64) : State :=
  { state := s.state.bv
    inc := s.inc.bv
    hasU32 := s.has_u32
    u32Buf := s.u32_buf.bv }

@[simp]
theorem pureLeft (s : State) : ofPure (toPure s) = s := by
  cases s
  rfl

@[simp]
theorem pureRight (s : BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64) :
    toPure (ofPure s) = s := by
  cases s
  rfl

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#from_parts().` -/
theorem partsMatch (state inc : U128) :
    BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.from_parts state inc =
      Result.ok (toPure (fromParts state.bv inc.bv)) := by
  rfl

private theorem shl64 (v : U128) :
    v <<< 64#i32 = Result.ok (⟨v.bv <<< 64⟩ : U128) := by
  rfl

private theorem wideOr (a b : BitVec 128) :
    (⟨a⟩ : U128) ||| (⟨b⟩ : U128) = (⟨a ||| b⟩ : U128) := by
  rfl

private theorem hiOk :
    BauxResearch.Generated.Pure.numpy_pcg.MULT_HI = (⟨multHi⟩ : U128) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.MULT_HI
  rfl

private theorem loOk :
    BauxResearch.Generated.Pure.numpy_pcg.MULT_LO = (⟨multLo⟩ : U128) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.MULT_LO
  rfl

private theorem multOk :
    BauxResearch.Generated.Pure.numpy_pcg.MULT =
      Result.ok (⟨multVal⟩ : U128) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.MULT multVal
  rw [hiOk, loOk, shl64, bind_ok, wideOr]

private theorem wideMul (a b : BitVec 128) :
    core.num.U128.wrapping_mul (⟨a⟩ : U128) (⟨b⟩ : U128) = (⟨a * b⟩ : U128) := by
  rfl

private theorem wideAdd (a b : BitVec 128) :
    core.num.U128.wrapping_add (⟨a⟩ : U128) (⟨b⟩ : U128) = (⟨a + b⟩ : U128) := by
  rfl

private theorem wideCast (a : BitVec 128) :
    UScalar.cast .U64 (⟨a⟩ : U128) = (⟨BitVec.setWidth 64 a⟩ : U64) := by
  rfl

private theorem wordCast (a : BitVec 64) :
    UScalar.cast .U32 (⟨a⟩ : U64) = (⟨BitVec.setWidth 32 a⟩ : U32) := by
  rfl

private theorem wordXor (a b : BitVec 64) :
    (⟨a⟩ : U64) ^^^ (⟨b⟩ : U64) = (⟨a ^^^ b⟩ : U64) := by
  rfl

private theorem shift64 (v : U128) :
    v >>> 64#i32 = Result.ok (⟨v.bv >>> 64⟩ : U128) := by
  rfl

private theorem shift58 (v : U64) :
    v >>> 58#i32 = Result.ok (⟨v.bv >>> 58⟩ : U64) := by
  rfl

private theorem shift32 (v : U64) :
    v >>> 32#i32 = Result.ok (⟨v.bv >>> 32⟩ : U64) := by
  rfl

private theorem shift11 (v : U64) :
    v >>> 11#i32 = Result.ok (⟨v.bv >>> 11⟩ : U64) := by
  rfl

private theorem rotateMatch (w high : BitVec 64) :
    core.num.U64.rotate_right (⟨w⟩ : U64) (⟨BitVec.setWidth 32 (high >>> 58)⟩ : U32) =
      (⟨w.rotateRight (high >>> 58).toNat⟩ : U64) := by
  apply U64.bv_eq_imp_eq
  change w.rotateRight (BitVec.setWidth 32 (high >>> 58)).toNat =
    w.rotateRight (high >>> 58).toNat
  have hlt : (high >>> 58).toNat < 64 := by
    rw [BitVec.toNat_ushiftRight, Nat.shiftRight_eq_div_pow]
    have h64 : high.toNat < 2 ^ 64 := BitVec.isLt _
    omega
  rw [BitVec.toNat_setWidth, Nat.mod_eq_of_lt (by omega)]

private theorem wordStep (s : State) (m : BitVec 128) :
    (do
      let i1 ← lift (core.num.U128.wrapping_mul (toPure s).state (⟨m⟩ : U128))
      let i2 ← lift (core.num.U128.wrapping_add i1 (toPure s).inc)
      let i3 ← i2 >>> 64#i32
      let high ← lift (UScalar.cast .U64 i3)
      let low ← lift (UScalar.cast .U64 i2)
      let i4 ← lift (high ^^^ low)
      let i5 ← high >>> 58#i32
      let i6 ← lift (UScalar.cast .U32 i5)
      let i7 ← lift (core.num.U64.rotate_right i4 i6)
      Result.ok (i7, { (toPure s) with state := i2 })) =
    (let state := s.state * m + s.inc
     let high : BitVec 64 := BitVec.setWidth 64 (state >>> 64)
     let low : BitVec 64 := BitVec.setWidth 64 state
     let rot : Nat := (high >>> 58).toNat
     let word := (high ^^^ low).rotateRight rot
     Result.ok ((⟨word⟩ : U64), toPure { s with state := state })) := by
  simp only [toPure, wideMul, wideAdd, shift64, shift58, wideCast, wordXor, wordCast,
    rotateMatch, lift, bind_ok]

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_u64().` -/
theorem wordMatch (s : State) :
    BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_u64 (toPure s) =
      Result.ok (⟨(nextWord s).1⟩, toPure (nextWord s).2) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_u64 nextWord
  rw [multOk, bind_ok]
  exact wordStep s multVal

private theorem halfStep (nw : BitVec 64 × State) :
    (do
      let (next, self1) ← Result.ok ((⟨nw.1⟩ : U64), toPure nw.2)
      let i ← next >>> 32#i32
      let i1 ← lift (UScalar.cast .U32 i)
      let i2 ← lift (UScalar.cast .U32 next)
      Result.ok (i2, { self1 with has_u32 := true, u32_buf := i1 })) =
    (let (next, s1) := nw
     let buf : BitVec 32 := BitVec.setWidth 32 (next >>> 32)
     Result.ok ((⟨BitVec.setWidth 32 next⟩ : U32),
       toPure { s1 with hasU32 := true, u32Buf := buf })) := by
  rcases nw with ⟨next, s1⟩
  simp only [Bind.bind, toPure, shift32, lift, bind_ok]
  rfl

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_u32().` -/
theorem halfMatch (s : State) :
    BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_u32 (toPure s) =
      Result.ok (⟨(nextHalf s).1⟩, toPure (nextHalf s).2) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_u32 nextHalf
  cases h : s.hasU32
  · have ht : (toPure s).has_u32 = false := h
    rw [ht]
    simp only [Bool.false_eq_true, ↓reduceIte]
    rw [wordMatch]
    exact halfStep (nextWord s)
  · have ht : (toPure s).has_u32 = true := h
    rw [ht]
    simp only [↓reduceIte, toPure]

/-- Convert a 53-bit integer word `w = next_u64 >>> 11` to `F64` matching `next_f64`. -/
def unitDouble (w : BitVec 64) : F64 :=
  F64.mul (F64.fromUnsigned (⟨w⟩ : U64))
    (F64.div (F64.literal false 1 1) (F64.literal false 9007199254740992 1))

private theorem doubleStep (nw : BitVec 64 × State) (inv : F64) :
    (do
      let (i, self1) ← Result.ok ((⟨nw.1⟩ : U64), toPure nw.2)
      let i1 ← i >>> 11#i32
      let fl ← lift (F64.fromUnsigned i1)
      let fl1 ← lift inv
      let fl2 ← lift (F64.mul fl fl1)
      Result.ok (fl2, self1)) =
    Result.ok (F64.mul (F64.fromUnsigned (⟨nw.1 >>> 11⟩ : U64)) inv, toPure nw.2) := by
  rcases nw with ⟨next, s1⟩
  simp only [shift11, lift, bind_ok]
  rfl

/-- Exact correspondence for `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#next_f64().` -/
theorem doubleMatch (s : State) :
    BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_f64 (toPure s) =
      Result.ok (unitDouble (nextFloat s).1, toPure (nextFloat s).2) := by
  unfold BauxResearch.Generated.Pure.numpy_pcg.NumpyPcg64.next_f64 nextFloat unitDouble
  rw [wordMatch]
  exact doubleStep (nextWord s)
    (F64.div (F64.literal false 1 1) (F64.literal false 9007199254740992 1))

/-- Box-Muller normal pair semantics for `rust:ennbo 0.4.1 numpy_pcg/NumpyPcg64#standard_normal_pair().` -/
def normalPair (s : State) : Result (F64 × F64 × State) := do
  let (u1Raw, s1) := nextFloat s
  let (u2Raw, s2) := nextFloat s1
  let lo := F64.literal false 1 1000000000000
  let hi := F64.sub (F64.literal false 1 1) lo
  let u1 ← F64.clamp (unitDouble u1Raw) lo hi
  let u2 := unitDouble u2Raw
  let lnU1 ← F64.ln u1
  let rad ← F64.sqrt (F64.mul (F64.literal true 2 1) lnU1)
  let piVal ← F64.pi
  let theta := F64.mul (F64.mul (F64.literal false 2 1) piVal) u2
  let c ← F64.cos theta
  let halfPi := F64.mul (F64.literal false 5 10) piVal
  let sn ← F64.cos (F64.sub theta halfPi)
  Result.ok (F64.mul rad c, F64.mul rad sn, s2)

#print axioms hashmixMatch
#print axioms mixMatch
#print axioms entropyLen
#print axioms poolLen
#print axioms seedLen
#print axioms sobolMatch
#print axioms partsMatch
#print axioms wordMatch
#print axioms halfMatch
#print axioms doubleMatch

end BauxResearch.Semantics.Seed
