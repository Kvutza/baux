import BauxResearch.Generated.Hash
import BauxResearch.Generated.Pure.Funs
import BauxResearch.Semantics.Hash

/-!
# Extracted ENN 0.4.1 `hash.rs` Correspondence

Covers all 6 reachable functions in `crates/ennbo/src/hash.rs`:
- `rust:ennbo 0.4.1 hash/splitmix64().` (`splitmixMatch`, `pureSplitmix`)
- `rust:ennbo 0.4.1 hash/u64_to_f53().` (`pureUniform`)
- `rust:ennbo 0.4.1 hash/box_muller().` (`pureBox`)
- `rust:ennbo 0.4.1 hash/normal_for_seed_index_metric().` (`pureIndex`)
- `rust:ennbo 0.4.1 hash/normal_for_seed_query_metric().` (`pureQuery`)
- `rust:ennbo 0.4.1 hash/unique_index_inverse().` (`Semantics.Hash.inverseLen`)
-/

namespace BauxResearch.Extraction.Hash

open Aeneas.Std Aeneas.Std.Result
open BauxResearch.Extraction.Native
open BauxResearch.Semantics.Hash

private theorem shift30 (value : U64) :
    value >>> 30#i32 = Result.ok (⟨value.bv >>> 30⟩ : U64) := by
  rfl

private theorem shift27 (value : U64) :
    value >>> 27#i32 = Result.ok (⟨value.bv >>> 27⟩ : U64) := by
  rfl

private theorem shift31 (value : U64) :
    value >>> 31#i32 = Result.ok (⟨value.bv >>> 31⟩ : U64) := by
  rfl

private theorem shift11 (value : U64) :
    value >>> 11#i32 = Result.ok (⟨value.bv >>> 11⟩ : U64) := by
  rfl

theorem splitmixMatch (value : U64) :
    BauxResearch.Generated.Hash.splitmix64 value =
      Result.ok (⟨BauxResearch.Semantics.Hash.splitmix value.bv⟩ : U64) := by
  simp only [BauxResearch.Generated.Hash.splitmix64,
    BauxResearch.Generated.Hash.golden,
    BauxResearch.Generated.Hash.multiplier1,
    BauxResearch.Generated.Hash.multiplier2,
    shift30, shift27, shift31, lift, bind_ok]
  apply congrArg Result.ok
  apply U64.bv_eq_imp_eq
  rfl

private theorem goldenOk :
    BauxResearch.Generated.Pure.hash.SM64_GOLDEN_RATIO = (⟨golden⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.SM64_GOLDEN_RATIO
  rfl

private theorem firstOk :
    BauxResearch.Generated.Pure.hash.SM64_MULTIPLIER_1 = (⟨multiplier1⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.SM64_MULTIPLIER_1
  rfl

private theorem secondOk :
    BauxResearch.Generated.Pure.hash.SM64_MULTIPLIER_2 = (⟨multiplier2⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.SM64_MULTIPLIER_2
  rfl

private theorem xorOk :
    BauxResearch.Generated.Pure.hash.SM64_XOR_OFFSET = (⟨xorOffset⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.SM64_XOR_OFFSET
  rfl

private theorem primeOk :
    BauxResearch.Generated.Pure.hash.SEED_PRIME = (⟨seedPrime⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.SEED_PRIME
  rfl

private theorem saltOk :
    BauxResearch.Generated.Pure.hash.normal_for_seed_query_metric.QUERY_SALT =
      (⟨querySalt⟩ : U64) := by
  unfold BauxResearch.Generated.Pure.hash.normal_for_seed_query_metric.QUERY_SALT
  rfl

private theorem wordAdd (a b : BitVec 64) :
    core.num.U64.wrapping_add (⟨a⟩ : U64) (⟨b⟩ : U64) = (⟨a + b⟩ : U64) := by
  rfl

private theorem wordMul (a b : BitVec 64) :
    core.num.U64.wrapping_mul (⟨a⟩ : U64) (⟨b⟩ : U64) = (⟨a * b⟩ : U64) := by
  rfl

private theorem wordXor (a b : BitVec 64) :
    (⟨a⟩ : U64) ^^^ (⟨b⟩ : U64) = (⟨a ^^^ b⟩ : U64) := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 hash/splitmix64().` -/
theorem pureSplitmix (value : U64) :
    BauxResearch.Generated.Pure.hash.splitmix64 value =
      Result.ok (⟨splitmix value.bv⟩ : U64) := by
  cases value with | mk vb =>
  simp only [BauxResearch.Generated.Pure.hash.splitmix64,
    goldenOk, firstOk, secondOk, wordAdd, wordMul, wordXor,
    shift30, shift27, shift31, lift, bind_ok, splitmix]

/-- Semantic 53-bit fraction conversion for `rust:ennbo 0.4.1 hash/u64_to_f53().` -/
def uniformPure (x : BitVec 64) : Result F64 := do
  let inv ← BauxResearch.Generated.Pure.hash.INV_2P53
  Result.ok (F64.mul (F64.fromUnsigned (⟨x >>> 11⟩ : U64)) inv)

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 hash/u64_to_f53().` -/
theorem pureUniform (x : U64) :
    BauxResearch.Generated.Pure.hash.u64_to_f53 x =
      uniformPure x.bv := by
  cases x with | mk xb =>
  unfold BauxResearch.Generated.Pure.hash.u64_to_f53 uniformPure
  rw [shift11, bind_ok]
  dsimp only [lift]
  rw [bind_ok]

/-- Semantic Box-Muller kernel for `rust:ennbo 0.4.1 hash/box_muller().` -/
def boxPure (u1 u2 : F64) : Result F64 := do
  let fl ← F64.ln u1
  let fl1 ← lift (F64.mul (F64.literal true 2 1) fl)
  let fl2 ← F64.sqrt fl1
  let fl3 ← F64.pi
  let fl4 ← lift (F64.mul (F64.literal false 2 1) fl3)
  let fl5 ← lift (F64.mul fl4 u2)
  let fl6 ← F64.cos fl5
  Result.ok (F64.mul fl2 fl6)

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 hash/box_muller().` -/
theorem pureBox (u1 u2 : F64) :
    BauxResearch.Generated.Pure.hash.box_muller u1 u2 =
      boxPure u1 u2 := by
  rfl

/-- Shared SplitMix64 + Box-Muller normal draw from combined 64-bit seed word. -/
def combinedPure (combined : BitVec 64) : Result F64 := do
  let u1 ← uniformPure (splitmix combined)
  let u2 ← uniformPure (splitmix (combined ^^^ xorOffset))
  let u1c ← F64.clamp u1
    BauxResearch.Generated.Pure.hash.CLIP_MIN
    BauxResearch.Generated.Pure.hash.CLIP_MAX
  boxPure u1c u2

private theorem signedCast (x : I64) :
    IScalar.hcast .U64 x = (⟨BitVec.signExtend 64 x.bv⟩ : U64) := by
  rfl

private theorem usizeCast (x : Usize) :
    UScalar.cast .U64 x = (⟨BitVec.setWidth 64 x.bv⟩ : U64) := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 hash/normal_for_seed_index_metric().` -/
theorem pureIndex (seed : U64) (idx : I64) (metric : Usize) :
    BauxResearch.Generated.Pure.hash.normal_for_seed_index_metric seed idx metric =
      combinedPure ((seed.bv * seedPrime + BitVec.signExtend 64 idx.bv) * seedPrime +
        BitVec.setWidth 64 metric.bv) := by
  cases seed with | mk sb =>
  unfold BauxResearch.Generated.Pure.hash.normal_for_seed_index_metric combinedPure
  rw [signedCast, usizeCast, primeOk]
  dsimp only [lift]
  rw [bind_ok, bind_ok, wordMul, bind_ok, wordAdd, bind_ok, wordMul, bind_ok, wordAdd, bind_ok,
    pureSplitmix, bind_ok, xorOk, wordXor, bind_ok, pureSplitmix, bind_ok,
    pureUniform, pureUniform]
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 hash/normal_for_seed_query_metric().` -/
theorem pureQuery (seed : U64) (query metric : Usize) :
    BauxResearch.Generated.Pure.hash.normal_for_seed_query_metric seed query metric =
      combinedPure (((seed.bv * seedPrime + BitVec.setWidth 64 query.bv) * seedPrime + querySalt) +
        BitVec.setWidth 64 metric.bv) := by
  cases seed with | mk sb =>
  unfold BauxResearch.Generated.Pure.hash.normal_for_seed_query_metric combinedPure
  rw [usizeCast, usizeCast, primeOk, saltOk]
  dsimp only [lift]
  rw [bind_ok, bind_ok, wordMul, bind_ok, wordAdd, bind_ok, wordMul, bind_ok,
    wordAdd, bind_ok, wordAdd, bind_ok, pureSplitmix, bind_ok, xorOk, wordXor, bind_ok,
    pureSplitmix, bind_ok, pureUniform, pureUniform]
  rfl

#print axioms splitmixMatch
#print axioms pureSplitmix
#print axioms pureUniform
#print axioms pureBox
#print axioms pureIndex
#print axioms pureQuery

end BauxResearch.Extraction.Hash
