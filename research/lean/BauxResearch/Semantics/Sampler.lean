import BauxResearch.Semantics.Gaussian
import BauxResearch.Generated.Optimizer.Funs

/-!
# Extracted CUDA sampler word generation

CUDA calls the shared production `gaussian::mix32`, `gaussian::bits` and
`gaussian::normal_bounds` functions. This module connects their extracted
arithmetic to the checked integer model. Metal uses a different generator.
Float acceptance, tables and complete rejection control flow are outside this
correspondence.
-/

namespace BauxResearch.Semantics.Sampler

open Aeneas.Std
open BauxResearch.Generated.Optimizer

private theorem shr16 (value : U32) :
    value >>> 16#i32 = Result.ok (⟨value.bv >>> 16⟩ : U32) := by rfl

private theorem shr15 (value : U32) :
    value >>> 15#i32 = Result.ok (⟨value.bv >>> 15⟩ : U32) := by rfl

private theorem shr32 (value : U64) :
    value >>> 32#i32 = Result.ok (⟨value.bv >>> 32⟩ : U64) := by rfl

/-- Exact extracted mixer correspondence, including checked fixed-width shifts. -/
theorem mixMatch (value : U32) :
    gaussian.mix32 value = Result.ok (⟨Gaussian.mix value.bv⟩ : U32) := by
  simp only [gaussian.mix32, shr16, shr15, lift, bind_ok]
  apply congrArg Result.ok
  apply U32.bv_eq_imp_eq
  simp [Gaussian.mix, Gaussian.shift16, Gaussian.shift15]

def domain (seed key : U64) : Gaussian.Word :=
  seed.bv.zeroExtend 32 ^^^ ((seed.bv >>> 32).zeroExtend 32 * 0xc2b2ae35) ^^^
    (key.bv.zeroExtend 32 * 0x27d4eb2f) ^^^
    ((key.bv >>> 32).zeroExtend 32 * 0x165667b1)

def counter (element : U64) : Gaussian.Word :=
  (element.bv.zeroExtend 32 * 0x9e3779b9) ^^^
    ((element.bv >>> 32).zeroExtend 32 * 0x85ebca6b)

def base (seed key element : U64) (stream : U32) : Gaussian.Word :=
  domain seed key ^^^ counter element ^^^ (stream.bv * 0xfd7046c5)

/-- Every cast, shift, XOR and wrapping product matches the production expression. -/
theorem bitsRaw (seed key element : U64) (attempt stream : U32) :
    gaussian.bits seed key element attempt stream = Result.ok (⟨Gaussian.mix
      (domain seed key ^^^ counter element ^^^ (attempt.bv * 0xd3a2646c) ^^^
        (stream.bv * 0xfd7046c5))⟩ : U32) := by
  simp only [gaussian.bits, shr32, lift, bind_ok, mixMatch]
  apply congrArg Result.ok
  apply U32.bv_eq_imp_eq
  simp [domain, counter, UScalar.cast]

/-- The fixed stream selects one input coset; attempt variation follows the proved model. -/
theorem bitsMatch (seed key element : U64) (attempt stream : U32) :
    gaussian.bits seed key element attempt stream = Result.ok
      (⟨Gaussian.mix (Gaussian.input (base seed key element stream) attempt.bv)⟩ : U32) := by
  rw [bitsRaw]
  apply congrArg (fun word : Gaussian.Word ↦ Result.ok (⟨Gaussian.mix word⟩ : U32))
  unfold Gaussian.input base
  simp only [BitVec.xor_assoc]
  rw [BitVec.xor_comm (attempt.bv * 0xd3a2646c) (stream.bv * 0xfd7046c5)]

/-- The extracted table-entry gate establishes the four normal-branch premises. -/
theorem boundsMatch (t0 t192 t129 t221 : U32) :
    gaussian.normal_bounds t0 t192 t129 t221 = Result.ok true ↔
      0 < t0.val ∧ 1753845952 < t192.val ∧ 787275391 < t129.val ∧ 1408362973 < t221.val := by
  simp only [gaussian.normal_bounds, UScalar.lt_equiv]
  split <;> simp_all
  split <;> simp_all
  split <;> simp_all

/-- Conditional finite reachability of the normal branch through the extracted Rust functions. -/
theorem normalReach (thresholds : Nat → U32) (seed key element : U64)
    (start stream : U32)
    (bounds : gaussian.normal_bounds
      (thresholds 0) (thresholds 192) (thresholds 129) (thresholds 221) = Result.ok true) :
    ∃ steps : ℕ, steps < 2 ^ 32 ∧ ∃ word : U32,
      gaussian.bits seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32) stream =
        Result.ok word ∧
      Gaussian.accepted (fun index ↦ (thresholds index).val) word.bv := by
  obtain ⟨zero, first, second, third⟩ := (boundsMatch _ _ _ _).mp bounds
  obtain ⟨steps, bound, accepted⟩ :=
    Gaussian.normalReach (fun index ↦ (thresholds index).val)
      (base seed key element stream) start.bv zero first second third
  refine ⟨steps, bound, ⟨Gaussian.mix (Gaussian.input (base seed key element stream)
    (start.bv + BitVec.ofNat 32 steps))⟩, ?_, accepted⟩
  simpa using bitsMatch seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32) stream

private theorem streamXor (x y a b : Gaussian.Word) :
    (x ^^^ a ^^^ y) ^^^ a ^^^ b = x ^^^ b ^^^ y := by
  ext i hi
  interval_cases i <;> simp [Bool.xor_comm, Bool.xor_left_comm]

/-- Stream 2 extracted word generation matches `Gaussian.other` applied to stream 1's input. -/
theorem otherMatch (seed key element : U64) (attempt : U32) :
    gaussian.bits seed key element attempt 2#u32 = Result.ok
      (⟨Gaussian.other (Gaussian.input (base seed key element 1#u32) attempt.bv)⟩ : U32) := by
  rw [bitsMatch]
  apply congrArg (fun word : Gaussian.Word ↦ Result.ok (⟨word⟩ : U32))
  change Gaussian.mix (domain seed key ^^^ counter element ^^^
    ((2 : Gaussian.Word) * 0xfd7046c5) ^^^ ((attempt.bv : Gaussian.Word) * 0xd3a2646c)) =
    Gaussian.mix ((domain seed key ^^^ counter element ^^^
      (0xfd7046c5 : Gaussian.Word) ^^^ ((attempt.bv : Gaussian.Word) * 0xd3a2646c)) ^^^
      0xfd7046c5 ^^^ (2 * 0xfd7046c5))
  apply congrArg Gaussian.mix
  exact (streamXor (domain seed key ^^^ counter element)
    ((attempt.bv : Gaussian.Word) * 0xd3a2646c) 0xfd7046c5 (2 * 0xfd7046c5)).symm

/-- Conditional finite reachability of the tail word pair through the extracted Rust functions. -/
theorem tailReach (passes : Gaussian.Word → Gaussian.Word → Prop)
    (seed key element : U64) (start : U32)
    (first : passes 0xffffffff 0x39ef859b) (second : passes 0xfffffffe 0x49c079da)
    (third : passes 0xfffffffc 0x0a72fd74) (fourth : passes 0xfffffffb 0x20394f67) :
    ∃ steps : ℕ, steps < 2 ^ 32 ∧ ∃ word1 word2 : U32,
      gaussian.bits seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32) 1#u32 =
        Result.ok word1 ∧
      gaussian.bits seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32) 2#u32 =
        Result.ok word2 ∧
      passes word1.bv word2.bv := by
  obtain ⟨steps, bound, passed⟩ :=
    Gaussian.tailReach passes (base seed key element 1#u32) start.bv first second third fourth
  let raw := Gaussian.input (base seed key element 1#u32) (start.bv + BitVec.ofNat 32 steps)
  refine ⟨steps, bound, ⟨Gaussian.mix raw⟩, ⟨Gaussian.other raw⟩, ?_, ?_, passed⟩
  · simpa [raw] using bitsMatch seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32) 1#u32
  · simpa [raw] using otherMatch seed key element (⟨start.bv + BitVec.ofNat 32 steps⟩ : U32)

#print axioms mixMatch
#print axioms bitsMatch
#print axioms boundsMatch
#print axioms normalReach
#print axioms otherMatch
#print axioms tailReach

end BauxResearch.Semantics.Sampler
