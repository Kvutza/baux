import BauxResearch.Generated.Floating.Funs

/-!
Correspondence for the native Rust `u64_f53` extraction. The specification below preserves
both floating-point operations and their rounding: converting the shifted integer and
multiplying it by the exactly represented reciprocal. No integer rewrite of Rust is used.
This theorem concerns the generated function under the local FloatLib bindings. It does not
establish the correctness of the extraction tool or equivalence of the full optimizer.
-/

namespace BauxResearch.Extraction.Conversion

open Aeneas.Std
open FloatLib.Floats.Formats.BinaryInterchange
open BauxResearch.Generated.Floating
open BauxResearch.Extraction.Native

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

def fraction (word : BitVec 64) : F64 :=
  Model.mul (Model.roundRat FloatFormat.binary64 false (word >>> 11).toNat 1)
    (Model.ofBits 0x3ca0000000000000)

theorem reciprocal : random.INV_2P53 =
    Result.ok (Model.ofBits 0x3ca0000000000000 : F64) := by
  unfold random.INV_2P53
  apply congrArg Result.ok
  decide

private theorem shift (word : U64) :
    word >>> 11#i32 = Result.ok (⟨word.bv >>> 11⟩ : U64) := by rfl

/-- All inputs, including both ends of the u64 domain, follow the FloatLib operation sequence. -/
theorem extracted (word : U64) :
    random.u64_f53 word = Result.ok (fraction word.bv) := by
  simp only [random.u64_f53, shift, reciprocal, lift, bind_ok]
  rfl

theorem singleRound (word : U32) :
    (float.single_from word >>= float.single_bits) = Result.ok word := by
  change Aeneas.Std.bind (Result.ok (Model.ofBits word.bv : F32))
    float.single_bits = Result.ok word
  rw [bind_ok]
  rfl

theorem doubleRound (word : U64) :
    (float.double_from word >>= float.double_bits) = Result.ok word := by
  change Aeneas.Std.bind (Result.ok (Model.ofBits word.bv : F64))
    float.double_bits = Result.ok word
  rw [bind_ok]
  rfl

def boxMuller (u1 u2 : F64) : F64 :=
  let r := Model.sqrt (Model.mul (F64.literal true 2 1) (Model.log u1))
  let theta := Model.mul (Model.mul (F64.literal false 2 1) (Model.pi _)) u2
  Model.mul r (Model.cos theta)

/-- Direct extraction of `random::box_muller` lowers `f64::consts::PI` and inherent
`ln`, `sqrt`, `cos` methods to the FloatLib model without external axioms. -/
theorem boxExtracted (u1 u2 : F64) :
    random.box_muller u1 u2 = Result.ok (boxMuller u1 u2) := by
  simp only [random.box_muller, F64.ln, F64.sqrt, F64.pi, F64.cos, lift, bind_ok]
  rfl

theorem zeroValue : (fraction 0).bits = 0 := by decide

theorem upperValue : (fraction 0xffffffffffffffff).bits = 0x3fefffffffffffff := by decide

#print axioms reciprocal
#print axioms extracted
#print axioms singleRound
#print axioms doubleRound
#print axioms boxExtracted
#print axioms zeroValue
#print axioms upperValue

end BauxResearch.Extraction.Conversion
