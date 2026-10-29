import BauxResearch.Extraction.Native

namespace BauxResearch.Extraction.NativeTest

open FloatLib.Floats.Formats.BinaryInterchange
open BauxResearch.Extraction.Native
open Aeneas.Std

set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

theorem singleBits (word : U32) :
    (F32.fromBits word >>= F32.toBits) = Result.ok word := by
  change Aeneas.Std.bind (Result.ok (Model.ofBits word.bv : F32)) F32.toBits = Result.ok word
  rw [bind_ok]
  rfl

theorem doubleBits (word : U64) :
    (F64.fromBits word >>= F64.toBits) = Result.ok word := by
  change Aeneas.Std.bind (Result.ok (Model.ofBits word.bv : F64)) F64.toBits = Result.ok word
  rw [bind_ok]
  rfl

theorem cloneExact (word : F64) : F64.clone word = Result.ok word := by rfl

/-- IEEE equality identifies the two zero encodings; negation keeps the sign bit. -/
theorem zeroSign :
    F64.eq (F64.literal false 0 1) (F64.literal true 0 1) = true ∧
    (F64.neg (F64.literal false 0 1)).bits = 0x8000000000000000 := by decide

/-- A NaN is unequal to itself and unordered in each non-strict direction. -/
theorem nanOrder :
    let nan : F64 := Model.ofBits 0x7ff8000000000001
    F64.eq nan nan = false ∧ F64.ne nan nan = true ∧
    F64.le nan nan = false ∧ F64.ge nan nan = false := by decide

/-- Non-finite literals preserve standard IEEE positive/negative infinity and quiet NaN encodings. -/
theorem nonfiniteLiterals :
    F32.posInf.bits = 0x7f800000 ∧
    F32.negInf.bits = 0xff800000 ∧
    F32.nan.bits = 0x7fc00000 ∧
    F64.posInf.bits = 0x7ff0000000000000 ∧
    F64.negInf.bits = 0xfff0000000000000 ∧
    F64.nan.bits = 0x7ff8000000000000 := by decide

/-- Integer conversion rounds ties to even at the binary64 significand boundary. -/
theorem castTie :
    (F64.fromUnsigned (9007199254740993#u64)).bits = 0x4340000000000000 := by decide

/-- Signed conversion handles the minimum signed integer without host overflow. -/
theorem signedLimit :
    (F64.fromSigned ((-9223372036854775808)#i64)).bits = 0xc3e0000000000000 := by decide

/-- Float-to-unsigned conversion saturates non-finites, signed zero, and out-of-range values. -/
theorem unsignedSaturate :
    F64.toUnsigned .U32 (F64.literal false 0 1) = 0#u32 ∧
    F64.toUnsigned .U32 (F64.literal true 0 1) = 0#u32 ∧
    F64.toUnsigned .U32 (F64.literal false 9 10) = 0#u32 ∧
    F64.toUnsigned .U32 (F64.literal true 9 10) = 0#u32 ∧
    F64.toUnsigned .U32 (F64.literal false 429 10) = 42#u32 ∧
    F64.toUnsigned .U32 (F64.literal true 429 10) = 0#u32 ∧
    F64.toUnsigned .U32 (F64.literal false 4294967295 1) = 4294967295#u32 ∧
    F64.toUnsigned .U32 (F64.literal false 1000000000000000 1) = 4294967295#u32 ∧
    F64.toUnsigned .U32 (F64.literal true 1000000000000000 1) = 0#u32 ∧
    F64.toUnsigned .U32 F64.posInf = 0xffffffff#u32 ∧
    F64.toUnsigned .U32 F64.negInf = 0#u32 ∧
    F64.toUnsigned .U32 F64.nan = 0#u32 := by decide

/-- Float-to-signed conversion saturates non-finites, signed zero, and truncates within limits. -/
theorem signedSaturate :
    F64.toSigned .I32 (F64.literal false 0 1) = 0#i32 ∧
    F64.toSigned .I32 (F64.literal true 0 1) = 0#i32 ∧
    F64.toSigned .I32 (F64.literal false 9 10) = 0#i32 ∧
    F64.toSigned .I32 (F64.literal true 9 10) = 0#i32 ∧
    F64.toSigned .I32 (F64.literal false 429 10) = 42#i32 ∧
    F64.toSigned .I32 (F64.literal true 429 10) = (-42)#i32 ∧
    F64.toSigned .I32 (F64.literal false 2147483647 1) = 2147483647#i32 ∧
    F64.toSigned .I32 (F64.literal true 2147483648 1) = (-2147483648)#i32 ∧
    F64.toSigned .I32 (F64.literal false 1000000000000000 1) = 2147483647#i32 ∧
    F64.toSigned .I32 (F64.literal true 1000000000000000 1) = (-2147483648)#i32 ∧
    F64.toSigned .I32 F64.posInf = 0x7fffffff#i32 ∧
    F64.toSigned .I32 F64.negInf = (-2147483648)#i32 ∧
    F64.toSigned .I32 F64.nan = 0#i32 := by decide

/--
Discriminating regression distinguishing Rust `std::ops::Rem` (`%` / `fmod`)
from IEEE 754-2019 §5.3.1 `remainder`.

On 5.5 % 2.0:
- The exact quotient is 2.75.
- Rust truncates the quotient toward zero (`q = 2`), giving `5.5 - 2 * 2.0 = 1.5`.
- IEEE 754 §5.3.1 rounds the quotient to the nearest even integer (`q = 3`), giving `5.5 - 3 * 2.0 = -0.5`.
- Consequently, Rust floating `%` and FloatLib's `Model.remainder` are strictly unequal.
-/
theorem remRegression :
    let x64 := F64.literal false 11 2   -- 5.5
    let y64 := F64.literal false 2 1    -- 2.0
    let x32 := F32.literal false 11 2   -- 5.5f32
    let y32 := F32.literal false 2 1    -- 2.0f32
    -- Rust remainder computes +1.5
    (F64.rem x64 y64).bits = (F64.literal false 3 2).bits ∧
    (F32.rem x32 y32).bits = (F32.literal false 3 2).bits ∧
    -- FloatLib Model.remainder computes -0.5
    (Model.remainder x64 y64).bits = (F64.literal true 1 2).bits ∧
    (Model.remainder x32 y32).bits = (F32.literal true 1 2).bits ∧
    -- Rust rem and IEEE remainder disagree on this input
    (F64.rem x64 y64).bits ≠ (Model.remainder x64 y64).bits ∧
    (F32.rem x32 y32).bits ≠ (Model.remainder x32 y32).bits := by decide

/--
Rust floating remainder sign, signed-zero, and magnitude edge cases.
Rem preserves the sign of the dividend and handles smaller-magnitude dividends.
-/
theorem remEdges :
    let xPos := F64.literal false 11 2   -- 5.5
    let xNeg := F64.literal true 11 2    -- -5.5
    let yPos := F64.literal false 2 1    -- 2.0
    let yNeg := F64.literal true 2 1     -- -2.0
    let resPos := F64.literal false 3 2  -- 1.5
    let resNeg := F64.literal true 3 2   -- -1.5
    -- Signed combinations: sign depends only on dividend
    (F64.rem xNeg yPos).bits = resNeg.bits ∧
    (F64.rem xPos yNeg).bits = resPos.bits ∧
    (F64.rem xNeg yNeg).bits = resNeg.bits ∧
    -- Zero dividends yield signed zeros matching dividend sign
    (F64.rem (F64.literal false 0 1) yPos).bits = 0x0000000000000000 ∧
    (F64.rem (F64.literal true 0 1) yPos).bits = 0x8000000000000000 ∧
    -- Exact zero remainder preserves dividend sign
    (F64.rem (F64.literal false 4 1) yPos).bits = 0x0000000000000000 ∧
    (F64.rem (F64.literal true 4 1) yPos).bits = 0x8000000000000000 ∧
    -- Dividend magnitude smaller than divisor returns dividend unchanged
    (F64.rem (F64.literal false 1 1) yPos).bits = (F64.literal false 1 1).bits ∧
    (F64.rem (F64.literal true 1 1) yPos).bits = (F64.literal true 1 1).bits := by decide

/--
Rust floating remainder non-finite cases:
- Division by zero yields canonical NaN.
- Infinite dividend yields canonical NaN.
- Finite dividend modulo infinity yields dividend unchanged.
-/
theorem remNonfinite :
    let finiteVal := F64.literal false 5 1
    (F64.rem finiteVal (F64.literal false 0 1)).bits = F64.nan.bits ∧
    (F64.rem finiteVal (F64.literal true 0 1)).bits = F64.nan.bits ∧
    (F64.rem F64.posInf finiteVal).bits = F64.nan.bits ∧
    (F64.rem F64.negInf finiteVal).bits = F64.nan.bits ∧
    (F64.rem finiteVal F64.posInf).bits = finiteVal.bits ∧
    (F64.rem finiteVal F64.negInf).bits = finiteVal.bits := by decide

/-!
## Source Semantic Correspondence & Platform Boundary

The Lean extraction binds Rust source calls to mathematical specifications in FloatLib.
The relation between source code and Lean models falls into three categories:

1. **IEEE 754-2008 Mandated Correctly-Rounded Operations:**
   - Basic arithmetic (`add`, `sub`, `mul`, `div`) and square root (`sqrt`) are mandated
     by IEEE 754-2008 §5.4.1 to be correctly rounded to the nearest representable float
     (roundTiesToEven, 0.5 ULP). FloatLib's `Model.add`, `Model.sub`, `Model.mul`,
     `Model.div`, and `Model.sqrt` implement this exact specification.

2. **Language-Specified Truncating Arithmetic & Saturating Casts:**
   - Rust `std::ops::Rem` (`%`) is specified as truncating quotient remainder `fmod`
     (`x - trunc(x/y)*y`), implemented exactly by `BauxResearch.Extraction.Native.fmod`.
   - Float-to-int casts (`as u32`, `as i32`, etc.) follow Rust RFC 2497 saturating conversions,
     implemented exactly by `Native.toUnsigned` and `Native.toSigned`.

3. **Idealized Transcendentals vs. Target Platform libm (Open Boundary):**
   - Transcendentals (`ln`, `cos`, `exp`, `powi`, `powf`) bind definitionally to FloatLib's
     idealized mathematical real-analytic models (`Model.log`, `Model.cos`, etc.).
   - However, IEEE 754-2008 §9.2 does not mandate correctly-rounded transcendentals, and
     standard host C math libraries (`libm` on Darwin, glibc, musl) exhibit 0.5–1.0 ULP
     variations across platforms and architectures.
   - **Open Boundary:** Definitional wrapper equality (`wrapperBindings`) confirms that our
     extraction faithfully targets FloatLib, but does NOT prove that a specific compiled
     native binary artifact using host `libm` has bit-identical outputs. That artifact-level
     correspondence remains an open platform boundary, to be established under explicit
     runtime/libm assumptions in downstream task `baux-x0b.2.9.49.3`.
-/

/--
Wrapper binding consistency: verifies that our extraction wrappers route directly
to FloatLib's mathematical models without intermediate translation artifacts.
(Note: this is an internal model consistency check, not a proof of host libm bit-equivalence.)
-/
theorem wrapperBindings (x y z : F64) (i : I32) (a b c : F32) :
    F64.isNan x = Result.ok (Model.isNaN x) ∧
    F64.isInfinite x = Result.ok (Model.isInf x) ∧
    F64.isFinite x = Result.ok (Model.isFinite x) ∧
    F64.abs x = Result.ok (Model.abs x) ∧
    F64.min x y = Result.ok (Model.minimumNumber x y) ∧
    F64.max x y = Result.ok (Model.maximumNumber x y) ∧
    F64.clamp x y z = clamp x y z ∧
    F64.sqrt x = Result.ok (Model.sqrt x) ∧
    F64.exp x = Result.ok (Model.exp x) ∧
    F64.ln x = Result.ok (Model.log x) ∧
    F64.cos x = Result.ok (Model.cos x) ∧
    F64.powi x i = Result.ok (Model.powInt x i.val) ∧
    F64.powf x y = Result.ok (Model.pow x y) ∧
    F64.pi = Result.ok (Model.pi _) ∧
    F32.isNan a = Result.ok (Model.isNaN a) ∧
    F32.isInfinite a = Result.ok (Model.isInf a) ∧
    F32.isFinite a = Result.ok (Model.isFinite a) ∧
    F32.abs a = Result.ok (Model.abs a) ∧
    F32.min a b = Result.ok (Model.minimumNumber a b) ∧
    F32.max a b = Result.ok (Model.maximumNumber a b) ∧
    F32.clamp a b c = clamp a b c ∧
    F32.sqrt a = Result.ok (Model.sqrt a) ∧
    F32.exp a = Result.ok (Model.exp a) ∧
    F32.ln a = Result.ok (Model.log a) ∧
    F32.cos a = Result.ok (Model.cos a) ∧
    F32.powi a i = Result.ok (Model.powInt a i.val) ∧
    F32.powf a b = Result.ok (Model.pow a b) ∧
    F32.pi = Result.ok (Model.pi _) := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Rust `clamp` bounds values to `[lo, hi]` and propagates NaN inputs. -/
theorem clampCases :
    F64.clamp (F64.literal true 2 1) (F64.literal false 0 1) (F64.literal false 1 1) =
      Result.ok (F64.literal false 0 1) ∧
    F64.clamp (F64.literal false 1 2) (F64.literal false 0 1) (F64.literal false 1 1) =
      Result.ok (F64.literal false 1 2) ∧
    F64.clamp (F64.literal false 3 1) (F64.literal false 0 1) (F64.literal false 1 1) =
      Result.ok (F64.literal false 1 1) ∧
    F64.clamp F64.nan (F64.literal false 0 1) (F64.literal false 1 1) =
      Result.ok F64.nan := by
  refine ⟨rfl, rfl, rfl, rfl⟩

/-- Binary32 multiplication retains the least positive subnormal. -/
theorem tinyProduct :
    (F32.mul (Model.ofBits 1) (F32.literal false 1 1)).bits = 1 := by decide

/-- Narrowing rounds the halfway case toward the even binary32 significand. -/
theorem narrowTie :
    (F32.fromDouble (F64.literal false 16777217 16777216)).bits = 0x3f800000 := by decide

/-- Shared-reference dynamic dispatch selects the exact concrete vtable instance and receiver. -/
theorem dynDispatch {Trait : Type u → Type v} {Self : Type u} {α β : Type w}
    (method : ∀ {T}, Trait T → T → α → Result β)
    (inst : Trait Self) (val : Self) (arg : α) :
    (fun (d : Aeneas.Std.Dyn Trait) (x0 : α) => method d.inst d.value x0)
      (Aeneas.Std.Dyn.mk Self inst val) arg =
    method inst val arg := rfl

/--
Mutable-reference dynamic dispatch followed by `dynBack` selects the concrete
vtable instance, updates the receiver in place, and writes back the exact
updated concrete state on `Result.ok`.
-/
theorem dynMutate {Trait : Type u → Type v} {Self : Type u} {α : Type w}
    (method : ∀ {T}, Trait T → T → Result (α × T))
    (inst : Trait Self) (init val val' : Self) (out : α)
    (h : method inst val = Result.ok (out, val')) :
    Aeneas.Std.bind
      ((fun (d : Aeneas.Std.Dyn Trait) =>
          Aeneas.Std.bind (method d.inst d.value)
            (fun (r0, value') => Result.ok (r0, { d with value := value' })))
        (Aeneas.Std.Dyn.mk Self inst val))
      (fun (r0, d') => Result.ok (r0, dynBack init d')) =
    Result.ok (out, val') := by
  simp [h]

/-- Mutable-reference dynamic dispatch propagates `Result.fail` unchanged. -/
theorem dynFail {Trait : Type u → Type v} {Self : Type u} {α : Type w}
    (method : ∀ {T}, Trait T → T → Result (α × T))
    (inst : Trait Self) (init val : Self) (err : Error)
    (h : method inst val = Result.fail err) :
    Aeneas.Std.bind
      ((fun (d : Aeneas.Std.Dyn Trait) =>
          Aeneas.Std.bind (method d.inst d.value)
            (fun (r0, value') => Result.ok (r0, { d with value := value' })))
        (Aeneas.Std.Dyn.mk Self inst val))
      (fun (r0, d') => Result.ok (r0, dynBack init d')) =
    Result.fail err := by
  simp [h]

#print axioms zeroSign
#print axioms singleBits
#print axioms doubleBits
#print axioms cloneExact
#print axioms nanOrder
#print axioms nonfiniteLiterals
#print axioms castTie
#print axioms signedLimit
#print axioms unsignedSaturate
#print axioms signedSaturate
#print axioms remRegression
#print axioms remEdges
#print axioms remNonfinite
#print axioms wrapperBindings
#print axioms clampCases
#print axioms tinyProduct
#print axioms narrowTie
#print axioms dynDispatch
#print axioms dynMutate
#print axioms dynFail

end BauxResearch.Extraction.NativeTest
