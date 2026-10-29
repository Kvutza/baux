module
public import Aeneas
public import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.ExpLog
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.Trig
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.Pi.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.Power
public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Model.ExactValue

@[expose] public section

/-!
Executable bindings for the local Aeneas floating-point extension. Arithmetic uses FloatLib's
nearest-even operations. Comparisons use IEEE numerical ordering rather than equality of words.
Casts, non-finite literals, remainder, square root and transcendental functions are bound
directly to FloatLib without axioms or sorry.
-/

namespace BauxResearch.Extraction.Native

open FloatLib.Floats.Formats.BinaryInterchange
open Aeneas.Std

abbrev F32 := Model FloatFormat.binary32
abbrev F64 := Model FloatFormat.binary64

def unsigned (fmt : FloatFormat) {ty : UScalarTy} (x : UScalar ty) : Model fmt :=
  Model.roundRat fmt false x.val 1

def signed (fmt : FloatFormat) {ty : IScalarTy} (x : IScalar ty) : Model fmt :=
  Model.roundRat fmt (decide (x.val < 0)) x.val.natAbs 1

def toUnsigned {fmt : FloatFormat} (tgt : UScalarTy) (x : Model fmt) : UScalar tgt :=
  let w := tgt.numBits
  let maxNat : Nat := 2^w - 1
  let maxInt : Int := (maxNat : Int)
  match Model.exactValue x with
  | .finite exact =>
      let n := Model.roundDyadicToInt .towardZero exact
      if n ≤ 0 then ⟨BitVec.ofNat w 0⟩
      else if n ≥ maxInt then ⟨BitVec.ofNat w maxNat⟩
      else ⟨BitVec.ofNat w n.toNat⟩
  | .infinity true => ⟨BitVec.ofNat w 0⟩
  | .infinity false => ⟨BitVec.ofNat w maxNat⟩
  | .nan .. => ⟨BitVec.ofNat w 0⟩

def toSigned {fmt : FloatFormat} (tgt : IScalarTy) (x : Model fmt) : IScalar tgt :=
  let w := tgt.numBits
  let minVal : Int := -2^(w - 1)
  let maxVal : Int := 2^(w - 1) - 1
  match Model.exactValue x with
  | .finite exact =>
      let n := Model.roundDyadicToInt .towardZero exact
      if n ≤ minVal then ⟨BitVec.ofInt w minVal⟩
      else if n ≥ maxVal then ⟨BitVec.ofInt w maxVal⟩
      else ⟨BitVec.ofInt w n⟩
  | .infinity true => ⟨BitVec.ofInt w minVal⟩
  | .infinity false => ⟨BitVec.ofInt w maxVal⟩
  | .nan .. => ⟨BitVec.ofInt w 0⟩

def equal {fmt : FloatFormat} (x y : Model fmt) : Bool :=
  Model.compare x y == some .eq

def less {fmt : FloatFormat} (x y : Model fmt) : Bool :=
  Model.compare x y == some .lt

def lessEqual {fmt : FloatFormat} (x y : Model fmt) : Bool :=
  match Model.compare x y with
  | some .lt | some .eq => true
  | _ => false

def clamp {fmt : FloatFormat} (x lo hi : Model fmt) : Result (Model fmt) :=
  if lessEqual lo hi then
    if less x lo then .ok lo
    else if less hi x then .ok hi
    else .ok x
  else .fail .panic

/--
Dyadic truncating remainder (`fmodDyadic`).

Computes `dividend - trunc(|dividend| / |divisor|) * |divisor|` with the dividend's sign.
For finite non-zero dyadics with `divisor.significand ≠ 0`, this operation is mathematically
exact without any rounding error.
-/
def fmodDyadic (dividend divisor : FloatLib.Numerics.Dyadic) : FloatLib.Numerics.Dyadic :=
  if dividend.significand == 0 then
    dividend
  else if divisor.exponent ≤ dividend.exponent then
    let shift := Int.toNat (dividend.exponent - divisor.exponent)
    let remSig :=
      (dividend.significand % divisor.significand *
        FloatLib.Numerics.modularPow 2 shift divisor.significand) % divisor.significand
    { negative := dividend.negative, significand := remSig, exponent := divisor.exponent }
  else
    let shift := Int.toNat (divisor.exponent - dividend.exponent)
    if dividend.significand.log2 < shift then
      dividend
    else
      let scaledDivisor := Nat.shiftLeft divisor.significand shift
      let remSig := dividend.significand % scaledDivisor
      { negative := dividend.negative, significand := remSig, exponent := dividend.exponent }

/--
Floating-point remainder matching Rust `std::ops::Rem` (`%`) and POSIX `fmod`.

Unlike IEEE 754-2019 §5.3.1 `remainder` (which uses round-to-nearest-even quotient `x - round(x/y)*y`),
Rust floating remainder computes `x - trunc(x/y)*y`. The sign of the result matches the dividend.
Special cases conform to IEEE 754 and Rust semantics:
- `x % 0.0` or `inf % y` yields `canonicalNaN`.
- `x % inf` for finite `x` yields `x`.
- `NaN` operands propagate quiet NaNs.
- Zero results preserve the sign of the dividend.
-/
def fmod {fmt : FloatFormat} (dividend divisor : Model fmt) : Model fmt :=
  match Model.exactValue dividend, Model.exactValue divisor with
  | .nan _ true _, _ =>
      Model.quietNaN dividend
  | _, .nan _ true _ =>
      Model.quietNaN divisor
  | .nan _ false _, _ =>
      Model.quietNaN dividend
  | _, .nan _ false _ =>
      Model.quietNaN divisor
  | .infinity _, _ =>
      Model.canonicalNaN fmt
  | .finite _, .infinity _ =>
      dividend
  | .finite exactDividend, .finite exactDivisor =>
      if exactDivisor.significand == 0 then
        Model.canonicalNaN fmt
      else
        let exact := fmodDyadic exactDividend exactDivisor
        if exact.significand == 0 then
          Model.zero fmt exactDividend.negative
        else
          Model.roundDyadic fmt exact

namespace F32

def clone (x : F32) : Result F32 := .ok x
def fromBits (x : U32) : Result F32 := .ok (Model.ofBits x.bv)
def toBits (x : F32) : Result U32 := .ok ⟨x.bits⟩

def literal (sign : Bool) (num den : Nat) : F32 :=
  Model.roundRat FloatFormat.binary32 sign num den

def posInf : F32 := Model.posInf FloatFormat.binary32
def negInf : F32 := Model.negInf FloatFormat.binary32
def nan : F32 := Model.canonicalNaN FloatFormat.binary32

def fromUnsigned {ty : UScalarTy} (x : UScalar ty) : F32 := unsigned _ x
def fromSigned {ty : IScalarTy} (x : IScalar ty) : F32 := signed _ x
def toUnsigned (tgt : UScalarTy) (x : F32) : UScalar tgt :=
  BauxResearch.Extraction.Native.toUnsigned tgt x
def toSigned (tgt : IScalarTy) (x : F32) : IScalar tgt :=
  BauxResearch.Extraction.Native.toSigned tgt x

def fromSingle (x : F32) : F32 := x
def fromDouble (x : F64) : F32 := Model.cast _ _ x
def add (x y : F32) : F32 := Model.add x y
def sub (x y : F32) : F32 := Model.sub x y
def mul (x y : F32) : F32 := Model.mul x y
def div (x y : F32) : F32 := Model.div x y
def rem (x y : F32) : F32 := fmod x y
def neg (x : F32) : F32 := Model.neg x
def eq (x y : F32) : Bool := equal x y
def ne (x y : F32) : Bool := !(equal x y)
def lt (x y : F32) : Bool := less x y
def le (x y : F32) : Bool := lessEqual x y
def gt (x y : F32) : Bool := less y x
def ge (x y : F32) : Bool := lessEqual y x

def isNan (x : F32) : Result Bool := .ok (Model.isNaN x)
def isInfinite (x : F32) : Result Bool := .ok (Model.isInf x)
def isFinite (x : F32) : Result Bool := .ok (Model.isFinite x)
def abs (x : F32) : Result F32 := .ok (Model.abs x)
def floor (x : F32) : Result F32 := .ok (Model.roundToIntegral x .down)
def ceil (x : F32) : Result F32 := .ok (Model.roundToIntegral x .up)
def round (x : F32) : Result F32 := .ok (Model.roundToIntegral x .nearestAway)
def trunc (x : F32) : Result F32 := .ok (Model.roundToIntegral x .towardZero)
def copysign (x y : F32) : Result F32 := .ok (Model.copySign x y)
def min (x y : F32) : Result F32 := .ok (Model.minimumNumber x y)
def max (x y : F32) : Result F32 := .ok (Model.maximumNumber x y)
def clamp (x lo hi : F32) : Result F32 := BauxResearch.Extraction.Native.clamp x lo hi
def ln (x : F32) : Result F32 := .ok (Model.log x)
def sqrt (x : F32) : Result F32 := .ok (Model.sqrt x)
def cos (x : F32) : Result F32 := .ok (Model.cos x)
def exp (x : F32) : Result F32 := .ok (Model.exp x)
def powi (x : F32) (i : I32) : Result F32 := .ok (Model.powInt x i.val)
def powf (x y : F32) : Result F32 := .ok (Model.pow x y)
def pi : Result F32 := .ok (Model.pi _)

end F32

namespace F64

def clone (x : F64) : Result F64 := .ok x
def fromBits (x : U64) : Result F64 := .ok (Model.ofBits x.bv)
def toBits (x : F64) : Result U64 := .ok ⟨x.bits⟩

def literal (sign : Bool) (num den : Nat) : F64 :=
  Model.roundRat FloatFormat.binary64 sign num den

def posInf : F64 := Model.posInf FloatFormat.binary64
def negInf : F64 := Model.negInf FloatFormat.binary64
def nan : F64 := Model.canonicalNaN FloatFormat.binary64

def fromUnsigned {ty : UScalarTy} (x : UScalar ty) : F64 := unsigned _ x
def fromSigned {ty : IScalarTy} (x : IScalar ty) : F64 := signed _ x
def toUnsigned (tgt : UScalarTy) (x : F64) : UScalar tgt :=
  BauxResearch.Extraction.Native.toUnsigned tgt x
def toSigned (tgt : IScalarTy) (x : F64) : IScalar tgt :=
  BauxResearch.Extraction.Native.toSigned tgt x

def fromSingle (x : F32) : F64 := Model.cast _ _ x
def fromDouble (x : F64) : F64 := x
def add (x y : F64) : F64 := Model.add x y
def sub (x y : F64) : F64 := Model.sub x y
def mul (x y : F64) : F64 := Model.mul x y
def div (x y : F64) : F64 := Model.div x y
def rem (x y : F64) : F64 := fmod x y
def neg (x : F64) : F64 := Model.neg x
def eq (x y : F64) : Bool := equal x y
def ne (x y : F64) : Bool := !(equal x y)
def lt (x y : F64) : Bool := less x y
def le (x y : F64) : Bool := lessEqual x y
def gt (x y : F64) : Bool := less y x
def ge (x y : F64) : Bool := lessEqual y x

def isNan (x : F64) : Result Bool := .ok (Model.isNaN x)
def isInfinite (x : F64) : Result Bool := .ok (Model.isInf x)
def isFinite (x : F64) : Result Bool := .ok (Model.isFinite x)
def abs (x : F64) : Result F64 := .ok (Model.abs x)
def floor (x : F64) : Result F64 := .ok (Model.roundToIntegral x .down)
def ceil (x : F64) : Result F64 := .ok (Model.roundToIntegral x .up)
def round (x : F64) : Result F64 := .ok (Model.roundToIntegral x .nearestAway)
def trunc (x : F64) : Result F64 := .ok (Model.roundToIntegral x .towardZero)
def copysign (x y : F64) : Result F64 := .ok (Model.copySign x y)
def min (x y : F64) : Result F64 := .ok (Model.minimumNumber x y)
def max (x y : F64) : Result F64 := .ok (Model.maximumNumber x y)
def clamp (x lo hi : F64) : Result F64 := BauxResearch.Extraction.Native.clamp x lo hi
def ln (x : F64) : Result F64 := .ok (Model.log x)
def sqrt (x : F64) : Result F64 := .ok (Model.sqrt x)
def cos (x : F64) : Result F64 := .ok (Model.cos x)
def exp (x : F64) : Result F64 := .ok (Model.exp x)
def powi (x : F64) (i : I32) : Result F64 := .ok (Model.powInt x i.val)
def powf (x y : F64) : Result F64 := .ok (Model.pow x y)
def pi : Result F64 := .ok (Model.pi _)

end F64

/--
Project the updated concrete receiver value after a mutable borrow cast
`&mut Self -> &mut dyn Trait` ends.

For any dynamic trait object `d : Dyn Trait` whose underlying concrete type
`d.self` equals `Self` (which holds definitionally as `rfl` for all translated
receiver-preserving dynamic trait updates `{ d with value := value' }`), returns
`h ▸ d.value`.
-/
noncomputable def dynBack {Trait : Type u → Type v} {Self : Type u}
    (init : Self) (d : Dyn Trait) : Self :=
  by
    classical
    exact if h : d.self = Self then h ▸ d.value else init

@[simp]
theorem dyn_mk {Trait : Type u → Type v} {Self : Type u}
    (init : Self) (inst : Trait Self) (val : Self) :
    dynBack init (Dyn.mk Self inst val) = val := by
  unfold dynBack
  simp

theorem dyn_cast {Trait : Type u → Type v} {Self : Type u}
    (init : Self) (d : Dyn Trait) (h : d.self = Self) :
    dynBack init d = h ▸ d.value := by
  unfold dynBack
  simp [h]

end BauxResearch.Extraction.Native
