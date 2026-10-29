import BauxResearch.Generated.Optimizer.Funs
import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.Runtime
import FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.Runtime
import FloatLib.Floats.Formats.BinaryInterchange.Format.Catalog
import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
import Mathlib.Data.BitVec
import Mathlib.Tactic

/-!
# Checked floating-point support for optimizer extraction

Bit-precise IEEE-754 binary32 and binary64 formalization backed by FloatLib:
1. Storage words and masks matching Aeneas-extracted definitions.
2. Bitwise predicates: NaN, signed zero, and finite classifications.
3. Order mapping and strict comparison matching native float semantics.
4. Algebraic IEEE properties: irreflexivity, asymmetry, NaN rejection, and zero equality.
5. Exact widening cast from binary32 to binary64 and narrowing cast to binary32.
6. Extraction correspondence theorems connecting generated Rust to FloatLib.
7. Optimizer constant fixtures: 1.0e-12, 1.0, 2.0, 0.5, and widening identity.
-/

namespace BauxResearch.Semantics.Float

open FloatLib.Floats.Formats.BinaryInterchange

/-! ## 1. Storage Words and Format Descriptors -/

abbrev WordSingle := Model FloatFormat.binary32
abbrev WordDouble := Model FloatFormat.binary64

def wordSingle (bits : BitVec 32) : WordSingle :=
  Model.ofBits bits

def wordDouble (bits : BitVec 64) : WordDouble :=
  Model.ofBits bits

def signSingle : BitVec 32 := BitVec.ofNat 32 0x80000000
def expSingle : BitVec 32 := BitVec.ofNat 32 0x7f800000
def fracSingle : BitVec 32 := BitVec.ofNat 32 0x007fffff

def signDouble : BitVec 64 := BitVec.ofNat 64 0x8000000000000000
def expDouble : BitVec 64 := BitVec.ofNat 64 0x7ff0000000000000
def fracDouble : BitVec 64 := BitVec.ofNat 64 0x000fffffffffffff

/-! ## 2. Optimizer Constants in Bit Representation -/

def singleZero : BitVec 32 := BitVec.ofNat 32 0
def singleOne : BitVec 32 := BitVec.ofNat 32 0x3f800000
def singleInf : BitVec 32 := BitVec.ofNat 32 0x7f800000
def singleNan : BitVec 32 := BitVec.ofNat 32 0x7fc00000

def doubleZero : BitVec 64 := BitVec.ofNat 64 0
def doubleOne : BitVec 64 := BitVec.ofNat 64 0x3ff0000000000000
def doubleTwo : BitVec 64 := BitVec.ofNat 64 0x4000000000000000
def doubleHalf : BitVec 64 := BitVec.ofNat 64 0x3fe0000000000000
def scaleDouble : BitVec 64 := BitVec.ofNat 64 0x3d719799812dea11
def priorDouble : BitVec 64 := BitVec.ofNat 64 0x3ff0000000000000
def doubleInf : BitVec 64 := BitVec.ofNat 64 0x7ff0000000000000
def doubleNan : BitVec 64 := BitVec.ofNat 64 0x7ff8000000000000

/-! ## 3. Bit-Level Predicates and Ordering -/

def nanSingle (bits : BitVec 32) : Bool :=
  decide ((bits &&& expSingle) = expSingle ∧ (bits &&& fracSingle) ≠ 0)

def zeroSingle (bits : BitVec 32) : Bool :=
  decide ((bits &&& ~~~signSingle) = 0)

def finiteSingle (bits : BitVec 32) : Bool :=
  decide ((bits &&& expSingle) ≠ expSingle)

def orderSingle (bits : BitVec 32) : BitVec 32 :=
  if bits &&& signSingle = 0 then bits ||| signSingle else ~~~bits

def ltSingle (x y : BitVec 32) : Bool :=
  if nanSingle x || nanSingle y then false
  else if zeroSingle x && zeroSingle y then false
  else decide (orderSingle x < orderSingle y)

def leSingle (x y : BitVec 32) : Bool :=
  if nanSingle x || nanSingle y then false
  else if zeroSingle x && zeroSingle y then true
  else decide (orderSingle x ≤ orderSingle y)

def eqSingle (x y : BitVec 32) : Bool :=
  if nanSingle x || nanSingle y then false
  else if zeroSingle x && zeroSingle y then true
  else decide (x = y)

def nanDouble (bits : BitVec 64) : Bool :=
  decide ((bits &&& expDouble) = expDouble ∧ (bits &&& fracDouble) ≠ 0)

def zeroDouble (bits : BitVec 64) : Bool :=
  decide ((bits &&& ~~~signDouble) = 0)

def finiteDouble (bits : BitVec 64) : Bool :=
  decide ((bits &&& expDouble) ≠ expDouble)

def orderDouble (bits : BitVec 64) : BitVec 64 :=
  if bits &&& signDouble = 0 then bits ||| signDouble else ~~~bits

def ltDouble (x y : BitVec 64) : Bool :=
  if nanDouble x || nanDouble y then false
  else if zeroDouble x && zeroDouble y then false
  else decide (orderDouble x < orderDouble y)

def leDouble (x y : BitVec 64) : Bool :=
  if nanDouble x || nanDouble y then false
  else if zeroDouble x && zeroDouble y then true
  else decide (orderDouble x ≤ orderDouble y)

def eqDouble (x y : BitVec 64) : Bool :=
  if nanDouble x || nanDouble y then false
  else if zeroDouble x && zeroDouble y then true
  else decide (x = y)

/-! ## 4. FloatLib Arithmetic and Casts -/

def addSingle (x y : WordSingle) : WordSingle :=
  Model.add x y

def subSingle (x y : WordSingle) : WordSingle :=
  Model.sub x y

def mulSingle (x y : WordSingle) : WordSingle :=
  Model.mul x y

def divSingle (x y : WordSingle) : WordSingle :=
  Model.div x y

def sqrtSingle (x : WordSingle) : WordSingle :=
  Model.sqrt x

def negSingle (x : WordSingle) : WordSingle :=
  Model.neg x

def absSingle (x : WordSingle) : WordSingle :=
  Model.abs x

def addDouble (x y : WordDouble) : WordDouble :=
  Model.add x y

def subDouble (x y : WordDouble) : WordDouble :=
  Model.sub x y

def mulDouble (x y : WordDouble) : WordDouble :=
  Model.mul x y

def divDouble (x y : WordDouble) : WordDouble :=
  Model.div x y

def sqrtDouble (x : WordDouble) : WordDouble :=
  Model.sqrt x

def negDouble (x : WordDouble) : WordDouble :=
  Model.neg x

def absDouble (x : WordDouble) : WordDouble :=
  Model.abs x

def castWide (x : WordSingle) : WordDouble :=
  Model.cast FloatFormat.binary32 FloatFormat.binary64 x

def castNarrow (x : WordDouble) : WordSingle :=
  Model.cast FloatFormat.binary64 FloatFormat.binary32 x

/-! ## 5. IEEE-754 Algebraic Properties -/

theorem singleIrrefl (word : BitVec 32) :
    ltSingle word word = false := by
  unfold ltSingle nanSingle zeroSingle orderSingle signSingle expSingle fracSingle
  bv_decide

theorem doubleIrrefl (word : BitVec 64) :
    ltDouble word word = false := by
  unfold ltDouble nanDouble zeroDouble orderDouble signDouble expDouble fracDouble
  bv_decide

theorem singleAsymm (left right : BitVec 32) :
    ltSingle left right = true → ltSingle right left = false := by
  unfold ltSingle nanSingle zeroSingle orderSingle signSingle expSingle fracSingle
  bv_decide

theorem doubleAsymm (left right : BitVec 64) :
    ltDouble left right = true → ltDouble right left = false := by
  unfold ltDouble nanDouble zeroDouble orderDouble signDouble expDouble fracDouble
  bv_decide

theorem singleReject (left right : BitVec 32) :
    nanSingle left = true → ltSingle left right = false ∧ ltSingle right left = false := by
  unfold ltSingle nanSingle zeroSingle orderSingle signSingle expSingle fracSingle
  bv_decide

theorem doubleReject (left right : BitVec 64) :
    nanDouble left = true → ltDouble left right = false ∧ ltDouble right left = false := by
  unfold ltDouble nanDouble zeroDouble orderDouble signDouble expDouble fracDouble
  bv_decide

theorem singleZeros :
    eqSingle singleZero (BitVec.ofNat 32 0x80000000) = true ∧
    ltSingle singleZero (BitVec.ofNat 32 0x80000000) = false := by
  decide

theorem doubleZeros :
    eqDouble doubleZero (BitVec.ofNat 64 0x8000000000000000) = true ∧
    ltDouble doubleZero (BitVec.ofNat 64 0x8000000000000000) = false := by
  decide

/-! ## 6. Translation Correspondence with Aeneas Extraction -/

theorem singleSign :
    BauxResearch.Generated.Optimizer.float.SIGN_SINGLE.bv = signSingle := by
  unfold BauxResearch.Generated.Optimizer.float.SIGN_SINGLE signSingle
  decide

theorem singleExp :
    BauxResearch.Generated.Optimizer.float.EXP_SINGLE.bv = expSingle := by
  unfold BauxResearch.Generated.Optimizer.float.EXP_SINGLE expSingle
  decide

theorem singleFrac :
    BauxResearch.Generated.Optimizer.float.FRAC_SINGLE.bv = fracSingle := by
  unfold BauxResearch.Generated.Optimizer.float.FRAC_SINGLE fracSingle
  decide

theorem doubleSign :
    BauxResearch.Generated.Optimizer.float.SIGN_DOUBLE.bv = signDouble := by
  unfold BauxResearch.Generated.Optimizer.float.SIGN_DOUBLE signDouble
  decide

theorem doubleExp :
    BauxResearch.Generated.Optimizer.float.EXP_DOUBLE.bv = expDouble := by
  unfold BauxResearch.Generated.Optimizer.float.EXP_DOUBLE expDouble
  decide

theorem doubleFrac :
    BauxResearch.Generated.Optimizer.float.FRAC_DOUBLE.bv = fracDouble := by
  unfold BauxResearch.Generated.Optimizer.float.FRAC_DOUBLE fracDouble
  decide

theorem nanMatch (bits : Aeneas.Std.U32) :
    BauxResearch.Generated.Optimizer.float.single_nan bits =
      Aeneas.Std.Result.ok (nanSingle bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.float.single_nan,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.EXP_SINGLE =
      (⟨expSingle⟩ : Aeneas.Std.U32) by exact Aeneas.Std.U32.bv_eq_imp_eq _ _ singleExp]
  rw [show BauxResearch.Generated.Optimizer.float.FRAC_SINGLE =
      (⟨fracSingle⟩ : Aeneas.Std.U32) by exact Aeneas.Std.U32.bv_eq_imp_eq _ _ singleFrac]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, nanSingle]
  split <;> simp_all [bne, BEq.beq]

theorem zeroMatch (bits : Aeneas.Std.U32) :
    BauxResearch.Generated.Optimizer.float.single_zero bits =
      Aeneas.Std.Result.ok (zeroSingle bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.float.single_zero,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.SIGN_SINGLE =
      (⟨signSingle⟩ : Aeneas.Std.U32) by exact Aeneas.Std.U32.bv_eq_imp_eq _ _ singleSign]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, zeroSingle]

theorem orderMatch (bits : Aeneas.Std.U32) :
    BauxResearch.Generated.Optimizer.float.single_order bits =
      Aeneas.Std.Result.ok (⟨orderSingle bits.bv⟩ : Aeneas.Std.U32) := by
  simp only [BauxResearch.Generated.Optimizer.float.single_order,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.SIGN_SINGLE =
      (⟨signSingle⟩ : Aeneas.Std.U32) by exact Aeneas.Std.U32.bv_eq_imp_eq _ _ singleSign]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, orderSingle]
  split <;> simp_all [Aeneas.Std.UScalar.eq_equiv_bv_eq]

private theorem bitLt (left right : BitVec 32) :
    ((↑(⟨left⟩ : Aeneas.Std.U32) : Nat) < ↑(⟨right⟩ : Aeneas.Std.U32)) =
      (left < right) := by
  rfl

theorem singleMatch (x y : Aeneas.Std.U32) :
    BauxResearch.Generated.Optimizer.float.single_lt x y =
      Aeneas.Std.Result.ok (ltSingle x.bv y.bv) := by
  simp [BauxResearch.Generated.Optimizer.float.single_lt,
    nanMatch, zeroMatch, orderMatch, bitLt, ltSingle]
  cases hnx : nanSingle x.bv <;>
    cases hny : nanSingle y.bv <;>
    cases hzx : zeroSingle x.bv <;>
    cases hzy : zeroSingle y.bv <;>
    cases hlt : decide (orderSingle x.bv < orderSingle y.bv) <;>
    simp_all

theorem wideNan (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.float.double_nan bits =
      Aeneas.Std.Result.ok (nanDouble bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.float.double_nan,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.EXP_DOUBLE =
      (⟨expDouble⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ doubleExp]
  rw [show BauxResearch.Generated.Optimizer.float.FRAC_DOUBLE =
      (⟨fracDouble⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ doubleFrac]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, nanDouble]
  split <;> simp_all [bne, BEq.beq]

theorem wideZero (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.float.double_zero bits =
      Aeneas.Std.Result.ok (zeroDouble bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.float.double_zero,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.SIGN_DOUBLE =
      (⟨signDouble⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ doubleSign]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, zeroDouble]

theorem wideOrder (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.float.double_order bits =
      Aeneas.Std.Result.ok (⟨orderDouble bits.bv⟩ : Aeneas.Std.U64) := by
  simp only [BauxResearch.Generated.Optimizer.float.double_order,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.float.SIGN_DOUBLE =
      (⟨signDouble⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ doubleSign]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, orderDouble]
  split <;> simp_all [Aeneas.Std.UScalar.eq_equiv_bv_eq]

private theorem wideLt (left right : BitVec 64) :
    ((↑(⟨left⟩ : Aeneas.Std.U64) : Nat) < ↑(⟨right⟩ : Aeneas.Std.U64)) =
      (left < right) := by
  rfl

theorem doubleMatch (x y : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.float.double_lt x y =
      Aeneas.Std.Result.ok (ltDouble x.bv y.bv) := by
  simp [BauxResearch.Generated.Optimizer.float.double_lt,
    wideNan, wideZero, wideOrder, wideLt, ltDouble]
  cases hnx : nanDouble x.bv <;>
    cases hny : nanDouble y.bv <;>
    cases hzx : zeroDouble x.bv <;>
    cases hzy : zeroDouble y.bv <;>
    cases hlt : decide (orderDouble x.bv < orderDouble y.bv) <;>
    simp_all

/-! ## 7. Deterministic Fixtures -/

theorem priorFixture :
    priorDouble = BitVec.ofNat 64 0x3ff0000000000000 := by
  decide

theorem scaleFixture :
    scaleDouble = BitVec.ofNat 64 0x3d719799812dea11 := by
  decide

theorem twoFixture :
    doubleTwo = BitVec.ofNat 64 0x4000000000000000 := by
  decide

theorem halfFixture :
    doubleHalf = BitVec.ofNat 64 0x3fe0000000000000 := by
  decide

theorem widenFixture :
    (castWide (wordSingle singleOne)).bits.toNat = doubleOne.toNat := by
  rfl

#print axioms singleIrrefl
#print axioms doubleIrrefl
#print axioms singleAsymm
#print axioms doubleAsymm
#print axioms singleReject
#print axioms doubleReject
#print axioms singleZeros
#print axioms doubleZeros
#print axioms singleSign
#print axioms singleExp
#print axioms singleFrac
#print axioms doubleSign
#print axioms doubleExp
#print axioms doubleFrac
#print axioms nanMatch
#print axioms zeroMatch
#print axioms orderMatch
#print axioms wideNan
#print axioms wideZero
#print axioms wideOrder
#print axioms singleMatch
#print axioms doubleMatch
#print axioms priorFixture
#print axioms scaleFixture
#print axioms twoFixture
#print axioms halfFixture
#print axioms widenFixture

end BauxResearch.Semantics.Float
