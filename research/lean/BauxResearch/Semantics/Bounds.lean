import BauxResearch.Semantics.Sampling
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

/-!
# Exact scalar semantics for ENN 0.4.1 output bounds (`y_bounds.rs`)

The source stores binary64 values and rounds after every primitive operation.
`f64::ln` and `f64::exp` remain explicit implementation boundaries; all branch
selection, arithmetic around those calls, validation, and error precedence are
executable here.

Covers all 17 reachable functions in `crates/ennbo/src/y_bounds.rs`:
- `rust:ennbo 0.4.1 y_bounds/validate_bounds()`
- `rust:ennbo 0.4.1 y_bounds/is_identity_bounds()`
- `rust:ennbo 0.4.1 y_bounds/bounds_match()`
- `rust:ennbo 0.4.1 y_bounds/warp_scalar()`
- `rust:ennbo 0.4.1 y_bounds/d_warp_dy()`
- `rust:ennbo 0.4.1 y_bounds/inv_scalar()`
- `rust:ennbo 0.4.1 y_bounds/d_inv_dz()`
- `rust:ennbo 0.4.1 y_bounds/warp_y()`
- `rust:ennbo 0.4.1 y_bounds/warp_y_into()`
- `rust:ennbo 0.4.1 y_bounds/warp_yvar()`
- `rust:ennbo 0.4.1 y_bounds/warp_yvar_into()`
- `rust:ennbo 0.4.1 y_bounds/unwarp_y()`
- `rust:ennbo 0.4.1 y_bounds/unwarp_y_into()`
- `rust:ennbo 0.4.1 y_bounds/unwarp_draws()`
- `rust:ennbo 0.4.1 y_bounds/unwarp_draws_into()`
- `rust:ennbo 0.4.1 y_bounds/naturalize_mu_se()`
- `rust:ennbo 0.4.1 y_bounds/naturalize_posterior()`
-/

namespace BauxResearch.Semantics.Bounds

open Sampling

def posInf : Word := 0x7ff0000000000000
def negInf : Word := 0xfff0000000000000
def posZero : Word := 0
def half : Word := 0x3fe0000000000000
def one : Word := 0x3ff0000000000000

def greater (left right : Word) : Bool :=
  FloatLib.Floats.Formats.BinaryInterchange.Model.compare
    (Float.wordDouble right) (Float.wordDouble left) == some .lt

/-- `coordinate_scalar::to_unit`, returning `0.5` when `high - low` is not
strictly positive under IEEE-754 binary64 comparison. -/
def toUnit (value low high : Word) : Word :=
  let range := sub high low
  if greater range posZero then divide (sub value low) range else half

abbrev LogOp := Word → Word
abbrev ExpOp := Word → Word

inductive BoundError where
  | nan
  | order
  | lowerOpen
  | upperOpen
  deriving Repr, DecidableEq

inductive ValueError where
  | nonfinite
  | lower
  | upper
  | warped
  | derivative
  deriving Repr, DecidableEq

/-- Error precedence and comparisons in `validate_bounds` for one metric. -/
def validateBound (a b : Word) : Except BoundError Unit :=
  if Float.nanDouble a || Float.nanDouble b then .error .nan
  else if Float.finiteDouble a && Float.finiteDouble b && !Float.ltDouble a b then
    .error .order
  else if !Float.finiteDouble a && !Float.eqDouble a negInf then .error .lowerOpen
  else if !Float.finiteDouble b && !Float.eqDouble b posInf then .error .upperOpen
  else .ok ()

/-- Open-interval and finite-value checks for one matrix element. -/
def validateValue (y a b : Word) : Except ValueError Unit :=
  if !Float.finiteDouble y then .error .nonfinite
  else if Float.finiteDouble a && Float.leDouble y a then .error .lower
  else if Float.finiteDouble b && Float.leDouble b y then .error .upper
  else .ok ()

/-- Source expression before the final finite-result check in `warp_scalar`. -/
def warpRaw (logOp : LogOp) (y a b : Word) : Word :=
  match Float.finiteDouble a, Float.finiteDouble b with
  | false, false => y
  | true, false => logOp (sub y a)
  | false, true => Float.negDouble (Float.wordDouble (logOp (sub b y))) |>.bits
  | true, true =>
      let u := divide (sub y a) (sub b a)
      logOp (divide u (sub one u))

/-- `warp_scalar`, including rejection of nonfinite transcendental output. -/
def warp (logOp : LogOp) (y a b : Word) : Except ValueError Word :=
  let z := warpRaw logOp y a b
  if Float.finiteDouble z then .ok z else .error .warped

/-- `inv_scalar`, with binary64 rounding after every primitive operation. -/
def inverse (expOp : ExpOp) (z a b : Word) : Word :=
  match Float.finiteDouble a, Float.finiteDouble b with
  | false, false => z
  | true, false => add a (expOp z)
  | false, true => sub b (expOp (Float.negDouble (Float.wordDouble z)).bits)
  | true, true =>
      let negative := Float.negDouble (Float.wordDouble z) |>.bits
      let s := divide one (add one (expOp negative))
      add a (mul (sub b a) s)

/-- `dφ⁻¹/dz` from the source, without the caller's later absolute value. -/
def inverseJac (expOp : ExpOp) (z a b : Word) : Word :=
  match Float.finiteDouble a, Float.finiteDouble b with
  | false, false => one
  | true, false => expOp z
  | false, true => expOp (Float.negDouble (Float.wordDouble z)).bits
  | true, true =>
      let negative := Float.negDouble (Float.wordDouble z) |>.bits
      let s := divide one (add one (expOp negative))
      mul (mul (sub b a) s) (sub one s)

/-- Source expression before the finite-result check in `d_warp_dy`. -/
def rawJac (y a b : Word) : Word :=
  match Float.finiteDouble a, Float.finiteDouble b with
  | false, false => one
  | true, false => divide one (sub y a)
  | false, true => divide one (sub b y)
  | true, true =>
      let width := sub b a
      let u := divide (sub y a) width
      divide one (mul (mul width u) (sub one u))

def warpJac (y a b : Word) : Except ValueError Word :=
  let d := rawJac y a b
  if Float.finiteDouble d then .ok d else .error .derivative

/-- Exact operation order for `yvar_z = d * d * yvar_y`. -/
def warpVariance (d variance : Word) : Except ValueError Word :=
  let result := mul (mul d d) variance
  if Float.finiteDouble result then .ok result else .error .warped

abbrev Bound := Word × Word
abbrev Batch := List (List Word)

inductive BatchError where
  | shape
  | value (cause : ValueError)
  deriving Repr, DecidableEq

/-- Row-major validation. A list row represents an ndarray row under its
rectangularity invariant; a length mismatch is the source column-shape error. -/
def checkRow : List Word → List Bound → Except BatchError Unit
  | [], [] => .ok ()
  | y :: ys, (a, b) :: bounds =>
      match validateValue y a b with
      | .error cause => .error (.value cause)
      | .ok () => checkRow ys bounds
  | _, _ => .error .shape

def checkBatch : Batch → List Bound → Except BatchError Unit
  | [], _ => .ok ()
  | row :: rows, bounds =>
      match checkRow row bounds with
      | .error cause => .error cause
      | .ok () => checkBatch rows bounds

def warpRow (logOp : LogOp) : List Word → List Bound → Except BatchError (List Word)
  | [], [] => .ok []
  | y :: ys, (a, b) :: bounds =>
      match warp logOp y a b with
      | .error cause => .error (.value cause)
      | .ok z =>
          match warpRow logOp ys bounds with
          | .error cause => .error cause
          | .ok zs => .ok (z :: zs)
  | _, _ => .error .shape

def warpRows (logOp : LogOp) (bounds : List Bound) :
    Batch → Except BatchError Batch
  | [] => .ok []
  | row :: rows =>
      match warpRow logOp row bounds with
      | .error cause => .error cause
      | .ok warped =>
          match warpRows logOp bounds rows with
          | .error cause => .error cause
          | .ok rest => .ok (warped :: rest)

/-- `warp_y` validates the entire input before executing any transcendental
warp. This distinction determines which error is returned. -/
def warpBatch (logOp : LogOp) (values : Batch) (bounds : List Bound) :
    Except BatchError Batch :=
  match checkBatch values bounds with
  | .error cause => .error cause
  | .ok () => warpRows logOp bounds values

/-- Equality of ndarray shapes under the rectangular-list representation used
by this module. The row count and every corresponding row width must agree. -/
def sameShape : Batch → Batch → Bool
  | [], [] => true
  | row :: rows, variance :: variances =>
      row.length == variance.length && sameShape rows variances
  | _, _ => false

/-- Row-major body of `warp_yvar`, after shape and interval validation. -/
def varRow : List Word → List Word → List Bound →
    Except BatchError (List Word)
  | [], [], [] => .ok []
  | y :: ys, variance :: variances, (a, b) :: bounds =>
      match warpJac y a b with
      | .error cause => .error (.value cause)
      | .ok d =>
          match warpVariance d variance with
          | .error cause => .error (.value cause)
          | .ok warped =>
              match varRow ys variances bounds with
              | .error cause => .error cause
              | .ok rest => .ok (warped :: rest)
  | _, _, _ => .error .shape

def varRows (bounds : List Bound) : Batch → Batch →
    Except BatchError Batch
  | [], [] => .ok []
  | row :: rows, variance :: variances =>
      match varRow row variance bounds with
      | .error cause => .error cause
      | .ok warped =>
          match varRows bounds rows variances with
          | .error cause => .error cause
          | .ok rest => .ok (warped :: rest)
  | _, _ => .error .shape

/-- Exact control order of `warp_yvar`: shape equality first, then complete
open-interval validation, then row-major Jacobian and variance arithmetic. -/
def warpVar (values variances : Batch) (bounds : List Bound) :
    Except BatchError Batch :=
  if !sameShape values variances then .error .shape
  else
    match checkBatch values bounds with
    | .error cause => .error cause
    | .ok () => varRows bounds values variances

structure Natural where
  mu : Batch
  se : Batch
  epi : Batch
  ale : Batch
  deriving Repr, DecidableEq

def identityBounds (bounds : List Bound) : Bool :=
  bounds.all fun pair =>
    Float.eqDouble pair.1 negInf && Float.eqDouble pair.2 posInf

def naturalRow (expOp : ExpOp) :
    List Word → List Word → List Word → List Word → List Bound →
      Except BatchError (List Word × List Word × List Word × List Word)
  | [], [], [], [], [] => .ok ([], [], [], [])
  | mu :: mus, se :: ses, epi :: epis, ale :: ales, (a, b) :: bounds =>
      let jac := (Float.absDouble (Float.wordDouble (inverseJac expOp mu a b))).bits
      match naturalRow expOp mus ses epis ales bounds with
      | .error cause => .error cause
      | .ok rest =>
          .ok (inverse expOp mu a b :: rest.1,
            mul se jac :: rest.2.1,
            mul epi jac :: rest.2.2.1,
            mul ale jac :: rest.2.2.2)
  | _, _, _, _, _ => .error .shape

def naturalRows (expOp : ExpOp) (bounds : List Bound) :
    Batch → Batch → Batch → Batch → Except BatchError Natural
  | [], [], [], [] => .ok ⟨[], [], [], []⟩
  | mu :: mus, se :: ses, epi :: epis, ale :: ales =>
      match naturalRow expOp mu se epi ale bounds,
          naturalRows expOp bounds mus ses epis ales with
      | .ok row, .ok rest =>
          .ok ⟨row.1 :: rest.mu, row.2.1 :: rest.se,
            row.2.2.1 :: rest.epi, row.2.2.2 :: rest.ale⟩
      | .error cause, _ => .error cause
      | _, .error cause => .error cause
  | _, _, _, _ => .error .shape

/-- Exact row-major mutation performed by `naturalize_mu_se`. The identity
bounds branch returns before touching any matrix storage. -/
def naturalize (expOp : ExpOp) (bounds : List Bound)
    (mu se epi ale : Batch) : Except BatchError Natural :=
  if identityBounds bounds then .ok ⟨mu, se, epi, ale⟩
  else naturalRows expOp bounds mu se epi ale

private theorem posOpen : Float.finiteDouble posInf = false := by
  decide +kernel

private theorem negOpen : Float.finiteDouble negInf = false := by
  decide +kernel

private theorem oneFinite : Float.finiteDouble one = true := by
  decide +kernel

private theorem zeroFinite : Float.finiteDouble 0 = true := by
  decide +kernel

theorem identityRaw (logOp : LogOp) (y : Word) :
    warpRaw logOp y negInf posInf = y := by
  rfl

theorem identityInverse (expOp : ExpOp) (z : Word) :
    inverse expOp z negInf posInf = z := by
  rfl

theorem identityJac (expOp : ExpOp) (z : Word) :
    inverseJac expOp z negInf posInf = one := by
  rfl

theorem lowerRaw (logOp : LogOp) (y a : Word)
    (ha : Float.finiteDouble a = true) :
    warpRaw logOp y a posInf = logOp (sub y a) := by
  simp only [warpRaw, ha, posOpen]

theorem upperRaw (logOp : LogOp) (y b : Word)
    (hb : Float.finiteDouble b = true) :
    warpRaw logOp y negInf b =
      (Float.negDouble (Float.wordDouble (logOp (sub b y)))).bits := by
  simp only [warpRaw, hb, negOpen]
  rfl

theorem closedRaw (logOp : LogOp) (y a b : Word)
    (ha : Float.finiteDouble a = true) (hb : Float.finiteDouble b = true) :
    warpRaw logOp y a b =
      let u := divide (sub y a) (sub b a)
      logOp (divide u (sub one u)) := by
  simp [warpRaw, ha, hb]

theorem badOrder :
    validateBound one one = .error .order := by
  decide +kernel

theorem wrongLower :
    validateBound posInf posInf = .error .lowerOpen := by
  decide +kernel

theorem strictLower :
    validateValue 0 0 one = .error .lower := by
  decide +kernel

theorem strictUpper :
    validateValue one 0 one = .error .upper := by
  decide +kernel

theorem varianceOrder (d variance : Word) :
    let expected := mul (mul d d) variance
    warpVariance d variance =
      if Float.finiteDouble expected then .ok expected else .error .warped := by
  rfl

theorem batchIdentity (logOp : LogOp) :
    warpBatch logOp [[one, 0]] [(negInf, posInf), (negInf, posInf)] =
      .ok [[one, 0]] := by
  simp [warpBatch, checkBatch, checkRow, warpRows, warpRow, validateValue,
    warp, warpRaw, Float.finiteDouble, Float.expDouble, one, negInf, posInf]

theorem batchShape (logOp : LogOp) :
    warpBatch logOp [[one]] [(negInf, posInf), (negInf, posInf)] =
      .error .shape := by
  rfl

/-- Validation of every input precedes evaluation of the first logarithm. -/
theorem batchPrecheck (logOp : LogOp) :
    warpBatch logOp [[one], [0]] [(0, posInf)] =
      .error (.value .lower) := by
  rfl

/-- A shape mismatch has precedence over an invalid observation value. -/
theorem shapeFirst :
    warpVar [[posInf]] [[]] [(0, posInf)] = .error .shape := by
  rfl

/-- All observations are validated before the first Jacobian is evaluated. -/
theorem varPrecheck :
    warpVar [[one], [0]] [[one], [one]] [(0, posInf)] =
      .error (.value .lower) := by
  rfl

theorem varIdentity :
    warpVar [[one, 0]] [[one, one]]
        [(negInf, posInf), (negInf, posInf)] =
      .ok [[one, one]] := by
  decide +kernel

theorem naturalIdentity (expOp : ExpOp) (mu se epi ale : Batch) :
    naturalize expOp [(negInf, posInf)] mu se epi ale =
      .ok ⟨mu, se, epi, ale⟩ := by
  rfl

theorem naturalShape (expOp : ExpOp) :
    naturalize expOp [(0, one)] [[0]] [] [[0]] [[0]] = .error .shape := by
  rfl

theorem unitFallback (value low high : Word)
    (h : greater (sub high low) posZero = false) :
    toUnit value low high = half := by
  simp [toUnit, h]

theorem unitActive (value low high : Word)
    (h : greater (sub high low) posZero = true) :
    toUnit value low high = divide (sub value low) (sub high low) := by
  simp [toUnit, h]

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 y_bounds/inv_scalar().` -/
def invPure (z a b : BauxResearch.Extraction.Native.F64) :
    Aeneas.Std.Result BauxResearch.Extraction.Native.F64 := do
  let b1 ← BauxResearch.Extraction.Native.F64.isFinite a
  let b2 ← BauxResearch.Extraction.Native.F64.isFinite b
  if b1 then
    if b2 then
      let fl ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.neg z)
      let fl1 ← BauxResearch.Extraction.Native.F64.exp fl
      let fl2 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.add
        (BauxResearch.Extraction.Native.F64.literal false 1 1) fl1)
      let s ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.div
        (BauxResearch.Extraction.Native.F64.literal false 1 1) fl2)
      let fl3 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.sub b a)
      let fl4 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl3 s)
      Aeneas.Std.Result.ok (BauxResearch.Extraction.Native.F64.add a fl4)
    else
      let fl ← BauxResearch.Extraction.Native.F64.exp z
      Aeneas.Std.Result.ok (BauxResearch.Extraction.Native.F64.add a fl)
  else
    if b2 then
      let fl ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.neg z)
      let fl1 ← BauxResearch.Extraction.Native.F64.exp fl
      Aeneas.Std.Result.ok (BauxResearch.Extraction.Native.F64.sub b fl1)
    else
      Aeneas.Std.Result.ok z

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 y_bounds/inv_scalar().` -/
theorem invMatch (z a b : BauxResearch.Extraction.Native.F64) :
    BauxResearch.Generated.Pure.y_bounds.inv_scalar z a b =
      invPure z a b := by
  rfl

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 y_bounds/d_inv_dz().` -/
def derivPure (z a b : BauxResearch.Extraction.Native.F64) :
    Aeneas.Std.Result BauxResearch.Extraction.Native.F64 := do
  let b1 ← BauxResearch.Extraction.Native.F64.isFinite a
  let b2 ← BauxResearch.Extraction.Native.F64.isFinite b
  if b1 then
    if b2 then
      let fl ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.neg z)
      let fl1 ← BauxResearch.Extraction.Native.F64.exp fl
      let fl2 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.add
        (BauxResearch.Extraction.Native.F64.literal false 1 1) fl1)
      let s ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.div
        (BauxResearch.Extraction.Native.F64.literal false 1 1) fl2)
      let fl3 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.sub b a)
      let fl4 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl3 s)
      let fl5 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.sub
        (BauxResearch.Extraction.Native.F64.literal false 1 1) s)
      Aeneas.Std.Result.ok (BauxResearch.Extraction.Native.F64.mul fl4 fl5)
    else
      BauxResearch.Extraction.Native.F64.exp z
  else
    if b2 then
      let fl ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.neg z)
      BauxResearch.Extraction.Native.F64.exp fl
    else
      Aeneas.Std.Result.ok (BauxResearch.Extraction.Native.F64.literal false 1 1)

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 y_bounds/d_inv_dz().` -/
theorem derivMatch (z a b : BauxResearch.Extraction.Native.F64) :
    BauxResearch.Generated.Pure.y_bounds.d_inv_dz z a b =
      derivPure z a b := by
  rfl

#print axioms identityRaw
#print axioms identityInverse
#print axioms identityJac
#print axioms lowerRaw
#print axioms upperRaw
#print axioms closedRaw
#print axioms badOrder
#print axioms wrongLower
#print axioms strictLower
#print axioms strictUpper
#print axioms varianceOrder
#print axioms batchIdentity
#print axioms batchShape
#print axioms batchPrecheck
#print axioms shapeFirst
#print axioms varPrecheck
#print axioms varIdentity
#print axioms naturalIdentity
#print axioms naturalShape
#print axioms unitFallback
#print axioms unitActive
#print axioms invMatch
#print axioms derivMatch

end BauxResearch.Semantics.Bounds
