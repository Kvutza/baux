import BauxResearch.Semantics.FitterAsk
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

/-!
# ENN 0.4.1 affine calibration (`calibration.rs`)

This module executes `sample_prefix`, `AffineCalibrator::fit`, `col_sums`,
`residual_c`, `margin`, and `project_scalar` in source order. Every arithmetic
operation except square root uses FloatLib binary64 semantics.

Covers all 14 reachable functions in `crates/ennbo/src/calibration.rs`:
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#identity()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#fit()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#fit_for_model()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_mu()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_mu_into()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_se()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_se_into()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_draws()`
- `rust:ennbo 0.4.1 calibration/AffineCalibrator#apply_draws_into()`
- `rust:ennbo 0.4.1 calibration/sample_prefix()`
- `rust:ennbo 0.4.1 calibration/col_sums()`
- `rust:ennbo 0.4.1 calibration/residual_c()`
- `rust:ennbo 0.4.1 calibration/margin()`
- `rust:ennbo 0.4.1 calibration/project_scalar()`
-/

namespace BauxResearch.Semantics.Calibration

open FloatLib.Floats.Formats.BinaryInterchange
open Sampling Bounds Fitter

abbrev SqrtOp := Word → Word
abbrev CalMatrix := Fitter.Matrix

def threshold : Word :=
  (Model.roundRat FloatFormat.binary64 false 1 1000000000000000000).bits

inductive CalError where
  | shape
  | rows
  | posterior
  deriving Repr, DecidableEq

structure Calibrator where
  a : List Word
  b : List Word
  c : List Word
  deriving Repr, DecidableEq

def identity (metrics : Nat) : Calibrator :=
  ⟨List.replicate metrics 0, List.replicate metrics Bounds.one,
    List.replicate metrics Bounds.one⟩

def swapAt (values : Array Nat) (left right : Nat) : Array Nat :=
  let first := values[left]!
  let second := values[right]!
  (values.set! left second).set! right first

def prefixLoop (n p : Nat) : Nat → Array Nat → Draw (Array Nat)
  | 0, order => pure order
  | left + 1, order => do
      let index := p - (left + 1)
      let offset ← rangeSample (n - index)
      prefixLoop n p left (swapAt order index (index + offset))

/-- The partial Fisher--Yates loop used by upstream `sample_prefix`. -/
def samplePrefix (n p : Nat) : Draw (List Nat) := do
  let amount := min p n
  let order ← prefixLoop n amount amount (List.range n).toArray
  return order.toList.take amount

def validMatrix (matrix : CalMatrix) : Bool :=
  matrix.data.length == matrix.rows &&
    matrix.data.all (fun row => row.length == matrix.cols)

structure Sums where
  mu : Word := 0
  y : Word := 0
  mumu : Word := 0
  muy : Word := 0
  deriving Repr, DecidableEq

def colLoop (mu y : CalMatrix) (column : Nat) : Nat → Sums → Sums
  | 0, sums => sums
  | left + 1, sums =>
      let row := mu.rows - (left + 1)
      let m := (mu.data[row]!)[column]!
      let target := (y.data[row]!)[column]!
      colLoop mu y column left
        ⟨add sums.mu m, add sums.y target,
          add sums.mumu (mul m m), add sums.muy (mul m target)⟩

/-- The four scalar accumulators from upstream `col_sums`. -/
def colSums (mu y : CalMatrix) (column : Nat) : Sums :=
  colLoop mu y column mu.rows {}

def absWord (value : Word) : Word :=
  (Float.absDouble (Float.wordDouble value)).bits

/-- Rust `f64::max(value, 1.0)` on the nonnegative absolute value. -/
def maxOne (value : Word) : Word :=
  if Float.nanDouble value || Float.ltDouble value Bounds.one then
    Bounds.one
  else value

def coeff (mu y : CalMatrix) (column : Nat) : Word × Word :=
  let sums := colSums mu y column
  let count := numWord mu.rows
  let lead := mul count sums.mumu
  let det := sub lead (mul sums.mu sums.mu)
  let scale := maxOne (absWord lead)
  let cutoff := mul threshold scale
  if Float.finiteDouble det && Float.leDouble cutoff (absWord det) then
    let a := divide
      (sub (mul sums.mumu sums.y) (mul sums.mu sums.muy)) det
    let b := divide
      (sub (mul count sums.muy) (mul sums.mu sums.y)) det
    if Float.finiteDouble a && Float.finiteDouble b then (a, b)
    else (0, Bounds.one)
  else (0, Bounds.one)

def fitCoeffs (mu y : CalMatrix) : List (Word × Word) :=
  if mu.rows < 2 then
    List.replicate mu.cols (0, Bounds.one)
  else (List.range mu.cols).map (coeff mu y)

def residualCol (sqrtOp : SqrtOp) (mu se y : CalMatrix)
    (a b : Word) (column : Nat) : Word :=
  let indices := List.range mu.rows
  let totals := indices.foldl (fun state row =>
    let pred := add a (mul b (mu.data[row]!)[column]!)
    let resid := sub (y.data[row]!)[column]! pred
    let sigma := (se.data[row]!)[column]!
    (add state.1 (mul resid resid), add state.2 (mul sigma sigma))) (0, 0)
  let count := numWord mu.rows
  let rmsR := sqrtOp (divide totals.1 count)
  let rmsS := sqrtOp (divide totals.2 count)
  if Float.finiteDouble rmsS && Float.ltDouble 0 rmsS &&
      Float.finiteDouble rmsR then divide rmsR rmsS
  else Bounds.one

def residuals (sqrtOp : SqrtOp) (mu se y : CalMatrix)
    (coeffs : List (Word × Word)) : List Word :=
  (List.range mu.cols).map fun column =>
    let pair := coeffs[column]!
    residualCol sqrtOp mu se y pair.1 pair.2 column

/-- Source-ordered `AffineCalibrator::fit`. Malformed matrix storage is an
explicit evaluator error; Rust's ndarray values satisfy that invariant. -/
def fit (sqrtOp : SqrtOp) (mu y : CalMatrix) (se : Option CalMatrix) :
    Except CalError Calibrator := do
  if !validMatrix mu || !validMatrix y || !sameDims mu y then throw .shape
  let coeffs := fitCoeffs mu y
  let a := coeffs.map Prod.fst
  let b := coeffs.map Prod.snd
  let c ← match se with
    | none => pure (List.replicate mu.cols Bounds.one)
    | some sigma =>
        if !validMatrix sigma || !sameDims sigma mu then throw .shape
        pure (residuals sqrtOp mu sigma y coeffs)
  return ⟨a, b, c⟩

def pickData : Batch → List Nat → Except CalError Batch
  | _, [] => .ok []
  | rows, index :: rest =>
      match rows[index]?, pickData rows rest with
      | some row, .ok tail => .ok (row :: tail)
      | _, _ => .error .rows

def pickMatrix (matrix : CalMatrix) (indices : List Nat) :
    Except CalError CalMatrix := do
  let data ← pickData matrix.data indices
  return ⟨indices.length, matrix.cols, data⟩

/-- The posterior query required by `fit_calibrator`. Its implementation is
the concrete neighbor/posterior layer, and its flags are fixed here to
exclude the nearest training row. -/
abbrev PostOp := CalMatrix → Params → Bool →
  Except CalError (CalMatrix × CalMatrix)

/-- `surrogate_affine::fit_calibrator`, including its early identity branch,
prefix sampling, row gather, posterior flags, and affine fit. -/
def fitCal (sqrtOp : SqrtOp) (postOp : PostOp) (x y : CalMatrix)
    (params : Params) (samples : Nat) (noise : Bool) : Draw (Except CalError Calibrator) := do
  if x.rows < 2 || samples = 0 then return .ok (identity y.cols)
  let indices ← samplePrefix x.rows (min samples x.rows)
  match pickMatrix x indices, pickMatrix y indices with
  | .ok xSel, .ok ySel =>
      match postOp xSel params noise with
      | .error cause => return .error cause
      | .ok (mu, se) => return fit sqrtOp mu ySel (some se)
  | .error cause, _ => return .error cause
  | _, .error cause => return .error cause

def rows (cal : Calibrator) : List (List Word) := [cal.a, cal.b, cal.c]

def boundEps : Word :=
  (Model.roundRat FloatFormat.binary64 false 1 1000).bits

def boundScale : Word :=
  (Model.roundRat FloatFormat.binary64 false 1 1000000).bits

def seMult : Word := 0x3fd0000000000000
def half : Word := 0x3fe0000000000000

def maxNum (left right : Word) : Word :=
  if Float.nanDouble left || Float.ltDouble left right then right else left

def minNum (left right : Word) : Word :=
  if Float.nanDouble left || Float.ltDouble right left then right else left

def margin (se : Option Word) (span : Word) : Word :=
  let floor := if Float.finiteDouble span then
    maxNum boundEps (mul boundScale span)
  else boundEps
  match se with
  | none => floor
  | some value => maxNum floor (mul seMult (absWord value))

def clamp (value low high : Word) : Word :=
  if Float.ltDouble value low then low
  else if Float.ltDouble high value then high
  else value

/-- Source `project_scalar`, including its uncertainty-dependent margin. -/
def project (value low high : Word) (se : Option Word) : Word :=
  if Float.finiteDouble low && Float.finiteDouble high then
    let eps := margin se (sub high low)
    let lowBound := add low eps
    let highBound := sub high eps
    if Float.leDouble highBound lowBound then mul half (add low high)
    else clamp value lowBound highBound
  else if Float.finiteDouble low then
    maxNum value (add low (margin se Bounds.posInf))
  else if Float.finiteDouble high then
    minNum value (sub high (margin se Bounds.posInf))
  else value

def scaleRow : List Word → List Word → Except CalError (List Word)
  | [], [] => .ok []
  | value :: values, c :: coeffs =>
      match scaleRow values coeffs with
      | .ok rest => .ok (mul value c :: rest)
      | .error cause => .error cause
  | _, _ => .error .shape

def scaleMatrix (matrix : CalMatrix) (coeffs : List Word) :
    Except CalError CalMatrix := do
  if !validMatrix matrix || coeffs.length != matrix.cols then throw .shape
  let data ← matrix.data.mapM (fun row => scaleRow row coeffs)
  return ⟨matrix.rows, matrix.cols, data⟩

def mapRow : List Word → List Word → List Word → List Bound → List Word →
    Except CalError (List Word)
  | [], [], [], [], [] => .ok []
  | mu :: mus, a :: aRest, b :: bs, (low, high) :: bounds, se :: ses =>
      match mapRow mus aRest bs bounds ses with
      | .ok rest =>
          .ok (project (add a (mul b mu)) low high (some se) :: rest)
      | .error cause => .error cause
  | _, _, _, _, _ => .error .shape

def drawRow : List Word → List Word → List Word → List Word → List Word →
    List Bound → List Word → Except CalError (List Word)
  | [], [], [], [], [], [], [] => .ok []
  | draw :: draws, mu :: mus, a :: aRest, b :: bs, c :: cs,
      (low, high) :: bounds, se :: ses =>
      match drawRow draws mus aRest bs cs bounds ses with
      | .ok rest =>
          let center := add a (mul b mu)
          let value := add center (mul c (sub draw mu))
          .ok (project value low high (some (mul se c)) :: rest)
      | .error cause => .error cause
  | _, _, _, _, _, _, _ => .error .shape

/-- `surrogate_affine::apply_prediction` after the model posterior has already
been naturalized into output units. -/
def applyPred (cal : Calibrator) (mu se : CalMatrix) (bounds : List Bound) :
    Except CalError (CalMatrix × CalMatrix) := do
  if !validMatrix mu || !validMatrix se || !sameDims mu se ||
      bounds.length != mu.cols || cal.a.length != mu.cols ||
      cal.b.length != mu.cols || cal.c.length != mu.cols then throw .shape
  let scaled ← scaleMatrix se cal.c
  let mapped ← (mu.data.zip scaled.data).mapM fun pair =>
    mapRow pair.1 cal.a cal.b bounds pair.2
  return (⟨mu.rows, mu.cols, mapped⟩, scaled)

/-- One `surrogate_affine::apply_draws` slice. The posterior mean and standard
error are already in natural units. -/
def applyDraw (cal : Calibrator) (draws mu se : CalMatrix)
    (bounds : List Bound) : Except CalError CalMatrix := do
  if !validMatrix draws || !validMatrix mu || !validMatrix se ||
      !sameDims draws mu || !sameDims mu se || bounds.length != draws.cols ||
      cal.a.length != draws.cols || cal.b.length != draws.cols ||
      cal.c.length != draws.cols then throw .shape
  let mapped ← (draws.data.zip (mu.data.zip se.data)).mapM fun pair =>
    drawRow pair.1 pair.2.1 cal.a cal.b cal.c bounds pair.2.2
  return ⟨draws.rows, draws.cols, mapped⟩

/-- Concrete affine-fit adapter for the calibrator seam in `FitterAsk.ask`.
Only the neighbor posterior operation remains explicit. -/
def askCal (sqrtOp : SqrtOp) (postOp : PostOp) (x y : CalMatrix)
    (noise : Bool) : FitterAsk.CalOp := fun params samples => do
  match ← fitCal sqrtOp postOp x y params samples noise with
  | .ok calibrator => return .ok (rows calibrator)
  | .error _ => return .error .calibrate

theorem emptyPrefix (n : Nat) (state : RngState) :
    run (samplePrefix n 0) state = (.ok [], state) := by
  rfl

theorem shortFit (sqrtOp : SqrtOp) :
    fit sqrtOp ⟨1, 1, [[Bounds.one]]⟩ ⟨1, 1, [[0]]⟩ none =
      .ok (identity 1) := by
  rfl

theorem flatFit (sqrtOp : SqrtOp) :
    fit sqrtOp ⟨2, 1, [[Bounds.one], [Bounds.one]]⟩
      ⟨2, 1, [[0], [Bounds.one]]⟩ none = .ok (identity 1) := by
  rfl

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 calibration/margin().` -/
def marginPure (se : Option BauxResearch.Extraction.Native.F64)
    (span : BauxResearch.Extraction.Native.F64) :
    Aeneas.Std.Result BauxResearch.Extraction.Native.F64 := do
  let b ← BauxResearch.Extraction.Native.F64.isFinite span
  let floor ←
    if b then do
      let fl ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul
        (BauxResearch.Extraction.Native.F64.literal false 99999999999999995 100000000000000000000000)
        span)
      BauxResearch.Extraction.Native.F64.max
        BauxResearch.Generated.Pure.calibration.BOUND_EPS fl
    else
      Aeneas.Std.Result.ok BauxResearch.Generated.Pure.calibration.BOUND_EPS
  match se with
  | none => Aeneas.Std.Result.ok floor
  | some s =>
      let fl ← BauxResearch.Extraction.Native.F64.abs s
      let fl1 ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul
        BauxResearch.Generated.Pure.calibration.BOUND_SE_MULT fl)
      BauxResearch.Extraction.Native.F64.max floor fl1

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 calibration/margin().` -/
theorem marginMatch (se : Option BauxResearch.Extraction.Native.F64)
    (span : BauxResearch.Extraction.Native.F64) :
    BauxResearch.Generated.Pure.calibration.margin se span =
      marginPure se span := by
  rfl

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 calibration/project_scalar().` -/
def projectPure (col lo hi : BauxResearch.Extraction.Native.F64)
    (se : Option BauxResearch.Extraction.Native.F64) :
    Aeneas.Std.Result BauxResearch.Extraction.Native.F64 :=
  BauxResearch.Generated.Pure.calibration.project_scalar col lo hi se

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 calibration/project_scalar().` -/
theorem projectMatch (col lo hi : BauxResearch.Extraction.Native.F64)
    (se : Option BauxResearch.Extraction.Native.F64) :
    BauxResearch.Generated.Pure.calibration.project_scalar col lo hi se =
      projectPure col lo hi se := by
  rfl

#print axioms emptyPrefix
#print axioms shortFit
#print axioms flatFit
#print axioms marginMatch
#print axioms projectMatch

end BauxResearch.Semantics.Calibration
