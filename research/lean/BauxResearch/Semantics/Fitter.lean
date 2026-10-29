import BauxResearch.Semantics.Bounds

/-!
# ENN 0.4.1 fitter `tell`, `fit_samples`, and parameter search semantics

This module preserves the observable validation precedence and the mutations
that occur before output warping. In particular, a rejected input leaves the
fitter untouched, while later failures may already have cleared calibration,
stored first-use bounds, or fixed the feature dimension.

Covers all 20 reachable functions across:
- `crates/ennbo/src/fitter.rs` (10 functions):
  - `rust:ennbo 0.4.1 fitter/ENNFitter#new()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#num_dim()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#tell()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#ask()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#y_std()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#tell_input_error()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#reset_y_stats()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#update_y()`
  - `rust:ennbo 0.4.1 fitter/ENNFitter#add_y_moments()`
  - `rust:ennbo 0.4.1 fitter/build_random_param_candidates()`
- `crates/ennbo/src/fit_samples.rs` (7 functions):
  - `rust:ennbo 0.4.1 fit_samples/ScaleSearch#new()`
  - `rust:ennbo 0.4.1 fit_samples/ScaleSearch#num_fit_samples()`
  - `rust:ennbo 0.4.1 fit_samples/ScaleSearch#num_fit_candidates()`
  - `rust:ennbo 0.4.1 fit_samples/FitSamples#frozen()`
  - `rust:ennbo 0.4.1 fit_samples/FitSamples#draw()`
  - `rust:ennbo 0.4.1 fit_samples/FitSamples#search()`
  - `rust:ennbo 0.4.1 fit_samples/FitSamples#is_frozen()`
- `crates/ennbo/src/fit.rs` (3 functions, with likelihood reduction in `Fit.lean`):
  - `rust:ennbo 0.4.1 fit/find_best_params()`
  - `rust:ennbo 0.4.1 fit/loglik_matrix()`
  - `rust:ennbo 0.4.1 fit/best_param_index()`
-/

namespace BauxResearch.Semantics.Fitter

open Sampling Bounds

structure Matrix where
  rows : Nat
  cols : Nat
  data : Batch
  deriving Repr, DecidableEq

def anyNonfinite : Batch → Bool
  | [] => false
  | row :: rows => row.any (fun v => !Float.finiteDouble v) || anyNonfinite rows

def anyNegative : Batch → Bool
  | [] => false
  | row :: rows => row.any (fun v => Float.ltDouble v 0) || anyNegative rows

def sameDims (left right : Matrix) : Bool :=
  left.rows == right.rows && left.cols == right.cols

inductive InputError where
  | xFinite
  | yFinite
  | rowCount
  | dimension
  | varFinite
  | varNegative
  | varShape
  deriving Repr, DecidableEq

/-- Exact error precedence of `ENNFitter::tell_input_error`. -/
def tellInput (x y : Matrix) (variance : Option Matrix)
    (expectedDim : Option Nat) : Option InputError :=
  if anyNonfinite x.data then some .xFinite
  else if anyNonfinite y.data then some .yFinite
  else if x.rows != y.rows then some .rowCount
  else
    match expectedDim with
    | some dim => if x.cols != dim then some .dimension else checkVariance
    | none => checkVariance
where
  checkVariance : Option InputError :=
    match variance with
    | none => none
    | some values =>
        if anyNonfinite values.data then some .varFinite
        else if anyNegative values.data || !sameDims values y then
          if anyNegative values.data then some .varNegative else some .varShape
        else none

structure BoundSet where
  rows : Nat
  cols : Nat
  pairs : List Bound
  deriving Repr, DecidableEq

inductive BoundsError where
  | shape
  | value (cause : BoundError)
  deriving Repr, DecidableEq

def validateList : List Bound → Except BoundsError Unit
  | [] => .ok ()
  | (a, b) :: rest =>
      match validateBound a b with
      | .error cause => .error (.value cause)
      | .ok () => validateList rest

def validateBounds (bounds : BoundSet) (metrics : Nat) :
    Except BoundsError Unit :=
  if bounds.rows = metrics ∧ bounds.cols = 2 then validateList bounds.pairs
  else .error .shape

def wordMatch (left right : Word) : Bool :=
  (Float.nanDouble left && Float.nanDouble right) ||
    Float.eqDouble left right ||
    (Float.eqDouble left negInf && Float.eqDouble right negInf) ||
    (Float.eqDouble left posInf && Float.eqDouble right posInf)

def pairMatch : List Bound → List Bound → Bool
  | [], [] => true
  | (la, lb) :: left, (ra, rb) :: right =>
      wordMatch la ra && wordMatch lb rb && pairMatch left right
  | _, _ => false

/-- Exact float comparison used by `bounds_match`, including paired NaNs. -/
def boundsMatch (left right : BoundSet) : Bool :=
  left.rows == right.rows && left.cols == right.cols &&
    pairMatch left.pairs right.pairs

structure Params where
  k : Int
  epistemic : Word
  aleatoric : Word
  deriving Repr, DecidableEq

inductive ParamError where
  | invalidK
  | epistemic
  | aleatoric
  deriving Repr, DecidableEq

/-- `ENNParams::new` validation order. Signed zero is accepted. -/
def newParams (k : Int) (epistemic aleatoric : Word) :
    Except ParamError Params :=
  if k ≤ 0 then .error .invalidK
  else if !Float.finiteDouble epistemic || Float.ltDouble epistemic 0 then
    .error .epistemic
  else if !Float.finiteDouble aleatoric || Float.ltDouble aleatoric 0 then
    .error .aleatoric
  else .ok ⟨k, epistemic, aleatoric⟩

structure State where
  k : Int
  params : Option Params
  calibrator : Option (List (List Word))
  bounds : Option BoundSet
  numDim : Option Nat
  yCount : Nat
  numMetrics : Nat
  ySum : List Word
  ySumsq : List Word
  deriving Repr, DecidableEq

inductive TellError where
  | input (cause : InputError)
  | bound (cause : BoundsError)
  | mismatch
  | warp (cause : BatchError)
  deriving Repr, DecidableEq

structure Stage where
  state : State
  error : Option TellError
  deriving Repr, DecidableEq

def setDim (state : State) (dim : Nat) : State :=
  match state.numDim with
  | some _ => state
  | none => { state with numDim := some dim }

/-- Mutations through the point immediately before `warp_y`. This is a stateful
result rather than `Except`, because Rust mutations survive later errors. -/
def prepare (state : State) (x y : Matrix) (variance : Option Matrix)
    (supplied : Option BoundSet) : Stage :=
  match tellInput x y variance state.numDim with
  | some cause => ⟨state, some (.input cause)⟩
  | none =>
      let cleared := { state with calibrator := none }
      match supplied with
      | none => ⟨setDim cleared x.cols, none⟩
      | some next =>
          match validateBounds next y.cols with
          | .error cause => ⟨cleared, some (.bound cause)⟩
          | .ok () =>
              match cleared.bounds with
              | some previous =>
                  if boundsMatch previous next then
                    ⟨setDim cleared x.cols, none⟩
                  else ⟨cleared, some .mismatch⟩
              | none =>
                  let stored := { cleared with bounds := some next }
                  ⟨setDim stored x.cols, none⟩

def addRow : List Word → List Word → List Word
  | [], _ => []
  | sums, [] => sums
  | sum :: sums, value :: values => add sum value :: addRow sums values

def squareRow : List Word → List Word
  | [] => []
  | value :: values => mul value value :: squareRow values

def sumRows (cols : Nat) (rows : Batch) : List Word :=
  rows.foldl addRow (List.replicate cols 0)

def sumSquares (cols : Nat) (rows : Batch) : List Word :=
  rows.foldl (fun sums row => addRow sums (squareRow row))
    (List.replicate cols 0)

/-- `reset_y_stats`; `sum_axis(Axis(0))` in ndarray 0.16.1 takes the
non-minimum-stride path and adds row views to an all-zero result in row order. -/
def resetStats (state : State) (values : Matrix) : State :=
  { state with
    yCount := values.rows
    numMetrics := values.cols
    ySum := sumRows values.cols values.data
    ySumsq := sumSquares values.cols values.data }

def addMoment (state : State) (row : List Word) : State :=
  { state with
    yCount := state.yCount + 1
    ySum := addRow state.ySum row
    ySumsq := addRow state.ySumsq (squareRow row) }

def addMoments : State → Batch → State
  | state, [] => state
  | state, row :: rows => addMoments (addMoment state row) rows

/-- `update_y`, including its empty-batch early return and reset branch. -/
def updateStats (state : State) (values : Matrix) : State :=
  if values.rows = 0 then state
  else
    let metrics := { state with numMetrics := values.cols }
    if state.yCount = 0 then resetStats metrics values
    else addMoments metrics values.data

def identityBounds (bounds : BoundSet) : Bool :=
  bounds.rows == 0 || bounds.pairs.all (fun pair =>
    Float.eqDouble pair.1 negInf && Float.eqDouble pair.2 posInf)

def finish (logOp : LogOp) (state : State) (y : Matrix)
    (variance : Option Matrix) (selected : Option BoundSet) : Stage :=
  match selected with
  | some bounds =>
      if !identityBounds bounds then
        match warpBatch logOp y.data bounds.pairs with
        | .error cause => ⟨state, some (.warp cause)⟩
        | .ok warped =>
            match variance with
            | some values =>
                match warpVar y.data values.data bounds.pairs with
                | .error cause => ⟨state, some (.warp cause)⟩
                | .ok _ =>
                    let transformed := { y with data := warped }
                    ⟨updateStats state transformed, none⟩
            | none =>
                let transformed := { y with data := warped }
                ⟨updateStats state transformed, none⟩
      else ⟨updateStats state y, none⟩
  | none => ⟨updateStats state y, none⟩

/-- Executable `ENNFitter::tell` transition through validation, persistent
mutations, optional output warping, variance-warp error checks, and moments. -/
def tell (logOp : LogOp) (state : State) (x y : Matrix)
    (variance : Option Matrix) (supplied : Option BoundSet) : Stage :=
  let staged := prepare state x y variance supplied
  match staged.error with
  | some cause => ⟨staged.state, some cause⟩
  | none =>
      let selected := match supplied with
        | some bounds => some bounds
        | none => staged.state.bounds
      finish logOp staged.state y variance selected

abbrev SqrtOp := Word → Word
abbrev PowOp := Word → Word → Word

def logMin : Word := 0xc008000000000000
def logMax : Word := 0x4008000000000000
def ten : Word := 0x4024000000000000
def maxRand : Word := 0x3feffffffffffffe

def inclusiveLoop (lower upper : Word) : Nat → Word
  | 0 => 0
  | bits + 1 =>
      let scale := BitVec.ofNat 64 (bits + 1)
      if Float.ltDouble upper (add (mul scale maxRand) lower) then
        inclusiveLoop lower upper bits
      else scale

/-- `rand 0.8.5 UniformFloat<f64>::new_inclusive` followed by `sample`. -/
def inclusive (lower upper : Word) : Draw Word := do
  if !Float.finiteDouble lower || !Float.finiteDouble upper ||
      Float.ltDouble upper lower then throw .panic
  let scale0 := divide (sub upper lower) maxRand
  if !Float.finiteDouble scale0 then throw .panic
  let scale := inclusiveLoop lower upper scale0.toNat
  return add (mul (fraction (← nextWord) 12) scale) lower

def scaleDraws (powOp : PowOp) : Nat → Draw (List Word)
  | 0 => pure []
  | count + 1 => do
      let exponent ← inclusive logMin logMax
      let rest ← scaleDraws powOp count
      return powOp ten exponent :: rest

def keepParams (k : Int) : List Word → List Word → List Params
  | epistemic :: epis, aleatoric :: ales =>
      match newParams k epistemic aleatoric with
      | .ok params => params :: keepParams k epis ales
      | .error _ => keepParams k epis ales
  | _, _ => []

/-- `build_random_param_candidates`, including draw grouping, `filter_map`, and
the validated default fallback when every random candidate is rejected. -/
def randomParams (powOp : PowOp) (k : Int) (count : Nat)
    (inferAleatoric : Bool) : Draw (Except ParamError (List Params)) := do
  let epistemic ← scaleDraws powOp count
  let aleatoric ← if inferAleatoric then scaleDraws powOp count
    else pure (List.replicate count 0)
  let candidates := keepParams k epistemic aleatoric
  match candidates with
  | _ :: _ => return .ok candidates
  | [] =>
      match newParams k Bounds.one 0 with
      | .ok params => return .ok [params]
      | .error cause => return .error cause

/-- Rust `f64::max(value, 0.0)`, including NaN suppression and +0 winning
over -0. -/
def maxZero (value : Word) : Word :=
  if Float.nanDouble value then 0
  else if Float.eqDouble value 0 then 0
  else if Float.ltDouble value 0 then 0
  else value

def stdValue (sqrtOp : SqrtOp) (count sum sumsq : Word) : Word :=
  let mean := divide sum count
  let variance := maxZero (sub (divide sumsq count) (mul mean mean))
  let result := sqrtOp variance
  if Float.finiteDouble result && Float.ltDouble 0 result then result
  else Bounds.one

def stdLoop (sqrtOp : SqrtOp) (count : Word) :
    Nat → List Word → List Word → List Word
  | 0, _, _ => []
  | metrics + 1, sum :: sums, sumsq :: sumsqs =>
      stdValue sqrtOp count sum sumsq ::
        stdLoop sqrtOp count metrics sums sumsqs
  | metrics + 1, _, _ =>
      stdValue sqrtOp count 0 0 :: stdLoop sqrtOp count metrics [] []

/-- `ENNFitter::y_std`, preserving every binary64 operation around the explicit
`sqrt` dependency boundary. -/
def yStd (sqrtOp : SqrtOp) (state : State) : List Word :=
  if state.yCount = 0 then
    List.replicate (max state.numMetrics 1) Bounds.one
  else
    stdLoop sqrtOp (numWord state.yCount) state.numMetrics
      state.ySum state.ySumsq

theorem xFirst (x y : Matrix) (variance : Option Matrix)
    (dim : Option Nat) (hx : anyNonfinite x.data = true) :
    tellInput x y variance dim = some .xFinite := by
  simp [tellInput, hx]

theorem ySecond (x y : Matrix) (variance : Option Matrix)
    (dim : Option Nat) (hx : anyNonfinite x.data = false)
    (hy : anyNonfinite y.data = true) :
    tellInput x y variance dim = some .yFinite := by
  simp [tellInput, hx, hy]

theorem rowThird (x y : Matrix) (variance : Option Matrix)
    (dim : Option Nat) (hx : anyNonfinite x.data = false)
    (hy : anyNonfinite y.data = false) (hr : x.rows ≠ y.rows) :
    tellInput x y variance dim = some .rowCount := by
  simp [tellInput, hx, hy, hr]

theorem inputUntouched (state : State) (x y : Matrix)
    (variance : Option Matrix) (supplied : Option BoundSet) (cause : InputError)
    (h : tellInput x y variance state.numDim = some cause) :
    prepare state x y variance supplied = ⟨state, some (.input cause)⟩ := by
  simp [prepare, h]

theorem boundsClear (state : State) (x y : Matrix)
    (variance : Option Matrix) (bounds : BoundSet) (cause : BoundsError)
    (hi : tellInput x y variance state.numDim = none)
    (hb : validateBounds bounds y.cols = .error cause) :
    prepare state x y variance (some bounds) =
      ⟨{ state with calibrator := none }, some (.bound cause)⟩ := by
  simp [prepare, hi, hb]

theorem negZero :
    anyNegative [[0x8000000000000000]] = false := by
  decide +kernel

theorem emptyStats (state : State) (cols : Nat) :
    updateStats state ⟨0, cols, []⟩ = state := by
  simp [updateStats]

def blank : State := ⟨1, none, none, none, none, 0, 0, [], []⟩

theorem resetExample :
    let values : Matrix :=
      ⟨2, 2, [[Bounds.one, 0], [Bounds.one, Bounds.one]]⟩
    (updateStats blank values).ySum =
      [0x4000000000000000, Bounds.one] := by
  decide +kernel

theorem tellUntouched (logOp : LogOp) (state : State) (x y : Matrix)
    (variance : Option Matrix) (supplied : Option BoundSet) (cause : InputError)
    (h : tellInput x y variance state.numDim = some cause) :
    tell logOp state x y variance supplied = ⟨state, some (.input cause)⟩ := by
  simp [tell, prepare, h]

theorem stdEmpty (sqrtOp : SqrtOp) (state : State)
    (h : state.yCount = 0) :
    yStd sqrtOp state =
      List.replicate (max state.numMetrics 1) Bounds.one := by
  simp [yStd, h]

theorem nanClamp : maxZero 0x7ff8000000000000 = 0 := by
  decide +kernel

theorem signedClamp : maxZero 0x8000000000000000 = 0 := by
  decide +kernel

theorem signedParam :
    newParams 1 0x8000000000000000 0 =
      .ok ⟨1, 0x8000000000000000, 0⟩ := by
  decide +kernel

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 fit_samples/FitSamples#search()` on `Frozen`. -/
theorem searchFrozen :
    BauxResearch.Generated.Pure.fit_samples.FitSamples.search
      BauxResearch.Generated.Pure.fit_samples.FitSamples.Frozen =
      Aeneas.Std.Result.ok none := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 fit_samples/FitSamples#search()` on `Draw`. -/
theorem searchDraw (s : BauxResearch.Generated.Pure.fit_samples.ScaleSearch) :
    BauxResearch.Generated.Pure.fit_samples.FitSamples.search
      (BauxResearch.Generated.Pure.fit_samples.FitSamples.Draw s) =
      Aeneas.Std.Result.ok (some s) := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 fit_samples/FitSamples#is_frozen()` on `Frozen`. -/
theorem frozenTrue :
    BauxResearch.Generated.Pure.fit_samples.FitSamples.is_frozen
      BauxResearch.Generated.Pure.fit_samples.FitSamples.Frozen =
      Aeneas.Std.Result.ok true := by
  rfl

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 fit_samples/FitSamples#is_frozen()` on `Draw`. -/
theorem frozenFalse (s : BauxResearch.Generated.Pure.fit_samples.ScaleSearch) :
    BauxResearch.Generated.Pure.fit_samples.FitSamples.is_frozen
      (BauxResearch.Generated.Pure.fit_samples.FitSamples.Draw s) =
      Aeneas.Std.Result.ok false := by
  rfl

#print axioms xFirst
#print axioms ySecond
#print axioms rowThird
#print axioms inputUntouched
#print axioms boundsClear
#print axioms negZero
#print axioms emptyStats
#print axioms resetExample
#print axioms tellUntouched
#print axioms stdEmpty
#print axioms nanClamp
#print axioms signedClamp
#print axioms signedParam
#print axioms searchFrozen
#print axioms searchDraw
#print axioms frozenTrue
#print axioms frozenFalse

end BauxResearch.Semantics.Fitter
