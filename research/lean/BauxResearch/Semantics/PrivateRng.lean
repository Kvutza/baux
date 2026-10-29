import BauxResearch.Semantics.ChaCha
import BauxResearch.Semantics.Sampling
import BauxResearch.Semantics.Surrogate
import BauxResearch.Semantics.Append
import BauxResearch.Semantics.Draw
import BauxResearch.Semantics.Acq

/-!
# Total private ChaCha12 RNG stream semantics

This module defines executable total semantics for ENN 0.4.1 private surrogate
and acquisition RNG streams, replacing finite observation-prefix assumptions
with exact on-demand ChaCha12 progression.

Draw order, IEEE binary64 arithmetic, rejection sampling, and non-exhaustion
errors are preserved. A stream divergence relation characterizes non-terminating
consumers without assuming divergence or relying on arbitrary callbacks.
-/

namespace BauxResearch.Semantics.PrivateRng

open Sampling Bounds Fitter

/-- Logical state of a total private ChaCha12 word stream. -/
structure Stream where
  core : ChaCha.State
  consumed : Nat := 0
  deriving Repr, DecidableEq

/-- Construct a fresh total stream from four 64-bit caller seed words. -/
def fromSeed (seed : Array Word) : Stream :=
  ⟨ChaCha.fromWords seed, 0⟩

/-- Seeded constructor for testing or single-u64 seed expansion. -/
def seeded (seed : Word) : Stream :=
  ⟨ChaCha.seeded seed, 0⟩

/-- Draw one 32-bit half on demand from the total stream. Never exhausts. -/
def nextHalf (s : Stream) : BitVec 32 × Stream :=
  let (word, core) := ChaCha.next32 s.core
  (word, ⟨core, s.consumed + 1⟩)

/-- Draw one 64-bit word on demand from the total stream. Never exhausts. -/
def nextWord (s : Stream) : Word × Stream :=
  let (word, core) := ChaCha.next64 s.core
  (word, ⟨core, s.consumed + 1⟩)

/-- Draw `n` consecutive 64-bit words, preserving stream order. -/
def takeWords : Nat → Stream → List Word × Stream
  | 0, s => ([], s)
  | n + 1, s =>
      let (w, s) := nextWord s
      let (rest, s) := takeWords n s
      (w :: rest, s)

/-- Finite observation prefix of the total stream. -/
def prefixRng (s : Stream) (count : Nat) : RngState :=
  { words := (takeWords count s).1 }

/-- Run a Draw action against an observation prefix of the total stream. -/
def runPrefix (action : Draw α) (s : Stream) (count : Nat) :
    Except Failure α × RngState :=
  Sampling.run action (prefixRng s count)

/-- An action diverges on a stream iff every finite observation prefix yields exhaustion. -/
def Diverges (action : Draw α) (s : Stream) : Prop :=
  ∀ n : Nat, (runPrefix action s n).1 = .error Failure.exhausted

/-- An action terminates on a stream with result `val`, consuming `used` words,
iff prefix evaluation with `used` words succeeds with `val`, consumes all `used` words,
and larger prefixes produce the identical value. -/
structure Terminates (action : Draw α) (s : Stream) (val : α)
    (used : Nat) (remPending : List (BitVec 32)) : Prop where
  exactDraw : runPrefix action s used =
    (.ok val, { words := [], pending := remPending, consumed := used })
  stable : ∀ n : Nat, n ≥ used →
    ∃ remWords, runPrefix action s n =
      (.ok val, { words := remWords, pending := remPending, consumed := used })

/-- Divergence and termination are mutually exclusive. -/
theorem divDisjoint (action : Draw α) (s : Stream) (val : α) (used : Nat)
    (pending : List (BitVec 32)) :
    Diverges action s → Terminates action s val used pending → False := by
  intro hdiv hterm
  have hexhaust := hdiv used
  have hexact := hterm.exactDraw
  simp [runPrefix] at hexhaust hexact
  rw [hexact] at hexhaust
  contradiction

/-- Total standard uniform double in [0, 1) using 53 random bits. Consumes one word. -/
def standard (s : Stream) : Word × Stream :=
  let (w, s') := nextWord s
  (Sampling.fraction w 11, s')

/-- Total inclusive float draw in [lower, upper]. Consumes exactly one 64-bit word. -/
def inclusive (lower upper : Word) (s : Stream) : Except Failure Word × Stream :=
  if !Float.finiteDouble lower || !Float.finiteDouble upper ||
      Float.ltDouble upper lower then (.error .panic, s)
  else
    let scale0 := divide (sub upper lower) Fitter.maxRand
    if !Float.finiteDouble scale0 then (.error .panic, s)
    else
      let scale := Fitter.inclusiveLoop lower upper scale0.toNat
      let (w, s') := nextWord s
      (.ok (add (mul (fraction w 12) scale) lower), s')

/-- Total sequence of parameter scale draws. Consumes exactly `count` words. -/
def scaleDraws (powOp : PowOp) : Nat → Stream → Except Failure (List Word) × Stream
  | 0, s => (.ok [], s)
  | count + 1, s =>
      match inclusive Fitter.logMin Fitter.logMax s with
      | (.error cause, s') => (.error cause, s')
      | (.ok exponent, s') =>
          match scaleDraws powOp count s' with
          | (.error cause, s'') => (.error cause, s'')
          | (.ok rest, s'') => (.ok (powOp Fitter.ten exponent :: rest), s'')

/-- Total candidate parameter generation on the stream. Consumes
`count * (if inferAleatoric then 2 else 1)` words without fixed caps. -/
def randomParams (powOp : PowOp) (k : Int) (count : Nat)
    (inferAleatoric : Bool) (s : Stream) :
    Except Failure (Except ParamError (List Params)) × Stream :=
  match scaleDraws powOp count s with
  | (.error cause, s') => (.error cause, s')
  | (.ok epistemic, s') =>
      let (aleRes, s'') : Except Failure (List Word) × Stream :=
        if inferAleatoric then
          match scaleDraws powOp count s' with
          | (.error cause, s'') => (.error cause, s'')
          | (.ok aleatoric, s'') => (.ok aleatoric, s'')
        else (.ok (List.replicate count 0), s')
      match aleRes with
      | .error cause => (.error cause, s'')
      | .ok aleatoric =>
          let candidates := Fitter.keepParams k epistemic aleatoric
          match candidates with
          | _ :: _ => (.ok (.ok candidates), s'')
          | [] =>
              match Fitter.newParams k Bounds.one 0 with
              | .ok params => (.ok (.ok [params]), s'')
              | .error cause => (.ok (.error cause), s'')

/-- Total surrogate function draw from the stream. Consumes exactly one word
when uncalibrated and two words when calibrated. -/
def sampleDraw (ops : Draw.Ops) (expOp : ExpOp) (backend : Neighbor.BackendOp)
    (ctx : Select.Context) (queries : Fitter.Matrix) (count : Nat)
    (s : Stream) : Except Select.SelectError Draw.Cube × Stream :=
  let (first, s1) := nextWord s
  match ctx.calibrator with
  | none =>
      match Batch.drawData ops.sqrt backend ctx.model queries ctx.params false false with
      | .error _ => (.error .posterior, s1)
      | .ok data => (.ok (Draw.drawMany ops data first count queries.rows
          ctx.model.neighbor.trainY.cols), s1)
  | some calibrator =>
      let (base, s2) := nextWord s1
      match Batch.drawData ops.sqrt backend ctx.model queries ctx.params false false with
      | .error _ => (.error .posterior, s2)
      | .ok data =>
          let warped := Draw.drawMany ops data base count queries.rows
            ctx.model.neighbor.trainY.cols
          match Draw.naturalCube expOp ctx.bounds warped with
          | .error _ => (.error .posterior, s2)
          | .ok draws =>
              match Batch.one ops.sqrt backend ctx.model queries ctx.params false false with
              | .error _ => (.error .posterior, s2)
              | .ok stats =>
                  match Bounds.naturalize expOp ctx.bounds stats.mu stats.se
                      stats.seEpi stats.seAle with
                  | .error _ => (.error .posterior, s2)
                  | .ok post => (Draw.calCube calibrator ctx.bounds post.mu post.se draws, s2)

/-- Total argmax tie-breaking when maximum is unique. Consumes zero words. -/
def argmaxUnique (values : List Word) (s : Stream) : Option (Nat × Stream) :=
  if values.isEmpty then some (0, s)
  else
    let choices := Acq.ties values
    if choices.length = 1 then some (choices[0]!, s)
    else none

/-- Fitter operations with total private RNG semantics: no localWords prefix field. -/
structure Ops where
  pow : PowOp
  sqrt : SqrtOp
  log : LogOp
  powi : Fit.PowiOp
  backend : Neighbor.BackendOp
  gate : Append.Gate

/-- Reconstruct Append.Ops with an exact consumed word count. -/
def innerOps (ops : Ops) (words : Nat) : Append.Ops :=
  ⟨ops.pow, ops.sqrt, ops.log, ops.powi, ops.backend, ops.gate, words⟩

/-- Recursively scale posterior batches by standard deviations. -/
def scaleAll (safe : List Word) :
    List Bounds.Batch → Except FitterAsk.AskError (List Bounds.Batch)
  | [] => .ok []
  | b :: bs => do
      let first ← FitterAsk.scaleBatch b safe
      let rest ← scaleAll safe bs
      return first :: rest

/-- Evaluate sync and release gate errors for an append plan. -/
def gateError (ops : Ops) (plan : Surrogate.AppendPlan) :
    Option Append.FitError :=
  if plan.sync then
    match ops.gate .sync with
    | .error c => some (.effect c)
    | .ok () => none
  else if plan.release then
    match ops.gate .release with
    | .error c => some (.effect c)
    | .ok () => none
  else none

/-- Construct an append step with private stream consumption evidence. -/
def stepResult (state : Append.FitState) (callerRng : RngState)
    (consumed : Nat) (err : Option Append.FitError) : Append.Step :=
  ⟨state, callerRng, some { words := [], consumed := consumed }, err⟩

def rejectFuel : Nat := 1024

def two32 : Nat := 2 ^ 32
def two64 : Nat := 2 ^ 64

def loop32 (low range zone : Nat) : Nat → Stream → Except Failure Nat × Stream
  | 0, s => (.error .exhausted, s)
  | fuel + 1, s =>
      let (word, s1) := nextHalf s
      let product := word.toNat * range
      if product % two32 ≤ zone then (.ok (low + product / two32), s1)
      else loop32 low range zone fuel s1

def range32 (low high : Nat) (s : Stream) : Except Failure Nat × Stream :=
  if high > Index.max32 || high < low then (.error .panic, s)
  else
    let range := high - low + 1
    if range == two32 then
      let (word, s1) := nextHalf s
      (.ok (low + word.toNat), s1)
    else
      loop32 low range (Index.zone32 range) rejectFuel s

def floydLoop (length amount : Nat) :
    Nat → List Nat → Nat → Stream → Except Failure (List Nat) × Stream
  | j, values, 0, s =>
      if j < length then (.error .exhausted, s) else (.ok values, s)
  | j, values, fuel + 1, s =>
      if j < length then
        match range32 0 j s with
        | (.error cause, s1) => (.error cause, s1)
        | (.ok target, s1) =>
            let next := if amount < 50 then
              match values.findIdx? (fun value => value == target) with
              | some pos => Index.insertAt values pos j
              | none => values ++ [target]
            else if values.contains target then values ++ [j]
            else values ++ [target]
            floydLoop length amount (j + 1) next fuel s1
      else (.ok values, s)

def shuffleLoop :
    Nat → Array Nat → Stream → Except Failure (Array Nat) × Stream
  | 0, values, s => (.ok values, s)
  | i + 1, values, s =>
      match range32 0 (i + 1) s with
      | (.error cause, s1) => (.error cause, s1)
      | (.ok target, s1) =>
          shuffleLoop i (Index.swapAt values (i + 1) target) s1

def sampleFloyd (length amount : Nat) (s : Stream) :
    Except Failure (List Nat) × Stream :=
  match floydLoop length amount (length - amount) [] amount s with
  | (.error cause, s1) => (.error cause, s1)
  | (.ok values, s1) =>
      if amount < 50 then (.ok values, s1)
      else
        match shuffleLoop (amount - 1) values.toArray s1 with
        | (.error cause, s2) => (.error cause, s2)
        | (.ok arr, s2) => (.ok arr.toList, s2)

def inplaceLoop (length amount : Nat) :
    Nat → Array Nat → Nat → Stream → Except Failure (Array Nat) × Stream
  | i, values, 0, s =>
      if i < amount then (.error .exhausted, s) else (.ok values, s)
  | i, values, fuel + 1, s =>
      if i < amount then
        match range32 i (length - 1) s with
        | (.error cause, s1) => (.error cause, s1)
        | (.ok target, s1) =>
            inplaceLoop length amount (i + 1) (Index.swapAt values i target) fuel s1
      else (.ok values, s)

def sampleInplace (length amount : Nat) (s : Stream) :
    Except Failure (List Nat) × Stream :=
  match inplaceLoop length amount 0 (List.range length).toArray amount s with
  | (.error cause, s1) => (.error cause, s1)
  | (.ok values, s1) => (.ok ((values.extract 0 amount).toList), s1)

def rangeLoop (range zone : Nat) : Nat → Stream → Except Failure Nat × Stream
  | 0, s => (.error .exhausted, s)
  | fuel + 1, s =>
      let (word, s1) := nextWord s
      let product := word.toNat * range
      let high := product / two64
      let low := product % two64
      if low ≤ zone then (.ok high, s1)
      else rangeLoop range zone fuel s1

def rangeSample (upper : Nat) (s : Stream) : Except Failure Nat × Stream :=
  if upper == 0 || two64 < upper then (.error .panic, s)
  else
    let range := if upper == two64 then 0 else upper
    if range == 0 then
      let (word, s1) := nextWord s
      (.ok word.toNat, s1)
    else
      let ints := (two64 - range) % range
      let zone := two64 - 1 - ints
      rangeLoop range zone rejectFuel s

def reject32 (length amount : Nat) :
    Nat → List Nat → Nat → Stream → Except Failure (List Nat) × Stream
  | count, values, 0, s =>
      if count < amount then (.error .exhausted, s) else (.ok values, s)
  | count, values, fuel + 1, s =>
      if count < amount then
        match range32 0 (length - 1) s with
        | (.error cause, s1) => (.error cause, s1)
        | (.ok target, s1) =>
            if values.contains target then
              reject32 length amount count values fuel s1
            else
              reject32 length amount (count + 1) (values ++ [target]) fuel s1
      else (.ok values, s)

def reject64 (length amount : Nat) :
    Nat → List Nat → Nat → Stream → Except Failure (List Nat) × Stream
  | count, values, 0, s =>
      if count < amount then (.error .exhausted, s) else (.ok values, s)
  | count, values, fuel + 1, s =>
      if count < amount then
        match rangeSample length s with
        | (.error cause, s1) => (.error cause, s1)
        | (.ok target, s1) =>
            if values.contains target then
              reject64 length amount count values fuel s1
            else
              reject64 length amount (count + 1) (values ++ [target]) fuel s1
      else (.ok values, s)

def sampleIndex (length amount : Nat) (s : Stream) :
    Except Failure (List Nat) × Stream :=
  if amount > length then (.error .panic, s)
  else if length > Index.max32 then
    reject64 length amount 0 [] rejectFuel s
  else if amount < 163 then
    if Index.shortInplace length amount then sampleInplace length amount s
    else sampleFloyd length amount s
  else if Index.largeInplace length amount then sampleInplace length amount s
  else reject32 length amount 0 [] rejectFuel s

/-- `FitterAsk.chooseRows` on the total private stream. -/
def chooseRows (total requested : Nat) (s : Stream) :
    Except Failure (List Nat) × Stream :=
  let n := min requested total
  if n == total then (.ok (List.range n), s)
  else sampleIndex total n s

def scaleMany (safe : List Word) :
    List Bounds.Batch → Except FitterAsk.AskError (List Bounds.Batch)
  | [] => .ok []
  | batch :: rest => do
      let first ← FitterAsk.scaleBatch batch safe
      let tail ← scaleMany safe rest
      return first :: tail

/-- `FitterAsk.subsample` (`subsample_loglik`) on the total private stream. -/
def subsample (ops : Ops) (model : Batch.Model) (modelObs : Nat)
    (x y : Fitter.Matrix) (params : List Params) (p : Nat) (stds : List Word)
    (s : Stream) :
    Except Failure (Except FitterAsk.AskError (List Word)) × Stream :=
  if x.rows != y.rows || p == 0 || params.isEmpty then
    (.ok (.error .fitInput), s)
  else if x.rows == 0 || modelObs ≤ 1 then
    (.ok (.ok (List.replicate params.length 0)), s)
  else if Fitter.anyNonfinite y.data then
    (.ok (.ok (List.replicate params.length Bounds.negInf)), s)
  else
    let amount := min p x.rows
    match chooseRows x.rows amount s with
    | (.error cause, s1) => (.error cause, s1)
    | (.ok indices, s1) =>
        match FitterAsk.pickMatrix x indices, FitterAsk.pickMatrix y indices with
        | .error cause, _ => (.ok (.error cause), s1)
        | _, .error cause => (.ok (.error cause), s1)
        | .ok xSel, .ok ySel =>
            match Batch.fitPost ops.sqrt ops.backend model xSel params with
            | .error cause => (.ok (.error cause), s1)
            | .ok post =>
                if !FitterAsk.shapePost params.length amount y.cols post.mu ||
                    !FitterAsk.shapePost params.length amount y.cols post.se then
                  (.ok (.error .postShape), s1)
                else
                  let safe := FitterAsk.safeStds stds
                  match FitterAsk.scaleBatch ySel.data safe with
                  | .error cause => (.ok (.error cause), s1)
                  | .ok yScaled =>
                      match scaleMany safe post.mu, scaleMany safe post.se with
                      | .error cause, _ => (.ok (.error cause), s1)
                      | _, .error cause => (.ok (.error cause), s1)
                      | .ok mu, .ok se =>
                          (.ok (FitterAsk.scorePosts ops.log ops.powi yScaled mu se), s1)

def prefixLoop (length amount i : Nat) (indices : Array Nat) (s : Stream) :
    Except Failure (Array Nat) × Stream :=
  if h : i < amount then
    match rangeSample (length - i) s with
    | (.error cause, s1) => (.error cause, s1)
    | (.ok offset, s1) =>
        have _ : amount - (i + 1) < amount - i := Nat.sub_lt_sub_left h (Nat.lt_succ_self i)
        prefixLoop length amount (i + 1) (Index.swapAt indices i (i + offset)) s1
  else (.ok indices, s)
termination_by amount - i

def samplePrefix (length amount : Nat) (s : Stream) :
    Except Failure (List Nat) × Stream :=
  let capped := min amount length
  let initial := (List.range length).toArray
  match prefixLoop length capped 0 initial s with
  | (.error cause, s1) => (.error cause, s1)
  | (.ok indices, s1) => (.ok ((indices.extract 0 capped).toList), s1)

def fitCal (sqrtOp : SqrtOp) (postOp : Calibration.PostOp)
    (x y : Calibration.CalMatrix) (params : Params) (samples : Nat) (noise : Bool)
    (s : Stream) :
    Except Failure (Except Calibration.CalError Calibration.Calibrator) × Stream :=
  if x.rows < 2 || samples == 0 then
    (.ok (.ok (Calibration.identity y.cols)), s)
  else
    match samplePrefix x.rows (min samples x.rows) s with
    | (.error cause, s1) => (.error cause, s1)
    | (.ok indices, s1) =>
        match Calibration.pickMatrix x indices, Calibration.pickMatrix y indices with
        | .error cause, _ => (.ok (.error cause), s1)
        | _, .error cause => (.ok (.error cause), s1)
        | .ok xSel, .ok ySel =>
            match postOp xSel params noise with
            | .error cause => (.ok (.error cause), s1)
            | .ok (mu, se) =>
                (.ok (Calibration.fit sqrtOp mu ySel (some se)), s1)

def askCal (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (model : Batch.Model) (params : Params) (samples : Nat) (fitNoise : Bool)
    (s : Stream) :
    Except Failure (Except FitterAsk.AskError FitterAsk.Calibrator) × Stream :=
  let postOp := Compose.calPost sqrtOp backend model
  match fitCal sqrtOp postOp model.neighbor.trainX model.neighbor.trainY
      params samples fitNoise s with
  | (.error cause, s1) => (.error cause, s1)
  | (.ok (.ok cal), s1) => (.ok (.ok (Calibration.rows cal)), s1)
  | (.ok (.error _), s1) => (.ok (.error .calibrate), s1)

/-- `FitterAsk.commit` on the total private stream, updating `state.params` and
`state.calibrator` after optional affine calibration. -/
def commit (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (state : Fitter.State) (model : Batch.Model) (search : FitterAsk.Search)
    (best : Params) (fitNoise : Bool) (s : Stream) :
    Except Failure (Except FitterAsk.AskError (Fitter.State × Params)) × Stream :=
  if search.affine then
    match askCal sqrtOp backend model best search.samples fitNoise s with
    | (.error cause, s1) => (.error cause, s1)
    | (.ok (.error cause), s1) => (.ok (.error cause), s1)
    | (.ok (.ok cal), s1) =>
        (.ok (.ok ({ state with params := some best, calibrator := some cal }, best)), s1)
  else
    (.ok (.ok ({ state with params := some best, calibrator := none }, best)), s)

/-- `Compose.ask` (`FitterAsk.ask`) on the total private stream, including
training-row subsampling, warm-start candidate injection, and calibration commit. -/
def askStream (ops : Ops) (state : Fitter.State) (model : Batch.Model)
    (search : FitterAsk.Search) (warm : Option Params) (fitNoise : Bool) (s : Stream) :
    Except Failure FitterAsk.Outcome × Stream :=
  let fm := Compose.fitterModel model
  let dimMismatch : Bool :=
    match state.numDim with
    | some dim => dim != fm.dim
    | none => false
  if dimMismatch then
    (.ok (FitterAsk.fail state .dimension), s)
  else if fm.obs < 2 then
    match Fitter.newParams state.k Bounds.one 0 with
    | .error cause => (.ok (FitterAsk.fail state (.param cause)), s)
    | .ok best =>
        match commit ops.sqrt ops.backend state model search best fitNoise s with
        | (.error cause, s1) => (.error cause, s1)
        | (.ok (.error cause), s1) =>
            (.ok ⟨{ state with params := some best }, .error cause⟩, s1)
        | (.ok (.ok (committed, best)), s1) => (.ok ⟨committed, .ok best⟩, s1)
  else if state.yCount == 0 then
    (.ok (FitterAsk.fail state .needTell), s)
  else
    match randomParams ops.pow state.k search.candidates search.inferAleatoric s with
    | (.error cause, s1) => (.error cause, s1)
    | (.ok (.error cause), s1) => (.ok (FitterAsk.fail state (.param cause)), s1)
    | (.ok (.ok random), s1) =>
        match FitterAsk.addWarm state search warm random with
        | .error cause => (.ok (FitterAsk.fail state cause), s1)
        | .ok params =>
            let amount := min search.samples fm.obs
            match chooseRows fm.obs amount s1 with
            | (.error cause, s2) => (.error cause, s2)
            | (.ok indices, s2) =>
                match FitterAsk.trainRows fm indices with
                | .error cause => (.ok (FitterAsk.fail state cause), s2)
                | .ok (trainX, trainY) =>
                    let stds := Fitter.yStd ops.sqrt state
                    match subsample ops model fm.obs trainX trainY params
                        search.samples stds s2 with
                    | (.error cause, s3) => (.error cause, s3)
                    | (.ok (.error cause), s3) => (.ok (FitterAsk.fail state cause), s3)
                    | (.ok (.ok scores), s3) =>
                        match params[Fit.bestIndex scores]? with
                        | none => (.ok (FitterAsk.fail state .fitInput), s3)
                        | some best =>
                            match commit ops.sqrt ops.backend state model search
                                best fitNoise s3 with
                            | (.error cause, s4) => (.error cause, s4)
                            | (.ok (.error cause), s4) =>
                                (.ok ⟨{ state with params := some best }, .error cause⟩, s4)
                            | (.ok (.ok (committed, best)), s4) =>
                                (.ok ⟨committed, .ok best⟩, s4)

/-- Execute `Append.applyPlan` on the total private stream. -/
def runPlan (ops : Ops) (config : Append.Config) (plan : Surrogate.AppendPlan)
    (state : Append.FitState) (callerRng : RngState) (s : Stream) : Append.Step :=
  let syncErr : Option Append.Effect :=
    if plan.sync then
      match ops.gate .sync with
      | .error cause => some cause
      | .ok () => none
    else none
  match syncErr with
  | some cause => stepResult state callerRng s.consumed (some (.effect cause))
  | none =>
      match state with
      | .searched model search fitter warm _ =>
          if plan.search then
            match askStream ops fitter model.batch search warm config.fitNoise s with
            | (.error cause, s1) =>
                stepResult state callerRng s1.consumed (some (.seed cause))
            | (.ok outcome, s1) =>
                match outcome.result with
                | .error cause =>
                    stepResult
                      (.searched model search outcome.state
                        outcome.state.params outcome.state.calibrator)
                      callerRng s1.consumed (some (.search cause))
                | .ok best =>
                    let searched := Append.FitState.searched model search outcome.state
                      (some best) outcome.state.calibrator
                    let relErr : Option Append.FitError :=
                      if plan.release then
                        match ops.gate .release with
                        | .error cause => some (.effect cause)
                        | .ok () => none
                      else none
                    stepResult searched callerRng s1.consumed relErr
          else
            let relErr : Option Append.FitError :=
              if plan.release then
                match ops.gate .release with
                | .error cause => some (.effect cause)
                | .ok () => none
              else none
            stepResult state callerRng s.consumed relErr
      | _ =>
          let relErr : Option Append.FitError :=
            if plan.release then
              match ops.gate .release with
              | .error cause => some (.effect cause)
              | .ok () => none
            else none
          stepResult state callerRng s.consumed relErr

/-- Executable total surrogate append: four caller words seed a total ChaCha12 stream.
Frozen fit plans and non-search branches execute directly with zero random draws.
Subsampling, candidate parameter generation, warm-start injection, and affine calibration
advance the stream on demand without fixed caps. -/
def append (ops : Ops) (config : Append.Config) (state : Append.FitState)
    (rng : RngState) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) : Append.Step :=
  let seeded := Sampling.run Surrogate.localSeed rng
  match seeded.1 with
  | .error cause => ⟨state, seeded.2, none, some (.seed cause)⟩
  | .ok seed =>
      let stream := fromSeed seed
      match state with
      | .frozen model params =>
          let added := Append.addModel ops.sqrt ops.log ops.gate model x y variance
          match added.error with
          | some cause =>
              stepResult (.frozen added.model params) seeded.2 stream.consumed
                (some (.model cause))
          | none =>
              let plan := Surrogate.appendPlan
                ⟨config.onDisk, x.rows, config.mode, true, false⟩
              stepResult (.frozen added.model params) seeded.2 stream.consumed
                (gateError ops plan)
      | .unfitted =>
          match Append.makeModel (innerOps ops 0) config x y variance with
          | .error cause =>
              stepResult .unfitted seeded.2 stream.consumed (some (.model cause))
          | .ok model =>
              let plan := Surrogate.appendPlan
                ⟨config.onDisk, x.rows, config.mode, false, true⟩
              match config.mode with
              | .frozen =>
                  match Fitter.newParams config.k Bounds.one 0 with
                  | .error cause =>
                      stepResult .unfitted seeded.2 stream.consumed (some (.params cause))
                  | .ok params =>
                      runPlan ops config plan (.frozen model params) seeded.2 stream
              | .draw =>
                  let fitter := { Fitter.blank with k := config.k }
                  let told := Fitter.tell ops.log fitter x y variance (some config.bounds)
                  match told.error with
                  | some cause =>
                      stepResult .unfitted seeded.2 stream.consumed (some (.fitter cause))
                  | none =>
                      let initial := Append.FitState.searched model config.search
                        told.state none none
                      runPlan ops config plan initial seeded.2 stream
      | .searched model search fitter params calibrator =>
          let added := Append.addModel ops.sqrt ops.log ops.gate model x y variance
          match added.error with
          | some cause =>
              stepResult (.searched added.model search fitter params calibrator)
                seeded.2 stream.consumed (some (.model cause))
          | none =>
              let told := Fitter.tell ops.log fitter x y variance (some config.bounds)
              let nextParams := params.or told.state.params
              let nextCal := calibrator.or told.state.calibrator
              let next := Append.FitState.searched added.model search told.state
                nextParams nextCal
              match told.error with
              | some cause =>
                  stepResult next seeded.2 stream.consumed (some (.fitter cause))
              | none =>
                  let plan := Surrogate.appendPlan
                    ⟨config.onDisk, x.rows, config.mode, params.isSome, false⟩
                  runPlan ops config plan next seeded.2 stream

/-- The four-word caller seed draw consumes exactly 4 words and preserves stream order. -/
theorem seedFour (a b c d : Word) (rest : List Word) (used : Nat) :
    Sampling.run Surrogate.localSeed
      { words := a :: b :: c :: d :: rest, consumed := used } =
        (.ok #[a, b, c, d], { words := rest, consumed := used + 4 }) := by
  rfl

/-- Caller seed prefix exhaustion preserves partial progress and consumed count. -/
theorem shortExhaust (a b c : Word) (used : Nat) :
    Sampling.run Surrogate.localSeed
      { words := [a, b, c], consumed := used } =
        (.error .exhausted, { words := [], consumed := used + 3 }) := by
  rfl

/-- Total stream `takeWords 0` is an identity on stream state. -/
theorem takeZero (s : Stream) :
    takeWords 0 s = ([], s) := by
  rfl

/-- One word draw matches `nextWord`. -/
theorem takeOne (s : Stream) :
    takeWords 1 s = ([nextWord s |>.1], (nextWord s).2) := by
  rfl

/-- Fitter plan execution on frozen models does not draw random words. -/
theorem frozenZero (ops : Append.Ops) (config : Append.Config)
    (plan : Surrogate.AppendPlan) (model : Append.Model) (params : Params)
    (rng : RngState) (hSync : ops.gate .sync = .ok ()) (hRel : ops.gate .release = .ok ()) :
    (Sampling.run (Append.applyPlan ops config plan (.frozen model params)) rng).2 = rng := by
  simp [Append.applyPlan, Sampling.run, hSync, hRel]
  rfl

/-- Uncalibrated surrogate sampling consumes exactly one private seed word. -/
theorem uncalDraw (ops : Draw.Ops) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Select.Context)
    (queries : Fitter.Matrix) (count : Nat) (hCal : ctx.calibrator = none)
    (w : Word) (rest : List Word) :
    let drawAction := Draw.sample ops expOp backend ctx queries count
    (Sampling.run drawAction { words := w :: rest }).2.consumed = 1 := by
  dsimp
  unfold Draw.sample
  rw [hCal]
  dsimp
  cases Batch.drawData ops.sqrt backend ctx.model queries ctx.params false false <;> rfl

/-- Drawing one word steps the stream and passes the drawn word to the continuation. -/
theorem stepWord (f : Word → Draw α) (w : Word) (rest : List Word) (used : Nat) :
    Sampling.run (Sampling.nextWord >>= f) { words := w :: rest, consumed := used } =
      Sampling.run (f w) { words := rest, consumed := used + 1 } := by
  rfl

/-- Calibrated surrogate sampling consumes exactly two private seed words. -/
theorem calDraw (ops : Draw.Ops) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Select.Context)
    (queries : Fitter.Matrix) (count : Nat) (cal : Calibration.Calibrator)
    (hCal : ctx.calibrator = some cal) (w1 w2 : Word) (rest : List Word) :
    let drawAction := Draw.sample ops expOp backend ctx queries count
    (Sampling.run drawAction { words := w1 :: w2 :: rest }).2.consumed = 2 := by
  dsimp
  unfold Draw.sample
  rw [hCal]
  dsimp
  rw [stepWord, stepWord]
  repeat (first | rfl | split)

/-- Argmax tie-breaking with unique maximum consumes zero random words. -/
theorem uniqueZero (values : List Word) (rng : RngState)
    (hUnique : (Acq.ties values).length = 1) :
    Sampling.run (Acq.argmaxTie values) rng =
      (.ok (Acq.ties values)[0]!, rng) := by
  simp [Acq.argmaxTie, Sampling.run]
  split
  · next hEmpty =>
      have hTies : (Acq.ties values).length = 0 := by
        simp [Acq.ties, hEmpty]
      omega
  · next hNonEmpty =>
      rfl

/-- Empty candidate selection in acquisition immediately returns error without draws. -/
theorem emptyArm (arms : Nat) (rng : RngState) :
    Sampling.run (Acq.randomSelect 0 arms) rng =
      (.ok (.error .empty), rng) := by
  rfl

/-- Argmax tie-breaking with empty score list consumes zero words and returns zero. -/
theorem tieEmpty (rng : RngState) :
    Sampling.run (Acq.argmaxTie []) rng =
      (.ok 0, rng) := by
  rfl

/-- Sequential draw order composition theorem: running two independent actions
chains the consumed word count additively. -/
theorem composeWords (a b : Word) (rest : List Word) (used : Nat) :
    Sampling.run (do let w1 ← Sampling.nextWord; let w2 ← Sampling.nextWord; pure (w1, w2))
      { words := a :: b :: rest, consumed := used } =
        (.ok (a, b), { words := rest, consumed := used + 2 }) := by
  rfl

/-- Half-word followed by whole-word draw order composition theorem. -/
theorem composeMixed (a b : Word) (rest : List Word) (used : Nat) :
    Sampling.run (do let h ← Sampling.nextHalf; let w ← Sampling.nextWord; pure (h, w))
      { words := a :: b :: rest, consumed := used } =
        let low := BitVec.ofNat 32 a.toNat
        let high := BitVec.ofNat 32 (a.toNat >>> 32)
        let nextLow := BitVec.ofNat 32 b.toNat
        let combined := BitVec.ofNat 64 (high.toNat + (nextLow.toNat <<< 32))
        (.ok (low, combined),
          { words := rest, pending := [BitVec.ofNat 32 (b.toNat >>> 32)],
            consumed := used + 2 }) := by
  rfl

/-- Surrogate append on a frozen model consumes zero stream words and leaves private RNG empty. -/
theorem frozenAppend (ops : Ops) (config : Append.Config) (model : Append.Model)
    (params : Params) (a b c d : Word) (rest : List Word) (used : Nat)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) :
    (append ops config (.frozen model params)
      { words := a :: b :: c :: d :: rest, consumed := used } x y variance).privateRng =
        some { words := [], consumed := 0 } := by
  dsimp [append, stepResult]
  cases Append.addModel ops.sqrt ops.log ops.gate model x y variance |>.error <;> rfl

/-- When requested samples cover all observations, `chooseRows` returns `0..total`
without consuming stream words. -/
theorem chooseAll (total requested : Nat) (s : Stream) (h : total ≤ requested) :
    chooseRows total requested s = (.ok (List.range total), s) := by
  have hmin : min requested total = total := Nat.min_eq_right h
  simp [chooseRows, hmin]

/-- Uncalibrated commit updates `state.params` and clears `state.calibrator`
without consuming stream words. -/
theorem commitPlain (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (state : Fitter.State) (model : Batch.Model) (search : FitterAsk.Search)
    (best : Params) (fitNoise : Bool) (s : Stream)
    (hcal : search.affine = false) :
    commit sqrtOp backend state model search best fitNoise s =
      (.ok (.ok ({ state with params := some best, calibrator := none }, best)), s) := by
  simp [commit, hcal]

/-- Calibrated commit updates both `state.params` and `state.calibrator` on the stream. -/
theorem commitCal (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (state : Fitter.State) (model : Batch.Model) (search : FitterAsk.Search)
    (best : Params) (fitNoise : Bool) (s s1 : Stream) (cal : FitterAsk.Calibrator)
    (hcal : search.affine = true)
    (hask : askCal sqrtOp backend model best search.samples fitNoise s = (.ok (.ok cal), s1)) :
    commit sqrtOp backend state model search best fitNoise s =
      (.ok (.ok ({ state with params := some best, calibrator := some cal }, best)), s1) := by
  simp [commit, hcal, hask]

#print axioms divDisjoint
#print axioms seedFour
#print axioms shortExhaust
#print axioms takeZero
#print axioms takeOne
#print axioms frozenZero
#print axioms uncalDraw
#print axioms stepWord
#print axioms calDraw
#print axioms uniqueZero
#print axioms emptyArm
#print axioms tieEmpty
#print axioms composeWords
#print axioms composeMixed
#print axioms frozenAppend
#print axioms chooseAll
#print axioms commitPlain
#print axioms commitCal

end BauxResearch.Semantics.PrivateRng
