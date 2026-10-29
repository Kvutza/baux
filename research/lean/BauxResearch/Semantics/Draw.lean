import BauxResearch.Semantics.Hash
import BauxResearch.Semantics.Select

/-!
# ENN 0.4.1 correlated function draws

This module executes the reachable `posterior_function_draw_warped`, surrogate
sample, and Thompson selection arithmetic. Hash normals are shared by neighbor
identity, matching the upstream correlated noise field.
-/

namespace BauxResearch.Semantics.Draw

open Sampling Bounds Fitter

abbrev Cube := List Batch

structure Ops where
  ln : Hash.LnOp
  sqrt : Hash.SqrtOp
  cos : Hash.CosOp

def drawCell (ops : Ops) (data : Batch.DrawData) (seed : Word)
    (query metric : Nat) : Word :=
  let indices := data.indices[query]!
  let weighted := (List.range indices.length).foldl (fun total neighbor =>
    let index := BitVec.ofNat 64 indices[neighbor]!
    let normal := Hash.indexNormal ops.ln ops.sqrt ops.cos seed index metric
    let weight := ((data.stats.weights[query]!)[neighbor]!)[metric]!
    add total (mul weight normal)) 0
  let l2 := (data.stats.l2[query]!)[metric]!
  let safe := Acq.maxWord l2 Acq.clipMin
  let scale := divide (data.stats.seEpi[query]!)[metric]! safe
  let base := add (data.stats.mu[query]!)[metric]! (mul scale weighted)
  let ale := (data.stats.seAle[query]!)[metric]!
  if Float.ltDouble 0 ale then
    add base (mul ale (Hash.queryNormal ops.ln ops.sqrt ops.cos seed query metric))
  else base

def drawOne (ops : Ops) (data : Batch.DrawData) (seed : Word)
    (queries metrics : Nat) : Batch :=
  (List.range queries).map fun query =>
    (List.range metrics).map fun metric => drawCell ops data seed query metric

def drawMany (ops : Ops) (data : Batch.DrawData) (base : Word)
    (count queries metrics : Nat) : Cube :=
  (List.range count).map fun offset =>
    drawOne ops data (base + BitVec.ofNat 64 offset) queries metrics

def naturalRow (expOp : ExpOp) : List Word → List Bound →
    Except Select.SelectError (List Word)
  | [], [] => .ok []
  | value :: values, (low, high) :: bounds =>
      match naturalRow expOp values bounds with
      | .ok rest => .ok (Bounds.inverse expOp value low high :: rest)
      | .error cause => .error cause
  | _, _ => .error .posterior

def naturalBatch (expOp : ExpOp) (bounds : List Bound) : Batch →
    Except Select.SelectError Batch
  | [] => .ok []
  | row :: rows => match naturalRow expOp row bounds,
      naturalBatch expOp bounds rows with
    | .ok first, .ok rest => .ok (first :: rest)
    | .error cause, _ => .error cause
    | _, .error cause => .error cause

def naturalCube (expOp : ExpOp) (bounds : List Bound) (draws : Cube) :
    Except Select.SelectError Cube :=
  draws.mapM (naturalBatch expOp bounds)

def calCube (cal : Calibration.Calibrator) (bounds : List Bound)
    (mu se : Batch) (draws : Cube) : Except Select.SelectError Cube :=
  draws.mapM fun draw =>
    (Calibration.applyDraw cal (Select.matrix draw)
      (Select.matrix mu) (Select.matrix se) bounds).map
        (fun result => result.data) |>.mapError (fun _ => .posterior)

/-- `ENNSurrogate::sample`, including the otherwise-discarded first seed word
in the calibrated branch. -/
def sample (ops : Ops) (expOp : ExpOp) (backend : Neighbor.BackendOp)
    (ctx : Select.Context) (queries : Fitter.Matrix) (count : Nat) :
    Draw (Except Select.SelectError Cube) := do
  let first ← nextWord
  match ctx.calibrator with
  | none =>
      match Batch.drawData ops.sqrt backend ctx.model queries ctx.params false false with
      | .error _ => return .error .posterior
      | .ok data => return .ok (drawMany ops data first count queries.rows
          ctx.model.neighbor.trainY.cols)
  | some calibrator =>
      let base ← nextWord
      match Batch.drawData ops.sqrt backend ctx.model queries ctx.params false false with
      | .error _ => return .error .posterior
      | .ok data =>
          let warped := drawMany ops data base count queries.rows
            ctx.model.neighbor.trainY.cols
          match naturalCube expOp ctx.bounds warped with
          | .error _ => return .error .posterior
          | .ok draws =>
              -- `calibrated_sample` calls `model.posterior` again after the
              -- function draw. Keep that second backend call and failure point.
              match Batch.one ops.sqrt backend ctx.model queries ctx.params false false with
              | .error _ => return .error .posterior
              | .ok stats =>
                  match Bounds.naturalize expOp ctx.bounds stats.mu stats.se
                      stats.seEpi stats.seAle with
                  | .error _ => return .error .posterior
                  | .ok post => return calCube calibrator ctx.bounds post.mu post.se draws

def oneScores : Batch → Except Select.SelectError (List Word)
  | [] => .ok []
  | row :: rows => match row, oneScores rows with
    | value :: _, .ok rest => .ok (value :: rest)
    | _, _ => .error .posterior

def plainLoop : Cube → Draw (Except Select.SelectError (List Nat))
  | [] => pure (.ok [])
  | draw :: draws => do
      match oneScores draw with
      | .error cause => return .error cause
      | .ok scores =>
          let index ← Acq.argmaxTie scores
          match ← plainLoop draws with
          | .ok rest => return .ok (index :: rest)
          | .error cause => return .error cause

def maskScores (scores : List Word) (chosen : List Nat) : List Word :=
  (List.range scores.length).map fun index =>
    if chosen.contains index then Bounds.negInf else scores[index]!

def morboLoop (config : Select.Morbo) : Cube → List Nat →
    Draw (Except Select.SelectError (List Nat))
  | [], chosen => pure (.ok chosen)
  | draw :: draws, chosen => do
      match Morbo.scalarize (Select.dense draw config.metrics)
          config.low config.high config.weights config.alpha config.metrics false with
      | .error _ => return .error .scalar
      | .ok scores =>
          let index ← Acq.argmaxTie (maskScores scores.toList chosen)
          morboLoop config draws (chosen ++ [index])

/-- Thompson arm indices after the source's unconditional naturalization at
the strategy layer. Calibrated samples therefore follow the upstream second
naturalization as written. -/
def select (ops : Ops) (expOp : ExpOp) (backend : Neighbor.BackendOp)
    (ctx : Select.Context) (candidates : Sampling.Matrix) (arms : Nat)
    (morbo : Option Select.Morbo) :
    Draw (Except Select.SelectError Sampling.Matrix) := do
  match ← sample ops expOp backend ctx (Select.queryMatrix candidates) arms with
  | .error cause => return .error cause
  | .ok raw => match naturalCube expOp ctx.bounds raw with
    | .error cause => return .error cause
    | .ok draws =>
        let indices ← match morbo with
          | none => plainLoop draws
          | some config => morboLoop config draws []
        match indices with
        | .error cause => return .error cause
        | .ok values => return Select.pick candidates values

/-- Complete `strategy::select_arms` dispatch. Every acquisition mode uses the
source random fallback when no fitted surrogate is present. -/
def selectAll (ops : Ops) (expOp : ExpOp) (backend : Neighbor.BackendOp)
    (ctx : Option Select.Context) (mode : Select.Mode)
    (candidates : Sampling.Matrix) (arms : Nat) :
    Draw (Except Select.SelectError Sampling.Matrix) :=
  match mode with
  | .thompson morbo => match ctx with
    | none => Select.randomPick candidates arms
    | some value => select ops expOp backend value candidates arms morbo
  | other => Select.select ops.sqrt expOp backend ctx other candidates arms

theorem emptyNatural (expOp : ExpOp) (bounds : List Bound) :
    naturalCube expOp bounds [] = .ok [] := by
  rfl

theorem absentThompson (ops : Ops) (expOp : ExpOp)
    (backend : Neighbor.BackendOp) (candidates : Sampling.Matrix)
    (arms : Nat) (morbo : Option Select.Morbo) :
    selectAll ops expOp backend none (.thompson morbo) candidates arms =
      Select.randomPick candidates arms := by
  rfl

#print axioms emptyNatural
#print axioms absentThompson

end BauxResearch.Semantics.Draw
