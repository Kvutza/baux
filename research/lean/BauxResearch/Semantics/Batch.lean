import BauxResearch.Semantics.Neighbor
import BauxResearch.Semantics.FitterAsk

/-!
# ENN 0.4.1 batch posterior API

This module composes neighbor refinement and weighted statistics into the batch
posterior operation used by fitter likelihood evaluation. Equal-k parameter
batches share one neighbor query; unequal-k batches query separately, matching
the upstream control split.
-/

namespace BauxResearch.Semantics.Batch

open Sampling Bounds Fitter

abbrev SqrtOp := Word → Word

inductive ApiError where
  | neighbor
  | shape
  | empty
  deriving Repr, DecidableEq

structure Model where
  neighbor : Neighbor.Model
  yvar : Option Fitter.Matrix
  yScale : List Word
  deriving Repr, DecidableEq

structure DrawData where
  indices : List (List Nat)
  stats : Posterior.Stats
  deriving Repr, DecidableEq

def gatherVar (model : Model) (indices : List (List Nat)) :
    Except ApiError (Option Batch) :=
  match model.yvar with
  | none => .ok none
  | some values =>
      match Neighbor.gatherY values.data indices with
      | .ok rows => .ok (some rows)
      | .error _ => .error .neighbor

def emptyStats (queries metrics : Nat) : Posterior.Stats :=
  let ones := List.replicate queries (List.replicate metrics Bounds.one)
  let zeros := List.replicate queries (List.replicate metrics 0)
  ⟨List.replicate queries [], ones, zeros, ones, ones, zeros⟩

def fromNeighbors (sqrtOp : SqrtOp) (model : Model)
    (queries : Fitter.Matrix) (params : Params) (noise : Bool)
    (neighbors : Neighbor.Neighbors) : Except ApiError Posterior.Stats := do
  let yvar ← gatherVar model neighbors.indices
  let input : Posterior.Input :=
    ⟨queries.rows, neighbors.k, model.neighbor.trainY.cols,
      neighbors.dist2s, neighbors.y, yvar, model.yScale⟩
  match Posterior.compute sqrtOp input params noise with
  | .ok stats => return stats
  | .error _ => throw .shape

def one (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp) (model : Model)
    (queries : Fitter.Matrix) (params : Params) (exclude noise : Bool) :
    Except ApiError Posterior.Stats := do
  let count := params.k.toNat
  match Neighbor.getData backend model.neighbor queries count exclude with
  | .error _ => throw .neighbor
  | .ok none => return emptyStats queries.rows model.neighbor.trainY.cols
  | .ok (some neighbors) => fromNeighbors sqrtOp model queries params noise neighbors

/-- The same posterior computation while retaining neighbor indices required by
the correlated function-draw path. -/
def drawData (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp) (model : Model)
    (queries : Fitter.Matrix) (params : Params) (exclude noise : Bool) :
    Except ApiError DrawData := do
  let count := params.k.toNat
  match Neighbor.getData backend model.neighbor queries count exclude with
  | .error _ => throw .neighbor
  | .ok none => return ⟨List.replicate queries.rows [],
      emptyStats queries.rows model.neighbor.trainY.cols⟩
  | .ok (some neighbors) =>
      let stats ← fromNeighbors sqrtOp model queries params noise neighbors
      return ⟨neighbors.indices, stats⟩

def sameK : List Params → Bool
  | [] => false
  | first :: rest => rest.all (fun params => params.k == first.k)

def shared (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp) (model : Model)
    (queries : Fitter.Matrix) (params : List Params) (exclude noise : Bool) :
    Except ApiError (List Posterior.Stats) := do
  let first ← match params with
    | [] => throw .empty
    | value :: _ => pure value
  match Neighbor.getData backend model.neighbor queries first.k.toNat exclude with
  | .error _ => throw .neighbor
  | .ok none =>
      return List.replicate params.length
        (emptyStats queries.rows model.neighbor.trainY.cols)
  | .ok (some neighbors) =>
      params.mapM fun current =>
        fromNeighbors sqrtOp model queries current noise neighbors

def separate (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp) (model : Model)
    (queries : Fitter.Matrix) (params : List Params) (exclude noise : Bool) :
    Except ApiError (List Posterior.Stats) :=
  params.mapM fun current => one sqrtOp backend model queries current exclude noise

/-- The `batch_posterior` control split and output stack used by fitter
likelihood evaluation. -/
def batch (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp) (model : Model)
    (queries : Fitter.Matrix) (params : List Params) (exclude noise : Bool) :
    Except ApiError FitterAsk.Post := do
  if params.isEmpty then throw .empty
  let stats ← if sameK params && model.neighbor.trainX.rows > 0 then
    shared sqrtOp backend model queries params exclude noise
  else separate sqrtOp backend model queries params exclude noise
  return ⟨stats.map (fun result => result.mu), stats.map (fun result => result.se)⟩

/-- Concrete fitter posterior adapter. The fitter always requests leave-one-out
neighbors and observation noise. -/
def fitPost (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (model : Model) : FitterAsk.PostOp := fun queries params =>
  match batch sqrtOp backend model queries params true true with
  | .ok result => .ok result
  | .error _ => .error .postShape

def noBackend : Neighbor.BackendOp := fun _ _ _ => .error .backend

def emptyModel : Model :=
  ⟨⟨⟨0, 1, []⟩, ⟨0, 1, []⟩, false, [Bounds.one], .flat⟩,
    none, [Bounds.one]⟩

theorem emptyOne (sqrtOp : SqrtOp) (queries : Fitter.Matrix)
    (params : Params) :
    one sqrtOp noBackend emptyModel queries params false true =
      .ok (emptyStats queries.rows 1) := by
  simp [one, emptyModel, Neighbor.getData, emptyStats]
  rfl

theorem emptyBatch (sqrtOp : SqrtOp) (backend : Neighbor.BackendOp)
    (model : Model) (queries : Fitter.Matrix) :
    batch sqrtOp backend model queries [] true true = .error .empty := by
  rfl

#print axioms emptyOne
#print axioms emptyBatch

end BauxResearch.Semantics.Batch
