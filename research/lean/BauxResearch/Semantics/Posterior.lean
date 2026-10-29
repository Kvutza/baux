import BauxResearch.Semantics.Fitter
import BauxResearch.Extraction.Native
import BauxResearch.Generated.Pure.Funs

/-!
# ENN 0.4.1 weighted posterior arithmetic (`posterior.rs` and `posterior/*`)

This module executes `compute_weighted_stats_impl`,
`stats_from_neighbor_weights`, and `se_from_variance_components` at binary64
rounding points. Neighbor discovery and backend row gathering supply the input
arrays; they are deliberately not replaced by guessed neighbors here.

Covers all 27 reachable functions across:
- `crates/ennbo/src/posterior.rs` (10 functions):
  - `rust:ennbo 0.4.1 posterior/se_from_variance_components()`
  - `rust:ennbo 0.4.1 posterior/compute_weighted_stats_impl()`
  - `rust:ennbo 0.4.1 posterior/compute_weighted_stats()`
  - `rust:ennbo 0.4.1 posterior/compute_weighted_stats_into()`
  - `rust:ennbo 0.4.1 posterior/stats_from_neighbor_weights()`
  - `rust:ennbo 0.4.1 posterior/stats_from_neighbor_weights_into()`
  - `rust:ennbo 0.4.1 posterior/EnnModel#posterior()`
  - `rust:ennbo 0.4.1 posterior/EnnModel#posterior_into()`
  - `rust:ennbo 0.4.1 posterior/EnnModel#batch_posterior()`
  - `rust:ennbo 0.4.1 posterior/EnnModel#batch_posterior_into()`
- `crates/ennbo/src/posterior/dual_api.rs` (5 functions):
  - `rust:ennbo 0.4.1 posterior/dual_api/EnnModel#dual_posterior()`
  - `rust:ennbo 0.4.1 posterior/dual_api/EnnModel#dual_posterior_into()`
  - `rust:ennbo 0.4.1 posterior/dual_api/EnnModel#dual_batch_posterior()`
  - `rust:ennbo 0.4.1 posterior/dual_api/EnnModel#dual_batch_posterior_into()`
  - `rust:ennbo 0.4.1 posterior/dual_api/compute_dual_stats()`
- `crates/ennbo/src/posterior/light.rs` (4 functions):
  - `rust:ennbo 0.4.1 posterior/light/EnnModel#posterior_light()`
  - `rust:ennbo 0.4.1 posterior/light/EnnModel#posterior_light_into()`
  - `rust:ennbo 0.4.1 posterior/light/compute_light_stats()`
  - `rust:ennbo 0.4.1 posterior/light/compute_light_stats_into()`
- `crates/ennbo/src/posterior/neighbor.rs` (4 functions):
  - `rust:ennbo 0.4.1 posterior/neighbor/EnnModel#get_neighbor_data()`
  - `rust:ennbo 0.4.1 posterior/neighbor/EnnModel#gather_y()`
  - `rust:ennbo 0.4.1 posterior/neighbor/EnnModel#gather_yvar()`
  - `rust:ennbo 0.4.1 posterior/neighbor/filter_nearest()`
- `crates/ennbo/src/posterior/draw_compute.rs` (3 functions):
  - `rust:ennbo 0.4.1 posterior/draw_compute/EnnModel#posterior_function_draw_warped()`
  - `rust:ennbo 0.4.1 posterior/draw_compute/EnnModel#posterior_function_draw_warped_into()`
  - `rust:ennbo 0.4.1 posterior/draw_compute/compute_draw_from_neighbors()`
- `crates/ennbo/src/posterior/neighbor_dist.rs` (1 function):
  - `rust:ennbo 0.4.1 posterior/neighbor_dist/EnnModel#nearest_neighbor_distances()`
-/

namespace BauxResearch.Semantics.Posterior

open FloatLib.Floats.Formats.BinaryInterchange
open Sampling Bounds Fitter

abbrev SqrtOp := Word → Word

def epsVar : Word :=
  (Model.roundRat FloatFormat.binary64 false 1 1000000000).bits

structure Input where
  queries : Nat
  neighbors : Nat
  metrics : Nat
  dist2s : Batch
  y : Batch
  yvar : Option Batch
  yScale : List Word
  deriving Repr, DecidableEq

structure Stats where
  weights : List Batch
  l2 : Batch
  mu : Batch
  se : Batch
  seEpi : Batch
  seAle : Batch
  deriving Repr, DecidableEq

structure Parts where
  se : Word
  epi : Word
  ale : Word
  deriving Repr, DecidableEq

inductive PostError where
  | shape
  deriving Repr, DecidableEq

def matrixShape (rows cols : Nat) (values : Batch) : Bool :=
  values.length == rows && values.all (fun row => row.length == cols)

def validInput (input : Input) : Bool :=
  matrixShape input.queries input.neighbors input.dist2s &&
    matrixShape (input.queries * input.neighbors) input.metrics input.y &&
    input.yScale.length == input.metrics &&
    match input.yvar with
    | none => true
    | some values =>
        matrixShape (input.queries * input.neighbors) input.metrics values

def at2 (values : Batch) (row col : Nat) : Except PostError Word :=
  match values[row]? with
  | none => .error .shape
  | some data => match data[col]? with
    | none => .error .shape
    | some value => .ok value

def mapN (count : Nat) (f : Nat → Except PostError α) :
    Except PostError (List α) :=
  let rec loop : Nat → List α → Except PostError (List α)
    | 0, values => .ok values.reverse
    | left + 1, values => do
        let index := count - (left + 1)
        let value ← f index
        loop left (value :: values)
  loop count []

def sumWords (values : List Word) : Word := values.foldl add 0

def seParts (sqrtOp : SqrtOp) (epi ale scale : Word) : Parts :=
  let total := add epi ale
  if Float.ltDouble total epsVar then
    let value := mul (sqrtOp epsVar) scale
    ⟨value, value, 0⟩
  else
    ⟨mul (sqrtOp total) scale,
      mul (sqrtOp epi) scale,
      mul (sqrtOp ale) scale⟩

def rawWeights (input : Input) (params : Params) (query metric : Nat) :
    Except PostError (List Word) :=
  mapN input.neighbors fun neighbor => do
    let distance ← at2 input.dist2s query neighbor
    let scaleSq := mul input.yScale[metric]! input.yScale[metric]!
    let variance ← match input.yvar with
      | none => pure 0
      | some values => do
          let value ← at2 values (query * input.neighbors + neighbor) metric
          pure (divide value scaleSq)
    let epistemic := mul params.epistemic distance
    let total := add (add (add epsVar epistemic) params.aleatoric) variance
    return divide Bounds.one total

def normWeights (weights : List Word) : List Word :=
  let inverse := divide Bounds.one (sumWords weights)
  weights.map (fun value => mul value inverse)

structure Cell where
  normalized : List Word
  l2 : Word
  mu : Word
  se : Word
  epi : Word
  ale : Word
  deriving Repr, DecidableEq

/-- One metric cell, parameterized by the target `sqrt` implementation. -/
def cell (sqrtOp : SqrtOp) (input : Input) (params : Params) (noise : Bool)
    (query metric : Nat) : Except PostError Cell := do
  let raw ← rawWeights input params query metric
  let normalized := normWeights raw
  let values ← mapN input.neighbors fun neighbor => do
    let y ← at2 input.y (query * input.neighbors + neighbor) metric
    let variance ← match input.yvar with
      | none => pure 0
      | some yvar => do
          let rawVar ← at2 yvar (query * input.neighbors + neighbor) metric
          let scaleSq := mul input.yScale[metric]! input.yScale[metric]!
          pure (divide rawVar scaleSq)
    return (y, variance)
  let pairs := normalized.zip values
  let l2Sq := pairs.foldl (fun total item =>
    add total (mul item.1 item.1)) 0
  let mean := pairs.foldl (fun total item =>
    add total (mul item.1 item.2.1)) 0
  let aleatoric := if noise then
    pairs.foldl (fun total item =>
      let variance := add params.aleatoric item.2.2
      add total (mul item.1 variance)) 0
    else 0
  let epistemic := divide Bounds.one (sumWords raw)
  let parts := seParts sqrtOp epistemic aleatoric input.yScale[metric]!
  return ⟨normalized, sqrtOp l2Sq, mean, parts.se, parts.epi, parts.ale⟩

def compute (sqrtOp : SqrtOp) (input : Input) (params : Params)
    (noise : Bool) : Except PostError Stats := do
  if !validInput input then throw .shape
  let cells ← mapN input.queries fun query =>
    mapN input.metrics fun metric => cell sqrtOp input params noise query metric
  let weights := cells.map fun row =>
    (List.range input.neighbors).map fun neighbor =>
      row.map fun value => value.normalized[neighbor]!
  let field (get : Cell → Word) : Batch := cells.map fun row => row.map get
  return ⟨weights, field (·.l2), field (·.mu), field (·.se),
    field (·.epi), field (·.ale)⟩

def oneInput : Input :=
  ⟨1, 1, 1, [[0]], [[Bounds.one]], none, [Bounds.one]⟩

theorem floorBranch :
    seParts (fun _ => Bounds.one) 0 0 Bounds.one =
      ⟨Bounds.one, Bounds.one, 0⟩ := by
  decide +kernel

/-- Even one normalized weight is not assumed to be exact real one: source
division followed by multiplication rounds to the adjacent lower binary64. -/
theorem oneMean :
    (compute (fun _ => Bounds.one) oneInput ⟨1, 0, 0⟩ false).map (·.mu) =
      .ok [[0x3fefffffffffffff]] := by
  decide +kernel

/-- Semantic kernel for extracted `rust:ennbo 0.4.1 posterior/se_from_variance_components().` -/
def sePure (epi ale scale : BauxResearch.Extraction.Native.F64) :
    Aeneas.Std.Result
      (BauxResearch.Extraction.Native.F64 ×
       BauxResearch.Extraction.Native.F64 ×
       BauxResearch.Extraction.Native.F64) := do
  let sum ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.add epi ale)
  let b ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.lt sum
    BauxResearch.Generated.Pure.error.EPS_VAR)
  if b then
    let fl ← BauxResearch.Extraction.Native.F64.sqrt
      BauxResearch.Generated.Pure.error.EPS_VAR
    let se ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl scale)
    Aeneas.Std.Result.ok (se, se, BauxResearch.Extraction.Native.F64.literal false 0 1)
  else
    let fl ← BauxResearch.Extraction.Native.F64.sqrt sum
    let se ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl scale)
    let fl1 ← BauxResearch.Extraction.Native.F64.sqrt epi
    let seEpi ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl1 scale)
    let fl2 ← BauxResearch.Extraction.Native.F64.sqrt ale
    let seAle ← Aeneas.Std.lift (BauxResearch.Extraction.Native.F64.mul fl2 scale)
    Aeneas.Std.Result.ok (se, seEpi, seAle)

/-- Exact correspondence for extracted `rust:ennbo 0.4.1 posterior/se_from_variance_components().` -/
theorem seMatch (epi ale scale : BauxResearch.Extraction.Native.F64) :
    BauxResearch.Generated.Pure.posterior.se_from_variance_components epi ale scale =
      sePure epi ale scale := by
  rfl

#print axioms floorBranch
#print axioms oneMean
#print axioms seMatch

end BauxResearch.Semantics.Posterior
