import BauxResearch.Semantics.Posterior
import BauxResearch.Semantics.Fit

/-!
# ENN 0.4.1 posterior neighbor refinement

This module executes the source-visible part of `posterior::index_search` and
`get_neighbor_data`: f64 distance recomputation, total-order reranking, output
width normalization, valid-prefix selection, and target-row gathering. The
Faiss or B+ANN candidate-label call is an explicit operation with its request
and result represented in the type.
-/

namespace BauxResearch.Semantics.Neighbor

open Sampling Bounds Fitter

abbrev Label := Int
abbrev Labels := List (List Label)

inductive Driver where
  | flat
  | bpann
  deriving Repr, DecidableEq

inductive NearError where
  | shape
  | index
  | backend
  | exclude
  deriving Repr, DecidableEq

structure Model where
  trainX : Fitter.Matrix
  trainY : Fitter.Matrix
  scaleX : Bool
  xScale : List Word
  driver : Driver
  deriving Repr, DecidableEq

structure Entry where
  dist : Word
  label : Label
  deriving Repr, DecidableEq

structure Search where
  dist2s : Batch
  labels : Labels
  deriving Repr, DecidableEq

structure Neighbors where
  dist2s : Batch
  indices : List (List Nat)
  y : Batch
  k : Nat
  deriving Repr, DecidableEq

/-- Candidate-label production by the selected compiled index backend. -/
abbrev BackendOp := Fitter.Matrix → Nat → Bool → Except NearError Labels

def validMatrix (matrix : Fitter.Matrix) : Bool :=
  matrix.data.length == matrix.rows &&
    matrix.data.all (fun row => row.length == matrix.cols)

def rowDist (scale : Bool) (x y scales : List Word) : Word :=
  let count := min x.length y.length
  let total := (List.range count).foldl (fun acc column =>
    let delta := if scale then
      sub (divide x[column]! scales[column]!)
        (divide y[column]! scales[column]!)
    else sub x[column]! y[column]!
    add acc (mul delta delta)) 0
  Fitter.maxZero total

def distance (model : Model) (query : List Word) (label : Label) : Word :=
  if label < 0 then posInf
  else
    match model.trainX.data[label.toNat]? with
    | none => posInf
    | some row => rowDist model.scaleX query row model.xScale

def before (left right : Entry) : Bool :=
  let lk := Fit.totalKey left.dist
  let rk := Fit.totalKey right.dist
  decide (lk < rk || (lk = rk && left.label ≤ right.label))

def insert (entry : Entry) : List Entry → List Entry
  | [] => [entry]
  | head :: tail =>
      if before entry head then entry :: head :: tail
      else head :: insert entry tail

def sortRow (entries : List Entry) : List Entry :=
  entries.foldr insert []

def refineRow (model : Model) (query : List Word) (labels : List Label) :
    List Entry :=
  sortRow (labels.map fun label => ⟨distance model query label, label⟩)

def padRow (width : Nat) (entries : List Entry) : List Entry :=
  entries.take width ++ List.replicate (width - entries.length) ⟨posInf, -1⟩

def refineRows (model : Model) (queries : Fitter.Matrix)
    (width : Nat) (labels : Labels) : Except NearError Search := do
  if labels.length != queries.rows then throw .shape
  let rows := (List.range queries.rows).map fun row =>
    padRow width (refineRow model queries.data[row]! labels[row]!)
  return ⟨rows.map (fun row => row.map (fun entry => entry.dist)),
    rows.map (fun row => row.map (fun entry => entry.label))⟩

def fetchWidth (driver : Driver) (searchK observations : Nat) : Nat :=
  match driver with
  | .flat => min observations (searchK + 32)
  | .bpann => min observations searchK

/-- `posterior::index_search` after index synchronization. The backend
operation owns scaling, f64-to-f32 conversion, native search, and exclusion;
the returned labels are always reranked using source-order f64 distances. -/
def indexSearch (backend : BackendOp) (model : Model)
    (queries : Fitter.Matrix) (searchK : Nat) (exclude : Bool) :
    Except NearError Search := do
  if !validMatrix queries || queries.cols != model.trainX.cols then throw .shape
  if searchK = 0 then return ⟨List.replicate queries.rows [],
    List.replicate queries.rows []⟩
  if model.trainX.rows = 0 then
    return ⟨List.replicate queries.rows (List.replicate searchK posInf),
      List.replicate queries.rows (List.replicate searchK (-1))⟩
  let width := fetchWidth model.driver searchK model.trainX.rows
  let labels ← backend queries width exclude
  refineRows model queries searchK labels

def validCount (labels : List Label) : Nat :=
  (labels.takeWhile fun label => decide (0 ≤ label)).length

def minValid : Labels → Nat
  | [] => 0
  | row :: rows => rows.foldl (fun count next => min count (validCount next))
      (validCount row)

def takeSearch (width : Nat) (search : Search) : Search :=
  ⟨search.dist2s.map (List.take width), search.labels.map (List.take width)⟩

def gatherRow (rows : Batch) (indices : List Nat) : Except NearError Batch :=
  indices.foldr (fun index rest => do
    let tail ← rest
    match rows[index]? with
    | some row => return row :: tail
    | none => throw .index) (.ok [])

def gatherY (rows : Batch) (indices : List (List Nat)) : Except NearError Batch :=
  indices.foldl (fun result row => do
    let prior ← result
    let next ← gatherRow rows row
    return prior ++ next) (.ok [])

/-- `get_neighbor_data`, from search-width selection through y-row gather. -/
def getData (backend : BackendOp) (model : Model) (queries : Fitter.Matrix)
    (neighbors : Nat) (exclude : Bool) : Except NearError (Option Neighbors) := do
  if exclude && model.trainX.rows ≤ 1 then throw .exclude
  let searchK := min model.trainX.rows (neighbors + if exclude then 1 else 0)
  if searchK = 0 then return none
  let full ← indexSearch backend model queries searchK exclude
  let available := full.labels.head?.map List.length |>.getD 0
  let width := min neighbors (min available (minValid full.labels))
  if width = 0 then return none
  let selected := takeSearch width full
  let indices := selected.labels.map fun row => row.map Int.toNat
  let y ← gatherY model.trainY.data indices
  return some ⟨selected.dist2s, indices, y, width⟩

def unitModel : Model :=
  ⟨⟨1, 1, [[Bounds.one]]⟩, ⟨1, 1, [[0]]⟩, false,
    [Bounds.one], .flat⟩

theorem negDistance (query : List Word) :
    distance unitModel query (-1) = posInf := by
  rfl

theorem tieIndex :
    (sortRow [⟨Bounds.one, 2⟩, ⟨Bounds.one, 1⟩]).map Entry.label =
      [1, 2] := by
  decide +kernel

theorem emptySearch (backend : BackendOp) (queries : Fitter.Matrix)
    (hvalid : validMatrix queries = true) (hcols : queries.cols = 1) :
    indexSearch backend unitModel queries 0 false =
      .ok ⟨List.replicate queries.rows [], List.replicate queries.rows []⟩ := by
  simp [indexSearch, hvalid, hcols, unitModel]
  rfl

#print axioms negDistance
#print axioms tieIndex
#print axioms emptySearch

end BauxResearch.Semantics.Neighbor
