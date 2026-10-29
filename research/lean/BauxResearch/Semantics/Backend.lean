import BauxResearch.Semantics.Neighbor
import BauxResearch.Semantics.Append
import BauxResearch.Extraction.Native

/-!
# ENN 0.4.1 compiled neighbor backend relation

This module formalizes the source-faithful checked relation and executable candidate
transitions for reachable Flat Faiss and BPANN neighbor backend operations from pinned
ENN 0.4.1 (`f8f6a5dada9e85d1a196b7281be4b1b070e0a3ef`) under
`rust/crates/{ennbo,bpann}`.

Specifically, this module models:
- Flat Faiss (`ennbo/src/index.rs`, `ennbo/src/knn/mod.rs`, `ennbo/src/knn/faiss_backend.rs`):
  coordinate scaling (`scaleRow`), `f64 -> f32` narrowing (`toSingle`), `f32` squared L2
  distance accumulation (`distSingle`), widening to `f64` (`fromSingle`), `k_eff` top-k
  selection (`flatTop`), farthest-label repetition padding (`padFlat`) prior to LOO
  nearest-self exclusion (`excludeFlat`), and trailing negative column trimming (`trimCols`).
- BPANN (`bpann/src/backend.rs`, `bpann/src/small_n_search.rs`, `bpann/src/large_n_search.rs`,
  `bpann/src/merge.rs`):
  `k_eff`, `pool_k`, and `index_k` sizing (`effWidth`, `poolWidth`, `indexWidth`),
  the `total <= 8192` (`smallLimit`) exhaustive `f64` fast path bypassing B+ tree and
  pending splits, the `total > 8192` dual-leg B+ tree (`[0, indexed)`) + pending scan
  (`[indexed, total)`) merge path, exact row equality fallback (`findSelf`, `findScaled`),
  arbitrary-rank self-exclusion (`excludeBpann`), and trailing negative column trimming
  (`trimCols`).
- Checked correspondence theorems connecting `BackendRel`, `backendOp`, `exactOp`,
  `flatSearch`, `bpannSearch`, `exactSearch`, and `Neighbor.refineRows`, plus compiled
  native bindings to `BauxResearch.Extraction.Native`.
-/

namespace BauxResearch.Semantics.Backend

open Sampling Bounds Fitter Neighbor Append

/-- Model and storage state of a compiled candidate backend. -/
structure State where
  driver : Neighbor.Driver
  numDim : Nat
  numMetrics : Nat
  scaleX : Bool
  xScale : List Word
  trainX : Fitter.Matrix
  trainY : Fitter.Matrix
  yvar : Option Fitter.Matrix
  indexed : Nat
  pending : Nat
  dirty : Bool
  persisted : Nat
  treeReady : Bool
  rowsSinceRelease : Nat
  flushSoft : Nat
  flushHard : Nat
  deriving Repr, DecidableEq

/-- Convert backend state into the pure Neighbor model view. -/
def toModel (state : State) : Neighbor.Model :=
  ⟨state.trainX, state.trainY, state.scaleX, state.xScale, state.driver⟩

/-- Coordinate scaling cell conversion. -/
def scaleCell (scale : Bool) (x s : Word) : Word :=
  if scale then divide x s else x

/-- Coordinate scaling row conversion. -/
def scaleRow (scale : Bool) (row scales : List Word) : List Word :=
  (List.range (min row.length scales.length)).map fun i ↦
    scaleCell scale row[i]! scales[i]!

/-- Coordinate scaling for full matrix. -/
def scaleMatrix (scale : Bool) (scales : List Word) (m : Fitter.Matrix) :
    Fitter.Matrix :=
  ⟨m.rows, m.cols, m.data.map fun row ↦ scaleRow scale row scales⟩

/-- Unscaled coordinate cell is identity. -/
theorem scaleId (x s : Word) : scaleCell false x s = x := rfl

/-- Scaled coordinate cell divides coordinate by metric scale. -/
theorem scaleDiv (x s : Word) : scaleCell true x s = divide x s := rfl

/-- Label validity check against observation count. -/
def validLabel (n : Nat) (lbl : Label) : Bool :=
  decide (0 ≤ lbl && lbl < (n : Int))

/-- Label bounds check: sentinel (-1) or valid observation index. -/
def labelBound (n : Nat) (lbl : Label) : Bool :=
  decide (lbl = -1 || (0 ≤ lbl && lbl < (n : Int)))

/-- All labels in a candidate row are within bounds. -/
def rowBound (n : Nat) (row : List Label) : Bool :=
  row.all (labelBound n)

/-- Deduplication check on candidate rows. -/
def hasDupe : List Label → Bool
  | [] => false
  | x :: xs => if xs.contains x then true else hasDupe xs

/-- Non-negative candidate labels in a row contain no duplicates. -/
def noDupes (row : List Label) : Bool :=
  let valids := row.filter fun l ↦ decide (0 ≤ l)
  !hasDupe valids

/-- Fetch width for Flat Faiss is bounded by observation count. -/
theorem widthFlat (searchK n : Nat) :
    fetchWidth .flat searchK n ≤ n := by
  simp [fetchWidth]

/-- Fetch width for BPANN is bounded by observation count. -/
theorem widthBpann (searchK n : Nat) :
    fetchWidth .bpann searchK n ≤ n := by
  simp [fetchWidth]

/-- Fetch width is bounded by observation count for any driver. -/
theorem widthBound (driver : Neighbor.Driver) (searchK n : Nat) :
    fetchWidth driver searchK n ≤ n := by
  cases driver <;> simp [fetchWidth]

/-- When distances tie, entries are ordered by label index. -/
theorem tieBreak (d : Word) (l1 l2 : Label) :
    before ⟨d, l1⟩ ⟨d, l2⟩ = decide (l1 ≤ l2) := by
  simp [before]

/-- Pad candidate labels with sentinel (-1) to the requested width. -/
def padLabels (width : Nat) (labels : List Label) : List Label :=
  labels.take width ++ List.replicate (width - labels.length) (-1)

def selfEps : Word := 0x3cd203af9ee75616

def nearSelf (dist : Word) : Bool :=
  Float.finiteDouble dist && Float.leDouble dist selfEps

/-- Pad candidate distances with positive infinity to the requested width. -/
def padDist (width : Nat) (dists : List Word) : List Word :=
  dists.take width ++ List.replicate (width - dists.length) posInf

/-- Pad Flat Faiss candidate labels by repeating the farthest valid label. -/
def padFlat (width : Nat) (labels : List Label) : List Label :=
  labels.take width ++ List.replicate (width - labels.length) (labels.getLast?.getD 0)

theorem distPad (width : Nat) (dists : List Word) :
    (padDist width dists).length = width := by
  simp [padDist]
  omega

theorem flatPad (width : Nat) (labels : List Label) :
    (padFlat width labels).length = width := by
  simp [padFlat]
  omega

theorem epsSelf : nearSelf selfEps = true := by decide

theorem infDistinct : nearSelf posInf = false := by decide

theorem flatRepeat : padFlat 3 [2, 4] = [2, 4, 4] := by decide

/-- Padded candidate label list has exact requested length. -/
theorem labelsLen (width : Nat) (labels : List Label) :
    (padLabels width labels).length = width := by
  simp [padLabels]
  omega

/-- Check whether column `col` is negative across all query candidate rows. -/
def allNeg (rows : Labels) (col : Nat) : Bool :=
  rows.all fun row ↦ decide (row.getD col (-1) < 0)

/-- Compute trimmed column width by dropping trailing all-negative columns
(`trim_trailing_invalid_neighbor_cols` in `ennbo/src/index.rs` and `bpann/src/backend.rs`). -/
def trimWidth : Nat → Labels → Nat
  | 0, _ => 0
  | col + 1, rows =>
      if allNeg rows col then trimWidth col rows else col + 1

/-- Trim trailing columns that are negative in every row. -/
def trimCols (width : Nat) (rows : Labels) : Labels :=
  let keep := if rows.isEmpty then 0 else trimWidth width rows
  rows.map fun row ↦ row.take keep

theorem trimLe : ∀ (width : Nat) (rows : Labels), trimWidth width rows ≤ width
  | 0, _ => Nat.le_refl 0
  | col + 1, rows => by
      simp only [trimWidth]
      split
      · exact Nat.le_trans (trimLe col rows) (Nat.le_succ col)
      · exact Nat.le_refl (col + 1)

theorem trimLen (width : Nat) (rows : Labels) :
    (trimCols width rows).length = rows.length := by
  simp [trimCols]

theorem trimBound (width : Nat) (rows : Labels) :
    ∀ row ∈ trimCols width rows, row.length ≤ width := by
  intro row hrow
  rcases List.mem_map.mp hrow with ⟨r, _, hr⟩
  rw [← hr]
  split
  · simp
  · exact Nat.le_trans (List.length_take_le _ _) (trimLe width rows)

/-- Trace of effects produced during backend initialization. -/
def constructTrace : List Append.Effect :=
  [.construct, .auto]

/-- Trace of effects produced during stateful row append. -/
def appendTrace (hasScale : Bool) (persist : Bool) : List Append.Effect :=
  let base := [Append.Effect.wait, Append.Effect.append]
  let withScale := if hasScale then base ++ [Append.Effect.scale] else base
  let withPersist := if persist then withScale ++ [Append.Effect.persist] else withScale
  withPersist ++ [Append.Effect.auto]

/-- Trace of effects produced during soft synchronization. -/
def syncTrace : List Append.Effect :=
  [.sync]

/-- Trace of effects produced during RSS page release. -/
def releaseTrace : List Append.Effect :=
  [.release]

/-- Checked relation specifying that all effects in a trace pass the gate. -/
def gatePass (gate : Append.Gate) (effects : List Append.Effect) : Prop :=
  ∀ e ∈ effects, gate e = .ok ()

/-- Errors produced by compiled storage and persistence operations. -/
inductive StorageError where
  | shape
  | param
  | io
  | flush
  deriving Repr, DecidableEq

/-- Relational specification of the backend storage append transition. -/
structure StorageRel (before : State) (x y : Fitter.Matrix)
    (after : State) : Prop where
  driverEq : after.driver = before.driver
  dimEq : after.numDim = before.numDim
  metricEq : after.numMetrics = before.numMetrics
  scaleEq : after.scaleX = before.scaleX ∧ after.xScale = before.xScale
  rowsX : after.trainX.rows = before.trainX.rows + x.rows
  colsX : after.trainX.cols = before.trainX.cols
  dataX : after.trainX.data = before.trainX.data ++ x.data
  rowsY : after.trainY.rows = before.trainY.rows + y.rows
  colsY : after.trainY.cols = before.trainY.cols
  dataY : after.trainY.data = before.trainY.data ++ y.data
  pendingEq : after.pending = before.pending + x.rows

/-- Stateful row append on compiled storage. -/
def appendStore (state : State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) : Except StorageError State := do
  if x.cols != state.numDim || y.cols != state.numMetrics || x.rows != y.rows then
    throw .shape
  return { state with
    trainX := appendMatrix state.trainX x
    trainY := appendMatrix state.trainY y
    yvar := appendVar state.yvar variance
    pending := state.pending + x.rows
    dirty := true
    rowsSinceRelease := state.rowsSinceRelease + x.rows }

/-- Soft index synchronization inserting unindexed rows into memory index. -/
def softSync (state : State) : State :=
  { state with
    indexed := state.trainX.rows
    pending := 0 }

/-- Hard persistence flushing index state to disk. -/
def persistIndex (state : State) : State :=
  { state with
    indexed := state.trainX.rows
    pending := 0
    dirty := false
    persisted := state.trainX.rows }

/-- Page release clearing resident pages from process RSS. -/
def releasePages (state : State) : State :=
  { state with rowsSinceRelease := 0 }

/-- Metric rescaling resetting index state for rebuild. -/
def rescaleMetric (state : State) (newScale : List Word) : State :=
  { state with
    xScale := newScale
    indexed := 0
    pending := state.trainX.rows
    dirty := true
    treeReady := false }

/-- Append increases trainX rows by the appended matrix row count. -/
theorem appendRows (state : State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (next : State)
    (h : appendStore state x y variance = .ok next) :
    next.trainX.rows = state.trainX.rows + x.rows := by
  unfold appendStore at h
  split at h <;> try contradiction
  injection h with h'
  rw [← h']
  rfl

/-- Append increases pending unindexed count by the appended row count. -/
theorem appendPending (state : State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (next : State)
    (h : appendStore state x y variance = .ok next) :
    next.pending = state.pending + x.rows := by
  unfold appendStore at h
  split at h <;> try contradiction
  injection h with h'
  rw [← h']

/-- Append sets the dirty flag. -/
theorem appendDirty (state : State) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix) (next : State)
    (h : appendStore state x y variance = .ok next) :
    next.dirty = true := by
  unfold appendStore at h
  split at h <;> try contradiction
  injection h with h'
  rw [← h']

/-- Soft sync clears pending row count. -/
theorem syncClear (state : State) :
    (softSync state).pending = 0 := rfl

/-- Soft sync updates indexed row count to full training rows. -/
theorem syncIndexed (state : State) :
    (softSync state).indexed = state.trainX.rows := rfl

/-- Soft sync preserves dirty status until hard persist. -/
theorem syncDirty (state : State) :
    (softSync state).dirty = state.dirty := rfl

/-- Persist clears dirty flag. -/
theorem persistClean (state : State) :
    (persistIndex state).dirty = false := rfl

/-- Persist updates persisted row count. -/
theorem persistCount (state : State) :
    (persistIndex state).persisted = state.trainX.rows := rfl

/-- Release pages resets rows since release. -/
theorem releaseZero (state : State) :
    (releasePages state).rowsSinceRelease = 0 := rfl

/-- Rescale resets indexed count to zero. -/
theorem rescaleReset (state : State) (newScale : List Word) :
    (rescaleMetric state newScale).indexed = 0 := rfl

/-- Rescale marks all training rows as pending unindexed. -/
theorem rescalePending (state : State) (newScale : List Word) :
    (rescaleMetric state newScale).pending = state.trainX.rows := rfl

/-- Rescale sets dirty flag. -/
theorem rescaleDirty (state : State) (newScale : List Word) :
    (rescaleMetric state newScale).dirty = true := rfl

/-- Padded row length matches requested width. -/
theorem padLen (width : Nat) (entries : List Entry) :
    (padRow width entries).length = width := by
  simp [padRow]
  omega

/-- Insertion sort element step increases length by one. -/
theorem insertLen (entry : Entry) (entries : List Entry) :
    (Neighbor.insert entry entries).length = entries.length + 1 := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
      simp only [Neighbor.insert]
      split
      · rfl
      · simp [ih]

/-- Distance sorting preserves entry list length. -/
theorem sortLen (entries : List Entry) :
    (sortRow entries).length = entries.length := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
      simp only [sortRow, List.foldr_cons]
      rw [insertLen]
      exact congrArg (· + 1) ih

/-- Row refinement preserves candidate label count. -/
theorem refineLen (model : Neighbor.Model) (query : List Word)
    (labels : List Label) :
    (refineRow model query labels).length = labels.length := by
  simp [refineRow, sortLen]

/-- Refining valid candidate rows produces a search with exact dimensions. -/
theorem rowsOk (model : Neighbor.Model) (queries : Fitter.Matrix)
    (width : Nat) (labels : Labels) (h : labels.length = queries.rows) :
    ∃ search, refineRows model queries width labels = .ok search ∧
      search.labels.length = queries.rows ∧
      search.dist2s.length = queries.rows ∧
      (∀ row ∈ search.labels, row.length = width) ∧
      (∀ row ∈ search.dist2s, row.length = width) := by
  simp [refineRows, h]
  refine ⟨⟨_, _⟩, rfl, ?_, ?_, ?_, ?_⟩
  · simp
  · simp
  · intro row hrow
    rcases List.mem_map.mp hrow with ⟨r, _, hr⟩
    rw [← hr]
    simp [padLen]
  · intro row hrow
    rcases List.mem_map.mp hrow with ⟨r, _, hr⟩
    rw [← hr]
    simp [padLen]

/-!
### Flat Faiss C-ABI Boundary & Rust Orchestration

In `ennbo/src/index.rs`, `ennbo/src/knn/mod.rs`, and `ennbo/src/knn/faiss_backend.rs`:
- Queries and training vectors are scaled by `xScale` if `scaleX = true`, then narrowed from
  IEEE-754 binary64 to binary32 (`as f32`, modeled by `toSingle` / `rowSingle`).
- Native `IndexFlatL2` search (`faiss_Index_search`) computes `f32` squared L2 distances
  (`distSingle`), widens them to `f64` (`fromSingle`), and selects the top `k_eff = min(width, n)`
  entries (`flatTop`).
- In `KNNIndex::search_Internal`, `pad_neighbor_cols_to_search_k` pads distances with `posInf`
  (`padDist`) and pads labels by repeating the farthest label (`padFlat`) *before*
  `apply_exclude_nearest_identity`.
- When `exclude = true`, `apply_exclude_nearest_identity` drops column 0 and shifts left
  (setting column `width - 1` to `(posInf, -1)`) for rows where `nearSelf` holds (`excludeFlat`),
  and then `trim_trailing_invalid_neighbor_cols` drops trailing all-negative columns (`trimCols`).
-/

/-- Narrow an IEEE-754 binary64 word to binary32 (`val as f32`). -/
def toSingle (w : Word) : Float.WordSingle :=
  Float.castNarrow (Float.wordDouble w)

/-- Widen an IEEE-754 binary32 word to binary64 (`val as f64`). -/
def fromSingle (w : Float.WordSingle) : Word :=
  (Float.castWide w).bits

/-- Scale a coordinate row in `f64` (when `scale = true`) and narrow each cell to `f32`
(`scaled_row_f32` / `scale_matrix_f32` in `ennbo/src/index.rs`). -/
def rowSingle (scale : Bool) (row scales : List Word) : List Float.WordSingle :=
  (List.range row.length).map fun c ↦
    let raw := row.getD c 0
    let sc := if scale then divide raw (scales.getD c Bounds.one) else raw
    toSingle sc

/-- Squared L2 distance in IEEE-754 binary32 (`IndexFlatL2` inner loop). -/
def distSingle (q x : List Float.WordSingle) (dim : Nat) : Float.WordSingle :=
  let zero := Float.wordSingle Float.singleZero
  (List.range dim).foldl (fun acc c ↦
    let qc := q.getD c zero
    let xc := x.getD c zero
    let diff := Float.subSingle qc xc
    let sq := Float.mulSingle diff diff
    Float.addSingle acc sq) zero

/-- Flat Faiss squared L2 distance widened back to `f64`. -/
def flatDist (scale : Bool) (scales : List Word) (dim : Nat)
    (query row : List Word) : Word :=
  let q32 := rowSingle scale query scales
  let x32 := rowSingle scale row scales
  fromSingle (distSingle q32 x32 dim)

/-- Exhaustive `f32` distance scan over all training rows in `train`. -/
def flatScan (scale : Bool) (scales : List Word) (dim : Nat)
    (train : Fitter.Matrix) (query : List Word) : List Entry :=
  (List.range train.rows).map fun idx ↦
    ⟨flatDist scale scales dim query (train.data.getD idx []), (idx : Label)⟩

/-- Top `k_eff = min(width, train.rows)` entries produced by `IndexFlatL2::search`. -/
def flatTop (scale : Bool) (scales : List Word) (dim : Nat)
    (train : Fitter.Matrix) (query : List Word) (width : Nat) : List Entry :=
  let kEff := min width train.rows
  (sortRow (flatScan scale scales dim train query)).take kEff

/-- Batch `(dists, labels)` output of `FaissFlatBackend::search` across all query rows. -/
def faissBatch (scale : Bool) (scales : List Word) (dim : Nat)
    (train queries : Fitter.Matrix) (width : Nat) : Batch × Labels :=
  let rows := (List.range queries.rows).map fun r ↦
    flatTop scale scales dim train (queries.data.getD r []) width
  (rows.map (fun row ↦ row.map Entry.dist), rows.map (fun row ↦ row.map Entry.label))

/-- Source-faithful C-ABI relation for native Faiss Flat exact L2 search (`libfaiss_c`).
Binds `dists` and `labels` to the `f32` scaled squared-L2 top-`min width train.rows`
candidates computed over `train` and `queries`. -/
def FaissRel (scale : Bool) (scales : List Word) (dim : Nat)
    (train queries : Fitter.Matrix) (width : Nat)
    (dists : Batch) (labels : Labels) : Prop :=
  labels.length = queries.rows ∧
  dists.length = queries.rows ∧
  ∀ r : Nat, r < queries.rows →
    let top := flatTop scale scales dim train (queries.data.getD r []) width
    dists.getD r [] = top.map Entry.dist ∧
    labels.getD r [] = top.map Entry.label

/-- `faissBatch` satisfies `FaissRel`. -/
theorem faissExact (scale : Bool) (scales : List Word) (dim : Nat)
    (train queries : Fitter.Matrix) (width : Nat) :
    let (dists, labels) := faissBatch scale scales dim train queries width
    FaissRel scale scales dim train queries width dists labels := by
  dsimp only [faissBatch]
  refine ⟨by simp, by simp, ?_⟩
  intro r hr
  simp [List.getD, hr]

/-- Correspondence between `toSingle`/`fromSingle` and `BauxResearch.Extraction.Native`. -/
theorem nativeCast (w64 : Word) (w32 : Float.WordSingle) :
    toSingle w64 =
      BauxResearch.Extraction.Native.F32.fromDouble (Float.wordDouble w64) ∧
    fromSingle w32 =
      (BauxResearch.Extraction.Native.F64.fromSingle w32).bits :=
  ⟨rfl, rfl⟩

/-- Exclude nearest observation under LOO identity when distance is within epsilon.
Drops column 0, shifts columns left, and pads the rightmost slot with `(posInf, -1)`. -/
def excludeFlat (dists : List Word) (labels : List Label) : List Word × List Label :=
  match dists, labels with
  | d :: ds, _l :: ls =>
      if nearSelf d then (ds ++ [posInf], ls ++ [-1])
      else (dists, labels)
  | _, _ => (dists, labels)

theorem epsExclude :
    excludeFlat [selfEps, Bounds.one] [0, 1] =
      ([Bounds.one, posInf], [1, -1]) := by decide

/-- Per-row Flat Faiss post-processing: pads `rowD` and `rowL` to `width` first
(`pad_neighbor_cols_to_search_k`), then applies LOO self-exclusion (`excludeFlat`). -/
def flatRow (width : Nat) (exclude : Bool) (rowD : List Word)
    (rowL : List Label) : List Label :=
  let paddedD := padDist width rowD
  let paddedL := padFlat width rowL
  let (_dists, labels) := if exclude then excludeFlat paddedD paddedL else (paddedD, paddedL)
  labels

/-- Untrimmed Flat Faiss candidate label rows across all queries. -/
def flatRaw (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (rawLabels : Labels) (rawDists : Batch) : Labels :=
  (List.range queries.rows).map fun r ↦
    flatRow width exclude (rawDists.getD r []) (rawLabels.getD r [])

/-- Rust orchestration around native Faiss search (`KNNIndex::search_Internal`):
pads to `width`, applies LOO nearest-neighbor exclusion when `exclude = true`,
and trims trailing negative columns (`trimCols`). -/
def flatSearch (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (rawLabels : Labels) (rawDists : Batch) : Labels :=
  let rows := flatRaw queries width exclude rawLabels rawDists
  if exclude then trimCols width rows else rows

/-!
### BPANN Small-N Fast Path, Indexed vs Pending Boundary & Rust Orchestration

In `bpann/src/backend.rs`, `bpann/src/small_n_search.rs`, `bpann/src/large_n_search.rs`,
and `bpann/src/merge.rs`:
- `k_eff = min(width, total)` (`effWidth`).
- `pool_k = min(k_eff + 1, total)` if `exclude` else `k_eff` (`poolWidth`).
- `index_k = max(pool_k, k_eff * 2)` if `exclude` else `k_eff` (`indexWidth`).
- When `total <= 8192` (`smallLimit`), `BpannBackend::search` executes `search_small_n`
  (`ScoreQueriesFlat`), scoring all `total` rows in `f64` and bypassing the B+ tree and
  pending split.
- When `total > 8192`, `search_large_n` queries Leg A (`[0, indexed)` via B+ tree up to
  `max(index_k, 1)`) and Leg B (`[indexed, total)` via brute-force `f64` scan up to `pool_k`),
  deduplicates, sorts, resolves `self_id` (`nearSelf` first, then `findSelf` exact row match),
  excludes `self_id` at any rank (`excludeBpann`), truncates to `pool_k` and `k_eff`, pads with
  `-1` to `k_eff`, and trims trailing all-negative columns (`trimCols`).
-/

/-- Small-N fast-path threshold (`SMALL_N_FLAT_SEARCH_MAX = 8192` in `bpann/src/backend.rs`). -/
def smallLimit : Nat := 8192

/-- Effective neighbor width `k_eff = min(width, total)`. -/
def effWidth (width total : Nat) : Nat :=
  min width total

/-- Candidate pool width `pool_k` before self-exclusion. -/
def poolWidth (width total : Nat) (exclude : Bool) : Nat :=
  let kEff := effWidth width total
  if exclude then min (kEff + 1) total else kEff

/-- B+ tree fetch width `index_k` for Leg A in large-N search. -/
def indexWidth (width total : Nat) (exclude : Bool) : Nat :=
  let kEff := effWidth width total
  let poolK := poolWidth width total exclude
  if exclude then max poolK (kEff * 2) else kEff

/-- Exact IEEE-754 bit-level row equality (`query_row == train_row`). -/
def rowEq (q x : List Word) (dim : Nat) : Bool :=
  (List.range dim).all fun c ↦ decide (q.getD c 0 = x.getD c 0)

/-- Reverse scan for the latest training row matching `query` (`find_query_train_id`). -/
def findSelf (train : Fitter.Matrix) (query : List Word) : Option Label :=
  (List.range train.rows).reverse.findSome? fun idx ↦
    if rowEq query (train.data.getD idx []) train.cols then some (idx : Label)
    else none

/-- Reverse scan for the latest scaled training row matching scaled `query`
(`find_query_train_id_flat` in `bpann/src/small_n_search.rs`). -/
def findScaled (scale : Bool) (scales : List Word) (train : Fitter.Matrix)
    (query : List Word) : Option Label :=
  let qScaled := scaleRow scale query scales
  (List.range train.rows).reverse.findSome? fun idx ↦
    let xScaled := scaleRow scale (train.data.getD idx []) scales
    if rowEq qScaled xScaled train.cols then some (idx : Label)
    else none

/-- Remove the first entry satisfying `pred` (`Vec::remove(pos)` in `bpann/src/merge.rs`). -/
def dropPred (pred : Entry → Bool) : List Entry → List Entry
  | [] => []
  | e :: es => if pred e then es else e :: dropPred pred es

/-- BPANN LOO self-exclusion (`bpann_apply_exclude_nearest` in `bpann/src/merge.rs`):
removes `selfId` at any rank when `some id` is found, or removes the first `nearSelf`
candidate otherwise. -/
def excludeBpann (entries : List Entry) (selfId : Option Label) : List Entry :=
  match selfId with
  | some id => dropPred (fun e ↦ decide (e.label = id)) entries
  | none => dropPred (fun e ↦ nearSelf e.dist) entries

/-- Resolve `self_id` in large-N search (`bpann/src/large_n_search.rs`):
checks if any candidate in `combined` has `nearSelf`, falling back to `findSelf`. -/
def resolveSelf (scale : Bool) (scales : List Word) (train : Fitter.Matrix)
    (query : List Word) (combined : List Entry) : Option Label :=
  match combined.find? (fun e ↦ nearSelf e.dist) with
  | some e => some e.label
  | none =>
      if scale then findScaled true scales train query
      else findSelf train query

/-- Brute-force distance scan over a specific row slice `[start, stop)` of `trainX`. -/
def scanRange (scale : Bool) (scales : List Word) (trainX : Fitter.Matrix)
    (query : List Word) (start stop : Nat) : List Entry :=
  (List.range (stop - start)).map fun offset ↦
    let idx := start + offset
    ⟨rowDist scale query (trainX.data.getD idx []) scales, (idx : Label)⟩

def uniqueEntries (entries : List Entry) : List Entry :=
  entries.foldl (fun seen entry ↦
    if seen.any (fun prior ↦ prior.label == entry.label) then seen
    else seen ++ [entry]) []

/-- Score a list of B+ tree leaf row indices in `[0, indexed)` using `rowDist`. -/
def scoreNodes (scale : Bool) (scales : List Word) (train : Fitter.Matrix)
    (query : List Word) (nodes : List Nat) : List Entry :=
  nodes.map fun idx ↦
    ⟨rowDist scale query (train.data.getD idx []) scales, (idx : Label)⟩

/-- Sorted top-`max indexK 1` B+ tree candidates from visited leaf rows `nodes`. -/
def treeTop (scale : Bool) (scales : List Word) (train : Fitter.Matrix)
    (query : List Word) (width : Nat) (exclude : Bool)
    (nodes : List Nat) : List Entry :=
  let indexK := indexWidth width train.rows exclude
  (sortRow (scoreNodes scale scales train query nodes)).take (max indexK 1)

/-- Full-prefix B+ tree candidate batch when the tree visits all indexed rows `[0, indexed)`. -/
def treeBatch (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) : List (List Entry) :=
  (List.range queries.rows).map fun r ↦
    treeTop state.scaleX state.xScale state.trainX
      (queries.data.getD r []) width exclude (List.range state.indexed)

/-- Source-faithful relation for BPANN B+ tree index search over `[0, indexed)`:
`entries` is the sorted top-`max indexK 1` slice of exact `f64` distances over a subset
`nodes` of indexed training rows `< indexed`. -/
def TreeRel (scale : Bool) (scales : List Word) (indexed _dim : Nat)
    (train : Fitter.Matrix) (query : List Word) (width : Nat)
    (exclude : Bool) (entries : List Entry) : Prop :=
  ∃ nodes : List Nat,
    (∀ idx ∈ nodes, idx < indexed) ∧
    entries = treeTop scale scales train query width exclude nodes

theorem memTake {α : Type} (x : α) : ∀ (n : Nat) (xs : List α), x ∈ xs.take n → x ∈ xs
  | 0, _, h => by contradiction
  | _ + 1, [], h => by contradiction
  | n + 1, y :: ys, h => by
      simp only [List.take_succ_cons, List.mem_cons] at h ⊢
      rcases h with rfl | htail
      · exact Or.inl rfl
      · exact Or.inr (memTake x n ys htail)

theorem memInsert (entry x : Entry) (entries : List Entry) :
    x ∈ Neighbor.insert entry entries ↔ x = entry ∨ x ∈ entries := by
  induction entries with
  | nil => simp [Neighbor.insert]
  | cons head tail ih =>
      simp only [Neighbor.insert]
      split
      · simp
      · simp only [List.mem_cons, ih]
        constructor
        · rintro (rfl | rfl | h)
          · exact Or.inr (Or.inl rfl)
          · exact Or.inl rfl
          · exact Or.inr (Or.inr h)
        · rintro (rfl | rfl | h)
          · exact Or.inr (Or.inl rfl)
          · exact Or.inl rfl
          · exact Or.inr (Or.inr h)

theorem memSort (x : Entry) (entries : List Entry) :
    x ∈ sortRow entries ↔ x ∈ entries := by
  induction entries with
  | nil => rfl
  | cons head tail ih =>
      change x ∈ Neighbor.insert head (sortRow tail) ↔ x ∈ head :: tail
      rw [memInsert, List.mem_cons, ih]

/-- Every entry satisfying `TreeRel` has a valid indexed label `< indexed` and exact `rowDist`. -/
theorem treeExact (scale : Bool) (scales : List Word) (indexed dim : Nat)
    (train : Fitter.Matrix) (query : List Word) (width : Nat)
    (exclude : Bool) (entries : List Entry)
    (h : TreeRel scale scales indexed dim train query width exclude entries) :
    ∀ e ∈ entries,
      0 ≤ e.label ∧ e.label < (indexed : Int) ∧
      e.dist = rowDist scale query (train.data.getD e.label.toNat []) scales := by
  rcases h with ⟨nodes, hnodes, rfl⟩
  intro e he
  have hmem : e ∈ sortRow (scoreNodes scale scales train query nodes) :=
    memTake e _ _ he
  rw [memSort] at hmem
  rcases List.mem_map.mp hmem with ⟨idx, hidx, rfl⟩
  have hlt : idx < indexed := hnodes idx hidx
  refine ⟨Int.natCast_nonneg idx, Int.ofNat_lt.mpr hlt, ?_⟩
  simp

/-- Merge indexed tree candidates (Leg A) and pending scan candidates (Leg B) with explicit
`poolK` truncation and arbitrary-rank self-exclusion (`merge_and_prune` in `bpann/src/merge.rs`). -/
def mergeSelf (kEff poolK : Nat) (legA legB : List Entry)
    (exclude : Bool) (selfId : Option Label) : List Label :=
  let combined := uniqueEntries (legA ++ legB)
  let sorted := sortRow combined
  let pruned := if exclude then
    (excludeBpann sorted selfId).take poolK
  else
    sorted.take poolK
  padLabels kEff ((pruned.take kEff).map Entry.label)

/-- Merge indexed tree candidates (Leg A) and pending scan candidates (Leg B)
using distance-based self-exclusion when `selfId` is not separately supplied. -/
def mergeLegs (width : Nat) (legA legB : List Entry) (exclude : Bool) : List Label :=
  let poolK := if exclude then width + 1 else width
  mergeSelf width poolK legA legB exclude none

theorem duplicateLeg :
    mergeLegs 2 [⟨0, 0⟩] [⟨0, 0⟩] false = [0, -1] := by decide

/-- Per-query BPANN candidate row before trailing-column trim:
executes the `total <= 8192` exhaustive `f64` branch (`search_small_n`) or the `total > 8192`
dual-leg B+ tree + pending scan branch (`search_large_n`). -/
def bpannRow (state : State) (query : List Word) (width : Nat)
    (exclude : Bool) (treeRow : List Entry) : List Label :=
  let total := state.trainX.rows
  let kEff := effWidth width total
  let poolK := poolWidth width total exclude
  let indexK := indexWidth width total exclude
  if total ≤ smallLimit then
    let allEntries := scanRange state.scaleX state.xScale state.trainX query 0 total
    let topPool := (sortRow allEntries).take poolK
    let selfId := if exclude then
      findScaled state.scaleX state.xScale state.trainX query
    else none
    let pruned := if exclude then
      (excludeBpann topPool selfId).take poolK
    else topPool
    padLabels kEff ((pruned.take kEff).map Entry.label)
  else
    let legA := if state.indexed > 0 && state.treeReady then
      treeRow.take (max indexK 1)
    else []
    let legB := if state.pending > 0 then
      (sortRow (scanRange state.scaleX state.xScale state.trainX query
        state.indexed total)).take poolK
    else []
    let selfId := if exclude then
      resolveSelf state.scaleX state.xScale state.trainX query (legA ++ legB)
    else none
    mergeSelf kEff poolK legA legB exclude selfId

/-- Untrimmed BPANN candidate rows across all queries. -/
def bpannRaw (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (treeCandidates : List (List Entry)) : Labels :=
  (List.range queries.rows).map fun r ↦
    bpannRow state (queries.data.getD r []) width exclude (treeCandidates.getD r [])

/-- BPANN candidate production (`BpannBackend::search` in `bpann/src/backend.rs`):
computes per-query rows via `bpannRow` and trims trailing all-negative columns when
`exclude = true`. -/
def bpannSearch (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (treeCandidates : List (List Entry)) : Labels :=
  let kEff := effWidth width state.trainX.rows
  let rows := bpannRaw state queries width exclude treeCandidates
  if exclude then trimCols kEff rows else rows

/-- Exact exhaustive `f64` reference search row over `[0, total)`. -/
def exactRow (state : State) (query : List Word) (width : Nat)
    (exclude : Bool) : List Label :=
  let total := state.trainX.rows
  let kEff := effWidth width total
  let poolK := poolWidth width total exclude
  let allEntries := scanRange state.scaleX state.xScale state.trainX query 0 total
  let topPool := (sortRow allEntries).take poolK
  let selfId := if exclude then
    findScaled state.scaleX state.xScale state.trainX query
  else none
  let pruned := if exclude then
    (excludeBpann topPool selfId).take poolK
  else topPool
  padLabels kEff ((pruned.take kEff).map Entry.label)

/-- Untrimmed exact exhaustive `f64` reference rows across all queries. -/
def exactRaw (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) : Labels :=
  (List.range queries.rows).map fun r ↦
    exactRow state (queries.data.getD r []) width exclude

/-- Exact exhaustive `f64` reference search with trailing negative column trimming. -/
def exactSearch (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) : Labels :=
  let kEff := effWidth width state.trainX.rows
  let rows := exactRaw state queries width exclude
  if exclude then trimCols kEff rows else rows

/-!
### Driver-Branched Dispatch, Concrete `BackendOp` & Relational Correspondence
-/

/-- Reachable candidate production branching explicitly by backend driver:
dispatches Flat Faiss exact search or BPANN small-N / dual-leg indexed+pending search. -/
def driverSearch (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (faissLabels : Labels) (faissDists : Batch)
    (treeEntries : List (List Entry)) : Labels :=
  match state.driver with
  | .flat => flatSearch queries width exclude faissLabels faissDists
  | .bpann => bpannSearch state queries width exclude treeEntries

/-- Output label row count matches query row count for all drivers. -/
theorem searchLen (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (faissLabels : Labels) (faissDists : Batch)
    (treeEntries : List (List Entry)) :
    (driverSearch state queries width exclude faissLabels faissDists treeEntries).length =
      queries.rows := by
  unfold driverSearch
  split <;> cases exclude <;> simp [flatSearch, flatRaw, bpannSearch, bpannRaw, trimLen]

/-- Relational backend candidate production parameterized by source-faithful native outputs:
- For `.flat`, `rawDists` and `rawLabels` must satisfy `FaissRel`.
- For `.bpann`, `treeEntries` must satisfy `TreeRel` on each query row. -/
inductive BackendRel (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool) : Labels → Prop where
  | flat (rawLabels : Labels) (rawDists : Batch)
      (hfaiss : FaissRel state.scaleX state.xScale state.numDim
        state.trainX queries width rawDists rawLabels)
      (hdrv : state.driver = .flat) :
      BackendRel state queries width exclude
        (flatSearch queries width exclude rawLabels rawDists)
  | bpann (treeEntries : List (List Entry))
      (hlen : treeEntries.length = queries.rows)
      (htree : ∀ r : Nat, r < queries.rows →
        TreeRel state.scaleX state.xScale state.indexed state.numDim
          state.trainX (queries.data.getD r []) width exclude
          (treeEntries.getD r []))
      (hdrv : state.driver = .bpann) :
      BackendRel state queries width exclude
        (bpannSearch state queries width exclude treeEntries)

/-- Any candidate labels produced by relational backend execution match query rows. -/
theorem relLen (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (labels : Labels)
    (h : BackendRel state queries width exclude labels) :
    labels.length = queries.rows := by
  cases h with
  | flat rawLabels rawDists hfaiss hdrv =>
      cases exclude <;> simp [flatSearch, flatRaw, trimLen]
  | bpann treeEntries hlen htree hdrv =>
      cases exclude <;> simp [bpannSearch, bpannRaw, trimLen]

theorem bpannPad (state : State) (query : List Word) (width : Nat)
    (exclude : Bool) (treeRow : List Entry) :
    (bpannRow state query width exclude treeRow).length =
      effWidth width state.trainX.rows := by
  dsimp only [bpannRow, mergeSelf]
  split <;> exact labelsLen _ _

/-- Flat Faiss candidate rows are bounded by requested fetch width (and equal to `width`
before trailing-column trim). -/
theorem flatWidth (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (rawLabels : Labels) (rawDists : Batch) :
    ∀ row ∈ flatSearch queries width exclude rawLabels rawDists,
      row.length ≤ width := by
  cases exclude
  · intro row hrow
    simp only [flatSearch, Bool.false_eq_true, ↓reduceIte, flatRaw, flatRow] at hrow
    rcases List.mem_map.mp hrow with ⟨r, _, hr⟩
    rw [← hr, flatPad]
  · intro row hrow
    simp only [flatSearch, ↓reduceIte] at hrow
    exact trimBound width _ row hrow

/-- BPANN candidate rows are bounded by requested fetch width. -/
theorem bpannWidth (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (treeEntries : List (List Entry)) :
    ∀ row ∈ bpannSearch state queries width exclude treeEntries,
      row.length ≤ width := by
  have hk : effWidth width state.trainX.rows ≤ width :=
    Nat.min_le_left width state.trainX.rows
  cases exclude
  · intro row hrow
    simp only [bpannSearch, Bool.false_eq_true, ↓reduceIte, bpannRaw] at hrow
    rcases List.mem_map.mp hrow with ⟨r, _, hr⟩
    rw [← hr, bpannPad]
    exact hk
  · intro row hrow
    simp only [bpannSearch, ↓reduceIte] at hrow
    exact Nat.le_trans (trimBound (effWidth width state.trainX.rows) _ row hrow) hk

/-- Any candidate row produced by relational backend execution is bounded by fetch width. -/
theorem relWidth (state : State) (queries : Fitter.Matrix) (width : Nat)
    (exclude : Bool) (labels : Labels)
    (h : BackendRel state queries width exclude labels) :
    ∀ row ∈ labels, row.length ≤ width := by
  cases h with
  | flat rawLabels rawDists hfaiss hdrv =>
      exact flatWidth queries width exclude rawLabels rawDists
  | bpann treeEntries hlen htree hdrv =>
      exact bpannWidth state queries width exclude treeEntries

/-- Exact Rust orchestration refinement: when candidate labels are produced
by relational backend execution with native outputs satisfying `FaissRel` or `TreeRel`,
`Neighbor.refineRows` succeeds and reranks candidates with exact search dimensions. -/
theorem relRefine (state : State) (queries : Fitter.Matrix)
    (searchK : Nat) (exclude : Bool) (labels : Labels)
    (hrel : BackendRel state queries
      (fetchWidth state.driver searchK state.trainX.rows) exclude labels) :
    ∃ search, refineRows (toModel state) queries searchK labels = .ok search ∧
      search.labels.length = queries.rows ∧
      search.dist2s.length = queries.rows ∧
      (∀ row ∈ search.labels, row.length = searchK) ∧
      (∀ row ∈ search.dist2s, row.length = searchK) := by
  have hlen : labels.length = queries.rows :=
    relLen state queries _ exclude labels hrel
  exact rowsOk (toModel state) queries searchK labels hlen

/-- Concrete `Neighbor.BackendOp` for `state`:
executes source-faithful `f32` Flat Faiss search (`faissBatch` + `flatSearch`) when
`state.driver = .flat`, and executes BPANN search (`treeBatch` + `bpannSearch`) when
`state.driver = .bpann`. -/
def backendOp (state : State) : Neighbor.BackendOp :=
  fun queries width exclude ↦
    if !validMatrix queries || queries.cols != state.trainX.cols then
      .error .shape
    else
      match state.driver with
      | .flat =>
          let batch :=
            faissBatch state.scaleX state.xScale state.numDim state.trainX queries width
          .ok (flatSearch queries width exclude batch.2 batch.1)
      | .bpann =>
          let trees := treeBatch state queries width exclude
          .ok (bpannSearch state queries width exclude trees)

/-- Concrete `Neighbor.BackendOp` for the exact exhaustive `f64` reference search. -/
def exactOp (state : State) : Neighbor.BackendOp :=
  fun queries width exclude ↦
    if !validMatrix queries || queries.cols != state.trainX.cols then
      .error .shape
    else
      .ok (exactSearch state queries width exclude)

/-- `backendOp` always produces candidate labels satisfying `BackendRel`. -/
theorem backendSound (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool) (labels : Labels)
    (h : backendOp state queries width exclude = .ok labels) :
    BackendRel state queries width exclude labels := by
  unfold backendOp at h
  split at h
  · contradiction
  · cases hdrv : state.driver <;> simp only [hdrv] at h
    · injection h with h'
      rw [← h']
      exact BackendRel.flat _ _
        (faissExact state.scaleX state.xScale state.numDim state.trainX queries width) hdrv
    · injection h with h'
      rw [← h']
      refine BackendRel.bpann (treeBatch state queries width exclude) (by simp [treeBatch]) ?_ hdrv
      intro r hr
      refine ⟨List.range state.indexed, ?_, ?_⟩
      · intro idx hidx
        exact List.mem_range.mp hidx
      · simp [treeBatch, List.getD, hr]

/-- For `.flat` driver states, `backendOp` computes `flatSearch` over `faissBatch`. -/
theorem flatExec (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool)
    (hdrv : state.driver = .flat)
    (hvalid : validMatrix queries = true)
    (hcols : queries.cols = state.trainX.cols) :
    let batch :=
      faissBatch state.scaleX state.xScale state.numDim state.trainX queries width
    backendOp state queries width exclude =
      .ok (flatSearch queries width exclude batch.2 batch.1) := by
  simp [backendOp, hdrv, hvalid, hcols]

/-- For `.bpann` driver states, `backendOp` computes `bpannSearch` over `treeBatch`. -/
theorem bpannExec (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool)
    (hdrv : state.driver = .bpann)
    (hvalid : validMatrix queries = true)
    (hcols : queries.cols = state.trainX.cols) :
    backendOp state queries width exclude =
      .ok (bpannSearch state queries width exclude
        (treeBatch state queries width exclude)) := by
  simp [backendOp, hdrv, hvalid, hcols]

/-- When `state.trainX.rows <= 8192` (`smallLimit`), `bpannSearch` is identical to the
exhaustive `f64` reference `exactSearch` regardless of `treeEntries`, `indexed`, or `pending`. -/
theorem bpannSmall (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool) (treeEntries : List (List Entry))
    (hsmall : state.trainX.rows ≤ smallLimit) :
    bpannSearch state queries width exclude treeEntries =
      exactSearch state queries width exclude := by
  simp [bpannSearch, exactSearch, bpannRaw, exactRaw, bpannRow, exactRow, hsmall]

/-- For `.bpann` driver states with `state.trainX.rows <= 8192`, `backendOp` and `exactOp`
are judgmentally equal on all query batches. -/
theorem smallExec (state : State) (queries : Fitter.Matrix)
    (width : Nat) (exclude : Bool)
    (hdrv : state.driver = .bpann)
    (hsmall : state.trainX.rows ≤ smallLimit) :
    backendOp state queries width exclude =
      exactOp state queries width exclude := by
  simp [backendOp, exactOp, hdrv, bpannSmall state queries width exclude _ hsmall]

theorem takeIdem {α : Type} : ∀ (n : Nat) (xs : List α), (xs.take n).take n = xs.take n
  | 0, _ => rfl
  | _ + 1, [] => rfl
  | n + 1, x :: xs => congrArg (x :: ·) (takeIdem n xs)

/-- In large-N mode (`total > 8192`) when all rows are pending (`indexed = 0`, `pending > 0`)
and `exclude = false`, `bpannRow` produces the exact same labels as `exactRow`. -/
theorem largePending (state : State) (query : List Word) (width : Nat)
    (treeRow : List Entry)
    (hlarge : ¬ state.trainX.rows ≤ smallLimit)
    (hidx : state.indexed = 0)
    (hpend : state.pending > 0)
    (hdupes : uniqueEntries
      ((sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) =
      (sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false))
    (hsorted : sortRow
      ((sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) =
      (sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) :
    bpannRow state query width false treeRow =
      exactRow state query width false := by
  simp [bpannRow, exactRow, mergeSelf, hlarge, hidx, hpend, hdupes, hsorted]
  simp [poolWidth, takeIdem]

/-- In large-N mode (`total > 8192`) when all rows are indexed (`indexed > 0`, `treeReady = true`,
`pending = 0`), `exclude = false`, and `treeRow` supplies the top `poolWidth` candidates from
`scanRange`, `bpannRow` matches `exactRow`. -/
theorem largeSynced (state : State) (query : List Word) (width : Nat)
    (treeRow : List Entry)
    (hlarge : ¬ state.trainX.rows ≤ smallLimit)
    (hidx : state.indexed > 0)
    (hready : state.treeReady = true)
    (hpend : state.pending = 0)
    (htree : treeRow.take (max (indexWidth width state.trainX.rows false) 1) =
      (sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false))
    (hdupes : uniqueEntries
      ((sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) =
      (sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false))
    (hsorted : sortRow
      ((sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) =
      (sortRow (scanRange state.scaleX state.xScale state.trainX query 0 state.trainX.rows)).take
        (poolWidth width state.trainX.rows false)) :
    bpannRow state query width false treeRow =
      exactRow state query width false := by
  simp [bpannRow, exactRow, mergeSelf, hlarge, hidx, hready, hpend, htree, hdupes, hsorted]
  simp [poolWidth, takeIdem]

/-- Step equation for gatherRow folding. -/
theorem gatherCons (rows : Batch) (idx : Nat) (tail : List Nat) :
    gatherRow rows (idx :: tail) = (do
      let rest ← gatherRow rows tail
      match rows[idx]? with
      | some row => pure (row :: rest)
      | none => throw NearError.index) := rfl

/-- Gathering in-bounds indices produces a batch matching the index length. -/
theorem gatherOk (rows : Batch) (indices : List Nat)
    (h : ∀ i ∈ indices, i < rows.length) :
    ∃ batch, gatherRow rows indices = .ok batch ∧ batch.length = indices.length := by
  induction indices with
  | nil =>
      refine ⟨[], rfl, rfl⟩
  | cons idx tail ih =>
      have htail : ∀ i ∈ tail, i < rows.length := fun i hi ↦ h i (List.mem_cons_of_mem idx hi)
      obtain ⟨tailBatch, htailOk, htailLen⟩ := ih htail
      have hidx : idx < rows.length := h idx (List.Mem.head tail)
      have hget : rows[idx]? = some rows[idx] := List.getElem?_eq_getElem hidx
      refine ⟨rows[idx] :: tailBatch, ?_, ?_⟩
      · rw [gatherCons, htailOk]
        simp [hget]
      · simp [htailLen]

/-- Sentinel distance evaluates to positive infinity. -/
theorem sentinelInf (model : Neighbor.Model) (query : List Word) :
    distance model query (-1) = posInf := by
  rfl

/-- Any negative label evaluates to positive infinity. -/
theorem sentinelNeg (model : Neighbor.Model) (query : List Word) (l : Label)
    (hl : l < 0) :
    distance model query l = posInf := by
  simp [distance, hl]

/-- Gate failure on wait preserves the original model without mutation. -/
theorem waitPreserve (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Append.Gate)
    (model : Append.Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (hvalid : validateAdd model x y variance = none)
    (hrows : x.rows > 0)
    (warped : Fitter.Matrix) (warpedVar : Option Fitter.Matrix)
    (hwarp : warpInput logOp model y variance = .ok (warped, warpedVar))
    (cause : Append.Effect)
    (hwait : gate .wait = .error cause) :
    addModel sqrtOp logOp gate model x y variance = ⟨model, some (.effect cause)⟩ := by
  have hpos : ¬x.rows = 0 := Nat.ne_of_gt hrows
  simp [addModel, hvalid, hpos, hwarp, hwait]

/-- Gate failure on append preserves the original model without mutation. -/
theorem appendPreserve (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Append.Gate)
    (model : Append.Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (hvalid : validateAdd model x y variance = none)
    (hrows : x.rows > 0)
    (warped : Fitter.Matrix) (warpedVar : Option Fitter.Matrix)
    (hwarp : warpInput logOp model y variance = .ok (warped, warpedVar))
    (hwait : gate .wait = .ok ())
    (cause : Append.Effect)
    (happend : gate .append = .error cause) :
    addModel sqrtOp logOp gate model x y variance = ⟨model, some (.effect cause)⟩ := by
  have hpos : ¬x.rows = 0 := Nat.ne_of_gt hrows
  simp [addModel, hvalid, hpos, hwarp, hwait, happend]

/-- Gate failure on persist retains the appended data in the model. -/
theorem persistRetain (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Append.Gate)
    (model : Append.Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (hvalid : validateAdd model x y variance = none)
    (hrows : x.rows > 0)
    (warped : Fitter.Matrix) (warpedVar : Option Fitter.Matrix)
    (hwarp : warpInput logOp model y variance = .ok (warped, warpedVar))
    (hwait : gate .wait = .ok ())
    (happend : gate .append = .ok ())
    (scaled : ModelStage)
    (hscale : updateScale sqrtOp logOp gate (appendData sqrtOp model x warped warpedVar) = scaled)
    (hscaleOk : scaled.error = none)
    (cause : Append.Effect)
    (hpersist : gate .persist = .error cause) :
    addModel sqrtOp logOp gate model x y variance = ⟨scaled.model, some (.effect cause)⟩ := by
  have hpos : ¬x.rows = 0 := Nat.ne_of_gt hrows
  simp [addModel, hvalid, hpos, hwarp, hwait, happend, hscale, hscaleOk, hpersist]

/-- Gate failure on auto retains the appended data in the model. -/
theorem autoRetain (sqrtOp : SqrtOp) (logOp : LogOp) (gate : Append.Gate)
    (model : Append.Model) (x y : Fitter.Matrix)
    (variance : Option Fitter.Matrix)
    (hvalid : validateAdd model x y variance = none)
    (hrows : x.rows > 0)
    (warped : Fitter.Matrix) (warpedVar : Option Fitter.Matrix)
    (hwarp : warpInput logOp model y variance = .ok (warped, warpedVar))
    (hwait : gate .wait = .ok ())
    (happend : gate .append = .ok ())
    (scaled : ModelStage)
    (hscale : updateScale sqrtOp logOp gate (appendData sqrtOp model x warped warpedVar) = scaled)
    (hscaleOk : scaled.error = none)
    (hpersist : gate .persist = .ok ())
    (cause : Append.Effect)
    (hauto : gate .auto = .error cause) :
    addModel sqrtOp logOp gate model x y variance = ⟨scaled.model, some (.effect cause)⟩ := by
  have hpos : ¬x.rows = 0 := Nat.ne_of_gt hrows
  simp [addModel, hvalid, hpos, hwarp, hwait, happend, hscale, hscaleOk, hpersist, hauto]

/-- External artifact evidence identifying compiled native shared libraries. -/
structure ArtifactEvidence where
  driver : Neighbor.Driver
  abiSymbol : String
  binaryHash : Word
  headerHash : Word
  deriving Repr, DecidableEq

/-- Compiled native C-ABI symbols bound to the Flat Faiss driver (`libfaiss_c`). -/
def faissSymbols : List String :=
  ["faiss_IndexFlatL2_new_with",
   "faiss_Index_add",
   "faiss_Index_search",
   "faiss_Index_reset",
   "faiss_Index_free"]

/-- Compiled Rust symbols bound to the BPANN driver (`bpann` + `ennbo`). -/
def bpannSymbols : List String :=
  ["bpann::backend::BpannBackend::search",
   "bpann::small_n_search::search_small_n",
   "bpann::large_n_search::search_large_n",
   "bpann::merge::merge_and_prune"]

/-- Driver-indexed native symbol table connecting `Neighbor.Driver` to compiled artifacts. -/
def driverSymbols (driver : Neighbor.Driver) : List String :=
  match driver with
  | .flat => faissSymbols
  | .bpann => bpannSymbols

/-- Artifact boundary description for each driver. -/
def abiBoundary (driver : Neighbor.Driver) : String :=
  match driver with
  | .flat => "libfaiss_c C-ABI IndexFlatL2 f32 search boundary (FaissRel)"
  | .bpann => "bpann small-N flat f64 + large-N B+ tree/pending search boundary (TreeRel)"

#print axioms scaleId
#print axioms scaleDiv
#print axioms widthFlat
#print axioms widthBpann
#print axioms widthBound
#print axioms tieBreak
#print axioms labelsLen
#print axioms distPad
#print axioms flatPad
#print axioms epsSelf
#print axioms infDistinct
#print axioms flatRepeat
#print axioms trimLe
#print axioms trimLen
#print axioms trimBound
#print axioms faissExact
#print axioms nativeCast
#print axioms epsExclude
#print axioms memTake
#print axioms memInsert
#print axioms memSort
#print axioms treeExact
#print axioms duplicateLeg
#print axioms appendRows
#print axioms appendPending
#print axioms appendDirty
#print axioms syncClear
#print axioms syncIndexed
#print axioms syncDirty
#print axioms persistClean
#print axioms persistCount
#print axioms releaseZero
#print axioms rescaleReset
#print axioms rescalePending
#print axioms rescaleDirty
#print axioms padLen
#print axioms insertLen
#print axioms sortLen
#print axioms refineLen
#print axioms rowsOk
#print axioms searchLen
#print axioms relLen
#print axioms bpannPad
#print axioms flatWidth
#print axioms bpannWidth
#print axioms relWidth
#print axioms relRefine
#print axioms backendSound
#print axioms flatExec
#print axioms bpannExec
#print axioms bpannSmall
#print axioms smallExec
#print axioms takeIdem
#print axioms largePending
#print axioms largeSynced
#print axioms gatherCons
#print axioms gatherOk
#print axioms sentinelInf
#print axioms sentinelNeg
#print axioms waitPreserve
#print axioms appendPreserve
#print axioms persistRetain
#print axioms autoRetain

end BauxResearch.Semantics.Backend
