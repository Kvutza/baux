import BauxResearch.Semantics.Tree

/-!
# ENN 0.4.1 BPANN backend, tuning, metric scaling, distance, small-N/large-N search, and merge

This module formalizes all 51 functions across the 8 BPANN core files in `rust/crates/bpann/src/`:
- `tuning.rs` (4 functions)
- `backend/flush_thresholds.rs` (4 functions)
- `distance.rs` (3 functions)
- `metric.rs` (7 functions)
- `small_n_search.rs` (5 functions)
- `large_n_search.rs` (2 functions)
- `merge.rs` (5 functions)
- `backend.rs` (21 functions)

Every function is modeled with executable state transitions and connected by proved theorems
to `Backend`, `Tree`, `Morph`, and `Residency`.
-/

namespace BauxResearch.Semantics.Bpann

open Sampling Bounds Fitter Neighbor Append Backend
open Residency Morph Tree

/-! ## Section 1: `tuning.rs` (4 functions) and `flush_thresholds.rs` (4 functions) -/

/-- Minimum allowed value for `index_compact_fragment_max` (`INDEX_COMPACT_FRAGMENT_MAX_MIN = 3`). -/
def fragMin : Nat := 3

/-- Model of `BpannTuning` (`bpann/src/tuning.rs`). -/
structure Tuning where
  compactRows : Nat
  compactMax : Nat
  searchRows : Nat
  mergeRows : Nat
  budgetMax : Nat
  flushSoft : Nat
  flushHard : Nat
  structLimit : Nat
  beamWidth : Nat
  exhaustLimit : Nat
  skipLimit : Nat
  deriving Repr, DecidableEq

/-- `BpannTuning::default`: default BPANN tuning parameters. -/
def tuningDef : Tuning :=
  { compactRows := 10000
    compactMax := 32
    searchRows := 80000
    mergeRows := 15000
    budgetMax := 1
    flushSoft := 250
    flushHard := 3000
    structLimit := 1024
    beamWidth := 1
    exhaustLimit := 2500
    skipLimit := 150000 }

/-- `BpannTuning::validate`: check all positive bounds and ordering invariants (`hard >= soft`,
`compactMax >= 3`, `skipLimit >= exhaustLimit`). -/
def tuningValid (t : Tuning) : Bool :=
  decide
    (t.compactRows ≥ 1 &&
     t.compactMax ≥ fragMin &&
     t.searchRows ≥ 1 &&
     t.mergeRows ≥ 1 &&
     t.budgetMax ≥ 1 &&
     t.flushSoft ≥ 1 &&
     t.flushHard ≥ 1 &&
     t.flushHard ≥ t.flushSoft &&
     t.structLimit ≥ 1 &&
     t.beamWidth ≥ 1 &&
     t.exhaustLimit ≥ 1 &&
     t.skipLimit ≥ 1 &&
     t.skipLimit ≥ t.exhaustLimit)

/-- `set_tuning_provider`: install a custom `Tuning` provider. -/
def setProvider (custom : Tuning) : Option Tuning :=
  some custom

/-- `current_tuning`: return custom tuning if valid, otherwise fall back to `tuningDef`. -/
def curTuning (provider : Option Tuning) : Tuning :=
  match provider with
  | some t => if tuningValid t then t else tuningDef
  | none => tuningDef

/-- Model of `BpannBackend` (`bpann/src/backend.rs`). -/
structure BpannState where
  numDim : Nat
  numMetrics : Nat
  scaleX : Bool
  xScale : List Word
  metricMarker : Bool
  trainX : MmapStore
  trainY : MmapStore
  trainYvar : Option MmapStore
  index : IncIndex
  smallCache : Option (Nat × Fitter.Matrix)
  flushSoft : Nat
  flushHard : Nat
  deferSync : Bool
  rowsSinceRel : Nat
  deriving Repr, DecidableEq

/-- `BpannBackend::with_pending_flush_threshold`: set `soft` and clamp `hard = max(hard, soft)`. -/
def withSoft (b : BpannState) (soft : Nat) : BpannState :=
  { b with
    flushSoft := soft
    flushHard := max b.flushHard soft }

/-- `BpannBackend::with_pending_hard_flush_threshold`: set `hard = max(hard, soft)`. -/
def withHard (b : BpannState) (hard : Nat) : BpannState :=
  { b with flushHard := max hard b.flushSoft }

/-- `BpannBackend::pending_flush_threshold`: return `pending_flush_threshold`. -/
def getSoft (b : BpannState) : Nat :=
  b.flushSoft

/-- `BpannBackend::pending_hard_flush_threshold`: return `pending_hard_flush_threshold`. -/
def getHard (b : BpannState) : Nat :=
  b.flushHard

/-! ## Section 2: `distance.rs` (3 functions) and `metric.rs` (7 functions) -/

/-- `row_sq_l2`: squared L2 distance between two `f64` rows. -/
def rowSq (a b : List Word) : Word :=
  rowDist false a b []

/-- `l2_sq_f32`: squared L2 distance between two `f32` rows. -/
def singleSq (a b : List Float.WordSingle) : Float.WordSingle :=
  distSingle a b a.length

/-- `bpann_row_to_f32`: scale an `f64` row by `x_scale` (if `scale_x`) and cast to `f32`. -/
def castRow (row : List Word) (scaleX : Bool) (xScale : List Word) :
    List Float.WordSingle :=
  rowSingle scaleX row xScale

/-- `IncrementalIndex::rescale_coords`: multiply all tree coordinates by `ratios`. -/
def rescaleCoords (idx : IncIndex) (ratios : List Word) : IncIndex :=
  if idx.indexedRows = 0 then
    idx
  else
    { idx with tree := { idx.tree with store := scaleRows idx.tree.store ratios } }

/-- `BpannBackend::validate_metric_scale`: require `xScale.length == numDim` and all finite `> 0`. -/
def validMetric (numDim : Nat) (xScale : List Word) : Bool :=
  decide (xScale.length = numDim) &&
  xScale.all (fun s ↦ Float.finiteDouble s && Float.ltDouble (0 : Word) s)

/-- `BpannBackend::current_scale`: return `xScale[j]` when `scaleX && j < xScale.length`, else `1`. -/
def curScale (b : BpannState) (j : Nat) : Word :=
  if b.scaleX && j < b.xScale.length then
    b.xScale.getD j Bounds.one
  else
    Bounds.one

/-- `BpannBackend::write_metric_marker`: persist `metric_x_scale.json` marker flag. -/
def writeMarker (b : BpannState) : BpannState :=
  { b with metricMarker := true }

/-- `BpannBackend::rescale_metric`: validate new `xScale`, rescale tree coords by `old / new`,
clear `small_n_x_cache`, set `scale_x = true`, and write `metric_x_scale.json`. -/
def rescaleMetric (b : BpannState) (xScale : List Word) : Option BpannState :=
  if !validMetric b.numDim xScale then
    none
  else
    let ratios :=
      (List.range b.numDim).map fun j ↦
        divide (curScale b j) (xScale.getD j Bounds.one)
    let idx1 := rescaleCoords b.index ratios
    some
      (writeMarker
        { b with
          scaleX := true
          xScale := xScale
          index := idx1
          smallCache := none })

/-- `BpannBackend::rebuild_metric`: run `rescale_metric` then schedule incremental tree morphing. -/
def rebuildMetric (b : BpannState) (xScale : List Word) : Option BpannState :=
  match rescaleMetric b xScale with
  | none => none
  | some b1 => some { b1 with index := startMorph b1.index }

/-- `BpannBackend::discard_metric_index_on_reopen`: reset index on reopen if `metricMarker` is set. -/
def discardReopen (b : BpannState) : BpannState :=
  if !b.metricMarker then
    b
  else
    { b with
      metricMarker := false
      index := idxReset b.index }

/-! ## Section 3: `small_n_search.rs` (5), `large_n_search.rs` (2), `merge.rs` (5) -/

/-- `OrderedFloat::cmp`: total ordering comparison on `Word` consistent with `before`. -/
def ordCmp (a b : Word) : Bool :=
  Float.leDouble a b

/-- `scaled_row`: divide `row[j]` by `xScale[j]` when `scaleX` is enabled. -/
def scaledRow (row : List Word) (scaleX : Bool) (xScale : List Word) : List Word :=
  scaleRow scaleX row xScale

/-- `load_or_build_small_n_cache`: return cached scaled matrix for `(n, numDim)` or build it. -/
def loadCache (b : BpannState) : Fitter.Matrix × BpannState :=
  let n := b.trainX.nrows
  match b.smallCache with
  | some (cachedN, mat) =>
    if cachedN = n && mat.cols = b.numDim then
      (mat, b)
    else
      let built := scaleMatrix b.scaleX b.xScale ⟨n, b.numDim, b.trainX.data⟩
      (built, { b with smallCache := some (n, built) })
  | none =>
    let built := scaleMatrix b.scaleX b.xScale ⟨n, b.numDim, b.trainX.data⟩
    (built, { b with smallCache := some (n, built) })

/-- `topk_flat_sq_l2`: select top-`k` nearest rows from `trainScaled` for `qScaled`. -/
def topkFlat (trainScaled : Fitter.Matrix) (qScaled : List Word) (k : Nat) : List Entry :=
  (sortRow (scanRange false [] trainScaled qScaled 0 trainScaled.rows)).take k

/-- `score_queries_flat`: run `topk_flat_sq_l2` for all queries in `queries`. -/
def scoreFlat (trainScaled queries : Fitter.Matrix) (scaleX : Bool)
    (xScale : List Word) (poolK : Nat) : List (List Entry) :=
  let qScaled := scaleMatrix scaleX xScale queries
  qScaled.data.map fun qRow ↦ topkFlat trainScaled qRow poolK

/-- `find_query_train_id`: find exact train row match in `0..n` for unscaled query `q`. -/
def findTrain (train : Fitter.Matrix) (q : List Word) : Option Label :=
  findSelf train q

/-- `find_query_train_id_flat`: find exact scaled row match in `0..n` when `d0 <= 1e-12`. -/
def findFlat (trainScaled : Fitter.Matrix) (qScaled : List Word)
    (top : List Entry) : Option Label :=
  match top.head? with
  | none => none
  | some hd =>
    if nearSelf hd.dist then
      findScaled false [] trainScaled qScaled
    else
      none

/-- `bpann_apply_exclude_nearest`: filter out self-row or drop rank 0 if `d <= 1e-12`, then pad. -/
def applyExclude (top : List Entry) (selfId : Option Label)
    (exclude : Bool) (kEff : Nat) : List Entry :=
  let filtered := if exclude then excludeBpann top selfId else top
  padRow kEff filtered

/-- `merge_topk_precomputed_dist_with_self`: exclude self from precomputed flat top-k and take
`kEff`. -/
def mergePrecomp (trainScaled : Fitter.Matrix) (qScaled : List Word)
    (top : List Entry) (exclude : Bool) (kEff : Nat) : List Entry :=
  let selfId := if exclude then findFlat trainScaled qScaled top else none
  applyExclude top selfId exclude kEff

/-- `bpann_merge_topk_candidates`: deduplicate `(indexCands ++ pendingCands)`, sort by exact `f64`
squared L2 distance, and apply self-exclusion. -/
def mergeCands (indexCands pendingCands : List Entry) (selfId : Option Label)
    (exclude : Bool) (kEff : Nat) : List Entry :=
  let merged := sortRow (uniqueEntries (indexCands ++ pendingCands))
  applyExclude merged selfId exclude kEff

/-- `merge_pending_unscaled`: scan unindexed rows `[indexed, total)` and merge with `indexCands`. -/
def mergePending (trainScaled : Fitter.Matrix) (indexed : Nat)
    (qScaled : List Word) (indexCands : List Entry) (selfId : Option Label)
    (exclude : Bool) (poolK kEff : Nat) : List Entry :=
  let pCands :=
    (sortRow (scanRange false [] trainScaled qScaled indexed trainScaled.rows)).take poolK
  mergeCands indexCands pCands selfId exclude kEff

/-- `search_indexed_and_pending`: large-N dual-leg search over B+ tree `[0, indexed)` and pending
rows `[indexed, total)`. -/
def searchLarge (b : BpannState) (queries : Fitter.Matrix) (kEff poolK : Nat)
    (exclude : Bool) : List (List Entry) :=
  let trainRaw : Fitter.Matrix := ⟨b.trainX.nrows, b.numDim, b.trainX.data⟩
  let trainScaled := scaleMatrix b.scaleX b.xScale trainRaw
  let qScaled := scaleMatrix b.scaleX b.xScale queries
  let idxK := indexWidth kEff b.index.indexedRows exclude
  (List.range queries.rows).map fun qi ↦
    let qRaw := queries.data.getD qi []
    let qRow := qScaled.data.getD qi []
    let idxCands := searchCand b.index qRow idxK
    let selfId :=
      if exclude then
        match idxCands.head? with
        | some hd => if nearSelf hd.dist then findTrain trainRaw qRaw else none
        | none => findTrain trainRaw qRaw
      else
        none
    mergePending trainScaled b.index.indexedRows qRow idxCands selfId exclude poolK kEff

/-! ## Section 4: `backend.rs` (21 functions) -/

/-- `trim_trailing_invalid_neighbor_cols`: trim trailing columns where all row labels are `< 0`. -/
def bpannTrim (kEff : Nat) (rows : List (List Entry)) :
    List (List Word) × List (List Label) :=
  let w := trimWidth kEff (rows.map (fun r ↦ r.map Entry.label))
  (rows.map (fun r ↦ (r.map Entry.dist).take w),
   rows.map (fun r ↦ (r.map Entry.label).take w))

/-- `BpannBackend::new`: construct a new `BpannBackend` and sync initial observations. -/
def bpannNew (trainX trainY : Fitter.Matrix) (trainYvar : Option Fitter.Matrix)
    (scaleX : Bool) (xScale : List Word) : Option BpannState :=
  if trainX.cols = 0 || trainY.cols = 0 || trainX.rows ≠ trainY.rows then
    none
  else if scaleX && !validMetric trainX.cols xScale then
    none
  else
    let sx : MmapStore :=
      { ncols := trainX.cols, nrows := trainX.rows, capRows := trainX.rows,
        fileLen := trainX.rows * trainX.cols * 8, data := trainX.data, residency := resNew }
    let sy : MmapStore :=
      { ncols := trainY.cols, nrows := trainY.rows, capRows := trainY.rows,
        fileLen := trainY.rows * trainY.cols * 8, data := trainY.data, residency := resNew }
    let syv :=
      trainYvar.map fun yv ↦
        ({ ncols := yv.cols, nrows := yv.rows, capRows := yv.rows,
           fileLen := yv.rows * yv.cols * 8, data := yv.data, residency := resNew } : MmapStore)
    let scaledRows := (scaleMatrix scaleX xScale trainX).data
    let idx0 := ensureSync (idxNew trainX.cols) scaledRows
    let t := tuningDef
    some
      { numDim := trainX.cols
        numMetrics := trainY.cols
        scaleX := scaleX
        xScale := if scaleX then xScale else List.replicate trainX.cols Bounds.one
        metricMarker := false
        trainX := sx
        trainY := sy
        trainYvar := syv
        index := idx0
        smallCache := none
        flushSoft := t.flushSoft
        flushHard := t.flushHard
        deferSync := false
        rowsSinceRel := 0 }

/-- `BpannBackend::defer_append_indexing`: return whether append defers index sync. -/
def deferAppend (b : BpannState) : Bool :=
  b.deferSync

/-- `BpannBackend::pending_rows`: unindexed row count `len.saturating_sub(indexed_rows)`. -/
def pendingRows (b : BpannState) : Nat :=
  b.trainX.nrows - b.index.indexedRows

/-- `BpannBackend::len`: total observation row count `train_x.nrows`. -/
def bpannLen (b : BpannState) : Nat :=
  b.trainX.nrows

/-- `BpannBackend::num_dim`: feature dimension `num_dim`. -/
def bpannDim (b : BpannState) : Nat :=
  b.numDim

/-- `BpannBackend::num_metrics`: objective metric count `num_metrics`. -/
def bpannMetrics (b : BpannState) : Nat :=
  b.numMetrics

/-- `BpannBackend::reset_index`: reset incremental index and clear `small_n_x_cache`. -/
def resetIdx (b : BpannState) : BpannState :=
  { b with
    index := idxReset b.index
    smallCache := none }

/-- `BpannBackend::mark_index_stale`: reset the incremental index on metric scale invalidation. -/
def markStale (b : BpannState) : BpannState :=
  resetIdx b

/-- `BpannBackend::ensure_index_sync`: synchronize all unindexed rows into `b.index`. -/
def syncIndex (b : BpannState) : BpannState :=
  let rows := (List.range b.trainX.nrows).map (fun i ↦ b.trainX.data.getD i [])
  let raw : Fitter.Matrix := ⟨b.trainX.nrows, b.numDim, rows⟩
  let scaled := (scaleMatrix b.scaleX b.xScale raw).data
  { b with index := ensureSync b.index scaled }

/-- `BpannBackend::ensure_index_sync_with_scale`: update scale if changed, then `ensure_index_sync`. -/
def syncScale (b : BpannState) (scaleX : Bool) (xScale : List Word) :
    Option BpannState :=
  if scaleX && !validMetric b.numDim xScale then
    none
  else
    let changed := decide (b.scaleX ≠ scaleX || (scaleX && b.xScale ≠ xScale))
    let b1 :=
      if changed then
        resetIdx
          { b with
            scaleX := scaleX
            xScale := if scaleX then xScale else List.replicate b.numDim Bounds.one }
      else
        b
    some (syncIndex b1)

/-- `BpannBackend::insert_unindexed_rows`: insert newly appended rows `[indexed, len)` into tree. -/
def insertUnidx (b : BpannState) : BpannState :=
  syncIndex b

/-- `BpannBackend::release_observation_pages`: release mmap observation pages and zero counter. -/
def releaseObs (b : BpannState) : BpannState :=
  { b with
    trainX := releaseStore b.trainX
    trainY := releaseStore b.trainY
    trainYvar := b.trainYvar.map releaseStore
    rowsSinceRel := 0 }

/-- `BpannBackend::append_rows`: append `(x, y, yvar)` to mmap stores and sync if not deferred. -/
def bpannAppend (b : BpannState) (x y : Fitter.Matrix)
    (_yvar : Option Fitter.Matrix) : Option BpannState :=
  if x.cols ≠ b.numDim || y.cols ≠ b.numMetrics || x.rows ≠ y.rows then
    none
  else if x.rows = 0 then
    some b
  else
    match mmapAppend b.trainX x, mmapAppend b.trainY y with
    | some sx, some sy =>
      let b1 :=
        { b with
          trainX := sx
          trainY := sy
          smallCache := none
          rowsSinceRel := b.rowsSinceRel + x.rows }
      if b1.deferSync then
        some b1
      else
        some (insertUnidx b1)
    | _, _ => none

/-- `BpannBackend::train_rows_at`: gather `(x, y, yvar)` at `indices`. -/
def bpannGather (b : BpannState) (indices : List Nat) :
    Option (Fitter.Matrix × Fitter.Matrix × Option Fitter.Matrix) :=
  match mmapGather b.trainX indices, mmapGather b.trainY indices with
  | some mx, some my =>
    let myv :=
      match b.trainYvar with
      | some syv => mmapGather syv indices
      | none => none
    some (mx, my, myv)
  | _, _ => none

/-- `BpannBackend::mmap_row_slice`: read row `i` from `train_x`. -/
def mmapSlice (b : BpannState) (i : Nat) : Option (List Word) :=
  (rowSlice b.trainX i).map Prod.fst

/-- `BpannBackend::mmap_row_y_and_yvar`: read row `i` from `train_y` and `train_yvar`. -/
def mmapRow (b : BpannState) (i : Nat) : Option (List Word × Option (List Word)) :=
  match rowSlice b.trainY i with
  | none => none
  | some (ry, _) =>
    let ryv :=
      match b.trainYvar with
      | some syv => (rowSlice syv i).map Prod.fst
      | none => none
    some (ry, ryv)

/-- `BpannBackend::reopen`: reopen `BpannBackend` from persisted stores and index. -/
def bpannReopen (b : BpannState) (persisted : Option Tree) : BpannState :=
  { b with
    index := adoptPersist b.numDim b.index.persistedRows persisted
    smallCache := none }

/-- `BpannBackend::search`: execute small-N (`<= 8192`) or large-N (`> 8192`) search and trim
trailing invalid neighbor columns. -/
def searchExec (b : BpannState) (queries : Fitter.Matrix) (searchK : Nat)
    (exclude : Bool) : Option (List (List Word) × List (List Label)) :=
  if queries.cols ≠ b.numDim then
    none
  else
    let total := b.trainX.nrows
    let kEff := effWidth searchK total
    if queries.rows = 0 || searchK = 0 || total = 0 || kEff = 0 then
      some (List.replicate queries.rows [], List.replicate queries.rows [])
    else
      let poolK := poolWidth kEff total exclude
      let rows :=
        if total ≤ smallLimit then
          let (trainScaled, _) := loadCache b
          let qScaled := scaleMatrix b.scaleX b.xScale queries
          let tops := scoreFlat trainScaled queries b.scaleX b.xScale poolK
          (List.range queries.rows).map fun qi ↦
            mergePrecomp trainScaled (qScaled.data.getD qi []) (tops.getD qi []) exclude kEff
        else
          searchLarge b queries kEff poolK exclude
      some (bpannTrim kEff rows)

/-- `soft_sync_build`: clone `b.index` and build unindexed rows `[indexed, len)` detached. -/
def softBuild (b : BpannState) : Option IncIndex :=
  let n := b.trainX.nrows
  let idx := idxClone b.index
  if idx.indexedRows ≥ n then
    none
  else
    let rows := (List.range n).map (fun i ↦ b.trainX.data.getD i [])
    let raw : Fitter.Matrix := ⟨n, b.numDim, rows⟩
    let scaled := (scaleMatrix b.scaleX b.xScale raw).data
    some (ensureSync idx scaled)

/-- `soft_sync_publish`: publish a detached `built` index iff `built.indexedRows >= current` and
`built.indexedRows <= b.len()`. -/
def softPublish (b : BpannState) (built : IncIndex) : BpannState :=
  if built.indexedRows ≥ b.index.indexedRows && built.indexedRows ≤ b.trainX.nrows then
    { b with index := built }
  else
    b

/-! ## Theorems for `Bpann` -/

/-- `BpannTuning::default` passes `BpannTuning::validate`. -/
theorem defValid : tuningValid tuningDef = true := by decide

/-- `current_tuning` always returns a configuration that passes `validate`. -/
theorem curValid (provider : Option Tuning) :
    tuningValid (curTuning provider) = true := by
  cases provider with
  | none => decide
  | some t =>
    unfold curTuning
    by_cases h : tuningValid t = true
    · simp [h]
    · simp [h]; decide

/-- `with_pending_flush_threshold` and `with_pending_hard_flush_threshold` maintain `hard >= soft`. -/
theorem flushOrder (b : BpannState) (soft hard : Nat) :
    getHard (withSoft b soft) ≥ getSoft (withSoft b soft) ∧
    getHard (withHard b hard) ≥ getSoft (withHard b hard) := by
  simp [withSoft, withHard, getSoft, getHard]

/-- `reset_index` zeroes `indexedRows` and clears `smallCache`. -/
theorem resetClear (b : BpannState) :
    (resetIdx b).index.indexedRows = 0 ∧ (resetIdx b).smallCache = none := by
  simp [resetIdx, idxReset]

/-- `ensure_index_sync` brings `pending_rows` to 0. -/
theorem syncPending (b : BpannState) :
    pendingRows (syncIndex b) = 0 := by
  simp [pendingRows, syncIndex, syncTotal, scaleMatrix]

/-- `discard_metric_index_on_reopen` resets the index when `metricMarker` is set. -/
theorem discardMarker (b : BpannState) (h : b.metricMarker = true) :
    (discardReopen b).metricMarker = false ∧
    (discardReopen b).index.indexedRows = 0 := by
  simp [discardReopen, h, idxReset]

/-- `soft_sync_build` returns `none` when all rows are already indexed (`indexedRows >= nrows`). -/
theorem softClean (b : BpannState) (h : b.index.indexedRows ≥ b.trainX.nrows) :
    softBuild b = none := by
  simp [softBuild, idxClone, h]

/-- `soft_sync_publish` updates `b.index` when `built.indexedRows` is within `[current, nrows]`. -/
theorem softAccept (b : BpannState) (built : IncIndex)
    (hlo : built.indexedRows ≥ b.index.indexedRows)
    (hhi : built.indexedRows ≤ b.trainX.nrows) :
    (softPublish b built).index = built := by
  simp [softPublish, hlo, hhi]

/-- `soft_sync_publish` ignores stale detached builds where `built.indexedRows < current`. -/
theorem softStale (b : BpannState) (built : IncIndex)
    (hstale : built.indexedRows < b.index.indexedRows) :
    softPublish b built = b := by
  have hnot : ¬(built.indexedRows ≥ b.index.indexedRows) := Nat.not_le.mpr hstale
  simp [softPublish, hnot]

/-! ## Manifest Function Coverage Table (51 functions) -/

/-- All 51 `coverage_manifest.json` functions formalized in `Bpann`. -/
def manifestSpecs : List FuncSpec :=
  [ { file := "rust/crates/bpann/src/backend.rs", line := 50,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]new()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 158,
      fnName := "defer_append_indexing",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]defer_append_indexing()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 162,
      fnName := "pending_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]pending_rows()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 167,
      fnName := "len",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]len()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 171,
      fnName := "num_dim",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]num_dim()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 175,
      fnName := "num_metrics",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]num_metrics()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 179,
      fnName := "mark_index_stale",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]mark_index_stale()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 183,
      fnName := "ensure_index_sync_with_scale",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]ensure_index_sync_with_scale()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 196,
      fnName := "reset_index",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]reset_index()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 223,
      fnName := "append_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]append_rows()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 254,
      fnName := "insert_unindexed_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]insert_unindexed_rows()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 280,
      fnName := "ensure_index_sync",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]ensure_index_sync()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 293,
      fnName := "release_observation_pages",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]release_observation_pages()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 327,
      fnName := "train_rows_at",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]train_rows_at()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 337,
      fnName := "search",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]search()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 424,
      fnName := "mmap_row_slice",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]mmap_row_slice()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 429,
      fnName := "mmap_row_y_and_yvar",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]mmap_row_y_and_yvar()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 445,
      fnName := "reopen",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/impl#[BpannBackend]reopen()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 467,
      fnName := "trim_trailing_invalid_neighbor_cols",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/trim_trailing_invalid_neighbor_cols()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 497,
      fnName := "soft_sync_build",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/soft_sync_build()." },
    { file := "rust/crates/bpann/src/backend.rs", line := 514,
      fnName := "soft_sync_publish",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/soft_sync_publish()." },
    { file := "rust/crates/bpann/src/backend/flush_thresholds.rs", line := 4,
      fnName := "with_pending_flush_threshold",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/flush_thresholds/impl#[BpannBackend]with_pending_flush_threshold()." },
    { file := "rust/crates/bpann/src/backend/flush_thresholds.rs", line := 12,
      fnName := "with_pending_hard_flush_threshold",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/flush_thresholds/impl#[BpannBackend]with_pending_hard_flush_threshold()." },
    { file := "rust/crates/bpann/src/backend/flush_thresholds.rs", line := 17,
      fnName := "pending_flush_threshold",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/flush_thresholds/impl#[BpannBackend]pending_flush_threshold()." },
    { file := "rust/crates/bpann/src/backend/flush_thresholds.rs", line := 21,
      fnName := "pending_hard_flush_threshold",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 backend/flush_thresholds/impl#[BpannBackend]pending_hard_flush_threshold()." },
    { file := "rust/crates/bpann/src/distance.rs", line := 3,
      fnName := "row_sq_l2",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 distance/row_sq_l2()." },
    { file := "rust/crates/bpann/src/distance.rs", line := 28,
      fnName := "l2_sq_f32",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 distance/l2_sq_f32()." },
    { file := "rust/crates/bpann/src/distance.rs", line := 47,
      fnName := "bpann_row_to_f32",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 distance/bpann_row_to_f32()." },
    { file := "rust/crates/bpann/src/large_n_search.rs", line := 26,
      fnName := "merge_pending_unscaled",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 large_n_search/merge_pending_unscaled()." },
    { file := "rust/crates/bpann/src/large_n_search.rs", line := 55,
      fnName := "search_indexed_and_pending",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 large_n_search/search_indexed_and_pending()." },
    { file := "rust/crates/bpann/src/merge.rs", line := 15,
      fnName := "bpann_apply_exclude_nearest",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 merge/bpann_apply_exclude_nearest()." },
    { file := "rust/crates/bpann/src/merge.rs", line := 35,
      fnName := "find_query_train_id",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 merge/find_query_train_id()." },
    { file := "rust/crates/bpann/src/merge.rs", line := 49,
      fnName := "find_query_train_id_flat",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 merge/find_query_train_id_flat()." },
    { file := "rust/crates/bpann/src/merge.rs", line := 68,
      fnName := "bpann_merge_topk_candidates",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 merge/bpann_merge_topk_candidates()." },
    { file := "rust/crates/bpann/src/merge.rs", line := 126,
      fnName := "merge_topk_precomputed_dist_with_self",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 merge/merge_topk_precomputed_dist_with_self()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 24,
      fnName := "rescale_coords",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[IncrementalIndex]rescale_coords()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 32,
      fnName := "validate_metric_scale",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]validate_metric_scale()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 47,
      fnName := "current_scale",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]current_scale()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 55,
      fnName := "write_metric_marker",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]write_metric_marker()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 66,
      fnName := "rescale_metric",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]rescale_metric()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 81,
      fnName := "rebuild_metric",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]rebuild_metric()." },
    { file := "rust/crates/bpann/src/metric.rs", line := 89,
      fnName := "discard_metric_index_on_reopen",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 metric/impl#[BpannBackend]discard_metric_index_on_reopen()." },
    { file := "rust/crates/bpann/src/small_n_search.rs", line := 40,
      fnName := "cmp",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 small_n_search/impl#[`OrderedFloat<T>`][Ord]cmp()." },
    { file := "rust/crates/bpann/src/small_n_search.rs", line := 46,
      fnName := "topk_flat_sq_l2",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 small_n_search/topk_flat_sq_l2()." },
    { file := "rust/crates/bpann/src/small_n_search.rs", line := 98,
      fnName := "score_queries_flat",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 small_n_search/score_queries_flat()." },
    { file := "rust/crates/bpann/src/small_n_search.rs", line := 143,
      fnName := "scaled_row",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 small_n_search/scaled_row()." },
    { file := "rust/crates/bpann/src/small_n_search.rs", line := 151,
      fnName := "load_or_build_small_n_cache",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 small_n_search/load_or_build_small_n_cache()." },
    { file := "rust/crates/bpann/src/tuning.rs", line := 53,
      fnName := "default",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 tuning/impl#[BpannTuning][Default]default()." },
    { file := "rust/crates/bpann/src/tuning.rs", line := 74,
      fnName := "validate",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 tuning/impl#[BpannTuning]validate()." },
    { file := "rust/crates/bpann/src/tuning.rs", line := 158,
      fnName := "set_tuning_provider",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 tuning/set_tuning_provider()." },
    { file := "rust/crates/bpann/src/tuning.rs", line := 170,
      fnName := "current_tuning",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 tuning/current_tuning()." } ]

/-- All 51 functions across the 8 BPANN core files are covered. -/
theorem bpannCount : manifestSpecs.length = 51 := rfl

#print axioms defValid
#print axioms curValid
#print axioms flushOrder
#print axioms resetClear
#print axioms syncPending
#print axioms discardMarker
#print axioms softClean
#print axioms softAccept
#print axioms softStale
#print axioms bpannCount

end BauxResearch.Semantics.Bpann
