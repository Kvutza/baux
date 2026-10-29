import BauxResearch.Semantics.Flush

/-!
# ENN 0.4.1 observation metadata, row storage, in-memory backend, and unified EnnBackend dispatch

This module formalizes all 66 functions across:
- `rust/crates/bpann/src/observation.rs` (17 functions)
- `rust/crates/ennbo/src/backend/disk_observation.rs` (2 functions)
- `rust/crates/ennbo/src/backend/row_storage.rs` (6 functions)
- `rust/crates/ennbo/src/backend/storage_wire.rs` (1 function)
- `rust/crates/ennbo/src/backend/in_memory.rs` (15 functions)
- `rust/crates/ennbo/src/backend/mod.rs` (25 functions)
-/

namespace BauxResearch.Semantics.Storage

open Sampling Bounds Fitter Neighbor Append Backend
open Residency Morph Tree Bpann Flush

/-! ## Section 1: `bpann/src/observation.rs` (17) and `disk_observation.rs` (2) -/

/-- Format version constant (`FORMAT_VERSION = 1`). -/
def formatVer : Nat := 1

/-- Maximum feature dimension (`MAX_NUM_DIM = 100_000`). -/
def maxDim : Nat := 100000

/-- Maximum record byte stride (`MAX_RECORD_STRIDE = 8 * 1024 * 1024 = 8_388_608`). -/
def maxStride : Nat := 8388608

/-- Maximum row count limit (`u32::MAX = 4_294_967_295`). -/
def rowLimit : Nat := 4294967295

/-- Metadata model for `metadata.json` and sidecar files (`num_obs.bin`, `indexed_rows.bin`). -/
structure ObsMeta where
  formatVersion : Nat
  numObs : Option Nat
  sidecarObs : Option Nat
  numDim : Nat
  numMetrics : Nat
  scaleX : Bool
  indexBackend : Option String
  indexedRows : Option Nat
  sidecarIndexed : Option Nat
  extFields : List (String × String)
  deriving Repr, DecidableEq

/-- `bpann_validate_dim_limits`: require `numDim <= 100_000` and `recordStride <= 8 MiB`. -/
def validDim (numDim recordStride : Nat) : Bool :=
  decide (numDim ≤ maxDim && recordStride ≤ maxStride)

/-- `bpann_check_append_row_limit`: require `newN < u32::MAX`. -/
def checkLimit (newN : Nat) : Bool :=
  decide (newN < rowLimit)

/-- `bpann_validate_index_backend`: succeed iff persisted backend is `none` or equals `expected`. -/
def validBackend (md : ObsMeta) (expected : String) : Bool :=
  match md.indexBackend with
  | some b => decide (b = expected)
  | none => true

/-- `bpann_remove_store`: reset an `MmapStore` to empty (0 rows). -/
def removeStore (s : MmapStore) : MmapStore :=
  { s with nrows := 0, capRows := 0, fileLen := 0, data := [] }

/-- `bpann_load_num_obs`: return `max(sidecar, meta)` when both exist, or whichever exists. -/
def loadObs (md : ObsMeta) : Option Nat :=
  match md.sidecarObs, md.numObs with
  | some a, some b => some (max a b)
  | some a, none => some a
  | none, some b => some b
  | none, none => none

/-- Model of `NumObsCounter` (`num_obs.bin` 8-byte mmap counter). -/
structure ObsCounter where
  count : Nat
  deriving Repr, DecidableEq

/-- `NumObsCounter::open`: open or create `num_obs.bin` with `initial` count. -/
def counterOpen (existing : Option Nat) (initial : Nat) : ObsCounter :=
  { count := existing.getD initial }

/-- `NumObsCounter::set`: write 64-bit little-endian observation count `n`. -/
def counterSet (c : ObsCounter) (n : Nat) : ObsCounter :=
  let _ := c
  { count := n }

/-- `write_indexed_rows`: write `indexed_rows.bin` 8-byte sidecar. -/
def writeIndexed (md : ObsMeta) (indexedRows : Nat) : ObsMeta :=
  { md with sidecarIndexed := some indexedRows }

/-- `bpann_load_indexed_rows`: prefer `sidecarIndexed` (`indexed_rows.bin`), falling back to
`metadata.json`. -/
def loadIndexed (md : ObsMeta) : Option Nat :=
  match md.sidecarIndexed with
  | some n => some n
  | none => md.indexedRows

/-- `bpann_load_index_backend`: load `index_backend` string field from `metadata.json`. -/
def loadBackend (md : ObsMeta) : Option String :=
  md.indexBackend

/-- `preserve_metadata_extension_fields`: preserve extension keys (`model_Config`, etc.). -/
def preserveExt (existing : ObsMeta) : List (String × String) :=
  existing.extFields

/-- `bpann_write_metadata`: write `metadata.json` and `indexed_rows.bin` while preserving
extension fields. -/
def writeMeta (existing : ObsMeta) (numObs numDim numMetrics : Nat) (scaleX : Bool)
    (indexedRows : Nat) (backend : String) : ObsMeta :=
  { formatVersion := formatVer
    numObs := some numObs
    sidecarObs := some numObs
    numDim := numDim
    numMetrics := numMetrics
    scaleX := scaleX
    indexBackend := some backend
    indexedRows := some indexedRows
    sidecarIndexed := some indexedRows
    extFields := preserveExt existing }

/-- `bpann_open_or_append_yvar`: open or create `train_yvar.bin` when `yvar` or file exists. -/
def openYvar (numMetrics : Nat) (knownRows : Option Nat)
    (existing : Option (List (List Word))) (yvar : Option Fitter.Matrix) :
    Option MmapStore :=
  match yvar with
  | some yv =>
    match mmapOpen numMetrics knownRows (existing.getD []) with
    | none => none
    | some st => if st.nrows = 0 then mmapAppend st yv else some st
  | none =>
    match existing with
    | some ex => mmapOpen numMetrics knownRows ex
    | none => none

/-- `bpann_append_yvar_on_add`: append `yvar` to existing or newly created `train_yvar.bin`. -/
def appendYvar (numMetrics : Nat) (knownRows : Option Nat)
    (slot : Option MmapStore) (yvar : Option Fitter.Matrix) : Option MmapStore :=
  match slot, yvar with
  | some st, some yv => mmapAppend st yv
  | none, some yv =>
    match mmapOpen numMetrics knownRows [] with
    | some st => mmapAppend st yv
    | none => none
  | some st, none => some st
  | none, none => none

/-- `bpann_train_rows_at`: gather `(x, y, yvar)` after checking all `indices < n`. -/
def obsGather (n : Nat) (tx ty : MmapStore) (tyv : Option MmapStore)
    (indices : List Nat) :
    Option (Fitter.Matrix × Fitter.Matrix × Option Fitter.Matrix) :=
  if indices.any (fun i ↦ decide (i ≥ n)) then
    none
  else
    match mmapGather tx indices, mmapGather ty indices with
    | some mx, some my =>
      let myv :=
        match tyv with
        | some s => mmapGather s indices
        | none => none
      some (mx, my, myv)
    | _, _ => none

/-- `parse_json_usize_field` (`bpann/src/observation.rs`): look up a numeric JSON field. -/
def parseUsize (fields : List (String × Nat)) (key : String) : Option Nat :=
  (fields.find? (fun kv ↦ decide (kv.1 = key))).map Prod.snd

/-- `bpann_parse_json_string_field`: look up a string JSON field. -/
def parseStr (fields : List (String × String)) (key : String) : Option String :=
  (fields.find? (fun kv ↦ decide (kv.1 = key))).map Prod.snd

/-- `disk_observation::set_index_stale`: set `index_stale` boolean latch to `true`. -/
def setStale (stale : Bool) : Bool :=
  let _ := stale
  true

/-- `disk_observation::parse_json_usize_field` (`ennbo/src/backend/disk_observation.rs`). -/
def diskUsize (fields : List (String × Nat)) (key : String) : Option Nat :=
  parseUsize fields key

/-! ## Section 2: `row_storage.rs` (6) and `storage_wire.rs` (1) -/

/-- Storage wire kind (`EnnStorage` in `ennbo/src/backend/mod.rs` and `storage_wire.rs`). -/
inductive StorageKind where
  | disk
  | memory
  deriving Repr, DecidableEq

/-- `EnnStorage::from_wire`: parse `"DISK"` or `"MEMORY"`, returning `none` otherwise. -/
def storageWire (wire : String) : Option StorageKind :=
  if wire = "DISK" then
    some .disk
  else if wire = "MEMORY" then
    some .memory
  else
    none

/-- Model of `RowStorage` (`ennbo/src/backend/row_storage.rs`). -/
structure RowStore where
  ncols : Nat
  nrows : Nat
  data : List (List Word)
  deriving Repr, DecidableEq

/-- `RowStorage::from_array2`: construct `RowStorage` from an `Array2<f64>` matrix. -/
def fromArray (m : Fitter.Matrix) : RowStore :=
  { ncols := m.cols, nrows := m.rows, data := m.data }

/-- `RowStorage::nrows`: return `self.nrows`. -/
def storeRows (s : RowStore) : Nat :=
  s.nrows

/-- `RowStorage::view`: return an `ArrayView2<f64>` matrix view of `RowStorage`. -/
def storeView (s : RowStore) : Fitter.Matrix :=
  ⟨s.nrows, s.ncols, s.data⟩

/-- `RowStorage::gather_rows`: gather rows at `indices` into a new `Matrix`. -/
def gatherRows (s : RowStore) (indices : List Nat) : Fitter.Matrix :=
  ⟨indices.length, s.ncols, indices.map (fun i ↦ s.data.getD i [])⟩

/-- `RowStorage::row_vec`: return row `i` as a `List Word`. -/
def rowVec (s : RowStore) (i : Nat) : List Word :=
  s.data.getD i []

/-- `RowStorage::push_rows`: append `rows` when `rows.rows > 0`. -/
def pushRows (s : RowStore) (rows : Fitter.Matrix) : RowStore :=
  if rows.rows = 0 then
    s
  else
    { s with
      nrows := s.nrows + rows.rows
      data := s.data ++ rows.data }

/-! ## Section 3: `in_memory.rs` (15 functions) -/

/-- Model of `InMemoryEnnBackend` (`ennbo/src/backend/in_memory.rs`). -/
structure MemBackend where
  trainX : RowStore
  trainY : RowStore
  trainYvar : Option RowStore
  driver : Neighbor.Driver
  index : Option Fitter.Matrix
  indexStale : Bool
  deriving Repr, DecidableEq

/-- `InMemoryEnnBackend::new`: construct an in-memory backend and build initial Flat index if
`trainX.rows > 0`. -/
def memNew (trainX trainY : Fitter.Matrix) (trainYvar : Option Fitter.Matrix)
    (driver : Neighbor.Driver) (scaleX : Bool) (xScale : List Word) :
    Option MemBackend :=
  if driver ≠ .flat then
    none
  else if trainX.cols = 0 || trainY.cols = 0 || trainX.rows ≠ trainY.rows then
    none
  else
    let idx :=
      if trainX.rows > 0 then
        some (scaleMatrix scaleX xScale trainX)
      else
        none
    some
      { trainX := fromArray trainX
        trainY := fromArray trainY
        trainYvar := trainYvar.map fromArray
        driver := driver
        index := idx
        indexStale := false }

/-- `InMemoryEnnBackend::len`: return `self.train_x.nrows()`. -/
def memLen (m : MemBackend) : Nat :=
  storeRows m.trainX

/-- `InMemoryEnnBackend::num_dim`: return `self.train_x.ncols`. -/
def memDim (m : MemBackend) : Nat :=
  m.trainX.ncols

/-- `InMemoryEnnBackend::num_metrics`: return `self.train_y.ncols`. -/
def memMetrics (m : MemBackend) : Nat :=
  m.trainY.ncols

/-- `InMemoryEnnBackend::driver`: return `self.driver`. -/
def memDriver (m : MemBackend) : Neighbor.Driver :=
  m.driver

/-- `InMemoryEnnBackend::train_x_view`: return `self.train_x.view()`. -/
def viewFeat (m : MemBackend) : Fitter.Matrix :=
  storeView m.trainX

/-- `InMemoryEnnBackend::train_y_view`: return `self.train_y.view()`. -/
def viewObj (m : MemBackend) : Fitter.Matrix :=
  storeView m.trainY

/-- `InMemoryEnnBackend::append_rows`: append `(x, y, yvar)` to `RowStorage`. -/
def memAppend (m : MemBackend) (x y : Fitter.Matrix)
    (yvar : Option Fitter.Matrix) : Option MemBackend :=
  if x.rows = 0 then
    some m
  else
    let yv1 :=
      match m.trainYvar, yvar with
      | some s, some v => some (pushRows s v)
      | none, some v => if m.trainX.nrows = 0 then some (fromArray v) else none
      | s, none => s
    some
      { m with
        trainX := pushRows m.trainX x
        trainY := pushRows m.trainY y
        trainYvar := yv1 }

/-- `InMemoryEnnBackend::mark_index_stale`: set `index_stale = true`. -/
def memStale (m : MemBackend) : MemBackend :=
  { m with indexStale := true }

/-- `InMemoryEnnBackend::ensure_index_sync`: rebuild or incrementally add rows to Flat index. -/
def memSync (m : MemBackend) (scaleX : Bool) (xScale : List Word) : MemBackend :=
  let total := memLen m
  if total = 0 then
    { m with index := none, indexStale := false }
  else
    let full := scaleMatrix scaleX xScale (viewFeat m)
    { m with index := some full, indexStale := false }

/-- `InMemoryEnnBackend::train_rows_at`: bounds-check `indices` and gather `(x, y, yvar)`. -/
def memGather (m : MemBackend) (indices : List Nat) :
    Option (Fitter.Matrix × Fitter.Matrix × Option Fitter.Matrix) :=
  let n := memLen m
  if indices.any (fun i ↦ decide (i ≥ n)) then
    none
  else
    some
      (gatherRows m.trainX indices,
       gatherRows m.trainY indices,
       m.trainYvar.map (fun s ↦ gatherRows s indices))

/-- `InMemoryEnnBackend::row_x`: bounds-check `i < len()` and return `train_x.row_vec(i)`. -/
def memFeat (m : MemBackend) (i : Nat) : Option (List Word) :=
  if i ≥ memLen m then none else some (rowVec m.trainX i)

/-- `InMemoryEnnBackend::row_y`: bounds-check `i < len()` and return `train_y.row_vec(i)`. -/
def memObj (m : MemBackend) (i : Nat) : Option (List Word) :=
  if i ≥ memLen m then none else some (rowVec m.trainY i)

/-- `InMemoryEnnBackend::row_yvar`: bounds-check `i < len()` and return `train_yvar` row. -/
def memVar (m : MemBackend) (i : Nat) : Option (Option (List Word)) :=
  if i ≥ memLen m then
    none
  else
    some (m.trainYvar.map (fun s ↦ rowVec s i))

/-- `InMemoryEnnBackend::search`: run Flat Faiss search over `m.index`. -/
def memSearch (m : MemBackend) (queries : Fitter.Matrix) (scaleX : Bool)
    (xScale : List Word) (searchK : Nat) (exclude : Bool) :
    Option (Batch × Labels) :=
  if m.indexStale then
    none
  else if queries.rows = 0 || searchK = 0 || memLen m = 0 then
    some (List.replicate queries.rows [], List.replicate queries.rows [])
  else
    let (rawD, rawL) := faissBatch scaleX xScale (memDim m) (viewFeat m) queries searchK
    let labels := flatSearch queries searchK exclude rawL rawD
    some (rawD, labels)

/-! ## Section 4: `ennbo/src/backend/mod.rs` (25 functions) -/

/-- Unified `EnnBackend` enum (`InMemory` vs `DiskBpann`). -/
inductive EnnBackend where
  | inMemory (b : MemBackend)
  | diskBpann (h : FlushHandle)
  deriving Repr, DecidableEq

/-- `EnnStorage::work_dir_from_env`: return `some path` for `.disk` when non-empty, else `none`. -/
def workEnv (s : StorageKind) (envVal : Option String) : Option String :=
  match s with
  | .memory => none
  | .disk =>
    match envVal with
    | some v => if v.isEmpty then none else some v
    | none => none

/-- `backend::disk_read`: acquire read guard on `DiskBackendHandle`. -/
def modRead (h : FlushHandle) : DiskBpann :=
  flushRead h

/-- `backend::disk_write`: apply a mutable update to `DiskBackendHandle.data`. -/
def modWrite (h : FlushHandle) (f : DiskBpann → DiskBpann) : FlushHandle :=
  { h with data := f h.data }

/-- `EnnBackend::new_in_memory`: construct `EnnBackend::InMemory`. -/
def newMem (trainX trainY : Fitter.Matrix) (trainYvar : Option Fitter.Matrix)
    (driver : Neighbor.Driver) (scaleX : Bool) (xScale : List Word) :
    Option EnnBackend :=
  (memNew trainX trainY trainYvar driver scaleX xScale).map .inMemory

/-- `EnnBackend::new_disk`: construct `EnnBackend::DiskBpann` when `driver == .bpann`. -/
def newDisk (reopenable : Bool) (existing : Option BpannState)
    (trainX trainY : Fitter.Matrix) (trainYvar : Option Fitter.Matrix)
    (driver : Neighbor.Driver) (scaleX : Bool) (xScale : List Word) (t : Tuning) :
    Option EnnBackend :=
  if driver ≠ .bpann then
    none
  else
    match diskNew reopenable existing trainX trainY trainYvar scaleX xScale t with
    | some d => some (.diskBpann (handleNew d))
    | none => none

/-- `EnnBackend::from_layout`: dispatch between `new_in_memory` and `new_disk` by `StorageKind`. -/
def fromLayout (storage : StorageKind) (hasWorkDir reopenable : Bool)
    (existing : Option BpannState) (trainX trainY : Fitter.Matrix)
    (trainYvar : Option Fitter.Matrix) (driver : Neighbor.Driver)
    (scaleX : Bool) (xScale : List Word) (t : Tuning) : Option EnnBackend :=
  match storage with
  | .memory =>
    if driver = .bpann then none
    else newMem trainX trainY trainYvar driver scaleX xScale
  | .disk =>
    if !hasWorkDir then none
    else newDisk reopenable existing trainX trainY trainYvar driver scaleX xScale t

/-- `EnnBackend::wait_for_flush`: no-op on `InMemory`, delegates to `waitFlush` on `DiskBpann`. -/
def ennWait (eb : EnnBackend) : Except EnnErr EnnBackend :=
  match eb with
  | .inMemory m => .ok (.inMemory m)
  | .diskBpann h =>
    match waitFlush h with
    | .ok h1 => .ok (.diskBpann h1)
    | .error e => .error e

/-- `EnnBackend::schedule_background_flush`: no-op on `InMemory`, delegates to `schedFlush` on
`DiskBpann`. -/
def ennSched (eb : EnnBackend) : Except EnnErr EnnBackend :=
  match eb with
  | .inMemory m => .ok (.inMemory m)
  | .diskBpann h =>
    match schedFlush h with
    | .ok h1 => .ok (.diskBpann h1)
    | .error e => .error e

/-- `EnnBackend::len`: observation count across `InMemory` or `DiskBpann`. -/
def ennLen (eb : EnnBackend) : Nat :=
  match eb with
  | .inMemory m => memLen m
  | .diskBpann h => diskLen (modRead h)

/-- `EnnBackend::num_dim`: feature dimension across `InMemory` or `DiskBpann`. -/
def ennDim (eb : EnnBackend) : Nat :=
  match eb with
  | .inMemory m => memDim m
  | .diskBpann h => diskDim (modRead h)

/-- `EnnBackend::num_metrics`: metric count across `InMemory` or `DiskBpann`. -/
def ennMetrics (eb : EnnBackend) : Nat :=
  match eb with
  | .inMemory m => memMetrics m
  | .diskBpann h => diskMetrics (modRead h)

/-- `EnnBackend::driver`: index driver across `InMemory` or `DiskBpann`. -/
def ennDriver (eb : EnnBackend) : Neighbor.Driver :=
  match eb with
  | .inMemory m => memDriver m
  | .diskBpann h => diskDriver (modRead h)

/-- `EnnBackend::mark_index_stale`: mark index stale across `InMemory` or `DiskBpann`. -/
def ennStale (eb : EnnBackend) : EnnBackend :=
  match eb with
  | .inMemory m => .inMemory (memStale m)
  | .diskBpann h => .diskBpann (modWrite h diskStale)

/-- `EnnBackend::append_rows`: append rows across `InMemory` or `DiskBpann`. -/
def ennAppend (eb : EnnBackend) (x y : Fitter.Matrix)
    (yvar : Option Fitter.Matrix) : Option EnnBackend :=
  match eb with
  | .inMemory m => (memAppend m x y yvar).map .inMemory
  | .diskBpann h =>
    match diskAppend h.data x y yvar with
    | some d1 => some (.diskBpann { h with data := d1 })
    | none => none

/-- `EnnBackend::defer_index_sync_for_search`: `false` for `InMemory`, `true` for `DiskBpann`. -/
def ennDefer (eb : EnnBackend) : Bool :=
  match eb with
  | .inMemory _ => false
  | .diskBpann h => deferSearch (modRead h)

/-- `EnnBackend::ensure_index_sync`: sync index across `InMemory` or `DiskBpann`. -/
def ennSync (eb : EnnBackend) (scaleX : Bool) (xScale : List Word) : Option EnnBackend :=
  match eb with
  | .inMemory m => some (.inMemory (memSync m scaleX xScale))
  | .diskBpann h =>
    match diskSync h.data scaleX xScale with
    | some d1 => some (.diskBpann { h with data := d1 })
    | none => none

/-- `EnnBackend::set_metric_scale`: mark stale on `InMemory`, or rescale/morph on `DiskBpann`. -/
def ennMetric (eb : EnnBackend) (xScale : List Word) (rebuild : Bool) : Option EnnBackend :=
  match eb with
  | .inMemory m => some (.inMemory (memStale m))
  | .diskBpann h =>
    match setMetric h.data xScale rebuild with
    | some d1 => some (.diskBpann { h with data := d1 })
    | none => none

/-- `EnnBackend::train_rows_at`: gather training rows across `InMemory` or `DiskBpann`. -/
def ennGather (eb : EnnBackend) (indices : List Nat) :
    Option (Fitter.Matrix × Fitter.Matrix × Option Fitter.Matrix) :=
  match eb with
  | .inMemory m => memGather m indices
  | .diskBpann h => diskGather (modRead h) indices

/-- `EnnBackend::row_x`: read feature row `i` across `InMemory` or `DiskBpann`. -/
def ennFeat (eb : EnnBackend) (i : Nat) : Option (List Word) :=
  match eb with
  | .inMemory m => memFeat m i
  | .diskBpann h => diskFeat (modRead h) i

/-- `EnnBackend::row_y`: read objective row `i` across `InMemory` or `DiskBpann`. -/
def ennObj (eb : EnnBackend) (i : Nat) : Option (List Word) :=
  match eb with
  | .inMemory m => memObj m i
  | .diskBpann h => diskObj (modRead h) i

/-- `EnnBackend::row_yvar`: read variance row `i` across `InMemory` or `DiskBpann`. -/
def ennVar (eb : EnnBackend) (i : Nat) : Option (Option (List Word)) :=
  match eb with
  | .inMemory m => memVar m i
  | .diskBpann h => diskVar (modRead h) i

/-- `EnnBackend::search`: search neighbors across `InMemory` or `DiskBpann`. -/
def ennSearch (eb : EnnBackend) (queries : Fitter.Matrix) (scaleX : Bool)
    (xScale : List Word) (searchK : Nat) (exclude : Bool) :
    Option (Batch × Labels) :=
  match eb with
  | .inMemory m => memSearch m queries scaleX xScale searchK exclude
  | .diskBpann h => diskSearch (modRead h) queries searchK exclude

/-- `EnnBackend::in_memory_train_x_view`: return `some train_x` for `InMemory`, `none` for Disk. -/
def ennMatx (eb : EnnBackend) : Option Fitter.Matrix :=
  match eb with
  | .inMemory m => some (viewFeat m)
  | .diskBpann _ => none

/-- `EnnBackend::in_memory_train_y_view`: return `some train_y` for `InMemory`, `none` for Disk. -/
def ennMaty (eb : EnnBackend) : Option Fitter.Matrix :=
  match eb with
  | .inMemory m => some (viewObj m)
  | .diskBpann _ => none

/-- `release_enn_observation_pages`: no-op for `InMemory`, releases mmap pages for `DiskBpann`. -/
def ennRelease (eb : EnnBackend) : EnnBackend :=
  match eb with
  | .inMemory m => .inMemory m
  | .diskBpann h => .diskBpann (modWrite h diskRelease)

/-! ## Theorems for `Storage` -/

/-- `bpann_load_num_obs` takes `max(sidecar, meta)` when both are present. -/
theorem obsMax (md : ObsMeta) (a b : Nat)
    (hs : md.sidecarObs = some a) (hm : md.numObs = some b) :
    loadObs md = some (max a b) := by
  simp [loadObs, hs, hm]

/-- `bpann_write_metadata` preserves extension fields while updating `numObs` and `indexedRows`. -/
theorem metaPreserve (existing : ObsMeta) (numObs numDim numMetrics : Nat)
    (scaleX : Bool) (indexedRows : Nat) (backend : String) :
    let m1 := writeMeta existing numObs numDim numMetrics scaleX indexedRows backend
    m1.extFields = existing.extFields ∧
    loadObs m1 = some numObs ∧
    loadIndexed m1 = some indexedRows ∧
    loadBackend m1 = some backend := by
  simp [writeMeta, preserveExt, loadObs, loadIndexed, loadBackend]

/-- `EnnStorage::from_wire` recognizes `"DISK"` and `"MEMORY"` and rejects other strings. -/
theorem wireSpec :
    storageWire "DISK" = some .disk ∧
    storageWire "MEMORY" = some .memory ∧
    storageWire "OTHER" = none := by
  decide

/-- `RowStorage::push_rows` increases `nrows` by `rows.rows`. -/
theorem pushCount (s : RowStore) (rows : Fitter.Matrix) :
    (pushRows s rows).nrows = s.nrows + rows.rows := by
  by_cases h : rows.rows = 0 <;> simp [pushRows, h]

/-- `EnnBackend::from_layout` rejects `.bpann` on `.memory` and `.flat` on `.disk`. -/
theorem layoutDriver (hasWorkDir reopenable : Bool) (existing : Option BpannState)
    (tx ty : Fitter.Matrix) (tyv : Option Fitter.Matrix)
    (scaleX : Bool) (xScale : List Word) (t : Tuning) :
    fromLayout .memory hasWorkDir reopenable existing tx ty tyv .bpann scaleX xScale t = none ∧
    fromLayout .disk hasWorkDir reopenable existing tx ty tyv .flat scaleX xScale t = none := by
  constructor
  · simp [fromLayout]
  · by_cases hw : hasWorkDir <;> simp [fromLayout, hw, newDisk]

/-- `EnnBackend::defer_index_sync_for_search` is `false` for `InMemory` and `true` for `DiskBpann`. -/
theorem deferSplit (m : MemBackend) (h : FlushHandle) :
    ennDefer (.inMemory m) = false ∧
    ennDefer (.diskBpann h) = true := by
  simp [ennDefer, deferSearch]

/-- `EnnBackend::in_memory_train_x_view` and `in_memory_train_y_view` return `some` iff `InMemory`. -/
theorem viewSplit (m : MemBackend) (h : FlushHandle) :
    (ennMatx (.inMemory m)).isSome = true ∧
    (ennMaty (.inMemory m)).isSome = true ∧
    ennMatx (.diskBpann h) = none ∧
    ennMaty (.diskBpann h) = none := by
  simp [ennMatx, ennMaty]

/-! ## Manifest Function Coverage Table (66 functions) -/

/-- All 66 `coverage_manifest.json` functions formalized in `Storage`. -/
def manifestSpecs : List FuncSpec :=
  [ { file := "rust/crates/bpann/src/observation.rs", line := 18,
      fnName := "bpann_validate_dim_limits",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_validate_dim_limits()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 33,
      fnName := "bpann_check_append_row_limit",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_check_append_row_limit()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 42,
      fnName := "bpann_validate_index_backend",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_validate_index_backend()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 67,
      fnName := "bpann_remove_store",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_remove_store()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 83,
      fnName := "bpann_load_num_obs",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_load_num_obs()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 118,
      fnName := "open",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/impl#[NumObsCounter]open()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 144,
      fnName := "set",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/impl#[NumObsCounter]set()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 149,
      fnName := "write_indexed_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/write_indexed_rows()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 157,
      fnName := "bpann_load_indexed_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_load_indexed_rows()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 168,
      fnName := "bpann_load_index_backend",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_load_index_backend()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 173,
      fnName := "bpann_write_metadata",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_write_metadata()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 199,
      fnName := "preserve_metadata_extension_fields",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/preserve_metadata_extension_fields()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 302,
      fnName := "bpann_open_or_append_yvar",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_open_or_append_yvar()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 325,
      fnName := "bpann_append_yvar_on_add",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_append_yvar_on_add()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 346,
      fnName := "bpann_train_rows_at",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_train_rows_at()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 372,
      fnName := "parse_json_usize_field",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/parse_json_usize_field()." },
    { file := "rust/crates/bpann/src/observation.rs", line := 382,
      fnName := "bpann_parse_json_string_field",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 observation/bpann_parse_json_string_field()." },
    { file := "rust/crates/ennbo/src/backend/disk_observation.rs", line := 27,
      fnName := "set_index_stale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/disk_observation/set_index_stale()." },
    { file := "rust/crates/ennbo/src/backend/disk_observation.rs", line := 283,
      fnName := "parse_json_usize_field",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/disk_observation/parse_json_usize_field()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 24,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]new()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 70,
      fnName := "len",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]len()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 79,
      fnName := "num_dim",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]num_dim()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 83,
      fnName := "num_metrics",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]num_metrics()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 87,
      fnName := "driver",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]driver()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 95,
      fnName := "train_x_view",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]train_x_view()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 99,
      fnName := "train_y_view",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]train_y_view()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 107,
      fnName := "append_rows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]append_rows()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 128,
      fnName := "mark_index_stale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]mark_index_stale()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 136,
      fnName := "ensure_index_sync",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]ensure_index_sync()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 194,
      fnName := "train_rows_at",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]train_rows_at()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 215,
      fnName := "row_x",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]row_x()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 225,
      fnName := "row_y",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]row_y()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 235,
      fnName := "row_yvar",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]row_yvar()." },
    { file := "rust/crates/ennbo/src/backend/in_memory.rs", line := 248,
      fnName := "search",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/in_memory/impl#[InMemoryEnnBackend]search()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 39,
      fnName := "work_dir_from_env",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnStorage]work_dir_from_env()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 49,
      fnName := "disk_read",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/disk_read()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 56,
      fnName := "disk_write",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/disk_write()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 71,
      fnName := "new_in_memory",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]new_in_memory()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 84,
      fnName := "new_disk",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]new_disk()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 130,
      fnName := "from_layout",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]from_layout()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 165,
      fnName := "wait_for_flush",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]wait_for_flush()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 172,
      fnName := "schedule_background_flush",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]schedule_background_flush()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 179,
      fnName := "len",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]len()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 190,
      fnName := "num_dim",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]num_dim()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 197,
      fnName := "num_metrics",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]num_metrics()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 204,
      fnName := "driver",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]driver()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 213,
      fnName := "mark_index_stale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]mark_index_stale()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 233,
      fnName := "append_rows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]append_rows()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 245,
      fnName := "defer_index_sync_for_search",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]defer_index_sync_for_search()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 254,
      fnName := "ensure_index_sync",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]ensure_index_sync()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 266,
      fnName := "set_metric_scale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]set_metric_scale()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 276,
      fnName := "train_rows_at",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]train_rows_at()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 286,
      fnName := "row_x",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]row_x()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 293,
      fnName := "row_y",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]row_y()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 300,
      fnName := "row_yvar",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]row_yvar()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 307,
      fnName := "search",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]search()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 340,
      fnName := "in_memory_train_x_view",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]in_memory_train_x_view()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 347,
      fnName := "in_memory_train_y_view",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/impl#[EnnBackend]in_memory_train_y_view()." },
    { file := "rust/crates/ennbo/src/backend/mod.rs", line := 356,
      fnName := "release_enn_observation_pages",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/release_enn_observation_pages()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 15,
      fnName := "from_array2",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]from_array2()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 25,
      fnName := "nrows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]nrows()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 29,
      fnName := "view",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]view()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 34,
      fnName := "gather_rows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]gather_rows()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 42,
      fnName := "row_vec",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]row_vec()." },
    { file := "rust/crates/ennbo/src/backend/row_storage.rs", line := 46,
      fnName := "push_rows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/row_storage/impl#[RowStorage]push_rows()." },
    { file := "rust/crates/ennbo/src/backend/storage_wire.rs", line := 15,
      fnName := "from_wire",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/storage_wire/impl#[EnnStorage]from_wire()." } ]

/-- All 66 functions across observation, row storage, in-memory, and unified backend are covered. -/
theorem storageCount : manifestSpecs.length = 66 := rfl

#print axioms obsMax
#print axioms metaPreserve
#print axioms wireSpec
#print axioms pushCount
#print axioms layoutDriver
#print axioms deferSplit
#print axioms viewSplit
#print axioms storageCount

end BauxResearch.Semantics.Storage
