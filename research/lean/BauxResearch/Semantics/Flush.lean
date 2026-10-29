import BauxResearch.Semantics.Bpann

/-!
# ENN 0.4.1 disk BPANN wrapper and background flush controller semantics

This module formalizes all 30 functions in:
- `rust/crates/ennbo/src/disk_bpann/enn_backend.rs` (24 functions)
- `rust/crates/ennbo/src/backend/flush_controller.rs` (6 functions)

It models `DiskBpannEnnBackend`, `DiskBackendHandle`, `wait_for_flush`, and
`schedule_background_flush` (including the `last_error` Sticky-Error Latch, the
`pending >= hard_threshold` synchronous backpressure wait + build + publish path, the
`job.is_some()` in-flight coalescing guard, and the `pending >= soft_threshold` background
detached soft-sync spawn).
-/

namespace BauxResearch.Semantics.Flush

open Sampling Bounds Fitter Neighbor Append Backend
open Residency Morph Tree Bpann

/-! ## Section 1: `ennbo/src/disk_bpann/enn_backend.rs` (24 functions) -/

/-- Error category mapped by `bpann_err` (`BpannError -> ENNError`). -/
inductive EnnErr where
  | invalidShape (expected got : String)
  | invalidParam (msg : String)
  deriving Repr, DecidableEq

/-- `bpann_err`: map `BpannError` variants one-to-one to `ENNError`. -/
def bpannErr (e : EnnErr) : EnnErr := e

/-- `apply_config_flush_threshold`: apply `current_tuning()` soft and hard flush thresholds. -/
def applyThresh (b : BpannState) (t : Tuning) : BpannState :=
  withHard (withSoft b t.flushSoft) t.flushHard

/-- Model of `DiskBpannEnnBackend` (`ennbo/src/disk_bpann/enn_backend.rs`). -/
structure DiskBpann where
  inner : BpannState
  driver : Neighbor.Driver
  numMetrics : Nat
  deriving Repr, DecidableEq

/-- `DiskBpannEnnBackend::new`: reopen when `trainX.rows == 0 && trainY.rows == 0 && exists`,
otherwise construct a fresh `BpannBackend`, and apply config flush thresholds. -/
def diskNew (reopenable : Bool) (existing : Option BpannState)
    (trainX trainY : Fitter.Matrix) (trainYvar : Option Fitter.Matrix)
    (scaleX : Bool) (xScale : List Word) (t : Tuning) : Option DiskBpann :=
  let baseOpt :=
    if trainX.rows = 0 && trainY.rows = 0 && reopenable then
      match existing with
      | some ex => some (discardReopen (bpannReopen ex (some ex.index.tree)))
      | none => none
    else
      bpannNew trainX trainY trainYvar scaleX xScale
  match baseOpt with
  | none => none
  | some b0 =>
    let b1 := applyThresh b0 t
    some
      { inner := b1
        driver := .bpann
        numMetrics := b1.numMetrics }

/-- `DiskBpannEnnBackend::driver`: return `IndexDriver::BpAnnDisk` (`.bpann`). -/
def diskDriver (d : DiskBpann) : Neighbor.Driver :=
  d.driver

/-- `DiskBpannEnnBackend::len`: return `inner.len()`. -/
def diskLen (d : DiskBpann) : Nat :=
  bpannLen d.inner

/-- `DiskBpannEnnBackend::num_dim`: return `inner.num_dim()`. -/
def diskDim (d : DiskBpann) : Nat :=
  bpannDim d.inner

/-- `DiskBpannEnnBackend::num_metrics`: return `self.num_metrics`. -/
def diskMetrics (d : DiskBpann) : Nat :=
  d.numMetrics

/-- `DiskBpannEnnBackend::defer_index_sync_for_search`: always `true` for disk BPANN. -/
def deferSearch (d : DiskBpann) : Bool :=
  let _ := d
  true

/-- `DiskBpannEnnBackend::mark_index_stale`: delegate to `inner.mark_index_stale()`. -/
def diskStale (d : DiskBpann) : DiskBpann :=
  { d with inner := markStale d.inner }

/-- `DiskBpannEnnBackend::append_rows`: delegate to `inner.append_rows(x, y, yvar)`. -/
def diskAppend (d : DiskBpann) (x y : Fitter.Matrix)
    (yvar : Option Fitter.Matrix) : Option DiskBpann :=
  match bpannAppend d.inner x y yvar with
  | some b1 => some { d with inner := b1 }
  | none => none

/-- `DiskBpannEnnBackend::ensure_index_sync`: delegate to `inner.ensure_index_sync_with_scale`. -/
def diskSync (d : DiskBpann) (scaleX : Bool) (xScale : List Word) : Option DiskBpann :=
  match syncScale d.inner scaleX xScale with
  | some b1 => some { d with inner := b1 }
  | none => none

/-- `DiskBpannEnnBackend::set_metric_scale`: call `rebuild_metric` if `rebuild` else
`rescale_metric`. -/
def setMetric (d : DiskBpann) (xScale : List Word) (rebuild : Bool) : Option DiskBpann :=
  let res :=
    if rebuild then
      rebuildMetric d.inner xScale
    else
      Bpann.rescaleMetric d.inner xScale
  match res with
  | some b1 => some { d with inner := b1 }
  | none => none

/-- `DiskBpannEnnBackend::release_observation_pages`: delegate to
`inner.release_observation_pages()`. -/
def diskRelease (d : DiskBpann) : DiskBpann :=
  { d with inner := releaseObs d.inner }

/-- `DiskBpannEnnBackend::train_rows_at`: delegate to `inner.train_rows_at(indices)`. -/
def diskGather (d : DiskBpann) (indices : List Nat) :
    Option (Fitter.Matrix × Fitter.Matrix × Option Fitter.Matrix) :=
  bpannGather d.inner indices

/-- `DiskBpannEnnBackend::row_x`: read observation row `i` from `train_x`. -/
def diskFeat (d : DiskBpann) (i : Nat) : Option (List Word) :=
  mmapSlice d.inner i

/-- `DiskBpannEnnBackend::row_y`: read objective row `i` from `train_y`. -/
def diskObj (d : DiskBpann) (i : Nat) : Option (List Word) :=
  (mmapRow d.inner i).map Prod.fst

/-- `DiskBpannEnnBackend::row_yvar`: read observation noise row `i` from `train_yvar`. -/
def diskVar (d : DiskBpann) (i : Nat) : Option (Option (List Word)) :=
  (mmapRow d.inner i).map Prod.snd

/-- `DiskBpannEnnBackend::search`: delegate to `inner.search(x, search_k as usize, exclude)`. -/
def diskSearch (d : DiskBpann) (queries : Fitter.Matrix) (searchK : Nat)
    (exclude : Bool) : Option (List (List Word) × List (List Label)) :=
  searchExec d.inner queries searchK exclude

/-- `DiskBpannEnnBackend::pending_unindexed_count`: return `inner.pending_rows()`. -/
def pendingCount (d : DiskBpann) : Nat :=
  pendingRows d.inner

/-- `DiskBpannEnnBackend::pending_flush_threshold`: return `inner.pending_flush_threshold()`. -/
def softThresh (d : DiskBpann) : Nat :=
  getSoft d.inner

/-- `DiskBpannEnnBackend::pending_hard_flush_threshold`: return
`inner.pending_hard_flush_threshold()`. -/
def hardThresh (d : DiskBpann) : Nat :=
  getHard d.inner

/-- `DiskBpannEnnBackend::defer_append_indexing_for_flush`: return
`inner.defer_append_indexing()`. -/
def deferFlush (d : DiskBpann) : Bool :=
  deferAppend d.inner

/-- `DiskBpannEnnBackend::soft_sync_build_detached`: build detached incremental index. -/
def buildDetached (d : DiskBpann) : Option IncIndex :=
  softBuild d.inner

/-- `DiskBpannEnnBackend::soft_sync_publish_detached`: publish detached incremental index. -/
def pubDetached (d : DiskBpann) (built : IncIndex) : DiskBpann :=
  { d with inner := softPublish d.inner built }

/-! ## Section 2: `ennbo/src/backend/flush_controller.rs` (6 functions) -/

/-- Model of `DiskBackendHandle` (`ennbo/src/backend/flush_controller.rs`). -/
structure FlushHandle where
  data : DiskBpann
  inFlight : Option IncIndex
  lastError : Option EnnErr
  testGate : Bool
  deriving Repr, DecidableEq

/-- `flush_controller::disk_read`: acquire read guard on `DiskBpannEnnBackend`. -/
def flushRead (h : FlushHandle) : DiskBpann :=
  h.data

/-- `maybe_wait_test_flush_gate`: test-only synchronization gate before background flush work. -/
def waitGate (h : FlushHandle) : FlushHandle :=
  { h with testGate := false }

/-- `DiskBackendHandle::new`: wrap `DiskBpannEnnBackend` with no in-flight job and no error. -/
def handleNew (d : DiskBpann) : FlushHandle :=
  { data := d
    inFlight := none
    lastError := none
    testGate := false }

/-- `DiskBackendHandle::data`: return the underlying `DiskBpannEnnBackend` handle. -/
def handleData (h : FlushHandle) : DiskBpann :=
  h.data

/-- `DiskBackendHandle::wait_for_flush`: join any in-flight flush job and surface `last_error`. -/
def waitFlush (h : FlushHandle) : Except EnnErr FlushHandle :=
  let h1 :=
    match h.inFlight with
    | some built => { h with data := pubDetached h.data built, inFlight := none }
    | none => h
  match h1.lastError with
  | some err => .error err
  | none => .ok h1

/-- `DiskBackendHandle::schedule_background_flush`:
1. If `last_error` is set, return `Err(err)`.
2. Read `(defer, pending, soft, hard)`.
3. If `defer && pending >= hard`, run `wait_for_flush()`, then synchronously build + publish.
4. If `inFlight.isSome`, return `Ok(h)` (coalesce with active background flush).
5. If `defer && pending >= soft`, spawn background detached soft-sync build. -/
def schedFlush (h : FlushHandle) : Except EnnErr FlushHandle :=
  match h.lastError with
  | some err => .error err
  | none =>
    let d := flushRead h
    let defer := deferFlush d
    let pending := pendingCount d
    let soft := softThresh d
    let hard := hardThresh d
    if defer && pending ≥ hard then
      match waitFlush h with
      | .error err => .error err
      | .ok h1 =>
        let h2 := waitGate h1
        match buildDetached h2.data with
        | some built => .ok { h2 with data := pubDetached h2.data built }
        | none => .ok h2
    else if h.inFlight.isSome then
      .ok h
    else if defer && pending ≥ soft then
      let h1 := waitGate h
      .ok { h1 with inFlight := buildDetached h1.data }
    else
      .ok h

/-! ## Theorems for `Flush` -/

/-- `DiskBpannEnnBackend::defer_index_sync_for_search` is always `true`. -/
theorem deferAlways (d : DiskBpann) :
    deferSearch d = true := rfl

/-- `DiskBackendHandle::new` starts with no in-flight job and no sticky error. -/
theorem handleInit (d : DiskBpann) :
    (handleNew d).inFlight = none ∧ (handleNew d).lastError = none := by
  simp [handleNew]

/-- `wait_for_flush` surfaces `lastError` whenever a sticky flush error is present. -/
theorem waitErr (h : FlushHandle) (err : EnnErr) (herr : h.lastError = some err) :
    waitFlush h = .error err := by
  rcases h with ⟨data, inFlight, lastError, testGate⟩
  cases inFlight <;> simp [waitFlush] at herr ⊢ <;> simp [herr]

/-- `wait_for_flush` clears `inFlight` and publishes the built index when `lastError = none`. -/
theorem waitPub (d : DiskBpann) (built : IncIndex) :
    waitFlush { data := d, inFlight := some built, lastError := none, testGate := false } =
      .ok { data := pubDetached d built, inFlight := none, lastError := none, testGate := false } :=
  rfl

/-- `schedule_background_flush` surfaces `lastError` immediately when present. -/
theorem schedErr (h : FlushHandle) (err : EnnErr) (herr : h.lastError = some err) :
    schedFlush h = .error err := by
  simp [schedFlush, herr]

/-- `schedule_background_flush` is a no-op when `pending < soft` and `pending < hard`. -/
theorem schedBelow (h : FlushHandle)
    (herr : h.lastError = none)
    (hjob : h.inFlight = none)
    (hsoft : pendingCount h.data < softThresh h.data)
    (hhard : pendingCount h.data < hardThresh h.data) :
    schedFlush h = .ok h := by
  have hnotHard : ¬(pendingCount h.data ≥ hardThresh h.data) := Nat.not_le.mpr hhard
  have hnotSoft : ¬(pendingCount h.data ≥ softThresh h.data) := Nat.not_le.mpr hsoft
  simp [schedFlush, herr, flushRead, hnotHard, hjob, hnotSoft]

/-- `schedule_background_flush` synchronously drains all pending rows when `defer = true`
and `pending >= hard`. -/
theorem schedHard (d : DiskBpann)
    (hdefer : deferFlush d = true)
    (hhard : pendingCount d ≥ hardThresh d)
    (hpos : hardThresh d ≥ 1) :
    ∃ h', schedFlush (handleNew d) = .ok h' ∧ pendingCount h'.data = 0 := by
  have hpendPos : pendingCount d ≥ 1 := Nat.le_trans hpos hhard
  have hlt : d.inner.index.indexedRows < d.inner.trainX.nrows := by
    unfold pendingCount pendingRows at hpendPos
    omega
  have hle : d.inner.index.indexedRows ≤ d.inner.trainX.nrows := Nat.le_of_lt hlt
  have hnotge : ¬(d.inner.index.indexedRows ≥ d.inner.trainX.nrows) := Nat.not_le.mpr hlt
  let rows := (List.range d.inner.trainX.nrows).map (fun i ↦ d.inner.trainX.data.getD i [])
  let built := ensureSync d.inner.index
    (scaleMatrix d.inner.scaleX d.inner.xScale
      ⟨d.inner.trainX.nrows, d.inner.numDim, rows⟩).data
  let h' : FlushHandle :=
    { data := pubDetached d built, inFlight := none, lastError := none, testGate := false }
  refine ⟨h', ?_, ?_⟩
  · dsimp only [h', built, rows]
    simp [schedFlush, handleNew, flushRead, hdefer, hhard, waitFlush, waitGate,
          buildDetached, softBuild, idxClone, hnotge]
  · dsimp only [h', built, rows]
    simp [pubDetached, softPublish, hle, pendingCount, pendingRows,
          syncTotal, scaleMatrix]

/-! ## Manifest Function Coverage Table (30 functions) -/

/-- All 30 `coverage_manifest.json` functions formalized in `Flush`. -/
def manifestSpecs : List FuncSpec :=
  [ { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 14,
      fnName := "disk_read",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/disk_read()." },
    { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 45,
      fnName := "maybe_wait_test_flush_gate",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/maybe_wait_test_flush_gate()." },
    { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 58,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/impl#[DiskBackendHandle]new()." },
    { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 66,
      fnName := "data",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/impl#[DiskBackendHandle]data()." },
    { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 70,
      fnName := "wait_for_flush",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/impl#[DiskBackendHandle]wait_for_flush()." },
    { file := "rust/crates/ennbo/src/backend/flush_controller.rs", line := 111,
      fnName := "schedule_background_flush",
      fnId := "rust-analyzer cargo ennbo 0.4.1 backend/flush_controller/impl#[DiskBackendHandle]schedule_background_flush()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 13,
      fnName := "bpann_err",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/bpann_err()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 20,
      fnName := "apply_config_flush_threshold",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/apply_config_flush_threshold()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 34,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]new()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 75,
      fnName := "driver",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]driver()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 80,
      fnName := "len",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]len()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 84,
      fnName := "num_dim",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]num_dim()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 88,
      fnName := "num_metrics",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]num_metrics()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 92,
      fnName := "defer_index_sync_for_search",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]defer_index_sync_for_search()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 100,
      fnName := "mark_index_stale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]mark_index_stale()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 108,
      fnName := "append_rows",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]append_rows()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 117,
      fnName := "ensure_index_sync",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]ensure_index_sync()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 129,
      fnName := "set_metric_scale",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]set_metric_scale()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 138,
      fnName := "release_observation_pages",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]release_observation_pages()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 142,
      fnName := "train_rows_at",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]train_rows_at()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 146,
      fnName := "row_x",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]row_x()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 152,
      fnName := "row_y",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]row_y()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 159,
      fnName := "row_yvar",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]row_yvar()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 164,
      fnName := "search",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]search()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 232,
      fnName := "pending_unindexed_count",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]pending_unindexed_count()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 236,
      fnName := "pending_flush_threshold",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]pending_flush_threshold()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 240,
      fnName := "pending_hard_flush_threshold",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]pending_hard_flush_threshold()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 252,
      fnName := "defer_append_indexing_for_flush",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]defer_append_indexing_for_flush()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 257,
      fnName := "soft_sync_build_detached",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]soft_sync_build_detached()." },
    { file := "rust/crates/ennbo/src/disk_bpann/enn_backend.rs", line := 264,
      fnName := "soft_sync_publish_detached",
      fnId := "rust-analyzer cargo ennbo 0.4.1 disk_bpann/enn_backend/impl#[DiskBpannEnnBackend]soft_sync_publish_detached()." } ]

/-- All 30 functions in `flush_controller.rs` and `disk_bpann/enn_backend.rs` are covered. -/
theorem flushCount : manifestSpecs.length = 30 := rfl

#print axioms deferAlways
#print axioms handleInit
#print axioms waitErr
#print axioms waitPub
#print axioms schedErr
#print axioms schedBelow
#print axioms schedHard
#print axioms flushCount

end BauxResearch.Semantics.Flush
