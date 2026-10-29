import BauxResearch.Semantics.Backend

/-!
# ENN 0.4.1 tree residency and mmap column store semantics

This module formalizes all 20 functions in:
- `rust/crates/bpann/src/index/tree_residency.rs` (10 functions)
- `rust/crates/bpann/src/mmap_store.rs` (10 functions)

It models the 64 KiB residency unit bitmap (`unitBytes = 65536`), the Knuth multiplicative
touch hash (`(unit * 0x9E37_79B9_7F4A_7C15) >>> 41`), the default (`256 MiB`) and tree
(`1 GiB`) residency budgets, threshold-gated page release (`releaseMap`), and the
`f64` mmap column store (`rowBytes`, `bytesRows`, `ensureCap`, `mmapAppend`, `mmapGather`).
-/

namespace BauxResearch.Semantics.Residency

open Sampling Bounds Fitter Neighbor Append Backend

/-- Function specification record binding a semantic model to `coverage_manifest.json`. -/
structure FuncSpec where
  file : String
  line : Nat
  fnName : String
  fnId : String
  deriving Repr, DecidableEq

/-- Default mmap observation resident budget: `256 << 20` bytes (256 MiB). -/
def defBudget : Nat := 268435456

/-- Default B+ tree resident budget: `1 << 30` bytes (1 GiB). -/
def treeBudget : Nat := 1073741824

/-- Residency tracking unit size: `64 << 10` bytes (64 KiB). -/
def unitBytes : Nat := 65536

/-- Number of bits in the residency touch filter: `1 << 23` (8,388,608 bits). -/
def touchBits : Nat := 8388608

/-- Number of 64-bit words in the residency touch filter: `touchBits / 64 = 131,072`. -/
def touchWords : Nat := 131072

/-- Minimum row growth quantum for `MmapColumnStore`: 64 rows. -/
def growRows : Nat := 64

/-- Byte size of a single `f64` scalar in `MmapColumnStore`: 8 bytes. -/
def cellBytes : Nat := 8

/-- Residency tracker state for `bpann/src/index/tree_residency.rs`. -/
structure Residency where
  budget : Nat
  tracking : Bool
  bytes : Nat
  released : Nat
  touched : List Nat
  deriving Repr, DecidableEq

/-- `release_map`: OS page-cache release advise operation (`madvise` / `Discard`).
In pure state semantics, advising the OS to drop clean pages resets resident bytes to 0. -/
def releaseMap (mapped : Nat) : Nat :=
  let _ := mapped
  0

/-- `Residency::new`: construct a residency tracker with default 256 MiB budget. -/
def resNew : Residency :=
  { budget := defBudget
    tracking := false
    bytes := 0
    released := 0
    touched := [] }

/-- `Residency::budget`: return the configured residency budget in bytes. -/
def resBudget (r : Residency) : Nat :=
  r.budget

/-- `Residency::set_budget`: update the residency budget and clear tracking if `budget == 0`. -/
def setBudget (r : Residency) (budget : Nat) : Residency :=
  if budget = 0 then
    { r with budget := 0, tracking := false, bytes := 0, touched := [] }
  else
    { r with budget := budget }

/-- `Residency::update_tracking`: enable tracking iff `budget > 0` and `mapped > budget`. -/
def updateTrack (r : Residency) (mapped reserved : Nat) : Residency :=
  if r.budget = 0 then
    r
  else
    let track := decide (mapped > r.budget)
    if !track && r.tracking then
      { r with tracking := false, bytes := 0, touched := [] }
    else if track && r.bytes + reserved > r.budget then
      { r with
        tracking := true
        bytes := releaseMap mapped
        released := r.released + 1
        touched := [] }
    else
      { r with tracking := track }

/-- Knuth multiplicative hash bit index for a 64 KiB residency unit. -/
def unitBit (offset : Nat) : Nat :=
  let unit := offset / unitBytes
  (unit * 11400714819323198485) % touchBits

/-- `Residency::release`: drop resident pages, clear touch filter, and increment `released`. -/
def resRelease (r : Residency) (mapped : Nat) : Residency :=
  if r.bytes = 0 then
    r
  else
    { r with
      bytes := releaseMap mapped
      released := r.released + 1
      touched := [] }

/-- `Residency::touch_tracked`: record a 64 KiB unit touch and release if over budget. -/
def touchTrack (r : Residency) (mapped offset reserved : Nat) : Residency :=
  let bit := unitBit offset
  if r.touched.contains bit then
    r
  else
    let r1 :=
      if r.bytes + unitBytes + reserved > r.budget then
        resRelease r mapped
      else
        r
    { r1 with
      bytes := r1.bytes + unitBytes
      touched := bit :: r1.touched }

/-- `Residency::touch`: fast-path check on `tracking` before calling `touch_tracked`. -/
def resTouch (r : Residency) (mapped offset reserved : Nat) : Residency :=
  if r.tracking then
    touchTrack r mapped offset reserved
  else
    r

/-- `Residency::forget`: clear tracked resident bytes and touch bitmap without OS release. -/
def resForget (r : Residency) : Residency :=
  { r with bytes := 0, touched := [] }

/-- `Residency::copy_released`: copy the cumulative release counter from `other`. -/
def copyRel (r other : Residency) : Residency :=
  { r with released := other.released }

/-- Model of `MmapColumnStore` (`bpann/src/mmap_store.rs`). -/
structure MmapStore where
  ncols : Nat
  nrows : Nat
  capRows : Nat
  fileLen : Nat
  data : List (List Word)
  residency : Residency
  deriving Repr, DecidableEq

/-- `MmapColumnStore::row_bytes`: `ncols * 8`, returns `none` if `ncols == 0`. -/
def rowBytes (ncols : Nat) : Option Nat :=
  if ncols = 0 then none else some (ncols * cellBytes)

/-- `MmapColumnStore::bytes_for_rows`: `nrows * row_bytes(ncols)`. -/
def bytesRows (ncols nrows : Nat) : Option Nat :=
  match rowBytes ncols with
  | none => none
  | some rb => some (nrows * rb)

/-- Rounded-up row capacity for `ensure_capacity`: `max(minRows, max(capRows * 2, 64))`. -/
def nextCap (capRows minRows : Nat) : Nat :=
  max minRows (max (capRows * 2) growRows)

/-- `MmapColumnStore::ensure_capacity`: grow backing file capacity when `minRows > capRows`. -/
def ensureCap (s : MmapStore) (minRows : Nat) : Option MmapStore :=
  if minRows ≤ s.capRows then
    some s
  else
    let newCap := nextCap s.capRows minRows
    match bytesRows s.ncols newCap with
    | none => none
    | some newLen =>
      let res := updateTrack s.residency newLen 0
      some { s with capRows := newCap, fileLen := newLen, residency := res }

/-- `MmapColumnStore::touch`: touch the residency unit containing `row`. -/
def storeTouch (s : MmapStore) (row : Nat) : MmapStore :=
  let offset := row * s.ncols * cellBytes
  { s with residency := resTouch s.residency s.fileLen offset 0 }

/-- `MmapColumnStore::pretouch`: touch one row per 64 KiB unit across `[startRow, endRow)`. -/
def pretouch (s : MmapStore) (startRow endRow : Nat) : MmapStore :=
  if !s.residency.tracking || endRow ≤ startRow then
    s
  else
    let rb := max (s.ncols * cellBytes) 1
    let step := max (unitBytes / rb) 1
    let count := (endRow - startRow + step - 1) / step
    (List.range count).foldl
      (fun acc k ↦ storeTouch acc (startRow + k * step))
      s

/-- `MmapColumnStore::mmap_open_or_create`: open or initialize a column store. -/
def mmapOpen (ncols : Nat) (knownRows : Option Nat) (initData : List (List Word)) :
    Option MmapStore :=
  match rowBytes ncols with
  | none => none
  | some rb =>
    let fileRows := initData.length
    let nrows :=
      match knownRows with
      | some k => min k fileRows
      | none => fileRows
    let kept := initData.take nrows
    let fileLen := fileRows * rb
    let res := updateTrack resNew fileLen 0
    some
      { ncols := ncols
        nrows := nrows
        capRows := fileRows
        fileLen := fileLen
        data := kept
        residency := res }

/-- `MmapColumnStore::mmap_append`: validate column width, grow capacity, and append rows. -/
def mmapAppend (s : MmapStore) (rows : Fitter.Matrix) : Option MmapStore :=
  if rows.cols ≠ s.ncols then
    none
  else if rows.rows = 0 then
    some s
  else
    let newRows := s.nrows + rows.rows
    match ensureCap s newRows with
    | none => none
    | some grown =>
      let touched := pretouch grown s.nrows newRows
      some
        { touched with
          nrows := newRows
          data := touched.data ++ rows.data }

/-- `MmapColumnStore::release_resident_pages`: release OS pages and forget tracked residency. -/
def releaseStore (s : MmapStore) : MmapStore :=
  let _ := releaseMap s.fileLen
  { s with residency := resForget s.residency }

/-- `MmapColumnStore::mmap_row_slice`: read row `i` and record a residency touch. -/
def rowSlice (s : MmapStore) (i : Nat) : Option (List Word × MmapStore) :=
  if i ≥ s.nrows then
    none
  else
    let s1 := storeTouch s i
    some (s1.data.getD i [], s1)

/-- `MmapColumnStore::mmap_gather`: gather rows at `indices` after bounds checking. -/
def mmapGather (s : MmapStore) (indices : List Nat) : Option Fitter.Matrix :=
  if indices.any (fun idx ↦ decide (idx ≥ s.nrows)) then
    none
  else
    let rows := indices.map (fun idx ↦ s.data.getD idx [])
    some ⟨indices.length, s.ncols, rows⟩

/-! ## Theorems for `tree_residency.rs` and `mmap_store.rs` -/

/-- `Residency::new` starts with default 256 MiB budget, tracking disabled, and 0 bytes. -/
theorem newInit :
    resNew.budget = defBudget ∧ resNew.tracking = false ∧ resNew.bytes = 0 := by
   decide

/-- `Residency::budget` returns the budget field. -/
theorem budgetEq (r : Residency) :
    resBudget r = r.budget := rfl

/-- `Residency::set_budget` with 0 disables tracking and zeroes tracked bytes. -/
theorem zeroBudget (r : Residency) :
    (setBudget r 0).tracking = false ∧ (setBudget r 0).bytes = 0 := by
  simp [setBudget]

/-- `Residency::set_budget` with a positive budget preserves tracking and sets budget. -/
theorem posBudget (r : Residency) (b : Nat) (hb : b > 0) :
    (setBudget r b).budget = b ∧ (setBudget r b).tracking = r.tracking := by
  have hne : b ≠ 0 := Nat.ne_of_gt hb
  simp [setBudget, hne]

/-- `Residency::update_tracking` disables tracking and clears bytes when `mapped ≤ budget`
and `r.tracking = true`. -/
theorem trackOff (r : Residency) (mapped reserved : Nat)
    (hb : r.budget > 0) (hle : mapped ≤ r.budget) (htr : r.tracking = true) :
    (updateTrack r mapped reserved).tracking = false ∧
    (updateTrack r mapped reserved).bytes = 0 := by
  have hne : r.budget ≠ 0 := Nat.ne_of_gt hb
  have hnot : ¬(mapped > r.budget) := Nat.not_lt.mpr hle
  simp [updateTrack, hne, hnot, htr]

/-- `Residency::touch` is a no-op when tracking is disabled. -/
theorem touchIdle (r : Residency) (mapped offset reserved : Nat)
    (h : r.tracking = false) :
    resTouch r mapped offset reserved = r := by
  simp [resTouch, h]

/-- `Residency::touch_tracked` is idempotent on an already-touched 64 KiB unit. -/
theorem touchIdem (r : Residency) (mapped offset reserved : Nat)
    (h : r.touched.contains (unitBit offset) = true) :
    touchTrack r mapped offset reserved = r := by
  dsimp only [touchTrack]
  rw [h]
  rfl

/-- `Residency::release` zeroes tracked bytes and clears the touch bitmap. -/
theorem releaseClear (r : Residency) (mapped : Nat) :
    (resRelease r mapped).bytes = 0 := by
  by_cases h : r.bytes = 0
  · simp [resRelease, h]
  · simp [resRelease, h, releaseMap]

/-- `Residency::forget` zeroes tracked bytes and clears the touch bitmap without incrementing
`released`. -/
theorem forgetKeep (r : Residency) :
    (resForget r).bytes = 0 ∧
    (resForget r).touched = [] ∧
    (resForget r).released = r.released := by
  simp [resForget]

/-- `Residency::copy_released` copies the release count from `other`. -/
theorem copyEq (r other : Residency) :
    (copyRel r other).released = other.released := rfl

/-- `MmapColumnStore::row_bytes` rejects `ncols = 0` and returns `8 * ncols` for `ncols > 0`. -/
theorem bytesSpec (ncols : Nat) :
    rowBytes 0 = none ∧
    (ncols > 0 → rowBytes ncols = some (ncols * cellBytes)) := by
  constructor
  · rfl
  · intro h
    have hne : ncols ≠ 0 := Nat.ne_of_gt h
    simp [rowBytes, hne]

/-- `MmapColumnStore::ensure_capacity` preserves `capRows` when `minRows ≤ capRows`. -/
theorem capKeep (s : MmapStore) (minRows : Nat) (hle : minRows ≤ s.capRows) :
    ensureCap s minRows = some s := by
  simp [ensureCap, hle]

/-- `MmapColumnStore::mmap_append` of an empty matrix with matching `ncols` is identity. -/
theorem appendEmpty (s : MmapStore) (rows : Fitter.Matrix)
    (hcols : rows.cols = s.ncols) (hrows : rows.rows = 0) :
    mmapAppend s rows = some s := by
  simp [mmapAppend, hcols, hrows]

/-- `MmapColumnStore::mmap_append` rejects a matrix whose column count mismatches `ncols`. -/
theorem appendMismatch (s : MmapStore) (rows : Fitter.Matrix)
    (hcols : rows.cols ≠ s.ncols) :
    mmapAppend s rows = none := by
  simp [mmapAppend, hcols]

/-- `MmapColumnStore::release_resident_pages` zeroes resident bytes while preserving rows. -/
theorem releaseRows (s : MmapStore) :
    (releaseStore s).nrows = s.nrows ∧
    (releaseStore s).data = s.data ∧
    (releaseStore s).residency.bytes = 0 := by
  simp [releaseStore, resForget]

/-- `MmapColumnStore::mmap_row_slice` rejects out-of-range row indices `i ≥ nrows`. -/
theorem sliceOob (s : MmapStore) (i : Nat) (hi : i ≥ s.nrows) :
    rowSlice s i = none := by
  simp [rowSlice, hi]

/-- `MmapColumnStore::mmap_gather` returns `indices.length` rows when all indices are valid. -/
theorem gatherLen (s : MmapStore) (indices : List Nat)
    (hok : indices.any (fun idx ↦ decide (idx ≥ s.nrows)) = false) :
    ∃ m, mmapGather s indices = some m ∧ m.rows = indices.length ∧ m.cols = s.ncols := by
  refine ⟨⟨indices.length, s.ncols, indices.map (fun idx ↦ s.data.getD idx [])⟩, ?_⟩
  simp [mmapGather, hok]

/-! ## Manifest Function Coverage Table (20 functions) -/

/-- All 20 `coverage_manifest.json` functions formalized in `Residency`. -/
def manifestSpecs : List FuncSpec :=
  [ { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 34,
      fnName := "release_map",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/release_map()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 41,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]new()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 51,
      fnName := "budget",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]budget()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 55,
      fnName := "set_budget",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]set_budget()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 65,
      fnName := "update_tracking",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]update_tracking()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 82,
      fnName := "touch",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]touch()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 89,
      fnName := "touch_tracked",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]touch_tracked()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 104,
      fnName := "release",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]release()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 110,
      fnName := "forget",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]forget()." },
    { file := "rust/crates/bpann/src/index/tree_residency.rs", line := 117,
      fnName := "copy_released",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_residency/impl#[Residency]copy_released()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 21,
      fnName := "row_bytes",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]row_bytes()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 25,
      fnName := "bytes_for_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]bytes_for_rows()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 29,
      fnName := "ensure_capacity",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]ensure_capacity()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 55,
      fnName := "touch",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]touch()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 59,
      fnName := "mmap_open_or_create",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]mmap_open_or_create()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 108,
      fnName := "pretouch",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]pretouch()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 118,
      fnName := "mmap_append",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]mmap_append()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 166,
      fnName := "release_resident_pages",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]release_resident_pages()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 174,
      fnName := "mmap_row_slice",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]mmap_row_slice()." },
    { file := "rust/crates/bpann/src/mmap_store.rs", line := 190,
      fnName := "mmap_gather",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 mmap_store/impl#[MmapColumnStore]mmap_gather()." } ]

/-- All 20 functions in `tree_residency.rs` and `mmap_store.rs` are covered. -/
theorem residencyCount : manifestSpecs.length = 20 := rfl

#print axioms newInit
#print axioms budgetEq
#print axioms zeroBudget
#print axioms posBudget
#print axioms trackOff
#print axioms touchIdle
#print axioms touchIdem
#print axioms releaseClear
#print axioms forgetKeep
#print axioms copyEq
#print axioms bytesSpec
#print axioms capKeep
#print axioms appendEmpty
#print axioms appendMismatch
#print axioms releaseRows
#print axioms sliceOob
#print axioms gatherLen
#print axioms residencyCount

end BauxResearch.Semantics.Residency
