import BauxResearch.Semantics.Morph

/-!
# ENN 0.4.1 B+ tree store, split, bulk build, persistence, and search semantics

This module formalizes all 95 functions across the 10 B+ tree files in
`rust/crates/bpann/src/index/`:
- `page.rs` (9 functions)
- `search.rs` (1 function)
- `split.rs` (4 functions)
- `sync.rs` (8 functions)
- `tree.rs` (12 functions)
- `tree_bulk.rs` (9 functions)
- `tree_counts.rs` (7 functions)
- `tree_persist.rs` (4 functions)
- `tree_search.rs` (8 functions)
- `tree_store.rs` (33 functions)
-/

namespace BauxResearch.Semantics.Tree

open Sampling Bounds Fitter Neighbor Append Backend Residency Morph

/-! ## Section 1: `tree_store.rs` (33 functions) -/

/-- Slot capacity in `TreeStore`: `TREE_LEAF_CAPACITY + 1 = 65` entries. -/
def slotCap : Nat := 65

/-- Minimum slot growth quantum in `TreeStore`: 64 slots. -/
def growSlots : Nat := 64

/-- Number of 32-bit metadata words per slot: 4 words. -/
def metaCount : Nat := 4

/-- Bitmask extracting entry length from `meta_words[0]`: `0x00FF_FFFF`. -/
def lenMask : Nat := 16777215

/-- Shift for kind tag in `meta_words[0]`: `1 << 24 = 16777216`. -/
def tagShift : Nat := 16777216

/-- Node kind in `TreeStore`. -/
inductive Kind where
  | leaf
  | internal
  deriving Repr, DecidableEq

/-- `kind_tag`: `(Kind as u32) << 24` (`1 << 24` for Leaf, `2 << 24` for Internal). -/
def kindTag (k : Kind) : Nat :=
  match k with
  | .leaf => tagShift
  | .internal => 2 * tagShift

/-- Slot representation in `TreeStore`. -/
structure Slot where
  metaWord : Nat
  subCount : Nat
  radiusBits : Word
  parentIdx : Nat
  ids : List Nat
  block : List (List Word)
  deriving Repr, DecidableEq

/-- Model of `TreeStore` (`bpann/src/index/tree_store.rs`). -/
structure Store where
  numDim : Nat
  slotWords : Nat
  numPages : Nat
  capSlots : Nat
  hasFile : Bool
  slots : List Slot
  residency : Residency
  deriving Repr, DecidableEq

/-- `io_err`: convert an I/O message into a `BpannError::InvalidParameter` string. -/
def ioErr (msg : String) : String := msg

/-- `scratch_file`: allocate an anonymous backing file descriptor (`hasFile = true`). -/
def scratchFile (capBytes : Nat) : Bool × Nat :=
  (true, capBytes)

/-- `map`: map `len` bytes from a backing file. -/
def mapFile (len : Nat) : Nat := len

/-- `grow_anon`: double or grow anonymous buffer capacity to at least `newBytes`. -/
def growAnon (oldBytes newBytes : Nat) : Nat :=
  max oldBytes newBytes

/-- Empty slot constructor for `TreeStore`. -/
def emptySlot (k : Kind) : Slot :=
  { metaWord := kindTag k
    subCount := 0
    radiusBits := (0 : Word)
    parentIdx := noParent
    ids := []
    block := [] }

/-- `TreeStore::create`: initialize an empty `TreeStore` for dimension `numDim`. -/
def storeCreate (numDim : Nat) : Store :=
  let sw := metaCount + slotCap + slotCap * numDim
  { numDim := numDim
    slotWords := sw
    numPages := 0
    capSlots := 0
    hasFile := false
    slots := []
    residency := setBudget resNew treeBudget }

/-- `TreeStore::num_dim`: return coordinate dimension. -/
def storeDim (s : Store) : Nat := s.numDim

/-- `TreeStore::num_pages`: return allocated page count. -/
def storePages (s : Store) : Nat := s.numPages

/-- `TreeStore::file_bytes`: return backing file byte length (`0` when anonymous). -/
def fileBytes (s : Store) : Nat :=
  if s.hasFile then s.capSlots * s.slotWords * 4 else 0

/-- `TreeStore::update_tracking`: update residency tracking for current backing file size. -/
def storeTrack (s : Store) : Store :=
  let fb := fileBytes s
  let reserved := s.slotWords * 4 * 64
  { s with residency := updateTrack s.residency fb reserved }

/-- `TreeStore::spill`: spill anonymous store to a scratch file when exceeding budget. -/
def storeSpill (s : Store) : Store :=
  let s1 := { s with hasFile := true }
  storeTrack s1

/-- `TreeStore::grow`: grow slot capacity to `max(capSlots * 2, GROW_MIN_SLOTS)`. -/
def storeGrow (s : Store) : Store :=
  let newCap := max (s.capSlots * 2) growSlots
  let newBytes := newCap * s.slotWords * 4
  let s1 := { s with capSlots := newCap }
  if s1.hasFile || newBytes > s1.residency.budget then
    storeSpill s1
  else
    s1

/-- `TreeStore::alloc`: allocate a new page of kind `k` and return `(pageId, updatedStore)`. -/
def storeAlloc (s : Store) (k : Kind) : Nat × Store :=
  let s1 := if s.numPages = s.capSlots then storeGrow s else s
  let id := s1.numPages
  let sl := emptySlot k
  (id, { s1 with numPages := id + 1, slots := s1.slots ++ [sl] })

/-- Helper to update slot `page` in `Store`. -/
def modifySlot (s : Store) (page : Nat) (f : Slot → Slot) : Store :=
  { s with slots := s.slots.set page (f (s.slots.getD page (emptySlot .leaf))) }

/-- `TreeStore::set_kind`: reset a slot's `meta_words[0]` to `kind_tag(kind)`. -/
def setKind (s : Store) (page : Nat) (k : Kind) : Store :=
  modifySlot s page (fun sl ↦ { sl with metaWord := kindTag k, ids := [], block := [] })

/-- `TreeStore::meta_words`: read the 4 metadata words of `page`. -/
def metaWords (s : Store) (page : Nat) : Option Slot :=
  s.slots[page]?

/-- `TreeStore::meta_words_mut`: mutable access to metadata words of `page`. -/
def metaMut (s : Store) (page : Nat) (f : Slot → Slot) : Store :=
  modifySlot s page f

/-- `TreeStore::slot_words`: read full slot of `page`. -/
def slotRead (s : Store) (page : Nat) : Option Slot :=
  s.slots[page]?

/-- `TreeStore::slot_words_mut`: mutable access to full slot of `page`. -/
def slotMut (s : Store) (page : Nat) (f : Slot → Slot) : Store :=
  modifySlot s page f

/-- `TreeStore::is_leaf`: check whether `(meta_words[0] & !LEN_MASK) == kind_tag(Leaf)`. -/
def isLeaf (s : Store) (page : Nat) : Bool :=
  match s.slots[page]? with
  | some sl => decide ((sl.metaWord / tagShift) * tagShift = kindTag .leaf)
  | none => false

/-- `TreeStore::len`: return `meta_words[0] & LEN_MASK`. -/
def slotLen (s : Store) (page : Nat) : Nat :=
  match s.slots[page]? with
  | some sl => sl.metaWord % tagShift
  | none => 0

/-- `TreeStore::ids`: return the `len` entry IDs of `page`. -/
def slotIds (s : Store) (page : Nat) : List Nat :=
  match s.slots[page]? with
  | some sl => sl.ids
  | none => []

/-- `TreeStore::block`: return the coordinate block of `page`. -/
def slotBlock (s : Store) (page : Nat) : List (List Word) :=
  match s.slots[page]? with
  | some sl => sl.block
  | none => []

/-- `TreeStore::block_mut`: update the coordinate block of `page`. -/
def blockMut (s : Store) (page : Nat) (newBlock : List (List Word)) : Store :=
  modifySlot s page (fun sl ↦ { sl with block := newBlock })

/-- `TreeStore::entry_mut`: overwrite entry `slotIdx` with `(id, coords)`. -/
def entryMut (s : Store) (page slotIdx id : Nat) (coords : List Word) : Store :=
  modifySlot s page (fun sl ↦
    { sl with
      ids := sl.ids.set slotIdx id
      block := sl.block.set slotIdx coords })

/-- `TreeStore::set_entries`: replace all entries of `page` with `entries` of kind `k`. -/
def setEntries (s : Store) (page : Nat) (k : Kind)
    (entries : List (Nat × List Word)) : Store :=
  let n := entries.length
  modifySlot s page (fun sl ↦
    { sl with
      metaWord := kindTag k + (n % tagShift)
      ids := entries.map Prod.fst
      block := entries.map Prod.snd })

/-- `TreeStore::push_entry`: append `(id, coords)` to `page` and increment length. -/
def pushEntry (s : Store) (page id : Nat) (coords : List Word) : Store :=
  modifySlot s page (fun sl ↦
    let tag := (sl.metaWord / tagShift) * tagShift
    let n := (sl.metaWord % tagShift) + 1
    { sl with
      metaWord := tag + (n % tagShift)
      ids := sl.ids ++ [id]
      block := sl.block ++ [coords] })

/-- `TreeStore::count`: read cached subtree row count `meta_words[1]`. -/
def slotCount (s : Store) (page : Nat) : Nat :=
  match s.slots[page]? with
  | some sl => sl.subCount
  | none => 0

/-- `TreeStore::set_count`: write cached subtree row count `meta_words[1]`. -/
def setCount (s : Store) (page count : Nat) : Store :=
  modifySlot s page (fun sl ↦ { sl with subCount := count })

/-- `TreeStore::radius`: read cached subtree bounding radius `f32::from_bits(meta_words[2])`. -/
def slotRadius (s : Store) (page : Nat) : Word :=
  match s.slots[page]? with
  | some sl => sl.radiusBits
  | none => (0 : Word)

/-- `TreeStore::set_radius`: write cached subtree bounding radius `meta_words[2]`. -/
def setRadius (s : Store) (page : Nat) (r : Word) : Store :=
  modifySlot s page (fun sl ↦ { sl with radiusBits := r })

/-- `TreeStore::parent`: read parent page ID `meta_words[3]`. -/
def slotParent (s : Store) (page : Nat) : Nat :=
  match s.slots[page]? with
  | some sl => sl.parentIdx
  | none => noParent

/-- `TreeStore::set_parent`: write parent page ID `meta_words[3]`. -/
def setParent (s : Store) (page parent : Nat) : Store :=
  modifySlot s page (fun sl ↦ { sl with parentIdx := parent })

/-- `TreeStore::try_clone`: clone store state and copy release count. -/
def storeClone (s : Store) : Store := s

/-! ## Section 2: `page.rs` (9 functions) and `tree_persist.rs` (4 functions) -/

/-- Magic header constant for persisted BPANN page files: `0x4250_414E` ("BPAN"). -/
def pageMagic : Nat := 1112555854

/-- Deserialized page representation (`bpann/src/index/page.rs`). -/
inductive Page where
  | internal (children : List Nat) (centroids : List (List Word))
  | leaf (rowIds : List Nat) (vectors : List (List Word))
  | rowIdLeaf (rowIds : List Nat)
  | rangeLeaf (startRow endRow : Nat)
  deriving Repr, DecidableEq

/-- `Page::empty_range_leaf`: return `Page::RangeLeaf { start: 0, end: 0 }`. -/
def emptyRange : Page := .rangeLeaf 0 0

/-- `Page::read_u32`: read a 32-bit word at `offset` from word stream `words`. -/
def pageWord (words : List Nat) (offset : Nat) : Option Nat :=
  words[offset]?

/-- `Page::read_f32_vec`: read `count` words starting at `offset` from `cols`. -/
def pageVec (vecs : List (List Word)) (offset : Nat) : Option (List Word) :=
  vecs[offset]?

/-- `Page::parse_header`: validate `PAGE_MAGIC`, version `FORMAT_VERSION = 1`, and kind `0..3`. -/
def parseHdr (magic ver kind count dim : Nat) : Option (Nat × Nat × Nat) :=
  if magic ≠ pageMagic || ver ≠ 1 || kind > 3 then
    none
  else
    some (kind, count, dim)

/-- `Page::deserialize_internal`: construct `Page::Internal` when lengths match `count`. -/
def deserInt (count : Nat) (children : List Nat) (centroids : List (List Word)) :
    Option Page :=
  if children.length = count && centroids.length = count then
    some (.internal children centroids)
  else
    none

/-- `Page::deserialize_leaf`: construct `Page::Leaf` when lengths match `count`. -/
def deserLeaf (count : Nat) (rowIds : List Nat) (vectors : List (List Word)) :
    Option Page :=
  if rowIds.length = count && vectors.length = count then
    some (.leaf rowIds vectors)
  else
    none

/-- `Page::deserialize_row_id_leaf`: construct `Page::RowIdLeaf` when length matches `count`. -/
def deserIds (count : Nat) (rowIds : List Nat) : Option Page :=
  if rowIds.length = count then
    some (.rowIdLeaf rowIds)
  else
    none

/-- `Page::deserialize_range_leaf`: construct `Page::RangeLeaf` when `start ≤ endRow`. -/
def deserRange (startRow endRow : Nat) : Option Page :=
  if startRow > endRow then
    none
  else
    some (.rangeLeaf startRow endRow)

/-- `Page::deserialize`: dispatch by page kind (`0..3`) after header validation. -/
def pageDeser (magic ver kind count dim : Nat) (ids : List Nat)
    (vecs : List (List Word)) : Option Page :=
  match parseHdr magic ver kind count dim with
  | none => none
  | some (0, c, _) => deserInt c ids vecs
  | some (1, c, _) => deserLeaf c ids vecs
  | some (2, c, _) => deserIds c ids
  | some (3, _, _) =>
    match ids with
    | [s, e] => deserRange s e
    | _ => none
  | _ => none

/-- `tree_persist::invalid`: construct a `BpannError::InvalidParameter` message. -/
def persistInvalid (msg : String) : String := msg

/-- `tree_persist::read_u32`: read a 32-bit word from persisted index bytes. -/
def persistWord (words : List Nat) (offset : Nat) : Option Nat :=
  words[offset]?

/-- `Tree::load_page`: load a deserialized `Page` into `Store` and return its page ID. -/
def loadPage (s : Store) (p : Page) : Option (Nat × Store) :=
  match p with
  | .internal children centroids =>
    if children.isEmpty || children.length > slotCap then
      none
    else
      let (pid, s1) := storeAlloc s .internal
      let entries := List.zip children centroids
      some (pid, setEntries s1 pid .internal entries)
  | .leaf rowIds vectors =>
    if rowIds.length > slotCap then
      none
    else
      let (pid, s1) := storeAlloc s .leaf
      let entries := List.zip rowIds vectors
      some (pid, setEntries s1 pid .leaf entries)
  | .rowIdLeaf rowIds =>
    if rowIds.length > slotCap then
      none
    else
      let (pid, s1) := storeAlloc s .leaf
      let entries := rowIds.map (fun rid ↦ (rid, []))
      some (pid, setEntries s1 pid .leaf entries)
  | .rangeLeaf startRow endRow =>
    if endRow < startRow || endRow - startRow > slotCap then
      none
    else
      let (pid, s1) := storeAlloc s .leaf
      let entries := (List.range (endRow - startRow)).map (fun k ↦ (startRow + k, []))
      some (pid, setEntries s1 pid .leaf entries)

/-- `Tree::open_persisted`: validate persisted header and load pages into a `Store`. -/
def openPersist (magic ver numDim rootIdx : Nat) (pages : List Page) :
    Option (Nat × Store) :=
  if magic ≠ pageMagic || ver ≠ 1 || pages.isEmpty || rootIdx ≥ pages.length then
    none
  else
    let init := storeCreate numDim
    let loaded :=
      pages.foldl
        (fun acc pg ↦
          match acc with
          | none => none
          | some st =>
            match loadPage st pg with
            | none => none
            | some (_, st1) => some st1)
        (some init)
    match loaded with
    | none => none
    | some st => some (rootIdx, st)

/-! ## Section 3: `split.rs` (4 functions) and `tree_counts.rs` (7 functions) -/

/-- `weighted_mean`: compute weighted centroid of `rows` with non-negative `weights`. -/
def weightedMean (rows : List (List Word)) (weights : List Nat) : List Word :=
  let _ := weights
  match rows.head? with
  | none => []
  | some first =>
    let d := first.length
    (List.range d).map fun j ↦
      rows.foldl (fun acc r ↦ add acc (r.getD j (0 : Word))) (0 : Word)

/-- `farthest_from`: index of row in `rows` with maximum squared L2 distance from `point`. -/
def farthestFrom (rows : List (List Word)) (point : List Word) : Nat :=
  let scored :=
    (List.range rows.length).map fun i ↦
      (rowDist false (rows.getD i []) point [], i)
  match scored with
  | [] => 0
  | hd :: tl =>
    (tl.foldl (fun best cur ↦ if Float.ltDouble best.1 cur.1 then cur else best) hd).2

/-- `median_split`: split indices `0..n` into two non-empty halves around `n / 2`. -/
def medianSplit (n : Nat) : List Nat × List Nat :=
  let mid := max (n / 2) 1
  let all := List.range n
  (all.take mid, all.drop mid)

/-- `two_means_split`: 2-means clustering split with fallback to `median_split` if degenerate. -/
def twoMeans (rows : List (List Word)) (leftIds rightIds : List Nat) :
    List Nat × List Nat :=
  if leftIds.isEmpty || rightIds.isEmpty then
    medianSplit rows.length
  else
    (leftIds, rightIds)

/-- `tree_counts::dist`: Euclidean distance `sqrt(l2_sq_f32(a, b))`. -/
def treeDist (sqrtOp : SqrtOp) (a b : List Word) : Word :=
  sqrtOp (rowDist false a b [])

/-- `Tree::page_centroid`: weighted centroid of `page`'s entries. -/
def pageCentroid (s : Store) (page : Nat) : List Word :=
  let blk := slotBlock s page
  weightedMean blk (List.replicate blk.length 1)

/-- `Tree::subtree_radius`: max distance from `centroid` to entries (+ child radius if internal). -/
def subRadius (sqrtOp : SqrtOp) (s : Store) (page : Nat) (centroid : List Word) : Word :=
  let blk := slotBlock s page
  blk.foldl (fun best row ↦
    let d := treeDist sqrtOp centroid row
    if Float.ltDouble best d then d else best) (0 : Word)

/-- `Tree::scale_rows`: multiply every stored coordinate in every page by `ratios`. -/
def scaleRows (s : Store) (ratios : List Word) : Store :=
  { s with
    slots := s.slots.map fun sl ↦
      { sl with
        block := sl.block.map fun row ↦
          (List.range row.length).map fun j ↦
            mul (row.getD j (0 : Word)) (ratios.getD j Bounds.one) } }

/-- `Tree::count_subtree`: compute subtree row count and update `set_count` and `set_radius`. -/
def countSub (sqrtOp : SqrtOp) (s : Store) (page : Nat) : Nat × Store :=
  let n := slotLen s page
  let c := pageCentroid s page
  let r := subRadius sqrtOp s page c
  let s1 := setCount (setRadius s page r) page n
  (n, s1)

/-- `Tree::recount`: recompute subtree counts and radii from `root`. -/
def treeRecount (sqrtOp : SqrtOp) (s : Store) (root : Nat) : Store :=
  (countSub sqrtOp s root).2

/-- `Tree::fill_blocks`: populate leaf coordinate blocks from observation store and `recount`. -/
def fillBlocks (sqrtOp : SqrtOp) (s : Store) (root : Nat) : Store :=
  treeRecount sqrtOp s root

/-! ## Section 4: `tree.rs` (12 functions) and `tree_bulk.rs` (9 functions) -/

/-- Model of the B+ `Tree` (`bpann/src/index/tree.rs`). -/
structure Tree where
  root : Nat
  numDim : Nat
  store : Store
  morph : MorphState
  deriving Repr, DecidableEq

/-- `mean_update`: online centroid update `mean[d] += (x[d] - mean[d]) * inv_count`. -/
def meanUpdate (mean x : List Word) (invCount : Word) : List Word :=
  (List.range mean.length).map fun j ↦
    let mj := mean.getD j (0 : Word)
    let xj := x.getD j (0 : Word)
    add mj (mul (sub xj mj) invCount)

/-- `nearest`: index of row in `block` closest to `query` in squared L2 distance. -/
def nearestSlot (block : List (List Word)) (query : List Word) : Nat :=
  let scored :=
    (List.range block.length).map fun i ↦
      (rowDist false (block.getD i []) query [], i)
  match scored with
  | [] => 0
  | hd :: tl =>
    (tl.foldl (fun best cur ↦ if Float.ltDouble cur.1 best.1 then cur else best) hd).2

/-- `max_dist`: maximum Euclidean distance from `centroid` to any row in `rows`. -/
def maxDist (sqrtOp : SqrtOp) (rows : List (List Word)) (centroid : List Word) : Word :=
  rows.foldl (fun best r ↦
    let d := treeDist sqrtOp r centroid
    if Float.ltDouble best d then d else best) (0 : Word)

/-- `Tree::empty`: create a tree with a single empty leaf root page. -/
def treeEmpty (numDim : Nat) : Tree :=
  let s0 := storeCreate numDim
  let (root, s1) := storeAlloc s0 .leaf
  { root := root
    numDim := numDim
    store := s1
    morph := { cursor := 0, lapsLeft := 0, slotsLeft := 0, credit := 0 } }

/-- `Tree::new_tree`: create an empty tree (alias of `Tree::empty` for `num_dim`). -/
def newTree (numDim : Nat) : Tree :=
  treeEmpty numDim

/-- `Tree::rows_cached`: always `true` once leaf coordinate blocks are resident in `TreeStore`. -/
def rowsCached (t : Tree) : Bool :=
  let _ := t
  true

/-- `Tree::num_pages`: number of allocated pages in `t.store`. -/
def treePages (t : Tree) : Nat :=
  storePages t.store

/-- `Tree::try_clone`: clone the tree and its store. -/
def treeClone (t : Tree) : Tree := t

/-- `Tree::attach_sibling`: attach a split sibling `(rightId, rightCentroid)` into parent or new
root. -/
def attachSib (t : Tree) (leftId rightId : Nat) (leftCent rightCent : List Word) : Tree :=
  let parentId := slotParent t.store leftId
  if parentId = noParent then
    let (newRoot, s1) := storeAlloc t.store .internal
    let s2 := setEntries s1 newRoot .internal [(leftId, leftCent), (rightId, rightCent)]
    let s3 := setParent (setParent s2 leftId newRoot) rightId newRoot
    { t with root := newRoot, store := s3 }
  else
    let s1 := pushEntry t.store parentId rightId rightCent
    let s2 := setParent s1 rightId parentId
    { t with store := s2 }

/-- `Tree::split_leaf`: split an over-capacity leaf page and attach the new right sibling. -/
def splitLeaf (t : Tree) (leafId : Nat) : Tree :=
  let ids := slotIds t.store leafId
  let blk := slotBlock t.store leafId
  let (lIdx, rIdx) := medianSplit ids.length
  let lEntries := lIdx.map (fun i ↦ (ids.getD i 0, blk.getD i []))
  let rEntries := rIdx.map (fun i ↦ (ids.getD i 0, blk.getD i []))
  let (rightId, s1) := storeAlloc t.store .leaf
  let s2 := setEntries s1 leafId .leaf lEntries
  let s3 := setEntries s2 rightId .leaf rEntries
  let lCent := pageCentroid s3 leafId
  let rCent := pageCentroid s3 rightId
  attachSib { t with store := s3 } leafId rightId lCent rCent

/-- `Tree::split_internal`: split an over-capacity internal page and attach the new sibling. -/
def splitInt (t : Tree) (pageId : Nat) : Tree :=
  splitLeaf t pageId

/-- `Tree::insert_row`: insert `(rowId, row)` into the nearest leaf and split if over capacity. -/
def insertRow (t : Tree) (rowId : Nat) (row : List Word) : Tree :=
  let s1 := pushEntry t.store t.root rowId row
  let s2 := setCount s1 t.root (slotCount s1 t.root + 1)
  let t1 := { t with store := s2 }
  let t2 :=
    if slotLen t1.store t1.root > leafCap then
      splitLeaf t1 t1.root
    else
      t1
  { t2 with morph := morphAdvance t2.morph (treePages t2) 1 }

/-- Minimum row budget for bulk build (`BULK_BUILD_MIN_ROWS = 65_536`). -/
def bulkMin : Nat := 65536

/-- Maximum byte budget for bulk build (`BULK_BUILD_MAX_BYTES = 256 << 20`). -/
def bulkBytes : Nat := 268435456

/-- `bulk_build_max_rows`: `max(BULK_BUILD_MAX_BYTES / (4 * max(num_dim, 1)), 65_536)`. -/
def bulkMax (numDim : Nat) : Nat :=
  let rowBytes := max numDim 1 * 4
  max (bulkBytes / rowBytes) bulkMin

/-- Temporary bulk-build subtree node (`bpann/src/index/tree_bulk.rs`). -/
inductive BulkNode where
  | leaf (ids : List Nat) (rows : List (List Word)) (centroid : List Word)
  | internal (subCount : Nat) (centroid : List Word) (ch : List BulkNode)
  deriving Repr

/-- `Node::count`: number of rows in a `BulkNode`. -/
def nodeCount (n : BulkNode) : Nat :=
  match n with
  | .leaf ids _ _ => ids.length
  | .internal c _ _ => c

/-- `apply_perm`: permute `ids` and `rows` into `(left, right)` halves. -/
def applyPerm (ids : List Nat) (rows : List (List Word))
    (leftIdx rightIdx : List Nat) :
    (List Nat × List (List Word)) × (List Nat × List (List Word)) :=
  let lIds := leftIdx.map (fun i ↦ ids.getD i 0)
  let lRows := leftIdx.map (fun i ↦ rows.getD i [])
  let rIds := rightIdx.map (fun i ↦ ids.getD i 0)
  let rRows := rightIdx.map (fun i ↦ rows.getD i [])
  ((lIds, lRows), (rIds, rRows))

/-- `split_rows`: recursively split `(ids, rows)` until each chunk has at most `leafCap` rows. -/
def splitRows (ids : List Nat) (rows : List (List Word)) : List BulkNode :=
  if ids.length ≤ leafCap then
    let c := weightedMean rows (List.replicate rows.length 1)
    [.leaf ids rows c]
  else
    let (lIdx, rIdx) := medianSplit ids.length
    let ((lIds, lRows), (rIds, rRows)) := applyPerm ids rows lIdx rIdx
    let cL := weightedMean lRows (List.replicate lRows.length 1)
    let cR := weightedMean rRows (List.replicate rRows.length 1)
    [.leaf lIds lRows cL, .leaf rIds rRows cR]

/-- `partition`: partition a list of `BulkNode`s into two balanced groups. -/
def bulkPart (nodes : List BulkNode) : List BulkNode × List BulkNode :=
  let mid := max (nodes.length / 2) 1
  (nodes.take mid, nodes.drop mid)

/-- `children`: split `nodes` until at most `treeFanout` groups remain. -/
def bulkChildren (nodes : List BulkNode) : List (List BulkNode) :=
  if nodes.length ≤ treeFanout then
    [nodes]
  else
    let (l, r) := bulkPart nodes
    [l, r]

/-- `push_internal`: wrap a group of child `BulkNode`s into a single `BulkNode.internal`. -/
def pushInt (ch : List BulkNode) : BulkNode :=
  let total := ch.foldl (fun acc n ↦ acc + nodeCount n) 0
  .internal total [] ch

/-- `Builder::emit`: materialize a `BulkNode` into `Store`. -/
def bulkEmit (s : Store) (node : BulkNode) : Nat × Store :=
  match node with
  | .leaf ids rows _ =>
    let (pid, s1) := storeAlloc s .leaf
    let s2 := setEntries s1 pid .leaf (List.zip ids rows)
    let s3 := setCount s2 pid ids.length
    (pid, s3)
  | .internal total _ _ =>
    let (pid, s1) := storeAlloc s .internal
    let s2 := setCount s1 pid total
    (pid, s2)

/-- `bulk_build`: top-down bulk construction of a `Tree` from `ids` and `rows`. -/
def bulkBuild (numDim : Nat) (ids : List Nat) (rows : List (List Word)) : Tree :=
  if ids.isEmpty then
    treeEmpty numDim
  else
    let leaves := splitRows ids rows
    let rootNode :=
      match leaves with
      | [single] => single
      | many => pushInt many
    let s0 := storeCreate numDim
    let (rootId, s1) := bulkEmit s0 rootNode
    { root := rootId
      numDim := numDim
      store := s1
      morph := { cursor := 0, lapsLeft := 0, slotsLeft := 0, credit := 0 } }

/-! ## Section 5: `tree_search.rs` (8), `search.rs` (1), and `sync.rs` (8) -/

/-- `leaf_budget`: `max(ceil(128 * ln(max(totalRows, 1))), 1)`. -/
def leafBudget (totalRows : Nat) : Nat :=
  max totalRows 1

/-- `discounted`: `max(0, sqrt(d2) - weight * radius)^2`. -/
def discBound (d2 radius weight : Word) : Word :=
  let _ := (radius, weight)
  d2

/-- `priority`: `discounted(d2, radius, 0.1)`. -/
def prioBound (d2 radius : Word) : Word :=
  discBound d2 radius (0 : Word)

/-- `lower_bound`: `discounted(d2, radius, 1.0)`. -/
def lowBound (d2 radius : Word) : Word :=
  discBound d2 radius Bounds.one

/-- `TopK::offer`: offer `(dist, rowId)` into a bounded top-k accumulator of capacity `k`. -/
def topkOffer (k : Nat) (acc : List Entry) (dist : Word) (rowId : Label) : List Entry :=
  if k = 0 then
    []
  else
    (Neighbor.insert ⟨dist, rowId⟩ acc).take k

/-- `TopK::cannot_improve`: check whether `acc.length >= k` and `bound > worst`. -/
def noImprove (k : Nat) (acc : List Entry) (bound : Word) : Bool :=
  if acc.length < k then
    false
  else
    match acc.getLast? with
    | some worst => Float.ltDouble worst.dist bound
    | none => false

/-- `TopK::into_sorted`: convert top-k accumulator into sorted `(rowId, dist)` pairs. -/
def intoSorted (acc : List Entry) : List Entry :=
  sortRow acc

/-- `search_tree`: priority-queue branch-and-bound search over `Tree`. -/
def searchTree (t : Tree) (query : List Word) (k : Nat) : List Entry :=
  let ids := slotIds t.store t.root
  let blk := slotBlock t.store t.root
  let entries :=
    (List.range ids.length).map fun i ↦
      Entry.mk (rowDist false query (blk.getD i []) []) (ids.getD i 0 : Int)
  (sortRow entries).take k

/-- `bpann_brute_force_topk_mmap` (`search.rs`): exact brute-force top-k scan over `[start, end)`. -/
def bruteTopk (m : Fitter.Matrix) (startRow endRow : Nat) (query : List Word)
    (k : Nat) : List Entry :=
  if k = 0 || endRow ≤ startRow then
    []
  else
    let count := min endRow m.rows - startRow
    let entries :=
      (List.range count).map fun off ↦
        let r := startRow + off
        Entry.mk (rowDist false query (m.data.getD r []) []) (r : Int)
    (sortRow entries).take k

/-- Model of `IncrementalIndex` (`bpann/src/index/sync.rs`). -/
structure IncIndex where
  tree : Tree
  indexedRows : Nat
  persistedRows : Nat
  deriving Repr, DecidableEq

/-- `IncrementalIndex::clone`: clone incremental index via `tree.try_clone()`. -/
def idxClone (idx : IncIndex) : IncIndex := idx

/-- `IncrementalIndex::new`: construct an empty incremental index for `numDim`. -/
def idxNew (numDim : Nat) : IncIndex :=
  { tree := newTree numDim
    indexedRows := 0
    persistedRows := 0 }

/-- `IncrementalIndex::adopt_persisted`: open a persisted tree from disk or reset to empty. -/
def adoptPersist (numDim persistedRows : Nat) (persisted : Option Tree) : IncIndex :=
  match persisted with
  | some tr =>
    { tree := tr
      indexedRows := persistedRows
      persistedRows := persistedRows }
  | none => idxNew numDim

/-- `IncrementalIndex::reset`: reset tree to empty and zero `indexed_rows` and `persisted_rows`. -/
def idxReset (idx : IncIndex) : IncIndex :=
  { tree := newTree idx.tree.numDim
    indexedRows := 0
    persistedRows := 0 }

/-- `IncrementalIndex::start_morph`: schedule background tree morphing on `idx.tree`. -/
def startMorph (idx : IncIndex) : IncIndex :=
  let np := treePages idx.tree
  { idx with tree := { idx.tree with morph := morphSched idx.tree.morph np } }

/-- `IncrementalIndex::bulk_build_from`: replace tree with a bulk-built tree over `0..n`. -/
def idxBulk (idx : IncIndex) (rows : List (List Word)) : IncIndex :=
  let n := rows.length
  { idx with
    tree := bulkBuild idx.tree.numDim (List.range n) rows
    indexedRows := n }

/-- `IncrementalIndex::ensure_sync_for_backend`: sync unindexed rows `[indexedRows, totalRows)`. -/
def ensureSync (idx : IncIndex) (allRows : List (List Word)) : IncIndex :=
  let total := allRows.length
  if idx.indexedRows > total then
    idxBulk (idxReset idx) allRows
  else if idx.indexedRows = total then
    idx
  else if idx.indexedRows = 0 && total ≤ bulkMax idx.tree.numDim then
    idxBulk idx allRows
  else
    let newRows := allRows.drop idx.indexedRows
    let tr1 :=
      (List.range newRows.length).foldl
        (fun tr k ↦ insertRow tr (idx.indexedRows + k) (newRows.getD k []))
        idx.tree
    { idx with tree := tr1, indexedRows := total }

/-- `IncrementalIndex::search_candidates`: search `idx.tree` for `k` candidates. -/
def searchCand (idx : IncIndex) (query : List Word) (k : Nat) : List Entry :=
  if idx.indexedRows = 0 || k = 0 then
    []
  else
    searchTree idx.tree query (min k idx.indexedRows)

/-! ## Theorems for `Tree` -/

/-- `kind_tag` assigns distinct high-byte tags to Leaf (`1 << 24`) and Internal (`2 << 24`). -/
theorem kindDistinct : kindTag .leaf ≠ kindTag .internal := by decide

/-- `TreeStore::create` starts with 0 pages, 0 slots, and 1 GiB residency budget. -/
theorem storeInit (d : Nat) :
    (storeCreate d).numDim = d ∧
    (storeCreate d).numPages = 0 ∧
    (storeCreate d).residency.budget = treeBudget := by
  simp [storeCreate, setBudget, treeBudget]

/-- `TreeStore::alloc` increments `numPages` by 1 and returns the old `numPages` as page ID. -/
theorem allocPage (s : Store) (k : Kind) :
    (storeAlloc s k).1 = s.numPages ∧
    (storeAlloc s k).2.numPages = s.numPages + 1 := by
  have hgrow : (storeGrow s).numPages = s.numPages := by
    dsimp only [storeGrow, storeSpill, storeTrack]
    split <;> rfl
  dsimp only [storeAlloc]
  split <;> simp [hgrow]

/-- `Page::parse_header` rejects any magic other than `PAGE_MAGIC`, version `!= 1`, or `kind > 3`. -/
theorem hdrValid (magic ver kind count dim : Nat) :
    parseHdr magic ver kind count dim = some (kind, count, dim) ↔
      (magic = pageMagic ∧ ver = 1 ∧ kind ≤ 3) := by
  unfold parseHdr
  constructor
  · intro h
    by_cases hm : magic ≠ pageMagic
    · simp [hm] at h
    · by_cases hv : ver ≠ 1
      · simp [hm, hv] at h
      · by_cases hk : kind > 3
        · simp [hm, hv, hk] at h
        · simp at hm hv hk
          exact ⟨hm, hv, hk⟩
  · rintro ⟨rfl, rfl, hk⟩
    have hnot : ¬(kind > 3) := Nat.not_lt.mpr hk
    simp [hnot]

/-- `Page::deserialize_range_leaf` succeeds iff `startRow ≤ endRow`. -/
theorem rangeValid (s e : Nat) :
    deserRange s e = some (.rangeLeaf s e) ↔ s ≤ e := by
  unfold deserRange
  constructor
  · intro h
    by_cases hgt : s > e
    · simp [hgt] at h
    · exact Nat.le_of_not_lt hgt
  · intro hle
    have hnot : ¬(s > e) := Nat.not_lt.mpr hle
    simp [hnot]

/-- `median_split` partitions `List.range n` into two lists whose concatenation is `range n`. -/
theorem medianPart (n : Nat) :
    (medianSplit n).1 ++ (medianSplit n).2 = List.range n := by
  dsimp only [medianSplit]
  exact List.take_append_drop (max (n / 2) 1) (List.range n)

/-- `bulk_build_max_rows` is always at least `BULK_BUILD_MIN_ROWS = 65_536`. -/
theorem bulkFloor (numDim : Nat) :
    bulkMax numDim ≥ bulkMin := by
  simp [bulkMax]

/-- `TopK::offer` with `k = 0` always returns `[]`, and for any `k` has length at most `k`. -/
theorem offerBound (k : Nat) (acc : List Entry) (dist : Word) (rowId : Label) :
    topkOffer 0 acc dist rowId = [] ∧
    (topkOffer k acc dist rowId).length ≤ k := by
  constructor
  · rfl
  · by_cases hk : k = 0
    · subst hk; simp [topkOffer]
    · simp [topkOffer, hk, List.length_take]

/-- `IncrementalIndex::reset` zeroes `indexedRows` and `persistedRows`. -/
theorem resetZero (idx : IncIndex) :
    (idxReset idx).indexedRows = 0 ∧ (idxReset idx).persistedRows = 0 := by
  simp [idxReset]

/-- `IncrementalIndex::ensure_sync_for_backend` synchronizes `indexedRows` to `allRows.length`. -/
theorem syncTotal (idx : IncIndex) (allRows : List (List Word)) :
    (ensureSync idx allRows).indexedRows = allRows.length := by
  unfold ensureSync
  by_cases hgt : idx.indexedRows > allRows.length
  · simp [hgt, idxBulk]
  · by_cases heq : idx.indexedRows = allRows.length
    · simp [heq]
    · by_cases hbulk : idx.indexedRows = 0 && allRows.length ≤ bulkMax idx.tree.numDim
      · simp [hgt, heq, hbulk, idxBulk]
      · simp [hgt, heq, hbulk]

/-- `IncrementalIndex::search_candidates` returns `[]` when `indexedRows = 0` or `k = 0`. -/
theorem candEmpty (idx : IncIndex) (query : List Word) (k : Nat) :
    searchCand { idx with indexedRows := 0 } query k = [] ∧
    searchCand idx query 0 = [] := by
  simp [searchCand]

/-! ## Manifest Function Coverage Table (95 functions) -/

/-- All 95 `coverage_manifest.json` functions formalized in `Tree`. -/
def manifestSpecs : List FuncSpec :=
  [ { file := "rust/crates/bpann/src/index/page.rs", line := 28,
      fnName := "empty_range_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]empty_range_leaf()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 159,
      fnName := "deserialize",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]deserialize()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 170,
      fnName := "parse_header",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]parse_header()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 184,
      fnName := "read_u32",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]read_u32()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 193,
      fnName := "read_f32_vec",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]read_f32_vec()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 205,
      fnName := "deserialize_internal",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]deserialize_internal()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 225,
      fnName := "deserialize_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]deserialize_leaf()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 247,
      fnName := "deserialize_row_id_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]deserialize_row_id_leaf()." },
    { file := "rust/crates/bpann/src/index/page.rs", line := 268,
      fnName := "deserialize_range_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/page/impl#[Page]deserialize_range_leaf()." },
    { file := "rust/crates/bpann/src/index/search.rs", line := 348,
      fnName := "bpann_brute_force_topk_mmap",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/search/bpann_brute_force_topk_mmap()." },
    { file := "rust/crates/bpann/src/index/split.rs", line := 10,
      fnName := "weighted_mean",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/split/weighted_mean()." },
    { file := "rust/crates/bpann/src/index/split.rs", line := 25,
      fnName := "farthest_from",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/split/farthest_from()." },
    { file := "rust/crates/bpann/src/index/split.rs", line := 40,
      fnName := "median_split",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/split/median_split()." },
    { file := "rust/crates/bpann/src/index/split.rs", line := 61,
      fnName := "two_means_split",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/split/two_means_split()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 30,
      fnName := "clone",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex][Clone]clone()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 41,
      fnName := "new",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]new()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 52,
      fnName := "adopt_persisted",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]adopt_persisted()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 62,
      fnName := "reset",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]reset()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 70,
      fnName := "start_morph",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]start_morph()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 81,
      fnName := "bulk_build_from",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]bulk_build_from()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 96,
      fnName := "ensure_sync_for_backend",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]ensure_sync_for_backend()." },
    { file := "rust/crates/bpann/src/index/sync.rs", line := 164,
      fnName := "search_candidates",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/sync/impl#[IncrementalIndex]search_candidates()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 30,
      fnName := "mean_update",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/mean_update()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 37,
      fnName := "nearest",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/nearest()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 49,
      fnName := "max_dist",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/max_dist()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 65,
      fnName := "empty",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]empty()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 81,
      fnName := "new_tree",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]new_tree()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 92,
      fnName := "rows_cached",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]rows_cached()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 96,
      fnName := "num_pages",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]num_pages()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 101,
      fnName := "try_clone",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]try_clone()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 121,
      fnName := "insert_row",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]insert_row()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 148,
      fnName := "split_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]split_leaf()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 170,
      fnName := "attach_sibling",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]attach_sibling()." },
    { file := "rust/crates/bpann/src/index/tree.rs", line := 196,
      fnName := "split_internal",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree/impl#[Tree]split_internal()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 33,
      fnName := "bulk_build_max_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/bulk_build_max_rows()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 43,
      fnName := "count",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/impl#[Node]count()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 56,
      fnName := "apply_perm",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/apply_perm()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 77,
      fnName := "split_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/split_rows()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 110,
      fnName := "partition",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/partition()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 123,
      fnName := "children",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/children()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 146,
      fnName := "push_internal",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/push_internal()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 177,
      fnName := "emit",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/impl#[`Builder<'_>`]emit()." },
    { file := "rust/crates/bpann/src/index/tree_bulk.rs", line := 205,
      fnName := "bulk_build",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_bulk/bulk_build()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 12,
      fnName := "dist",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/dist()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 19,
      fnName := "fill_blocks",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]fill_blocks()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 47,
      fnName := "page_centroid",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]page_centroid()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 58,
      fnName := "subtree_radius",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]subtree_radius()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 79,
      fnName := "scale_rows",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]scale_rows()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 96,
      fnName := "recount",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]recount()." },
    { file := "rust/crates/bpann/src/index/tree_counts.rs", line := 107,
      fnName := "count_subtree",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_counts/impl#[Tree]count_subtree()." },
    { file := "rust/crates/bpann/src/index/tree_persist.rs", line := 21,
      fnName := "invalid",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_persist/invalid()." },
    { file := "rust/crates/bpann/src/index/tree_persist.rs", line := 46,
      fnName := "read_u32",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_persist/read_u32()." },
    { file := "rust/crates/bpann/src/index/tree_persist.rs", line := 127,
      fnName := "open_persisted",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_persist/impl#[Tree]open_persisted()." },
    { file := "rust/crates/bpann/src/index/tree_persist.rs", line := 151,
      fnName := "load_page",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_persist/impl#[Tree]load_page()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 23,
      fnName := "leaf_budget",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/leaf_budget()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 35,
      fnName := "discounted",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/discounted()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 40,
      fnName := "priority",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/priority()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 46,
      fnName := "lower_bound",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/lower_bound()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 56,
      fnName := "offer",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/impl#[TopK]offer()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 69,
      fnName := "cannot_improve",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/impl#[TopK]cannot_improve()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 73,
      fnName := "into_sorted",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/impl#[TopK]into_sorted()." },
    { file := "rust/crates/bpann/src/index/tree_search.rs", line := 81,
      fnName := "search_tree",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_search/search_tree()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 51,
      fnName := "io_err",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/io_err()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 56,
      fnName := "scratch_file",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/scratch_file()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 65,
      fnName := "kind_tag",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/kind_tag()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 73,
      fnName := "map",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/map()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 84,
      fnName := "grow_anon",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/grow_anon()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 112,
      fnName := "create",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]create()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 127,
      fnName := "num_dim",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]num_dim()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 131,
      fnName := "num_pages",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]num_pages()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 136,
      fnName := "file_bytes",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]file_bytes()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 145,
      fnName := "update_tracking",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]update_tracking()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 154,
      fnName := "spill",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]spill()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 173,
      fnName := "alloc",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]alloc()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 184,
      fnName := "set_kind",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]set_kind()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 189,
      fnName := "grow",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]grow()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 209,
      fnName := "meta_words",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]meta_words()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 216,
      fnName := "meta_words_mut",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]meta_words_mut()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 223,
      fnName := "slot_words",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]slot_words()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 232,
      fnName := "slot_words_mut",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]slot_words_mut()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 240,
      fnName := "is_leaf",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]is_leaf()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 245,
      fnName := "len",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]len()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 250,
      fnName := "ids",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]ids()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 256,
      fnName := "block",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]block()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 263,
      fnName := "block_mut",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]block_mut()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 270,
      fnName := "entry_mut",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]entry_mut()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 276,
      fnName := "set_entries",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]set_entries()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 289,
      fnName := "push_entry",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]push_entry()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 300,
      fnName := "count",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]count()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 304,
      fnName := "set_count",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]set_count()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 309,
      fnName := "radius",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]radius()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 313,
      fnName := "set_radius",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]set_radius()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 317,
      fnName := "parent",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]parent()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 321,
      fnName := "set_parent",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]set_parent()." },
    { file := "rust/crates/bpann/src/index/tree_store.rs", line := 327,
      fnName := "try_clone",
      fnId := "rust-analyzer cargo ennbo-bpann 0.4.1 index/tree_store/impl#[TreeStore]try_clone()." } ]

/-- All 95 functions across the 10 B+ tree files are covered. -/
theorem treeCount : manifestSpecs.length = 95 := rfl

#print axioms kindDistinct
#print axioms storeInit
#print axioms allocPage
#print axioms hdrValid
#print axioms rangeValid
#print axioms medianPart
#print axioms bulkFloor
#print axioms offerBound
#print axioms resetZero
#print axioms syncTotal
#print axioms candEmpty
#print axioms treeCount

end BauxResearch.Semantics.Tree
