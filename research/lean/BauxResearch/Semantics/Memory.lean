import BauxResearch.Semantics.Partition

/-!
# Packed stores across leaves and tiles

The host accepts contiguous leaves and allocates two bytes per parameter.
Prefix sums model those checked offsets. This module derives global coverage
and uniqueness of scalar byte ownership from the complete leaf and tile
partition, rather than assuming nonoverlap between different tiles or leaves.
It does not identify host buffer objects, validate aliasing of kernel arguments,
or prove shader/native semantics and synchronization.
-/

namespace BauxResearch.Semantics.Memory

/-- Offset of a leaf in a contiguous allocation. -/
def offset (lengths : List ℕ) (leaf : ℕ) : ℕ := (lengths.take leaf).sum

/-- Length of one leaf, with zero making out-of-range leaves uninhabited. -/
def size (lengths : List ℕ) (leaf : ℕ) : ℕ := lengths.getD leaf 0

/-- Contiguous leaf construction covers every element of its allocation. -/
theorem leafCover (lengths : List ℕ) (element : ℕ) (inside : element < lengths.sum) :
    ∃ leaf item, leaf < lengths.length ∧ item < size lengths leaf ∧
      offset lengths leaf + item = element := by
  induction lengths generalizing element with
  | nil => simp at inside
  | cons head tail ih =>
    by_cases first : element < head
    · exact ⟨0, element, by simp, by simpa [size] using first, by simp [offset]⟩
    · have tailInside : element - head < tail.sum := by simp only [List.sum_cons] at inside; omega
      obtain ⟨leaf, item, leafBound, active, same⟩ := ih (element - head) tailInside
      refine ⟨leaf + 1, item, by simp; omega, by simpa [size] using active, ?_⟩
      simp only [offset, List.take_succ_cons, List.sum_cons] at same ⊢
      omega

/-- Contiguous leaf construction cannot assign one element to two leaves. -/
theorem leafUnique (lengths : List ℕ) (first second a b : ℕ)
    (left : a < size lengths first) (right : b < size lengths second)
    (same : offset lengths first + a = offset lengths second + b) :
    first = second ∧ a = b := by
  induction lengths generalizing first second a b with
  | nil => simp [size] at left
  | cons head tail ih =>
    cases first with
    | zero =>
      cases second with
      | zero => simpa [offset] using same
      | succ second =>
        simp only [size, List.getD_cons_zero] at left
        simp only [offset, List.take_zero, List.sum_nil, zero_add,
          List.take_succ_cons, List.sum_cons] at same
        omega
    | succ first =>
      cases second with
      | zero =>
        simp only [size, List.getD_cons_zero] at right
        simp only [offset, List.take_zero, List.sum_nil, zero_add,
          List.take_succ_cons, List.sum_cons] at same
        omega
      | succ second =>
        simp only [size, List.getD_cons_succ] at left right
        have tailSame : offset tail first + a = offset tail second + b := by
          simp only [offset, List.take_succ_cons, List.sum_cons] at same ⊢
          omega
        obtain ⟨leaves, items⟩ := ih first second a b left right tailSame
        exact ⟨by omega, items⟩

/-- Every active leaf address lies inside the whole allocation. -/
theorem leafBound (lengths : List ℕ) (leaf item : ℕ) (active : item < size lengths leaf) :
    offset lengths leaf + item < lengths.sum := by
  induction lengths generalizing leaf with
  | nil => simp [size] at active
  | cons head tail ih =>
    cases leaf with
    | zero => simp [size] at active; simp [offset]; omega
    | succ leaf =>
      have bound := ih leaf (by simpa [size] using active)
      simpa [offset, Nat.add_assoc] using Nat.add_lt_add_left bound head

/-- Full identity of a byte written by a packed parameter kernel. -/
structure Access where
  leaf : ℕ
  tile : ℕ
  lane : ℕ
  iteration : ℕ
  component : ℕ
  byte : ℕ

def index (access : Access) : ℕ :=
  Dispatch.packed 256 access.lane access.iteration access.component

def element (lengths : List ℕ) (access : Access) : ℕ :=
  offset lengths access.leaf + Partition.start access.tile + index access

def address (lengths : List ℕ) (access : Access) : ℕ := 2 * element lengths access + access.byte

/-- Valid packed stores include both the loop guard and the selected tail guard. -/
def Valid (lengths : List ℕ) (access : Access) : Prop :=
  access.lane < 256 ∧ access.component < 2 ∧ access.byte < 2 ∧
    Dispatch.packed 256 access.lane access.iteration 0 <
      Partition.extent (size lengths access.leaf) access.tile ∧
    index access < Partition.extent (size lengths access.leaf) access.tile

/-- Every byte of the parameter allocation has an active packed-store owner. -/
theorem cover (lengths : List ℕ) (location : ℕ) (inside : location < 2 * lengths.sum) :
    ∃ access, Valid lengths access ∧ address lengths access = location := by
  have elementInside : location / 2 < lengths.sum := by omega
  obtain ⟨leaf, item, _, active, leafEq⟩ := leafCover lengths (location / 2) elementInside
  obtain ⟨tile, lane, iteration, component, laneBound, componentBound, pairActive,
    localActive, localEq⟩ := Partition.packedCover (size lengths leaf) item active
  refine ⟨⟨leaf, tile, lane, iteration, component, location % 2⟩, ?_, ?_⟩
  · exact ⟨laneBound, componentBound, Nat.mod_lt _ (by decide), pairActive, localActive⟩
  · unfold address element index
    dsimp
    have split := Nat.mod_add_div location 2
    omega

/-- Every active store's byte address is below the allocated byte count. -/
theorem addressBound (lengths : List ℕ) (access : Access) (valid : Valid lengths access) :
    address lengths access < 2 * lengths.sum := by
  have localBound := (Partition.localBound (size lengths access.leaf) access.tile
    (index access) valid.2.2.2.2).2
  have globalBound := leafBound lengths access.leaf _ localBound
  unfold address element
  have byteBound := valid.2.2.1
  omega

/-- All store identity fields are determined by the global byte address.
Consequently different leaves, tiles, lanes, iterations or packed components
cannot overlap even at the byte level within this one allocation. -/
theorem unique (lengths : List ℕ) (left right : Access)
    (leftValid : Valid lengths left) (rightValid : Valid lengths right)
    (same : address lengths left = address lengths right) : left = right := by
  have elementEq : element lengths left = element lengths right := by
    unfold address at same
    have a := leftValid.2.2.1
    have b := rightValid.2.2.1
    omega
  have byteEq : left.byte = right.byte := by unfold address at same; omega
  have leftBound := (Partition.localBound (size lengths left.leaf) left.tile
    (index left) leftValid.2.2.2.2).2
  have rightBound := (Partition.localBound (size lengths right.leaf) right.tile
    (index right) rightValid.2.2.2.2).2
  obtain ⟨leafEq, offsetEq⟩ := leafUnique lengths left.leaf right.leaf
    (Partition.start left.tile + index left) (Partition.start right.tile + index right)
    leftBound rightBound (by simpa [element, Nat.add_assoc] using elementEq)
  have rightActive : index right < Partition.extent (size lengths left.leaf) right.tile := by
    simpa [leafEq] using rightValid.2.2.2.2
  obtain ⟨tileEq, laneEq, iterationEq, componentEq⟩ :=
    Partition.packedUnique (size lengths left.leaf) left.tile right.tile left.lane right.lane
      left.iteration right.iteration left.component right.component leftValid.1 rightValid.1
      leftValid.2.1 rightValid.2.1 leftValid.2.2.2.2 rightActive offsetEq
  cases left
  cases right
  simp_all

/-- The vector fast path's complete-pair guard keeps all four bytes in the allocation. -/
theorem pairRange (lengths : List ℕ) (leaf tile lane iteration : ℕ)
    (complete : Dispatch.packed 256 lane iteration 0 + 1 <
      Partition.extent (size lengths leaf) tile) :
    2 * (offset lengths leaf + Partition.start tile +
      Dispatch.packed 256 lane iteration 0) + 4 ≤ 2 * lengths.sum := by
  have tileBound := (Partition.localBound (size lengths leaf) tile _ complete).2
  have bound := leafBound lengths leaf _ tileBound
  omega

/-- An even element offset gives a four-byte-aligned relative byte offset.
Actual `ushort2` pointer alignment also requires an aligned allocation base.
The shader tests this guard; an odd leaf offset uses scalar loads/stores instead. -/
theorem pairAligned (index : ℕ) (aligned : index % 2 = 0) : (2 * index) % 4 = 0 := by
  omega

/-- For a uint64-sized allocation the full byte address itself does not truncate. -/
theorem machineBound (lengths : List ℕ) (access : Access)
    (allocated : 2 * lengths.sum ≤ 2 ^ 64) (valid : Valid lengths access) :
    (BitVec.ofNat 64 (address lengths access)).toNat = address lengths access := by
  exact Nat.mod_eq_of_lt (lt_of_lt_of_le (addressBound lengths access valid) allocated)

/-- Every intermediate integer address operation is below the uint64 bound.
This includes the leaf offset, tile addition, local addition, byte scaling and byte addition. -/
theorem stages (lengths : List ℕ) (access : Access)
    (allocated : 2 * lengths.sum ≤ 2 ^ 64) (valid : Valid lengths access) :
    offset lengths access.leaf < 2 ^ 64 ∧
      offset lengths access.leaf + Partition.start access.tile < 2 ^ 64 ∧
      element lengths access < 2 ^ 64 ∧
      2 * element lengths access < 2 ^ 64 ∧ address lengths access < 2 ^ 64 := by
  have bound := addressBound lengths access valid
  unfold address element at *
  omega

/-- Sequential uint64 additions and byte scaling compute exactly the proved address.
This is integer offset arithmetic, not a claim about GPU pointer provenance. -/
theorem machineAddress (lengths : List ℕ) (access : Access)
    (allocated : 2 * lengths.sum ≤ 2 ^ 64) (valid : Valid lengths access) :
    ((BitVec.ofNat 64 2) *
      (BitVec.ofNat 64 (offset lengths access.leaf) +
        BitVec.ofNat 64 (Partition.start access.tile) + BitVec.ofNat 64 (index access)) +
      BitVec.ofNat 64 access.byte).toNat = address lengths access := by
  rw [← BitVec.ofNat_add, ← BitVec.ofNat_add, ← BitVec.ofNat_mul, ← BitVec.ofNat_add]
  exact machineBound lengths access allocated valid

#print axioms leafCover
#print axioms leafUnique
#print axioms cover
#print axioms unique
#print axioms machineBound
#print axioms machineAddress

end BauxResearch.Semantics.Memory
