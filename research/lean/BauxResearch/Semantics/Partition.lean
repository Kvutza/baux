import BauxResearch.Semantics.Tile

/-!
# Complete fixed-width tile partition

These definitions describe the 65536-element geometry used by the host layout
constructor. Theorems connect this geometry to scalar/packed ownership. The
connection to the extracted host start and length functions is stated separately;
this module does not assume that different tiles are disjoint.
-/

namespace BauxResearch.Semantics.Partition

/-- Element offset of one host tile within its leaf. -/
def start (tile : ℕ) : ℕ := tile * 65536

/-- Extent of a host tile, including its truncated final tile. -/
def extent (length tile : ℕ) : ℕ := min 65536 (length - start tile)

theorem startEven (tile : ℕ) : start tile % 2 = 0 := by simp [start, Nat.mul_mod]

theorem localBound (length tile item : ℕ) (active : item < extent length tile) :
    item < 65536 ∧ start tile + item < length := by
  unfold extent at active
  omega

/-- Every leaf element belongs to its quotient-indexed tile. -/
theorem cover (length element : ℕ) (inside : element < length) :
    element % 65536 < extent length (element / 65536) ∧
      start (element / 65536) + element % 65536 = element := by
  have remainder := Nat.mod_lt element (by decide : 0 < 65536)
  have split := Nat.mod_add_div element 65536
  unfold extent start
  omega

/-- Earlier tile intervals end before any element of a later tile. -/
theorem separated (length first second item : ℕ) (ordered : first < second)
    (active : item < extent length first) :
    start first + item < start second := by
  have bound := (localBound length first item active).1
  unfold start
  omega

/-- Global leaf offsets recover both the tile and its local element uniquely. -/
theorem unique (length first second a b : ℕ)
    (left : a < extent length first) (right : b < extent length second)
    (same : start first + a = start second + b) : first = second ∧ a = b := by
  have leftBound := (localBound length first a left).1
  have rightBound := (localBound length second b right).1
  unfold start at same
  omega

/-- An active tile has a representable ordinal whenever its leaf fits uint32. -/
theorem ordinalBound (length tile item : ℕ) (leafBound : length < 2 ^ 32)
    (active : item < extent length tile) : tile < 65536 := by
  have bound := (localBound length tile item active).2
  unfold start at bound
  norm_num at leafBound
  omega

/-- Complete leaf coverage by the actual 256-lane scalar geometry. -/
theorem scalarCover (length element : ℕ) (inside : element < length) :
    ∃ tile lane iteration, lane < 256 ∧
      Dispatch.scalar 256 lane iteration < extent length tile ∧
      start tile + Dispatch.scalar 256 lane iteration = element := by
  obtain ⟨insideTile, elementEq⟩ := cover length element inside
  obtain ⟨lane, iteration, laneBound, active, localEq⟩ :=
    Dispatch.scalarCover 256 (extent length (element / 65536)) (element % 65536)
      (by decide) insideTile
  exact ⟨element / 65536, lane, iteration, laneBound, active, by omega⟩

/-- Complete leaf coverage by packed accesses, including a final unpaired element. -/
theorem packedCover (length element : ℕ) (inside : element < length) :
    ∃ tile lane iteration component, lane < 256 ∧ component < 2 ∧
      Dispatch.packed 256 lane iteration 0 < extent length tile ∧
      Dispatch.packed 256 lane iteration component < extent length tile ∧
      start tile + Dispatch.packed 256 lane iteration component = element := by
  obtain ⟨insideTile, elementEq⟩ := cover length element inside
  obtain ⟨lane, iteration, component, laneBound, componentBound, active, localEq⟩ :=
    Dispatch.packedCover 256 (extent length (element / 65536)) (element % 65536)
      (by decide) insideTile
  exact ⟨element / 65536, lane, iteration, component, laneBound, componentBound,
    active, by omega, by omega⟩

/-- Packed stores from different tiles or lanes cannot target the same element. -/
theorem packedUnique (length first second left right a b x y : ℕ)
    (leftLane : left < 256) (rightLane : right < 256)
    (leftPart : x < 2) (rightPart : y < 2)
    (leftActive : Dispatch.packed 256 left a x < extent length first)
    (rightActive : Dispatch.packed 256 right b y < extent length second)
    (same : start first + Dispatch.packed 256 left a x =
      start second + Dispatch.packed 256 right b y) :
    first = second ∧ left = right ∧ a = b ∧ x = y := by
  obtain ⟨tileEq, elementEq⟩ := unique length first second _ _ leftActive rightActive same
  obtain ⟨laneEq, iterationEq, componentEq⟩ := Dispatch.packedUnique 256 left right a b x y
    leftLane rightLane leftPart rightPart elementEq
  exact ⟨tileEq, laneEq, iterationEq, componentEq⟩

#print axioms cover
#print axioms packedCover
#print axioms packedUnique

end BauxResearch.Semantics.Partition
