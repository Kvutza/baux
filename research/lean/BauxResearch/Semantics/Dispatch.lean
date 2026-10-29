import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

/-!
# Ownership of strided GPU tile accesses

The scalar and packed indexing forms below occur in `resident_search.metal`:
`item = tid; item += 256` and `item = tid * 2; item += 512`.
The host dispatches 256 threads per tile. These proofs establish indexing,
unique ownership within one tile, tail coverage, and explicit no-wrap conditions.
They do not establish the host tile partition or even tile starts, and do not
model shader compilation, physical memory, the numerical kernel body, barriers,
SIMD reductions, or emitted device instructions. The loop variant concerns the
outer indexing loop; termination of called sampling routines is separate.
-/

namespace BauxResearch.Semantics.Dispatch

/-- Index visited by one lane at one iteration of a strided loop. -/
def scalar (width lane iteration : ℕ) : ℕ := lane + width * iteration

/-- Element selected inside a two-element packed loop iteration. -/
def packed (width lane iteration component : ℕ) : ℕ :=
  2 * scalar width lane iteration + component

/-- The original lane is recovered from every scalar-loop access. -/
theorem scalarLane (width lane iteration : ℕ) (bound : lane < width) :
    scalar width lane iteration % width = lane := by
  simp [scalar, Nat.mod_eq_of_lt bound]

/-- The iteration number is recovered from every scalar-loop access. -/
theorem scalarIter (width lane iteration : ℕ) (bound : lane < width) :
    scalar width lane iteration / width = iteration := by
  have positive : 0 < width := by omega
  simp [scalar, Nat.add_mul_div_left _ _ positive, Nat.div_eq_of_lt bound]

/-- Distinct lane/iteration pairs cannot write the same scalar element. -/
theorem scalarUnique (width left right first second : ℕ)
    (leftBound : left < width) (rightBound : right < width)
    (same : scalar width left first = scalar width right second) :
    left = right ∧ first = second := by
  constructor
  · simpa only [scalarLane width left first leftBound,
      scalarLane width right second rightBound] using congrArg (· % width) same
  · simpa only [scalarIter width left first leftBound,
      scalarIter width right second rightBound] using congrArg (· / width) same

/-- Every element below the tile length has one active scalar-loop owner. -/
theorem scalarCover (width length element : ℕ) (positive : 0 < width)
    (inside : element < length) :
    ∃ lane iteration, lane < width ∧ scalar width lane iteration < length ∧
      scalar width lane iteration = element := by
  refine ⟨element % width, element / width, Nat.mod_lt _ positive, ?_, ?_⟩
  · simpa [scalar, Nat.mod_add_div] using inside
  · exact Nat.mod_add_div element width

/-- Packed coordinates agree only when lane, iteration, and component all agree. -/
theorem packedUnique (width left right first second a b : ℕ)
    (leftBound : left < width) (rightBound : right < width)
    (aBound : a < 2) (bBound : b < 2)
    (same : packed width left first a = packed width right second b) :
    left = right ∧ first = second ∧ a = b := by
  have indices : scalar width left first = scalar width right second := by
    unfold packed at same
    omega
  obtain ⟨lanes, iterations⟩ := scalarUnique width left right first second
    leftBound rightBound indices
  exact ⟨lanes, iterations, by unfold packed at same; omega⟩

/-- Every valid element is reached by a packed loop, including an odd tail.
The pair start passes the loop guard; the selected component passes its guard. -/
theorem packedCover (width length element : ℕ) (positive : 0 < width)
    (inside : element < length) :
    ∃ lane iteration component, lane < width ∧ component < 2 ∧
      packed width lane iteration 0 < length ∧
      packed width lane iteration component = element := by
  have pairEq := Nat.mod_add_div (element / 2) width
  have elementEq := Nat.mod_add_div element 2
  have componentBound := Nat.mod_lt element (by decide : 0 < 2)
  refine ⟨element / 2 % width, element / 2 / width, element % 2,
    Nat.mod_lt _ positive, componentBound, ?_, ?_⟩
  · simp only [packed, scalar, pairEq, Nat.add_zero]
    omega
  · simp only [packed, scalar, pairEq]
    omega

/-- Every guarded local access remains inside its declared enclosing allocation. -/
theorem addressBound (offset length capacity item : ℕ)
    (allocated : offset + length ≤ capacity) (inside : item < length) :
    offset + item < capacity := by omega

/-- An allocated range within the machine address space also prevents address wrap.
This is an element-address result; byte scaling and physical allocation are separate. -/
theorem machineAddress (bits offset length capacity item : ℕ)
    (allocated : offset + length ≤ capacity) (addressSpace : capacity ≤ 2 ^ bits)
    (inside : item < length) :
    (BitVec.ofNat bits offset + BitVec.ofNat bits item).toNat = offset + item := by
  have address := addressBound offset length capacity item allocated inside
  have bound : offset + item < 2 ^ bits := by omega
  have offsetBound : offset < 2 ^ bits := by omega
  have itemBound : item < 2 ^ bits := by omega
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt offsetBound, Nat.mod_eq_of_lt itemBound, Nat.mod_eq_of_lt bound]

/-- A bound on length plus stride also covers the final increment before exit. -/
theorem stepBound (bits length stride item : ℕ)
    (capacity : length + stride ≤ 2 ^ bits) (active : item < length) :
    item + stride < 2 ^ bits := by omega

/-- Every active positive-stride step strictly decreases the remaining work. -/
theorem stepProgress (length stride item : ℕ) (positive : 0 < stride)
    (active : item < length) :
    length - (item + stride) < length - item := by omega

/-- Machine addition agrees with natural addition under the loop's no-wrap bound. -/
theorem machineStep (bits length stride item : ℕ)
    (capacity : length + stride ≤ 2 ^ bits) (active : item < length) :
    (BitVec.ofNat bits item + BitVec.ofNat bits stride).toNat = item + stride := by
  have bound := stepBound bits length stride item capacity active
  have itemBound : item < 2 ^ bits := by omega
  have strideBound : stride < 2 ^ bits := by omega
  simp only [BitVec.toNat_add, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt itemBound, Nat.mod_eq_of_lt strideBound, Nat.mod_eq_of_lt bound]

/-- Every active scalar or packed increment fits uint32 for a 65536-element tile. -/
theorem tileStep (length stride item : ℕ) (tileBound : length ≤ 65536)
    (strideBound : stride ≤ 512) (active : item < length) :
    (BitVec.ofNat 32 item + BitVec.ofNat 32 stride).toNat = item + stride := by
  apply machineStep 32 length stride item
  · norm_num
    omega
  · exact active

/-- Distinct scalar-element writes commute, independently of execution order.
This models writes of already computed values, not reads from mutable inputs. -/
theorem writeCommute {Value : Type*} (state : ℕ → Value)
    (left right : ℕ) (a b : Value) (distinct : left ≠ right) :
    Function.update (Function.update state left a) right b =
      Function.update (Function.update state right b) left a := by
  exact Function.update_comm distinct a b state

/-- Merely checking `item < length` does not prevent a uint32 loop from wrapping. -/
theorem wrapExample :
    4294967040 < (4294967295 : ℕ) ∧
      (BitVec.ofNat 32 4294967040 + BitVec.ofNat 32 256).toNat = 0 := by decide

#print axioms scalarUnique
#print axioms scalarCover
#print axioms packedUnique
#print axioms packedCover
#print axioms machineStep
#print axioms writeCommute

end BauxResearch.Semantics.Dispatch
