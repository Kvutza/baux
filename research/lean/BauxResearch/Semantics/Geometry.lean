import BauxResearch.Semantics.Partition

/-!
# Extracted host tile origins

The production layout uses `dispatch::tile_start` and `dispatch::tile_len`.
Their extracted results implement the complete finite geometry proved in
`Partition`. The enclosing Rust allocation/loop and device launches are not
extracted here; acceptance, iteration and pointer obligations stay explicit.
-/

namespace BauxResearch.Semantics.Geometry

open Aeneas.Std
open BauxResearch.Generated.Optimizer

def word (ordinal : U32) : U32 := ⟨BitVec.ofNat 32 (Partition.start ordinal.val)⟩

private theorem capVal : dispatch.TILE_ELEMENTS.val = 65536 := by
  unfold dispatch.TILE_ELEMENTS
  rfl

theorem wordVal (ordinal : U32) (bound : ordinal.val < 65536) :
    (word ordinal).val = Partition.start ordinal.val := by
  change Partition.start ordinal.val % 2 ^ 32 = _
  apply Nat.mod_eq_of_lt
  unfold Partition.start
  omega

/-- Exact start computation, including its ordinal and leaf rejection guards. -/
theorem startMatch (leaf ordinal : U32) :
    dispatch.tile_start leaf ordinal = Result.ok
      (if ordinal.val < 65536 ∧ Partition.start ordinal.val < leaf.val then
        some (word ordinal) else none) := by
  by_cases bound : ordinal.val < 65536
  · have outside : ¬65536 ≤ ordinal.val := by omega
    have product : ordinal.val * 65536 < 2 ^ 32 := by omega
    have multiply : UScalar.mul ordinal dispatch.TILE_ELEMENTS = Result.ok (word ordinal) := by
      simp only [UScalar.mul, capVal, UScalar.tryMk, UScalar.tryMkOpt,
        UScalar.check_bounds, UScalarTy.numBits, product, decide_true, ↓reduceDIte,
        Result.ofOption]
      apply congrArg Result.ok
      apply UScalar.val_eq_imp
      simpa [Partition.start] using (wordVal ordinal bound).symm
    simp only [dispatch.tile_start, UScalar.le_equiv, capVal, outside, ↓reduceIte]
    change Aeneas.Std.bind (UScalar.mul ordinal dispatch.TILE_ELEMENTS)
      (fun start ↦ if start < leaf then Result.ok (some start) else Result.ok none) = _
    rw [multiply, bind_ok]
    simp only [UScalar.lt_equiv, wordVal ordinal bound, bound, true_and]
    split <;> rfl
  · have outside : 65536 ≤ ordinal.val := by omega
    simp [dispatch.tile_start, capVal, outside, bound]

/-- Accepted origins are in range and even; they are not arbitrary caller offsets. -/
theorem startBounds (leaf ordinal value : U32)
    (accepted : dispatch.tile_start leaf ordinal = Result.ok (some value)) :
    ordinal.val < 65536 ∧ value.val = Partition.start ordinal.val ∧
      value.val < leaf.val ∧ value.val % 2 = 0 := by
  rw [startMatch] at accepted
  split at accepted
  · rename_i guard
    have same : word ordinal = value := by simpa using accepted
    subst value
    rw [wordVal ordinal guard.1]
    exact ⟨guard.1, rfl, guard.2, Partition.startEven _⟩
  · simp at accepted

/-- Every successful production start/length pair has precisely the proved geometry. -/
theorem tileGeometry (leaf ordinal origin length : U32)
    (started : dispatch.tile_start leaf ordinal = Result.ok (some origin))
    (sized : dispatch.tile_len leaf origin = Result.ok (some length)) :
    origin.val = Partition.start ordinal.val ∧
      length.val = Partition.extent leaf.val ordinal.val := by
  obtain ⟨_, originEq, inside, _⟩ := startBounds leaf ordinal origin started
  rw [Tile.tileMatch, ite_eq_left inside] at sized
  have same : Tile.length leaf origin = length := by simpa using sized
  subst length
  exact ⟨originEq, by simp [Tile.lengthVal, Partition.extent, originEq]⟩

/-- Every leaf element is reached by an accepted extracted start/length pair. -/
theorem cover (leaf : U32) (element : ℕ) (inside : element < leaf.val) :
    ∃ ordinal origin length : U32,
      dispatch.tile_start leaf ordinal = Result.ok (some origin) ∧
      dispatch.tile_len leaf origin = Result.ok (some length) ∧
      element % 65536 < length.val ∧ origin.val + element % 65536 = element := by
  obtain ⟨active, equal⟩ := Partition.cover leaf.val element inside
  have leafBound : leaf.val < 2 ^ 32 := leaf.hBounds
  have bound := Partition.ordinalBound leaf.val (element / 65536) (element % 65536)
    leafBound active
  let ordinal : U32 := ⟨BitVec.ofNat 32 (element / 65536)⟩
  have ordinalVal : ordinal.val = element / 65536 := by
    change (element / 65536) % 2 ^ 32 = _
    apply Nat.mod_eq_of_lt
    omega
  have ordinalBound : ordinal.val < 65536 := by omega
  have originVal := wordVal ordinal ordinalBound
  rw [ordinalVal] at originVal
  have originInside : (word ordinal).val < leaf.val := by omega
  refine ⟨ordinal, word ordinal, Tile.length leaf (word ordinal), ?_, ?_, ?_, ?_⟩
  · rw [startMatch]
    have startInside : Partition.start ordinal.val < leaf.val := by
      rw [ordinalVal, ← originVal]
      exact originInside
    simp [ordinalBound, startInside]
  · rw [Tile.tileMatch]
    simp [originInside]
  · rw [Tile.lengthVal]
    simpa [Partition.extent, ← originVal] using active
  · omega

/-- The host's increment after an accepted ordinal, including its final iteration, fits uint32. -/
theorem nextBound (leaf ordinal origin : U32)
    (started : dispatch.tile_start leaf ordinal = Result.ok (some origin)) :
    (ordinal.bv + BitVec.ofNat 32 1).toNat = ordinal.val + 1 := by
  have bound := (startBounds leaf ordinal origin started).1
  have safe := Dispatch.machineStep 32 65536 1 ordinal.val (by decide) bound
  simpa using safe

/-- Two accepted production tiles cannot overlap unless their ordinals agree. -/
theorem unique (leaf a b first second left right : U32) (i j : ℕ)
    (firstStart : dispatch.tile_start leaf a = Result.ok (some first))
    (secondStart : dispatch.tile_start leaf b = Result.ok (some second))
    (firstLength : dispatch.tile_len leaf first = Result.ok (some left))
    (secondLength : dispatch.tile_len leaf second = Result.ok (some right))
    (firstActive : i < left.val) (secondActive : j < right.val)
    (same : first.val + i = second.val + j) : a = b ∧ i = j := by
  obtain ⟨firstEq, leftEq⟩ := tileGeometry leaf a first left firstStart firstLength
  obtain ⟨secondEq, rightEq⟩ := tileGeometry leaf b second right secondStart secondLength
  obtain ⟨ordinals, items⟩ := Partition.unique leaf.val a.val b.val i j
    (by omega) (by omega) (by omega)
  exact ⟨UScalar.val_eq_imp a b ordinals, items⟩

#print axioms startMatch
#print axioms tileGeometry
#print axioms cover
#print axioms unique

end BauxResearch.Semantics.Geometry
