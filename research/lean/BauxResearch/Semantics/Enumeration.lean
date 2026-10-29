import BauxResearch.Semantics.Geometry

/-!
# Production tile enumeration

This module refines the extracted Rust `tiles` loop to its complete ordered
list of tiles. Termination and successful result are proved for every uint32 leaf;
physical allocator failure remains outside Aeneas's abstract vector model.
-/

namespace BauxResearch.Semantics.Enumeration

open Aeneas Aeneas.Std
open BauxResearch.Generated.Optimizer
open alloc.vec

/-- Number of nonempty 65536-element tiles, including a truncated final tile. -/
def count (leaf : U32) : ℕ := (leaf.val + 65535) / 65536

theorem countBound (leaf : U32) : count leaf ≤ 65536 := by
  have bound : leaf.val < 2 ^ 32 := leaf.hBounds
  unfold count
  omega

theorem active (leaf ordinal : U32) :
    (ordinal.val < 65536 ∧ Partition.start ordinal.val < leaf.val) ↔
      ordinal.val < count leaf := by
  have bound := countBound leaf
  unfold count Partition.start at *
  omega

/-- The exact machine words stored for the given mathematical ordinal. -/
def pair (leaf : U32) (ordinal : ℕ) : U32 × U32 :=
  let origin : U32 := ⟨BitVec.ofNat 32 (Partition.start ordinal)⟩
  (origin, Tile.length leaf origin)

/-- Remaining ordered tile records from an ordinal up to the final tile. -/
def suffix (leaf : U32) (ordinal : ℕ) : List (U32 × U32) :=
  (List.range' ordinal (count leaf - ordinal)).map (pair leaf)

theorem suffixStep (leaf : U32) (ordinal : ℕ) (inside : ordinal < count leaf) :
    suffix leaf ordinal = pair leaf ordinal :: suffix leaf (ordinal + 1) := by
  have remaining : count leaf - ordinal = (count leaf - (ordinal + 1)) + 1 := by omega
  unfold suffix
  rw [remaining, List.range'_succ, List.map_cons]

def next (ordinal : U32) : U32 := ⟨BitVec.ofNat 32 (ordinal.val + 1)⟩

theorem nextVal (ordinal : U32) (bound : ordinal.val < 65536) :
    (next ordinal).val = ordinal.val + 1 := by
  change (ordinal.val + 1) % 2 ^ 32 = _
  apply Nat.mod_eq_of_lt
  omega

theorem nextMatch (ordinal : U32) (bound : ordinal.val < 65536) :
    UScalar.add ordinal 1#u32 = Result.ok (next ordinal) := by
  have safe : ordinal.val + 1 < 2 ^ 32 := by omega
  have one : (1#u32).val = 1 := rfl
  simp only [UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, UScalar.check_bounds,
    UScalarTy.numBits, one, safe, decide_true, ↓reduceDIte, Result.ofOption]
  apply congrArg Result.ok
  apply UScalar.val_eq_imp
  simpa using (nextVal ordinal bound).symm

theorem pushMatch (tiles : Vec (U32 × U32)) (value : U32 × U32)
    (bound : tiles.val.length < 65536) :
    ∃ result, Vec.push tiles value = Result.ok result ∧ result.val = tiles.val ++ [value] := by
  have small : tiles.val.length + 1 ≤ U32.max := by norm_num [U32.max, U32.numBits]; omega
  simp only [Vec.push, small, decide_true, Bool.true_or, ↓reduceDIte]
  exact ⟨_, rfl, by simp⟩

theorem bodyStep (leaf ordinal : U32) (tiles : Vec (U32 × U32))
    (inside : ordinal.val < count leaf) (size : tiles.val.length = ordinal.val) :
    ∃ result, dispatch.tiles_loop.body leaf ordinal tiles =
        Result.ok (ControlFlow.cont (next ordinal, result)) ∧
      result.val = tiles.val ++ [pair leaf ordinal.val] := by
  have guard := (active leaf ordinal).mpr inside
  have originInside : (Geometry.word ordinal).val < leaf.val := by
    rw [Geometry.wordVal ordinal guard.1]
    exact guard.2
  obtain ⟨result, pushed, value⟩ := pushMatch tiles (pair leaf ordinal.val) (by omega)
  refine ⟨result, ?_, value⟩
  have increment := nextMatch ordinal guard.1
  change Vec.push tiles (Geometry.word ordinal, Tile.length leaf (Geometry.word ordinal)) =
    Result.ok result at pushed
  simp [dispatch.tiles_loop.body, Geometry.startMatch, guard, Tile.tileMatch,
    originInside, pushed]
  change Aeneas.Std.bind (UScalar.add ordinal 1#u32)
    (fun value ↦ Result.ok (ControlFlow.cont (value, result))) = _
  rw [increment, bind_ok]

theorem bodyDone (leaf ordinal : U32) (tiles : Vec (U32 × U32))
    (outside : count leaf ≤ ordinal.val) :
    dispatch.tiles_loop.body leaf ordinal tiles =
      Result.ok (ControlFlow.done (some tiles)) := by
  have guard : ¬(ordinal.val < 65536 ∧ Partition.start ordinal.val < leaf.val) := by
    rw [active]
    omega
  simp [dispatch.tiles_loop.body, Geometry.startMatch, guard]

/-- Full extracted-loop refinement, with the remaining tile count as termination measure. -/
theorem loopMatch (leaf ordinal : U32) (tiles : Vec (U32 × U32))
    (bound : ordinal.val ≤ count leaf) (size : tiles.val.length = ordinal.val) :
    ∃ result, dispatch.tiles_loop ordinal leaf tiles = Result.ok (some result) ∧
      result.val = tiles.val ++ suffix leaf ordinal.val := by
  by_cases inside : ordinal.val < count leaf
  · obtain ⟨appended, step, appendedVal⟩ := bodyStep leaf ordinal tiles inside size
    have ordinalBound : ordinal.val < 65536 := (active leaf ordinal).mpr inside |>.1
    have nextValue := nextVal ordinal ordinalBound
    obtain ⟨result, finished, resultVal⟩ := loopMatch leaf (next ordinal) appended
      (by omega) (by simp [appendedVal, size, nextValue])
    refine ⟨result, ?_, ?_⟩
    · unfold dispatch.tiles_loop
      rw [loop]
      change Aeneas.Std.bind (dispatch.tiles_loop.body leaf ordinal tiles) _ = _
      rw [step, bind_ok]
      exact finished
    · rw [resultVal, appendedVal, nextValue, suffixStep leaf ordinal.val inside]
      simp [List.append_assoc]
  · refine ⟨tiles, ?_, ?_⟩
    · unfold dispatch.tiles_loop
      rw [loop]
      change Aeneas.Std.bind (dispatch.tiles_loop.body leaf ordinal tiles) _ = _
      rw [bodyDone leaf ordinal tiles (by omega), bind_ok]
    · simp [suffix, Nat.sub_eq_zero_of_le (by omega : count leaf ≤ ordinal.val)]
termination_by count leaf - ordinal.val
decreasing_by omega

/-- The actual production helper terminates successfully and returns precisely the ordered tiles. -/
theorem layoutMatch (leaf : U32) :
    ∃ result, dispatch.tiles leaf = Result.ok (some result) ∧
      result.val = suffix leaf 0 := by
  obtain ⟨result, finished, values⟩ := loopMatch leaf 0#u32 (Vec.new _) (by simp) (by simp)
  exact ⟨result, finished, by simpa using values⟩

theorem layoutValues (leaf : U32) (result : Vec (U32 × U32))
    (accepted : dispatch.tiles leaf = Result.ok (some result)) :
    result.val = suffix leaf 0 := by
  obtain ⟨expected, computed, values⟩ := layoutMatch leaf
  have same : result = expected := by simpa using accepted.symm.trans computed
  simpa [same] using values

theorem layoutLength (leaf : U32) (result : Vec (U32 × U32))
    (accepted : dispatch.tiles leaf = Result.ok (some result)) :
    result.val.length = count leaf := by
  simp [layoutValues leaf result accepted, suffix]

/-- Successful output preserves ordinal order and contains exactly one entry per ordinal. -/
theorem entryAt (leaf : U32) (result : Vec (U32 × U32)) (ordinal : ℕ)
    (accepted : dispatch.tiles leaf = Result.ok (some result)) (inside : ordinal < count leaf) :
    result.val[ordinal]? = some (pair leaf ordinal) := by
  rw [layoutValues leaf result accepted]
  simp [suffix, List.getElem?_range' inside]

theorem pairValues (leaf : U32) (ordinal : ℕ) (inside : ordinal < count leaf) :
    (pair leaf ordinal).1.val = Partition.start ordinal ∧
      (pair leaf ordinal).2.val = Partition.extent leaf.val ordinal := by
  have bound := countBound leaf
  have safe : Partition.start ordinal < 2 ^ 32 := by unfold Partition.start; omega
  have origin : (⟨BitVec.ofNat 32 (Partition.start ordinal)⟩ : U32).val =
      Partition.start ordinal := Nat.mod_eq_of_lt safe
  simp [pair, Tile.lengthVal, origin, Partition.extent]

/-- Distinct ordinal positions have strictly ordered origins, hence no duplicate tiles. -/
theorem ordered (leaf : U32) (first second : ℕ) (order : first < second)
    (inside : second < count leaf) :
    (pair leaf first).1.val < (pair leaf second).1.val := by
  rw [(pairValues leaf first (by omega)).1, (pairValues leaf second inside).1]
  unfold Partition.start
  omega

/-- Consecutive output tiles meet exactly; there is no skipped or repeated element. -/
theorem adjacent (leaf : U32) (ordinal : ℕ) (inside : ordinal + 1 < count leaf) :
    (pair leaf ordinal).1.val + (pair leaf ordinal).2.val =
      (pair leaf (ordinal + 1)).1.val := by
  obtain ⟨origin, length⟩ := pairValues leaf ordinal (by omega)
  obtain ⟨nextOrigin, _⟩ := pairValues leaf (ordinal + 1) inside
  rw [origin, length, nextOrigin]
  unfold Partition.extent Partition.start count at *
  omega

/-- The final returned tile ends at exactly the leaf length. -/
theorem finalEnd (leaf : U32) (nonempty : 0 < count leaf) :
    (pair leaf (count leaf - 1)).1.val + (pair leaf (count leaf - 1)).2.val = leaf.val := by
  obtain ⟨origin, length⟩ := pairValues leaf (count leaf - 1) (by omega)
  rw [origin, length]
  unfold Partition.extent Partition.start count at *
  omega

/-- Every leaf element occurs inside an actual returned tile. -/
theorem cover (leaf : U32) (result : Vec (U32 × U32)) (element : ℕ)
    (accepted : dispatch.tiles leaf = Result.ok (some result)) (inside : element < leaf.val) :
    ∃ origin length : U32, (origin, length) ∈ result.val ∧
      element % 65536 < length.val ∧ origin.val + element % 65536 = element := by
  have ordinalBound : element / 65536 < count leaf := by unfold count; omega
  have index := entryAt leaf result (element / 65536) accepted ordinalBound
  have member : pair leaf (element / 65536) ∈ result.val := List.mem_of_getElem? index
  obtain ⟨origin, length⟩ := pairValues leaf (element / 65536) ordinalBound
  obtain ⟨active, equal⟩ := Partition.cover leaf.val element inside
  exact ⟨(pair leaf (element / 65536)).1, (pair leaf (element / 65536)).2,
    member, by omega, by omega⟩

#print axioms layoutMatch
#print axioms cover
#print axioms adjacent

end BauxResearch.Semantics.Enumeration
