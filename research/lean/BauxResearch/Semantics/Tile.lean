import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Semantics.Dispatch

/-!
# Production tile-length bounds

The extracted Rust function constructs the length stored in each Metal tile.
Its range supplies the no-wrap premise for the scalar and packed dispatch loops.
Tile partitioning, source-to-device compilation, numerical operations and physical
allocation remain separate obligations.
-/

namespace BauxResearch.Semantics.Tile

open Aeneas.Std
open BauxResearch.Generated.Optimizer

/-- The finite mathematical tile length, encoded without truncation. -/
def length (leaf start : U32) : U32 :=
  ⟨BitVec.ofNat 32 (min 65536 (leaf.val - start.val))⟩

theorem lengthVal (leaf start : U32) :
    (length leaf start).val = min 65536 (leaf.val - start.val) := by
  change (min 65536 (leaf.val - start.val)) % 2 ^ 32 = _
  apply Nat.mod_eq_of_lt
  have := min_le_left 65536 (leaf.val - start.val)
  omega

/-- Exact correspondence for the extracted production function, including rejection. -/
theorem tileMatch (leaf start : U32) :
    dispatch.tile_len leaf start =
      Result.ok (if start.val < leaf.val then some (length leaf start) else none) := by
  by_cases active : start.val < leaf.val
  · have outside : ¬ leaf.val ≤ start.val := by omega
    have underflow : ¬ leaf.val < start.val := by omega
    have remaining : leaf.val - start.val < 2 ^ 32 := by
      have := leaf.hBounds
      change leaf.val < 2 ^ 32 at this
      omega
    simp only [dispatch.tile_len, UScalar.le_equiv, outside, ↓reduceIte]
    change Aeneas.Std.bind (UScalar.sub leaf start) (fun rest ↦
      if rest > dispatch.TILE_ELEMENTS then Result.ok (some dispatch.TILE_ELEMENTS)
      else Result.ok (some rest)) = _
    simp only [UScalar.sub, underflow, ↓reduceIte, bind_ok]
    simp only [dispatch.TILE_ELEMENTS, UScalar.lt_equiv]
    change (if 65536 < (leaf.val - start.val) % 2 ^ 32 then
      Result.ok (some (65536#u32)) else
      Result.ok (some (⟨BitVec.ofNat 32 (leaf.val - start.val)⟩ : U32))) = _
    rw [Nat.mod_eq_of_lt remaining]
    by_cases large : 65536 < leaf.val - start.val
    · simp [large, active, length, Nat.min_eq_left (by omega : 65536 ≤ leaf.val - start.val)]
      decide
    · simp [large, active, length, Nat.min_eq_right (by omega : leaf.val - start.val ≤ 65536)]
  · have outside : leaf.val ≤ start.val := by omega
    simp [dispatch.tile_len, outside, active]

/-- Every accepted tile is nonempty, within the fixed cap, and within its leaf. -/
theorem tileBounds (leaf start value : U32)
    (accepted : dispatch.tile_len leaf start = Result.ok (some value)) :
    0 < value.val ∧ value.val ≤ 65536 ∧ start.val + value.val ≤ leaf.val := by
  rw [tileMatch] at accepted
  split at accepted
  · have same : length leaf start = value := by simpa using accepted
    subst value
    rw [lengthVal]
    omega
  · simp at accepted

/-- The production bound discharges uint32 no-wrap for both 256/512-stride loops. -/
theorem tileStep (leaf start value : U32) (stride item : ℕ)
    (accepted : dispatch.tile_len leaf start = Result.ok (some value))
    (strideBound : stride ≤ 512) (active : item < value.val) :
    (BitVec.ofNat 32 item + BitVec.ofNat 32 stride).toNat = item + stride := by
  exact Dispatch.tileStep value.val stride item (tileBounds leaf start value accepted).2.1
    strideBound active

#print axioms tileMatch
#print axioms tileBounds
#print axioms tileStep

end BauxResearch.Semantics.Tile
