import BauxResearch.Generated.Optimizer.Funs
import Mathlib.Tactic

/-!
# Executed host buffer guards

These are the extracted Rust checks called by Metal views, not a second
implementation of their range arithmetic. Allocation identity, base-pointer
alignment and a kernel respecting its declared view remain separate premises.
-/

namespace BauxResearch.Semantics.Buffer

open Aeneas.Std
open BauxResearch.Generated.Optimizer

theorem byteCount (rows cols width result : U64)
    (accepted : memory.view_bytes rows cols width = Result.ok (some result)) :
    result.val = rows.val * cols.val * width.val := by
  simp only [memory.view_bytes, lift, bind_ok] at accepted
  cases first : U64.checked_mul rows cols with
  | none => simp [first] at accepted
  | some elements =>
    have firstSpec := U64.checked_mul_bv_spec rows cols
    simp only [first] at firstSpec
    have secondSpec := U64.checked_mul_bv_spec elements width
    have second : U64.checked_mul elements width = some result := by
      simpa [first] using accepted
    simp only [second] at secondSpec
    rw [secondSpec.2.1, firstSpec.2.1]

def endWord (offset bytes : U64) : U64 :=
  ⟨BitVec.ofNat 64 (offset.val + bytes.val)⟩

theorem spanMatch (capacity offset bytes : U64) :
    memory.span_end capacity offset bytes = Result.ok
      (if offset.val + bytes.val ≤ capacity.val then some (endWord offset bytes) else none) := by
  by_cases outside : capacity.val < offset.val
  · have overflow : ¬offset.val + bytes.val ≤ capacity.val := by omega
    simp [memory.span_end, outside, overflow]
  · have remaining : capacity.val - offset.val < 2 ^ 64 := by
      have := capacity.hBounds
      change capacity.val < 2 ^ 64 at this
      omega
    simp only [memory.span_end, UScalar.lt_equiv, outside, ↓reduceIte]
    change Aeneas.Std.bind (UScalar.sub capacity offset) (fun rest ↦
      if bytes > rest then Result.ok none else
        Aeneas.Std.bind (UScalar.add offset bytes) (fun result ↦ Result.ok (some result))) = _
    simp only [UScalar.sub, outside, ↓reduceIte, bind_ok]
    change (if (capacity.val - offset.val) % 2 ^ 64 < bytes.val then Result.ok none
      else Aeneas.Std.bind (UScalar.add offset bytes) (fun result ↦ Result.ok (some result))) = _
    rw [Nat.mod_eq_of_lt remaining]
    by_cases large : capacity.val - offset.val < bytes.val
    · have overflow : ¬offset.val + bytes.val ≤ capacity.val := by omega
      simp [large, overflow]
    · have fits : offset.val + bytes.val ≤ capacity.val := by omega
      have bound : offset.val + bytes.val < 2 ^ 64 := by
        have := capacity.hBounds
        change capacity.val < 2 ^ 64 at this
        omega
      change offset.val + bytes.val < 18446744073709551616 at bound
      simp [large, fits, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
        UScalar.check_bounds, UScalarTy.numBits, bound, endWord, UScalar.ofNatCore]
      apply BitVec.eq_of_toNat_eq
      exact (Nat.mod_eq_of_lt bound).symm

theorem spanBound (capacity offset bytes result : U64)
    (accepted : memory.span_end capacity offset bytes = Result.ok (some result)) :
    result.val = offset.val + bytes.val ∧ result.val ≤ capacity.val := by
  rw [spanMatch] at accepted
  split at accepted
  · rename_i fits
    have same : endWord offset bytes = result := by simpa using accepted
    subst result
    have bound : offset.val + bytes.val < 2 ^ 64 := by
      have := capacity.hBounds
      change capacity.val < 2 ^ 64 at this
      omega
    have exactVal : (endWord offset bytes).val = offset.val + bytes.val :=
      Nat.mod_eq_of_lt bound
    exact ⟨exactVal, by omega⟩
  · simp at accepted

theorem intersection (a b c d : U64) :
    memory.overlaps a b c d = Result.ok true ↔ ∃ byte : ℕ,
      a.val ≤ byte ∧ byte < b.val ∧ c.val ≤ byte ∧ byte < d.val := by
  have equivalent : (a.val < b.val ∧ c.val < d.val ∧ a.val < d.val ∧ c.val < b.val) ↔
      ∃ byte : ℕ, a.val ≤ byte ∧ byte < b.val ∧ c.val ≤ byte ∧ byte < d.val := by
    constructor
    · intro bounds
      exact ⟨max a.val c.val, by omega, by omega, by omega, by omega⟩
    · rintro ⟨byte, bounds⟩
      omega
  rw [← equivalent]
  simp only [memory.overlaps, UScalar.lt_equiv]
  split <;> simp_all
  split <;> simp_all
  split <;> simp_all

/-- A checked subwindow stays in its parent's allocation, including empty windows. -/
theorem windowBound (offset rows stride start count capacity : ℕ)
    (parent : offset + rows * stride ≤ capacity) (inside : start + count ≤ rows) :
    offset + start * stride + count * stride ≤ capacity := by
  have bound := Nat.mul_le_mul_right stride inside
  nlinarith

#print axioms byteCount
#print axioms spanBound
#print axioms intersection

end BauxResearch.Semantics.Buffer
