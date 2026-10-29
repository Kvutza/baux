import BauxResearch.Forward.Algebra

/-!
For an observed patch `t`, read `encode` componentwise as the sum of
position-weighted token embeddings.  For a masked patch its output is the
learned mask vector.  The supplied `Arithmetic.store` remains part of the
formula; this is not a claim that compression preserves token distinctions.
-/

namespace BauxResearch.Forward.Encoder
noncomputable section
open scoped BigOperators

/-- The source ABI requires complete observed patches or completely masked patches. -/
abbrev Patch (patch vocabulary : ℕ) := Option (Fin patch → Fin vocabulary)

structure Weights (patch vocabulary width : ℕ) where
  embedding : Fin vocabulary → Vector width
  position : Fin patch → Vector width
  mask : Vector width

/-- `patch_encode`: learned position-wise multiplication and summation, not mean pooling. -/
def encode {patch vocabulary width : ℕ} (math : Arithmetic)
    (weights : Weights patch vocabulary width) (tokens : Patch patch vocabulary) :
    Vector width :=
  match tokens with
  | none => weights.mask
  | some observed => fun column => math.store
      (∑ index, weights.position index column * weights.embedding (observed index) column)

theorem masked {patch vocabulary width : ℕ} (math : Arithmetic)
    (weights : Weights patch vocabulary width) : encode math weights none = weights.mask := rfl

/-- Equal position-weighted embedding contributions are indistinguishable to this encoder. -/
theorem indistinguishable {patch vocabulary width : ℕ} (math : Arithmetic)
    (weights : Weights patch vocabulary width) (first second : Fin patch → Fin vocabulary)
    (same : ∀ index column,
      weights.position index column * weights.embedding (first index) column =
      weights.position index column * weights.embedding (second index) column) :
    encode math weights (some first) = encode math weights (some second) := by
  funext column
  simp only [encode]
  congr 1
  exact Finset.sum_congr rfl (fun index _ => same index column)

/-- Compression can erase all token distinctions for an admissible weight state.
This is not a claim about learned weights or dimension-counting injectivity. -/
theorem zeroPositions {patch vocabulary width : ℕ}
    (weights : Weights patch vocabulary width) (tokens : Fin patch → Fin vocabulary)
    (zero : ∀ index column, weights.position index column = 0) :
    encode Arithmetic.exact weights (some tokens) = fun _ => 0 := by
  funext column
  simp [encode, zero, Arithmetic.exact]

end
end BauxResearch.Forward.Encoder
