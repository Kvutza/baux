import BauxResearch.Forward.Algebra

/-! Selected-support, shared-K/V multihead attention. The support is explicit
input data: neither the PISA beam search nor cached shadow refinement is hidden
behind a claim that this computes their decisions. See Selection.lean. -/
/-!
For row `r`, head `h`, and supplied support `S_r`, the attended value is the
exponentially weighted average of `V_i` over `i ∈ S_r`, using the score of
`Q_{r,h}` against `K_i`.  The support is an input to this module, not a
theorem about which keys PISA or a cache would select.  `locality` is
conditional on that same support and the relevant query/key/value equality.
-/
namespace BauxResearch.Forward.Attention
noncomputable section
open scoped BigOperators

def get {width : ℕ} (values : Vector width) (index : ℕ) : ℝ :=
  if bound : index < width then values ⟨index, bound⟩ else 0

def rotate {width : ℕ} (math : Arithmetic) (cosine sine : ℕ → ℝ)
    (values : Vector width) : Vector width :=
  fun index => math.store (if index.val % 2 = 0 then
    cosine (index.val / 2) * values index - sine (index.val / 2) * get values (index.val + 1)
    else sine (index.val / 2) * get values (index.val - 1) +
      cosine (index.val / 2) * values index)

structure Weights (width heads keyWidth : ℕ) where
  query : Fin heads → Fin width → Fin keyWidth → ℝ
  key : Fin width → Fin keyWidth → ℝ
  value : Fin width → Fin keyWidth → ℝ
  output : Fin heads → Fin keyWidth → Fin width → ℝ

structure Features (rows heads keyWidth : ℕ) where
  query : Fin rows → Fin heads → Vector keyWidth
  key : Rows rows keyWidth
  value : Rows rows keyWidth

def features {rows width heads keyWidth : ℕ} (math : Arithmetic)
    (weights : Weights width heads keyWidth) (input : Rows rows width)
    (cosine sine : Fin rows → ℕ → ℝ) : Features rows heads keyWidth where
  query := fun row head => rotate math (cosine row) (sine row)
    (project math (weights.query head) (input row))
  key := fun row => rotate math (cosine row) (sine row)
    (project math weights.key (input row))
  value := fun row => project math weights.value (input row)

abbrev Support (rows : ℕ) := Fin rows → Finset (Fin rows)

def Causal {rows : ℕ} (support : Support rows) : Prop :=
  ∀ row index, index ∈ support row → index.val ≤ row.val

def score {rows heads keyWidth : ℕ} (scale : ℝ)
    (state : Features rows heads keyWidth) (row : Fin rows) (head : Fin heads)
    (index : Fin rows) : ℝ := scale * dot (state.query row head) (state.key index)

def attend {rows heads keyWidth : ℕ} (math : Arithmetic) (scale : ℝ)
    (support : Support rows) (state : Features rows heads keyWidth)
    (row : Fin rows) (head : Fin heads) : Vector keyWidth :=
  fun column => math.store
    ((∑ index ∈ support row, Real.exp (score scale state row head index) * state.value index column) /
      ∑ index ∈ support row, Real.exp (score scale state row head index))

def branch {rows width heads keyWidth : ℕ} (math : Arithmetic) (scale : ℝ)
    (weights : Weights width heads keyWidth) (support : Support rows)
    (state : Features rows heads keyWidth) : Rows rows width :=
  fun row column => math.store (∑ head, ∑ index,
    attend math scale support state row head index * weights.output head index column)

/-- At fixed support, only selected keys/values and the current query can matter. -/
theorem locality {rows heads keyWidth : ℕ} (math : Arithmetic) (scale : ℝ)
    (support : Support rows) (first second : Features rows heads keyWidth)
    (row : Fin rows) (head : Fin heads)
    (sameQuery : first.query row head = second.query row head)
    (sameKey : ∀ index ∈ support row, first.key index = second.key index)
    (sameValue : ∀ index ∈ support row, first.value index = second.value index) :
    attend math scale support first row head = attend math scale support second row head := by
  have scores : ∀ index ∈ support row,
      score scale first row head index = score scale second row head index := by
    intro index member
    simp only [score, sameQuery, sameKey index member]
  funext column
  unfold attend
  congr 2
  · apply Finset.sum_congr rfl
    intro index member
    rw [scores index member, sameValue index member]
  · exact Finset.sum_congr rfl (fun index member => congrArg Real.exp (scores index member))

end
end BauxResearch.Forward.Attention
