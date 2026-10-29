import BauxResearch.Forward.Attention

/-! Fresh PISA beam-selection view. Mandatory first/previous/current blocks,
binary descent, score ordering, and final score pruning are explicit. Cached
shadow refinement and chunked KV state are not identified with this selector. -/
namespace BauxResearch.Forward.Selection
noncomputable section
open scoped BigOperators

def summary {rows width : ℕ} (math : Arithmetic) (blockSize : ℕ)
    (keys : Rows rows width) : ℕ → ℕ → Vector width
  | 0, node => fun column => math.store
      ((∑ index, if node * blockSize ≤ index.val ∧ index.val < (node + 1) * blockSize
        then keys index column else 0) / Real.sqrt blockSize)
  | level + 1, node => fun column => math.store
      ((summary math blockSize keys level (2 * node) column +
        summary math blockSize keys level (2 * node + 1) column) / Real.sqrt 2)

def contains (node level leaf : ℕ) : Bool :=
  decide (node * 2^level ≤ leaf ∧ leaf < (node + 1) * 2^level)

def mandatory (current level node : ℕ) : Bool :=
  contains node level 0 || contains node level (current - 1) || contains node level current

def descend (count current level : ℕ) (threshold : ℝ)
    (score : ℕ → ℝ) (nodes : List ℕ) : List ℕ :=
  let valid := nodes.filter fun node => mandatory current level node ||
    decide ((node + 1) * 2^level - 1 < current)
  let ordered := valid.mergeSort fun left right =>
    if mandatory current level left != mandatory current level right then
      mandatory current level left
    else decide (score right < score left ∨ (score right = score left ∧ left ≤ right))
  let finite := valid.filter fun node => !mandatory current level node
  let maximum := (finite.map score).foldl max (score (finite.headD 0))
  let chosen := (ordered.take count).filter fun node =>
    decide (level ≠ 0) || mandatory current level node || finite.isEmpty ||
      decide (maximum - threshold ≤ score node)
  if level = 0 then chosen else chosen.flatMap fun node => [2 * node, 2 * node + 1]

def leaves {rows heads width : ℕ} (math : Arithmetic) (blockSize count : ℕ)
    (threshold : ℝ) (state : Attention.Features rows heads width)
    (row : Fin rows) : List ℕ :=
  let current := row.val / blockSize
  let depth := Nat.log2 (rows / blockSize) - 4
  let query := fun column => ∑ head, state.query row head column
  let levels := (List.range (depth + 1)).reverse
  levels.foldl (fun nodes level => descend count current level threshold
    (fun node => dot query (summary math blockSize state.key level node)) nodes) (List.range 16)

def support {rows heads width : ℕ} (math : Arithmetic) (blockSize count : ℕ)
    (threshold : ℝ) (state : Attention.Features rows heads width) : Attention.Support rows :=
  fun row => Finset.univ.filter fun index =>
    index.val ≤ row.val ∧ index.val / blockSize ∈ leaves math blockSize count threshold state row

theorem causal {rows heads width : ℕ} (math : Arithmetic) (blockSize count : ℕ)
    (threshold : ℝ) (state : Attention.Features rows heads width) :
    Attention.Causal (support math blockSize count threshold state) := by
  intro row index member
  exact (Finset.mem_filter.mp member).2.1

end
end BauxResearch.Forward.Selection
