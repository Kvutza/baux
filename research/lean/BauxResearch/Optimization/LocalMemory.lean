import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Data.List.TakeDrop
import Mathlib.Data.List.FinRange
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic.Linarith

/-!
# Bounded retention with explicit local recall

For a supplied finite metric cover, keep the latest k observations in each
cell. `kept` specifies the statistic from complete history; `recordLaw` proves
that it is updated from bounded memory alone. Identifiers can carry observations
or replay descriptors; the cost of representing states and finding cells is
separate. This is a research alternative to incumbent-only compaction.
-/

namespace BauxResearch.Optimization.LocalMemory

open scoped BigOperators

variable {Index Cell : Type*} [DecidableEq Cell]

/-- Complete-history specification, with newest observations first. -/
def kept (neighbors : ℕ) (cell : Index → Cell) (history : List Index) (center : Cell) : List Index :=
  (history.filter fun index => decide (cell index = center)).take neighbors

/-- The executable bounded-state update needs only one bucket and the new item. -/
def record (neighbors : ℕ) (cell : Index → Cell) (memory : Cell → List Index) (item : Index) :
    Cell → List Index :=
  Function.update memory (cell item) ((item :: memory (cell item)).take neighbors)

/-- No full-history rescan is required to maintain the specified statistic. -/
theorem recordLaw (neighbors : ℕ) (cell : Index → Cell) (history : List Index) (item : Index) :
    kept neighbors cell (item :: history) = record neighbors cell (kept neighbors cell history) item := by
  funext center
  by_cases same : cell item = center
  · subst center
    cases neighbors <;> simp [kept, record, List.take_take]
  · simp [kept, record, same, Ne.symm same]

/-- Observation slots grow with local recall and cover size, not run horizon. -/
theorem slotBound [Fintype Cell] (neighbors : ℕ) (cell : Index → Cell) (history : List Index) :
    (∑ center : Cell, (kept neighbors cell history center).length) ≤
      Fintype.card Cell * neighbors := by
  calc
    _ ≤ ∑ _center : Cell, neighbors :=
      Finset.sum_le_sum fun center _ => List.length_take_le neighbors _
    _ = _ := by simp

/-- Saturated cells retain exactly k observations. -/
theorem fullLength (neighbors : ℕ) (cell : Index → Cell) (history : List Index) (center : Cell)
    (enough : neighbors ≤ (history.filter fun index => decide (cell index = center)).length) :
    (kept neighbors cell history center).length = neighbors := by
  simp [kept, List.length_take, Nat.min_eq_left enough]

theorem membership (neighbors : ℕ) (cell : Index → Cell) (history : List Index)
    (center : Cell) (item : Index) (member : item ∈ kept neighbors cell history center) :
    item ∈ history ∧ cell item = center := by
  simpa only [decide_eq_true_eq] using List.mem_filter.mp (List.mem_of_mem_take member)

/-- Recall preserves distinct observations when the input history has no
duplicate observation identifiers. It does not require distinct parameter states. -/
theorem distinct (neighbors : ℕ) (cell : Index → Cell) (history : List Index) (center : Cell)
    (unique : history.Nodup) : (kept neighbors cell history center).Nodup :=
  (unique.filter _).take

section Metric

variable {State : Type*} [PseudoMetricSpace State]

/-- Every retained observation in the query cell lies within twice its cover
radius. The metric and cover are explicit; no low-dimensional embedding is assumed. -/
theorem localRadius (neighbors : ℕ) (cell : Index → Cell) (history : List Index)
    (point : Index → State) (center : Cell → State) (radius : ℝ)
    (covered : ∀ index, dist (point index) (center (cell index)) ≤ radius)
    (query item : Index) (member : item ∈ kept neighbors cell history (cell query)) :
    dist (point item) (point query) ≤ 2 * radius := by
  have same := (membership neighbors cell history (cell query) item member).2
  have near := covered item
  rw [same] at near
  have next := covered query
  rw [dist_comm (point query)] at next
  exact (dist_triangle (point item) (center (cell query)) (point query)).trans (by linarith)

end Metric

#print axioms recordLaw
#print axioms slotBound
#print axioms fullLength
#print axioms localRadius

end BauxResearch.Optimization.LocalMemory
