import BauxResearch.Forward.Mixing
import Mathlib.Analysis.Normed.Operator.Basic

/-!
# Perturbation propagation through learned mixing

Fixed stochastic transport is not a proof of full-block contraction. The
coefficient predictor and branch also change. The source-shaped decomposition
below retains those terms. The operator view uses Mathlib's continuous linear
maps and operator norm; concrete predictor constants remain to be derived.
-/
namespace BauxResearch.Forward.Sensitivity
noncomputable section
open scoped BigOperators

/-- Exact real view of the existing mHC update, including learned coefficients. -/
theorem updateSplit {streams width : ℕ}
    (before after : Mixing.Coefficients streams) (oldState newState : Rows streams width)
    (oldBranch newBranch : Vector width) (row : Fin streams) (column : Fin width) :
    Mixing.update Arithmetic.exact after newState newBranch row column -
      Mixing.update Arithmetic.exact before oldState oldBranch row column =
    (∑ source, after.residual row source * (newState source column - oldState source column)) +
    (∑ source, (after.residual row source - before.residual row source) * oldState source column) +
    after.output row * (newBranch column - oldBranch column) +
    (after.output row - before.output row) * oldBranch column := by
  simp only [Mixing.update, Arithmetic.exact, id_eq, mul_sub, sub_mul,
    Finset.sum_sub_distrib]
  ring

/-- Entrywise perturbation bound for the exact mHC mixing update, separating
state movement, residual-matrix movement, branch movement, and output-gate
movement. -/
theorem updateEntry {streams width : ℕ}
    (before after : Mixing.Coefficients streams) (oldState newState : Rows streams width)
    (oldBranch newBranch : Vector width) (row : Fin streams) (column : Fin width) :
    |Mixing.update Arithmetic.exact after newState newBranch row column -
      Mixing.update Arithmetic.exact before oldState oldBranch row column| ≤
    (∑ source, |after.residual row source| *
      |newState source column - oldState source column|) +
    (∑ source, |after.residual row source - before.residual row source| *
      |oldState source column|) +
    |after.output row| * |newBranch column - oldBranch column| +
    |after.output row - before.output row| * |oldBranch column| := by
  rw [updateSplit]
  have sum1 :
      |∑ source, after.residual row source *
          (newState source column - oldState source column)| ≤
        ∑ source, |after.residual row source| *
          |newState source column - oldState source column| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
    intro source _
    rw [abs_mul]
  have sum2 :
      |∑ source, (after.residual row source - before.residual row source) *
          oldState source column| ≤
        ∑ source, |after.residual row source - before.residual row source| *
          |oldState source column| := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum ?_)
    intro source _
    rw [abs_mul]
  have term3 :
      |after.output row * (newBranch column - oldBranch column)| =
        |after.output row| * |newBranch column - oldBranch column| :=
    abs_mul _ _
  have term4 :
      |(after.output row - before.output row) * oldBranch column| =
        |after.output row - before.output row| * |oldBranch column| :=
    abs_mul _ _
  have step1 := abs_add_le
    ((∑ source, after.residual row source * (newState source column - oldState source column)) +
      (∑ source, (after.residual row source - before.residual row source) * oldState source column) +
      after.output row * (newBranch column - oldBranch column))
    ((after.output row - before.output row) * oldBranch column)
  have step2 := abs_add_le
    ((∑ source, after.residual row source * (newState source column - oldState source column)) +
      (∑ source, (after.residual row source - before.residual row source) * oldState source column))
    (after.output row * (newBranch column - oldBranch column))
  have step3 := abs_add_le
    (∑ source, after.residual row source * (newState source column - oldState source column))
    (∑ source, (after.residual row source - before.residual row source) * oldState source column)
  linarith

variable {Space : Type*} [NormedAddCommGroup Space] [NormedSpace ℝ Space]

def block (transport : Space → Space →L[ℝ] Space) (branch : Space → Space)
    (state : Space) : Space := transport state state + branch state

theorem difference (transport : Space → Space →L[ℝ] Space) (branch : Space → Space)
    (before after : Space) :
    block transport branch after - block transport branch before =
      transport after (after - before) + (transport after - transport before) before +
        (branch after - branch before) := by
  simp only [block, map_sub, sub_apply]
  abel

theorem normBound (transport : Space → Space →L[ℝ] Space) (branch : Space → Space)
    (before after : Space) :
    ‖block transport branch after - block transport branch before‖ ≤
      ‖transport after‖ * ‖after - before‖ +
        ‖transport after - transport before‖ * ‖before‖ + ‖branch after - branch before‖ := by
  rw [difference]
  exact (norm_add_le _ _).trans (add_le_add
    ((norm_add_le _ _).trans (add_le_add
      ((transport after).le_opNorm _) ((transport after - transport before).le_opNorm _))) (le_refl _))

/-- Predictor sensitivity and branch sensitivity are not removed by a unit
transport norm. This theorem does not assign those constants to the source. -/
theorem propagated (transport : Space → Space →L[ℝ] Space) (branch : Space → Space)
    (before after : Space) (transportCap predictorLip branchLip radius : ℝ)
    (predictorPositive : 0 ≤ predictorLip)
    (transportBound : ‖transport after‖ ≤ transportCap)
    (predictorBound : ‖transport after - transport before‖ ≤ predictorLip * ‖after - before‖)
    (branchBound : ‖branch after - branch before‖ ≤ branchLip * ‖after - before‖)
    (stateBound : ‖before‖ ≤ radius) :
    ‖block transport branch after - block transport branch before‖ ≤
      (transportCap + predictorLip * radius + branchLip) * ‖after - before‖ := by
  have first := mul_le_mul_of_nonneg_right transportBound (norm_nonneg (after - before))
  have second := mul_le_mul_of_nonneg_right predictorBound (norm_nonneg before)
  have third := mul_le_mul_of_nonneg_left stateBound
    (mul_nonneg predictorPositive (norm_nonneg (after - before)))
  have total := normBound transport branch before after
  nlinarith

/-- Explicit scalar error recurrence along a finite layer schedule. Each layer
multiplies incoming state movement by `scale layer` and adds direct parameter
shift `shift layer`. -/
def foldBound {Layer : Type*} (scale shift : Layer → ℝ)
    (schedule : List Layer) (initial : ℝ) : ℝ :=
  schedule.foldl (fun bound layer => scale layer * bound + shift layer) initial

/-- Monotonicity of the layer-schedule error envelope in its initial bound when
per-layer amplification factors are nonnegative. -/
theorem foldMono {Layer : Type*} (scale shift : Layer → ℝ)
    (schedule : List Layer)
    (scaleNonneg : ∀ layer ∈ schedule, 0 ≤ scale layer)
    {first second : ℝ} (ordered : first ≤ second) :
    foldBound scale shift schedule first ≤ foldBound scale shift schedule second := by
  induction schedule generalizing first second with
  | nil => exact ordered
  | cons layer rest ih =>
    simp only [foldBound, List.foldl_cons]
    apply ih
    · intro next member
      exact scaleNonneg next (List.mem_cons_of_mem layer member)
    · have nonneg := scaleNonneg layer List.mem_cons_self
      linarith [mul_le_mul_of_nonneg_left ordered nonneg]

/-- Multi-layer forward perturbation propagation along any finite layer
schedule (such as `Program.core`'s recurrent layer schedule `[0,1,2,1,2,3,4]`),
composing single-layer affine sensitivity bounds through `List.foldl`. -/
theorem scheduleBound {State Layer : Type*} [PseudoMetricSpace State]
    (beforeStep afterStep : State → Layer → State)
    (scale shift : Layer → ℝ)
    (schedule : List Layer)
    (scaleNonneg : ∀ layer ∈ schedule, 0 ≤ scale layer)
    (layerBound : ∀ layer ∈ schedule, ∀ before after : State,
      dist (afterStep after layer) (beforeStep before layer) ≤
        scale layer * dist after before + shift layer)
    (before after : State)
    (initial : ℝ) (initialBound : dist after before ≤ initial) :
    dist (schedule.foldl afterStep after) (schedule.foldl beforeStep before) ≤
      foldBound scale shift schedule initial := by
  induction schedule generalizing before after initial with
  | nil => exact initialBound
  | cons layer rest ih =>
    simp only [List.foldl_cons, foldBound]
    have step := layerBound layer List.mem_cons_self before after
    have nextBound :
        dist (afterStep after layer) (beforeStep before layer) ≤
          scale layer * initial + shift layer := by
      have nonneg := scaleNonneg layer List.mem_cons_self
      linarith [mul_le_mul_of_nonneg_left initialBound nonneg]
    exact ih
      (fun next member => scaleNonneg next (List.mem_cons_of_mem layer member))
      (fun next member => layerBound next (List.mem_cons_of_mem layer member))
      (beforeStep before layer) (afterStep after layer)
      (scale layer * initial + shift layer) nextBound

#print axioms updateSplit
#print axioms updateEntry
#print axioms normBound
#print axioms propagated
#print axioms foldMono
#print axioms scheduleBound
end
end BauxResearch.Forward.Sensitivity
