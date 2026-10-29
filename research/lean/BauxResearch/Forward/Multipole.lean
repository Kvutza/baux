import BauxResearch.Forward.Attention
import BauxResearch.Forward.SelectionCell
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Multipole Softmax Tail Certificate for Tree Attention

Formally establishes the exponential decay bound on pruned attention mass when
hierarchical tree attention (from `Selection.lean` and Fast Multipole Methods)
replaces full dense attention over long contexts (1M tokens).

If every unselected/pruned key cluster has score bounded strictly below the active
maximum by threshold `τ`, the fraction of partition mass omitted by the sparse
support is bounded by `|unchosen| * exp(-τ)`.
-/

namespace BauxResearch.Forward.Multipole

noncomputable section
open scoped BigOperators

variable {Index : Type*} [Fintype Index] [DecidableEq Index]

/-- The total partition sum (softmax denominator) over a set of indices. -/
def partition (score : Index → ℝ) (indices : Finset Index) : ℝ :=
  ∑ i ∈ indices, Real.exp (score i)

/-- Positive partition sum when indices is non-empty. -/
theorem partitionPos (score : Index → ℝ) {indices : Finset Index} (h : indices.Nonempty) :
    0 < partition score indices := by
  obtain ⟨x, hx⟩ := h
  unfold partition
  have hpos : 0 < Real.exp (score x) := Real.exp_pos (score x)
  have hnonneg : ∀ i ∈ indices, 0 ≤ Real.exp (score i) := fun i _ => le_of_lt (Real.exp_pos (score i))
  exact Finset.sum_pos' (fun i hi => hnonneg i hi) ⟨x, hx, hpos⟩

/-- When an active support contains an anchor scoring at least `M`, and all
unchosen/pruned keys have score bounded by `M - τ`, the sum of exp scores over
the unchosen set is bounded by `|unchosen| * exp(M - τ)`. -/
theorem unchosenBound
    (score : Index → ℝ)
    (unchosen : Finset Index)
    (maxActive : ℝ)
    (threshold : ℝ)
    (hBound : ∀ i ∈ unchosen, score i ≤ maxActive - threshold) :
    partition score unchosen ≤ (unchosen.card : ℝ) * Real.exp (maxActive - threshold) := by
  unfold partition
  have hterm : ∀ i ∈ unchosen, Real.exp (score i) ≤ Real.exp (maxActive - threshold) := by
    intro i hi
    exact Real.exp_le_exp.mpr (hBound i hi)
  calc
    ∑ i ∈ unchosen, Real.exp (score i)
      ≤ ∑ i ∈ unchosen, Real.exp (maxActive - threshold) := Finset.sum_le_sum hterm
    _ = (unchosen.card : ℝ) * Real.exp (maxActive - threshold) := by
      rw [Finset.sum_const, nsmul_eq_mul]

/-- Exponential tail suppression: the relative error mass of the unchosen tree blocks
relative to the active peak `exp(maxActive)` is strictly bounded by `|unchosen| * exp(-τ)`. -/
theorem tailSuppression
    (score : Index → ℝ)
    (unchosen : Finset Index)
    (maxActive : ℝ)
    (threshold : ℝ)
    (hBound : ∀ i ∈ unchosen, score i ≤ maxActive - threshold) :
    partition score unchosen ≤ Real.exp maxActive * ((unchosen.card : ℝ) * Real.exp (-threshold)) := by
  have bound1 := unchosenBound score unchosen maxActive threshold hBound
  have hexp : Real.exp (maxActive - threshold) = Real.exp maxActive * Real.exp (-threshold) := by
    rw [sub_eq_add_neg, Real.exp_add]
  calc
    partition score unchosen ≤ (unchosen.card : ℝ) * Real.exp (maxActive - threshold) := bound1
    _ = (unchosen.card : ℝ) * (Real.exp maxActive * Real.exp (-threshold)) := by rw [hexp]
    _ = Real.exp maxActive * ((unchosen.card : ℝ) * Real.exp (-threshold)) := by ring

/-- The normalized attention output for a scalar feature over a set of indices. -/
def attentionOutput (score : Index → ℝ) (value : Index → ℝ) (indices : Finset Index) : ℝ :=
  (∑ i ∈ indices, Real.exp (score i) * value i) / partition score indices

/-- Partition sum is non-negative on any set of indices. -/
theorem partitionNonneg (score : Index → ℝ) (indices : Finset Index) :
    0 ≤ partition score indices := by
  unfold partition
  exact Finset.sum_nonneg (fun i _ => le_of_lt (Real.exp_pos (score i)))

/-- Partition sum on a set containing an anchor is at least the anchor's exponentiated score. -/
theorem partitionAnchor
    (score : Index → ℝ)
    (indices : Finset Index)
    (anchor : Index)
    (hAnchor : anchor ∈ indices) :
    Real.exp (score anchor) ≤ partition score indices := by
  unfold partition
  have hpos : ∀ i ∈ indices, 0 ≤ Real.exp (score i) :=
    fun i _ => le_of_lt (Real.exp_pos (score i))
  exact Finset.single_le_sum hpos hAnchor

/-- Bounding the weighted value sum by the maximum absolute value times partition. -/
theorem valueBound
    (score : Index → ℝ)
    (value : Index → ℝ)
    (indices : Finset Index)
    (valBound : ℝ)
    (hVal : ∀ i ∈ indices, |value i| ≤ valBound) :
    |∑ i ∈ indices, Real.exp (score i) * value i| ≤ valBound * partition score indices := by
  unfold partition
  have h1 : |∑ i ∈ indices, Real.exp (score i) * value i| ≤
      ∑ i ∈ indices, |Real.exp (score i) * value i| :=
    Finset.abs_sum_le_sum_abs _ _
  have h2 : (∑ i ∈ indices, |Real.exp (score i) * value i|) =
      ∑ i ∈ indices, Real.exp (score i) * |value i| := by
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [abs_mul, abs_of_pos (Real.exp_pos (score i))]
  have h3 : (∑ i ∈ indices, Real.exp (score i) * |value i|) ≤
      ∑ i ∈ indices, Real.exp (score i) * valBound := by
    refine Finset.sum_le_sum (fun i hi => ?_)
    exact mul_le_mul_of_nonneg_left (hVal i hi) (le_of_lt (Real.exp_pos (score i)))
  have h4 : (∑ i ∈ indices, Real.exp (score i) * valBound) =
      valBound * (∑ i ∈ indices, Real.exp (score i)) := by
    rw [← Finset.sum_mul, mul_comm]
  linarith

theorem quotientDiff {a b c d : ℝ} (hb : b ≠ 0) (hbd : b + d ≠ 0) :
    a / b - (a + c) / (b + d) = (a * d - b * c) / (b * (b + d)) := by
  rw [div_sub_div a (a + c) hb hbd]
  congr 1
  ring

theorem quotientBound
    {a b c d V : ℝ}
    (ha : |a| ≤ V * b)
    (hc : |c| ≤ V * d)
    (hb : 0 < b)
    (hd : 0 ≤ d) :
    |a / b - (a + c) / (b + d)| ≤ (2 * V * d) / (b + d) := by
  have hb_ne : b ≠ 0 := ne_of_gt hb
  have hbd : 0 < b + d := add_pos_of_pos_of_nonneg hb hd
  have hbd_ne : b + d ≠ 0 := ne_of_gt hbd
  rw [quotientDiff hb_ne hbd_ne, abs_div]
  have hdenom : |b * (b + d)| = b * (b + d) := abs_of_pos (mul_pos hb hbd)
  rw [hdenom]
  have hnum : |a * d - b * c| ≤ 2 * V * b * d := by
    have htri : |a * d - b * c| ≤ |a * d| + |b * c| := abs_sub _ _
    have hmul1 : |a * d| = |a| * d := by rw [abs_mul, abs_of_nonneg hd]
    have hmul2 : |b * c| = b * |c| := by rw [abs_mul, abs_of_pos hb]
    have hterm1 : |a| * d ≤ V * b * d := mul_le_mul_of_nonneg_right ha hd
    have hterm2 : b * |c| ≤ V * b * d := by
      calc
        b * |c| ≤ b * (V * d) := mul_le_mul_of_nonneg_left hc (le_of_lt hb)
        _ = V * b * d := by ring
    linarith
  have hfrac : |a * d - b * c| / (b * (b + d)) ≤ (2 * V * b * d) / (b * (b + d)) :=
    div_le_div_of_nonneg_right hnum (le_of_lt (mul_pos hb hbd))
  have heq : (2 * V * b * d) / (b * (b + d)) = (2 * V * d) / (b + d) := by
    calc
      (2 * V * b * d) / (b * (b + d)) = (b * (2 * V * d)) / (b * (b + d)) := by
        congr 1; ring
      _ = (2 * V * d) / (b + d) := mul_div_mul_left _ _ hb_ne
  exact hfrac.trans_eq heq

/-- Under the condition that all unchosen blocks score at least threshold `τ` below
the active maximum, the absolute difference between the normalized attention output
on the active support and the true dense attention is bounded by
`(2 * valBound) * (|unchosen| * exp(-τ))`. -/
theorem denseApproximation
    (score : Index → ℝ)
    (value : Index → ℝ)
    (active : Finset Index)
    (unchosen : Finset Index)
    (hDisjoint : Disjoint active unchosen)
    (anchor : Index)
    (hAnchor : anchor ∈ active)
    (maxActive : ℝ)
    (hMax : maxActive ≤ score anchor)
    (threshold : ℝ)
    (hBound : ∀ i ∈ unchosen, score i ≤ maxActive - threshold)
    (valBound : ℝ)
    (hVal : ∀ i, |value i| ≤ valBound) :
    |attentionOutput score value active - attentionOutput score value (active ∪ unchosen)| ≤
      (2 * valBound) * ((unchosen.card : ℝ) * Real.exp (-threshold)) := by
  let pA := partition score active
  let pU := partition score unchosen
  let sA := ∑ i ∈ active, Real.exp (score i) * value i
  let sU := ∑ i ∈ unchosen, Real.exp (score i) * value i
  have hExpMaxPos : 0 < Real.exp maxActive := Real.exp_pos maxActive
  have hAnchorExp : Real.exp (score anchor) ≤ pA := partitionAnchor score active anchor hAnchor
  have hMaxExp : Real.exp maxActive ≤ Real.exp (score anchor) := Real.exp_le_exp.mpr hMax
  have hpA : 0 < pA := hExpMaxPos.trans_le (hMaxExp.trans hAnchorExp)
  have hpUNonneg : 0 ≤ pU := partitionNonneg score unchosen
  have hValNonneg : 0 ≤ valBound := (abs_nonneg (value anchor)).trans (hVal anchor)
  have hsaBound : |sA| ≤ valBound * pA := valueBound score value active valBound (fun i _ => hVal i)
  have hsuBound : |sU| ≤ valBound * pU := valueBound score value unchosen valBound (fun i _ => hVal i)
  have hPartUnion : partition score (active ∪ unchosen) = pA + pU := by
    unfold partition
    exact Finset.sum_union hDisjoint
  have hSumUnion : (∑ i ∈ active ∪ unchosen, Real.exp (score i) * value i) = sA + sU :=
    Finset.sum_union hDisjoint
  have hAttA : attentionOutput score value active = sA / pA := rfl
  have hAttD : attentionOutput score value (active ∪ unchosen) = (sA + sU) / (pA + pU) := by
    unfold attentionOutput
    rw [hSumUnion, hPartUnion]
  rw [hAttA, hAttD]
  have hDiffBound := quotientBound hsaBound hsuBound hpA hpUNonneg
  have hle_denom : (2 * valBound * pU) / (pA + pU) ≤ (2 * valBound * pU) / pA := by
    have hnum_nonneg : 0 ≤ 2 * valBound * pU := by
      have : 0 ≤ 2 * valBound := by linarith
      exact mul_nonneg this hpUNonneg
    have hle_add : pA ≤ pA + pU := by linarith
    exact div_le_div_of_nonneg_left hnum_nonneg hpA hle_add
  have hdecomp : (2 * valBound * pU) / pA = (2 * valBound) * (pU / pA) := by
    ring
  have hTail := tailSuppression score unchosen maxActive threshold hBound
  have hratio : pU / pA ≤ (unchosen.card : ℝ) * Real.exp (-threshold) := by
    have h1 : pU / pA ≤ (Real.exp maxActive * ((unchosen.card : ℝ) * Real.exp (-threshold))) / pA :=
      div_le_div_of_nonneg_right hTail (le_of_lt hpA)
    have h2 : (Real.exp maxActive * ((unchosen.card : ℝ) * Real.exp (-threshold))) / pA =
        (Real.exp maxActive / pA) * ((unchosen.card : ℝ) * Real.exp (-threshold)) := by
      ring
    have hExpLePA : Real.exp maxActive ≤ pA := hMaxExp.trans hAnchorExp
    have h3 : Real.exp maxActive / pA ≤ 1 := by
      rw [div_le_iff₀ hpA]
      linarith
    have hcard_exp_nonneg : 0 ≤ (unchosen.card : ℝ) * Real.exp (-threshold) :=
      mul_nonneg (Nat.cast_nonneg _) (le_of_lt (Real.exp_pos _))
    calc
      pU / pA ≤ (Real.exp maxActive * ((unchosen.card : ℝ) * Real.exp (-threshold))) / pA := h1
      _ = (Real.exp maxActive / pA) * ((unchosen.card : ℝ) * Real.exp (-threshold)) := h2
      _ ≤ 1 * ((unchosen.card : ℝ) * Real.exp (-threshold)) :=
        mul_le_mul_of_nonneg_right h3 hcard_exp_nonneg
      _ = (unchosen.card : ℝ) * Real.exp (-threshold) := by ring
  have hFinal : (2 * valBound) * (pU / pA) ≤
      (2 * valBound) * ((unchosen.card : ℝ) * Real.exp (-threshold)) := by
    have : 0 ≤ 2 * valBound := by linarith
    exact mul_le_mul_of_nonneg_left hratio this
  calc
    |sA / pA - (sA + sU) / (pA + pU)| ≤ (2 * valBound * pU) / (pA + pU) := hDiffBound
    _ ≤ (2 * valBound * pU) / pA := hle_denom
    _ = (2 * valBound) * (pU / pA) := hdecomp
    _ ≤ (2 * valBound) * ((unchosen.card : ℝ) * Real.exp (-threshold)) := hFinal

end

end BauxResearch.Forward.Multipole
