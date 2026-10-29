import BauxResearch.Forward.Mixing
import BauxResearch.Forward.Clipping
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Predictor-seed sensitivity

The source subtracts a finite raw maximum, clips at -80, and exponentiates.
Each stage has a checked stability bound. The logarithmic seed bound is the
natural input to projective Sinkhorn sensitivity.
-/

namespace BauxResearch.Forward.SeedSensitivity

noncomputable section
open Set
open scoped NNReal

theorem expIic : LipschitzOnWith 1 Real.exp (Set.Iic 0) := by
  apply Convex.lipschitzOnWith_of_nnnorm_deriv_le
  · intro value _
    exact (Real.hasDerivAt_exp value).differentiableAt
  · intro value member
    rw [Real.deriv_exp]
    change ‖Real.exp value‖ ≤ (1 : ℝ)
    simpa [Real.norm_eq_abs, abs_of_pos (Real.exp_pos value)] using
      Real.exp_le_one_iff.mpr member
  · exact convex_Iic 0

/-- Centering each raw matrix by its own finite maximum doubles entrywise sup
error at most. -/
theorem centerBound {streams : ℕ} (positive : 0 < streams)
    (first second : Transport.Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta)
    (row column : Fin streams) :
    |(second row column - Mixing.rawMax positive second) -
        (first row column - Mixing.rawMax positive first)| ≤ 2 * delta := by
  have entry := abs_le.mp (shift row column)
  have maximum := abs_le.mp (Mixing.maxBound positive first second delta shift)
  apply abs_le.mpr
  constructor <;> linarith

/-- Lower clipping is one-Lipschitz, so it preserves the centered two-delta
bound. -/
theorem clipBound {streams : ℕ} (positive : 0 < streams)
    (first second : Transport.Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta)
    (row column : Fin streams) :
    |max (second row column - Mixing.rawMax positive second) (-80) -
        max (first row column - Mixing.rawMax positive first) (-80)| ≤ 2 * delta := by
  have clipped := (Clipping.floorLipschitz (-80)).dist_le_mul
    (second row column - Mixing.rawMax positive second)
    (first row column - Mixing.rawMax positive first)
  have centered := centerBound positive first second delta shift row column
  have clippedBound :
      |max (second row column - Mixing.rawMax positive second) (-80) -
          max (first row column - Mixing.rawMax positive first) (-80)| ≤
        |second row column - Mixing.rawMax positive second +
          Mixing.rawMax positive first - first row column| := by
    simpa [Clipping.floor, Real.dist_eq] using clipped
  have centeredBound :
      |second row column - Mixing.rawMax positive second +
          Mixing.rawMax positive first - first row column| ≤ 2 * delta := by
    rw [show second row column - Mixing.rawMax positive second +
      Mixing.rawMax positive first - first row column =
      (second row column - Mixing.rawMax positive second) -
        (first row column - Mixing.rawMax positive first) by ring]
    exact centered
  exact clippedBound.trans centeredBound

/-- Absolute seed-entry error is at most twice the raw-score sup error. -/
theorem seedBound {streams : ℕ} (positive : 0 < streams)
    (first second : Transport.Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta)
    (row column : Fin streams) :
    |Transport.seed (Mixing.rawMax positive second) second row column -
        Transport.seed (Mixing.rawMax positive first) first row column| ≤ 2 * delta := by
  let secondClip := max (second row column - Mixing.rawMax positive second) (-80)
  let firstClip := max (first row column - Mixing.rawMax positive first) (-80)
  have secondMember : secondClip ∈ Set.Iic 0 := by
    change secondClip ≤ 0
    apply max_le
    · exact sub_nonpos.mpr (Mixing.rawUpper positive second row column)
    · norm_num
  have firstMember : firstClip ∈ Set.Iic 0 := by
    change firstClip ≤ 0
    apply max_le
    · exact sub_nonpos.mpr (Mixing.rawUpper positive first row column)
    · norm_num
  have exponential := expIic.dist_le_mul secondClip secondMember firstClip firstMember
  have clipped := clipBound positive first second delta shift row column
  have exponentialBound : |Real.exp secondClip - Real.exp firstClip| ≤
      |secondClip - firstClip| := by
    simpa [Real.dist_eq] using exponential
  have total := exponentialBound.trans clipped
  simpa [Transport.seed, secondClip, firstClip, Real.dist_eq] using total

/-- More importantly for Sinkhorn, the log-entry displacement is at most twice
the raw-score sup error, independently of exponentially small seed entries. -/
theorem logBound {streams : ℕ} (positive : 0 < streams)
    (first second : Transport.Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta)
    (row column : Fin streams) :
    |Real.log (Transport.seed (Mixing.rawMax positive second) second row column) -
        Real.log (Transport.seed (Mixing.rawMax positive first) first row column)| ≤
      2 * delta := by
  simpa [Transport.seed, Real.log_exp] using
    clipBound positive first second delta shift row column

#print axioms expIic
#print axioms centerBound
#print axioms clipBound
#print axioms seedBound
#print axioms logBound

end

end BauxResearch.Forward.SeedSensitivity
