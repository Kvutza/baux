import Mathlib.Tactic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.BitVec

/-!
# Integer reachability for the CUDA Gaussian rejection sampler

The word expressions follow `backends/cuda/device/src/fbt_math.rs`. This module does not yet
prove extraction correspondence, float/log predicates, the generated host tables,
or termination of the complete GPU loop. Successful branch witnesses below have
explicit table or floating-point premises. The retry multiplier is divisible by
four: a fixed seed visits one input coset, not every 32-bit word.
-/

namespace BauxResearch.Semantics.Gaussian

abbrev Word := BitVec 32

def shift16 (x : Word) : Word := x ^^^ (x >>> 16)
def shift15 (x : Word) : Word := x ^^^ (x >>> 15)
def undo15 (x : Word) : Word := x ^^^ (x >>> 15) ^^^ (x >>> 30)

def mix (x : Word) : Word :=
  shift16 (shift15 (shift16 x * 0x7feb352d) * 0x846ca68b)

def unmix (x : Word) : Word :=
  shift16 (undo15 (shift16 x * 0x43021123) * 0x1d69e2a5)

theorem cancel16 (x : Word) : shift16 (shift16 x) = x := by
  ext i hi
  interval_cases i <;> simp [shift16, Bool.xor_assoc]

theorem cancel15 (x : Word) : undo15 (shift15 x) = x := by
  ext i hi
  interval_cases i <;> simp [undo15, shift15, Bool.xor_assoc]

theorem mulFirst (x : Word) : x * 0x7feb352d * 0x1d69e2a5 = x := by
  rw [BitVec.mul_assoc]
  change x * 1 = x
  exact BitVec.mul_one x

theorem mulSecond (x : Word) : x * 0x846ca68b * 0x43021123 = x := by
  rw [BitVec.mul_assoc]
  change x * 1 = x
  exact BitVec.mul_one x

theorem unmixMix (x : Word) : unmix (mix x) = x := by
  simp only [unmix, mix, cancel16, mulSecond, cancel15, mulFirst]

theorem mixInjective : Function.Injective mix := by
  intro x y same
  have h := congrArg unmix same
  simpa only [unmixMix] using h

theorem mixBijective : Function.Bijective mix := by
  letI : Finite Word := Finite.of_injective BitVec.toFin BitVec.toFin_injective
  exact ⟨mixInjective, Finite.surjective_of_injective mixInjective⟩

/-- One stream's fixed seed/key/element contribution is absorbed into `base`. -/
def input (base attempt : Word) : Word := base ^^^ (attempt * 0xd3a2646c)

/-- Solve the counter equation inside its low-two-bit coset. -/
def witness (base target : Word) : Word :=
  ((base ^^^ target) >>> 2) * 0x70eb1913

theorem reachable (base target : Word) (coset : base &&& 3 = target &&& 3) :
    input base (witness base target) = target := by
  unfold input witness
  rw [BitVec.mul_assoc]
  change base ^^^ (((base ^^^ target) >>> 2) * BitVec.twoPow 32 2) = target
  rw [BitVec.mul_twoPow_eq_shiftLeft]
  have low0 : base[0] = target[0] := by
    have same := congrArg (fun value : Word ↦ value[0]) coset
    simpa using same
  have low1 : base[1] = target[1] := by
    have same := congrArg (fun value : Word ↦ value[1]) coset
    simpa using same
  ext i hi
  interval_cases i <;> simp_all [Bool.xor_assoc]

/-- Every starting retry counter reaches a coset witness before wrapping twice. -/
theorem retryWitness (base target start : Word)
    (coset : base &&& 3 = target &&& 3) :
    ∃ steps : Word, input base (start + steps) = target := by
  refine ⟨witness base target - start, ?_⟩
  have cancel : start + (witness base target - start) = witness base target := by
    rw [BitVec.add_comm, BitVec.sub_add_cancel]
  rw [cancel]
  exact reachable base target coset

/-- A successful word predicate is reached within one full counter cycle. -/
theorem finiteReach (base target start : Word) (predicate : Word → Prop)
    (coset : base &&& 3 = target &&& 3) (passes : predicate target) :
    ∃ steps : ℕ, steps < 2 ^ 32 ∧
      predicate (input base (start + BitVec.ofNat 32 steps)) := by
  obtain ⟨steps, reached⟩ := retryWitness base target start coset
  refine ⟨steps.toNat, steps.isLt, ?_⟩
  simpa only [BitVec.ofNat_toNat, BitVec.setWidth_eq, reached] using passes

theorem cosets (base : Word) :
    base &&& 3 = 0 ∨ base &&& 3 = 1 ∨ base &&& 3 = 2 ∨ base &&& 3 = 3 := by
  have bound : (base &&& 3).toNat ≤ 3 := by
    simpa using (Nat.and_le_right : base.toNat &&& 3 ≤ 3)
  have residues : (base &&& 3).toNat = 0 ∨ (base &&& 3).toNat = 1 ∨
      (base &&& 3).toNat = 2 ∨ (base &&& 3).toNat = 3 := by omega
  rcases residues with zero | one | two | three
  · exact Or.inl (BitVec.eq_of_toNat_eq zero)
  · exact Or.inr (Or.inl (BitVec.eq_of_toNat_eq one))
  · exact Or.inr (Or.inr (Or.inl (BitVec.eq_of_toNat_eq two)))
  · exact Or.inr (Or.inr (Or.inr (BitVec.eq_of_toNat_eq three)))

def magnitude (bits : Word) : Word := if bits.toNat < 2 ^ 31 then bits else -bits

def accepted (thresholds : Nat → Nat) (bits : Word) : Prop :=
  (magnitude bits).toNat < thresholds (bits &&& 255).toNat

/-- Four exact integer witnesses; the production table bounds remain premises. -/
theorem normalWitness (thresholds : Nat → Nat)
    (zero : 0 < thresholds 0) (first : 1753845952 < thresholds 192)
    (second : 787275391 < thresholds 129) (third : 1408362973 < thresholds 221) :
    accepted thresholds (mix 0) ∧ accepted thresholds (mix 1) ∧
      accepted thresholds (mix 2) ∧ accepted thresholds (mix 3) := by
  exact ⟨zero, first, second, third⟩

/-- Conditional finite reachability of the normal integer acceptance branch.
An earlier tail branch may still block the complete sampler; this is not its termination proof. -/
theorem normalReach (thresholds : Nat → Nat) (base start : Word)
    (zero : 0 < thresholds 0) (first : 1753845952 < thresholds 192)
    (second : 787275391 < thresholds 129) (third : 1408362973 < thresholds 221) :
    ∃ steps : ℕ, steps < 2 ^ 32 ∧
      accepted thresholds (mix (input base (start + BitVec.ofNat 32 steps))) := by
  obtain ⟨a, b, c, d⟩ := normalWitness thresholds zero first second third
  rcases cosets base with h | h | h | h
  · exact finiteReach base 0 start (fun raw ↦ accepted thresholds (mix raw))
      (by simpa using h) a
  · exact finiteReach base 1 start (fun raw ↦ accepted thresholds (mix raw))
      (by simpa using h) b
  · exact finiteReach base 2 start (fun raw ↦ accepted thresholds (mix raw))
      (by simpa using h) c
  · exact finiteReach base 3 start (fun raw ↦ accepted thresholds (mix raw))
      (by simpa using h) d

/-- Stream 2 differs from stream 1 by these two exact wrapping XOR constants. -/
def other (raw : Word) : Word := mix (raw ^^^ 0xfd7046c5 ^^^ (2 * 0xfd7046c5))

def tailInput (coset : Fin 4) : Word :=
  match coset.val with
  | 0 => 0xd5f9efc4
  | 1 => 0x33984aa9
  | 2 => 0xf6ab832a
  | _ => 0x20320b4b

theorem tailCoset (coset : Fin 4) : tailInput coset &&& 3 = BitVec.ofNat 32 coset.val := by
  fin_cases coset <;> decide

/-- The four word pairs to pass through the actual CUDA float/log acceptance test. -/
theorem tailWords :
    (mix 0xd5f9efc4 = 0xffffffff ∧ other 0xd5f9efc4 = 0x39ef859b) ∧
    (mix 0x33984aa9 = 0xfffffffe ∧ other 0x33984aa9 = 0x49c079da) ∧
    (mix 0xf6ab832a = 0xfffffffc ∧ other 0xf6ab832a = 0x0a72fd74) ∧
    (mix 0x20320b4b = 0xfffffffb ∧ other 0x20320b4b = 0x20394f67) := by
  decide

/-- Every retry stream reaches a passing tail word pair within one full cycle,
conditional on the four actual float/log predicate checks. -/
theorem tailReach (passes : Word → Word → Prop) (base start : Word)
    (first : passes 0xffffffff 0x39ef859b) (second : passes 0xfffffffe 0x49c079da)
    (third : passes 0xfffffffc 0x0a72fd74) (fourth : passes 0xfffffffb 0x20394f67) :
    ∃ steps : ℕ, steps < 2 ^ 32 ∧
      passes (mix (input base (start + BitVec.ofNat 32 steps)))
        (other (input base (start + BitVec.ofNat 32 steps))) := by
  rcases cosets base with h | h | h | h
  · exact finiteReach base 0xd5f9efc4 start (fun raw ↦ passes (mix raw) (other raw))
      (by simpa using h) first
  · exact finiteReach base 0x33984aa9 start (fun raw ↦ passes (mix raw) (other raw))
      (by simpa using h) second
  · exact finiteReach base 0xf6ab832a start (fun raw ↦ passes (mix raw) (other raw))
      (by simpa using h) third
  · exact finiteReach base 0x20320b4b start (fun raw ↦ passes (mix raw) (other raw))
      (by simpa using h) fourth

#print axioms unmixMix
#print axioms mixBijective
#print axioms reachable
#print axioms retryWitness
#print axioms normalWitness
#print axioms normalReach
#print axioms tailReach
#print axioms tailWords

end BauxResearch.Semantics.Gaussian
