import BauxResearch.Forward.Transport

/-!
The predictor first scales a state `X` by
`s(X) = 1 / sqrt((∑ Xᵢⱼ²)/(streams * width) + 1/100000)`.
Its three coefficient groups are distinct functions of this normalized state.
The finite raw maximum and the subsequent transport operations are part of
the source-shaped formula; calling the result a mixing matrix does not by
itself establish stochasticity or stability.
-/

namespace BauxResearch.Forward.Mixing
noncomputable section
open scoped BigOperators

structure Weights (streams width : ℕ) where
  input : Fin streams → Fin width → Fin streams → ℝ
  residual : Fin streams → Fin width → Fin streams → Fin streams → ℝ
  output : Fin streams → Fin width → Fin streams → ℝ
  inputBias : Vector streams
  residualBias : Transport.Square streams
  outputBias : Vector streams
  control : Fin 3 → ℝ
  lambda : ℝ
  norm : Vector width

structure Coefficients (streams : ℕ) where
  input : Vector streams
  residual : Transport.Square streams
  output : Vector streams

namespace FoldMax

theorem initial (values : List ℝ) (start : ℝ) :
    start ≤ values.foldl max start := by
  induction values generalizing start with
  | nil => exact le_rfl
  | cons value values hypothesis =>
      exact (le_max_left start value).trans (hypothesis (max start value))

theorem member (values : List ℝ) (start value : ℝ)
    (present : value ∈ values) : value ≤ values.foldl max start := by
  induction values generalizing start with
  | nil => simp at present
  | cons head tail hypothesis =>
      rw [List.foldl_cons]
      rcases List.mem_cons.mp present with rfl | member
      · exact (le_max_right start value).trans (initial tail (max start value))
      · exact hypothesis (max start head) member

theorem attained (values : List ℝ) (start : ℝ) :
    values.foldl max start ∈ start :: values := by
  induction values generalizing start with
  | nil => simp
  | cons head tail hypothesis =>
      rw [List.foldl_cons]
      have result := hypothesis (max start head)
      rcases List.mem_cons.mp result with equal | member
      · rw [equal]
        rcases le_total start head with order | order
        · rw [max_eq_right order]
          simp
        · rw [max_eq_left order]
          simp
      · exact List.mem_cons_of_mem start (List.mem_cons_of_mem head member)

end FoldMax

def rawList {streams : ℕ} (raw : Transport.Square streams) : List ℝ :=
  (List.ofFn fun row : Fin streams => row).flatMap
    (fun row => List.ofFn fun column : Fin streams => raw row column)

def rawMax {streams : ℕ} (positive : 0 < streams)
    (raw : Transport.Square streams) : ℝ :=
  (rawList raw).foldl max (raw ⟨0, positive⟩ ⟨0, positive⟩)

theorem rawUpper {streams : ℕ} (positive : 0 < streams)
    (raw : Transport.Square streams) (row column : Fin streams) :
    raw row column ≤ rawMax positive raw := by
  apply FoldMax.member
  apply List.mem_flatMap.mpr
  exact ⟨row, List.mem_ofFn.mpr ⟨row, rfl⟩,
    List.mem_ofFn.mpr ⟨column, rfl⟩⟩

theorem rawWitness {streams : ℕ} (positive : 0 < streams)
    (raw : Transport.Square streams) :
    ∃ row column, rawMax positive raw = raw row column := by
  have attained := FoldMax.attained (rawList raw)
    (raw ⟨0, positive⟩ ⟨0, positive⟩)
  rcases List.mem_cons.mp attained with initial | listed
  · exact ⟨⟨0, positive⟩, ⟨0, positive⟩, initial⟩
  · rcases List.mem_flatMap.mp listed with ⟨row, _, member⟩
    rcases List.mem_ofFn.mp member with ⟨column, equal⟩
    exact ⟨row, column, equal.symm⟩

/-- The finite raw maximum is one-Lipschitz in entrywise sup error. -/
theorem maxBound {streams : ℕ} (positive : 0 < streams)
    (first second : Transport.Square streams) (delta : ℝ)
    (shift : ∀ row column, |second row column - first row column| ≤ delta) :
    |rawMax positive second - rawMax positive first| ≤ delta := by
  obtain ⟨secondRow, secondColumn, secondWitness⟩ := rawWitness positive second
  obtain ⟨firstRow, firstColumn, firstWitness⟩ := rawWitness positive first
  have secondShift := abs_le.mp (shift secondRow secondColumn)
  have firstShift := abs_le.mp (shift firstRow firstColumn)
  have firstUpper := rawUpper positive first secondRow secondColumn
  have secondUpper := rawUpper positive second firstRow firstColumn
  apply abs_le.mpr
  constructor <;> linarith

/-- The exact RMS scale used by all three source predictor groups. -/
def stateScale {streams width : ℕ} (state : Rows streams width) : ℝ :=
  1 / Real.sqrt ((∑ stream, ∑ column, state stream column ^ 2) /
    (streams * width) + 1 / 100000)

/-- State entries after the source predictor's shared RMS normalization. -/
def normState {streams width : ℕ} (state : Rows streams width) : Rows streams width :=
  fun stream column => stateScale state * state stream column

/-- The residual predictor's normalized dot product before its control and bias. -/
def residualDot {streams width : ℕ} (weights : Weights streams width)
    (state : Rows streams width) (destination source : Fin streams) : ℝ :=
  stateScale state * ∑ stream, ∑ column,
    state stream column * weights.residual stream column destination source

/-- The exact residual logits consumed by source clipping and Sinkhorn. -/
def residualRaw {streams width : ℕ} (weights : Weights streams width)
    (state : Rows streams width) : Transport.Square streams :=
  fun destination source => weights.residualBias destination source +
    weights.control 1 * residualDot weights state destination source

def predict {streams width : ℕ} (positive : 0 < streams) (iterations : ℕ)
    (weights : Weights streams width) (state : Rows streams width) : Coefficients streams :=
  let scale := stateScale state
  let input := fun destination => sigmoid (weights.inputBias destination + weights.control 0 * scale *
    ∑ stream, ∑ column, state stream column * weights.input stream column destination)
  let raw := residualRaw weights state
  let bound := rawMax positive raw
  { input := fun index => input index / ∑ other, input other
    residual := Transport.blend (min 1 (max 0 weights.lambda))
      (Transport.sinkhorn iterations (1 / 10^20) (Transport.seed bound raw))
    output := fun destination => 2 * sigmoid (weights.outputBias destination +
      weights.control 2 * scale * ∑ stream, ∑ column,
        state stream column * weights.output stream column destination) }

def mix {streams width : ℕ} (math : Arithmetic) (epsilon : ℝ)
    (norm : Vector width) (coefficients : Vector streams) (state : Rows streams width) :
    Vector width :=
  rms math epsilon norm (fun column => math.store (∑ stream, coefficients stream * state stream column))

def update {streams width : ℕ} (math : Arithmetic) (coefficients : Coefficients streams)
    (state : Rows streams width) (branch : Vector width) : Rows streams width :=
  fun destination column => math.store
    ((∑ source, coefficients.residual destination source * state source column) +
      coefficients.output destination * branch column)

#print axioms FoldMax.initial
#print axioms FoldMax.member
#print axioms FoldMax.attained
#print axioms rawUpper
#print axioms rawWitness
#print axioms maxBound

end
end BauxResearch.Forward.Mixing
