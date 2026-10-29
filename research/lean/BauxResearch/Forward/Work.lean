import BauxResearch.Forward.Program

/-! Symbolic arithmetic counts for the dense projections in this view.
These omit selector/reduction work, packing, padding, memory traffic, and kernel
scheduling. They are neither total FLOPs nor millisecond measurements. -/
namespace BauxResearch.Forward.Work

def projection (rows input output : ℕ) : ℕ := rows * input * output

def predictor (shape : Program.Shape) : ℕ :=
  projection shape.rows (shape.streams * shape.width) (shape.streams^2 + 2 * shape.streams)

def attention (shape : Program.Shape) : ℕ :=
  projection shape.rows shape.width ((shape.heads + 2) * shape.keyWidth) +
  projection shape.rows (shape.heads * shape.keyWidth) shape.width

def experts (shape : Program.Shape) (active : ℕ) : ℕ :=
  projection shape.rows shape.width shape.experts +
    (active + 1) * 3 * projection shape.rows shape.width shape.hidden

def core (shape : Program.Shape) (active : ℕ) : ℕ :=
  (Program.schedule shape).length * (2 * predictor shape + attention shape + experts shape active)

def normalizations (shape : Program.Shape) : ℕ :=
  (Program.schedule shape).length * 2 * shape.rows * shape.iterations * 2 * shape.streams^2

def encoder (shape : Program.Shape) : ℕ :=
  projection shape.rows shape.patch shape.width

def decoder (shape : Program.Shape) (patches : ℕ) : ℕ :=
  projection patches shape.width shape.keyWidth +
    patches * shape.patch * 2 * shape.levels * shape.radix * shape.keyWidth

/-- Multiply-accumulate count represented by the current exact-real forward view. -/
def roundMacs (shape : Program.Shape) : ℕ :=
  encoder shape + core shape shape.active + decoder shape shape.rows

def patchTokens : ℕ := 1

/-- Two arithmetic operations are charged for each multiply-accumulate. -/
def roundFlops (shape : Program.Shape) : ℕ :=
  (64 / patchTokens) * 2 * roundMacs shape

/-- The partial dense-contraction account derived by the Rust performance
certificate from the same source shape. -/
def contractFlops : ℕ := 35_243_427_889_152

theorem currentMacs : roundMacs Program.current = 275_339_280_384 := by
  decide

theorem currentFlops : roundFlops Program.current = 35_243_427_889_152 := by
  decide

/-- Rust and Lean now derive the same partial dense-contraction count. This is
not a total machine-work theorem: both boundaries still omit the work named in
the module documentation. -/
theorem contractAgreement :
    contractFlops = roundFlops Program.current := by
  decide

#print axioms currentMacs
#print axioms currentFlops
#print axioms contractAgreement

end BauxResearch.Forward.Work
