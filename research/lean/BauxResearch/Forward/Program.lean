import BauxResearch.Forward.Encoder
import BauxResearch.Forward.Mixing
import BauxResearch.Forward.Selection
import BauxResearch.Forward.Experts
import BauxResearch.Forward.Decoder

/-! Composition of the single-sequence, fresh-index real-arithmetic forward
view. Explicit source gaps: cached shadow selection, chunked KV state, F32
ordered accumulation, stable log-sum-exp, integer RNG, and parameter packing.
No arbitrary `forward` function is supplied to this composition. -/
/-!
For fixed shape and arithmetic, read `evaluate math weights input` as the
defined sequence of encoder, layer schedule, readout, and autonomous decoder
results.  A shorthand `F(θ, x)` denotes this composition only after specifying
how a full parameter state `θ` becomes `Weights` and what `x` contains.
`meanLoss` averages rollout losses; it is not itself a population objective.
-/
namespace BauxResearch.Forward.Program
noncomputable section
open scoped BigOperators

structure Shape where
  rows : ℕ
  patch : ℕ
  vocabulary : ℕ
  width : ℕ
  keyWidth : ℕ
  streams : ℕ
  heads : ℕ
  hidden : ℕ
  experts : ℕ
  active : ℕ
  layers : ℕ
  radix : ℕ
  levels : ℕ
  blockSize : ℕ
  selected : ℕ
  iterations : ℕ
  first : ℕ
  span : ℕ
  visits : ℕ
  streamPositive : 0 < streams
  radixPositive : 0 < radix
  layerPositive : 0 < layers
  headPositive : 0 < heads
  queryBound : keyWidth ≤ width

def current : Shape where
  rows := 16384
  patch := 64
  vocabulary := 262144
  width := 512
  keyWidth := 64
  streams := 4
  heads := 8
  hidden := 216
  experts := 625
  active := 3
  layers := 5
  radix := 8
  levels := 6
  blockSize := 64
  selected := 8
  iterations := 20
  first := 1
  span := 2
  visits := 2
  streamPositive := by decide
  radixPositive := by decide
  layerPositive := by decide
  headPositive := by decide
  queryBound := by decide

/-- Restrictions for this source family, separate from totalized formula definitions. -/
def Valid (shape : Shape) : Prop :=
  0 < shape.patch ∧ 0 < shape.width ∧ 0 < shape.keyWidth ∧
  0 < shape.blockSize ∧ shape.rows % shape.blockSize = 0 ∧
  (∃ depth, shape.rows / shape.blockSize = 2^depth ∧ 4 ≤ depth) ∧
  shape.vocabulary = shape.radix^shape.levels ∧ 0 < shape.levels ∧
  0 < shape.active ∧ shape.active ≤ shape.experts ∧
  shape.first + shape.span ≤ shape.layers ∧ 0 < shape.span ∧
  1 ≤ shape.visits ∧ shape.visits ≤ 4 ∧ shape.keyWidth % 2 = 0

theorem Current.valid : Valid current := by
  refine ⟨by decide, by decide, by decide, by decide, by decide,
    ⟨8, by decide, by decide⟩, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide⟩

/-- Count the declared scalar fields, respecting the readout aliases below.
Equivalence to the Rust flattened storage index remains a refinement task. -/
def parameterSlots (shape : Shape) : ℕ :=
  let mix := (shape.streams * shape.width) * (shape.streams^2 + 2 * shape.streams) +
    shape.streams^2 + 2 * shape.streams + 4 + shape.width
  let attention := shape.width * ((shape.heads + 2) * shape.keyWidth) +
    shape.heads * shape.keyWidth * shape.width
  let experts := shape.width * shape.experts +
    (shape.experts + 1) * 3 * shape.width * shape.hidden
  shape.vocabulary * shape.width + shape.patch * shape.width + shape.width +
    shape.layers * (2 * mix + attention + experts) + shape.width

theorem Current.slots : parameterSlots current = 1177756440 := by
  decide

/-- Native `fbt-pisa1-mhc4-v1` token-level parameter slot count after removing
the legacy `(patch + 1) * width` patch-position and mask embedding rows. -/
def nativeSlots (shape : Shape) : ℕ :=
  let mix := (shape.streams * shape.width) * (shape.streams^2 + 2 * shape.streams) +
    shape.streams^2 + 2 * shape.streams + 4 + shape.width
  let attention := shape.width * ((shape.heads + 2) * shape.keyWidth) +
    shape.heads * shape.keyWidth * shape.width
  let experts := shape.width * shape.experts +
    (shape.experts + 1) * 3 * shape.width * shape.hidden
  shape.vocabulary * shape.width +
    shape.layers * (2 * mix + attention + experts) + shape.width

theorem Current.nativeSlots : nativeSlots current = 1177723160 := by
  decide

theorem slotsDecomp (shape : Shape) :
    parameterSlots shape = nativeSlots shape + (shape.patch + 1) * shape.width := by
  unfold parameterSlots nativeSlots
  ring

structure Layer (shape : Shape) where
  attentionMix : Mixing.Weights shape.streams shape.width
  expertMix : Mixing.Weights shape.streams shape.width
  attention : Attention.Weights shape.width shape.heads shape.keyWidth
  experts : Experts.Weights shape.width shape.hidden shape.experts

structure Weights (shape : Shape) where
  encoder : Encoder.Weights shape.patch shape.vocabulary shape.width
  layers : Fin shape.layers → Layer shape
  finalNorm : Vector shape.width

abbrev Streams (shape : Shape) := Fin shape.rows → Rows shape.streams shape.width

def schedule (shape : Shape) : List (Fin shape.layers) :=
  let all := List.ofFn fun index : Fin shape.layers => index
  all.take shape.first ++
    (List.replicate shape.visits ((all.drop shape.first).take shape.span)).flatten ++
    all.drop (shape.first + shape.span)

theorem Schedule.current : (schedule current).map Fin.val = [0, 1, 2, 1, 2, 3, 4] := by
  decide

def layer {shape : Shape} (math : Arithmetic) (weights : Layer shape)
    (cosine sine : Fin shape.rows → ℕ → ℝ) (state : Streams shape) : Streams shape :=
  let first := fun row => Mixing.predict shape.streamPositive shape.iterations
    weights.attentionMix (state row)
  let input := fun row => Mixing.mix math (1 / 100000) weights.attentionMix.norm
    (first row).input (state row)
  let features := Attention.features math weights.attention input cosine sine
  let support := Selection.support math shape.blockSize shape.selected 12 features
  let branch := Attention.branch math (1 / Real.sqrt shape.keyWidth)
    weights.attention support features
  let carried := fun row => Mixing.update math (first row) (state row) (branch row)
  let second := fun row => Mixing.predict shape.streamPositive shape.iterations
    weights.expertMix (carried row)
  let expertInput := fun row => Mixing.mix math (1 / 100000) weights.expertMix.norm
    (second row).input (carried row)
  fun row => Mixing.update math (second row) (carried row)
    (Experts.branch math shape.active weights.experts (expertInput row))

def core {shape : Shape} (math : Arithmetic) (weights : Weights shape)
    (input : Fin shape.rows → Encoder.Patch shape.patch shape.vocabulary)
    (cosine sine : Fin shape.rows → ℕ → ℝ) : Rows shape.rows shape.width :=
  let encoded := fun row => Encoder.encode math weights.encoder (input row)
  let initial : Streams shape := fun row _ => encoded row
  let final := (schedule shape).foldl (fun state index =>
    layer math (weights.layers index) cosine sine state) initial
  fun row => Mixing.mix math (1 / 100000) weights.finalNorm
    (fun _ => 1 / shape.streams) (final row)

/-- Decoder embeddings/positions alias the encoder; the base projection aliases
the first query head of physical layer zero, as in the audited source. -/
def readout {shape : Shape} (weights : Weights shape) :
    Decoder.Weights shape.patch shape.vocabulary shape.width shape.keyWidth where
  embedding := fun token column => weights.encoder.embedding token (Fin.castLE shape.queryBound column)
  position := fun index column => weights.encoder.position index (Fin.castLE shape.queryBound column)
  projection := (weights.layers ⟨0, shape.layerPositive⟩).attention.query ⟨0, shape.headPositive⟩

structure Input (shape : Shape) where
  patches : Fin shape.rows → Encoder.Patch shape.patch shape.vocabulary
  cosine : Fin shape.rows → ℕ → ℝ
  sine : Fin shape.rows → ℕ → ℝ
  generationRows : List (Fin shape.rows)
  targets : Fin shape.rows → Fin shape.patch → Fin shape.vocabulary
  uniform : Fin shape.rows → Fin shape.patch → ℕ → ℕ → ℝ
  temperature : ℝ

def evaluate {shape : Shape} (math : Arithmetic) (weights : Weights shape)
    (input : Input shape) : List (Decoder.Result shape.keyWidth) :=
  let states := core math weights input.patches input.cosine input.sine
  input.generationRows.map fun row => Decoder.patch shape.radixPositive math shape.levels
    input.temperature (readout weights) (states row) (input.targets row) (input.uniform row)

/-- Mean loss computed directly across the autonomous generative rollouts. -/
def meanLoss {query : ℕ} (results : List (Decoder.Result query)) : ℝ :=
  let losses := results.flatMap (fun result => result.losses)
  losses.sum / losses.length

/-- Tokens generated autonomously across all evaluated rows. -/
def generatedTokens {query : ℕ} (results : List (Decoder.Result query)) : List ℕ :=
  results.flatMap (fun result => result.tokens)

theorem evaluateLength {shape : Shape} (math : Arithmetic) (weights : Weights shape)
    (input : Input shape) :
    (evaluate math weights input).length = input.generationRows.length := by
  simp [evaluate]

#print axioms evaluateLength

end
end BauxResearch.Forward.Program
