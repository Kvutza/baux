import BauxResearch.Forward.Program
import BauxResearch.Forward.Regularity
import BauxResearch.Foundation
import Mathlib.Data.Fintype.EquivFin
import Mathlib.Tactic.DeriveFintype

/-!
# Every independent parameter of the real forward view

Typed indices enumerate actual tensor fields, including biases, controls, norms,
and shared experts. Decoder aliases do not create extra independent parameters.
Mathlib finite-type cardinality and equivalences supply the scalar-coordinate
view. The generic Fin numbering is an analytical enumeration, NOT the Rust
buffer order, CUDA transpose/packing, or a verified hardware storage layout.
-/
namespace BauxResearch.Forward.Parameters
noncomputable section

inductive EncoderIndex (patch vocabulary width : ℕ)
  | embedding (token : Fin vocabulary) (column : Fin width)
  | position (index : Fin patch) (column : Fin width)
  | mask (column : Fin width)
  deriving DecidableEq, Fintype

inductive MixIndex (streams width : ℕ)
  | input (stream : Fin streams) (column : Fin width) (destination : Fin streams)
  | residual (stream : Fin streams) (column : Fin width) (destination source : Fin streams)
  | output (stream : Fin streams) (column : Fin width) (destination : Fin streams)
  | inputBias (destination : Fin streams)
  | residualBias (destination source : Fin streams)
  | outputBias (destination : Fin streams)
  | control (index : Fin 3)
  | blend
  | norm (column : Fin width)
  deriving DecidableEq, Fintype

inductive AttentionIndex (width heads keyWidth : ℕ)
  | query (head : Fin heads) (row : Fin width) (column : Fin keyWidth)
  | key (row : Fin width) (column : Fin keyWidth)
  | value (row : Fin width) (column : Fin keyWidth)
  | output (head : Fin heads) (row : Fin keyWidth) (column : Fin width)
  deriving DecidableEq, Fintype

inductive ExpertIndex (width hidden : ℕ)
  | gate (row : Fin width) (column : Fin hidden)
  | up (row : Fin width) (column : Fin hidden)
  | down (row : Fin hidden) (column : Fin width)
  deriving DecidableEq, Fintype

inductive ExpertsIndex (width hidden experts : ℕ)
  | router (row : Fin width) (column : Fin experts)
  | routed (expert : Fin experts) (index : ExpertIndex width hidden)
  | shared (index : ExpertIndex width hidden)
  deriving DecidableEq, Fintype

inductive LayerIndex (shape : Program.Shape)
  | attentionMix (index : MixIndex shape.streams shape.width)
  | expertMix (index : MixIndex shape.streams shape.width)
  | attention (index : AttentionIndex shape.width shape.heads shape.keyWidth)
  | experts (index : ExpertsIndex shape.width shape.hidden shape.experts)
  deriving DecidableEq, Fintype

inductive Index (shape : Program.Shape)
  | encoder (index : EncoderIndex shape.patch shape.vocabulary shape.width)
  | layer (layer : Fin shape.layers) (index : LayerIndex shape)
  | finalNorm (column : Fin shape.width)
  deriving DecidableEq, Fintype

def encoderRead {patch vocabulary width : ℕ} (weights : Encoder.Weights patch vocabulary width) :
    EncoderIndex patch vocabulary width → ℝ
  | .embedding token column => weights.embedding token column
  | .position index column => weights.position index column
  | .mask column => weights.mask column

def encoderBuild {patch vocabulary width : ℕ} (values : EncoderIndex patch vocabulary width → ℝ) :
    Encoder.Weights patch vocabulary width where
  embedding := fun token column => values (.embedding token column)
  position := fun index column => values (.position index column)
  mask := fun column => values (.mask column)

def mixRead {streams width : ℕ} (weights : Mixing.Weights streams width) : MixIndex streams width → ℝ
  | .input stream column destination => weights.input stream column destination
  | .residual stream column destination source => weights.residual stream column destination source
  | .output stream column destination => weights.output stream column destination
  | .inputBias destination => weights.inputBias destination
  | .residualBias destination source => weights.residualBias destination source
  | .outputBias destination => weights.outputBias destination
  | .control index => weights.control index
  | .blend => weights.lambda
  | .norm column => weights.norm column

def mixBuild {streams width : ℕ} (values : MixIndex streams width → ℝ) : Mixing.Weights streams width where
  input := fun stream column destination => values (.input stream column destination)
  residual := fun stream column destination source => values (.residual stream column destination source)
  output := fun stream column destination => values (.output stream column destination)
  inputBias := fun destination => values (.inputBias destination)
  residualBias := fun destination source => values (.residualBias destination source)
  outputBias := fun destination => values (.outputBias destination)
  control := fun index => values (.control index)
  lambda := values .blend
  norm := fun column => values (.norm column)

def attentionRead {width heads keyWidth : ℕ} (weights : Attention.Weights width heads keyWidth) :
    AttentionIndex width heads keyWidth → ℝ
  | .query head row column => weights.query head row column
  | .key row column => weights.key row column
  | .value row column => weights.value row column
  | .output head row column => weights.output head row column

def attentionBuild {width heads keyWidth : ℕ} (values : AttentionIndex width heads keyWidth → ℝ) :
    Attention.Weights width heads keyWidth where
  query := fun head row column => values (.query head row column)
  key := fun row column => values (.key row column)
  value := fun row column => values (.value row column)
  output := fun head row column => values (.output head row column)

def expertRead {width hidden : ℕ} (weights : Experts.Expert width hidden) : ExpertIndex width hidden → ℝ
  | .gate row column => weights.gate row column
  | .up row column => weights.up row column
  | .down row column => weights.down row column

def expertBuild {width hidden : ℕ} (values : ExpertIndex width hidden → ℝ) : Experts.Expert width hidden where
  gate := fun row column => values (.gate row column)
  up := fun row column => values (.up row column)
  down := fun row column => values (.down row column)

def expertsRead {width hidden experts : ℕ} (weights : Experts.Weights width hidden experts) :
    ExpertsIndex width hidden experts → ℝ
  | .router row column => weights.router row column
  | .routed expert index => expertRead (weights.routed expert) index
  | .shared index => expertRead weights.shared index

def expertsBuild {width hidden experts : ℕ} (values : ExpertsIndex width hidden experts → ℝ) :
    Experts.Weights width hidden experts where
  router := fun row column => values (.router row column)
  routed := fun expert => expertBuild (fun index => values (.routed expert index))
  shared := expertBuild (fun index => values (.shared index))

def layerRead {shape : Program.Shape} (weights : Program.Layer shape) : LayerIndex shape → ℝ
  | .attentionMix index => mixRead weights.attentionMix index
  | .expertMix index => mixRead weights.expertMix index
  | .attention index => attentionRead weights.attention index
  | .experts index => expertsRead weights.experts index

def layerBuild {shape : Program.Shape} (values : LayerIndex shape → ℝ) : Program.Layer shape where
  attentionMix := mixBuild (fun index => values (.attentionMix index))
  expertMix := mixBuild (fun index => values (.expertMix index))
  attention := attentionBuild (fun index => values (.attention index))
  experts := expertsBuild (fun index => values (.experts index))

def read {shape : Program.Shape} (weights : Program.Weights shape) : Index shape → ℝ
  | .encoder index => encoderRead weights.encoder index
  | .layer layer index => layerRead (weights.layers layer) index
  | .finalNorm column => weights.finalNorm column

def build {shape : Program.Shape} (values : Index shape → ℝ) : Program.Weights shape where
  encoder := encoderBuild (fun index => values (.encoder index))
  layers := fun layer => layerBuild (fun index => values (.layer layer index))
  finalNorm := fun column => values (.finalNorm column)

@[simp] theorem Encoder.rebuild {patch vocabulary width : ℕ}
    (weights : Encoder.Weights patch vocabulary width) : encoderBuild (encoderRead weights) = weights := by
  cases weights
  rfl

@[simp] theorem Mix.rebuild {streams width : ℕ} (weights : Mixing.Weights streams width) :
    mixBuild (mixRead weights) = weights := by cases weights; rfl

@[simp] theorem Attention.rebuild {width heads keyWidth : ℕ} (weights : Attention.Weights width heads keyWidth) :
    attentionBuild (attentionRead weights) = weights := by cases weights; rfl

@[simp] theorem Expert.rebuild {width hidden : ℕ} (weights : Experts.Expert width hidden) :
    expertBuild (expertRead weights) = weights := by cases weights; rfl

@[simp] theorem Experts.rebuild {width hidden experts : ℕ} (weights : Experts.Weights width hidden experts) :
    expertsBuild (expertsRead weights) = weights := by
  cases weights
  simp only [expertsBuild, expertsRead, Expert.rebuild]

@[simp] theorem Layer.rebuild {shape : Program.Shape} (weights : Program.Layer shape) :
    layerBuild (layerRead weights) = weights := by
  cases weights
  simp only [layerBuild, layerRead, Mix.rebuild, Attention.rebuild, Experts.rebuild]

@[simp] theorem rebuild {shape : Program.Shape} (weights : Program.Weights shape) :
    build (read weights) = weights := by
  cases weights
  simp only [build, read, Encoder.rebuild, Layer.rebuild]

@[simp] theorem reread {shape : Program.Shape} (values : Index shape → ℝ) : read (build values) = values := by
  funext index
  cases index with
  | encoder index => cases index <;> rfl
  | finalNorm column => rfl
  | layer layer index =>
    cases index with
    | attentionMix index => cases index <;> rfl
    | expertMix index => cases index <;> rfl
    | attention index => cases index <;> rfl
    | experts index =>
      cases index with
      | router row column => rfl
      | shared index => cases index <;> rfl
      | routed expert index => cases index <;> rfl

/-- Every independent scalar is represented once in the structured real view. -/
def coordinates (shape : Program.Shape) : Program.Weights shape ≃ (Index shape → ℝ) where
  toFun := read
  invFun := build
  left_inv := rebuild
  right_inv := reread

def perturb {shape : Program.Shape} (weights : Program.Weights shape) (delta : Index shape → ℝ) :
    Program.Weights shape := build (fun index => read weights index + delta index)

theorem Perturb.read {shape : Program.Shape} (weights : Program.Weights shape) (delta : Index shape → ℝ) :
    read (perturb weights delta) = fun index => read weights index + delta index := reread _

theorem Alias.embedding {shape : Program.Shape} (weights : Program.Weights shape)
    (token : Fin shape.vocabulary) (column : Fin shape.keyWidth) :
    (Program.readout weights).embedding token column =
      read weights (.encoder (.embedding token (Fin.castLE shape.queryBound column))) := rfl

theorem Alias.position {shape : Program.Shape} (weights : Program.Weights shape)
    (index : Fin shape.patch) (column : Fin shape.keyWidth) :
    (Program.readout weights).position index column =
      read weights (.encoder (.position index (Fin.castLE shape.queryBound column))) := rfl

theorem Alias.projection {shape : Program.Shape} (weights : Program.Weights shape)
    (row : Fin shape.width) (column : Fin shape.keyWidth) :
    (Program.readout weights).projection row column =
      read weights (.layer ⟨0, shape.layerPositive⟩
        (.attention (.query ⟨0, shape.headPositive⟩ row column))) := rfl

/-- A logical encoder update changes its tied decoder view by the same scalar. -/
theorem Alias.perturbed {shape : Program.Shape} (weights : Program.Weights shape)
    (delta : Index shape → ℝ) (token : Fin shape.vocabulary) (column : Fin shape.keyWidth) :
    (Program.readout (perturb weights delta)).embedding token column =
      (Program.readout weights).embedding token column +
        delta (.encoder (.embedding token (Fin.castLE shape.queryBound column))) := rfl

theorem Mix.card (streams width : ℕ) :
    Fintype.card (MixIndex streams width) =
      (streams * width) * (streams^2 + 2 * streams) + streams^2 + 2 * streams + 4 + width := by
  rw [← Fintype.card_congr (proxy_equiv% MixIndex streams width)]
  simp
  ring

theorem Attention.card (width heads keyWidth : ℕ) :
    Fintype.card (AttentionIndex width heads keyWidth) =
      width * ((heads + 2) * keyWidth) + heads * keyWidth * width := by
  rw [← Fintype.card_congr (proxy_equiv% AttentionIndex width heads keyWidth)]
  simp
  ring

theorem Expert.card (width hidden : ℕ) :
    Fintype.card (ExpertIndex width hidden) = 3 * width * hidden := by
  rw [← Fintype.card_congr (proxy_equiv% ExpertIndex width hidden)]
  simp
  ring

theorem Experts.card (width hidden experts : ℕ) :
    Fintype.card (ExpertsIndex width hidden experts) =
      width * experts + (experts + 1) * 3 * width * hidden := by
  rw [← Fintype.card_congr (proxy_equiv% ExpertsIndex width hidden experts)]
  simp [Expert.card]
  ring

theorem Encoder.card (patch vocabulary width : ℕ) :
    Fintype.card (EncoderIndex patch vocabulary width) = vocabulary * width + patch * width + width := by
  rw [← Fintype.card_congr (proxy_equiv% EncoderIndex patch vocabulary width)]
  simp
  ring

theorem Layer.card (shape : Program.Shape) : Fintype.card (LayerIndex shape) =
    2 * ((shape.streams * shape.width) * (shape.streams^2 + 2 * shape.streams) +
      shape.streams^2 + 2 * shape.streams + 4 + shape.width) +
    (shape.width * ((shape.heads + 2) * shape.keyWidth) + shape.heads * shape.keyWidth * shape.width) +
    (shape.width * shape.experts + (shape.experts + 1) * 3 * shape.width * shape.hidden) := by
  rw [← Fintype.card_congr (proxy_equiv% LayerIndex shape)]
  simp [Mix.card, Attention.card, Experts.card]
  ring

theorem cardinality (shape : Program.Shape) : Fintype.card (Index shape) = Program.parameterSlots shape := by
  rw [← Fintype.card_congr (proxy_equiv% Index shape)]
  simp [Fintype.card_sigma, Program.parameterSlots, Encoder.card, Layer.card]
  ring

def numbering (shape : Program.Shape) : Index shape ≃ Fin (Program.parameterSlots shape) :=
  Fintype.equivFinOfCardEq (cardinality shape)

/-- This enumeration connects the full Euclidean analysis state to actual
tensor fields. It does not certify backend address order or numeric execution. -/
def fromFlat {shape : Program.Shape} (state : Parameter.State (Program.parameterSlots shape)) :
    Program.Weights shape := build (fun index => state (numbering shape index))

def toFlat {shape : Program.Shape} (weights : Program.Weights shape) :
    Parameter.State (Program.parameterSlots shape) :=
  WithLp.toLp 2 (fun index => read weights ((numbering shape).symm index))

theorem flatRoundtrip {shape : Program.Shape} (state : Parameter.State (Program.parameterSlots shape)) :
    toFlat (fromFlat state) = state := by
  apply (WithLp.equiv 2 _).injective
  funext index
  simp [toFlat, fromFlat, reread]

theorem weightRoundtrip {shape : Program.Shape} (weights : Program.Weights shape) :
    fromFlat (toFlat weights) = weights := by
  simp [fromFlat, toFlat]

def flatEquiv (shape : Program.Shape) : Parameter.State (Program.parameterSlots shape) ≃ Program.Weights shape where
  toFun := fromFlat
  invFun := toFlat
  left_inv := flatRoundtrip
  right_inv := weightRoundtrip

/-- Every actual tensor coordinate is a smooth linear projection of the full
parameter vector. This does not prove smoothness through routing or clipping. -/
theorem coordinateSmooth (shape : Program.Shape) (index : Index shape) :
    ContDiff ℝ ⊤ (fun state : Parameter.State (Program.parameterSlots shape) =>
      read (fromFlat state) index) := by
  have linear := (EuclideanSpace.proj (numbering shape index) :
    Parameter.State (Program.parameterSlots shape) →L[ℝ] ℝ).contDiff (n := ⊤)
  simpa only [fromFlat, reread, EuclideanSpace.coe_proj] using linear

theorem Current.cardinality : Fintype.card (Index Program.current) = 1177756440 :=
  (Parameters.cardinality Program.current).trans Program.Current.slots

/-- The 17 architectural parameter families of `fbt-pisa1-mhc4-v1` (Section 6.2
of `TYPED_TENSOR_INFERENCE_AND_OPTIMIZER_ARCHITECTURE.txt`). -/
inductive ParamFamily
  | tokenEmbed
  | mhcAlpha
  | mhcPre
  | mhcPost
  | mhcMix
  | attnNorm
  | attnQ
  | attnK
  | attnV
  | attnO
  | ffnNorm
  | router
  | sharedGate
  | sharedUp
  | sharedDown
  | routedGateUp
  | routedDown
  deriving DecidableEq

def ParamFamily.all : List ParamFamily :=
  [.tokenEmbed, .mhcAlpha, .mhcPre, .mhcPost, .mhcMix, .attnNorm, .attnQ, .attnK,
   .attnV, .attnO, .ffnNorm, .router, .sharedGate, .sharedUp, .sharedDown,
   .routedGateUp, .routedDown]

theorem ParamFamily.card : ParamFamily.all.length = 17 ∧
    ∀ f : ParamFamily, f ∈ ParamFamily.all := by
  refine ⟨rfl, ?_⟩
  intro f
  cases f <;> decide

/-- Typed 17-family parameter PyTree (`ParamTree<T>`). -/
structure ParamTree (α : Type*) where
  tokenEmbed : α
  mhcAlpha : α
  mhcPre : α
  mhcPost : α
  mhcMix : α
  attnNorm : α
  attnQ : α
  attnK : α
  attnV : α
  attnO : α
  ffnNorm : α
  router : α
  sharedGate : α
  sharedUp : α
  sharedDown : α
  routedGateUp : α
  routedDown : α

def ParamTree.get {α : Type*} (tree : ParamTree α) : ParamFamily → α
  | .tokenEmbed => tree.tokenEmbed
  | .mhcAlpha => tree.mhcAlpha
  | .mhcPre => tree.mhcPre
  | .mhcPost => tree.mhcPost
  | .mhcMix => tree.mhcMix
  | .attnNorm => tree.attnNorm
  | .attnQ => tree.attnQ
  | .attnK => tree.attnK
  | .attnV => tree.attnV
  | .attnO => tree.attnO
  | .ffnNorm => tree.ffnNorm
  | .router => tree.router
  | .sharedGate => tree.sharedGate
  | .sharedUp => tree.sharedUp
  | .sharedDown => tree.sharedDown
  | .routedGateUp => tree.routedGateUp
  | .routedDown => tree.routedDown

def ParamTree.ofFn {α : Type*} (f : ParamFamily → α) : ParamTree α where
  tokenEmbed := f .tokenEmbed
  mhcAlpha := f .mhcAlpha
  mhcPre := f .mhcPre
  mhcPost := f .mhcPost
  mhcMix := f .mhcMix
  attnNorm := f .attnNorm
  attnQ := f .attnQ
  attnK := f .attnK
  attnV := f .attnV
  attnO := f .attnO
  ffnNorm := f .ffnNorm
  router := f .router
  sharedGate := f .sharedGate
  sharedUp := f .sharedUp
  sharedDown := f .sharedDown
  routedGateUp := f .routedGateUp
  routedDown := f .routedDown

@[simp] theorem ParamTree.get_ofFn {α : Type*} (f : ParamFamily → α) (family : ParamFamily) :
    (ParamTree.ofFn f).get family = f family := by
  cases family <;> rfl

@[simp] theorem ParamTree.ofFn_get {α : Type*} (tree : ParamTree α) :
    ParamTree.ofFn tree.get = tree := by
  cases tree
  rfl

def ParamTree.equiv (α : Type*) : ParamTree α ≃ (ParamFamily → α) where
  toFun := ParamTree.get
  invFun := ParamTree.ofFn
  left_inv := ParamTree.ofFn_get
  right_inv := fun f => funext (ParamTree.get_ofFn f)

/-- Classify any mHC `MixIndex` into its architectural `ParamFamily`, Given
whether the enclosing block is the attention mix (`isAttn = true`) or FFN mix. -/
def mixFamily (isAttn : Bool) {streams width : ℕ} : MixIndex streams width → ParamFamily
  | .input _ _ _ => .mhcPre
  | .inputBias _ => .mhcPre
  | .output _ _ _ => .mhcPost
  | .outputBias _ => .mhcPost
  | .residual _ _ _ _ => .mhcMix
  | .residualBias _ _ => .mhcMix
  | .control _ => .mhcAlpha
  | .blend => .mhcAlpha
  | .norm _ => if isAttn then .attnNorm else .ffnNorm

/-- Classify every independent coordinate of `Index shape` into its unique
architectural `ParamFamily`. -/
def familyOf {shape : Program.Shape} : Index shape → ParamFamily
  | .encoder _ => .tokenEmbed
  | .finalNorm _ => .ffnNorm
  | .layer _ (.attentionMix idx) => mixFamily true idx
  | .layer _ (.expertMix idx) => mixFamily false idx
  | .layer _ (.attention (.query _ _ _)) => .attnQ
  | .layer _ (.attention (.key _ _)) => .attnK
  | .layer _ (.attention (.value _ _)) => .attnV
  | .layer _ (.attention (.output _ _ _)) => .attnO
  | .layer _ (.experts (.router _ _)) => .router
  | .layer _ (.experts (.shared (.gate _ _))) => .sharedGate
  | .layer _ (.experts (.shared (.up _ _))) => .sharedUp
  | .layer _ (.experts (.shared (.down _ _))) => .sharedDown
  | .layer _ (.experts (.routed _ (.gate _ _))) => .routedGateUp
  | .layer _ (.experts (.routed _ (.up _ _))) => .routedGateUp
  | .layer _ (.experts (.routed _ (.down _ _))) => .routedDown

/-- Apply per-family perturbation scales from `ParamTree ℝ` to a parameter
displacement vector `delta : Index shape → ℝ`. -/
def familyScale {shape : Program.Shape} (scales : ParamTree ℝ)
    (delta : Index shape → ℝ) : Index shape → ℝ :=
  fun index => scales.get (familyOf index) * delta index

/-- Freezing a parameter family (`scales.get f = 0`) guarantees zero update on
every coordinate belonging to that family. -/
theorem frozenZero {shape : Program.Shape} (scales : ParamTree ℝ)
    (delta : Index shape → ℝ) (family : ParamFamily)
    (frozen : scales.get family = 0) (index : Index shape)
    (belongs : familyOf index = family) :
    familyScale scales delta index = 0 := by
  simp [familyScale, belongs, frozen]

/-- Canonical generative rollout objective evaluated directly without teacher forcing. -/
def objective (shape : Program.Shape) (input : Program.Input shape)
    (state : Parameter.State (Program.parameterSlots shape)) : ℝ :=
  Program.meanLoss (Program.evaluate Arithmetic.exact (fromFlat state) input)

#print axioms coordinates
#print axioms cardinality
#print axioms flatEquiv
#print axioms coordinateSmooth
#print axioms Alias.perturbed
#print axioms ParamFamily.card
#print axioms ParamTree.equiv
#print axioms frozenZero
end
end BauxResearch.Forward.Parameters
