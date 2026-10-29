import BauxResearch.Forward.Algebra

/-! Hierarchical mHC-4 readout with autonomous token sampling and rollout prefix
states. The forward generator models genuine student rollouts without teacher forcing.
Log-sum-exp is the real-arithmetic form, not an F32 reduction implementation. -/
namespace BauxResearch.Forward.Decoder
noncomputable section
open scoped BigOperators

structure Weights (patch vocabulary width query : ℕ) where
  embedding : Fin vocabulary → Vector query
  position : Fin patch → Vector query
  projection : Fin width → Fin query → ℝ

def lookup {vocabulary query : ℕ} (embedding : Fin vocabulary → Vector query)
    (token : ℕ) : Vector query :=
  if valid : token < vocabulary then embedding ⟨token, valid⟩ else fun _ => 0

def node {vocabulary query : ℕ} (radix levels level parent child : ℕ)
    (embedding : Fin vocabulary → Vector query) : Vector query :=
  let index := parent * radix + child
  if level + 1 < levels then
    let leaves := radix ^ (levels - level - 1)
    fun column => (∑ token, if index * leaves ≤ token.val ∧ token.val < (index + 1) * leaves
      then embedding token column else 0) / Real.sqrt leaves
  else lookup embedding index

def logits {vocabulary query radix : ℕ} (levels level parent : ℕ)
    (embedding : Fin vocabulary → Vector query) (input : Vector query) : Vector radix :=
  fun child => dot input (node radix levels level parent child.val embedding)

def sampleToken {vocabulary query radix : ℕ} (positive : 0 < radix)
    (levels : ℕ) (temperature : ℝ) (uniform : ℕ → ℕ → ℝ)
    (embedding : Fin vocabulary → Vector query) (input : Vector query) : ℕ :=
  (List.range levels).foldl (fun parent level =>
    let scores := logits (radix := radix) levels level parent embedding input
    let perturbed := fun child => if temperature = 0 then scores child else
      scores child / temperature - Real.log (-Real.log (uniform level (parent * radix + child.val)))
    parent * radix + (winner positive perturbed).val) 0

def targetLoss {vocabulary query radix : ℕ} (positive : 0 < radix)
    (levels : ℕ) (target : Fin vocabulary)
    (embedding : Fin vocabulary → Vector query) (input : Vector query) : ℝ :=
  ((List.range levels).map fun level =>
    let parent := target.val / radix^(levels - level)
    let child : Fin radix := ⟨target.val / radix^(levels - level - 1) % radix,
      Nat.mod_lt _ positive⟩
    logLoss (logits levels level parent embedding input) child).sum

def advance {query : ℕ} (position embedding carried : Vector query) : Vector query :=
  fun column =>
    let retention := sigmoid (position column + 10986123 / 10000000)
    retention * carried column +
      Real.sqrt (max (1 / 1000000) (1 - retention^2)) * embedding column

structure Result (query : ℕ) where
  sampled : Vector query
  reference : Vector query
  tokens : List ℕ
  losses : List ℝ

def tokenStep {patch vocabulary width query radix : ℕ} (positive : 0 < radix)
    (math : Arithmetic) (levels : ℕ) (temperature : ℝ)
    (weights : Weights patch vocabulary width query) (base : Vector query)
    (targets : Fin patch → Fin vocabulary) (uniform : Fin patch → ℕ → ℕ → ℝ)
    (state : Result query) (index : Fin patch) : Result query :=
    let sampled := fun column => math.store (base column + weights.position index column +
      state.sampled column)
    let reference := fun column => math.store (base column + weights.position index column +
      state.reference column)
    let token := sampleToken positive levels temperature (uniform index) weights.embedding sampled
    { sampled := advance (weights.position index) (lookup weights.embedding token) state.sampled
      reference := advance (weights.position index) (weights.embedding (targets index)) state.reference
      tokens := state.tokens ++ [token]
      losses := state.losses ++ [targetLoss positive levels (targets index) weights.embedding reference] }

def patch {patch vocabulary width query radix : ℕ} (positive : 0 < radix)
    (math : Arithmetic) (levels : ℕ) (temperature : ℝ)
    (weights : Weights patch vocabulary width query) (coarse : Vector width)
    (targets : Fin patch → Fin vocabulary) (uniform : Fin patch → ℕ → ℕ → ℝ) : Result query :=
  let base := fun column => (∑ index, coarse index * weights.projection index column) / Real.sqrt width
  (List.ofFn fun index : Fin patch => index).foldl
    (tokenStep positive math levels temperature weights base targets uniform)
    { sampled := fun _ => 0, reference := fun _ => 0, tokens := [], losses := [] }

theorem Target.nonnegative {vocabulary query radix : ℕ} (positive : 0 < radix)
    (levels : ℕ) (target : Fin vocabulary)
    (embedding : Fin vocabulary → Vector query) (input : Vector query) :
    0 ≤ targetLoss positive levels target embedding input := by
  unfold targetLoss
  apply List.sum_nonneg
  intro value member
  obtain ⟨level, _, rfl⟩ := List.mem_map.mp member
  exact Loss.nonnegative _ _

#print axioms Target.nonnegative
end
end BauxResearch.Forward.Decoder
