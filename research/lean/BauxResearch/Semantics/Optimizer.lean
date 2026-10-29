import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Optimization.EnnCommit
import BauxResearch.Optimization.Reliability
import BauxResearch.Semantics.Tell

namespace BauxResearch.Semantics.Optimizer

/-!
# Executable optimizer semantics

The generated definition in this module is extracted from the production Rust
implementation by Charon and Aeneas.  The handwritten definition remains the
mathematical interface used by the rest of the formalization.  `actionMatch`
checks all Boolean controller states and proves that the two definitions agree.
-/


private def signMask : BitVec 64 := BitVec.ofNat 64 0x8000000000000000
private def expMask : BitVec 64 := BitVec.ofNat 64 0x7ff0000000000000
private def fracMask : BitVec 64 := BitVec.ofNat 64 0x000fffffffffffff

private def nanBits (bits : BitVec 64) : Bool :=
  decide ((bits &&& expMask) = expMask ∧ (bits &&& fracMask) ≠ 0)

private def zeroBits (bits : BitVec 64) : Bool :=
  decide ((bits &&& ~~~signMask) = 0)

private def orderBits (bits : BitVec 64) : BitVec 64 :=
  if bits &&& signMask = 0 then bits ||| signMask else ~~~bits

/-- IEEE-754 binary64 strict order, including unordered NaNs and equal signed
zeros, stated independently of the generated Rust translation. -/
def acceptBits (improvement margin : BitVec 64) : Bool :=
  if nanBits improvement || nanBits margin then false
  else if zeroBits improvement && zeroBits margin then false
  else decide (orderBits margin < orderBits improvement)

private theorem signMatch :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.SIGN_MASK.bv = signMask := by
  unfold BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.SIGN_MASK
  decide

private theorem expMatch :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.EXP_MASK.bv = expMask := by
  unfold BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.EXP_MASK
  decide

private theorem fracMatch :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.FRAC_MASK.bv = fracMask := by
  unfold BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.FRAC_MASK
  decide

private theorem nanMatch (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.nan_bits bits =
      Aeneas.Std.Result.ok (nanBits bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.nan_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.EXP_MASK =
      (⟨expMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ expMatch]
  rw [show BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.FRAC_MASK =
      (⟨fracMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ fracMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, nanBits]
  split <;> simp_all [bne, BEq.beq]

private theorem zeroMatch (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.zero_bits bits =
      Aeneas.Std.Result.ok (zeroBits bits.bv) := by
  simp only [BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.zero_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.SIGN_MASK =
      (⟨signMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ signMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, zeroBits]

private theorem orderMatch (bits : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.order_bits bits =
      Aeneas.Std.Result.ok (⟨orderBits bits.bv⟩ : Aeneas.Std.U64) := by
  simp only [BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.order_bits,
    Aeneas.Std.lift, Aeneas.Std.bind_ok]
  rw [show BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.SIGN_MASK =
      (⟨signMask⟩ : Aeneas.Std.U64) by exact Aeneas.Std.U64.bv_eq_imp_eq _ _ signMatch]
  simp [Aeneas.Std.UScalar.eq_equiv_bv_eq, orderBits]
  split <;> simp_all [Aeneas.Std.UScalar.eq_equiv_bv_eq]

private theorem bitLt (left right : BitVec 64) :
    ((↑(⟨left⟩ : Aeneas.Std.U64) : Nat) < ↑(⟨right⟩ : Aeneas.Std.U64)) =
      (left < right) := by
  rfl

/-- The integer-only comparison extracted from production Rust implements the
independent binary64 bit-level specification for every pair of words. -/
theorem bitsMatch (improvement margin : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.accept_bits improvement margin =
      Aeneas.Std.Result.ok (acceptBits improvement.bv margin.bv) := by
  simp [BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.accept_bits,
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.accept_decision,
    nanMatch, zeroMatch, orderMatch, bitLt, acceptBits]
  cases hni : nanBits improvement.bv <;>
    cases hnm : nanBits margin.bv <;>
    cases hzi : zeroBits improvement.bv <;>
    cases hzm : zeroBits margin.bv <;>
    cases hlt : decide (orderBits margin.bv < orderBits improvement.bv) <;>
    simp_all

/-- The extracted production commit boundary agrees with the Mathlib-facing
acceptance law once the runtime comparison has supplied its Boolean result. -/
theorem acceptMatch (improvement margin : ℝ) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.accept_decision
      (decide (BauxResearch.Optimization.EnnCommit.accepts improvement margin)) =
        Aeneas.Std.Result.ok
          (decide (BauxResearch.Optimization.EnnCommit.accepts improvement margin)) := by
  rfl

abbrev CodeAction :=
  BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.RadiusAction

/-- Translate the handwritten action vocabulary to the Rust model's vocabulary. -/
def codeAction : BauxResearch.Optimization.ReliabilityAction → CodeAction
  | .hold => .Hold
  | .expand => .Expand
  | .shrink => .Shrink

/-- The Aeneas model of the production Rust decision law equals its handwritten
Mathlib-facing specification for every controller state. -/
theorem actionMatch
    (modelAdequate modelInadequate sharp accepted positiveRatio boundaryStep failure : Bool) :
    BauxResearch.Generated.Optimizer.optimizer.trust_region.controller.decide_action
      modelAdequate modelInadequate sharp accepted positiveRatio
      boundaryStep failure =
        Aeneas.Std.Result.ok (codeAction (BauxResearch.Optimization.decideAction
          modelAdequate modelInadequate sharp accepted positiveRatio boundaryStep failure)) := by
  cases modelAdequate <;> cases modelInadequate <;> cases sharp <;> cases accepted <;>
    cases positiveRatio <;> cases boundaryStep <;> cases failure <;> rfl

/-- Bit-level 64-bit L-order cell key packing (`arXiv:2305.01356`):
high 32 bits store biased scale level `level`, low 32 bits store horizontal grid index `grid`. -/
def cellKey (level grid : BitVec 32) : BitVec 64 :=
  (level.zeroExtend 64 <<< 32) ||| grid.zeroExtend 64

def cellLevel (key : BitVec 64) : BitVec 32 :=
  (key >>> 32).truncate 32

def cellGrid (key : BitVec 64) : BitVec 32 :=
  key.truncate 32

theorem cellMatch (level grid : BitVec 32) :
    cellLevel (cellKey level grid) = level ∧
    cellGrid (cellKey level grid) = grid := by
  unfold cellLevel cellGrid cellKey
  constructor <;> bv_decide

/-- IEEE-754 binary64 strict inequality is irreflexive on all 64-bit words,
including signed zeros, infinities, subnormals, and all NaN payloads. -/
theorem acceptIrrefl (word : BitVec 64) :
    acceptBits word word = false := by
  unfold acceptBits nanBits zeroBits orderBits signMask expMask fracMask
  bv_decide

/-- IEEE-754 binary64 strict inequality is asymmetric across all 64-bit word pairs. -/
theorem acceptAsymm (left right : BitVec 64) :
    acceptBits left right = true → acceptBits right left = false := by
  unfold acceptBits nanBits zeroBits orderBits signMask expMask fracMask
  bv_decide

/-- Any IEEE-754 NaN word on either side is rejected by `acceptBits`. -/
theorem nanReject (left right : BitVec 64) :
    nanBits left = true → acceptBits left right = false ∧ acceptBits right left = false := by
  unfold acceptBits nanBits zeroBits orderBits signMask expMask fracMask
  bv_decide

/-- Positive zero (`+0.0`) and negative zero (`-0.0`) compare equal and never trigger acceptance. -/
theorem zeroEqual :
    acceptBits (BitVec.ofNat 64 0) (BitVec.ofNat 64 0x8000000000000000) = false ∧
    acceptBits (BitVec.ofNat 64 0x8000000000000000) (BitVec.ofNat 64 0) = false := by
  native_decide

/-- Aeneas-extracted `provenance::same_world` implements exact `WorldId` equality (P2/P3). -/
theorem worldMatch (left right : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.provenance.same_world left right =
      Aeneas.Std.Result.ok (decide (left = right)) := by
  simp [BauxResearch.Generated.Optimizer.optimizer.provenance.same_world]

/-- Aeneas-extracted `provenance::exact_anchored` classifies rotating-window steps
with `from_step ≠ to_step` as drifted chains (`WindowPath.errorSplit`). -/
theorem chainMatch (isFixed : Bool) (fromStep toStep : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.provenance.exact_anchored isFixed fromStep toStep =
      Aeneas.Std.Result.ok (isFixed || decide (fromStep = toStep)) := by
  cases isFixed <;> simp [BauxResearch.Generated.Optimizer.optimizer.provenance.exact_anchored]

/-- Aeneas-extracted `param_tree::routed_count` matches the 5-layer `fbt-pisa1-mhc4-v1`
routed MoE parameter count `1_036_800_000` (P4/P5). -/
theorem routedMatch :
    BauxResearch.Generated.Optimizer.optimizer.param_tree.routed_count =
      Aeneas.Std.Result.ok 1036800000#u64 := by
  rfl

/-- Aeneas-extracted `param_tree::total_count` matches the native `fbt-pisa1-mhc4-v1`
total parameter count `1_177_723_160` at `patch_size = 1` (P4/P5). -/
theorem totalMatch :
    BauxResearch.Generated.Optimizer.optimizer.param_tree.total_count =
      Aeneas.Std.Result.ok 1177723160#u64 := by
  rfl

/-- Aeneas-extracted `window_plan::verify_full` requires `planned > 0`, `prefilled = 0`,
and `scored = planned`. -/
theorem windowMatch (planned prefilled scored : Aeneas.Std.U64) :
    BauxResearch.Generated.Optimizer.optimizer.window_plan.verify_full planned prefilled scored =
      Aeneas.Std.Result.ok
        (decide (planned > 0#u64) && decide (prefilled = 0#u64) && decide (scored = planned)) := by
  unfold BauxResearch.Generated.Optimizer.optimizer.window_plan.verify_full
  by_cases hp : planned > 0#u64 <;> by_cases hpre : prefilled = 0#u64 <;> simp [hp, hpre]

/-- The extracted window-receipt verifier rejects the `2048` prefill + `1024` scored
token-skipping shortcut on a `1,048,576`-token window. -/
theorem shortcutReject :
    BauxResearch.Generated.Optimizer.optimizer.window_plan.verify_full 1048576#u64 2048#u64 1024#u64 =
      Aeneas.Std.Result.ok false := by
  rfl

/-- Aeneas-extracted `reuse::conditioned_slot` preserves `slot &&& 3` on acceptance
and flips the low antipodal bit `(slot &&& 3) ^^^ 1` on rejection. -/
theorem slotMatch (slot : Aeneas.Std.U64) (accepted : Bool) :
    BauxResearch.Generated.Optimizer.optimizer.reuse.conditioned_slot slot accepted =
      Aeneas.Std.Result.ok
        (if accepted then slot &&& 3#u64 else (slot &&& 3#u64) ^^^ 1#u64) := by
  cases accepted <;>
    simp [BauxResearch.Generated.Optimizer.optimizer.reuse.conditioned_slot,
      Aeneas.Std.lift, Aeneas.Std.bind_ok]

/-- Two consecutive antipodal rejections on the same stream involutively restore
the original masked slot `slot.bv &&& 3#64`. -/
theorem slotFlip (slot : BitVec 64) :
    (((slot &&& 3#64) ^^^ 1#64) &&& 3#64) ^^^ 1#64 = slot &&& 3#64 := by
  bv_decide

/-- Aeneas-extracted `reuse::conditioned_sign` preserves `reversed` on acceptance
and negates `!reversed` on rejection (`Guide.lean`: `reverseProgress`, `signedProgress`). -/
theorem dirMatch (reversed accepted : Bool) :
    BauxResearch.Generated.Optimizer.optimizer.reuse.conditioned_sign reversed accepted =
      Aeneas.Std.Result.ok (if accepted then reversed else !reversed) := by
  cases accepted <;> simp [BauxResearch.Generated.Optimizer.optimizer.reuse.conditioned_sign]

/-- Two consecutive sign reversals involutively restore the original direction sign. -/
theorem dirFlip (reversed : Bool) :
    (!(!reversed)) = reversed := by
  cases reversed <;> rfl

/-- C-ABI exported Lean 4 scalar acceptance gate (`@[export baux_accept]`). -/
@[export baux_accept]
def exportAccept (improvement margin : UInt64) : Bool :=
  let s : UInt64 := 0x8000000000000000
  let m : UInt64 := 0x7fffffffffffffff
  let e : UInt64 := 0x7ff0000000000000
  let f : UInt64 := 0x000fffffffffffff
  let nanI := decide ((improvement &&& e) = e ∧ (improvement &&& f) ≠ 0)
  let nanM := decide ((margin &&& e) = e ∧ (margin &&& f) ≠ 0)
  let zeroI := decide ((improvement &&& m) = 0)
  let zeroM := decide ((margin &&& m) = 0)
  let ordI := if (improvement &&& s) = 0 then improvement ||| s else ~~~improvement
  let ordM := if (margin &&& s) = 0 then margin ||| s else ~~~margin
  if nanI || nanM then false
  else if zeroI && zeroM then false
  else decide (ordM < ordI)

/-- C-ABI exported Lean 4 controller decision gate (`@[export baux_action]`). -/
@[export baux_action]
def exportAction
    (adequate inadequate sharp accepted positive boundary failure : Bool) : UInt8 :=
  match BauxResearch.Optimization.decideAction
      adequate inadequate sharp accepted positive boundary failure with
  | .hold => 0
  | .expand => 1
  | .shrink => 2

/-- C-ABI exported Lean 4 same-world provenance gate (`@[export baux_world]`). -/
@[export baux_world]
def exportWorld (left right : UInt64) : Bool :=
  decide (left = right)

/-- C-ABI exported Lean 4 exact-anchored chain gate (`@[export baux_anchored]`). -/
@[export baux_anchored]
def exportAnchored (isFixed : Bool) (fromStep toStep : UInt64) : Bool :=
  if isFixed then true else decide (fromStep = toStep)

/-- C-ABI exported Lean 4 routed MoE parameter count (`@[export baux_routed]`). -/
@[export baux_routed]
def exportRouted (_ : UInt64) : UInt64 :=
  1036800000

/-- C-ABI exported Lean 4 total native mHC-4 parameter count (`@[export baux_total]`). -/
@[export baux_total]
def exportTotal (_ : UInt64) : UInt64 :=
  1177723160

/-- C-ABI exported Lean 4 full-window receipt verifier (`@[export baux_window]`). -/
@[export baux_window]
def exportWindow (planned prefilled scored : UInt64) : Bool :=
  decide (planned > 0) && decide (prefilled = 0) && decide (scored = planned)

/-- C-ABI exported Lean 4 L-order cell key packer (`@[export baux_cell]`). -/
@[export baux_cell]
def exportCell (level grid : UInt32) : UInt64 :=
  (level.toUInt64 <<< 32) ||| grid.toUInt64

/-- C-ABI exported Lean 4 conditioned antipodal slot selector (`@[export baux_slot]`). -/
@[export baux_slot]
def exportSlot (slot : UInt64) (accepted : Bool) : UInt64 :=
  if accepted then slot &&& 3 else (slot &&& 3) ^^^ 1

/-- C-ABI exported Lean 4 conditioned direction sign selector (`@[export baux_sign]`). -/
@[export baux_sign]
def exportSign (reversed accepted : Bool) : Bool :=
  if accepted then reversed else !reversed

theorem exportProof (improvement margin : UInt64) :
    exportAccept improvement margin = acceptBits improvement.toBitVec margin.toBitVec := by
  unfold exportAccept acceptBits nanBits zeroBits orderBits signMask expMask fracMask
  bv_decide

theorem slotExport (slot : UInt64) :
    exportSlot (exportSlot slot false) false = exportSlot slot true := by
  unfold exportSlot
  bv_decide

theorem dirExport (reversed : Bool) :
    exportSign (exportSign reversed false) false = exportSign reversed true := by
  cases reversed <;> rfl

/-!
# ENN 0.4.1 optimizer API

This module gives `ask` and `tell` one shared state and one transition system.
An ask changes only the lifecycle projection.  A tell runs the complete
state-retaining transition from `Tell`.  `run` makes finite public API
histories executable and stops at the first source-visible error.
-/


abbrev State := Tell.State

structure Ops where
  ask : Lifecycle.Ops
  tell : Tell.Ops

abbrev Config := Tell.Config

inductive Command where
  | ask (arms : Nat)
  | tell (x y : Fitter.Matrix) (variance : Option Fitter.Matrix)
  deriving Repr, DecidableEq

inductive Reply where
  | candidates (values : Sampling.Matrix)
  | told
  deriving Repr, DecidableEq

inductive Error where
  | ask (cause : Lifecycle.AskError)
  | tell (cause : Tell.TellError)
  deriving Repr, DecidableEq

structure Step where
  result : Except Error Reply
  state : State
  deriving Repr, DecidableEq

/-- One public optimizer call.  Earlier mutations are retained in `state` when
the result is an error, matching the component transitions. -/
def step (ops : Ops) (config : Config) (state : State) : Command → Step
  | .ask arms =>
      let asked := Lifecycle.ask ops.ask state.life arms
      let next := { state with life := asked.state }
      match asked.result with
      | .error cause => ⟨.error (.ask cause), next⟩
      | .ok values => ⟨.ok (.candidates values), next⟩
  | .tell x y variance =>
      let told := Tell.tell ops.tell config state x y variance
      match told.error with
      | some cause => ⟨.error (.tell cause), told.state⟩
      | none => ⟨.ok .told, told.state⟩

structure Trace where
  replies : List Reply
  result : Option Error
  state : State
  deriving Repr, DecidableEq

/-- Execute a finite API history, committing every transition and stopping at
the first error exactly where the public caller regains control. -/
def run (ops : Ops) (config : Config) : List Command → State → Trace
  | [], state => ⟨[], none, state⟩
  | command :: commands, state =>
      let current := step ops config state command
      match current.result with
      | .error cause => ⟨[], some cause, current.state⟩
      | .ok reply =>
          let rest := run ops config commands current.state
          ⟨reply :: rest.replies, rest.result, rest.state⟩

theorem runNil (ops : Ops) (config : Config) (state : State) :
    run ops config [] state = ⟨[], none, state⟩ := by
  rfl

theorem askLife (ops : Ops) (config : Config) (state : State) (arms : Nat) :
    (step ops config state (.ask arms)).state.life =
      (Lifecycle.ask ops.ask state.life arms).state := by
  simp only [step]
  generalize Lifecycle.ask ops.ask state.life arms = asked
  cases asked.result <;> rfl

theorem tellState (ops : Ops) (config : Config) (state : State)
    (x y : Fitter.Matrix) (variance : Option Fitter.Matrix) :
    (step ops config state (.tell x y variance)).state =
      (Tell.tell ops.tell config state x y variance).state := by
  simp only [step]
  generalize Tell.tell ops.tell config state x y variance = told
  cases told.error <;> rfl

/-- A one-call simulation obligation between two realizations of the optimizer
API.  It includes source-visible results and the successor state, including
the partially mutated state returned by an error. -/
def StepSim (upOps downOps : Ops) (upConfig downConfig : Config)
    (rel : State → State → Prop) : Prop :=
  ∀ {upState downState}, rel upState downState → ∀ command,
    let upStep := step upOps upConfig upState command
    let downStep := step downOps downConfig downState command
    upStep.result = downStep.result ∧ rel upStep.state downStep.state

/-- Observable equality and private-state correspondence for complete public
API histories. -/
structure TraceRel (rel : State → State → Prop) (up down : Trace) : Prop where
  replies : up.replies = down.replies
  result : up.result = down.result
  state : rel up.state down.state

/-- Per-call correspondence composes over every finite mixed `ask`/`tell`
history, including first-error termination and retained partial mutation. -/
theorem runRel (upOps downOps : Ops) (upConfig downConfig : Config)
    (rel : State → State → Prop) (hsim : StepSim upOps downOps upConfig downConfig rel)
    (commands : List Command) {upState downState : State}
    (hstate : rel upState downState) :
    TraceRel rel (run upOps upConfig commands upState)
      (run downOps downConfig commands downState) := by
  induction commands generalizing upState downState with
  | nil => exact ⟨rfl, rfl, hstate⟩
  | cons command commands ih =>
      generalize hup : step upOps upConfig upState command = upStep
      generalize hdown : step downOps downConfig downState command = downStep
      have hone := hsim hstate command
      rw [hup, hdown] at hone
      rcases hone with ⟨hresult, hnext⟩
      simp only [run, hup, hdown, hresult]
      cases hreply : downStep.result with
      | error cause => exact ⟨rfl, rfl, hnext⟩
      | ok reply =>
          have hrest := ih hnext
          exact ⟨congrArg (List.cons reply) hrest.replies,
            hrest.result, hrest.state⟩

#print axioms runNil
#print axioms askLife
#print axioms tellState
#print axioms runRel

end BauxResearch.Semantics.Optimizer
