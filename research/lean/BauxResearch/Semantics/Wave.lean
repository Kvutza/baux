import BauxResearch.Semantics.Buffer

/-!
# Production dispatch hazard decisions

The conflict predicate and both bounded scans are the extracted functions called
by the host Wave tracker. Resource identifiers mean buffer-object identity;
physical aliases between distinct objects remain outside this model. Cross-dispatch
conflicts are separate from a kernel's own internal argument alias contract.
-/

namespace BauxResearch.Semantics.Wave

open Aeneas Aeneas.Std
open BauxResearch.Generated.Optimizer

/-- A shared byte on one resource object, with at least one writer. -/
def Clash (first second : hazard.Access) : Prop :=
  first.buffer = second.buffer ∧ (first.write = true ∨ second.write = true) ∧
    first.start.val < first.end.val ∧ second.start.val < second.end.val ∧
    first.start.val < second.end.val ∧ second.start.val < first.end.val

instance (first second : hazard.Access) : Decidable (Clash first second) := by
  unfold Clash
  infer_instance

theorem clashWitness (first second : hazard.Access) :
    Clash first second ↔ first.buffer = second.buffer ∧
      (first.write = true ∨ second.write = true) ∧ ∃ byte : ℕ,
        first.start.val ≤ byte ∧ byte < first.end.val ∧
        second.start.val ≤ byte ∧ byte < second.end.val := by
  unfold Clash
  constructor
  · rintro ⟨same, writer, bounds⟩
    exact ⟨same, writer, max first.start.val second.start.val, by omega, by omega,
      by omega, by omega⟩
  · rintro ⟨same, writer, byte, bounds⟩
    exact ⟨same, writer, by omega, by omega, by omega, by omega⟩

theorem conflictMatch (first second : hazard.Access) :
    hazard.conflict first second = Result.ok (decide (Clash first second)) := by
  simp only [hazard.conflict, memory.overlaps, UScalar.lt_equiv]
  split <;> simp_all [Clash]
  split <;> simp_all
  all_goals split <;> simp_all
  all_goals split <;> simp_all
  all_goals split <;> simp_all
  all_goals split <;> simp_all

theorem indexMatch (values : Slice hazard.Access) (index : Usize)
    (inside : index.val < values.val.length) :
    Slice.index_usize values index = Result.ok values.val[index.val] := by
  simp [Slice.index_usize, List.getElem?_eq_getElem inside]

theorem increment (index : Usize) (bound : index.val < Usize.max) :
    ∃ result : Usize, index + 1#usize = Result.ok result ∧ result.val = index.val + 1 := by
  have one : (1#usize).val = 1 := by scalar_tac
  have safe : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
  let result : Usize := ⟨BitVec.ofNat _ (index.val + 1)⟩
  have value : result.val = index.val + 1 := Nat.mod_eq_of_lt safe
  refine ⟨result, ?_, value⟩
  change UScalar.add index 1#usize = _
  simp only [UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, UScalar.check_bounds,
    one, safe, decide_true, ↓reduceDIte, Result.ofOption]
  apply congrArg Result.ok
  apply UScalar.val_eq_imp
  simpa using value.symm

/-- Inner extracted scan terminates and detects exactly the remaining conflicts. -/
theorem againstLoop (active : Slice hazard.Access) (pending : hazard.Access) (prior : Usize) :
    hazard.against_loop active pending prior =
      Result.ok ((active.val.drop prior.val).any (fun item ↦ decide (Clash item pending))) := by
  by_cases inside : prior.val < active.val.length
  · have indexed := indexMatch active prior inside
    rw [List.drop_eq_getElem_cons inside, List.any_cons]
    by_cases hit : Clash active.val[prior.val] pending
    · have body : hazard.against_loop.body active pending prior =
          Result.ok (ControlFlow.done true) := by
        simp [hazard.against_loop.body, UScalar.lt_equiv, inside, indexed, conflictMatch, hit]
      unfold hazard.against_loop
      rw [loop]
      change Aeneas.Std.bind (hazard.against_loop.body active pending prior) _ = _
      rw [body, bind_ok]
      simp [hit]
    · have maximum := active.length_ineq
      obtain ⟨next, increment, nextValue⟩ := increment prior (by omega)
      have tail := againstLoop active pending next
      have body : hazard.against_loop.body active pending prior =
          Result.ok (ControlFlow.cont next) := by
        simp [hazard.against_loop.body, UScalar.lt_equiv, inside, indexed,
          conflictMatch, hit, increment]
      unfold hazard.against_loop
      rw [loop]
      change Aeneas.Std.bind (hazard.against_loop.body active pending prior) _ = _
      rw [body, bind_ok]
      simpa [hazard.against_loop, hit, nextValue] using tail
  · have outside : active.val.length ≤ prior.val := by omega
    unfold hazard.against_loop
    rw [loop]
    simp [hazard.against_loop.body, UScalar.lt_equiv,
      inside, List.drop_eq_nil_of_le outside]
termination_by active.val.length - prior.val
decreasing_by omega

theorem againstMatch (active : Slice hazard.Access) (pending : hazard.Access) :
    hazard.against active pending =
      Result.ok (active.val.any (fun item ↦ decide (Clash item pending))) := by
  simpa [hazard.against] using againstLoop active pending 0#usize

/-- Outer extracted scan terminates and returns the complete cross-dispatch conflict test. -/
theorem flushLoop (active pending : Slice hazard.Access) (next : Usize) :
    hazard.flush_loop active pending next = Result.ok
      ((pending.val.drop next.val).any (fun item ↦
        active.val.any (fun prior ↦ decide (Clash prior item)))) := by
  by_cases inside : next.val < pending.val.length
  · have indexed := indexMatch pending next inside
    rw [List.drop_eq_getElem_cons inside, List.any_cons]
    by_cases hit : active.val.any (fun prior ↦ decide (Clash prior pending.val[next.val])) = true
    · have found : ∃ prior ∈ active.val, Clash prior pending.val[next.val] := by simpa using hit
      have body : hazard.flush_loop.body active pending next =
          Result.ok (ControlFlow.done true) := by
        simp [hazard.flush_loop.body, UScalar.lt_equiv, inside, indexed, againstMatch, found]
      unfold hazard.flush_loop
      rw [loop]
      change Aeneas.Std.bind (hazard.flush_loop.body active pending next) _ = _
      rw [body, bind_ok]
      simp [hit]
    · have maximum := pending.length_ineq
      obtain ⟨after, increment, afterValue⟩ := increment next (by omega)
      have tail := flushLoop active pending after
      have absent : ¬∃ prior ∈ active.val, Clash prior pending.val[next.val] := by simpa using hit
      have body : hazard.flush_loop.body active pending next =
          Result.ok (ControlFlow.cont after) := by
        simp [hazard.flush_loop.body, UScalar.lt_equiv, inside, indexed,
          againstMatch, absent, increment]
      unfold hazard.flush_loop
      rw [loop]
      change Aeneas.Std.bind (hazard.flush_loop.body active pending next) _ = _
      rw [body, bind_ok]
      simpa [hazard.flush_loop, hit, afterValue] using tail
  · have outside : pending.val.length ≤ next.val := by omega
    unfold hazard.flush_loop
    rw [loop]
    simp [hazard.flush_loop.body, UScalar.lt_equiv,
      inside, List.drop_eq_nil_of_le outside]
termination_by pending.val.length - next.val
decreasing_by omega

theorem flushMatch (active pending : Slice hazard.Access) :
    hazard.flush active pending = Result.ok
      (pending.val.any (fun item ↦ active.val.any (fun prior ↦ decide (Clash prior item)))) := by
  simpa [hazard.flush] using flushLoop active pending 0#usize

/-- Exact capacity decision, including invalid previous counts, without machine overflow. -/
theorem capacityMatch (reads writes incoming outgoing : U32) :
    hazard.capacity reads writes incoming outgoing = Result.ok
      (decide (32 < reads.val + incoming.val ∨ 32 < writes.val + outgoing.val)) := by
  have cap : (32#u32).val = 32 := rfl
  by_cases readOutside : 32 < reads.val
  · have full : 32 < reads.val + incoming.val ∨ 32 < writes.val + outgoing.val := by omega
    simp [hazard.capacity, UScalar.lt_equiv, cap, readOutside, full]
  · by_cases writeOutside : 32 < writes.val
    · have full : 32 < reads.val + incoming.val ∨ 32 < writes.val + outgoing.val := by omega
      simp [hazard.capacity, UScalar.lt_equiv, cap, readOutside, writeOutside, full]
    · let left : U32 := ⟨BitVec.ofNat 32 (32 - reads.val)⟩
      let right : U32 := ⟨BitVec.ofNat 32 (32 - writes.val)⟩
      have leftVal : left.val = 32 - reads.val := Nat.mod_eq_of_lt (by omega)
      have rightVal : right.val = 32 - writes.val := Nat.mod_eq_of_lt (by omega)
      have leftSub : (32#u32 - reads) = Result.ok left := by
        change UScalar.sub 32#u32 reads = _
        simp [UScalar.sub, cap, readOutside, left]
      have rightSub : (32#u32 - writes) = Result.ok right := by
        change UScalar.sub 32#u32 writes = _
        simp [UScalar.sub, cap, writeOutside, right]
      by_cases leftFull : 32 - reads.val < incoming.val
      · have full : 32 < reads.val + incoming.val ∨ 32 < writes.val + outgoing.val := by omega
        simp [hazard.capacity, UScalar.lt_equiv, cap, readOutside, writeOutside,
          leftSub, leftVal, leftFull, full]
      · have full : (32 < reads.val + incoming.val ∨ 32 < writes.val + outgoing.val) ↔
            32 - writes.val < outgoing.val := by omega
        simp [hazard.capacity, UScalar.lt_equiv, cap, readOutside, writeOutside,
          leftSub, rightSub, leftVal, rightVal, leftFull, full]

/-- After either hazard or capacity flushing, appending one bounded dispatch fits both arrays. -/
theorem capacitySafe (reads writes incoming outgoing : U32) (conflict full : Bool)
    (readBound : incoming.val ≤ 32) (writeBound : outgoing.val ≤ 32)
    (decision : hazard.capacity reads writes incoming outgoing = Result.ok full) :
    (if full || conflict then incoming.val else reads.val + incoming.val) ≤ 32 ∧
      (if full || conflict then outgoing.val else writes.val + outgoing.val) ≤ 32 := by
  cases full <;> cases conflict <;> simp_all [capacityMatch]

/-- The extracted scan demands a barrier exactly for a cross-dispatch hazard. -/
theorem flushWitness (active pending : Slice hazard.Access) :
    hazard.flush active pending = Result.ok true ↔
      ∃ next ∈ pending.val, ∃ prior ∈ active.val, Clash prior next := by
  rw [flushMatch]
  simp

/-- No-barrier decisions exclude all RAW, WAR and WAW byte conflicts. -/
theorem clear (active pending : Slice hazard.Access)
    (decision : hazard.flush active pending = Result.ok false) :
    ∀ prior ∈ active.val, ∀ next ∈ pending.val, ¬Clash prior next := by
  intro prior previous next incoming conflict
  have required := (flushWitness active pending).mpr ⟨next, incoming, prior, previous, conflict⟩
  rw [decision] at required
  simp at required

/-- A dispatch contains declared reads/writes; independence concerns different dispatches. -/
def Independent (first second : List hazard.Access) : Prop :=
  ∀ a ∈ first, ∀ b ∈ second, ¬Clash a b

/-- No declared cross-dispatch memory hazard occurs within an epoch. -/
def Safe (epoch : List (List hazard.Access)) : Prop := epoch.Pairwise Independent

/-- Host transition induced by the checked decision, including conservative capacity flushing. -/
def advance (epoch : List (List hazard.Access)) (pending : List hazard.Access)
    (flush capacity : Bool) : List (List hazard.Access) :=
  if capacity || flush then [pending] else epoch ++ [pending]

/-- Applying the actual extracted hazard decision preserves conflict-free epochs.
The representation premise connects tracked spans to dispatch history. Hardware
ordering between the resulting epochs requires an emitted Metal barrier. -/
theorem epochSafe (epoch : List (List hazard.Access)) (active pending : Slice hazard.Access)
    (flush capacity : Bool)
    (represented : ∀ access, access ∈ active.val ↔ access ∈ epoch.flatten)
    (decision : capacity = false → hazard.flush active pending = Result.ok flush)
    (safe : Safe epoch) :
    Safe (advance epoch pending.val flush capacity) := by
  by_cases barrier : (capacity || flush) = true
  · simp only [advance, barrier, ↓reduceIte, Safe]
    exact List.pairwise_singleton _ _
  · have noFlush : flush = false := by cases flush <;> simp_all
    have noCapacity : capacity = false := by cases capacity <;> simp_all
    have independent := clear active pending (by simpa [noFlush] using decision noCapacity)
    simp only [advance, barrier, Bool.false_eq_true, ↓reduceIte, Safe,
      List.pairwise_append, List.pairwise_singleton, true_and]
    refine ⟨safe, ?_⟩
    intro prior member next selected
    have nextEq : next = pending.val := by simpa using selected
    subst next
    intro a previous b incoming
    apply independent a _ b incoming
    apply (represented a).mpr
    exact List.mem_flatten.mpr ⟨prior, member, previous⟩

#print axioms capacityMatch
#print axioms capacitySafe
#print axioms epochSafe
#print axioms conflictMatch
#print axioms flushMatch

end BauxResearch.Semantics.Wave
