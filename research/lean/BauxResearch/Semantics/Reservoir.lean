import Mathlib.Data.BitVec
import BauxResearch.Semantics.Metric
import BauxResearch.Semantics.Pcg

/-!
# ENN 0.4.1 Algorithm R reservoir sampling semantics (`reservoir.rs`)

This module proves the function-level semantics and invariants for all 8
reachable functions in `crates/ennbo/src/reservoir.rs`:
- `rust:ennbo 0.4.1 reservoir/Reservoir#new()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#num_dim()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#num_outputs()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#len()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#num_seen()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#x()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#y()`
- `rust:ennbo 0.4.1 reservoir/Reservoir#push_row()`
-/

namespace BauxResearch.Semantics.Reservoir

open BauxResearch.Semantics.Sampling
open BauxResearch.Semantics.Metric

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#new()` constructor for arbitrary capacity. -/
def newRes (capacity dim outputs : Nat) (seed : Word) : Reservoir :=
  ⟨capacity, dim, outputs, seed, Pcg.fromSeed seed, 0, [], []⟩

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#num_dim()` accessor. -/
def numDim (r : Reservoir) : Nat :=
  r.dim

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#num_outputs()` accessor. -/
def numOutputs (r : Reservoir) : Nat :=
  r.outputs

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#len()` accessor. -/
def resLen (r : Reservoir) : Nat :=
  r.xs.length

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#num_seen()` accessor. -/
def numSeen (r : Reservoir) : Nat :=
  r.numSeen

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#x()` accessor. -/
def rowsX (r : Reservoir) : List (List Word) :=
  r.xs

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#y()` accessor. -/
def rowsY (r : Reservoir) : List (List Word) :=
  r.ys

theorem newSeen (cap dim outputs : Nat) (seed : Word) :
    numSeen (newRes cap dim outputs seed) = 0 := by
  rfl

theorem newLen (cap dim outputs : Nat) (seed : Word) :
    resLen (newRes cap dim outputs seed) = 0 := by
  rfl

theorem dimEq (cap dim outputs : Nat) (seed : Word) :
    numDim (newRes cap dim outputs seed) = dim := by
  rfl

theorem outEq (cap dim outputs : Nat) (seed : Word) :
    numOutputs (newRes cap dim outputs seed) = outputs := by
  rfl

theorem xEmpty (cap dim outputs : Nat) (seed : Word) :
    rowsX (newRes cap dim outputs seed) = [] := by
  rfl

theorem yEmpty (cap dim outputs : Nat) (seed : Word) :
    rowsY (newRes cap dim outputs seed) = [] := by
  rfl

/-- Every `rust:ennbo 0.4.1 reservoir/Reservoir#push_row()` transition increments `num_seen` by 1. -/
theorem seenStep {r next : Reservoir} {x y : List Word}
    (h : PushRel r x y next) :
    numSeen next = numSeen r + 1 := by
  cases h <;> rfl

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#push_row()` preserves the capacity bound `len ≤ capacity`. -/
theorem capStep {r next : Reservoir} {x y : List Word}
    (hcap : resLen r ≤ r.capacity) (h : PushRel r x y next) :
    resLen next ≤ next.capacity := by
  cases h with
  | fill hlt =>
    simp only [resLen, List.length_append, List.length_singleton]
    omega
  | replace slot nextRng hge _ hslot =>
    simp only [resLen, List.length_set]
    exact hcap
  | discard slot nextRng hge _ hout =>
    exact hcap

/-- `rust:ennbo 0.4.1 reservoir/Reservoir#push_row()` is deterministic. -/
theorem pushDet {r r1 r2 : Reservoir} {x y : List Word}
    (h1 : PushRel r x y r1) (h2 : PushRel r x y r2) :
    r1 = r2 := by
  cases h1 with
  | fill hlt1 =>
    cases h2 with
    | fill _ => rfl
    | replace _ _ hge2 _ _ => omega
    | discard _ _ hge2 _ _ => omega
  | replace s1 rng1 hge1 hd1 hs1 =>
    cases h2 with
    | fill hlt2 => omega
    | replace s2 rng2 _ hd2 _ =>
      rcases Pcg.detHigh hd1 hd2 with ⟨rfl, rfl⟩
      rfl
    | discard s2 rng2 _ hd2 hout2 =>
      rcases Pcg.detHigh hd1 hd2 with ⟨rfl, rfl⟩
      omega
  | discard s1 rng1 hge1 hd1 hout1 =>
    cases h2 with
    | fill hlt2 => omega
    | replace s2 rng2 _ hd2 hs2 =>
      rcases Pcg.detHigh hd1 hd2 with ⟨rfl, rfl⟩
      omega
    | discard s2 rng2 _ hd2 _ =>
      rcases Pcg.detHigh hd1 hd2 with ⟨rfl, rfl⟩
      rfl

/-- Fuel-bounded evaluation `pushFuel` is sound with respect to `PushRel`. -/
theorem soundPush (fuel : Nat) (r next : Reservoir) (x y : List Word)
    (h : pushFuel fuel r x y = some next) :
    PushRel r x y next :=
  pushSound fuel r next x y h

#print axioms newSeen
#print axioms newLen
#print axioms dimEq
#print axioms outEq
#print axioms xEmpty
#print axioms yEmpty
#print axioms seenStep
#print axioms capStep
#print axioms pushDet
#print axioms soundPush

end BauxResearch.Semantics.Reservoir
