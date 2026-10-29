import BauxResearch.Optimization.Reachability

/-!
# Fixed proposal support

The long pretraining runner derives each root seed from a configured seed and
the round index. With that configuration fixed, a completed ask selects one
deterministic candidate. Its conditional proposal law is a point mass. A
probabilistic gain claim needs an explicit random experiment over seeds or
some other source of randomness; a Gaussian generator label alone is not one.

The second theorem records the value-only obstruction for any finite candidate
pool. It says why a lower bound on useful proposal mass requires objective
structure, independent of how accurately the ENN ranks the pool.
-/

namespace BauxResearch.Optimization.SeedSupport

noncomputable section

/-- A fixed completed ask has either full or zero useful-move mass. This is
the degenerate finite law for the one candidate actually evaluated. -/
theorem fixedMass {State : Type*} (objective : Objective State)
    (incumbent candidate : State) (gain : ℝ) :
    ObjectiveSearch.goodMass (ObjectiveSearch.uniformLaw 1) objective incumbent
      (fun _ : Fin 1 => candidate) gain =
        if gain ≤ progress objective incumbent candidate then 1 else 0 := by
  classical
  by_cases useful : gain ≤ progress objective incumbent candidate
  · simp [ObjectiveSearch.goodMass, ObjectiveSearch.uniformLaw, useful]
  · simp [ObjectiveSearch.goodMass, ObjectiveSearch.uniformLaw, useful]

/-- If the state space has more points than the incumbent and finite candidate
pool combined, some loss has zero gain on every proposed state and unit gain at
an unproposed state. Hence finite support alone gives no uniform improvement
guarantee over arbitrary objectives. -/
theorem blindPool {Seed State : Type*} [Fintype Seed] [Fintype State]
    (candidate : Seed → State) (incumbent : State)
    (room : Fintype.card Seed + 1 < Fintype.card State) :
    ∃ (objective : Objective State) (target : State),
      objective incumbent = 0 ∧
      (∀ seed, progress objective incumbent (candidate seed) = 0) ∧
      progress objective incumbent target = 1 := by
  classical
  let visited : Finset State :=
    insert incumbent (Finset.univ.image candidate)
  have visitedCard : visited.card < (Finset.univ : Finset State).card := by
    calc
      visited.card ≤ (Finset.univ.image candidate).card + 1 :=
        Finset.card_insert_le _ _
      _ ≤ Fintype.card Seed + 1 :=
        Nat.add_le_add_right Finset.card_image_le _
      _ < Fintype.card State := room
      _ = (Finset.univ : Finset State).card := by simp
  obtain ⟨target, _, outside⟩ :=
    Finset.exists_mem_notMem_of_card_lt_card visitedCard
  let objective : Objective State := fun state => if state = target then -1 else 0
  have incumbentOutside : incumbent ≠ target := by
    intro equal
    apply outside
    exact equal ▸ Finset.mem_insert_self incumbent _
  have candidateOutside (seed : Seed) : candidate seed ≠ target := by
    intro equal
    apply outside
    exact Finset.mem_insert_of_mem (Finset.mem_image.mpr
      ⟨seed, Finset.mem_univ _, equal⟩)
  refine ⟨objective, target, ?_, ?_, ?_⟩
  · simp [objective, incumbentOutside]
  · intro seed
    simp [progress, objective, incumbentOutside, candidateOutside seed]
  · simp [progress, objective, incumbentOutside]

#print axioms fixedMass
#print axioms blindPool

end
end BauxResearch.Optimization.SeedSupport
