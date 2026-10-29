import BauxResearch.Optimization.Regret

/-!
# Joint policy and certificate synthesis

A policy queries executable states and updates retained memory from declared
feedback. Its interface contains neither the objective nor its optimum. An
oracle supplies feedback, conditional finite probabilities, and abstract work
charges. Its law may depend on time, memory, and query; independence of successive
observations is not assumed. Policy randomization can be included in the finite
feedback branches, but a concrete sampling implementation is a separate task.

Local potential inequalities certify expected terminal simple regret. A natural
credit certifies total work along every noise path. Work units are declared by
the oracle; they are not milliseconds without an implementation/machine account.
The horizon, bound, environment class, and budget belong to a fixed `Task`.
`Solution` couples the policy with certificates for every admissible environment.
No automated search procedure or production BAUX regret result is claimed here.
-/

namespace BauxResearch.Optimization.Synthesis

open scoped BigOperators
open Regret

variable {State Feedback Memory Noise : Type*} [Fintype Noise]

structure Policy (State Feedback Memory : Type*) where
  initial : Memory
  query : ℕ → Memory → State
  update : ℕ → Memory → State → Feedback → Memory
  recommend : Memory → State

structure Oracle (State Feedback Memory Noise : Type*) [Fintype Noise] where
  value : ℕ → Memory → State → Noise → Feedback
  weight : ℕ → Memory → State → Noise → ℝ
  nonnegative : ∀ time memory state noise, 0 ≤ weight time memory state noise
  mass : ∀ time memory state, ∑ noise, weight time memory state noise = 1
  charge : ℕ → Memory → State → Noise → ℕ

def advance (policy : Policy State Feedback Memory) (oracle : Oracle State Feedback Memory Noise)
    (time : ℕ) (memory : Memory) (noise : Noise) : Memory :=
  let state := policy.query time memory
  policy.update time memory state (oracle.value time memory state noise)

/-- Execute a policy on a supplied oracle feedback path. -/
def run (policy : Policy State Feedback Memory) (oracle : Oracle State Feedback Memory Noise) :
    ℕ → ℕ → Memory → (ℕ → Noise) → Memory
  | _, 0, memory, _ => memory
  | time, steps + 1, memory, path =>
      run policy oracle (time + 1) steps
        (advance policy oracle time memory (path time)) path

/-- Exact finite conditional expectation, evaluated backwards from a terminal loss. -/
def expected (policy : Policy State Feedback Memory) (oracle : Oracle State Feedback Memory Noise)
    (objective : State → ℝ) (best : State) : ℕ → ℕ → Memory → ℝ
  | _, 0, memory => simpleRegret objective best (policy.recommend memory)
  | time, steps + 1, memory =>
      ∑ noise, oracle.weight time memory (policy.query time memory) noise *
        expected policy oracle objective best (time + 1) steps
          (advance policy oracle time memory noise)

/-- Total abstract work along one complete feedback path. -/
def spent (policy : Policy State Feedback Memory) (oracle : Oracle State Feedback Memory Noise) :
    ℕ → ℕ → Memory → (ℕ → Noise) → ℕ
  | _, 0, _, _ => 0
  | time, steps + 1, memory, path =>
      oracle.charge time memory (policy.query time memory) (path time) +
        spent policy oracle (time + 1) steps
          (advance policy oracle time memory (path time)) path

structure Certificate (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (horizon : ℕ) (bound : ℝ) (budget : ℕ) where
  potential : ℕ → Memory → ℝ
  initialBound : potential 0 policy.initial ≤ bound
  terminalBound : ∀ memory,
    simpleRegret objective best (policy.recommend memory) ≤ potential horizon memory
  stepBound : ∀ time, time < horizon → ∀ memory,
    (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      potential (time + 1) (advance policy oracle time memory noise)) ≤
        potential time memory
  credit : ℕ → Memory → ℕ
  workInitial : credit 0 policy.initial ≤ budget
  workStep : ∀ time, time < horizon → ∀ memory noise,
    oracle.charge time memory (policy.query time memory) noise +
      credit (time + 1) (advance policy oracle time memory noise) ≤ credit time memory

theorem Certificate.controls
    {policy : Policy State Feedback Memory} {oracle : Oracle State Feedback Memory Noise}
    {objective : State → ℝ} {best : State} {horizon budget : ℕ} {bound : ℝ}
    (certificate : Certificate policy oracle objective best horizon bound budget)
    (steps time : ℕ) (memory : Memory) (ends : time + steps = horizon) :
    expected policy oracle objective best time steps memory ≤
      certificate.potential time memory := by
  induction steps generalizing time memory with
  | zero =>
      have timeEq : time = horizon := by omega
      simpa [expected, timeEq] using certificate.terminalBound memory
  | succ steps induction =>
      have before : time < horizon := by omega
      have nextEnds : time + 1 + steps = horizon := by omega
      calc
        expected policy oracle objective best time (steps + 1) memory ≤
            ∑ noise, oracle.weight time memory (policy.query time memory) noise *
              certificate.potential (time + 1) (advance policy oracle time memory noise) := by
          unfold expected
          apply Finset.sum_le_sum
          intro noise _
          exact mul_le_mul_of_nonneg_left
            (induction (time + 1) (advance policy oracle time memory noise) nextEnds)
            (oracle.nonnegative time memory (policy.query time memory) noise)
        _ ≤ certificate.potential time memory := certificate.stepBound time before memory

theorem Certificate.workControls
    {policy : Policy State Feedback Memory} {oracle : Oracle State Feedback Memory Noise}
    {objective : State → ℝ} {best : State} {horizon budget : ℕ} {bound : ℝ}
    (certificate : Certificate policy oracle objective best horizon bound budget)
    (steps time : ℕ) (memory : Memory) (path : ℕ → Noise)
    (ends : time + steps = horizon) :
    spent policy oracle time steps memory path ≤ certificate.credit time memory := by
  induction steps generalizing time memory with
  | zero => simp [spent]
  | succ steps induction =>
      have before : time < horizon := by omega
      have nextEnds : time + 1 + steps = horizon := by omega
      calc
        spent policy oracle time (steps + 1) memory path ≤
            oracle.charge time memory (policy.query time memory) (path time) +
              certificate.credit (time + 1)
                (advance policy oracle time memory (path time)) := by
          exact Nat.add_le_add_left
            (induction (time + 1) (advance policy oracle time memory (path time)) nextEnds) _
        _ ≤ certificate.credit time memory :=
          certificate.workStep time before memory (path time)

theorem Certificate.regretBound
    {policy : Policy State Feedback Memory} {oracle : Oracle State Feedback Memory Noise}
    {objective : State → ℝ} {best : State} {horizon budget : ℕ} {bound : ℝ}
    (certificate : Certificate policy oracle objective best horizon bound budget) :
    expected policy oracle objective best 0 horizon policy.initial ≤ bound :=
  (certificate.controls horizon 0 policy.initial (by omega)).trans certificate.initialBound

theorem Certificate.workBound
    {policy : Policy State Feedback Memory} {oracle : Oracle State Feedback Memory Noise}
    {objective : State → ℝ} {best : State} {horizon budget : ℕ} {bound : ℝ}
    (certificate : Certificate policy oracle objective best horizon bound budget)
    (path : ℕ → Noise) : spent policy oracle 0 horizon policy.initial path ≤ budget :=
  (certificate.workControls horizon 0 policy.initial path (by omega)).trans certificate.workInitial

/-- A fixed, nonempty objective/oracle class and targets supplied before search. -/
structure Task (State Feedback Memory Noise : Type*) [Fintype Noise] where
  Environment : Type
  examples : Nonempty Environment
  oracle : Environment → Oracle State Feedback Memory Noise
  objective : Environment → State → ℝ
  best : Environment → State
  optimal : ∀ environment, Optimal (objective environment) (best environment)
  horizon : ℕ
  bound : ℝ
  budget : ℕ

/-- The same policy must meet the fixed task for every admissible environment. -/
structure Solution (task : Task State Feedback Memory Noise) where
  policy : Policy State Feedback Memory
  certificate : ∀ environment,
    Certificate policy (task.oracle environment) (task.objective environment)
      (task.best environment) task.horizon task.bound task.budget

theorem Solution.regretBound {task : Task State Feedback Memory Noise}
    (solution : Solution task) (environment : task.Environment) :
    expected solution.policy (task.oracle environment) (task.objective environment)
      (task.best environment) 0 task.horizon solution.policy.initial ≤ task.bound :=
  (solution.certificate environment).regretBound

theorem Solution.workBound {task : Task State Feedback Memory Noise}
    (solution : Solution task) (environment : task.Environment) (path : ℕ → Noise) :
    spent solution.policy (task.oracle environment) 0 task.horizon
      solution.policy.initial path ≤ task.budget :=
  (solution.certificate environment).workBound path

end BauxResearch.Optimization.Synthesis
