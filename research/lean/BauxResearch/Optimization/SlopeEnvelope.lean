import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Pointwise-slope metric envelopes, misspecification slack, and causal packing

This module formalizes McShane--Whitney metric envelopes with heterogeneous
pointwise slopes `slope : Index → ℝ`, bounded observation intervals
`[lower i, upper i]`, and explicit slope-misspecification slack
`slack : Index → ℝ`.

Main results:
1. `slackBounds` and `exactBounds`: pointwise envelope bracketing with explicit
   misspecification slack `slackTerm`, specializing to exact bracketing when
   `slack = 0`.
2. `orderIff`, `inconsistentFalse`, and `crossedFalse`: `below x ≤ above x`
   holds at all states `x` if and only if the finite pairwise condition
   `lower i - upper j ≤ min (slope i) (slope j) * dist (point i) (point j)`
   holds on the recorded archive; violation of this finite inequality certifies
   that no compatible objective exists.
3. `midpointSplit`, `estimateBound`, and `midpointBound`: for any estimator
   `estimate`, its worst-case error over the envelope bracket decomposes into
   the envelope half-width `halfWidth` plus the midpoint calibration offset
   `|estimate - midpoint|` plus misspecification slack.
4. `exclusionIff`, `domRefl`, `domTrans`, `domBelow`, and `deleteBelow`:
   exclusion-ball characterization of feasible sublevel sets and preorder
   cone-dominance deletion.
5. `stepDichotomy`, `causalSep`, `causalOverlap`, `bandDisjoint`, `bandCount`,
   and `multiscaleCount`: single-evaluation descent/excision dichotomy, causal
   separation `min (rad p) (rad q) ≤ dist (center p) (center q)`, a concrete 1D
   counterexample showing that naive variable-radius half-balls can overlap
   under one-way causal exclusion, and pairwise disjointness + packing bounds
   within each scale band.
6. `chordSlope`, `chordEval`, `chordMeet`, and `chordLower`: two-point chord
   slope, McShane--Whitney cone coincidence at the chord interpolant, and
   pointwise minimal slope optimality.
-/

namespace BauxResearch.Optimization.SlopeEnvelope

open scoped BigOperators

variable {State Index : Type*} [PseudoMetricSpace State]

/-- Interval observation constraints at recorded states. -/
def fits (records : Finset Index) (point : Index → State)
    (lower upper : Index → ℝ) (objective : State → ℝ) : Prop :=
  ∀ index ∈ records, lower index ≤ objective (point index) ∧
    objective (point index) ≤ upper index

/-- Pointwise cone regularity anchored at recorded states with misspecification
slack `slack index`. -/
def coneValid (records : Finset Index) (point : Index → State)
    (slope slack : Index → ℝ) (objective : State → ℝ) : Prop :=
  ∀ index ∈ records, ∀ query : State,
    |objective query - objective (point index)| ≤
      (slope index + slack index) * dist query (point index)

/-- Exact pointwise cone regularity anchored at recorded states (`slack = 0`). -/
def coneExact (records : Finset Index) (point : Index → State)
    (slope : Index → ℝ) (objective : State → ℝ) : Prop :=
  ∀ index ∈ records, ∀ query : State,
    |objective query - objective (point index)| ≤
      slope index * dist query (point index)

/-- Finite pairwise compatibility condition for heterogeneous slopes: the chord
between `point first` and `point second` is constrained by the smaller of the
two pointwise slopes. -/
def consistent (records : Finset Index) (point : Index → State)
    (lower upper slope : Index → ℝ) : Prop :=
  ∀ first ∈ records, ∀ second ∈ records,
    lower first - upper second ≤
      min (slope first) (slope second) * dist (point first) (point second)

/-- Lower McShane--Whitney envelope with heterogeneous slopes. -/
noncomputable def below (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower slope : Index → ℝ) (query : State) : ℝ :=
  records.sup' nonempty fun index ↦ lower index - slope index * dist query (point index)

/-- Upper McShane--Whitney envelope with heterogeneous slopes. -/
noncomputable def above (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (upper slope : Index → ℝ) (query : State) : ℝ :=
  records.inf' nonempty fun index ↦ upper index + slope index * dist query (point index)

/-- Worst-case misspecification slack at `query`. -/
noncomputable def slackTerm (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (slack : Index → ℝ) (query : State) : ℝ :=
  records.sup' nonempty fun index ↦ slack index * dist query (point index)

/-- Pointwise envelope bracketing with explicit slope-misspecification slack. -/
theorem slackBounds (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope slack : Index → ℝ)
    (objective : State → ℝ) (observed : fits records point lower upper objective)
    (regular : coneValid records point slope slack objective) (query : State) :
    below records nonempty point lower slope query -
        slackTerm records nonempty point slack query ≤ objective query ∧
      objective query ≤
        above records nonempty point upper slope query +
          slackTerm records nonempty point slack query := by
  constructor
  · rw [sub_le_iff_le_add]
    apply Finset.sup'_le
    intro index member
    have obs := (observed index member).1
    have cone := (abs_le.mp (regular index member query)).1
    have slk := Finset.le_sup' (fun j ↦ slack j * dist query (point j)) member
    change _ ≤ slackTerm records nonempty point slack query at slk
    linarith
  · rw [← sub_le_iff_le_add]
    apply Finset.le_inf'
    intro index member
    have obs := (observed index member).2
    have cone := (abs_le.mp (regular index member query)).2
    have slk := Finset.le_sup' (fun j ↦ slack j * dist query (point j)) member
    change _ ≤ slackTerm records nonempty point slack query at slk
    linarith

/-- Exact pointwise envelope bracketing when all slope bounds are valid (`slack = 0`). -/
theorem exactBounds (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ)
    (objective : State → ℝ) (observed : fits records point lower upper objective)
    (regular : coneExact records point slope objective) (query : State) :
    below records nonempty point lower slope query ≤ objective query ∧
      objective query ≤ above records nonempty point upper slope query := by
  constructor
  · apply Finset.sup'_le
    intro index member
    have obs := (observed index member).1
    have cone := (abs_le.mp (regular index member query)).1
    linarith
  · apply Finset.le_inf'
    intro index member
    have obs := (observed index member).2
    have cone := (abs_le.mp (regular index member query)).2
    linarith

/-- Pairwise consistency implies `below query ≤ above query` at every state. -/
theorem crossedOrder (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ)
    (nonneg : ∀ index ∈ records, 0 ≤ slope index)
    (compatible : consistent records point lower upper slope) (query : State) :
    below records nonempty point lower slope query ≤
      above records nonempty point upper slope query := by
  apply Finset.sup'_le
  intro first firstIn
  apply Finset.le_inf'
  intro second secondIn
  have pair := compatible first firstIn second secondIn
  have tri : dist (point first) (point second) ≤
      dist query (point first) + dist query (point second) := by
    simpa only [dist_comm (point first) query] using
      dist_triangle (point first) query (point second)
  have minNonneg : 0 ≤ min (slope first) (slope second) :=
    le_min (nonneg first firstIn) (nonneg second secondIn)
  have step1 : min (slope first) (slope second) * dist (point first) (point second) ≤
      min (slope first) (slope second) *
        (dist query (point first) + dist query (point second)) :=
    mul_le_mul_of_nonneg_left tri minNonneg
  have step2 : min (slope first) (slope second) * dist query (point first) ≤
      slope first * dist query (point first) :=
    mul_le_mul_of_nonneg_right (min_le_left _ _) dist_nonneg
  have step3 : min (slope first) (slope second) * dist query (point second) ≤
      slope second * dist query (point second) :=
    mul_le_mul_of_nonneg_right (min_le_right _ _) dist_nonneg
  linarith

/-- Necessary and sufficient condition for the lower envelope to lie below the
upper envelope everywhere: the finite pairwise condition `consistent` holds. -/
theorem orderIff (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ)
    (nonneg : ∀ index ∈ records, 0 ≤ slope index) :
    (∀ query : State,
      below records nonempty point lower slope query ≤
        above records nonempty point upper slope query) ↔
      consistent records point lower upper slope := by
  constructor
  · intro ordered first firstIn second secondIn
    have atSecond := ordered (point second)
    have lowSecond := Finset.le_sup'
      (fun index ↦ lower index - slope index * dist (point second) (point index)) firstIn
    have highSecond := Finset.inf'_le
      (fun index ↦ upper index + slope index * dist (point second) (point index)) secondIn
    change _ ≤ below records nonempty point lower slope (point second) at lowSecond
    change above records nonempty point upper slope (point second) ≤ _ at highSecond
    rw [dist_self, mul_zero, add_zero] at highSecond
    rw [dist_comm (point second) (point first)] at lowSecond
    have boundFirst : lower first - upper second ≤
        slope first * dist (point first) (point second) := by linarith
    have atFirst := ordered (point first)
    have lowFirst := Finset.le_sup'
      (fun index ↦ lower index - slope index * dist (point first) (point index)) firstIn
    have highFirst := Finset.inf'_le
      (fun index ↦ upper index + slope index * dist (point first) (point index)) secondIn
    change _ ≤ below records nonempty point lower slope (point first) at lowFirst
    change above records nonempty point upper slope (point first) ≤ _ at highFirst
    rw [dist_self, mul_zero, sub_zero] at lowFirst
    have boundSecond : lower first - upper second ≤
        slope second * dist (point first) (point second) := by linarith
    rcases le_total (slope first) (slope second) with hle | hle
    · simpa only [min_eq_left hle] using boundFirst
    · simpa only [min_eq_right hle] using boundSecond
  · exact crossedOrder records nonempty point lower upper slope nonneg

/-- Any objective satisfying the observation intervals and exact pointwise cones
must satisfy the finite pairwise consistency condition. -/
theorem inconsistentFalse (records : Finset Index)
    (point : Index → State) (lower upper slope : Index → ℝ)
    (objective : State → ℝ) (observed : fits records point lower upper objective)
    (regular : coneExact records point slope objective) :
    consistent records point lower upper slope := by
  intro first firstIn second secondIn
  have obsFirst := (observed first firstIn).1
  have obsSecond := (observed second secondIn).2
  have coneFirst := (abs_le.mp (regular first firstIn (point second))).1
  rw [dist_comm (point second) (point first)] at coneFirst
  have coneSecond := (abs_le.mp (regular second secondIn (point first))).2
  rcases le_total (slope first) (slope second) with hle | hle
  · rw [min_eq_left hle]
    linarith
  · rw [min_eq_right hle]
    linarith

/-- If the envelopes cross at any state (`above query < below query`), no
compatible objective exists. -/
theorem crossedFalse (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ)
    (query : State)
    (crossed : above records nonempty point upper slope query <
      below records nonempty point lower slope query) :
    ¬∃ objective : State → ℝ,
      fits records point lower upper objective ∧
        coneExact records point slope objective := by
  rintro ⟨objective, observed, regular⟩
  have within := exactBounds records nonempty point lower upper slope objective observed regular query
  linarith

/-- Envelope midpoint at `query`. -/
noncomputable def midpoint (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ) (query : State) : ℝ :=
  (below records nonempty point lower slope query +
    above records nonempty point upper slope query) / 2

/-- Envelope half-width at `query`. -/
noncomputable def halfWidth (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope : Index → ℝ) (query : State) : ℝ :=
  (above records nonempty point upper slope query -
    below records nonempty point lower slope query) / 2

/-- Exact algebraic decomposition of the worst-case distance from `estimate` to
an interval `[low, high]` into half-width plus midpoint calibration offset. -/
theorem midpointSplit (low high estimate : ℝ) (ordered : low ≤ high) :
    max |estimate - low| |estimate - high| =
      (high - low) / 2 + |estimate - (low + high) / 2| := by
  rcases le_total estimate ((low + high) / 2) with hle | hle
  · have diffNonpos : estimate - (low + high) / 2 ≤ 0 := by linarith
    have highNonpos : estimate - high ≤ 0 := by linarith
    rw [abs_of_nonpos diffNonpos, abs_of_nonpos highNonpos]
    have maxEq : max |estimate - low| (-(estimate - high)) = -(estimate - high) := by
      apply max_eq_right
      rw [abs_le]
      constructor <;> linarith
    linarith
  · have diffNonneg : 0 ≤ estimate - (low + high) / 2 := by linarith
    have lowNonneg : 0 ≤ estimate - low := by linarith
    rw [abs_of_nonneg diffNonneg, abs_of_nonneg lowNonneg]
    have maxEq : max (estimate - low) |estimate - high| = estimate - low := by
      apply max_eq_left
      rw [abs_le]
      constructor <;> linarith
    linarith

/-- Sharp error bound for any estimator `estimate` at `query`, decomposing into
the envelope half-width, the midpoint offset `|estimate - midpoint|`, and the
misspecification slack `slackTerm`. -/
theorem estimateBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope slack : Index → ℝ)
    (objective : State → ℝ) (observed : fits records point lower upper objective)
    (regular : coneValid records point slope slack objective) (query : State)
    (ordered : below records nonempty point lower slope query ≤
      above records nonempty point upper slope query)
    (estimate : ℝ) :
    |estimate - objective query| ≤
      halfWidth records nonempty point lower upper slope query +
        |estimate - midpoint records nonempty point lower upper slope query| +
        slackTerm records nonempty point slack query := by
  have within := slackBounds records nonempty point lower upper slope slack
    objective observed regular query
  have split := midpointSplit
    (below records nonempty point lower slope query)
    (above records nonempty point upper slope query)
    estimate ordered
  have left := abs_le.mp (le_max_left
    |estimate - below records nonempty point lower slope query|
    |estimate - above records nonempty point upper slope query|)
  have right := abs_le.mp (le_max_right
    |estimate - below records nonempty point lower slope query|
    |estimate - above records nonempty point upper slope query|)
  change _ = halfWidth records nonempty point lower upper slope query +
    |estimate - midpoint records nonempty point lower upper slope query| at split
  rw [← split]
  apply abs_le.mpr
  constructor <;> linarith

/-- Specializing `estimateBound` to the envelope midpoint eliminates the
calibration offset term. -/
theorem midpointBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper slope slack : Index → ℝ)
    (objective : State → ℝ) (observed : fits records point lower upper objective)
    (regular : coneValid records point slope slack objective) (query : State) :
    |midpoint records nonempty point lower upper slope query - objective query| ≤
      halfWidth records nonempty point lower upper slope query +
        slackTerm records nonempty point slack query := by
  have within := slackBounds records nonempty point lower upper slope slack
    objective observed regular query
  unfold midpoint halfWidth
  apply abs_le.mpr
  constructor <;> linarith

/-- Exclusion radius around recorded state `index` at target level `level`. -/
noncomputable def radius (lower slope : Index → ℝ) (level : ℝ) (index : Index) : ℝ :=
  (lower index - level) / slope index

/-- Sublevel feasibility predicate `below query ≤ level`. -/
def feasible (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower slope : Index → ℝ) (level : ℝ) (query : State) : Prop :=
  below records nonempty point lower slope query ≤ level

/-- Exclusion-ball characterization of the feasible sublevel set. -/
theorem exclusionIff (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower slope : Index → ℝ)
    (pos : ∀ index ∈ records, 0 < slope index) (level : ℝ) (query : State) :
    feasible records nonempty point lower slope level query ↔
      ∀ index ∈ records, radius lower slope level index ≤ dist query (point index) := by
  unfold feasible below radius
  rw [Finset.sup'_le_iff]
  constructor
  · intro bound index member
    have h := bound index member
    have hpos := pos index member
    rw [div_le_iff₀ hpos]
    linarith
  · intro bound index member
    have h := bound index member
    have hpos := pos index member
    rw [div_le_iff₀ hpos] at h
    linarith

/-- Pairwise cone-dominance relation `dominates point lower slope j i` meaning
constraint `i` dominates constraint `j` everywhere. -/
def dominates (point : Index → State) (lower slope : Index → ℝ) (j i : Index) : Prop :=
  0 ≤ slope i ∧ slope i ≤ slope j ∧
    lower j ≤ lower i - slope i * dist (point i) (point j)

theorem domRefl (point : Index → State) (lower slope : Index → ℝ)
    (i : Index) (nonneg : 0 ≤ slope i) :
    dominates point lower slope i i := by
  refine ⟨nonneg, le_rfl, ?_⟩
  simp only [dist_self, mul_zero, sub_zero, le_rfl]

theorem domTrans (point : Index → State) (lower slope : Index → ℝ)
    {k j i : Index}
    (kj : dominates point lower slope k j)
    (ji : dominates point lower slope j i) :
    dominates point lower slope k i := by
  rcases kj with ⟨_, slopeKJ, boundKJ⟩
  rcases ji with ⟨nonnegI, slopeJI, boundJI⟩
  refine ⟨nonnegI, slopeJI.trans slopeKJ, ?_⟩
  have tri := mul_le_mul_of_nonneg_left
    (dist_triangle (point i) (point j) (point k)) nonnegI
  have step : slope i * dist (point j) (point k) ≤ slope j * dist (point j) (point k) :=
    mul_le_mul_of_nonneg_right slopeJI dist_nonneg
  linarith

/-- If constraint `i` dominates constraint `j`, its lower cone lies above `j`'s
lower cone at every query state. -/
theorem domBelow (point : Index → State) (lower slope : Index → ℝ)
    {j i : Index} (dom : dominates point lower slope j i) (query : State) :
    lower j - slope j * dist query (point j) ≤
      lower i - slope i * dist query (point i) := by
  rcases dom with ⟨nonnegI, slopeIJ, boundIJ⟩
  have tri : dist query (point i) ≤ dist (point i) (point j) + dist query (point j) := by
    simpa only [dist_comm query (point i), dist_comm query (point j)] using
      dist_triangle (point i) (point j) query
  have step1 := mul_le_mul_of_nonneg_left tri nonnegI
  have step2 : slope i * dist query (point j) ≤ slope j * dist query (point j) :=
    mul_le_mul_of_nonneg_right slopeIJ dist_nonneg
  linarith

/-- Deleting all constraints dominated by a retained subset leaves the lower
envelope identical at every state. -/
theorem deleteBelow (full retained : Finset Index) (nonempty : retained.Nonempty)
    (subset : retained ⊆ full) (point : Index → State) (lower slope : Index → ℝ)
    (covered : ∀ j ∈ full, ∃ i ∈ retained, dominates point lower slope j i)
    (query : State) :
    below retained nonempty point lower slope query =
      below full (nonempty.mono subset) point lower slope query := by
  apply le_antisymm
  · apply Finset.sup'_le
    intro i member
    exact Finset.le_sup' (fun k ↦ lower k - slope k * dist query (point k)) (subset member)
  · apply Finset.sup'_le
    intro j member
    rcases covered j member with ⟨i, inRetained, dom⟩
    have step := domBelow point lower slope dom query
    have bound := Finset.le_sup' (fun k ↦ lower k - slope k * dist query (point k)) inRetained
    change _ ≤ below retained nonempty point lower slope query at bound
    linarith

/-- Single-evaluation descent vs. ball-excision dichotomy. -/
theorem stepDichotomy (best gamma alpha y err lam : ℝ) (fx : ℝ)
    (obs : |y - fx| ≤ err) (posLam : 0 < lam) :
    (y + err ≤ best - alpha * gamma → fx ≤ y + err ∧ y + err ≤ best - alpha * gamma) ∧
      (best - alpha * gamma < y + err →
        ((1 - alpha) * gamma - 2 * err) / lam < (y - err - (best - gamma)) / lam) := by
  constructor
  · intro descent
    have fxBound : fx ≤ y + err := by linarith [(abs_le.mp obs).1]
    exact ⟨fxBound, descent⟩
  · intro excise
    rw [div_lt_div_iff_of_pos_right posLam]
    linarith

/-- Causal separation: if every later evaluation `q > p` lies outside the ball
of radius `rad p` around `center p`, then every distinct pair `p ≠ q` is
separated by at least `min (rad p) (rad q)`. -/
theorem causalSep (steps : Finset ℕ) (center : ℕ → State) (rad : ℕ → ℝ)
    (causal : ∀ p ∈ steps, ∀ q ∈ steps, p < q → rad p ≤ dist (center q) (center p))
    {p q : ℕ} (pIn : p ∈ steps) (qIn : q ∈ steps) (neq : p ≠ q) :
    min (rad p) (rad q) ≤ dist (center p) (center q) := by
  rcases lt_or_gt_of_ne neq with hlt | hgt
  · have step := causal p pIn q qIn hlt
    rw [dist_comm] at step
    exact (min_le_left (rad p) (rad q)).trans step
  · have step := causal q qIn p pIn hgt
    exact (min_le_right (rad p) (rad q)).trans step

/-- Concrete 1D counterexample: one-way causal separation `rad 0 ≤ dist (x 1) (x 0)`
does NOT make the variable-radius half-balls `B(x p, rad p / 2)` disjoint when
`rad 1 > rad 0`. -/
theorem causalOverlap :
    let x0 : ℝ := 0
    let x1 : ℝ := 1
    let r0 : ℝ := 1
    let r1 : ℝ := 4
    r0 ≤ dist x1 x0 ∧
      ∃ z : ℝ, dist z x0 < r0 / 2 ∧ dist z x1 < r1 / 2 := by
  dsimp only
  constructor
  · norm_num [Real.dist_eq]
  · refine ⟨(1 : ℝ) / 4, ?_, ?_⟩ <;> norm_num [Real.dist_eq]

/-- Within any scale band where `rho ≤ rad p` for all `p ∈ band`, causal
exclusion makes the open balls of radius `rho / 2` pairwise disjoint. -/
theorem bandDisjoint (steps band : Finset ℕ) (subset : band ⊆ steps)
    (center : ℕ → State) (rad : ℕ → ℝ) (rho : ℝ)
    (bandLower : ∀ p ∈ band, rho ≤ rad p)
    (causal : ∀ p ∈ steps, ∀ q ∈ steps, p < q → rad p ≤ dist (center q) (center p))
    {p q : ℕ} (pIn : p ∈ band) (qIn : q ∈ band) (neq : p ≠ q) :
    Disjoint (Metric.ball (center p) (rho / 2)) (Metric.ball (center q) (rho / 2)) := by
  rw [Set.disjoint_left]
  intro z zInP zInQ
  rw [Metric.mem_ball] at zInP zInQ
  have sep := causalSep steps center rad causal (subset pIn) (subset qIn) neq
  have minLow : rho ≤ min (rad p) (rad q) :=
    le_min (bandLower p pIn) (bandLower q qIn)
  have tri : dist (center p) (center q) ≤ dist (center p) z + dist z (center q) :=
    dist_triangle (center p) z (center q)
  rw [dist_comm z (center p)] at zInP
  linarith

/-- Measure-packing cardinality bound on a single scale band: if each disjoint
half-ball in `band` has measure `mass p ≥ unitMass > 0` and their sum is at most
`totalMass`, then `band.card ≤ totalMass / unitMass`. -/
theorem bandCount (band : Finset ℕ) (mass : ℕ → ℝ) (unitMass totalMass : ℝ)
    (posUnit : 0 < unitMass)
    (lowerMass : ∀ p ∈ band, unitMass ≤ mass p)
    (upperSum : ∑ p ∈ band, mass p ≤ totalMass) :
    (band.card : ℝ) ≤ totalMass / unitMass := by
  rw [le_div_iff₀ posUnit]
  have sumLower : (band.card : ℝ) * unitMass ≤ ∑ p ∈ band, mass p := by
    simpa only [Finset.sum_const, nsmul_eq_mul] using
      Finset.sum_le_sum lowerMass
  exact sumLower.trans upperSum

/-- Multiscale cardinality bound across a finite family of scale bands covering
all evaluated steps. -/
theorem multiscaleCount {Band : Type*}
    (bands : Finset Band) (steps : Finset ℕ) (member : Band → Finset ℕ)
    (covered : steps ⊆ bands.biUnion member)
    (cap : Band → ℝ)
    (bounded : ∀ b ∈ bands, ((member b).card : ℝ) ≤ cap b) :
    (steps.card : ℝ) ≤ ∑ b ∈ bands, cap b := by
  have cardSub : steps.card ≤ (bands.biUnion member).card :=
    Finset.card_le_card covered
  have cardUnion : (bands.biUnion member).card ≤ ∑ b ∈ bands, (member b).card :=
    Finset.card_biUnion_le
  have castBound : (steps.card : ℝ) ≤ ((∑ b ∈ bands, (member b).card : ℕ) : ℝ) :=
    Nat.cast_le.mpr (cardSub.trans cardUnion)
  have castSum : ((∑ b ∈ bands, (member b).card : ℕ) : ℝ) =
      ∑ b ∈ bands, ((member b).card : ℝ) := by
    push_cast
    rfl
  rw [castSum] at castBound
  exact castBound.trans (Finset.sum_le_sum bounded)

/-- Two-point chord slope between `point first` and `point second` evaluated at `query`. -/
noncomputable def chordSlope (point : Index → State) (value : Index → ℝ)
    (query : State) (first second : Index) : ℝ :=
  (value second - value first) / (dist query (point second) + dist query (point first))

/-- Two-point chord interpolant between `point first` and `point second` at `query`. -/
noncomputable def chordEval (point : Index → State) (value : Index → ℝ)
    (query : State) (first second : Index) : ℝ :=
  value first + chordSlope point value query first second * dist query (point first)

/-- Whenever `query` has positive total distance to `point first` and `point second`,
the upper cone from `first` and the lower cone from `second` with slope `chordSlope`
coincide at `chordEval`. -/
theorem chordMeet (point : Index → State) (value : Index → ℝ)
    (query : State) (first second : Index)
    (posDist : 0 < dist query (point second) + dist query (point first)) :
    chordEval point value query first second =
      value second - chordSlope point value query first second * dist query (point second) := by
  unfold chordEval chordSlope
  have neDist : dist query (point second) + dist query (point first) ≠ 0 := ne_of_gt posDist
  field_simp [neDist]
  ring

/-- Pointwise minimal slope optimality: any function `objective` matching `value`
at `point first` and `point second` with local slope `lam` at `query` satisfies
`chordSlope point value query first second ≤ lam`. -/
theorem chordLower (point : Index → State) (value : Index → ℝ)
    (objective : State → ℝ) (query : State) (first second : Index) (lam : ℝ)
    (posDist : 0 < dist query (point second) + dist query (point first))
    (fitFirst : objective (point first) = value first)
    (fitSecond : objective (point second) = value second)
    (coneFirst : |objective query - objective (point first)| ≤ lam * dist query (point first))
    (coneSecond : |objective (point second) - objective query| ≤ lam * dist query (point second)) :
    chordSlope point value query first second ≤ lam := by
  unfold chordSlope
  rw [div_le_iff₀ posDist]
  have firstBound := (abs_le.mp coneFirst).2
  have secondBound := (abs_le.mp coneSecond).2
  linarith

/-- Any slope `lam` upper-bounding the two-point chord slope `chordSlope` keeps
the lower cone from `second` at or below the upper cone from `first` at `query`. -/
theorem chordOrder (point : Index → State) (value : Index → ℝ)
    (query : State) (first second : Index) (lam : ℝ)
    (posDist : 0 < dist query (point second) + dist query (point first))
    (slopeLe : chordSlope point value query first second ≤ lam) :
    value second - lam * dist query (point second) ≤
      value first + lam * dist query (point first) := by
  unfold chordSlope at slopeLe
  rw [div_le_iff₀ posDist] at slopeLe
  linarith

#print axioms slackBounds
#print axioms exactBounds
#print axioms crossedOrder
#print axioms orderIff
#print axioms inconsistentFalse
#print axioms crossedFalse
#print axioms midpointSplit
#print axioms estimateBound
#print axioms midpointBound
#print axioms exclusionIff
#print axioms domRefl
#print axioms domTrans
#print axioms domBelow
#print axioms deleteBelow
#print axioms stepDichotomy
#print axioms causalSep
#print axioms causalOverlap
#print axioms bandDisjoint
#print axioms bandCount
#print axioms multiscaleCount
#print axioms chordMeet
#print axioms chordLower
#print axioms chordOrder

end BauxResearch.Optimization.SlopeEnvelope
