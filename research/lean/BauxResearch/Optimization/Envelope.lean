import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Tactic.Linarith

/-!
# Exact information in interval observations

For a fixed metric and a justified Lipschitz constant, noisy observations
constrain an objective to intervals at recorded states. Finite lower and upper
Lipschitz envelopes describe exactly which values remain possible at any query.
They give a necessary and sufficient finite test for lossless deletion of
observation constraints. This is set-membership information, not a posterior
distribution, sufficient statistic for ENN fitting, or production archive rule.

The constant and observation intervals must be justified independently. No
small archive, cheap distance computation, or learning rate follows merely from
the ability to test redundancy. The construction is classical McShane--Whitney
envelope mathematics, not a claim of a new theorem in the literature.
-/

namespace BauxResearch.Optimization.Envelope

open scoped NNReal

variable {State Index : Type*} [PseudoMetricSpace State]

/-- Constraints on objective values; regularity is stated separately. -/
def fits (records : Finset Index) (point : Index → State)
    (lower upper : Index → ℝ) (objective : State → ℝ) : Prop :=
  ∀ index ∈ records, lower index ≤ objective (point index) ∧
    objective (point index) ≤ upper index

/-- The finite compatibility test includes validity of each interval by taking
the same index twice. It uses only recorded intervals and pairwise distances. -/
def consistent (records : Finset Index) (point : Index → State)
    (lower upper : Index → ℝ) (constant : ℝ≥0) : Prop :=
  ∀ first ∈ records, ∀ second ∈ records,
    lower first ≤ upper second + constant * dist (point first) (point second)

noncomputable def below (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower : Index → ℝ) (constant : ℝ≥0) (query : State) : ℝ :=
  records.sup' nonempty fun index => lower index - constant * dist (point index) query

noncomputable def above (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (upper : Index → ℝ) (constant : ℝ≥0) (query : State) : ℝ :=
  records.inf' nonempty fun index => upper index + constant * dist (point index) query

theorem belowLip (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower : Index → ℝ) (constant : ℝ≥0) :
    LipschitzWith constant (below records nonempty point lower constant) := by
  apply LipschitzWith.of_le_add_mul
  intro left right
  apply Finset.sup'_le
  intro index member
  have bound := Finset.le_sup' (fun index => lower index - constant * dist (point index) right) member
  have triangle := mul_le_mul_of_nonneg_left
    (dist_triangle (point index) left right) constant.coe_nonneg
  change _ ≤ below records nonempty point lower constant right at bound
  linarith

theorem aboveLip (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (upper : Index → ℝ) (constant : ℝ≥0) :
    LipschitzWith constant (above records nonempty point upper constant) := by
  apply LipschitzWith.of_le_add_mul
  intro left right
  obtain ⟨index, member, equality⟩ := Finset.exists_mem_eq_inf' nonempty
    (fun index => upper index + constant * dist (point index) right)
  have bound := Finset.inf'_le (fun index => upper index + constant * dist (point index) left) member
  have triangle := mul_le_mul_of_nonneg_left
    (dist_triangle (point index) right left) constant.coe_nonneg
  rw [dist_comm right left] at triangle
  change above records nonempty point upper constant left ≤ _ at bound
  change above records nonempty point upper constant right = _ at equality
  linarith

/-- Every compatible objective lies between the envelopes at every state. -/
theorem bounds (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : fits records point lower upper objective) (query : State) :
    below records nonempty point lower constant query ≤ objective query ∧
      objective query ≤ above records nonempty point upper constant query := by
  constructor
  · apply Finset.sup'_le
    intro index member
    have observation := (observed index member).1
    have transport := regular.le_add_mul (point index) query
    linarith
  · apply Finset.le_inf'
    intro index member
    have observation := (observed index member).2
    have transport := regular.le_add_mul query (point index)
    rw [dist_comm query (point index)] at transport
    linarith

/-- Compatibility constructs an admissible objective, rather than presuming
that the recorded intervals describe one. The lower envelope is a witness. -/
theorem belowFits (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (compatible : consistent records point lower upper constant) :
    fits records point lower upper (below records nonempty point lower constant) := by
  intro index member
  constructor
  · simpa only [below, dist_self, mul_zero, sub_zero] using Finset.le_sup'
      (fun other => lower other - constant * dist (point other) (point index)) member
  · apply Finset.sup'_le
    intro other inside
    have pair := compatible other inside index member
    linarith

theorem aboveFits (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (compatible : consistent records point lower upper constant) :
    fits records point lower upper (above records nonempty point upper constant) := by
  intro index member
  constructor
  · apply Finset.le_inf'
    intro other inside
    have pair := compatible index member other inside
    simpa only [dist_comm (point index) (point other)] using pair
  · simpa only [above, dist_self, mul_zero, add_zero] using Finset.inf'_le
      (fun other => upper other + constant * dist (point other) (point index)) member

/-- A finite, necessary and sufficient consistency condition for this model
class. An inconsistent archive cannot establish guarantees by vacuity. -/
theorem feasibleIff (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0) :
    (∃ objective, LipschitzWith constant objective ∧ fits records point lower upper objective) ↔
      consistent records point lower upper constant := by
  constructor
  · rintro ⟨objective, regular, observed⟩ first firstIn second secondIn
    have firstBound := (observed first firstIn).1
    have secondBound := (observed second secondIn).2
    have transport := regular.le_add_mul (point first) (point second)
    linarith
  · intro compatible
    exact ⟨_, belowLip records nonempty point lower constant,
      belowFits records nonempty point lower upper constant compatible⟩

/-- Every value in the envelope interval is attained by a compatible objective.
Mathlib's Lipschitz max/min closure constructs the witness by clamping a
constant between the envelopes. Thus the interval is exact, not just an upper
bound on possible uncertainty. -/
theorem valueIff (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (compatible : consistent records point lower upper constant) (query : State) (value : ℝ) :
    (∃ objective, LipschitzWith constant objective ∧
      fits records point lower upper objective ∧ objective query = value) ↔
    below records nonempty point lower constant query ≤ value ∧
      value ≤ above records nonempty point upper constant query := by
  constructor
  · rintro ⟨objective, regular, observed, equality⟩
    simpa only [equality] using bounds records nonempty point lower upper constant
      objective regular observed query
  · intro within
    let low := below records nonempty point lower constant
    let high := above records nonempty point upper constant
    refine ⟨fun state => min (max (low state) value) (high state), ?_, ?_, ?_⟩
    · simpa only [max_self] using
        ((belowLip records nonempty point lower constant).max_const value).min
          (aboveLip records nonempty point upper constant)
    · intro index member
      have lowFit := belowFits records nonempty point lower upper constant compatible index member
      have highFit := aboveFits records nonempty point lower upper constant compatible index member
      exact ⟨le_min (lowFit.1.trans (le_max_left _ _)) highFit.1,
        (min_le_right _ _).trans highFit.2⟩
    · exact (congrArg (fun x => min x (high query)) (max_eq_right within.1)).trans
        (min_eq_left within.2)

/-- Exact lossless deletion test. All witnesses are retained observations; a
batch of deletions cannot justify each record by another record being deleted.
This preserves the compatible objective class, not ENN weights or its mean. -/
theorem retentionIff (full retained : Finset Index) (nonempty : retained.Nonempty)
    (subset : retained ⊆ full) (point : Index → State) (lower upper : Index → ℝ)
    (constant : ℝ≥0) (compatible : consistent retained point lower upper constant) :
    (∀ objective, LipschitzWith constant objective →
      (fits retained point lower upper objective ↔ fits full point lower upper objective)) ↔
    (∀ index ∈ full,
      lower index ≤ below retained nonempty point lower constant (point index) ∧
        above retained nonempty point upper constant (point index) ≤ upper index) := by
  constructor
  · intro preserved index member
    have lowFit := (preserved _ (belowLip retained nonempty point lower constant)).mp
      (belowFits retained nonempty point lower upper constant compatible)
    have highFit := (preserved _ (aboveLip retained nonempty point upper constant)).mp
      (aboveFits retained nonempty point lower upper constant compatible)
    exact ⟨(lowFit index member).1, (highFit index member).2⟩
  · intro implied objective regular
    constructor
    · intro observed index member
      have bound := bounds retained nonempty point lower upper constant objective regular observed (point index)
      exact ⟨(implied index member).1.trans bound.1, bound.2.trans (implied index member).2⟩
    · intro observed index member
      exact observed index (subset member)

/-- A finite, approximate deletion test controls both envelopes everywhere.
The allowance is a value error, not a geometric radius or a count of deleted
records. This does not guarantee the existence of a small qualifying subset. -/
theorem retentionBound (full retained : Finset Index) (nonempty : retained.Nonempty)
    (subset : retained ⊆ full) (point : Index → State) (lower upper : Index → ℝ)
    (constant : ℝ≥0) (allowance : ℝ)
    (implied : ∀ index ∈ full,
      lower index - allowance ≤ below retained nonempty point lower constant (point index) ∧
        above retained nonempty point upper constant (point index) ≤ upper index + allowance)
    (query : State) :
    below full (nonempty.mono subset) point lower constant query - allowance ≤
        below retained nonempty point lower constant query ∧
      below retained nonempty point lower constant query ≤
        below full (nonempty.mono subset) point lower constant query ∧
      above full (nonempty.mono subset) point upper constant query ≤
        above retained nonempty point upper constant query ∧
      above retained nonempty point upper constant query ≤
        above full (nonempty.mono subset) point upper constant query + allowance := by
  have low : below full (nonempty.mono subset) point lower constant query ≤
      below retained nonempty point lower constant query + allowance := by
    apply Finset.sup'_le
    intro index member
    have bound := (implied index member).1
    have transport := (belowLip retained nonempty point lower constant).le_add_mul (point index) query
    linarith
  have high : above retained nonempty point upper constant query - allowance ≤
      above full (nonempty.mono subset) point upper constant query := by
    apply Finset.le_inf'
    intro index member
    have bound := (implied index member).2
    have transport := (aboveLip retained nonempty point upper constant).le_add_mul query (point index)
    rw [dist_comm query (point index)] at transport
    linarith
  refine ⟨by linarith, ?_, ?_, by linarith⟩
  · apply Finset.sup'_le
    intro index member
    exact Finset.le_sup' (fun index => lower index - constant * dist (point index) query)
      (subset member)
  · apply Finset.le_inf'
    intro index member
    exact Finset.inf'_le (fun index => upper index + constant * dist (point index) query)
      (subset member)

/-- An interval gives a valid radius around any estimator, including a fitted
ENN mean. Neither response-independent weights nor calibrated ENN variance is
needed for this deterministic implication. The interval assumptions remain. -/
theorem estimateBound (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (objective : State → ℝ) (regular : LipschitzWith constant objective)
    (observed : fits records point lower upper objective) (query : State) (estimate : ℝ) :
    |estimate - objective query| ≤
      max |estimate - below records nonempty point lower constant query|
        |estimate - above records nonempty point upper constant query| := by
  have within := bounds records nonempty point lower upper constant objective regular observed query
  have left := abs_le.mp (le_max_left
    |estimate - below records nonempty point lower constant query|
    |estimate - above records nonempty point upper constant query|)
  have right := abs_le.mp (le_max_right
    |estimate - below records nonempty point lower constant query|
    |estimate - above records nonempty point upper constant query|)
  apply abs_le.mpr
  constructor <;> linarith

/-- The preceding radius is attained by a compatible objective. It cannot be
uniformly reduced using only these interval observations and this regularity
class, even when the estimate depends arbitrarily on the observations. -/
theorem attainedError (records : Finset Index) (nonempty : records.Nonempty)
    (point : Index → State) (lower upper : Index → ℝ) (constant : ℝ≥0)
    (compatible : consistent records point lower upper constant) (query : State) (estimate : ℝ) :
    ∃ objective, LipschitzWith constant objective ∧ fits records point lower upper objective ∧
      |estimate - objective query| =
        max |estimate - below records nonempty point lower constant query|
          |estimate - above records nonempty point upper constant query| := by
  by_cases larger : |estimate - below records nonempty point lower constant query| ≤
      |estimate - above records nonempty point upper constant query|
  · exact ⟨_, aboveLip records nonempty point upper constant,
      aboveFits records nonempty point lower upper constant compatible, (max_eq_right larger).symm⟩
  · exact ⟨_, belowLip records nonempty point lower constant,
      belowFits records nonempty point lower upper constant compatible,
      (max_eq_left (le_of_not_ge larger)).symm⟩

/-- Saturated observations determine every intermediate value exactly. Thus
nearest-observation distance alone need not equal the uncertainty for a fixed
observed data set: on [0,1], these two observations leave zero uncertainty. -/
theorem saturated (objective : ℝ → ℝ) (regular : LipschitzWith 1 objective)
    (zero : objective 0 = 0) (one : objective 1 = 1)
    (query : ℝ) (left : 0 ≤ query) (right : query ≤ 1) : objective query = query := by
  have lower := regular.le_add_mul 1 query
  have upper := regular.le_add_mul query 0
  simp only [NNReal.coe_one, one_mul, Real.dist_eq, one,
    abs_of_nonneg (sub_nonneg.mpr right)] at lower
  simp only [NNReal.coe_one, one_mul, Real.dist_eq, zero, sub_zero,
    abs_of_nonneg left, zero_add] at upper
  linarith

#print axioms belowLip
#print axioms aboveLip
#print axioms feasibleIff
#print axioms valueIff
#print axioms retentionIff
#print axioms retentionBound
#print axioms estimateBound
#print axioms attainedError
#print axioms saturated

end BauxResearch.Optimization.Envelope
