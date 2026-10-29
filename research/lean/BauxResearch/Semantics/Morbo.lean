import BauxResearch.Semantics.Candidate
import BauxResearch.Semantics.Opt

/-!
# Exact ENN 0.4.1 MORBO scalarization and reseeding segments

This module models the source operation order for MORBO weight resampling,
Chebyshev scalarization, and per-ask Sobol reseeding.  `LogOp` is an explicit
boundary for Rust's platform `f64::ln`; no bitwise correspondence theorem for
that external implementation is assumed here.
-/

namespace BauxResearch.Semantics.Morbo

open Sampling

/-- Source configuration controlling when MORBO redraws scalarization weights. -/
inductive Rescalarize where
  | onRestart
  | onPropose
  deriving Repr, DecidableEq

/-- Platform logarithm boundary used by `f64::ln`. -/
abbrev LogOp := Word → Word

def positiveInfinity : Word := 0x7ff0000000000000
def negativeInfinity : Word := 0xfff0000000000000
def dirichletFloor : Word := 0x01a56e1fc2f8f359

/-- Rust core's source-level `f64::min` fallback: ignore one NaN and select the RHS on equality.

Native code may select either operand for equal inputs, notably signed zeros; executable-artifact
correspondence must therefore use the corresponding relation rather than assume this RHS choice.
-/
def minNumber (left right : Word) : Word :=
  if Float.nanDouble left || Float.leDouble right left then right else left

/-- Rust core's source-level `f64::max` fallback, with the same equal-input qualification. -/
def maxNumber (left right : Word) : Word :=
  if Float.nanDouble left || Float.leDouble left right then right else left

/-- Exact draw and binary64 operation order of `sample_dirichlet_weights`, parameterized only by
the platform logarithm implementation. -/
def sampleWeights (ln : LogOp) (count : Nat) : Draw Row := do
  let mut samples := #[]
  let mut sum : Word := 0
  for _ in [:count] do
    let unit ← standard
    let logWord := ln unit
    let positiveLog := (Float.negDouble (Float.wordDouble logWord)).bits
    let sample := maxNumber positiveLog dirichletFloor
    samples := samples.push sample
    sum := add sum sample
  let mut weights := #[]
  for sample in samples do
    weights := weights.push (divide sample sum)
  return weights

structure WeightState where
  mode : Rescalarize
  count : Nat
  weights : Row
  deriving Repr, DecidableEq

/-- `TrustRegionState::resample_on_propose`, including its caller-owned RNG transition. -/
def redraw (ln : LogOp) (state : WeightState) : Draw WeightState := do
  match state.mode with
  | .onRestart => return state
  | .onPropose =>
    let weights ← sampleWeights ln state.count
    return { state with weights }

/-- An ndarray-shaped binary64 matrix. The row storage is checked when indexed. -/
structure DenseMatrix where
  rows : Matrix
  cols : Nat
  deriving Repr, DecidableEq

/-- Exact binary64 loop and rounding order of `scalarize_with_ranges`.

The source checks the declared ndarray column count and range lengths. A short stored row or weight
vector is represented as the source's indexing panic; such ragged rows cannot be constructed by
ndarray itself.
-/
def scalarize (y : DenseMatrix) (yMin yMax weights : Row) (alpha : Word)
    (numMetrics : Nat) (clip : Bool) : Except Failure Row := do
  if y.cols != numMetrics || yMin.size != numMetrics || yMax.size != numMetrics then
    throw .invalid
  let mut scores := #[]
  for rowIndex in [:y.rows.size] do
    let some row := y.rows[rowIndex]? | throw .panic
    let mut tMin := positiveInfinity
    let mut tSum : Word := 0
    for metric in [:numMetrics] do
      let some value := row[metric]? | throw .panic
      let some low := yMin[metric]? | throw .panic
      let some high := yMax[metric]? | throw .panic
      let some weight := weights[metric]? | throw .panic
      let denominator := sub high low
      let normalized := if Float.leDouble denominator 0 then
          0x3fe0000000000000
        else
          let raw := divide (sub value low) denominator
          if clip then Opt.clampUnit raw else raw
      let term := mul normalized weight
      tMin := minNumber tMin term
      tSum := add tSum term
    scores := scores.push (add tMin (mul alpha tSum))
  return scores

/-- Candidate-local min/max scan used by incumbent selection. -/
def localRanges (y : DenseMatrix) : Except Failure (Row × Row) := do
  let mut yMin := Array.replicate y.cols positiveInfinity
  let mut yMax := Array.replicate y.cols negativeInfinity
  for rowIndex in [:y.rows.size] do
    let some row := y.rows[rowIndex]? | throw .panic
    for metric in [:y.cols] do
      let some value := row[metric]? | throw .panic
      yMin := yMin.set! metric (minNumber yMin[metric]! value)
      yMax := yMax.set! metric (maxNumber yMax[metric]! value)
  return (yMin, yMax)

/-- ENN 0.4.1 incumbent scalarization derives ranges from the candidate rows. -/
def scalarizeLocal (y : DenseMatrix) (weights : Row) (alpha : Word)
    (numMetrics : Nat) : Except Failure Row := do
  let (yMin, yMax) ← localRanges y
  scalarize y yMin yMax weights alpha numMetrics true

/-- Fresh scrambled Sobol state for a MORBO ask. This consumes a private StdRng seeded from the
optimizer state, not the caller's RNG. -/
def reseedSobol (base restart observations arms : Word) (engine : Option Sobol.Engine) :
    Except Failure (Option Sobol.Engine) :=
  match engine with
  | none => .ok none
  | some current =>
    match Sobol.create current.dirs.size with
    | none => .error .invalid
    | some fresh =>
      let seed := Candidate.sobolSeed base restart observations arms
      let (result, _) := run (Sobol.scramble fresh) (seeded seed fresh.dirs.size)
      result.map some

theorem emptyWeights (ln : LogOp) (rng : RngState) :
    run (sampleWeights ln 0) rng = (.ok #[], rng) := by
  simp [sampleWeights, run]
  rfl

theorem restartKeeps (ln : LogOp) (count : Nat) (weights : Row) (rng : RngState) :
    run (redraw ln { mode := .onRestart, count, weights }) rng =
      (.ok { mode := .onRestart, count, weights }, rng) := by
  rfl

theorem shapeMismatch :
    scalarize { rows := #[#[one]], cols := 1 } #[0] #[one] #[one] 0 2 true =
      .error .invalid := by
  decide +kernel

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem localFixture :
    scalarizeLocal
      { rows := #[#[numWord 10, numWord 9], #[numWord 9, numWord 10]], cols := 2 }
      #[0x3fe0000000000000, 0x3fe0000000000000]
      0x3fb999999999999a 2 =
      .ok #[0x3fa999999999999a, 0x3fa999999999999a] := by
  simp [scalarizeLocal, localRanges, scalarize, List.range'_succ]
  decide +kernel

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
/-- The upstream degenerate-range regression, checked at every binary64 rounding point. -/
theorem degenerateFixture :
    scalarize
      { rows := #[#[one, numWord 2], #[one, numWord 3]], cols := 2 }
      #[one, numWord 2] #[one, numWord 5]
      #[0x3fe0000000000000, 0x3fe0000000000000]
      0x3fb999999999999a 2 true =
      .ok #[0x3f9999999999999a, 0x3fcaaaaaaaaaaaaa] := by
  simp [scalarize, List.range'_succ]
  decide +kernel

theorem reseedAbsent (base restart observations arms : Word) :
    reseedSobol base restart observations arms none = .ok none := by
  rfl

theorem zeroReject (base restart observations arms : Word) :
    reseedSobol base restart observations arms
      (some { dirs := #[], previous := #[] }) = .error .invalid := by
  rfl

def oneReady : Bool :=
    match reseedSobol 0 0 0 1 (Sobol.create 1) with
    | .ok (some engine) =>
      engine.count == 0 && engine.previous == #[0] && engine.shift.map Array.size == some 1
    | _ => false

set_option maxHeartbeats 0 in
set_option maxRecDepth 65536 in
theorem oneReseed : oneReady = true := by
  decide +kernel

#print axioms emptyWeights
#print axioms restartKeeps
#print axioms shapeMismatch
#print axioms localFixture
#print axioms degenerateFixture
#print axioms reseedAbsent
#print axioms zeroReject
#print axioms oneReseed

end BauxResearch.Semantics.Morbo
