import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Generated.Enn.Funs
import BauxResearch.Semantics.Optimizer

/-!
# Extracted ENN 0.4.1 Optimizer Correspondence

Concrete zero-axiom correspondence theorems connecting the Aeneas-extracted
upstream `ennbo` optimizer functions in `BauxResearch.Generated.Optimizer`
to the verified executable semantics in `BauxResearch.Semantics.Optimizer`,
`Lifecycle`, `Tell`, `Predict`, `Incumbent`, and `Bounds`.
-/

namespace BauxResearch.Extraction.Optimizer

open Aeneas Aeneas.Std Aeneas.Std.WP Result
open BauxResearch.Generated.Optimizer
open BauxResearch.Semantics

/-! ## 1. Coordinate & Matrix Correspondence (`Bounds`, `Tell`, `Optimizer`) -/

/-- `toUnitWord` in the extracted external bindings is definitionally equal to
`Bounds.toUnit`. -/
theorem unitWord (v low high : Sampling.Word) :
    toUnitWord v low high = Bounds.toUnit v low high := by
  rfl

/-- `fromUnitWord` in the extracted external bindings is definitionally equal to
the binary64 affine map `low + x * (high - low)` in `Optimizer.fromUnitRow`. -/
theorem fromWord (x low high : Sampling.Word) :
    fromUnitWord x low high =
      Sampling.add low (Sampling.mul x (Sampling.sub high low)) := by
  rfl

/-! ## 2. `TrustRegionState::is_morbo` Correspondence (`Lifecycle.isMorbo`) -/

/-- Extracted `TrustRegionState.is_morbo` on `Turbo` agrees with `Lifecycle.isMorbo`. -/
theorem morboTurbo (tr : trust_region.TurboTrustRegion) (st : Lifecycle.State)
    (h : st.morboWeight = none) :
    optimizer.tr_state.TrustRegionState.is_morbo (.Turbo tr) =
      Result.ok (Lifecycle.isMorbo st) := by
  simp [optimizer.tr_state.TrustRegionState.is_morbo, Lifecycle.isMorbo, h]

/-- Extracted `TrustRegionState.is_morbo` on `Morbo` agrees with `Lifecycle.isMorbo`. -/
theorem morboState (m : morbo_trust_region.MorboTrustRegion) (st : Lifecycle.State)
    (w : Morbo.WeightState) (h : st.morboWeight = some w) :
    optimizer.tr_state.TrustRegionState.is_morbo (.Morbo m) =
      Result.ok (Lifecycle.isMorbo st) := by
  simp [optimizer.tr_state.TrustRegionState.is_morbo, Lifecycle.isMorbo, h]

/-! ## 3. `IncrementalIncumbentTracker::reset` Correspondence (`Incumbent.reset`) -/

/-- When the mutable slice iterator has reached the end of `per_metric_entries`,
`reset_loop` terminates in one step and applies the writeback continuation `back`. -/
theorem resetDone
    (iter : core.slice.iter.IterMut (alloc.vec.Vec (Usize × Extraction.Native.F64)))
    (back : core.slice.iter.IterMut (alloc.vec.Vec (Usize × Extraction.Native.F64)) →
      core.slice.iter.IterMut (alloc.vec.Vec (Usize × Extraction.Native.F64)))
    (h : ¬ (iter.i < iter.slice.val.length)) :
    incumbent_tracker.IncrementalIncumbentTracker.reset_loop iter back =
      Result.ok (back iter) := by
  unfold incumbent_tracker.IncrementalIncumbentTracker.reset_loop
  rw [loop.eq_1]
  simp [incumbent_tracker.IncrementalIncumbentTracker.reset_loop.body,
    core.slice.iter.IteratorIterMut.next, h, bind_ok]

/-- Extracted `IncrementalIncumbentTracker.reset` clears all index/entry vectors,
zeros `observation_count` and `max_initialized`, and resets `max_y` to IEEE-754
`negInf`, matching `Incumbent.reset`. -/
theorem resetEq
    (tr : incumbent_tracker.IncrementalIncumbentTracker)
    (st : Incumbent.State)
    (hEmpty : tr.per_metric_entries = alloc.vec.Vec.new _) :
    incumbent_tracker.IncrementalIncumbentTracker.reset tr =
      Result.ok
        { tr with
          observation_count := 0#usize
          all_indices := alloc.vec.Vec.new _
          max_y := Extraction.Native.F64.negInf
          max_indices := alloc.vec.Vec.new _
          max_initialized := (Incumbent.reset st).maxInitialized
          top_entries := alloc.vec.Vec.new _
          per_metric_entries := alloc.vec.Vec.new _ } ∧
      Extraction.Native.F64.negInf.bits = (Incumbent.reset st).maxY := by
  refine ⟨?_, rfl⟩
  unfold incumbent_tracker.IncrementalIncumbentTracker.reset
  rw [hEmpty]
  have hDone :
      incumbent_tracker.IncrementalIncumbentTracker.reset_loop
        { slice := (alloc.vec.Vec.new (alloc.vec.Vec (Usize × Extraction.Native.F64))).slice, i := 0 }
        (fun im => im) =
      Result.ok { slice := (alloc.vec.Vec.new (alloc.vec.Vec (Usize × Extraction.Native.F64))).slice, i := 0 } := by
    apply resetDone
    simp [alloc.vec.Vec.new, alloc.vec.Vec.from]
  simp [alloc.vec.opaque_external_impl_DefPathHash_Fingerprint_2597906387018489422_14690868890430580445___1.clear,
    core.f64.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_7831059766625668781__.NEG_INFINITY,
    MutAVec.Insts.CoreIterTraitsCollectIntoIteratorMutATIterMut.into_iter,
    Incumbent.reset, hDone, bind_ok]

/-! ## 4. `Optimizer::begin_local_restart` Correspondence (`Lifecycle.restartLocal`) -/

/-- Under MORBO trust-region state, `begin_local_restart` is a no-op returning
`Ok ()` and preserving `opt`, matching `Lifecycle.restartLocal`. -/
theorem restartMorbo
    (opt : optimizer.Optimizer) (m : morbo_trust_region.MorboTrustRegion)
    (hTr : opt.tr_state = .Morbo m) :
    optimizer.restart.Optimizer.begin_local_restart opt =
      Result.ok (core.result.Result.Ok (), opt) := by
  unfold optimizer.restart.Optimizer.begin_local_restart
  rw [hTr]
  simp [optimizer.tr_state.TrustRegionState.is_morbo]

/-- Under TuRBO trust-region state with no surrogate (`surrogate = none`),
`begin_local_restart` clears fallback observations, resets the incumbent tracker,
restarts the TuRBO trust region, increments `restart_generation`, and sets
`reinit_left` from `local_init_budget`, matching `Lifecycle.restartLocal`. -/
theorem restartNone
    (opt : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (nextGen : Usize)
    (hTr : opt.tr_state = .Turbo tr)
    (hSurr : opt.surrogate = none)
    (hMetric : opt.incumbent_tracker.per_metric_entries = alloc.vec.Vec.new _)
    (hGen : opt.restart_generation + 1#usize = Result.ok nextGen)
    (hBudget : opt.local_init_budget > 0#usize) :
    optimizer.restart.Optimizer.begin_local_restart opt =
      Result.ok (core.result.Result.Ok (),
        { opt with
          tr_state := .Turbo (restartTurbo tr)
          surrogate := none
          fallback_x := alloc.vec.Vec.new _
          fallback_y := alloc.vec.Vec.new _
          incumbent_idx := none
          incumbent_x_unit := none
          incumbent_y_scalar := none
          restart_generation := nextGen
          incumbent_tracker :=
            { opt.incumbent_tracker with
              observation_count := 0#usize
              all_indices := alloc.vec.Vec.new _
              max_y := Extraction.Native.F64.negInf
              max_indices := alloc.vec.Vec.new _
              max_initialized := false
              top_entries := alloc.vec.Vec.new _
              per_metric_entries := alloc.vec.Vec.new _ }
          reinit_left := some opt.local_init_budget }) := by
  have hReset := (resetEq opt.incumbent_tracker
    { m := 1, numMetrics := 1, mode := .all, observationCount := 0,
      allIndices := [], maxY := Bounds.negInf, maxIndices := [],
      maxInitialized := false, topEntries := [], perMetricEntries := [] } hMetric).1
  unfold optimizer.restart.Optimizer.begin_local_restart
  rw [hTr, hSurr]
  simp [optimizer.tr_state.TrustRegionState.is_morbo,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_mut,
    alloc.vec.opaque_external_impl_DefPathHash_Fingerprint_2597906387018489422_14690868890430580445___1.clear,
    optimizer.tr_state.TrustRegionState.restart_local,
    Incumbent.reset, hReset, hGen, hBudget, bind_ok]

/-- Under TuRBO trust-region state with a `Dyn Surrogate` whose `clear_observations`
vtable call succeeds, `begin_local_restart` updates the dynamic trait object
with the cleared surrogate value and performs the local restart. -/
theorem restartClear
    {S : Type} (inst : surrogate.Surrogate S) (val val' : S)
    (opt : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (nextGen : Usize)
    (hTr : opt.tr_state = .Turbo tr)
    (hSurr : opt.surrogate = some (Dyn.mk S inst val))
    (hClear : inst.clear_observations val = Result.ok (core.result.Result.Ok (), val'))
    (hMetric : opt.incumbent_tracker.per_metric_entries = alloc.vec.Vec.new _)
    (hGen : opt.restart_generation + 1#usize = Result.ok nextGen)
    (hBudget : opt.local_init_budget > 0#usize) :
    optimizer.restart.Optimizer.begin_local_restart opt =
      Result.ok (core.result.Result.Ok (),
        { opt with
          tr_state := .Turbo (restartTurbo tr)
          surrogate := some (Dyn.mk S inst val')
          fallback_x := alloc.vec.Vec.new _
          fallback_y := alloc.vec.Vec.new _
          incumbent_idx := none
          incumbent_x_unit := none
          incumbent_y_scalar := none
          restart_generation := nextGen
          incumbent_tracker :=
            { opt.incumbent_tracker with
              observation_count := 0#usize
              all_indices := alloc.vec.Vec.new _
              max_y := Extraction.Native.F64.negInf
              max_indices := alloc.vec.Vec.new _
              max_initialized := false
              top_entries := alloc.vec.Vec.new _
              per_metric_entries := alloc.vec.Vec.new _ }
          reinit_left := some opt.local_init_budget }) := by
  have hReset := (resetEq opt.incumbent_tracker
    { m := 1, numMetrics := 1, mode := .all, observationCount := 0,
      allIndices := [], maxY := Bounds.negInf, maxIndices := [],
      maxInitialized := false, topEntries := [], perMetricEntries := [] } hMetric).1
  unfold optimizer.restart.Optimizer.begin_local_restart
  rw [hTr, hSurr]
  simp [optimizer.tr_state.TrustRegionState.is_morbo,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_mut,
    hClear, core.result.Result.Insts.CoreOpsTry.branch,
    alloc.vec.opaque_external_impl_DefPathHash_Fingerprint_2597906387018489422_14690868890430580445___1.clear,
    optimizer.tr_state.TrustRegionState.restart_local,
    Incumbent.reset, hReset, hGen, hBudget, bind_ok]

/-- When `Surrogate::clear_observations` returns `Err err` via dynamic dispatch,
`begin_local_restart` aborts the restart, preserves the updated dynamic trait
object, and propagates `Err err`. -/
theorem restartFail
    {S : Type} (inst : surrogate.Surrogate S) (val val' : S)
    (err : error.ENNError)
    (opt : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (hTr : opt.tr_state = .Turbo tr)
    (hSurr : opt.surrogate = some (Dyn.mk S inst val))
    (hClear : inst.clear_observations val = Result.ok (core.result.Result.Err err, val')) :
    optimizer.restart.Optimizer.begin_local_restart opt =
      Result.ok (core.result.Result.Err err,
        { opt with surrogate := some (Dyn.mk S inst val') }) := by
  unfold optimizer.restart.Optimizer.begin_local_restart
  rw [hTr, hSurr]
  simp [optimizer.tr_state.TrustRegionState.is_morbo,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_mut,
    hClear, core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual,
    bind_ok]

/-! ## 5. `Strategy::ask` & `Optimizer::ask` Correspondence (`Lifecycle.ask`, `Optimizer.ask`) -/

/-- When `opt.reinit_left = some budget`, extracted `Strategy.ask` dispatches to
`strategy.ask_local_reinit` regardless of `strat`, matching `Lifecycle.initKind`
selecting `some st.localKind`. -/
theorem reinitAsk
    (strat : strategy.Strategy) (opt : optimizer.Optimizer)
    (arms budget : Usize) (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (st : Lifecycle.State)
    (hReinit : opt.reinit_left = some budget)
    (hSt : st.reinitLeft = some budget.val) :
    strategy.Strategy.ask strat opt arms tel rng =
      (do
        let (r, opt', rng') ← strategy.ask_local_reinit opt arms rng
        ok (r, opt', tel, rng')) ∧
    Lifecycle.initKind st = some st.localKind := by
  refine ⟨?_, ?_⟩
  · unfold strategy.Strategy.ask
    simp [optimizer.restart.Optimizer.reinit_left,
      core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
      hReinit]
  · simp [Lifecycle.initKind, hSt]

/-- When `opt.reinit_left = none` and `strat = .Init state`, extracted `Strategy.ask`
dispatches to `strategy.ask_init`, matching `Lifecycle.initKind st = some kind`. -/
theorem initAsk
    (state : strategy.InitStrategyState) (opt : optimizer.Optimizer)
    (arms : Usize) (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (st : Lifecycle.State) (kind : Opt.InitKind)
    (hReinit : opt.reinit_left = none)
    (hStReinit : st.reinitLeft = none)
    (hStStrat : st.strategy = .init kind state.completed.val state.num_init.val) :
    strategy.Strategy.ask (.Init state) opt arms tel rng =
      (do
        let (r, opt', rng') ← strategy.ask_init state opt arms rng
        ok (r, opt', tel, rng')) ∧
    Lifecycle.initKind st = some kind := by
  refine ⟨?_, ?_⟩
  · unfold strategy.Strategy.ask
    simp [optimizer.restart.Optimizer.reinit_left,
      core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
      hReinit]
  · simp [Lifecycle.initKind, hStReinit, hStStrat]

/-- When `opt.reinit_left = none` and `strat = .Turbo tss`, extracted `Strategy.ask`
dispatches to `strategy.ask_turbo`, matching `Lifecycle.initKind st = none`. -/
theorem turboAsk
    (tss : strategy.TurboStrategyState) (opt : optimizer.Optimizer)
    (arms : Usize) (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (st : Lifecycle.State)
    (hReinit : opt.reinit_left = none)
    (hStReinit : st.reinitLeft = none)
    (hStStrat : st.strategy = .turbo) :
    strategy.Strategy.ask (.Turbo tss) opt arms tel rng =
      strategy.ask_turbo opt arms tel rng ∧
    Lifecycle.initKind st = none := by
  refine ⟨?_, ?_⟩
  · unfold strategy.Strategy.ask
    simp [optimizer.restart.Optimizer.reinit_left,
      core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
      hReinit]
  · simp [Lifecycle.initKind, hStReinit, hStStrat]

/-- When `opt.reinit_left = none` and `strat = .Hybrid init tss true`, extracted
`Strategy.ask` dispatches to `strategy.ask_init_hybrid`, matching
`Lifecycle.initKind st = some kind`. -/
theorem hybridInit
    (init : strategy.InitStrategyState) (tss : strategy.TurboStrategyState)
    (opt : optimizer.Optimizer) (arms : Usize) (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (st : Lifecycle.State) (kind : Opt.InitKind)
    (hReinit : opt.reinit_left = none)
    (hStReinit : st.reinitLeft = none)
    (hStStrat : st.strategy = .hybrid kind init.completed.val init.num_init.val true) :
    strategy.Strategy.ask (.Hybrid init tss true) opt arms tel rng =
      (do
        let (r, opt', rng') ← strategy.ask_init_hybrid init opt arms rng
        ok (r, opt', tel, rng')) ∧
    Lifecycle.initKind st = some kind := by
  refine ⟨?_, ?_⟩
  · unfold strategy.Strategy.ask
    simp [optimizer.restart.Optimizer.reinit_left,
      core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
      hReinit]
  · simp [Lifecycle.initKind, hStReinit, hStStrat]

/-- When `opt.reinit_left = none` and `strat = .Hybrid init tss false`, extracted
`Strategy.ask` dispatches to `strategy.ask_turbo`, matching
`Lifecycle.initKind st = none`. -/
theorem hybridTurbo
    (init : strategy.InitStrategyState) (tss : strategy.TurboStrategyState)
    (opt : optimizer.Optimizer) (arms : Usize) (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (st : Lifecycle.State) (kind : Opt.InitKind)
    (hReinit : opt.reinit_left = none)
    (hStReinit : st.reinitLeft = none)
    (hStStrat : st.strategy = .hybrid kind init.completed.val init.num_init.val false) :
    strategy.Strategy.ask (.Hybrid init tss false) opt arms tel rng =
      strategy.ask_turbo opt arms tel rng ∧
    Lifecycle.initKind st = none := by
  refine ⟨?_, ?_⟩
  · unfold strategy.Strategy.ask
    simp [optimizer.restart.Optimizer.reinit_left,
      core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
      hReinit]
  · simp [Lifecycle.initKind, hStReinit, hStStrat]

/-- Wrapper `baux_extract.optimizer_ask` is definitionally `Optimizer.ask`. -/
theorem askWrap (opt : optimizer.Optimizer) (arms : Usize) :
    baux_extract.optimizer_ask opt arms =
      optimizer.Optimizer.ask opt arms := by
  rfl

/-- Requesting `0` arms returns `Err (InvalidParameter ...)` without mutating
the optimizer state, matching `Optimizer.ask` (`if numArms = 0 then (.error .zeroArms, st)`). -/
theorem askZero (opt : optimizer.Optimizer) :
    optimizer.Optimizer.ask opt 0#usize =
      Result.ok (core.result.Result.Err (.InvalidParameter "formatted error"), opt) := by
  unfold optimizer.Optimizer.ask
  simp [core.fmt.rt.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_1449683905130053284__.new_display,
    core.fmt.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_14732008022145090150___4.new,
    alloc.fmt.format, core.hint.must_use, bind_ok]

/-- When `arms ≠ 0#usize` and `tr_state` is TuRBO with `needsRestart = false`,
`reinit_left = none`, and `strategy = .Turbo ()`, `Optimizer.ask` casts `&mut self.rng`
to `&mut dyn RngCore`, dispatches `strategy.Strategy.ask` to `strategy.ask_turbo`,
writes back the concrete RNG via `Native.dynBack`, restores `self.strategy`,
and maps the unit-cube candidates to natural coordinates via `candidates.from_unit`. -/
theorem askTurbo
    (opt opt' : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (arms : Usize)
    (unitCand : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (tel' : optimizer.Telemetry) (rng' : rand.rngs.std.StdRng)
    (hArms : arms ≠ 0#usize)
    (hTr : opt.tr_state = .Turbo tr)
    (hNoRestart : tr.needsRestart = false)
    (hReinit : opt.reinit_left = none)
    (hStrat : opt.strategy = .Turbo ())
    (hTurbo :
      let z := Extraction.Native.F64.literal false 0 1
      let defTel : optimizer.Telemetry :=
        { dt_fit := z, dt_gen := z, dt_sel := z, dt_tell := z, num_candidates := 0#usize }
      strategy.ask_turbo
        { opt with strategy := .Turbo (), telemetry := defTel }
        arms opt.telemetry
        (Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore opt.rng) =
      Result.ok (core.result.Result.Ok unitCand, opt', tel',
        Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore rng'))
    (hSurr : opt'.surrogate = none) :
    let natRows := unitCand.entries.map (fun row =>
      _root_.Array.ofFn (n := opt'.bounds.entries.size) fun j =>
        let bj := opt'.bounds.entries[j]
        fromUnitWord (row.getD j.1 0#64) (bj.getD 0 0#64) (bj.getD 1 0#64))
    let z := Extraction.Native.F64.literal false 0 1
    optimizer.Optimizer.ask opt arms =
      Result.ok (core.result.Result.Ok
        { rows := unitCand.rows, cols := unitCand.cols, entries := natRows },
        { opt' with
          strategy := opt.strategy
          telemetry := { tel' with dt_gen := z }
          rng := rng' }) := by
  intro natRows z
  dsimp only at hTurbo
  rw [hTr, hReinit] at hTurbo
  unfold optimizer.Optimizer.ask
  rw [ite_eq_right_iff.mpr (fun h => False.elim (hArms h)), hTr, hStrat]
  simp [optimizer.tr_state.TrustRegionState.is_morbo,
    optimizer.tr_state.TrustRegionState.needs_restart, hNoRestart,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.now,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.elapsed,
    core.time.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_10380918174615376333__.as_secs_f64,
    rand.rngs.std.StdRng.Insts.CoreCloneClone.clone,
    strategy.Strategy.turbo, core.mem.replace, core.mem.take,
    optimizer.Telemetry.Insts.CoreDefaultDefault.default,
    strategy.Strategy.ask, optimizer.restart.Optimizer.reinit_left, hReinit,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hTurbo,
    core.result.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_12597149523846346601__.is_ok,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hSurr,
    core.result.Result.Insts.CoreOpsTry.branch,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___c23948124e078a22,
    candidates.from_unit, Extraction.Native.dyn_mk, natRows, z]

/-- When `opt'.surrogate = some (Dyn.mk S inst val)` and `inst.schedule_background_flush val`
succeeds, `Optimizer.ask` dispatches `schedule_background_flush` through the
dynamic trait vtable before returning natural-coordinate candidates. -/
theorem askFlush
    {S : Type} (inst : surrogate.Surrogate S) (val : S)
    (opt opt' : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (arms : Usize)
    (unitCand : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (tel' : optimizer.Telemetry) (rng' : rand.rngs.std.StdRng)
    (hArms : arms ≠ 0#usize)
    (hTr : opt.tr_state = .Turbo tr)
    (hNoRestart : tr.needsRestart = false)
    (hReinit : opt.reinit_left = none)
    (hStrat : opt.strategy = .Turbo ())
    (hTurbo :
      let z := Extraction.Native.F64.literal false 0 1
      let defTel : optimizer.Telemetry :=
        { dt_fit := z, dt_gen := z, dt_sel := z, dt_tell := z, num_candidates := 0#usize }
      strategy.ask_turbo
        { opt with strategy := .Turbo (), telemetry := defTel }
        arms opt.telemetry
        (Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore opt.rng) =
      Result.ok (core.result.Result.Ok unitCand, opt', tel',
        Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore rng'))
    (hSurr : opt'.surrogate = some (Dyn.mk S inst val))
    (hFlush : inst.schedule_background_flush val = Result.ok (core.result.Result.Ok ())) :
    let natRows := unitCand.entries.map (fun row =>
      _root_.Array.ofFn (n := opt'.bounds.entries.size) fun j =>
        let bj := opt'.bounds.entries[j]
        fromUnitWord (row.getD j.1 0#64) (bj.getD 0 0#64) (bj.getD 1 0#64))
    let z := Extraction.Native.F64.literal false 0 1
    optimizer.Optimizer.ask opt arms =
      Result.ok (core.result.Result.Ok
        { rows := unitCand.rows, cols := unitCand.cols, entries := natRows },
        { opt' with
          strategy := opt.strategy
          telemetry := { tel' with dt_gen := z }
          rng := rng' }) := by
  intro natRows z
  dsimp only at hTurbo
  rw [hTr, hReinit] at hTurbo
  unfold optimizer.Optimizer.ask
  rw [ite_eq_right_iff.mpr (fun h => False.elim (hArms h)), hTr, hStrat]
  simp [optimizer.tr_state.TrustRegionState.is_morbo,
    optimizer.tr_state.TrustRegionState.needs_restart, hNoRestart,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.now,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.elapsed,
    core.time.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_10380918174615376333__.as_secs_f64,
    rand.rngs.std.StdRng.Insts.CoreCloneClone.clone,
    strategy.Strategy.turbo, core.mem.replace, core.mem.take,
    optimizer.Telemetry.Insts.CoreDefaultDefault.default,
    strategy.Strategy.ask, optimizer.restart.Optimizer.reinit_left, hReinit,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hTurbo,
    core.result.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_12597149523846346601__.is_ok,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hSurr, hFlush,
    core.result.Result.Insts.CoreOpsTry.branch,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___c23948124e078a22,
    candidates.from_unit, Extraction.Native.dyn_mk, natRows, z]

/-! ## 6. `Strategy::tell` & `Optimizer::tell` Correspondence (`Tell.tell`) -/

/-- When `opt.reinit_left = some budget`, extracted `Strategy.tell` dispatches to
`strategy.tell_local_reinit` and preserves both `strat` and `tel`. -/
theorem reinitTell
    (strat : strategy.Strategy) (opt : optimizer.Optimizer) (budget : Usize)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (tel : optimizer.Telemetry)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hReinit : opt.reinit_left = some budget) :
    strategy.Strategy.tell strat opt x y yvar tel rng =
      (do
        let (r, opt', rng') ← strategy.tell_local_reinit opt x y yvar rng
        ok (r, strat, opt', tel, rng')) := by
  unfold strategy.Strategy.tell
  simp [optimizer.restart.Optimizer.reinit_left,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hReinit]

/-- When `opt.reinit_left = none` and `strat = .Hybrid init tss true`, if
`strategy.tell_init` succeeds with `init1.completed >= init1.num_init`,
`Strategy.tell` transitions `in_init` from `true` to `false` (`.Hybrid init1 tss false`). -/
theorem hybridDone
    (init init1 : strategy.InitStrategyState) (tss : strategy.TurboStrategyState)
    (opt opt' : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (tel : optimizer.Telemetry)
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hReinit : opt.reinit_left = none)
    (hInit : strategy.tell_init init opt x y yvar rng =
      Result.ok (core.result.Result.Ok (), init1, opt', rng'))
    (hDone : init1.completed >= init1.num_init) :
    strategy.Strategy.tell (.Hybrid init tss true) opt x y yvar tel rng =
      Result.ok (core.result.Result.Ok (), .Hybrid init1 tss false, opt', tel, rng') := by
  unfold strategy.Strategy.tell
  simp [optimizer.restart.Optimizer.reinit_left,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hReinit, hInit, core.result.Result.Insts.CoreOpsTry.branch]
  exact hDone

/-- When `opt.reinit_left = none` and `strat = .Hybrid init tss true`, if
`strategy.tell_init` succeeds with `¬ (init1.completed >= init1.num_init)`,
`Strategy.tell` remains in the hybrid init phase (`.Hybrid init1 tss true`). -/
theorem hybridStay
    (init init1 : strategy.InitStrategyState) (tss : strategy.TurboStrategyState)
    (opt opt' : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (tel : optimizer.Telemetry)
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hReinit : opt.reinit_left = none)
    (hInit : strategy.tell_init init opt x y yvar rng =
      Result.ok (core.result.Result.Ok (), init1, opt', rng'))
    (hNotDone : ¬ (init1.completed >= init1.num_init)) :
    strategy.Strategy.tell (.Hybrid init tss true) opt x y yvar tel rng =
      Result.ok (core.result.Result.Ok (), .Hybrid init1 tss true, opt', tel, rng') := by
  unfold strategy.Strategy.tell
  simp [optimizer.restart.Optimizer.reinit_left,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hReinit, hInit, core.result.Result.Insts.CoreOpsTry.branch]
  simp at hNotDone
  exact hNotDone

/-- Wrapper `baux_extract.optimizer_tell` is definitionally `Optimizer.tell`. -/
theorem tellWrap
    (opt : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)) :
    baux_extract.optimizer_tell opt x y yvar =
      optimizer.Optimizer.tell opt x y yvar := by
  rfl

/-- An empty batch (`x.rows = 0#usize` and `y.rows = 0#usize` with matching
columns) returns `Ok ()` immediately without mutating `yvar_on_tell` or state,
matching `Tell.emptyInert`. -/
theorem tellEmpty
    (opt : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (hCols : x.cols = opt.num_dim)
    (hRowsX : x.rows = 0#usize)
    (hRowsY : y.rows = 0#usize) :
    optimizer.Optimizer.tell opt x y yvar =
      Result.ok (core.result.Result.Ok (), opt) := by
  unfold optimizer.Optimizer.tell
  simp [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.nrows,
    hCols, hRowsX, hRowsY, bind_ok]

/-- Passing inconsistent `yvar` presence after `yvar_on_tell = some expected`
has been latched returns `Err (InvalidParameter ...)` without mutating state,
matching `Tell.tell` (`.error (.yvarConsistency expected)`). -/
theorem tellYvar
    (opt : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (expected : Bool)
    (hCols : x.cols = opt.num_dim)
    (hRows : y.rows = x.rows)
    (hNonEmpty : x.rows ≠ 0#usize)
    (hExpected : opt.yvar_on_tell = some expected)
    (hMismatch : (expected != yvar.isSome) = true) :
    optimizer.Optimizer.tell opt x y yvar =
      Result.ok (core.result.Result.Err (.InvalidParameter "formatted error"), opt) := by
  unfold optimizer.Optimizer.tell
  simp only [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.nrows,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hCols, hRows, hNonEmpty, hExpected, hMismatch, bne_self_eq_false,
    Bool.false_eq_true, ite_false, ite_true, bind_ok]
  cases expected <;>
    simp [core.fmt.rt.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_1449683905130053284__.new_display,
      core.fmt.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_14732008022145090150___4.new,
      alloc.fmt.format, core.hint.must_use, bind_ok]

/-- On the first non-empty `tell` (`yvar_on_tell = none`) with `surrogate = none`,
`Optimizer.tell` latches `yvar_on_tell := some yvar.isSome`, maps `x` to unit
coordinates via `candidates.to_unit`, invokes `strategy.Strategy.tell` with the
dynamically-cast `&mut dyn RngCore`, and writes back the concrete RNG via `dynBack`. -/
theorem tellNone
    (opt opt' : optimizer.Optimizer) (tel' : optimizer.Telemetry)
    (rng' : rand.rngs.std.StdRng)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (hCols : x.cols = opt.num_dim)
    (hRows : y.rows = x.rows)
    (hNonEmpty : x.rows ≠ 0#usize)
    (hFirst : opt.yvar_on_tell = none)
    (hSurr : opt.surrogate = none)
    (hReinit : opt.reinit_left = none)
    (hStrat : opt.strategy = .Turbo ())
    (hTurbo :
      let z := Extraction.Native.F64.literal false 0 1
      let defTel : optimizer.Telemetry :=
        { dt_fit := z, dt_gen := z, dt_sel := z, dt_tell := z, num_candidates := 0#usize }
      let unitRows := x.entries.map (fun row =>
        _root_.Array.ofFn (n := opt.bounds.entries.size) fun j =>
          let bj := opt.bounds.entries[j]
          toUnitWord (row.getD j.1 0#64) (bj.getD 0 0#64) (bj.getD 1 0#64))
      strategy.tell_turbo
        { opt with strategy := .Turbo (), telemetry := defTel, yvar_on_tell := some yvar.isSome }
        { rows := x.rows, cols := x.cols, entries := unitRows }
        y yvar opt.telemetry
        (Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore opt.rng) =
      Result.ok (core.result.Result.Ok (), opt', tel',
        Dyn.mk _ rand.rngs.std.StdRng.Insts.Rand_coreRngCore rng'))
    (hSurr' : opt'.surrogate = none) :
    let z := Extraction.Native.F64.literal false 0 1
    optimizer.Optimizer.tell opt x y yvar =
      Result.ok (core.result.Result.Ok (),
        { opt' with
          strategy := .Turbo ()
          telemetry := { tel' with dt_tell := z }
          rng := rng' }) := by
  intro z
  dsimp only at hTurbo
  rw [hCols, hSurr, hReinit] at hTurbo
  simp only [Array.getD_eq_getD_getElem?, Fin.getElem_fin] at hTurbo
  unfold optimizer.Optimizer.tell
  rw [hStrat]
  simp [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.nrows,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.is_some,
    hCols, hRows, hNonEmpty, hFirst, hSurr,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___c23948124e078a22,
    candidates.to_unit,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.now,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.elapsed,
    core.time.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_10380918174615376333__.as_secs_f64,
    rand.rngs.std.StdRng.Insts.CoreCloneClone.clone,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    strategy.Strategy.turbo, core.mem.replace, core.mem.take,
    optimizer.Telemetry.Insts.CoreDefaultDefault.default,
    strategy.Strategy.tell, optimizer.restart.Optimizer.reinit_left, hReinit,
    hTurbo, hSurr',
    core.result.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_12597149523846346601__.is_ok,
    Extraction.Native.dyn_mk, z]

/-! ## 7. `Optimizer::posterior_mu` Correspondence (`Predict.posteriorMu`) -/

/-- Wrapper `baux_extract.optimizer_mu` is definitionally `Optimizer.posterior_mu`. -/
theorem muWrap
    (opt : optimizer.Optimizer)
    (x : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64) :
    baux_extract.optimizer_mu opt x =
      optimizer.Optimizer.posterior_mu opt x := by
  rfl

/-- Calling `posterior_mu` when `surrogate = none` invokes the extracted closure
`posterior_mu.closure` via `Option::ok_or_else` and returns
`Err (InvalidParameter "No surrogate")`, matching `Predict.noModel`. -/
theorem muNone
    (opt : optimizer.Optimizer)
    (x : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (hCols : x.cols = opt.num_dim)
    (hSurr : opt.surrogate = none) :
    optimizer.Optimizer.posterior_mu opt x =
      Result.ok (core.result.Result.Err (.InvalidParameter "No surrogate")) := by
  unfold optimizer.Optimizer.posterior_mu
  simp [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    hCols, hSurr,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.ok_or_else,
    optimizer.Optimizer.posterior_mu.closure.Insts.CoreOpsFunctionFnOnceTupleENNError.call_once,
    alloc.string.ToString.Blanket.to_string,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual,
    bind_ok]

/-- Calling `posterior_mu` when `surrogate = some (Dyn.mk S inst val)` transforms
`x` to unit coordinates via `candidates.to_unit`, dispatches `inst.predict val`
through the dynamic trait vtable, and extracts `pred.mu`, matching
`Predict.posteriorMu`. -/
theorem muPredict
    {S : Type} (inst : surrogate.Surrogate S) (val : S)
    (pred : surrogate.SurrogatePrediction)
    (opt : optimizer.Optimizer)
    (x : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (hCols : x.cols = opt.num_dim)
    (hSurr : opt.surrogate = some (Dyn.mk S inst val))
    (hPred : ∀ u, inst.predict val u = Result.ok (core.result.Result.Ok pred)) :
    optimizer.Optimizer.posterior_mu opt x =
      Result.ok (core.result.Result.Ok pred.mu) := by
  unfold optimizer.Optimizer.posterior_mu
  simp [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    hCols, hSurr,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.ok_or_else,
    core.result.Result.Insts.CoreOpsTry.branch,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___c23948124e078a22,
    candidates.to_unit, hPred, bind_ok]

/-- When `Surrogate::predict` returns `Err err` via dynamic dispatch,
`posterior_mu` propagates `Err err` via `FromResidual::from_residual`. -/
theorem muFail
    {S : Type} (inst : surrogate.Surrogate S) (val : S)
    (err : error.ENNError)
    (opt : optimizer.Optimizer)
    (x : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (hCols : x.cols = opt.num_dim)
    (hSurr : opt.surrogate = some (Dyn.mk S inst val))
    (hPred : ∀ u, inst.predict val u = Result.ok (core.result.Result.Err err)) :
    optimizer.Optimizer.posterior_mu opt x =
      Result.ok (core.result.Result.Err err) := by
  unfold optimizer.Optimizer.posterior_mu
  simp [ndarray.impl_2d.opaque_external_impl_DefPathHash_Fingerprint_4096256263176452413_10435391855463334347__.ncols,
    hCols, hSurr,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.ok_or_else,
    core.result.Result.Insts.CoreOpsTry.branch,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___c23948124e078a22,
    candidates.to_unit, hPred,
    core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual,
    bind_ok]

/-! ## 8. Extracted Strategy Phase & Restart Helpers (`consume_reinit`, `init_plan`, `init_progress`, `ask_local_reinit`, `tell_local_reinit`, `tell_init`, `draw_turbo_candidates`, `morbo_sync_ranges_from_obs`, `reseed_sobol`, `ask_turbo`, `tell_turbo`) -/

/-- `Optimizer.consume_reinit` when `reinit_left = some left` and `left.saturating_sub n = 0`
clears `reinit_left` to `none`, matching `Lifecycle.consumeReinit`. -/
theorem consumeZero
    (opt : optimizer.Optimizer) (left n : Usize)
    (hLeft : opt.reinit_left = some left)
    (hSub : core.num.Usize.saturating_sub left n = 0#usize) :
    optimizer.restart.Optimizer.consume_reinit opt n =
      Result.ok { opt with reinit_left := none } := by
  unfold optimizer.restart.Optimizer.consume_reinit
  simp [hLeft,
    core.num.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_18170079207194785758___11.saturating_sub,
    hSub]

/-- `Optimizer.consume_reinit` when `reinit_left = some left` and `left.saturating_sub n = rem ≠ 0`
updates `reinit_left` to `some rem`, matching `Lifecycle.consumeReinit`. -/
theorem consumeKeep
    (opt : optimizer.Optimizer) (left n rem : Usize)
    (hLeft : opt.reinit_left = some left)
    (hSub : core.num.Usize.saturating_sub left n = rem)
    (hRem : rem ≠ 0#usize) :
    optimizer.restart.Optimizer.consume_reinit opt n =
      Result.ok { opt with reinit_left := some rem } := by
  unfold optimizer.restart.Optimizer.consume_reinit
  simp [hLeft,
    core.num.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_18170079207194785758___11.saturating_sub,
    hSub, hRem]

/-- `Strategy.init_plan` and `Strategy.init_progress` agree across `Init`, `Turbo`,
and `Hybrid` strategy states. -/
theorem planMatch (s : strategy.InitStrategyState) :
    strategy.Strategy.init_plan (.Init s) = Result.ok (s.num_init, s.strategy_type) ∧
    strategy.Strategy.init_plan (.Turbo ()) = Result.ok (0#usize, .LHD) ∧
    strategy.Strategy.init_plan (.Hybrid s () true) = Result.ok (s.num_init, s.strategy_type) ∧
    strategy.Strategy.init_progress (.Init s) = Result.ok (some (s.completed, s.num_init)) ∧
    strategy.Strategy.init_progress (.Turbo ()) = Result.ok none ∧
    strategy.Strategy.init_progress (.Hybrid s () true) = Result.ok (some (s.completed, s.num_init)) ∧
    strategy.Strategy.init_progress (.Hybrid s () false) = Result.ok none := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `strategy.ask_local_reinit` clamps the batch size to `n = max (min num_arms left) 1`,
constructs a fresh `InitStrategyState` with `opt.local_init_kind`, and delegates
to `strategy.ask_init`. -/
theorem reinitBatch
    (opt : optimizer.Optimizer) (arms left : Usize)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hLeft : opt.reinit_left = some left) :
    let n := core.cmp.impls.OrdUsize.max (core.cmp.impls.OrdUsize.min arms left) 1#usize
    strategy.ask_local_reinit opt arms rng =
      strategy.ask_init
        { strategy_type := opt.local_init_kind, num_init := n, completed := 0#usize }
        opt n rng := by
  intro n
  unfold strategy.ask_local_reinit
  simp [optimizer.restart.Optimizer.reinit_left, hLeft,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.unwrap_or,
    optimizer.restart.Optimizer.local_init_kind,
    strategy.InitStrategyState.new, lift, bind_ok, n]

/-- `strategy.tell_local_reinit` invokes `strategy.tell_common` with `telemetry = none`
and then decrements `reinit_left` by `x.rows` via `Optimizer.consume_reinit`. -/
theorem reinitConsume
    (opt opt' : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (telOut : Option optimizer.Telemetry)
    (hCommon : strategy.tell_common opt x y yvar none rng =
      Result.ok (core.result.Result.Ok (), opt', telOut, rng'))
    (hExhaust : opt'.reinit_left = some x.rows) :
    strategy.tell_local_reinit opt x y yvar rng =
      Result.ok (core.result.Result.Ok (), { opt' with reinit_left := none }, rng') := by
  unfold strategy.tell_local_reinit
  have hSub : core.num.Usize.saturating_sub x.rows x.rows = 0#usize := by
    apply UScalar.eq_of_val_eq
    simp [core.num.Usize.saturating_sub, UScalar.saturating_sub]
    rfl
  have hCons := consumeZero opt' x.rows x.rows hExhaust hSub
  simp [hCommon, core.result.Result.Insts.CoreOpsTry.branch,
    ndarray.opaque_callsite.nrows_DefPathHash_Fingerprint_4096256263176452413_8599771642494443633___c9a192dd741bdd5a,
    hCons, bind_ok]

/-- `strategy.tell_init` advances `state.completed` by `x.rows` and delegates
observation ingestion and surrogate fitting to `strategy.tell_common`. -/
theorem initStep
    (state : strategy.InitStrategyState) (opt opt' : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (telOut : Option optimizer.Telemetry)
    (nextComp : Usize)
    (hAdd : state.completed + x.rows = Result.ok nextComp)
    (hCommon : strategy.tell_common opt x y yvar none rng =
      Result.ok (core.result.Result.Ok (), opt', telOut, rng')) :
    strategy.tell_init state opt x y yvar rng =
      Result.ok (core.result.Result.Ok (), { state with completed := nextComp }, opt', rng') := by
  unfold strategy.tell_init
  simp [ndarray.opaque_callsite.nrows_DefPathHash_Fingerprint_4096256263176452413_8599771642494443633___c9a192dd741bdd5a,
    hAdd, hCommon, bind_ok]

/-- When `raasp_fast = true` and `candidate_rv = .RAASP`, `strategy.draw_turbo_candidates`
constructs the first-class `rand.rng.Rng (Dyn RngCore)` dictionary and dispatches
to `candidates_fast.generate_tr_candidates_fast`. -/
theorem drawFast
    (opt : optimizer.Optimizer)
    (xCenter : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (lower1d upper1d : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (numCand : Usize)
    (rng : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hFast : opt.config.candidates.raasp_fast = true)
    (hRv : opt.config.candidates.candidate_rv = .RAASP) :
    let dynRngInst : rand_core.RngCore (Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn)) :=
      { next_u32 := fun d => Aeneas.Std.bind (d.inst.next_u32 d.value) (fun (r0, v') => ok (r0, { d with value := v' }))
        next_u64 := fun d => Aeneas.Std.bind (d.inst.next_u64 d.value) (fun (r0, v') => ok (r0, { d with value := v' }))
        fill_bytes := fun d x0 => Aeneas.Std.bind (d.inst.fill_bytes d.value x0) (fun (v', r1) => ok ({ d with value := v' }, r1))
        try_fill_bytes := fun d x0 => Aeneas.Std.bind (d.inst.try_fill_bytes d.value x0) (fun (r0, v', r2) => ok (r0, { d with value := v' }, r2)) }
    strategy.draw_turbo_candidates opt xCenter lower1d upper1d numCand rng =
      (do
        let (r, rng1) ←
          candidates_fast.generate_tr_candidates_fast (rand.rng.Rng.Blanket dynRngInst)
            xCenter lower1d upper1d numCand rng 20#usize
        ok (r, opt, rng1)) := by
  intro dynRngInst
  unfold strategy.draw_turbo_candidates
  simp [optimizer.Optimizer.impl.config, hFast, hRv,
    candidates.CandidateRV.Insts.CoreCmpPartialEqCandidateRV.eq, dynRngInst]

/-- Under MORBO trust region (`tr_state = .Morbo m`) with `raasp_fast = false`,
`candidate_rv = .Sobol`, and `sobol_engine = some engine`, `strategy.draw_turbo_candidates`
dispatches to `candidates.generate_sobol_masked` and writes back the updated `SobolEngine`. -/
theorem drawSobol
    (opt : optimizer.Optimizer) (m : morbo_trust_region.MorboTrustRegion)
    (engine engine' : candidates.SobolEngine)
    (xCenter : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (lower1d upper1d : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (candOut : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (numCand : Usize)
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (hFast : opt.config.candidates.raasp_fast = false)
    (hTr : opt.tr_state = .Morbo m)
    (hRv : opt.config.candidates.candidate_rv = .Sobol)
    (hEng : opt.sobol_engine = some engine)
    (hMask : ∀ inst,
      candidates.generate_sobol_masked inst xCenter lower1d upper1d numCand rng engine 20#usize =
        Result.ok (core.result.Result.Ok candOut, rng', engine')) :
    strategy.draw_turbo_candidates opt xCenter lower1d upper1d numCand rng =
      Result.ok (core.result.Result.Ok candOut,
        { opt with sobol_engine := some engine' }, rng') := by
  unfold strategy.draw_turbo_candidates
  simp [optimizer.Optimizer.impl.config, hFast,
    optimizer.Optimizer.trust_region, hTr,
    optimizer.tr_state.TrustRegionState.is_morbo, hRv,
    candidates.CandidateRV.Insts.CoreCmpPartialEqCandidateRV.eq,
    optimizer.Optimizer.sobol_engine_mut,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_mut,
    hEng, hMask, bind_ok]

/-- `strategy.morbo_sync_ranges_from_obs` updates MORBO objective ranges when
`tr_state = .Morbo m` and `y_obs` returns a non-empty observation matrix, and
is a no-op under TuRBO. -/
theorem morboSync
    (opt : optimizer.Optimizer) (tr : trust_region.TurboTrustRegion)
    (hTr : opt.tr_state = .Turbo tr) :
    strategy.morbo_sync_ranges_from_obs opt =
      Result.ok (core.result.Result.Ok (), opt) := by
  unfold strategy.morbo_sync_ranges_from_obs
  simp [optimizer.Optimizer.trust_region, hTr,
    optimizer.tr_state.TrustRegionState.is_morbo]

/-- `Optimizer.reseed_sobol` is a no-op when `sobol_engine = none`. -/
theorem reseedNone (opt : optimizer.Optimizer) (arms : Usize)
    (hNone : opt.sobol_engine = none) :
    optimizer.Optimizer.reseed_sobol opt arms =
      Result.ok (core.result.Result.Ok (), opt) := by
  unfold optimizer.Optimizer.reseed_sobol
  simp [core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hNone]

/-- When `strategy.tell_common` succeeds during `strategy.tell_turbo` but
`opt'.incumbent_y_scalar = none`, `tell_turbo` invokes `tell_turbo.closure` via
`Option::ok_or_else` and returns `Err (InvalidParameter "Missing incumbent y")`. -/
theorem tellMissing
    (opt opt' : optimizer.Optimizer)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (tel tel' : optimizer.Telemetry)
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (nObs : Usize)
    (hCommon : strategy.tell_common opt x y yvar (some tel) rng =
      Result.ok (core.result.Result.Ok (), opt', some tel', rng'))
    (hObs : optimizer.Optimizer.obs_count opt' = Result.ok nObs)
    (hNoInc : opt'.incumbent_y_scalar = none) :
    strategy.tell_turbo opt x y yvar tel rng =
      Result.ok (core.result.Result.Err (.InvalidParameter "No surrogate"), opt', tel', rng') := by
  unfold strategy.tell_turbo
  simp [hCommon, core.result.Result.Insts.CoreOpsTry.branch, hObs,
    optimizer.Optimizer.impl.incumbent_y_scalar,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hNoInc,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.ok_or_else,
    strategy.tell_turbo.closure.Insts.CoreOpsFunctionFnOnceTupleENNError.call_once,
    alloc.string.ToString.Blanket.to_string,
    core.result.Result.Insts.CoreOpsTry_traitFromResidualResult.from_residual,
    bind_ok]

/-- Full extracted `strategy.tell_turbo` execution under TuRBO trust region when
`turbo_prev_num_obs ≠ 0`: after `tell_common`, it sets `num_arms := x.rows`,
updates the trust region via `tell_update_new_batch`, checks `needs_restart`,
and returns the updated optimizer and telemetry. -/
theorem tellPhase
    (opt opt' : optimizer.Optimizer)
    (tr tr1 tr2 : trust_region.TurboTrustRegion)
    (x y : ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (yvar : Option (ndarray.ArrayBase (ndarray.ViewRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64))
    (incY : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (tel tel' : optimizer.Telemetry)
    (rng rng' : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (nObs prevObs : Usize)
    (hCommon : strategy.tell_common opt x y yvar (some tel) rng =
      Result.ok (core.result.Result.Ok (), opt', some tel', rng'))
    (hObs : optimizer.Optimizer.obs_count opt' = Result.ok nObs)
    (hInc : opt'.incumbent_y_scalar = some incY)
    (hTr : opt'.tr_state = .Turbo tr)
    (hArms : optimizer.tr_state.TrustRegionState.set_num_arms (.Turbo tr) x.rows =
      Result.ok (.Turbo tr1))
    (hPrev : optimizer.tr_state.TrustRegionState.turbo_prev_num_obs (.Turbo tr1) =
      Result.ok prevObs)
    (hPrevNe : prevObs ≠ 0#usize)
    (hUpdate : optimizer.tr_state.TrustRegionState.tell_update_new_batch (.Turbo tr1) y
      { rows := incY.rows, cols := incY.cols, entries := incY.entries } nObs =
      Result.ok (core.result.Result.Ok (), .Turbo tr2)) :
    strategy.tell_turbo opt x y yvar tel rng =
      Result.ok (core.result.Result.Ok (),
        { opt' with tr_state := .Turbo tr2 }, tel', rng') := by
  unfold strategy.tell_turbo
  simp [hCommon, core.result.Result.Insts.CoreOpsTry.branch, hObs,
    optimizer.Optimizer.impl.incumbent_y_scalar,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hInc,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.ok_or_else,
    ndarray.opaque_callsite.to_owned_DefPathHash_Fingerprint_4096256263176452413_11463498080844012744___d0478050df5c2fc6,
    optimizer.Optimizer.trust_region_mut,
    ndarray.opaque_callsite.nrows_DefPathHash_Fingerprint_4096256263176452413_8599771642494443633___abba9b3754c36058,
    hTr, hArms, optimizer.Optimizer.trust_region,
    optimizer.tr_state.TrustRegionState.is_morbo,
    hPrev, hPrevNe,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___2155ad556a83c3c2,
    hUpdate, optimizer.tr_state.TrustRegionState.needs_restart, bind_ok]

/-- Full extracted `strategy.ask_turbo` execution under TuRBO trust region with a
dynamic `Surrogate` trait object (`Dyn.mk S inst val`):
1. Calls `resample_on_propose` and `set_num_arms` on `tr_state`,
2. Dispatches `inst.lengthscales val` through the `Surrogate` vtable via `ask_turbo.closure_1`,
3. Computes 1D TR bounds via `compute_bounds_1d`,
4. Computes `num_candidates` via `CandidateConfig.num_candidates`,
5. Draws candidates via `draw_turbo_candidates`, and
6. Selects `arms` via `strategy.select.select_arms` and records `num_candidates` in `telemetry`. -/
theorem askPhase
    {S : Type} (inst : surrogate.Surrogate S) (val : S)
    (opt : optimizer.Optimizer) (tr tr1 tr2 : trust_region.TurboTrustRegion)
    (arms : Usize) (tel : optimizer.Telemetry)
    (rng rng1 rng2 rng3 : Dyn (fun (_dyn : Type) => rand_core.RngCore _dyn))
    (ls lower1d upper1d : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 1#usize)) Extraction.Native.F64)
    (candPool armsOut : ndarray.ArrayBase (ndarray.data_repr.OwnedRepr Extraction.Native.F64)
      (ndarray.dimension.dim.Dim (Aeneas.Std.Array Usize 2#usize)) Extraction.Native.F64)
    (hTr : opt.tr_state = .Turbo tr)
    (hResample : optimizer.tr_state.TrustRegionState.resample_on_propose (.Turbo tr) rng =
      Result.ok (.Turbo tr1, rng1))
    (hArms : optimizer.tr_state.TrustRegionState.set_num_arms (.Turbo tr1) arms =
      Result.ok (.Turbo tr2))
    (hIncX : opt.incumbent_x_unit = none)
    (hSurr : opt.surrogate = some (Dyn.mk S inst val))
    (hLs : inst.lengthscales val = Result.ok (some ls))
    (hBounds : ∀ center lsView,
      optimizer.tr_state.TrustRegionState.compute_bounds_1d (.Turbo tr2) center lsView =
        Result.ok (lower1d, upper1d))
    (hFast : opt.config.candidates.raasp_fast = true)
    (hRv : opt.config.candidates.candidate_rv = .RAASP)
    (hGenFast : ∀ rInst center nCand,
      candidates_fast.generate_tr_candidates_fast rInst center lower1d upper1d nCand rng1 20#usize =
        Result.ok (core.result.Result.Ok candPool, rng2))
    (hSelect : ∀ optIn poolView,
      strategy.select.select_arms optIn poolView arms rng2 =
        Result.ok (core.result.Result.Ok armsOut, rng3)) :
    ∃ nCand,
      config.CandidateConfig.num_candidates opt.config.candidates opt.num_dim arms = Result.ok nCand ∧
      strategy.ask_turbo opt arms tel rng =
        Result.ok (core.result.Result.Ok armsOut,
          { opt with tr_state := .Turbo tr2 },
          { tel with dt_sel := Extraction.Native.F64.literal false 0 1, num_candidates := nCand },
          rng3) := by
  obtain ⟨nCand, hNumCand⟩ : ∃ nCand,
      config.CandidateConfig.num_candidates opt.config.candidates opt.num_dim arms = Result.ok nCand := by
    unfold config.CandidateConfig.num_candidates
    simp [core.num.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_18170079207194785758___11.saturating_mul,
      core.num.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_18170079207194785758___11.saturating_add,
      lift, bind_ok]
  refine ⟨nCand, hNumCand, ?_⟩
  have hClone : config.CandidateConfig.Insts.CoreCloneClone.clone opt.config.candidates =
      Result.ok opt.config.candidates := by
    cases opt.config.candidates
    simp [config.CandidateConfig.Insts.CoreCloneClone.clone,
      core.clone.impls.CloneUsize.clone, core.clone.impls.CloneBool.clone,
      candidates.CandidateRV.Insts.CoreCloneClone.clone, lift, bind_ok]
  unfold strategy.ask_turbo
  simp [optimizer.Optimizer.trust_region_mut, hTr, hResample, hArms,
    optimizer.Optimizer.impl.num_dim,
    ndarray.opaque_callsite.from_elem_DefPathHash_Fingerprint_4096256263176452413_6558715655683973886___3817dd7a1a061a04,
    optimizer.Optimizer.impl.incumbent_x_unit,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.as_ref,
    hIncX,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.map,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.unwrap_or,
    optimizer.Optimizer.impl.surrogate, hSurr,
    optimizer.Optimizer.surrogate.closure.Insts.CoreOpsFunctionFnOnceTupleShared0BoxSharedP.call_once,
    core.option.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_6987516322565337525__.and_then,
    strategy.ask_turbo.closure_1.Insts.CoreOpsFunctionFnOnceTupleShared0POptionArrayBaseOwnedReprF64DimArrayUsize1F64.call_once,
    hLs,
    strategy.ask_turbo.closure_2.Insts.CoreOpsFunctionFnOnceTupleShared0ArrayBaseOwnedReprF64DimArrayUsize1F64ArrayBaseViewReprSharedF64DimArrayUsize1F64.call_once,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___2155ad556a83c3c2,
    optimizer.Optimizer.trust_region, hBounds,
    optimizer.Optimizer.impl.config, hClone, hNumCand,
    optimizer.tr_state.TrustRegionState.is_morbo,
    strategy.draw_turbo_candidates, hFast, hRv,
    candidates.CandidateRV.Insts.CoreCmpPartialEqCandidateRV.eq,
    hGenFast, core.result.Result.Insts.CoreOpsTry.branch,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.now,
    std.time.opaque_external_impl_DefPathHash_Fingerprint_6454413375510392809_1741903903668520123__.elapsed,
    core.time.opaque_external_impl_DefPathHash_Fingerprint_2386793481758271246_10380918174615376333__.as_secs_f64,
    ndarray.opaque_callsite.view_DefPathHash_Fingerprint_4096256263176452413_7910469139750098697___3d4e5a36ec71ed96,
    hSelect, bind_ok]

#print axioms unitWord
#print axioms fromWord
#print axioms morboTurbo
#print axioms morboState
#print axioms resetDone
#print axioms resetEq
#print axioms restartMorbo
#print axioms restartNone
#print axioms restartClear
#print axioms restartFail
#print axioms reinitAsk
#print axioms initAsk
#print axioms turboAsk
#print axioms hybridInit
#print axioms hybridTurbo
#print axioms askWrap
#print axioms askZero
#print axioms askTurbo
#print axioms askFlush
#print axioms reinitTell
#print axioms hybridDone
#print axioms hybridStay
#print axioms tellWrap
#print axioms tellEmpty
#print axioms tellYvar
#print axioms tellNone
#print axioms muWrap
#print axioms muNone
#print axioms muPredict
#print axioms muFail
#print axioms consumeZero
#print axioms consumeKeep
#print axioms planMatch
#print axioms reinitBatch
#print axioms reinitConsume
#print axioms initStep
#print axioms drawFast
#print axioms drawSobol
#print axioms morboSync
#print axioms reseedNone
#print axioms tellMissing
#print axioms tellPhase
#print axioms askPhase

end BauxResearch.Extraction.Optimizer
