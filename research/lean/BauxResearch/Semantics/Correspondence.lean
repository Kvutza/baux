import BauxResearch.Semantics.TurboApi

/-!
# Relational optimizer semantics

This file states correspondence between an upstream optimizer state and a
downstream implementation state.  RNG and surrogate representations may be
different.  The public lifecycle state must agree, while representation
relations connect the private states.
-/

namespace BauxResearch.Semantics.Correspondence

open TurboApi

noncomputable section

/-- Publicly observable optimizer state, excluding representation-private RNG
and surrogate values. -/
structure Visible (dim : Nat) where
  bounds : Fin dim → BoundPair
  tr : TrState
  strat : Strategy
  obsX : Array (Fin dim → ℝ)
  obsY : Array ℝ
  bestIdx : Option Nat
  bestX : Option (Fin dim → ℝ)
  bestY : Option ℝ
  restartGen : Nat
  telemetry : Telemetry
  yvarTell : Option Bool
  reinitLeft : Option Nat
  localBudget : Nat

/-- Erase private implementation representations from an optimizer state. -/
def observe {dim : Nat} {R S : Type} (st : OptState dim R S) : Visible dim where
  bounds := st.bounds
  tr := st.tr
  strat := st.strat
  obsX := st.obsX
  obsY := st.obsY
  bestIdx := st.bestIdx
  bestX := st.bestX
  bestY := st.bestY
  restartGen := st.restartGen
  telemetry := st.telemetry
  yvarTell := st.yvarTell
  reinitLeft := st.reinitLeft
  localBudget := st.localBudget

/-- Lift a relation through optional implementation state. -/
inductive OptRel {A B : Type} (rel : A → B → Prop) : Option A → Option B → Prop
  | none : OptRel rel none none
  | some {a b} (h : rel a b) : OptRel rel (some a) (some b)

/-- Relational invariant between upstream and downstream optimizer states. -/
structure StateRel {dim : Nat} {RU RL SU SL : Type}
    (rngRel : RU → RL → Prop) (surrRel : SU → SL → Prop)
    (up : OptState dim RU SU) (down : OptState dim RL SL) : Prop where
  visible : observe up = observe down
  rng : rngRel up.rng down.rng
  surrogate : OptRel surrRel up.surrogate down.surrogate

/-- The two surrogate clear operations preserve their representation relation. -/
def ClearSim {dim : Nat} {SU SL : Type}
    (surrRel : SU → SL → Prop)
    (up : SeamSurr SU dim) (down : SeamSurr SL dim) : Prop :=
  ∀ {a b}, surrRel a b → surrRel (up.clear a) (down.clear b)

/-- Candidate generators return identical points and related successor RNGs. -/
structure RngSim {RU RL : Type} (rngRel : RU → RL → Prop)
    (up : SeamRng RU) (down : SeamRng RL) : Prop where
  init : ∀ {ru rl}, rngRel ru rl → ∀ dim arms,
    let upOut := up.genInit ru dim arms
    let downOut := down.genInit rl dim arms
    upOut.2 = downOut.2 ∧ rngRel upOut.1 downOut.1
  turbo : ∀ {ru rl}, rngRel ru rl → ∀ dim arms lb ub center,
    let upOut := turboGen up ru dim arms lb ub center
    let downOut := turboGen down rl dim arms lb ub center
    upOut.2 = downOut.2 ∧ rngRel upOut.1 downOut.1

/-- Lift a representation relation through matching errors or successes. -/
inductive ExceptRel {E A B : Type} (rel : A → B → Prop) :
    Except E A → Except E B → Prop
  | error (err : E) : ExceptRel rel (.error err) (.error err)
  | ok {a b} (hrel : rel a b) : ExceptRel rel (.ok a) (.ok b)

/-- Surrogate append operations preserve the private representation relation. -/
def AppendSim {dim : Nat} {SU SL : Type} (surrRel : SU → SL → Prop)
    (up : SeamSurr SU dim) (down : SeamSurr SL dim) : Prop :=
  ∀ {su sl}, surrRel su sl → ∀ x y yv,
    ExceptRel surrRel (up.fitAppend su x y yv) (down.fitAppend sl x y yv)

/-- Related surrogates make the same acquisition selection decision. -/
def SelectSim {dim : Nat} {SU SL : Type} (surrRel : SU → SL → Prop)
    (up : SeamSurr SU dim) (down : SeamSurr SL dim) : Prop :=
  ∀ {su sl}, surrRel su sl → ∀ candidates arms,
    up.selectArms su candidates arms = down.selectArms sl candidates arms

/-- Error fallback and successful acquisition preserve selected candidates. -/
theorem choose_rel {dim : Nat} {SU SL : Type} {surrRel : SU → SL → Prop}
    (up : SeamSurr SU dim) (down : SeamSurr SL dim)
    (hsim : SelectSim surrRel up down)
    {su : Option SU} {sl : Option SL} (hrel : OptRel surrRel su sl)
    (candidates : Array (Fin dim → ℝ)) (arms : Nat) :
    chooseArms up su candidates arms = chooseArms down sl candidates arms := by
  cases hrel with
  | none => rfl
  | some hs => simp [chooseArms, hsim hs candidates arms]

/-- The optimizer's error-tolerant surrogate update preserves correspondence. -/
theorem append_rel {dim : Nat} {SU SL : Type} {surrRel : SU → SL → Prop}
    (up : SeamSurr SU dim) (down : SeamSurr SL dim)
    (hsim : AppendSim surrRel up down)
    {su : Option SU} {sl : Option SL} (hrel : OptRel surrRel su sl)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ)) :
    OptRel surrRel (appendSurr up su x y yv) (appendSurr down sl x y yv) := by
  cases hrel with
  | none => exact OptRel.none
  | some hs =>
      have hfit := hsim hs x y yv
      generalize hu : up.fitAppend _ x y yv = upOut at hfit
      generalize hd : down.fitAppend _ x y yv = downOut at hfit
      cases upOut with
      | error upErr =>
          cases downOut with
          | error downErr =>
              cases hfit
              simp [appendSurr, hu, hd]
              exact OptRel.some hs
          | ok downNext => cases hfit
      | ok upNext =>
          cases downOut with
          | error downErr => cases hfit
          | ok downNext =>
              cases hfit with
              | ok hnext =>
                  simp [appendSurr, hu, hd]
                  exact OptRel.some hnext

/-- A validated tell body preserves correspondence when surrogate appends do. -/
theorem proceed_tell {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hsim : AppendSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ)) :
    ExceptRel (StateRel rngRel surrRel)
      (proceedTell upSurr up x y yv) (proceedTell downSurr down x y yv) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  rcases hvis with ⟨hb, htr, hstrat, hx, hy, hidx, hbestx, hbesty,
    hgen, htel, hmode, hreinit, hbudget⟩
  have happ := append_rel upSurr downSurr hsim hrel.surrogate
    (x.map (fun point => fun i => toUnit (up.bounds i) (point i))) y yv
  rw [hb] at happ
  generalize hu : appendSurr upSurr up.surrogate
    (x.map (fun point => fun i => toUnit (down.bounds i) (point i))) y yv = upNext at happ
  generalize hd : appendSurr downSurr down.surrogate
    (x.map (fun point => fun i => toUnit (down.bounds i) (point i))) y yv = downNext at happ
  cases happ with
  | none =>
      simp [proceedTell, hb, htr, hstrat, hx, hy, hidx, hbestx, hbesty,
        hgen, htel, hmode, hreinit, hbudget, hu, hd]
      split
      · exact ExceptRel.error _
      · apply ExceptRel.ok
        exact ⟨rfl, hrel.rng, OptRel.none⟩
  | some hsurr =>
      simp [proceedTell, hb, htr, hstrat, hx, hy, hidx, hbestx, hbesty,
        hgen, htel, hmode, hreinit, hbudget, hu, hd]
      split
      · exact ExceptRel.error _
      · apply ExceptRel.ok
        exact ⟨rfl, hrel.rng, OptRel.some hsurr⟩
private theorem map_rel {A B : Type} {rel : A → B → Prop}
    {a : Option A} {b : Option B} {f : A → A} {g : B → B}
    (hsim : ∀ {x y}, rel x y → rel (f x) (g y))
    (hrel : OptRel rel a b) : OptRel rel (a.map f) (b.map g) := by
  cases hrel with
  | none => exact OptRel.none
  | some h => exact OptRel.some (hsim h)

/-- Results correspond when errors agree or successful observations agree and
their successor states remain related. -/
inductive ResultRel {E O U D : Type} (rel : U → D → Prop) :
    Except E (O × U) → Except E (O × D) → Prop
  | error (err : E) : ResultRel rel (.error err) (.error err)
  | ok {obs : O} {up : U} {down : D} (hrel : rel up down) :
      ResultRel rel (.ok (obs, up)) (.ok (obs, down))

/-- A synchronized local restart preserves optimizer correspondence. -/
theorem restart_preserves {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSeam : SeamSurr SU dim) (localSeam : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hsim : ClearSim surrRel upSeam localSeam)
    (hrel : StateRel rngRel surrRel up down) :
    StateRel rngRel surrRel (optRestart upSeam up) (optRestart localSeam down) := by
  cases hrel with
  | mk hvis hrng hsurr =>
      constructor
      · simp only [observe, optRestart, Visible.mk.injEq] at hvis ⊢
        aesop
      · exact hrng
      · exact map_rel hsim hsurr

/-- The validated initialization branch preserves correspondence for init,
hybrid-init, and restart-budget asks alike. -/
theorem init_ready {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hrel : StateRel rngRel surrRel up down)
    (arms : Nat)
    (hup : (isInit up.strat || up.reinitLeft.isSome) = true)
    (hdown : (isInit down.strat || down.reinitLeft.isSome) = true) :
    ResultRel (StateRel rngRel surrRel)
      (.ok (askReady upRng upSurr up arms) : Except EnnError _)
      (.ok (askReady downRng downSurr down arms) : Except EnnError _) := by
  have hgen := hrng.init hrel.rng dim arms
  generalize hu : upRng.genInit up.rng dim arms = upOut at hgen
  generalize hd : downRng.genInit down.rng dim arms = downOut at hgen
  rcases upOut with ⟨upNext, upCand⟩
  rcases downOut with ⟨downNext, downCand⟩
  simp only at hgen
  rcases hgen with ⟨hcand, hnext⟩
  subst downCand
  have hb : up.bounds = down.bounds := congrArg Visible.bounds hrel.visible
  simp [askReady, hup, hdown, hu, hd]
  rw [← hb]
  apply ResultRel.ok
  constructor
  · simp only [observe, Visible.mk.injEq]
    have hvis := hrel.visible
    simp only [observe, Visible.mk.injEq] at hvis
    aesop
  · exact hnext
  · exact hrel.surrogate

/-- The validated TuRBO branch preserves correspondence. -/
theorem turbo_ready {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hselect : SelectSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down)
    (arms : Nat)
    (hup : (isInit up.strat || up.reinitLeft.isSome) = false)
    (hdown : (isInit down.strat || down.reinitLeft.isSome) = false) :
    ResultRel (StateRel rngRel surrRel)
      (.ok (askReady upRng upSurr up arms) : Except EnnError _)
      (.ok (askReady downRng downSurr down arms) : Except EnnError _) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  rcases hvis with ⟨hb, htr, hstrat, hx, hy, hidx, hbestx, hbesty,
    hgen, htel, hmode, hreinit, hbudget⟩
  let center := turboCenter up.bestX
  let region := trBounds dim up.tr.length center
  have hr := hrng.turbo hrel.rng dim arms region.1 region.2 center
  generalize hu : turboGen upRng up.rng dim arms region.1 region.2 center = upOut at hr
  generalize hd : turboGen downRng down.rng dim arms region.1 region.2 center = downOut at hr
  rcases upOut with ⟨upNext, upRaw⟩
  rcases downOut with ⟨downNext, downRaw⟩
  simp only at hr
  rcases hr with ⟨hraw, hnext⟩
  subst downRaw
  dsimp [region, center] at hu hd
  rw [htr, hbestx] at hu hd
  have hchosen := choose_rel upSurr downSurr hselect hrel.surrogate upRaw arms
  generalize hc : chooseArms upSurr up.surrogate upRaw arms = chosen at hchosen
  have hdc := hchosen.symm
  simp [askReady, hup, hdown, htr, hbestx]
  rw [hu, hd, hc, hdc]
  rw [← hb]
  apply ResultRel.ok
  constructor
  · simp only [observe, hstrat, hx, hy, hidx, hbesty, hgen, htel, hmode,
      hreinit, hbudget]
  · exact hnext
  · exact hrel.surrogate

/-- Every validated ask body takes the same branch and preserves
correspondence. -/
theorem ask_ready {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hselect : SelectSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down) (arms : Nat) :
    ResultRel (StateRel rngRel surrRel)
      (.ok (askReady upRng upSurr up arms) : Except EnnError _)
      (.ok (askReady downRng downSurr down arms) : Except EnnError _) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  have hbranch : (isInit up.strat || up.reinitLeft.isSome) =
      (isInit down.strat || down.reinitLeft.isSome) := by aesop
  cases hup : isInit up.strat || up.reinitLeft.isSome with
  | false =>
      have hdown : (isInit down.strat || down.reinitLeft.isSome) = false := by
        rw [← hbranch, hup]
      exact turbo_ready upRng downRng upSurr downSurr hrng hselect hrel arms hup hdown
  | true =>
      have hdown : (isInit down.strat || down.reinitLeft.isSome) = true := by
        rw [← hbranch, hup]
      exact init_ready upRng downRng upSurr downSurr hrng hrel arms hup hdown

/-- Both implementations reject a zero-sized ask with the same error. -/
theorem ask_zero {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    (up : OptState dim RU SU) (down : OptState dim RL SL) :
    ResultRel (StateRel rngRel surrRel)
      (optAsk upRng upSurr up 0) (optAsk downRng downSurr down 0) := by
  simp only [optAsk, ↓reduceIte]
  exact ResultRel.error (EnnError.invalidParam "numArms must be positive")

/-- Complete ask correspondence, including validation, synchronized restart,
initialization, hybrid/reinitialization, TuRBO generation, and acquisition. -/
theorem ask_correspondence {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hclear : ClearSim surrRel upSurr downSurr)
    (hselect : SelectSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down) (arms : Nat) :
    ResultRel (StateRel rngRel surrRel)
      (optAsk upRng upSurr up arms) (optAsk downRng downSurr down arms) := by
  by_cases hzero : arms = 0
  · subst arms
    exact ask_zero upRng downRng upSurr downSurr up down
  have htr : up.tr = down.tr := congrArg Visible.tr hrel.visible
  have hneeds : trNeeds up.tr = trNeeds down.tr := congrArg trNeeds htr
  cases hup : trNeeds up.tr with
  | false =>
      have hdown : trNeeds down.tr = false := by rw [← hneeds, hup]
      simp only [optAsk, hzero, ↓reduceIte, hup, hdown]
      exact ask_ready upRng downRng upSurr downSurr hrng hselect hrel arms
  | true =>
      have hdown : trNeeds down.tr = true := by rw [← hneeds, hup]
      simp only [optAsk, hzero, ↓reduceIte, hup, hdown]
      exact ask_ready upRng downRng upSurr downSurr hrng hselect
        (restart_preserves upSurr downSurr hclear hrel) arms

/-- Initialization asks return identical natural-space candidates and preserve
the state relation. -/
theorem init_ask {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hrel : StateRel rngRel surrRel up down)
    (budget done arms : Nat)
    (hup : up.strat = Strategy.initMode budget done)
    (hdown : down.strat = Strategy.initMode budget done)
    (hupTr : trNeeds up.tr = false) (hdownTr : trNeeds down.tr = false)
    (hpos : arms ≠ 0) :
    ResultRel (StateRel rngRel surrRel)
      (optAsk upRng upSurr up arms)
      (optAsk downRng downSurr down arms) := by
  have hgen := hrng.init hrel.rng dim arms
  generalize hu : upRng.genInit up.rng dim arms = upOut at hgen
  generalize hd : downRng.genInit down.rng dim arms = downOut at hgen
  rcases upOut with ⟨upNext, upCand⟩
  rcases downOut with ⟨downNext, downCand⟩
  simp only at hgen
  rcases hgen with ⟨hcand, hnext⟩
  subst downCand
  have hb : up.bounds = down.bounds := congrArg Visible.bounds hrel.visible
  simp [optAsk, askReady, isInit, hpos, hupTr, hdownTr, hup, hdown, hu, hd]
  rw [← hb]
  apply ResultRel.ok
  constructor
  · simp only [observe, Visible.mk.injEq]
    have hvis := hrel.visible
    simp only [observe, Visible.mk.injEq] at hvis
    aesop
  · exact hnext
  · exact hrel.surrogate

/-- TuRBO asks preserve correspondence when generation and acquisition agree. -/
theorem turbo_ask {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrng : RngSim rngRel upRng downRng)
    (hselect : SelectSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down)
    (arms : Nat) (hpos : arms ≠ 0)
    (hup : up.strat = Strategy.turboMode)
    (hdown : down.strat = Strategy.turboMode)
    (hdownTr : trNeeds down.tr = false)
    (hupInit : up.reinitLeft = none) (hdownInit : down.reinitLeft = none) :
    ResultRel (StateRel rngRel surrRel)
      (optAsk upRng upSurr up arms)
      (optAsk downRng downSurr down arms) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  rcases hvis with ⟨hb, htr, hstrat, hx, hy, hidx, hbestx, hbesty,
    hgen, htel, hmode, hreinit, hbudget⟩
  let center := turboCenter up.bestX
  let region := trBounds dim up.tr.length center
  have hr := hrng.turbo hrel.rng dim arms region.1 region.2 center
  generalize hu : turboGen upRng up.rng dim arms region.1 region.2 center = upOut at hr
  generalize hd : turboGen downRng down.rng dim arms region.1 region.2 center = downOut at hr
  rcases upOut with ⟨upNext, upRaw⟩
  rcases downOut with ⟨downNext, downRaw⟩
  simp only at hr
  rcases hr with ⟨hraw, hnext⟩
  subst downRaw
  dsimp [region, center] at hu hd
  rw [htr, hbestx] at hu hd
  have hchosen := choose_rel upSurr downSurr hselect hrel.surrogate upRaw arms
  generalize hc : chooseArms upSurr up.surrogate upRaw arms = chosen at hchosen
  have hdc := hchosen.symm
  simp [optAsk, askReady, isInit, hpos, hup, hdown, hdownTr, hupInit, hdownInit,
    htr, hbestx]
  rw [hu, hd, hc, hdc]
  rw [← hb]
  apply ResultRel.ok
  constructor
  · simp only [observe, hx, hy, hidx, hbesty, hgen, htel, hmode, hbudget]
  · exact hnext
  · exact hrel.surrogate

/-- Both implementations reject the same row-count mismatch. -/
theorem tell_shape {dim : Nat} {RU RL SU SL : Type}
    (upSurr : SeamSurr SU dim) (localSurr : SeamSurr SL dim)
    (up : OptState dim RU SU) (down : OptState dim RL SL)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ))
    (hne : x.size ≠ y.size) :
    optTell upSurr up x y yv = Except.error (EnnError.invalidShape [x.size] [y.size]) ∧
      optTell localSurr down x y yv =
        Except.error (EnnError.invalidShape [x.size] [y.size]) := by
  exact ⟨shapeGate upSurr up x y yv hne, shapeGate localSurr down x y yv hne⟩

/-- Empty tells are observationally inert in both implementations. -/
theorem tell_empty {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSurr : SeamSurr SU dim) (localSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (_hrel : StateRel rngRel surrRel up down) :
    optTell upSurr up #[] #[] none = Except.ok up ∧
      optTell localSurr down #[] #[] none = Except.ok down := by
  constructor <;> simp [optTell]

/-- Lift a tell transition into the common observable-result shape. -/
def tellResult {dim : Nat} {R S : Type} (seam : SeamSurr S dim)
    (state : OptState dim R S) (x : Array (Fin dim → ℝ)) (y : Array ℝ)
    (yv : Option (Array ℝ)) : Except EnnError (Unit × OptState dim R S) :=
  (optTell seam state x y yv).map (fun next => ((), next))

private theorem unit_result {E U D : Type} {rel : U → D → Prop}
    {up : Except E U} {down : Except E D} (hrel : ExceptRel rel up down) :
    ResultRel rel (up.map (fun next => ((), next)))
      (down.map (fun next => ((), next))) := by
  cases hrel with
  | error err => exact ResultRel.error err
  | ok hnext => exact ResultRel.ok hnext

/-- Recording the same variance mode preserves every representation relation. -/
theorem yvarPreserved {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrel : StateRel rngRel surrRel up down) (mode : Bool) :
    StateRel rngRel surrRel { up with yvarTell := some mode }
      { down with yvarTell := some mode } := by
  constructor
  · have hvis := hrel.visible
    simp only [observe, Visible.mk.injEq] at hvis ⊢
    aesop
  · exact hrel.rng
  · exact hrel.surrogate

/-- Complete tell correspondence, including shape validation, empty batches,
first-call variance-mode establishment, mismatch rejection, trust-region
errors, observation updates, strategy progression, and surrogate append. -/
theorem tell_correspondence {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hsim : AppendSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ)) :
    ResultRel (StateRel rngRel surrRel)
      (tellResult upSurr up x y yv) (tellResult downSurr down x y yv) := by
  unfold tellResult
  by_cases hshape : x.size ≠ y.size
  · simp [optTell, hshape]
    exact ResultRel.error (EnnError.invalidShape [x.size] [y.size])
  · by_cases hzero : x.size = 0
    · have hyzero : y.size = 0 := by omega
      simp [optTell, hzero, hyzero]
      exact ResultRel.ok hrel
    · have hvis := hrel.visible
      simp only [observe, Visible.mk.injEq] at hvis
      have hmode : up.yvarTell = down.yvarTell := by aesop
      cases hup : up.yvarTell with
      | none =>
          have hdown : down.yvarTell = none := by rw [← hmode, hup]
          simp [optTell, hshape, hzero, hup, hdown]
          exact unit_result (proceed_tell upSurr downSurr hsim
            (yvarPreserved hrel yv.isSome) x y yv)
      | some expected =>
          have hdown : down.yvarTell = some expected := by rw [← hmode, hup]
          by_cases hmis : expected ≠ yv.isSome
          · simp [optTell, hshape, hzero, hup, hdown, hmis]
            exact ResultRel.error (EnnError.invalidParam "yvar mismatch on tell")
          · simp [optTell, hshape, hzero, hup, hdown, hmis]
            exact unit_result (proceed_tell upSurr downSurr hsim hrel x y yv)

/-- A nonempty tell in the established variance mode preserves correspondence. -/
theorem tell_success {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hsim : AppendSim surrRel upSurr downSurr)
    (hrel : StateRel rngRel surrRel up down)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ))
    (hsize : x.size = y.size) (hpos : 0 < x.size)
    (hmode : up.yvarTell = some yv.isSome) :
    ResultRel (StateRel rngRel surrRel)
      (tellResult upSurr up x y yv) (tellResult downSurr down x y yv) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  have hdown : down.yvarTell = some yv.isSome := by aesop
  have hcore := proceed_tell upSurr downSurr hsim hrel x y yv
  have hmatch : ¬x.size ≠ y.size := by simp [hsize]
  have hxzero : ¬x.size = 0 := ne_of_gt hpos
  unfold tellResult
  simp only [optTell, hmatch, ↓reduceIte, hxzero, hmode, hdown]
  simp only [ne_eq, not_true_eq_false, ↓reduceIte]
  exact unit_result hcore

/-- Related states reject the same variance-mode mismatch. -/
theorem yvar_mismatch {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrel : StateRel rngRel surrRel up down)
    (x : Array (Fin dim → ℝ)) (y : Array ℝ) (yv : Option (Array ℝ))
    (hsize : x.size = y.size) (hpos : 0 < x.size)
    (expected : Bool) (hmode : up.yvarTell = some expected)
    (hmis : expected ≠ yv.isSome) :
    ResultRel (StateRel rngRel surrRel)
      (tellResult upSurr up x y yv) (tellResult downSurr down x y yv) := by
  have hvis := hrel.visible
  simp only [observe, Visible.mk.injEq] at hvis
  have hdown : down.yvarTell = some expected := by aesop
  rw [tellResult, yvarGate upSurr up x y yv hsize hpos expected hmode hmis,
    tellResult, yvarGate downSurr down x y yv hsize hpos expected hdown hmis]
  exact ResultRel.error (EnnError.invalidParam "yvar mismatch on tell")

/-! ## Trace composition -/

/-- A pair of transition functions simulates every command from related states. -/
def StepSim {C E O U D : Type} (rel : U → D → Prop)
    (upStep : U → C → Except E (O × U))
    (downStep : D → C → Except E (O × D)) : Prop :=
  ∀ {up down}, rel up down → ∀ cmd, ResultRel rel (upStep up cmd) (downStep down cmd)

/-- Execute a command trace, collecting its observable outputs. -/
def runTrace {C E O S : Type} (step : S → C → Except E (O × S))
    (state : S) : List C → Except E (List O × S)
  | [] => .ok ([], state)
  | cmd :: rest => do
      let (obs, next) ← step state cmd
      let (tail, final) ← runTrace step next rest
      pure (obs :: tail, final)

private theorem map_result {E O U D : Type} {rel : U → D → Prop}
    {up : Except E (List O × U)} {down : Except E (List O × D)}
    (obs : O) (hrel : ResultRel rel up down) :
    ResultRel rel
      (up.map (fun pair => (obs :: pair.1, pair.2)))
      (down.map (fun pair => (obs :: pair.1, pair.2))) := by
  cases hrel with
  | error err => exact ResultRel.error err
  | ok hnext => exact ResultRel.ok hnext

/-- One-step relational simulation composes over every finite command trace. -/
theorem trace_preserves {C E O U D : Type} {rel : U → D → Prop}
    {upStep : U → C → Except E (O × U)}
    {downStep : D → C → Except E (O × D)}
    (hsim : StepSim rel upStep downStep)
    {up : U} {down : D} (hrel : rel up down) (commands : List C) :
    ResultRel rel (runTrace upStep up commands) (runTrace downStep down commands) := by
  induction commands generalizing up down with
  | nil => exact ResultRel.ok hrel
  | cons cmd rest ih =>
      simp only [runTrace]
      have hpair := hsim hrel cmd
      cases hup : upStep up cmd with
      | error upErr =>
          cases hdown : downStep down cmd with
          | error downErr =>
              rw [hup, hdown] at hpair
              cases hpair
              exact ResultRel.error upErr
          | ok downPair =>
              rw [hup, hdown] at hpair
              cases hpair
      | ok upPair =>
          cases hdown : downStep down cmd with
          | error downErr =>
              rw [hup, hdown] at hpair
              cases hpair
          | ok downPair =>
              rw [hup, hdown] at hpair
              cases hpair
              change ResultRel rel
                ((runTrace upStep _ rest).map (fun pair => (_ :: pair.1, pair.2)))
                ((runTrace downStep _ rest).map (fun pair => (_ :: pair.1, pair.2)))
              exact map_result _ (ih ‹rel _ _›)

private theorem map_observation {E O P U D : Type} {rel : U → D → Prop}
    {up : Except E (O × U)} {down : Except E (O × D)}
    (f : O → P) (hrel : ResultRel rel up down) :
    ResultRel rel
      (up.map (fun pair ↦ (f pair.1, pair.2)))
      (down.map (fun pair ↦ (f pair.1, pair.2))) := by
  cases hrel with
  | error err => exact ResultRel.error err
  | ok hnext => exact ResultRel.ok hnext

/-- Every public optimizer command preserves correspondence. -/
theorem stepSimulates {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    (hrng : RngSim rngRel upRng downRng)
    (hclear : ClearSim surrRel upSurr downSurr)
    (hselect : SelectSim surrRel upSurr downSurr)
    (happend : AppendSim surrRel upSurr downSurr) :
    StepSim (StateRel rngRel surrRel)
      (optimizerStep upRng upSurr) (optimizerStep downRng downSurr) := by
  intro up down hrel command
  cases command with
  | ask arms =>
      exact map_observation OptOutput.asked
        (ask_correspondence upRng downRng upSurr downSurr hrng hclear hselect hrel arms)
  | tell x y yvar =>
      have ht := tell_correspondence upSurr downSurr happend hrel x y yvar
      cases hu : optTell upSurr up x y yvar with
      | error upErr =>
          cases hd : optTell downSurr down x y yvar with
          | error downErr =>
              unfold tellResult at ht
              rw [hu, hd] at ht
              simp only [Except.map] at ht
              cases ht
              simp [optimizerStep, hu, hd]
              exact ResultRel.error upErr
          | ok downNext =>
              unfold tellResult at ht
              rw [hu, hd] at ht
              simp only [Except.map] at ht
              cases ht
      | ok upNext =>
          cases hd : optTell downSurr down x y yvar with
          | error downErr =>
              unfold tellResult at ht
              rw [hu, hd] at ht
              simp only [Except.map] at ht
              cases ht
          | ok downNext =>
              unfold tellResult at ht
              rw [hu, hd] at ht
              simp only [Except.map] at ht
              cases ht with
              | ok hnext =>
                  simp [optimizerStep, hu, hd]
                  exact ResultRel.ok hnext

/-- Arbitrary finite sequences of public asks and tells preserve all observable
outputs, matching errors, RNG/surrogate relations, and optimizer state. -/
theorem traceCorresponds {dim : Nat} {RU RL SU SL : Type}
    {rngRel : RU → RL → Prop} {surrRel : SU → SL → Prop}
    (upRng : SeamRng RU) (downRng : SeamRng RL)
    (upSurr : SeamSurr SU dim) (downSurr : SeamSurr SL dim)
    (hrng : RngSim rngRel upRng downRng)
    (hclear : ClearSim surrRel upSurr downSurr)
    (hselect : SelectSim surrRel upSurr downSurr)
    (happend : AppendSim surrRel upSurr downSurr)
    {up : OptState dim RU SU} {down : OptState dim RL SL}
    (hrel : StateRel rngRel surrRel up down)
    (commands : List (OptCommand dim)) :
    ResultRel (StateRel rngRel surrRel)
      (runTrace (optimizerStep upRng upSurr) up commands)
      (runTrace (optimizerStep downRng downSurr) down commands) :=
  trace_preserves
    (stepSimulates upRng downRng upSurr downSurr hrng hclear hselect happend)
    hrel commands

end

end BauxResearch.Semantics.Correspondence
