import BauxResearch.Generated.LocalCoord.Funs

/-!
Checked correspondence for the scalar binary64 kernel used by production
`to_unit`. The generated definition comes from the source-bound local Charon
slice; array traversal and surrogate dispatch remain separate obligations.
-/

namespace BauxResearch.Extraction.LocalCoord

open Aeneas Aeneas.Std Aeneas.Std.WP Result
open BauxResearch.Extraction.Native
open BauxResearch.Generated.LocalCoord

def spec (value lower upper : F64) : F64 :=
  let range := F64.sub upper lower
  if F64.gt range (F64.literal false 0 1) then
    F64.div (F64.sub value lower) range
  else
    F64.literal false 5 10

def fromSpec (value lower upper : F64) : F64 :=
  F64.add lower (F64.mul value (F64.sub upper lower))

theorem fromExtracted (value lower upper : F64) :
    coordinate_scalar.from_unit value lower upper =
      Result.ok (fromSpec value lower upper) := by
  simp [coordinate_scalar.from_unit, fromSpec, lift, bind_ok]

theorem extracted (value lower upper : F64) :
    coordinate_scalar.to_unit value lower upper = Result.ok (spec value lower upper) := by
  simp only [coordinate_scalar.to_unit, spec, lift, bind_ok]
  split <;> rfl

def shapeReady (rows columns : Usize) (values bounds : Slice F64) : Prop :=
  ∃ cells boundCells,
    Usize.checked_mul rows columns = some cells ∧
    Usize.checked_mul columns 2#usize = some boundCells ∧
    Slice.len values = cells ∧ Slice.len bounds = boundCells

theorem shapeAccepted (rows columns : Usize) (values bounds : Slice F64) :
    coordinate_flat.shape_ok rows columns values bounds = Result.ok true ↔
      shapeReady rows columns values bounds := by
  simp only [coordinate_flat.shape_ok, lift, bind_ok]
  cases cellsEq : Usize.checked_mul rows columns with
  | none => simp [cellsEq, shapeReady]
  | some cells =>
      cases boundsEq : Usize.checked_mul columns 2#usize with
      | none => simp [cellsEq, boundsEq, shapeReady]
      | some boundCells =>
          simp only [cellsEq, boundsEq, shapeReady, UScalar.eq_equiv, Slice.len_val]
          by_cases valuesEq : values.val.length = cells.val <;>
            by_cases boundsEq : bounds.val.length = boundCells.val <;>
            simp [valuesEq, boundsEq]

theorem shapeLengths (rows columns : Usize) (values bounds : Slice F64)
    (accepted : coordinate_flat.shape_ok rows columns values bounds = Result.ok true) :
    values.val.length = rows.val * columns.val ∧
      bounds.val.length = columns.val * 2 := by
  obtain ⟨cells, boundCells, cellsEq, boundsEq, valuesEq, boundsLen⟩ :=
    (shapeAccepted rows columns values bounds).mp accepted
  have cellsSpec := Usize.checked_mul_bv_spec rows columns
  have boundsSpec := Usize.checked_mul_bv_spec columns 2#usize
  simp only [cellsEq] at cellsSpec
  simp only [boundsEq] at boundsSpec
  have two : (2#usize).val = 2 := rfl
  constructor
  · simpa [Slice.len_val, cellsSpec.2.1] using congrArg UScalar.val valuesEq
  · simpa [Slice.len_val, boundsSpec.2.1, two] using congrArg UScalar.val boundsLen

def cellLow (columns : Usize) (bounds : Slice F64) (k : Nat) : F64 :=
  bounds.val[(k % columns.val) * 2]!

def cellUp (columns : Usize) (bounds : Slice F64) (k : Nat) : F64 :=
  bounds.val[(k % columns.val) * 2 + 1]!

def toCell (columns : Usize) (bounds : Slice F64) (k : Nat) (v : F64) : F64 :=
  spec v (cellLow columns bounds k) (cellUp columns bounds k)

def fromCell (columns : Usize) (bounds : Slice F64) (k : Nat) (v : F64) : F64 :=
  fromSpec v (cellLow columns bounds k) (cellUp columns bounds k)

theorem toPrefix (columns : Usize) (values bounds : Slice F64) (index : Usize)
    (columnPos : 0 < columns.val)
    (boundsLen : bounds.val.length = columns.val * 2)
    (hindex : index.val ≤ values.val.length) :
    coordinate_flat.to_unit_loop columns values bounds index ⦃ result =>
      result.val.length = values.val.length ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        k < index.val → result[k]! = values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        index.val ≤ k → result[k]! = toCell columns bounds k values[k]!) ⦄ := by
  unfold coordinate_flat.to_unit_loop
  apply loop.spec_decr_nat
    (measure := fun state => values.val.length - state.2.val)
    (inv := fun state =>
      state.1.val.length = values.val.length ∧
      index.val ≤ state.2.val ∧
      state.2.val ≤ values.val.length ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        k < index.val → state.1[k]! = values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        index.val ≤ k → k < state.2.val →
          state.1[k]! = toCell columns bounds k values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        state.2.val ≤ k → state.1[k]! = values[k]!))
  · intro state hInv
    rcases state with ⟨current, currentIndex⟩
    dsimp only at hInv
    rcases hInv with ⟨hlen, hle1, hle2, hpre, hmid, hsuf⟩
    simp only [coordinate_flat.to_unit_loop.body]
    by_cases inside : currentIndex.val < current.val.length
    · simp [UScalar.lt_equiv, inside]
      have hInsideVal : currentIndex.val < values.val.length := by omega
      have hmod : currentIndex.val % columns.val < columns.val :=
        Nat.mod_lt _ columnPos
      have hbounds1 : (currentIndex.val % columns.val) * 2 < bounds.val.length := by
        omega
      have hbounds2 : (currentIndex.val % columns.val) * 2 + 1 < bounds.val.length := by
        omega
      step*
      simp only [extracted]
      step*
      have hsLen : s.val.length = values.val.length := by
        subst s_post
        simpa [Slice.set_length] using hlen
      have hidx1 : index.val ≤ index1.val := by
        have := index1_post
        omega
      have hidx2 : index1.val ≤ values.val.length := by
        have := index1_post
        omega
      have hneq : ∀ (k : Nat), k < index.val → currentIndex.val ≠ k := by
        intro k _hk; omega
      have hPre' : ∀ (k : Nat) (_hk : k < values.val.length),
          k < index.val → s[k]! = values[k]! := by
        intro k hk hlt
        subst s_post
        rw [Slice.getElem!_Nat_set_ne current currentIndex k _ (hneq k hlt)]
        exact hpre k hk hlt
      have hidx_i1 : i1.val = (currentIndex.val % columns.val) * 2 := by
        have := i1_post; have := column_post; omega
      have hi1 : i1.val < bounds.val.length := by
        rw [hidx_i1]; exact hbounds1
      have hLow' : lower = cellLow columns bounds currentIndex.val := by
        have h1 : bounds.val[i1.val] = bounds.val[i1.val]! :=
          List.Inhabited_getElem_eq_getElem! bounds.val i1.val hi1
        rw [lower_post, h1]
        dsimp only [cellLow]
        exact congrArg (fun j => bounds.val[j]!) hidx_i1
      have hidx_i2 : i2.val = (currentIndex.val % columns.val) * 2 + 1 := by
        have := i2_post; have := i1_post; have := column_post; omega
      have hi2 : i2.val < bounds.val.length := by
        rw [hidx_i2]; exact hbounds2
      have hUp' : upper = cellUp columns bounds currentIndex.val := by
        have h2 : bounds.val[i2.val] = bounds.val[i2.val]! :=
          List.Inhabited_getElem_eq_getElem! bounds.val i2.val hi2
        rw [upper_post, h2]
        dsimp only [cellUp]
        exact congrArg (fun j => bounds.val[j]!) hidx_i2
      have hfl : fl = values[currentIndex.val]! := by
        have h1 : current.val[currentIndex.val] = current.val[currentIndex.val]! :=
          List.Inhabited_getElem_eq_getElem! current.val currentIndex.val (by omega)
        have h2 : current[currentIndex.val]! = current.val[currentIndex.val]! :=
          Slice.getElem!_Nat_eq current currentIndex.val
        rw [fl_post, h1, ← h2]
        exact hsuf currentIndex.val (by omega) le_rfl
      have hcellEq : spec fl lower upper =
          toCell columns bounds currentIndex.val values[currentIndex.val]! := by
        dsimp only [toCell]
        rw [hfl, hLow', hUp']
      have hMid' : ∀ (k : Nat) (_hk : k < values.val.length),
          index.val ≤ k → k < index1.val →
            s[k]! = toCell columns bounds k values[k]! := by
        intro k hk hge _hlt
        have _hle : k ≤ currentIndex.val := by
          have := index1_post; omega
        by_cases heq : k = currentIndex.val
        · subst s_post
          have hEq' : currentIndex.val = k := heq.symm
          have hLt' : k < current.length := by
            have : current.length = current.val.length := rfl
            omega
          have hset : (current.set currentIndex (spec fl lower upper))[k]! =
              spec fl lower upper :=
            Slice.getElem!_Nat_set_eq current currentIndex k (spec fl lower upper) ⟨hEq', hLt'⟩
          rw [hset, hcellEq, heq]
        · subst s_post
          have hne : currentIndex.val ≠ k := Ne.symm heq
          rw [Slice.getElem!_Nat_set_ne current currentIndex k _ hne]
          exact hmid k hk hge (by omega)
      have hSuf' : ∀ (k : Nat) (_hk : k < values.val.length),
          index1.val ≤ k → s[k]! = values[k]! := by
        intro k hk hge
        subst s_post
        have hne : currentIndex.val ≠ k := by
          have := index1_post; omega
        rw [Slice.getElem!_Nat_set_ne current currentIndex k _ hne]
        exact hsuf k hk (by have := index1_post; omega)
      refine ⟨hsLen, hidx1, hidx2, hPre', hMid', hSuf', ?_⟩
      have := index1_post
      omega
    · have outside : ¬currentIndex.val < values.val.length := by
        simpa [hlen] using inside
      simp [UScalar.lt_equiv, outside, hlen]
      exact ⟨hpre, fun k hk hge => hmid k hk hge (by omega)⟩
  · dsimp only
    refine ⟨rfl, le_rfl, hindex, fun _ _ _ => rfl, fun _ _ _ h => by omega, fun _ _ _ => rfl⟩

theorem toLoop (columns : Usize) (values bounds : Slice F64)
    (columnPos : 0 < columns.val)
    (boundsLen : bounds.val.length = columns.val * 2) :
    coordinate_flat.to_unit_loop columns values bounds 0#usize ⦃ result =>
      result.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        result[k]! = toCell columns bounds k values[k]! ⦄ := by
  have hzero : (0#usize).val ≤ values.val.length := Nat.zero_le _
  have hpre := toPrefix columns values bounds 0#usize columnPos boundsLen hzero
  apply spec_mono hpre
  intro result ⟨hlen, _, hpost⟩
  refine ⟨hlen, fun k hk => hpost k hk (Nat.zero_le _)⟩

theorem toNil (columns : Usize) (values bounds : Slice F64)
    (hlen : values.val.length = 0) :
    coordinate_flat.to_unit_loop columns values bounds 0#usize ⦃ result =>
      result.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        result[k]! = toCell columns bounds k values[k]! ⦄ := by
  unfold coordinate_flat.to_unit_loop
  apply loop.spec_decr_nat
    (measure := fun state => values.val.length - state.2.val)
    (inv := fun state => state.1.val.length = values.val.length)
  · intro state sameLength
    rcases state with ⟨current, currentIndex⟩
    dsimp only at sameLength
    simp only [coordinate_flat.to_unit_loop.body]
    by_cases inside : currentIndex.val < current.val.length
    · have : False := by
        have := sameLength
        have := hlen
        omega
      contradiction
    · have outside : ¬currentIndex.val < values.val.length := by
        simpa [sameLength] using inside
      simp [UScalar.lt_equiv, outside, sameLength]
      intro k hk
      have := hlen
      omega
  · rfl

theorem fromPrefix (columns : Usize) (values bounds : Slice F64) (index : Usize)
    (columnPos : 0 < columns.val)
    (boundsLen : bounds.val.length = columns.val * 2)
    (hindex : index.val ≤ values.val.length) :
    coordinate_flat.from_unit_loop columns values bounds index ⦃ result =>
      result.val.length = values.val.length ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        k < index.val → result[k]! = values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        index.val ≤ k → result[k]! = fromCell columns bounds k values[k]!) ⦄ := by
  unfold coordinate_flat.from_unit_loop
  apply loop.spec_decr_nat
    (measure := fun state => values.val.length - state.2.val)
    (inv := fun state =>
      state.1.val.length = values.val.length ∧
      index.val ≤ state.2.val ∧
      state.2.val ≤ values.val.length ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        k < index.val → state.1[k]! = values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        index.val ≤ k → k < state.2.val →
          state.1[k]! = fromCell columns bounds k values[k]!) ∧
      (∀ (k : Nat) (_hk : k < values.val.length),
        state.2.val ≤ k → state.1[k]! = values[k]!))
  · intro state hInv
    rcases state with ⟨current, currentIndex⟩
    dsimp only at hInv
    rcases hInv with ⟨hlen, hle1, hle2, hpre, hmid, hsuf⟩
    simp only [coordinate_flat.from_unit_loop.body]
    by_cases inside : currentIndex.val < current.val.length
    · simp [UScalar.lt_equiv, inside]
      have hInsideVal : currentIndex.val < values.val.length := by omega
      have hmod : currentIndex.val % columns.val < columns.val :=
        Nat.mod_lt _ columnPos
      have hbounds1 : (currentIndex.val % columns.val) * 2 < bounds.val.length := by
        omega
      have hbounds2 : (currentIndex.val % columns.val) * 2 + 1 < bounds.val.length := by
        omega
      step*
      simp only [fromExtracted]
      step*
      have hsLen : s.val.length = values.val.length := by
        subst s_post
        simpa [Slice.set_length] using hlen
      have hidx1 : index.val ≤ index1.val := by
        have := index1_post
        omega
      have hidx2 : index1.val ≤ values.val.length := by
        have := index1_post
        omega
      have hneq : ∀ (k : Nat), k < index.val → currentIndex.val ≠ k := by
        intro k _hk; omega
      have hPre' : ∀ (k : Nat) (_hk : k < values.val.length),
          k < index.val → s[k]! = values[k]! := by
        intro k hk hlt
        subst s_post
        rw [Slice.getElem!_Nat_set_ne current currentIndex k _ (hneq k hlt)]
        exact hpre k hk hlt
      have hidx_i1 : i1.val = (currentIndex.val % columns.val) * 2 := by
        have := i1_post; have := column_post; omega
      have hi1 : i1.val < bounds.val.length := by
        rw [hidx_i1]; exact hbounds1
      have hLow' : lower = cellLow columns bounds currentIndex.val := by
        have h1 : bounds.val[i1.val] = bounds.val[i1.val]! :=
          List.Inhabited_getElem_eq_getElem! bounds.val i1.val hi1
        rw [lower_post, h1]
        dsimp only [cellLow]
        exact congrArg (fun j => bounds.val[j]!) hidx_i1
      have hidx_i2 : i2.val = (currentIndex.val % columns.val) * 2 + 1 := by
        have := i2_post; have := i1_post; have := column_post; omega
      have hi2 : i2.val < bounds.val.length := by
        rw [hidx_i2]; exact hbounds2
      have hUp' : upper = cellUp columns bounds currentIndex.val := by
        have h2 : bounds.val[i2.val] = bounds.val[i2.val]! :=
          List.Inhabited_getElem_eq_getElem! bounds.val i2.val hi2
        rw [upper_post, h2]
        dsimp only [cellUp]
        exact congrArg (fun j => bounds.val[j]!) hidx_i2
      have hfl : fl = values[currentIndex.val]! := by
        have h1 : current.val[currentIndex.val] = current.val[currentIndex.val]! :=
          List.Inhabited_getElem_eq_getElem! current.val currentIndex.val (by omega)
        have h2 : current[currentIndex.val]! = current.val[currentIndex.val]! :=
          Slice.getElem!_Nat_eq current currentIndex.val
        rw [fl_post, h1, ← h2]
        exact hsuf currentIndex.val (by omega) le_rfl
      have hcellEq : fromSpec fl lower upper =
          fromCell columns bounds currentIndex.val values[currentIndex.val]! := by
        dsimp only [fromCell]
        rw [hfl, hLow', hUp']
      have hMid' : ∀ (k : Nat) (_hk : k < values.val.length),
          index.val ≤ k → k < index1.val →
            s[k]! = fromCell columns bounds k values[k]! := by
        intro k hk hge _hlt
        have _hle : k ≤ currentIndex.val := by
          have := index1_post; omega
        by_cases heq : k = currentIndex.val
        · subst s_post
          have hEq' : currentIndex.val = k := heq.symm
          have hLt' : k < current.length := by
            have : current.length = current.val.length := rfl
            omega
          have hset : (current.set currentIndex (fromSpec fl lower upper))[k]! =
              fromSpec fl lower upper :=
            Slice.getElem!_Nat_set_eq current currentIndex k (fromSpec fl lower upper) ⟨hEq', hLt'⟩
          rw [hset, hcellEq, heq]
        · subst s_post
          have hne : currentIndex.val ≠ k := Ne.symm heq
          rw [Slice.getElem!_Nat_set_ne current currentIndex k _ hne]
          exact hmid k hk hge (by omega)
      have hSuf' : ∀ (k : Nat) (_hk : k < values.val.length),
          index1.val ≤ k → s[k]! = values[k]! := by
        intro k hk hge
        subst s_post
        have hne : currentIndex.val ≠ k := by
          have := index1_post; omega
        rw [Slice.getElem!_Nat_set_ne current currentIndex k _ hne]
        exact hsuf k hk (by have := index1_post; omega)
      refine ⟨hsLen, hidx1, hidx2, hPre', hMid', hSuf', ?_⟩
      have := index1_post
      omega
    · have outside : ¬currentIndex.val < values.val.length := by
        simpa [hlen] using inside
      simp [UScalar.lt_equiv, outside, hlen]
      exact ⟨hpre, fun k hk hge => hmid k hk hge (by omega)⟩
  · dsimp only
    refine ⟨rfl, le_rfl, hindex, fun _ _ _ => rfl, fun _ _ _ h => by omega, fun _ _ _ => rfl⟩

theorem fromLoop (columns : Usize) (values bounds : Slice F64)
    (columnPos : 0 < columns.val)
    (boundsLen : bounds.val.length = columns.val * 2) :
    coordinate_flat.from_unit_loop columns values bounds 0#usize ⦃ result =>
      result.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        result[k]! = fromCell columns bounds k values[k]! ⦄ := by
  have hzero : (0#usize).val ≤ values.val.length := Nat.zero_le _
  have hpre := fromPrefix columns values bounds 0#usize columnPos boundsLen hzero
  apply spec_mono hpre
  intro result ⟨hlen, _, hpost⟩
  refine ⟨hlen, fun k hk => hpost k hk (Nat.zero_le _)⟩

theorem fromNil (columns : Usize) (values bounds : Slice F64)
    (hlen : values.val.length = 0) :
    coordinate_flat.from_unit_loop columns values bounds 0#usize ⦃ result =>
      result.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        result[k]! = fromCell columns bounds k values[k]! ⦄ := by
  unfold coordinate_flat.from_unit_loop
  apply loop.spec_decr_nat
    (measure := fun state => values.val.length - state.2.val)
    (inv := fun state => state.1.val.length = values.val.length)
  · intro state sameLength
    rcases state with ⟨current, currentIndex⟩
    dsimp only at sameLength
    simp only [coordinate_flat.from_unit_loop.body]
    by_cases inside : currentIndex.val < current.val.length
    · have : False := by
        have := sameLength
        have := hlen
        omega
      contradiction
    · have outside : ¬currentIndex.val < values.val.length := by
        simpa [sameLength] using inside
      simp [UScalar.lt_equiv, outside, sameLength]
      intro k hk
      have := hlen
      omega
  · rfl

/-- The extracted in-place loop is total on every shape accepted by the Rust
adapter and preserves the flat buffer extent. -/
@[step] theorem toLoopSafe (columns : Usize) (values bounds : Slice F64) (index : Usize)
    (columnPos : 0 < columns.val)
    (boundsLen : bounds.val.length = columns.val * 2) :
    coordinate_flat.to_unit_loop columns values bounds index ⦃ result =>
      result.val.length = values.val.length ⦄ := by
  unfold coordinate_flat.to_unit_loop
  apply loop.spec_decr_nat
    (measure := fun state => state.1.val.length - state.2.val)
    (inv := fun state => state.1.val.length = values.val.length)
  · intro state sameLength
    rcases state with ⟨current, currentIndex⟩
    simp only [coordinate_flat.to_unit_loop.body]
    by_cases inside : currentIndex.val < current.val.length
    · simp [UScalar.lt_equiv, inside]
      step*
      simp only [extracted]
      step*
    · have outside : ¬currentIndex.val < values.val.length := by
        simpa [sameLength] using inside
      simp [UScalar.lt_equiv, outside, sameLength]
  · rfl

theorem toFail (rows columns : Usize) (values bounds : Slice F64)
    (rejected : coordinate_flat.shape_ok rows columns values bounds = Result.ok false) :
    coordinate_flat.to_unit rows columns values bounds = Result.ok (false, values) := by
  simp [coordinate_flat.to_unit, rejected]

theorem fromFail (rows columns : Usize) (values bounds : Slice F64)
    (rejected : coordinate_flat.shape_ok rows columns values bounds = Result.ok false) :
    coordinate_flat.from_unit rows columns values bounds = Result.ok (false, values) := by
  simp [coordinate_flat.from_unit, rejected]

theorem toOk (rows columns : Usize) (values bounds : Slice F64)
    (accepted : coordinate_flat.shape_ok rows columns values bounds = Result.ok true) :
    coordinate_flat.to_unit rows columns values bounds ⦃ res =>
      res.1 = true ∧
      res.2.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        res.2[k]! = toCell columns bounds k values[k]! ⦄ := by
  have ⟨hvalLen, hboundsLen⟩ := shapeLengths rows columns values bounds accepted
  simp only [coordinate_flat.to_unit, accepted, bind_ok, ite_true]
  by_cases hcol : 0 < columns.val
  · have hloop := toLoop columns values bounds hcol hboundsLen
    apply spec_bind hloop
    intro result hres
    simp only [spec_ok]
    exact ⟨trivial, hres.1, hres.2⟩
  · have hcol0 : columns.val = 0 := by omega
    have hzero : values.val.length = 0 := by
      rw [hvalLen, hcol0, Nat.mul_zero]
    have hloop := toNil columns values bounds hzero
    apply spec_bind hloop
    intro result hres
    simp only [spec_ok]
    exact ⟨trivial, hres.1, hres.2⟩

theorem fromOk (rows columns : Usize) (values bounds : Slice F64)
    (accepted : coordinate_flat.shape_ok rows columns values bounds = Result.ok true) :
    coordinate_flat.from_unit rows columns values bounds ⦃ res =>
      res.1 = true ∧
      res.2.val.length = values.val.length ∧
      ∀ (k : Nat) (_hk : k < values.val.length),
        res.2[k]! = fromCell columns bounds k values[k]! ⦄ := by
  have ⟨hvalLen, hboundsLen⟩ := shapeLengths rows columns values bounds accepted
  simp only [coordinate_flat.from_unit, accepted, bind_ok, ite_true]
  by_cases hcol : 0 < columns.val
  · have hloop := fromLoop columns values bounds hcol hboundsLen
    apply spec_bind hloop
    intro result hres
    simp only [spec_ok]
    exact ⟨trivial, hres.1, hres.2⟩
  · have hcol0 : columns.val = 0 := by omega
    have hzero : values.val.length = 0 := by
      rw [hvalLen, hcol0, Nat.mul_zero]
    have hloop := fromNil columns values bounds hzero
    apply spec_bind hloop
    intro result hres
    simp only [spec_ok]
    exact ⟨trivial, hres.1, hres.2⟩

/-- List model of the production row traversal. Shape truncation agrees with
the two nested ndarray loops once the rectangularity invariant is supplied by
the caller. Each arithmetic cell is the extracted Rust operation above. -/
def extractedRow : List (F64 × F64) → List F64 → Result (List F64)
  | [], _ => .ok []
  | _, [] => .ok []
  | (lower, upper) :: bounds, value :: values => do
      let head ← coordinate_scalar.to_unit value lower upper
      let tail ← extractedRow bounds values
      Result.ok (head :: tail)

def specRow : List (F64 × F64) → List F64 → List F64
  | [], _ => []
  | _, [] => []
  | (lower, upper) :: bounds, value :: values =>
      spec value lower upper :: specRow bounds values

theorem rowExtracted (bounds : List (F64 × F64)) (values : List F64) :
    extractedRow bounds values = .ok (specRow bounds values) := by
  induction bounds generalizing values with
  | nil => simp [extractedRow, specRow]
  | cons bound bounds ih =>
      cases values with
      | nil => simp [extractedRow, specRow]
      | cons value values =>
          rcases bound with ⟨lower, upper⟩
          simp [extractedRow, specRow, extracted, ih]

def extractedMatrix (bounds : List (F64 × F64)) :
    List (List F64) → Result (List (List F64))
  | [] => .ok []
  | row :: rows => do
      let head ← extractedRow bounds row
      let tail ← extractedMatrix bounds rows
      Result.ok (head :: tail)

def specMatrix (bounds : List (F64 × F64)) (rows : List (List F64)) :
    List (List F64) := rows.map (specRow bounds)

theorem matrixExtracted (bounds : List (F64 × F64))
    (rows : List (List F64)) :
    extractedMatrix bounds rows = .ok (specMatrix bounds rows) := by
  induction rows with
  | nil => rfl
  | cons row rows ih =>
      simp [extractedMatrix, specMatrix, rowExtracted, ih]

#print axioms extracted
#print axioms fromExtracted
#print axioms shapeAccepted
#print axioms shapeLengths
#print axioms toPrefix
#print axioms toLoop
#print axioms toNil
#print axioms fromPrefix
#print axioms fromLoop
#print axioms fromNil
#print axioms toLoopSafe
#print axioms toFail
#print axioms fromFail
#print axioms toOk
#print axioms fromOk
#print axioms rowExtracted
#print axioms matrixExtracted

end BauxResearch.Extraction.LocalCoord
