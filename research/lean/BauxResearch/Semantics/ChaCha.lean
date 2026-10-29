import Mathlib.Data.BitVec

/-!
# Executable seeded ChaCha12 word stream

The scalar rounds and PCG seed expansion follow rand_chacha 0.3.1 and rand_core 0.6.4,
as pinned by upstream ENN. A state records the key, block counter and unconsumed words.
These definitions calculate words; expected outputs appear only in regression theorems.
Correspondence to the upstream SIMD implementation remains a separate proof obligation.
-/

namespace BauxResearch.Semantics.ChaCha

abbrev Block := Array (BitVec 32)

/-- One ChaCha quarter round, with wrapping 32-bit arithmetic. -/
def quarter (x : Block) (a b c d : Nat) : Block := Id.run do
  let mut x := x
  x := x.set! a (x[a]! + x[b]!)
  x := x.set! d ((x[d]! ^^^ x[a]!).rotateLeft 16)
  x := x.set! c (x[c]! + x[d]!)
  x := x.set! b ((x[b]! ^^^ x[c]!).rotateLeft 12)
  x := x.set! a (x[a]! + x[b]!)
  x := x.set! d ((x[d]! ^^^ x[a]!).rotateLeft 8)
  x := x.set! c (x[c]! + x[d]!)
  x := x.set! b ((x[b]! ^^^ x[c]!).rotateLeft 7)
  return x

/-- Column round followed by diagonal round. -/
def roundPair (x : Block) : Block :=
  let x := quarter x 0 4 8 12
  let x := quarter x 1 5 9 13
  let x := quarter x 2 6 10 14
  let x := quarter x 3 7 11 15
  let x := quarter x 0 5 10 15
  let x := quarter x 1 6 11 12
  let x := quarter x 2 7 8 13
  quarter x 3 4 9 14

/-- ChaCha12 with a 64-bit block counter and 64-bit stream identifier. -/
def block (key : Block) (counter stream : BitVec 64) : Block :=
  let initial : Block := (#[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574] : Block) ++ key ++
    #[BitVec.ofNat 32 counter.toNat, BitVec.ofNat 32 (counter.toNat >>> 32),
      BitVec.ofNat 32 stream.toNat, BitVec.ofNat 32 (stream.toNat >>> 32)]
  let mixed := (List.range 6).foldl (fun state _ ↦ roundPair state) initial
  (List.range 16).toArray.map (fun i ↦ mixed[i]! + initial[i]!)

/-- The default SeedableRng::seed_from_u64 PCG expansion step. -/
def seedStep (state : BitVec 64) : BitVec 32 × BitVec 64 :=
  let state := state * 6364136223846793005 + 11634580027462260723
  let word := BitVec.ofNat 32 ((((state >>> 18) ^^^ state) >>> 27).toNat)
  (word.rotateRight ((state >>> 59).toNat), state)

/-- Expand one u64 to the eight u32 words of a ChaCha key. -/
def seedKey (seed : BitVec 64) : Block :=
  ((List.range 8).foldl (fun (acc : Block × BitVec 64) _ ↦
    let (word, state) := seedStep acc.2
    (acc.1.push word, state)) (#[], seed)).1

/-- Logical stream state; buffering four blocks upstream does not change this word order. -/
structure State where
  key : Block
  counter : BitVec 64 := 0
  stream : BitVec 64 := 0
  pending : List (BitVec 32) := []
  deriving Repr, DecidableEq

def seeded (seed : BitVec 64) : State := { key := seedKey seed }

/-- Interpret the 32 bytes passed to `StdRng::from_seed`. `fill_bytes` emits
each caller `u64` little-endian and rand_chacha reads eight little-endian
`u32` key words, so every input word contributes its low half first. -/
def directKey (seed : Array (BitVec 64)) : Block :=
  seed.foldl (fun key word =>
    key.push (BitVec.ofNat 32 word.toNat) |>.push
      (BitVec.ofNat 32 (word.toNat >>> 32))) #[]

/-- The fresh ChaCha12 state constructed by ENN's private surrogate RNG. -/
def fromWords (seed : Array (BitVec 64)) : State :=
  { key := directKey seed }

theorem directOrder (a b c d : BitVec 64) :
    (fromWords #[a, b, c, d]).key =
      #[BitVec.ofNat 32 a.toNat, BitVec.ofNat 32 (a.toNat >>> 32),
        BitVec.ofNat 32 b.toNat, BitVec.ofNat 32 (b.toNat >>> 32),
        BitVec.ofNat 32 c.toNat, BitVec.ofNat 32 (c.toNat >>> 32),
        BitVec.ofNat 32 d.toNat, BitVec.ofNat 32 (d.toNat >>> 32)] := by
  rfl

def next32 (state : State) : BitVec 32 × State :=
  match state.pending with
  | word :: rest => (word, { state with pending := rest })
  | [] =>
    let words := (block state.key state.counter state.stream).toList
    (words.headD 0, { state with counter := state.counter + 1, pending := words.drop 1 })

/-- rand_core combines the low u32 first, then the high u32. -/
def next64 (state : State) : BitVec 64 × State :=
  let (low, state) := next32 state
  let (high, state) := next32 state
  (BitVec.ofNat 64 (low.toNat + (high.toNat <<< 32)), state)

/-- Any finite prefix of the concrete stream, together with its residual state. -/
def takeWords : Nat → State → List (BitVec 64) × State
  | 0, state => ([], state)
  | n + 1, state =>
    let (word, state) := next64 state
    let (rest, state) := takeWords n state
    (word :: rest, state)

set_option maxRecDepth 8192 in
/-- Independent rand::StdRng seed-42 word fixture, crossing the first block boundary. -/
theorem seedFixture :
    (takeWords 10 (seeded 42)).1 =
      [9713269763989775522, 10011513049433592189, 11740708795755607249,
       7487565853151867058, 633513173585076202, 7654602743214997928,
       13603079691933612283, 15665927001465599799, 2421668070752551774,
       59990589126097438] := by decide +kernel

#print axioms seedFixture
#print axioms directOrder

end BauxResearch.Semantics.ChaCha
