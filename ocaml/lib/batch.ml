module Iarray = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u
module P = Parallel.Arrays.Iarray

type backend = Scalar | Simd

type block = {
  input : floatarray;
  output : floatarray;
  first : int;
  count : int;
}

type workspace = {
  problem : Problem.t;
  count : int;
  block_pairs : int;
  blocks : block P.t;
  mutable initialized : bool;
  mutable ready : bool;
}

let create ?(block_pairs = 128) problem ~length =
  let dimension = Problem.dimension problem in
  if
    length < 0 || block_pairs < 1
    || block_pairs > Sys.max_array_length / 2
    || dimension > Sys.max_array_length / (2 * block_pairs)
  then invalid_arg "Invalid packed batch size";
  let pairs = (length / 2) + (length mod 2) in
  let nblocks =
    (pairs / block_pairs) + if pairs mod block_pairs = 0 then 0 else 1
  in
  let blocks =
    P.init Parallel.sequential nblocks ~f:(fun _ index ->
        let first = 2 * index * block_pairs in
        let count = min (2 * block_pairs) (length - first) in
        let padded = count + (count mod 2) in
        {
          input = Float.Array.make (padded * dimension) 0.;
          output = Float.Array.make padded 0.;
          first;
          count;
        })
  in
  {
    problem;
    count = length;
    block_pairs;
    blocks;
    initialized = false;
    ready = false;
  }

let length w = w.count

let refill w ~(f : (int -> int -> float#) @ portable) =
  w.initialized <- false;
  w.ready <- false;
  let dimension = Problem.dimension w.problem in
  P.iter Parallel.sequential w.blocks ~f:(fun _ b ->
      for row = 0 to b.count - 1 do
        for col = 0 to dimension - 1 do
          let x = U.to_float (f (b.first + row) col) in
          if not (Float.is_finite x) then invalid_arg "Non-finite coordinate";
          let offset = (row / 2 * 2 * dimension) + (2 * col) + (row mod 2) in
          Float.Array.set b.input offset x
        done
      done;
      if b.count mod 2 = 1 then
        for col = 0 to dimension - 1 do
          let offset = (b.count / 2 * 2 * dimension) + (2 * col) in
          Float.Array.set b.input (offset + 1) (Float.Array.get b.input offset)
        done);
  w.initialized <- true

let init ?block_pairs problem ~length ~f =
  let w = create ?block_pairs problem ~length in
  refill w ~f;
  w

let prepare ?block_pairs problem (points : Problem.point iarray) =
  Iarray.iter
    (fun p ->
      if Iarray.length p <> Problem.dimension problem then
        invalid_arg "Point dimension mismatch")
    points;
  init ?block_pairs problem ~length:(Iarray.length points) ~f:(fun row col ->
      U.of_float (Iarray.get (Iarray.get points row) col))

let run parallel backend w =
  if not w.initialized then invalid_arg "Packed inputs require refill";
  w.ready <- false;
  let problem = w.problem in
  let dimension = Problem.dimension problem in
  P.iter parallel w.blocks ~f:(fun _ b ->
      match backend with
      | Simd ->
          for pair = 0 to ((b.count + 1) / 2) - 1 do
            Numeric.Neon.store b.output (2 * pair)
              (Problem.Packed.simd problem b.input
                 ~offset:(pair * 2 * dimension))
          done
      | Scalar ->
          for row = 0 to b.count - 1 do
            let offset = (row / 2 * 2 * dimension) + (row mod 2) in
            Float.Array.set b.output row
              (U.to_float (Problem.Packed.scalar problem b.input ~offset))
          done);
  w.ready <- true

let require_results w =
  if not w.ready then invalid_arg "Results require a successful run"

let get w i =
  require_results w;
  if i < 0 || i >= w.count then invalid_arg "Result index out of bounds";
  let block_size = 2 * w.block_pairs in
  U.of_float
    (P.extract w.blocks (i / block_size) (fun b ->
         Float.Array.get b.output (i mod block_size)))

let fold_results w ~init ~(f : (int -> float# -> float# -> float#) @ portable) =
  require_results w;
  let rec blocks index acc =
    if index = (w.count + (2 * w.block_pairs) - 1) / (2 * w.block_pairs) then
      acc
    else
      let next =
        P.extract w.blocks index (fun b ->
            let rec rows row acc =
              if row = b.count then U.to_float acc
              else
                rows (row + 1)
                  (f (b.first + row) acc
                     (U.of_float (Float.Array.get b.output row)))
            in
            rows 0 acc)
      in
      blocks (index + 1) (U.of_float next)
  in
  blocks 0 init

let snapshot w =
  require_results w;
  let values = Base.Array.create ~len:w.count #0. in
  for i = 0 to w.count - 1 do
    Base.Array.set values i (get w i)
  done;
  (* Fresh nonescaping storage; never mutated after freezing. *)
  Base.Iarray.unsafe_of_array__promise_no_mutation values

let evaluate ?(backend = Scalar) parallel (problem : Problem.t)
    (points : Problem.point iarray) =
  match backend with
  | Scalar ->
      let values =
        (P.init [@kind float64]) parallel (Iarray.length points) ~f:(fun _ i ->
            Problem.evaluate problem (Iarray.get points i))
      in
      P.to_iarray values
  | Simd ->
      let w = prepare problem points in
      run parallel Simd w;
      snapshot w
