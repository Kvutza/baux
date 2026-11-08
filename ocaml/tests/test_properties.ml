open Baux_benchmarks
module I = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u
module Q = Base_quickcheck

module Case = struct
  type t = int * int * int * int

  let sexp_of_t (n, count, block, seed) =
    Base.Sexp.List
      (List.map
         (fun n -> Base.Sexp.Atom (string_of_int n))
         [ n; count; block; seed ])

  let quickcheck_generator =
    let open Q.Generator.Let_syntax in
    let%bind n = Q.Generator.int_inclusive 1 32 in
    let%bind count = Q.Generator.int_inclusive 0 129 in
    let%bind block = Q.Generator.int_inclusive 1 64 in
    let%map seed = Q.Generator.int_inclusive 0 100000 in
    (n, count, block, seed)

  let quickcheck_shrinker =
    Q.Shrinker.create (fun (n, count, block, seed) ->
        Base.Sequence.of_list
          (List.filter
             (fun x -> x <> (n, count, block, seed))
             [
               (max 1 (n / 2), count, block, seed);
               (n, count / 2, block, seed);
               (n, count, max 1 (block / 2), seed);
               (n, count, block, seed / 2);
             ]))
end

let check id expected actual =
  let actual = U.to_float actual in
  let tolerance = if id = "bukin6" then 1e-6 else 2e-10 in
  if
    (not (Float.is_finite expected && Float.is_finite actual))
    || Float.abs (expected -. actual)
       > tolerance *. Float.max 1. (Float.abs expected)
  then failwith (Printf.sprintf "%s: %.17g <> %.17g" id expected actual)

let rejects f =
  match f () with
  | exception Invalid_argument _ -> ()
  | _ -> failwith "Expected Invalid_argument"

let exercise (parallel @ local) (d : Problem.description) (n, count, block, seed)
    =
  let n =
    match d.dimensions with
    | Fixed n -> n
    | At_least k -> max n k
    | Multiple_of k -> k * max 1 ((n + k - 1) / k)
  in
  let problem = Problem.create ~dimension:n d.id in
  let bounds = Problem.bounds problem in
  let coordinate seed row col =
    let low, high = I.get bounds col in
    let r = (seed + (row * 104729) + (col * 7919)) mod 1009 in
    U.of_float (low +. ((high -. low) *. float_of_int r /. 1008.))
  in
  let w =
    Batch.init ~block_pairs:block problem ~length:count ~f:(coordinate seed)
  in
  Batch.run parallel Simd w;
  let snapshot = Batch.snapshot w in
  for row = 0 to count - 1 do
    let point = I.init n (fun col -> U.to_float (coordinate seed row col)) in
    check d.id (U.to_float (Problem.evaluate problem point)) (Batch.get w row)
  done;
  let sum = Batch.fold_results w ~init:#0. ~f:(fun _ a b -> U.add a b) in
  let expected_sum = ref 0. in
  for row = 0 to count - 1 do
    expected_sum := !expected_sum +. U.to_float (I.get snapshot row)
  done;
  check d.id !expected_sum sum;
  Batch.refill w ~f:(coordinate (seed + 1));
  rejects (fun () -> U.to_float (Batch.get w 0));
  Batch.run Parallel.sequential Scalar w;
  let scalar = Batch.snapshot w in
  Batch.run parallel Simd w;
  for row = 0 to count - 1 do
    check d.id (U.to_float (I.get scalar row)) (Batch.get w row);
    let point = I.init n (fun col -> U.to_float (coordinate seed row col)) in
    check d.id
      (U.to_float (Problem.evaluate problem point))
      (I.get snapshot row)
  done;
  if count > 0 then (
    rejects (fun () -> Batch.refill w ~f:(fun _ _ -> U.of_float Float.nan));
    rejects (fun () -> Batch.run parallel Simd w);
    Batch.refill w ~f:(coordinate seed);
    Batch.run parallel Simd w)

let () =
  Parallel_scheduler.with_parallel ~max_workers:4 (fun parallel ->
      let rec run = function
        | [] -> ()
        | d :: rest ->
            let config =
              {
                Q.Test.default_config with
                test_count = 50;
                seed = Deterministic ("baux-packed-" ^ d.Problem.id);
              }
            in
            Q.Test.run_exn (module Case) ~config ~f:(exercise parallel d)
            [@nontail];
            run rest
      in
      run Problem.catalogue [@nontail]);
  let w = Batch.create (Problem.create "sphere") ~length:2 in
  rejects (fun () -> Batch.run Parallel.sequential Simd w);
  rejects (fun () -> Batch.snapshot w);
  print_endline
    "2800 deterministic Quickcheck cases across all 56 functions passed."
