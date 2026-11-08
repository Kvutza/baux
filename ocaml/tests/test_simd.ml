open Baux_benchmarks
module I = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u

let checks = ref 0

let check id expected actual =
  let actual = U.to_float actual in
  let tolerance = if id = "bukin6" then 1e-6 else 2e-10 in
  if
    (not (Float.is_finite expected && Float.is_finite actual))
    || Float.abs (expected -. actual)
       > tolerance *. Float.max 1. (Float.abs expected)
  then
    failwith
      (Printf.sprintf "%s: expected %.17g, SIMD %.17g" id expected actual);
  incr checks

let rejects label f =
  match f () with
  | exception Invalid_argument _ -> ()
  | _ -> failwith (label ^ ": expected Invalid_argument")

let fixtures () =
  let ch = open_in "reference.tsv" in
  Fun.protect
    ~finally:(fun () -> close_in ch)
    (fun () ->
      try
        while true do
          let line = input_line ch in
          if line <> "" && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ id; n; coordinates; expected ] ->
                let p = Problem.create ~dimension:(int_of_string n) id in
                let x =
                  I.of_list
                    (List.map float_of_string
                       (String.split_on_char ',' coordinates))
                in
                let w = Batch.prepare p [: x; x; x :] in
                Batch.run Parallel.sequential Simd w;
                for i = 0 to 2 do
                  check id (float_of_string expected) (Batch.get w i)
                done
            | _ -> failwith "Invalid fixture"
        done
      with End_of_file -> ())

let dimensions (d : Problem.description) =
  match d.dimensions with
  | Fixed n -> [ n ]
  | At_least n -> List.sort_uniq compare [ n; max n 3; max n 16 ]
  | Multiple_of n -> [ n; 2 * n; 4 * n ]

let exercise (parallel @ local) p count block_pairs =
  let id = (Problem.describe p).id in
  let n = Problem.dimension p in
  let points =
    I.init count (fun row ->
        I.init n (fun col ->
            let low, high = I.get (Problem.bounds p) col in
            let fraction =
              float_of_int (((row * 19) + (col * 31)) mod 257) /. 256.
            in
            low +. ((high -. low) *. fraction)))
  in
  let w = Batch.prepare ~block_pairs p points in
  Batch.run Parallel.sequential Scalar w;
  let scalar = Batch.snapshot w in
  Batch.run parallel Simd w;
  let first = Batch.snapshot w in
  for i = 0 to count - 1 do
    check id (U.to_float (I.get scalar i)) (I.get first i)
  done;
  Batch.run Parallel.sequential Simd w;
  for i = 0 to count - 1 do
    if
      Int64.bits_of_float (U.to_float (I.get first i))
      <> Int64.bits_of_float (U.to_float (Batch.get w i))
    then failwith (id ^ ": SIMD result depends on scheduling")
  done;
  Batch.run parallel Scalar w;
  for i = 0 to count - 1 do
    if
      Int64.bits_of_float (U.to_float (I.get scalar i))
      <> Int64.bits_of_float (U.to_float (Batch.get w i))
    then failwith (id ^ ": scalar workspace reuse mismatch");
    check id (U.to_float (Batch.get w i)) (I.get first i)
  done;
  rejects "negative result index" (fun () -> U.to_float (Batch.get w (-1)));
  rejects "past-end result index" (fun () -> U.to_float (Batch.get w count))

let catalogue (parallel @ local) =
  let rec descriptions = function
    | [] -> ()
    | d :: rest ->
        let rec dims = function
          | [] -> ()
          | n :: ns ->
              let p = Problem.create ~dimension:n d.Problem.id in
              let rec sizes = function
                | [] -> ()
                | count :: counts ->
                    exercise parallel p count 7;
                    exercise parallel p count 128;
                    sizes counts
              in
              sizes [ 0; 1; 2; 3; 15; 256; 257 ] [@nontail];
              dims ns
        in
        dims (dimensions d) [@nontail];
        descriptions rest
  in
  descriptions Problem.catalogue [@nontail]

let () =
  fixtures ();
  Parallel_scheduler.with_parallel ~max_workers:4 catalogue;
  let p = Problem.create "sphere" in
  rejects "zero block size" (fun () -> Batch.prepare ~block_pairs:0 p [::]);
  rejects "wrong dimension" (fun () -> Batch.prepare p [: [: 1. :] :]);
  rejects "non-finite coordinate" (fun () ->
      Batch.prepare p [: [: Float.nan; 0. :] :]);
  Printf.printf
    "%d SIMD reference, tail, layout, reuse and multicore checks passed.\n"
    !checks
