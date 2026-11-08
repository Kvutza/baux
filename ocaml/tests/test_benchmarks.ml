open Baux_benchmarks
module Iarray = Stdlib_stable.Iarray
module F = Stdlib_upstream_compatible.Float_u

let check ?(tolerance = 1e-10) label expected actual =
  let actual = F.to_float actual in
  if
    (not (Float.is_finite actual))
    || Float.abs (actual -. expected)
       > tolerance *. Float.max 1. (Float.abs expected)
  then
    failwith
      (Printf.sprintf "%s: expected %.17g; got %.17g" label expected actual)

let evaluate id x =
  Problem.evaluate (Problem.create ~dimension:(Iarray.length x) id) x

let rejects label f =
  match f () with
  | exception Invalid_argument _ -> ()
  | _ -> failwith (label ^ ": expected Invalid_argument")

let reference_cases () =
  let input = open_in "reference.tsv" in
  let count = ref 0 and covered = Hashtbl.create 64 in
  Fun.protect
    ~finally:(fun () -> close_in input)
    (fun () ->
      try
        while true do
          let line = input_line input in
          if String.length line > 0 && line.[0] <> '#' then
            match String.split_on_char '\t' line with
            | [ id; dimension; coordinates; expected ] ->
                let dimension = int_of_string dimension in
                let values =
                  List.map float_of_string
                    (String.split_on_char ',' coordinates)
                in
                let point = Iarray.of_list values in
                let problem = Problem.create ~dimension id in
                check
                  (id ^ ": " ^ coordinates)
                  (float_of_string expected)
                  (Problem.evaluate problem point);
                incr count;
                Hashtbl.replace covered id ()
            | _ -> failwith ("Malformed reference fixture: " ^ line)
        done
      with End_of_file -> ());
  List.iter
    (fun (d : Problem.description) ->
      if not (Hashtbl.mem covered d.id) then
        failwith ("Missing reference cases: " ^ d.id))
    Problem.catalogue;
  Printf.printf "%d numerical reference cases passed.\n" !count

let analytic_cases () =
  List.iter
    (fun id -> check (id ^ " origin") 0. (evaluate id [: 0.; 0. :]))
    [
      "sphere";
      "ackley";
      "rastrigin";
      "griewank";
      "sum-squares";
      "rotated-hyperellipsoid";
      "sum-powers";
      "zakharov";
      "matyas";
      "three-hump-camel";
      "bohachevsky1";
      "bohachevsky2";
      "bohachevsky3";
    ];
  check "sphere" 14. (evaluate "sphere" [: 1.; 2.; 3. :]);
  check "rosenbrock minimum" 0. (evaluate "rosenbrock" [: 1.; 1.; 1. :]);
  check "rosenbrock off minimum" 104. (evaluate "rosenbrock" [: -1.; 2. :]);
  check "beale minimum" 0. (evaluate "beale" [: 3.; 0.5 :]);
  check "booth minimum" 0. (evaluate "booth" [: 1.; 3. :]);
  (* The square root amplifies FMA rounding near Bukin's cusp. *)
  check ~tolerance:1e-6 "bukin minimum" 0. (evaluate "bukin6" [: -10.; 1. :]);
  check "goldstein minimum" 3. (evaluate "goldstein-price" [: 0.; -1. :]);
  check "easom minimum" (-1.) (evaluate "easom" [: Float.pi; Float.pi :]);
  check "powell minimum" 0. (evaluate "powell" [: 0.; 0.; 0.; 0. :]);
  check "power sum minimum" 0. (evaluate "power-sum" [: 1.; 2.; 2.; 3. :]);
  check "perm minimum" 0. (evaluate "perm" [: 1.; 2.; 3. :]);
  check "perm zero minimum" 0. (evaluate "perm-zero" [: 1.; 0.5; 1. /. 3. :]);
  check "trid minimum" (-50.) (evaluate "trid" [: 6.; 10.; 12.; 12.; 10.; 6. :]);
  check "levy dimension one" 0. (evaluate "levy" [: 1. :]);
  check "sphere dimension one" 4. (evaluate "sphere" [: 2. :])

let validation_cases () =
  rejects "unknown" (fun () -> Problem.create "unknown");
  List.iter
    (fun id ->
      rejects (id ^ " zero dimension") (fun () ->
          Problem.create ~dimension:0 id);
      rejects (id ^ " negative dimension") (fun () ->
          Problem.create ~dimension:(-1) id))
    [ "sphere"; "powell"; "beale" ];
  rejects "fixed dimension" (fun () -> Problem.create ~dimension:3 "beale");
  rejects "minimum dimension" (fun () ->
      Problem.create ~dimension:1 "rosenbrock");
  rejects "block dimension" (fun () -> Problem.create ~dimension:6 "powell");
  let p = Problem.create "sphere" in
  rejects "point dimension" (fun () -> F.to_float (Problem.evaluate p [: 1. :]));
  List.iter
    (fun v ->
      rejects "non-finite input" (fun () ->
          F.to_float (Problem.evaluate p [: v; 0. :])))
    [ Float.nan; Float.infinity; Float.neg_infinity ];
  if Problem.within_bounds p [: Float.nan; 0. :] then failwith "NaN in domain";
  if Problem.within_bounds p [: 0. :] then failwith "Wrong dimension in domain";
  if Problem.within_bounds p [: 6.; 0. :] then failwith "Outside domain";
  check "no implicit clamping" 36. (Problem.evaluate p [: 6.; 0. :]);
  let ids =
    List.map (fun (d : Problem.description) -> d.id) Problem.catalogue
  in
  if List.length ids <> List.length (List.sort_uniq String.compare ids) then
    failwith "Duplicate identifiers";
  List.iter
    (fun id ->
      let p = Problem.create id in
      let low = Iarray.map fst (Problem.bounds p) in
      let high = Iarray.map snd (Problem.bounds p) in
      if not (Problem.within_bounds p low && Problem.within_bounds p high) then
        failwith ("Invalid bounds: " ^ id))
    ids

let batch_cases (parallel @ local) =
  let rec loop = function
    | [] -> ()
    | (d : Problem.description) :: rest ->
        let p = Problem.create d.id in
        let points =
          Iarray.init 257 (fun row ->
              Iarray.init (Problem.dimension p) (fun col ->
                  let low, high = Iarray.get (Problem.bounds p) col in
                  low
                  +. (high -. low)
                     *. float_of_int (((row * 7) + (col * 13)) mod 257)
                     /. 257.))
        in
        let result = Batch.evaluate ~backend:Batch.Scalar parallel p points in
        if Iarray.length result <> Iarray.length points then
          failwith "Batch length";
        for i = 0 to Iarray.length points - 1 do
          let expected =
            F.to_float (Problem.evaluate p (Iarray.get points i))
          in
          let actual = F.to_float (Iarray.get result i) in
          if Int64.bits_of_float expected <> Int64.bits_of_float actual then
            failwith (Printf.sprintf "%s: batch mismatch at %d" d.id i)
        done;
        if Iarray.length (Batch.evaluate parallel p [::]) <> 0 then
          failwith "Empty batch";
        loop rest
  in
  loop Problem.catalogue [@nontail]

let () =
  reference_cases ();
  analytic_cases ();
  validation_cases ();
  batch_cases Parallel.sequential;
  Parallel_scheduler.with_parallel ~max_workers:4 batch_cases;
  let p = Problem.create "sphere" in
  let malformed =
    Iarray.init 257 (fun i -> if i = 128 then [: 1. :] else [: 1.; 2. :])
  in
  rejects "sequential batch validation" (fun () ->
      Iarray.length (Batch.evaluate Parallel.sequential p malformed));
  rejects "parallel batch validation" (fun () ->
      Parallel_scheduler.with_parallel ~max_workers:4 (fun parallel ->
          Iarray.length (Batch.evaluate parallel p malformed)));
  print_endline
    "Analytic, validation, and sequential/multicore batch checks passed."
