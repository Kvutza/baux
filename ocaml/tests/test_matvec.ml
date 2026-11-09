open Baux_hardware
module M = Matvec
module S = Matvec_sim

let check condition message = if not condition then failwith message

let fast =
  { S.ready_period = 1; ready_cycles = 1; initial_stall = 0; load_gap = 0 }

let blocked =
  { S.ready_period = 7; ready_cycles = 1; initial_stall = 200; load_gap = 1 }

let run d schedule =
  let rng = Random.State.make [| d.M.rows; d.cols |] in
  let values n =
    Array.init n (fun i ->
        match i mod 7 with
        | 0 -> -1.
        | 1 -> 1.
        | _ -> Random.State.float rng 2. -. 1.)
  in
  let sim = Hardcaml.Cyclesim.create (M.create d) in
  let weights = values (d.rows * d.cols) in
  let inputs = Array.init 3 (fun _ -> values d.cols) in
  let first, c = S.simulate d sim schedule weights inputs in
  let again, repeated = S.simulate d sim schedule weights inputs in
  check
    (first = again && c = repeated)
    "reused simulator changed outputs or cycles";
  let flipped, _ =
    S.simulate d sim schedule (Array.map ( ~-. ) weights) inputs
  in
  Array.iteri
    (fun i output ->
      Array.iteri
        (fun j value ->
          check (value = -.first.(i).(j)) "runtime weight reload failed")
        output)
    flipped;
  check (c.queue_peak <= d.queue) "FIFO capacity exceeded";
  check (c.reads = 3 * d.rows * M.tiles d * M.passes d) "read accounting";
  check
    (c.bank_wait = 3 * d.rows * M.tiles d * (M.passes d - 1))
    "bank wait accounting";
  if schedule = fast then
    check
      (c.compute
      = 3
        * (2
          + d.rows
            * ((M.tiles d * (M.passes d + 1 + if d.pipeline then 1 else 0)) + 1)
          ))
      "unstalled cycle model mismatch";
  c

let () =
  let count = ref 0 in
  List.iter
    (fun (rows, cols) ->
      List.iter
        (fun bits ->
          List.iter
            (fun lanes ->
              List.iter
                (fun banks ->
                  List.iter
                    (fun pipeline ->
                      List.iter
                        (fun queue ->
                          let d =
                            {
                              M.rows;
                              cols;
                              bits;
                              lanes;
                              banks;
                              pipeline;
                              queue;
                            }
                          in
                          ignore (run d fast);
                          ignore (run d blocked);
                          incr count)
                        [ 1; 4 ])
                    [ false; true ])
                [ 1; 4; 8 ])
            [ 1; 4; 8 ])
        [ 4; 12; 20 ])
    [ (1, 1); (3, 7); (8, 16) ];
  let base =
    {
      M.rows = 8;
      cols = 16;
      bits = 12;
      lanes = 8;
      banks = 1;
      pipeline = false;
      queue = 1;
    }
  in
  let narrow = run base fast and wide = run { base with banks = 8 } fast in
  check
    (narrow.compute > wide.compute && narrow.weights > wide.weights)
    "banks must affect cycles";
  let stalled = run { base with banks = 8 } blocked in
  check
    (stalled.fifo_stall > 0 && stalled.output_stall > 0)
    "backpressure not exercised";
  Printf.printf
    "Matrix accelerator: %d designs, signed/tail/runtime \
     reload/cache/backpressure parity passed\n\
     %!"
    !count

let protocol_checks () =
  let open Hardcaml in
  let d =
    {
      M.rows = 3;
      cols = 7;
      bits = 8;
      lanes = 8;
      banks = 4;
      pipeline = true;
      queue = 1;
    }
  in
  let sim = Cyclesim.create (M.create d) in
  let set key value =
    let port = Cyclesim.in_port sim key in
    port := Bits.of_int_trunc ~width:(Bits.width !port) value
  in
  let get key = Bits.to_int_trunc !(Cyclesim.out_port sim key) in
  let tick () = Cyclesim.cycle sim in
  List.iter
    (fun (_, p) -> p := Bits.zero (Bits.width !p))
    (Cyclesim.in_ports sim);
  set "clear" 1;
  tick ();
  set "clear" 0;
  set "w_row" 3;
  set "w_valid" 1;
  tick ();
  check (get "w_ready" = 0 && get "start_ready" = 0) "invalid row accepted";
  set "w_valid" 0;
  set "start" 1;
  tick ();
  set "start" 0;
  set "w_valid" 1;
  set "x_valid" 1;
  tick ();
  check (get "w_ready" = 0 && get "x_ready" = 0) "busy RAM accepted programming";
  set "w_valid" 0;
  set "x_valid" 0;
  for _ = 1 to 30 do
    tick ()
  done;
  check (get "out_valid" = 1 && get "fifo_stall" = 1) "FIFO did not fill";
  set "clear" 1;
  tick ();
  set "clear" 0;
  tick ();
  check
    (get "out_valid" = 0 && get "busy" = 0)
    "clear did not abort queued work";
  for _ = 1 to 30 do
    tick ()
  done;
  check (get "out_valid" = 0) "aborted output escaped";
  ignore (S.simulate d sim fast (Array.make 21 1.) [| Array.make 7 (-1.) |])

let worker_checks () =
  let open Yojson.Basic.Util in
  let evaluate = Matvec_search.create 1 in
  let request split seed =
    `Assoc
      [
        ("version", `Int 1);
        ("rows", `Int 3);
        ("cols", `Int 7);
        ("bits", `Int 8);
        ("lanes", `Int 8);
        ("banks", `Int 4);
        ("pipeline", `Bool true);
        ("queue", `Int 1);
        ("weights", `List (List.init 21 (fun _ -> `Float 0.25)));
        ("samples", `Int 4);
        ("target_seed", `Int seed);
        ("data_seed", `Int 123);
        ("split", `String split);
        ("ready_period", `Int 7);
        ("ready_cycles", `Int 1);
        ("initial_stall", `Int 100);
        ("load_gap", `Int 1);
      ]
  in
  let a = evaluate (request "train" 9) in
  let b = evaluate (request "train" 9) in
  let v = evaluate (request "validation" 9) in
  let t = evaluate (request "test" 9) in
  let changed = evaluate (request "train" 10) in
  check (not (a |> member "cache_hit" |> to_bool)) "cold matrix cache hit";
  check (b |> member "cache_hit" |> to_bool) "matrix cache miss on reuse";
  check
    (member "loss" a = member "loss" b)
    "cached evaluation not deterministic";
  check
    (member "loss" a <> member "loss" v && member "loss" v <> member "loss" t)
    "training, validation and test streams overlap";
  check (member "loss" a <> member "loss" changed) "teacher change ignored";
  List.iter
    (fun value ->
      check
        (member "cycles" a = member "cycles" value)
        "query cycles depend on data split")
    [ v; t; changed ];
  let rtl =
    match request "train" 9 with
    | `Assoc fields -> evaluate (`Assoc (("op", `String "rtl") :: fields))
    | _ -> failwith "fixture"
  in
  check
    (String.length (rtl |> member "rtl" |> to_string) > 1000)
    "RTL export empty"

let () =
  protocol_checks ();
  worker_checks ();
  let d =
    {
      M.rows = 256;
      cols = 4096;
      bits = 20;
      lanes = 32;
      banks = 32;
      pipeline = true;
      queue = 32;
    }
  in
  let _, profile = Search.measure (fun () -> ignore (run d fast)) in
  Printf.printf "Million-weight matrix: %s\n%!" (Yojson.Basic.to_string profile)

let () =
  let count = ref 0 in
  let _, profile =
    Search.measure (fun () ->
        List.iter
          (fun bits ->
            List.iter
              (fun lanes ->
                List.iter
                  (fun banks ->
                    List.iter
                      (fun pipeline ->
                        List.iter
                          (fun queue ->
                            ignore
                              (run
                                 {
                                   M.rows = 3;
                                   cols = 7;
                                   bits;
                                   lanes;
                                   banks;
                                   pipeline;
                                   queue;
                                 }
                                 blocked);
                            incr count)
                          [ 1; 2; 4; 8 ])
                      [ false; true ])
                  [ 1; 2; 4; 8; 16 ])
              [ 1; 2; 4; 8; 16 ])
          [ 4; 8; 12; 16; 20 ])
  in
  Printf.printf "Matrix grid: %d variants constructed and verified: %s\n%!"
    !count
    (Yojson.Basic.to_string profile)
