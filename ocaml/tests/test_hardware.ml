open Baux_hardware
module D = Distance
module E = Experiment

let check condition message = if not condition then failwith message

let rejects f =
  match f () with
  | _ -> failwith "accepted invalid hardware configuration"
  | exception Invalid_argument _ -> ()

let protocol_checks pipeline =
  let open Hardcaml in
  let d = { D.dimension = 3; bits = 8; lanes = 2; pipeline } in
  let sim = Cyclesim.create (D.create d) in
  let set name width value =
    Cyclesim.in_port sim name := Bits.of_int_trunc ~width value
  in
  let ready = Cyclesim.out_port sim "valid_out" in
  let result = Cyclesim.out_port sim "distance" in
  let tick () = Cyclesim.cycle sim in
  Cyclesim.reset sim;
  (* Start, then abort a vector by clearing all control and data registers. *)
  set "valid" 1 1;
  set "first" 1 1;
  set "last" 1 0;
  set "x0" 8 64;
  set "y0" 8 (-64);
  set "x1" 8 64;
  set "y1" 8 (-64);
  tick ();
  set "clear" 1 1;
  tick ();
  check (Bits.to_unsigned_int !ready = 0) "clear leaked valid";
  set "clear" 1 0;
  set "valid" 1 0;
  tick ();
  tick ();
  check (Bits.to_unsigned_int !ready = 0) "aborted input produced output";
  (* Send three unit coordinates, padded with an inactive zero lane. *)
  set "valid" 1 1;
  set "first" 1 1;
  set "x0" 8 64;
  set "y0" 8 0;
  set "x1" 8 64;
  set "y1" 8 0;
  tick ();
  set "first" 1 0;
  set "last" 1 1;
  set "x1" 8 0;
  tick ();
  if pipeline then (
    set "valid" 1 0;
    tick ());
  check (Bits.to_unsigned_int !ready = 1) "missing valid after reset";
  check
    (Bits.to_unsigned_int !result = 3 * 64 * 64)
    "reset retained stale accumulator"

let () =
  let configurations = ref 0 and checked = ref 0 in
  List.iter
    (fun dimension ->
      List.iter
        (fun bits ->
          List.iter
            (fun lanes ->
              if lanes <= dimension then
                List.iter
                  (fun pipeline ->
                    let d = { D.dimension; bits; lanes; pipeline } in
                    let circuit = D.create d in
                    let samples = D.corpus ~dimension ~samples:32 ~seed:718 in
                    let outputs, cycles = D.simulate d circuit samples in
                    let stalled, _ =
                      D.simulate ~bubbles:true d circuit samples
                    in
                    check
                      (List.map (fun (o : D.observation) -> o.raw) outputs
                      = List.map (fun (o : D.observation) -> o.raw) stalled)
                      "bubbles changed result order or values";
                    let chunks = (dimension + lanes - 1) / lanes in
                    check
                      (cycles = (32 * chunks) + if pipeline then 1 else 0)
                      "incorrect cycle accounting";
                    check
                      ((List.nth outputs 1).value = 4. *. float_of_int dimension)
                      "extreme distance overflowed";
                    check ((List.nth outputs 3).raw = 0) "equal inputs not zero";
                    check
                      (D.resource_bits circuit "signed_multipliers"
                      = lanes * 2 * (bits + 1))
                      "unexpected multiplier graph metric";
                    let rtl = D.rtl circuit in
                    check (String.length rtl > 100) "missing generated RTL";
                    incr configurations;
                    checked := !checked + 64)
                  [ false; true ])
            [ 1; 2; 3; 8; 32 ])
        [ 4; 8; 12; 20 ])
    [ 1; 3; 17; 64; 1024 ];
  List.iter protocol_checks [ false; true ];
  let d = { D.dimension = 1; bits = 4; lanes = 1; pipeline = false } in
  check
    (D.quantize d 0.125 = 1 && D.quantize d (-0.125) = -1)
    "rounding contract";
  List.iter
    (fun x -> rejects (fun () -> D.quantize d x))
    [ nan; infinity; neg_infinity; 1.001; -1.001 ];
  rejects (fun () -> D.create { d with bits = 21 });
  rejects (fun () -> D.create { d with lanes = 2 });
  rejects (fun () -> D.create { d with dimension = 0 });
  let config = E.load "../hardcaml.toml" in
  let text = E.to_string config in
  check (E.of_string text = config) "TOML roundtrip";
  List.iter
    (fun (before, after) ->
      let broken = Str.global_replace (Str.regexp_string before) after text in
      check (broken <> text) "invalid fixture replacement";
      rejects (fun () -> E.of_string broken))
    [
      ("version = 1", "version = 2");
      ("dimension = 64", "dimension = 0");
      ("samples = 128", "samples = 0");
      ("bits = [8, 12, 16, 20]", "bits = [8, 8]");
      ("lanes = [1, 2, 4, 8, 16, 32]", "lanes = [65]");
      ("pipelines = [false, true]", "pipelines = []");
      ("max_error = 0.001", "max_error = nan");
      ("max_multiplier_bits = 256", "max_multiplier_bits = 0");
    ];
  rejects (fun () -> E.of_string (text ^ "\nunknown = 1\n"));
  let directory = Filename.temp_file "baux-hardware-test-" "" in
  Sys.remove directory;
  Unix.mkdir directory 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter
        (fun file -> Sys.remove (Filename.concat directory file))
        (Sys.readdir directory);
      Unix.rmdir directory)
    (fun () ->
      let open Yojson.Basic.Util in
      let small =
        {
          config with
          samples = 8;
          bits = [ 8; 12 ];
          lanes = [ 1; 8 ];
          pipelines = [ false ];
        }
      in
      let design =
        { D.dimension = 64; bits = 12; lanes = 8; pipeline = false }
      in
      let feasible = E.evaluate small design directory in
      check
        (feasible |> member "feasible" |> to_bool)
        "expected feasible design";
      check
        (feasible |> member "reward" |> to_float > 0.)
        "feasible reward sign";
      let limited = { small with max_multiplier_bits = 1 } in
      let infeasible = E.evaluate limited design directory in
      check
        (not (infeasible |> member "feasible" |> to_bool))
        "ignored resource constraint";
      check
        (infeasible |> member "reward" |> to_float < 0.)
        "infeasible reward sign";
      let inaccurate =
        E.evaluate
          { small with max_error = 1e-12 }
          { design with bits = 8 } directory
      in
      check
        (not (inaccurate |> member "feasible" |> to_bool))
        "ignored accuracy constraint";
      rejects (fun () -> E.evaluate small { design with bits = 16 } directory);
      E.run limited directory;
      let summary =
        Yojson.Basic.from_file (Filename.concat directory "summary.json")
      in
      check
        (summary |> member "feasible_designs" |> to_int = 0)
        "incorrect empty feasible set";
      check (summary |> member "best" = `Null) "fabricated feasible winner";
      rejects (fun () -> E.run limited directory));
  Printf.printf
    "Hardcaml: %d designs, %d checked outputs; bubbles, reset, tails, \
     overflow, TOML passed\n"
    !configurations !checked
