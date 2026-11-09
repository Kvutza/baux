module D = Distance
open Yojson.Basic.Util

let clock = Core.Or_error.ok_exn Core_unix.Clock.gettime
let now () = Core.Int63.to_float (clock Core_unix.Clock.Monotonic) *. 1e-9

let measure action =
  let before = Gc.quick_stat () in
  let start = now () in
  let value = action () in
  let seconds = now () -. start in
  let after = Gc.quick_stat () in
  ( value,
    `Assoc
      [
        ("seconds", `Float seconds);
        ( "allocated_words",
          `Float
            (after.minor_words +. after.major_words -. after.promoted_words
           -. before.minor_words -. before.major_words +. before.promoted_words
            ) );
        ( "minor_collections",
          `Int (after.minor_collections - before.minor_collections) );
        ( "major_collections",
          `Int (after.major_collections - before.major_collections) );
        ("heap_words_after", `Int after.heap_words);
      ] )

let target ~dimension ~seed =
  let rng = Random.State.make [| seed |] in
  Array.init dimension (fun _ -> Random.State.float rng 1.5 -. 0.75)

let create capacity =
  if capacity < 1 || capacity > 1024 then
    invalid_arg "cache capacity must be in 1..1024";
  let entries = ref [] in
  fun request ->
    let integer key = request |> member key |> to_int in
    if integer "version" <> 1 then invalid_arg "unsupported worker protocol";
    let d =
      {
        D.dimension = integer "dimension";
        bits = integer "bits";
        lanes = integer "lanes";
        pipeline = request |> member "pipeline" |> to_bool;
      }
    in
    D.validate d;
    let x =
      request |> member "weights" |> to_list |> List.map to_number
      |> Array.of_list
    in
    if Array.length x <> d.dimension then
      invalid_arg "weight dimension mismatch";
    Array.iter (fun x -> ignore (D.quantize d x)) x;
    let samples, corpus_profile =
      measure (fun () ->
          [|
            {
              D.x;
              y = target ~dimension:d.dimension ~seed:(integer "target_seed");
            };
          |])
    in
    let key = D.name d in
    let hit = List.assoc_opt key !entries in
    let (circuit, sim), build_profile =
      measure (fun () ->
          match hit with
          | Some entry -> entry
          | None ->
              let circuit = D.create d in
              (circuit, Hardcaml.Cyclesim.create circuit))
    in
    let rec take n = function
      | _ when n = 0 -> []
      | [] -> []
      | x :: xs -> x :: take (n - 1) xs
    in
    entries :=
      (key, (circuit, sim))
      :: take (capacity - 1) (List.remove_assoc key !entries);
    let (observations, cycles), simulation_profile =
      measure (fun () -> D.simulate ~sim d circuit samples)
    in
    let o =
      match observations with [ o ] -> o | _ -> failwith "expected one output"
    in
    `Assoc
      [
        ("version", `Int 1);
        ("design", `String key);
        ("loss", `Float (o.value /. float_of_int d.dimension));
        ("reference_loss", `Float (o.reference /. float_of_int d.dimension));
        ("cycles", `Int cycles);
        ("reset_and_check_cycles", `Int 3);
        ("cache_hit", `Bool (Option.is_some hit));
        ("cache_entries", `Int (List.length !entries));
        ("corpus", corpus_profile);
        ("construction", build_profile);
        ("simulation_and_reference_check", simulation_profile);
      ]

let serve_evaluator evaluate =
  Printf.printf "{\"ready\":true,\"version\":1}\n%!";
  let rec loop () =
    match input_line stdin with
    | line ->
        let response =
          try evaluate (Yojson.Basic.from_string line) with
          | Invalid_argument message
          | Failure message
          | Yojson.Json_error message ->
              `Assoc [ ("error", `String message) ]
          | Type_error (message, _) -> `Assoc [ ("error", `String message) ]
        in
        print_endline (Yojson.Basic.to_string response);
        flush stdout;
        loop ()
    | exception End_of_file -> ()
  in
  loop ()

let serve capacity = serve_evaluator (create capacity)
