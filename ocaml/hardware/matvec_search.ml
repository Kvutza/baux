module M = Matvec
module S = Matvec_sim
open Yojson.Basic.Util

let design request =
  let int key = request |> member key |> to_int in
  {
    M.rows = int "rows";
    cols = int "cols";
    bits = int "bits";
    lanes = int "lanes";
    banks = int "banks";
    queue = int "queue";
    pipeline = request |> member "pipeline" |> to_bool;
  }

let schedule request =
  let int key = request |> member key |> to_int in
  {
    S.ready_period = int "ready_period";
    ready_cycles = int "ready_cycles";
    initial_stall = int "initial_stall";
    load_gap = int "load_gap";
  }

let corpus d request =
  let int key = request |> member key |> to_int in
  let split =
    match request |> member "split" |> to_string with
    | "train" -> 0
    | "validation" -> 1
    | "test" -> 2
    | _ -> invalid_arg "unknown corpus split"
  in
  let count = int "samples" in
  if count < 1 || count > 4096 then invalid_arg "samples must be in 1..4096";
  let teacher =
    Search.target ~dimension:(d.M.rows * d.cols) ~seed:(int "target_seed")
  in
  let rng = Random.State.make [| int "data_seed"; split; d.rows; d.cols |] in
  let inputs =
    Array.init count (fun _ ->
        Array.init d.cols (fun _ -> Random.State.float rng 2. -. 1.))
  in
  (inputs, Array.map (S.reference d teacher) inputs)

let losses d weights inputs targets outputs =
  let loss = ref 0. and reference_loss = ref 0. and error = ref 0. in
  Array.iteri
    (fun i input ->
      let reference = S.reference d weights input in
      Array.iteri
        (fun row value ->
          let target = targets.(i).(row) in
          loss := !loss +. ((value -. target) ** 2.);
          reference_loss :=
            !reference_loss +. ((reference.(row) -. target) ** 2.);
          error := max !error (Float.abs (value -. reference.(row))))
        outputs.(i))
    inputs;
  let divisor = float_of_int (Array.length inputs * d.rows * d.cols) in
  [
    ("loss", `Float (!loss /. divisor));
    ("reference_loss", `Float (!reference_loss /. divisor));
    ("max_quantization_error", `Float (!error /. sqrt (float_of_int d.cols)));
  ]

let create capacity =
  if capacity < 1 || capacity > 1024 then invalid_arg "invalid cache capacity";
  let entries = ref [] in
  fun request ->
    if request |> member "version" |> to_int <> 1 then
      invalid_arg "unsupported matrix worker protocol";
    let d = design request in
    M.validate d;
    let key = M.name d in
    let hit = List.assoc_opt key !entries in
    let (circuit, sim), construction =
      Search.measure (fun () ->
          match hit with
          | Some entry -> entry
          | None ->
              let c = M.create d in
              (c, Hardcaml.Cyclesim.create c))
    in
    let rec take n = function
      | _ when n = 0 -> []
      | [] -> []
      | x :: xs -> x :: take (n - 1) xs
    in
    entries :=
      (key, (circuit, sim))
      :: take (capacity - 1) (List.remove_assoc key !entries);
    if request |> member "op" = `String "rtl" then
      `Assoc [ ("rtl", `String (M.rtl circuit)); ("design", `String key) ]
    else
      let weights =
        request |> member "weights" |> to_list |> List.map to_number
        |> Array.of_list
      in
      let (inputs, targets), corpus_profile =
        Search.measure (fun () -> corpus d request)
      in
      let (outputs, counts), simulation =
        Search.measure (fun () ->
            S.simulate d sim (schedule request) weights inputs)
      in
      `Assoc
        (losses d weights inputs targets outputs
        @ [
            ("version", `Int 1);
            ("design", `String key);
            ("cycles", `Int (S.cycles counts));
            ("cycle_breakdown", S.to_json counts);
            ("reset_and_check_cycles", `Int 0);
            ("cache_hit", `Bool (Option.is_some hit));
            ("cache_entries", `Int (List.length !entries));
            ("corpus", corpus_profile);
            ("construction", construction);
            ("simulation_and_reference_check", simulation);
          ])

let serve capacity = Search.serve_evaluator (create capacity)
