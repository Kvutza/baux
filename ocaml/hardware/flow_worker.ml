module M = Flow_model
module B = Flow_backend
module C = Flow_core
open Yojson.Basic.Util

let model j =
  let n k = j |> member k |> to_int in
  let c =
    {
      M.levels = n "levels";
      steps = n "steps";
      width = n "width";
      depth = n "depth";
      heads = n "heads";
      expansion = n "expansion";
    }
  in
  M.validate c;
  c

let model_json c =
  `Assoc
    [
      ("levels", `Int c.M.levels);
      ("steps", `Int c.steps);
      ("width", `Int c.width);
      ("depth", `Int c.depth);
      ("heads", `Int c.heads);
      ("expansion", `Int c.expansion);
    ]

let design request =
  let n k = request |> member k |> to_int in
  {
    C.bits = n "bits";
    lanes = n "lanes";
    banks = n "banks";
    queue = n "queue";
    pipeline = request |> member "pipeline" |> to_bool;
  }

let array j = j |> to_list |> List.map to_number |> Array.of_list

let actual_weights c request =
  let initial =
    match request |> member "checkpoint" with
    | `String filename ->
        let document = Yojson.Basic.from_file filename in
        if
          member "version" document <> `Int 1
          || model (member "model" document) <> c
        then invalid_arg "FlowLOB checkpoint shape/version mismatch";
        array (member "weights" document)
    | _ -> M.initial c (request |> member "initial_seed" |> to_int)
  in
  let coordinates = array (member "weights" request) in
  let radius = request |> member "radius" |> to_number in
  if
    Array.length initial <> M.parameters c
    || Array.length coordinates <> Array.length initial
    || (not (Float.is_finite radius))
    || radius <= 0. || radius > 1.
  then invalid_arg "invalid FlowLOB parameter coordinates";
  Array.mapi
    (fun i x ->
      if (not (Float.is_finite x)) || abs_float x > 1. then
        invalid_arg "FlowLOB search coordinate outside [-1,1]";
      initial.(i) +. (radius *. x))
    coordinates

let corpus c request =
  let n k = request |> member k |> to_int in
  let split = request |> member "split" |> to_string in
  let split_id =
    match split with
    | "train" -> 0
    | "validation" -> 1
    | "test" -> 2
    | _ -> invalid_arg "unknown split"
  in
  let count = n "samples" in
  if count < 1 || count > 128 then invalid_arg "FlowLOB samples must be 1..128";
  Array.init count (fun i ->
      let seed = n "data_seed" + (7919 * split_id) + (104729 * i) in
      let example =
        match request |> member "dataset" with
        | `String filename ->
            Flow_data.load c filename split ((n "window" * count) + i)
        | _ -> Flow_data.synthetic c ~seed ~regime:(n "target_seed")
      in
      (example, seed))

let matrix_json x =
  `List
    (Array.to_list
       (Array.map
          (fun row -> `List (Array.to_list (Array.map (fun x -> `Float x) row)))
          x))

let raw_books c example windows =
  let anchor = ref example.Flow_data.previous_mid in
  let number x = if Float.is_finite x then `Float x else `Null in
  `List
    (Array.to_list
       (Array.map
          (fun window ->
            let books =
              Flow_data.decode c ~tick:example.tick ~previous:!anchor window
            in
            anchor := Flow_data.mid books.(Array.length books - 1);
            `List
              (Array.to_list
                 (Array.map
                    (fun book ->
                      `Assoc
                        [
                          ( "prices",
                            `List
                              (Array.to_list
                                 (Array.map number book.Flow_data.prices)) );
                          ( "volumes",
                            `List
                              (Array.to_list (Array.map number book.volumes)) );
                        ])
                    books)))
          windows))

let max_error a b =
  let result = ref 0. in
  Array.iteri
    (fun i x ->
      Array.iteri
        (fun j x -> result := max !result (abs_float (x -. b.(i).(j))))
        x)
    a;
  !result

let create capacity =
  if capacity < 1 || capacity > 1024 then invalid_arg "invalid cache capacity";
  let entries = ref [] in
  fun request ->
    if member "version" request <> `Int 1 then
      invalid_arg "unknown FlowLOB protocol";
    let c = model (member "model" request) and d = design request in
    C.validate d;
    if member "op" request = `String "plan" then
      Flow_plan.profile c d
        (request |> member "ready_period" |> to_int)
        (request |> member "samples" |> to_int)
    else if member "op" request = `String "rtl" then
      `Assoc
        [
          ("rtl", `String (Distance.rtl (C.create d)));
          ("design", `String (C.name d));
        ]
    else
      let backend = B.mode (request |> member "backend" |> to_string) in
      if backend = B.Float then
        invalid_arg "benchmark backend must be integer or rtl";
      let period = request |> member "ready_period" |> to_int in
      let key =
        C.name d ^ ":" ^ string_of_int period ^ ":"
        ^ (request |> member "backend" |> to_string)
      in
      let hit = List.assoc_opt key !entries in
      let machine, construction =
        Search.measure (fun () ->
            match hit with
            | Some value -> value
            | None -> B.create ~period backend d)
      in
      let rec take n = function
        | _ when n = 0 -> []
        | [] -> []
        | x :: xs -> x :: take (n - 1) xs
      in
      entries :=
        (key, machine) :: take (capacity - 1) (List.remove_assoc key !entries);
      B.reset_stats machine;
      let (weights, examples), corpus_profile =
        Search.measure (fun () -> (actual_weights c request, corpus c request))
      in
      let reference = B.create B.Float d in
      let generation = member "op" request = `String "generate" in
      let compute machine =
        if generation then (
          let example, seed = examples.(0) in
          let count =
            match member "rollout_windows" request with
            | `Null -> 1
            | j -> to_int j
          in
          if count < 1 || count > 16 then
            invalid_arg "invalid rollout window count";
          let condition = ref example.condition and total = ref 0 in
          let results =
            Array.init count (fun window ->
                let result, nfe =
                  Flow_sampling.generate machine c weights ~condition:!condition
                    ~noise:(Flow_data.noise c (seed + (window * 1009)))
                    ~method_:(request |> member "solver" |> to_string)
                    ~steps:(request |> member "solver_steps" |> to_int)
                    ~guidance:(request |> member "guidance" |> to_number)
                in
                total := !total + nfe;
                condition :=
                  Array.mapi
                    (fun i old ->
                      let next = Array.copy old in
                      next.(0) <- result.(i).(0);
                      next.(1) <- result.(i).(1);
                      next)
                    !condition;
                result)
          in
          (Flow_data.mse results.(0) example.target, results, !total))
        else
          let values =
            Array.map
              (fun (example, seed) ->
                Flow_sampling.training machine c weights example seed
                  ~dropout:(member "split" request = `String "train"))
              examples
          in
          ( Array.fold_left (fun s (loss, _) -> s +. loss) 0. values
            /. float_of_int (Array.length values),
            Array.map snd values,
            Array.length values )
      in
      let (loss, outputs, nfe), simulation =
        Search.measure (fun () -> compute machine)
      in
      let (reference_loss, reference_outputs, _), reference_profile =
        Search.measure (fun () -> compute reference)
      in
      let upload = Matvec.ceil_div (Array.length weights) d.banks in
      let io =
        Matvec.ceil_div (M.tokens c * 9 * Array.length examples) d.banks
      in
      let cycles = machine.stats.cycles + upload + io in
      let common =
        [
          ("version", `Int 1);
          ("design", `String (C.name d));
          ("model", model_json c);
          ("parameters", `Int (Array.length weights));
          ( "representation",
            `String
              "time-major outer-ask to outer-bid; price ticks /8; log1p volume \
               /12; fixed explicit scales" );
          ( "execution",
            `String
              "host-orchestrated streaming RTL kernels with external \
               bank-limited SRAM model" );
          ("loss", `Float loss);
          ("reference_loss", `Float reference_loss);
          ("cycles", `Int cycles);
          ("reset_and_check_cycles", `Int 0);
          ("weight_load_cycles", `Int upload);
          ("input_load_cycles", `Int io);
          ("kernel_counters", B.json machine);
          ("neural_function_evaluations", `Int nfe);
          ( "max_reference_error",
            `Float
              (Array.fold_left max 0.
                 (Array.mapi
                    (fun i x -> max_error x reference_outputs.(i))
                    outputs)) );
          ("cache_hit", `Bool (Option.is_some hit));
          ("cache_entries", `Int (List.length !entries));
          ("corpus", corpus_profile);
          ("construction", construction);
          ("simulation_and_reference_check", simulation);
          ("float_reference", reference_profile);
          ( "data_source",
            `String
              (match member "dataset" request with
              | `String p -> p
              | _ -> "synthetic LOB fixture; no market calibration") );
        ]
      in
      if not generation then `Assoc common
      else
        `Assoc
          (common
          @ [
              ("generated", matrix_json outputs.(0));
              ("rollout", `List (Array.to_list (Array.map matrix_json outputs)));
              ("raw_books", raw_books c (fst examples.(0)) outputs);
              ( "book_projection",
                `String
                  "none; invalid generated books retained and reported; \
                   nonfinite decoded volumes are null" );
              ("model_profile", M.profile c);
              ( "metrics",
                Flow_data.metrics c outputs.(0) (fst examples.(0)).target );
              ( "checkpoint",
                `Assoc
                  [
                    ("version", `Int 1);
                    ("model", model_json c);
                    ( "weights",
                      `List
                        (Array.to_list (Array.map (fun x -> `Float x) weights))
                    );
                    ( "provenance",
                      `String
                        "paper-based architecture; BAUX coordinate adaptation"
                    );
                  ] );
            ])

let serve capacity = Search.serve_evaluator (create capacity)
