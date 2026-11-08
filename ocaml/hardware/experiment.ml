module D = Distance

type t = {
  dimension : int;
  bits : int list;
  lanes : int list;
  pipelines : bool list;
  samples : int;
  seed : int;
  max_error : float;
  max_multiplier_bits : int;
}

let decode document =
  let open Otoml in
  let fail s = invalid_arg ("Hardcaml TOML: " ^ s) in
  let fields =
    match document with TomlTable xs -> xs | _ -> fail "expected table"
  in
  let allowed =
    [
      "version";
      "experiment";
      "dimension";
      "bits";
      "lanes";
      "pipelines";
      "samples";
      "seed";
      "max_error";
      "max_multiplier_bits";
    ]
  in
  List.iter
    (fun (k, _) -> if not (List.mem k allowed) then fail ("unknown field " ^ k))
    fields;
  let get k =
    match List.assoc_opt k fields with
    | Some x -> x
    | None -> fail ("missing " ^ k)
  in
  let integer = function TomlInteger n -> n | _ -> fail "expected integer" in
  let boolean = function TomlBoolean b -> b | _ -> fail "expected boolean" in
  let array f key =
    let xs =
      match get key with
      | TomlArray xs -> List.map f xs
      | _ -> fail (key ^ " must be array")
    in
    if xs = [] || List.length xs <> List.length (List.sort_uniq compare xs) then
      fail (key ^ " must be nonempty and unique");
    xs
  in
  if integer (get "version") <> 1 then fail "unsupported version";
  (match get "experiment" with
  | TomlString "hardcaml-distance" -> ()
  | _ -> fail "unknown experiment");
  let max_error =
    match get "max_error" with
    | TomlFloat x -> x
    | TomlInteger n -> float_of_int n
    | _ -> fail "expected max_error number"
  in
  let t =
    {
      dimension = integer (get "dimension");
      bits = array integer "bits";
      lanes = array integer "lanes";
      pipelines = array boolean "pipelines";
      samples = integer (get "samples");
      seed = integer (get "seed");
      max_error;
      max_multiplier_bits = integer (get "max_multiplier_bits");
    }
  in
  if t.samples < 8 || t.samples > 10000 then fail "samples must be in 8..10000";
  if (not (Float.is_finite t.max_error)) || t.max_error <= 0. then
    fail "max_error must be finite and positive";
  if t.max_multiplier_bits <= 0 then fail "max_multiplier_bits must be positive";
  List.iter
    (fun bits ->
      List.iter
        (fun lanes ->
          D.validate { dimension = t.dimension; bits; lanes; pipeline = false })
        t.lanes)
    t.bits;
  t

let of_string text =
  match Otoml.Parser.from_string_result text with
  | Ok doc -> decode doc
  | Error message -> invalid_arg message

let load path =
  match Otoml.Parser.from_file_result path with
  | Ok doc -> decode doc
  | Error message -> invalid_arg message

let to_string t =
  let array f xs = "[" ^ String.concat ", " (List.map f xs) ^ "]" in
  Printf.sprintf
    "version = 1\n\
     experiment = \"hardcaml-distance\"\n\
     dimension = %d\n\
     bits = %s\n\
     lanes = %s\n\
     pipelines = %s\n\
     samples = %d\n\
     seed = %d\n\
     max_error = %.17g\n\
     max_multiplier_bits = %d\n"
    t.dimension
    (array string_of_int t.bits)
    (array string_of_int t.lanes)
    (array string_of_bool t.pipelines)
    t.samples t.seed t.max_error t.max_multiplier_bits

let write path text =
  let out = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out out)
    (fun () -> output_string out text)

let designs t =
  List.concat_map
    (fun bits ->
      List.concat_map
        (fun lanes ->
          List.map
            (fun pipeline ->
              { D.dimension = t.dimension; bits; lanes; pipeline })
            t.pipelines)
        t.lanes)
    t.bits

let space t =
  `Assoc
    [
      ("schema", `String "baux.hardcaml-distance.space.v1");
      ("dimension", `Int t.dimension);
      ("bits", `List (List.map (fun x -> `Int x) t.bits));
      ("lanes", `List (List.map (fun x -> `Int x) t.lanes));
      ("pipelines", `List (List.map (fun x -> `Bool x) t.pipelines));
      ("direction", `String "maximize");
      ("objective", `String "reward");
      ("max_error", `Float t.max_error);
      ("max_multiplier_bits", `Int t.max_multiplier_bits);
      ("samples", `Int t.samples);
      ("seed", `Int t.seed);
      ("deterministic", `Bool true);
    ]

let evaluate t d directory =
  D.validate d;
  if d.D.dimension <> t.dimension then
    invalid_arg "design dimension differs from workload";
  if
    not
      (List.mem d.bits t.bits && List.mem d.lanes t.lanes
      && List.mem d.pipeline t.pipelines)
  then invalid_arg "design is outside the configured search space";
  let circuit = D.create d in
  let samples =
    D.corpus ~dimension:t.dimension ~samples:t.samples ~seed:t.seed
  in
  let observations, cycles, error = D.evaluate d samples circuit in
  let multiplier_bits = D.resource_bits circuit "signed_multipliers" in
  let register_bits = D.resource_bits circuit "registers" in
  let feasible =
    error <= t.max_error && multiplier_bits <= t.max_multiplier_bits
  in
  let cycles_per_vector = float_of_int cycles /. float_of_int t.samples in
  (* Feasible rewards are in (0,1]; infeasible rewards are negative. *)
  let reward =
    if feasible then 1. /. cycles_per_vector
    else
      -.(max 0. ((error /. t.max_error) -. 1.)
        +. max 0.
             (float_of_int multiplier_bits
              /. float_of_int t.max_multiplier_bits
             -. 1.))
  in
  let design_id = D.name d in
  let prefix = Filename.concat directory design_id in
  write (prefix ^ ".v") (D.rtl circuit);
  write (prefix ^ ".sexp") (Base.Sexp.to_string_hum (D.utilization circuit));
  let json =
    `Assoc
      [
        ("schema", `String "baux.hardcaml-distance.v1");
        ("design", `String design_id);
        ("dimension", `Int d.dimension);
        ("bits", `Int d.bits);
        ("fractional_bits", `Int (d.bits - 2));
        ("lanes", `Int d.lanes);
        ("pipeline", `Bool d.pipeline);
        ("samples", `Int t.samples);
        ("seed", `Int t.seed);
        ("max_error", `Float t.max_error);
        ("max_multiplier_bits", `Int t.max_multiplier_bits);
        ("cycles", `Int cycles);
        ("cycles_per_vector", `Float cycles_per_vector);
        ("max_absolute_error_per_dimension", `Float error);
        ("signed_multiplier_output_bits", `Int multiplier_bits);
        ("register_bits", `Int register_bits);
        ("feasible", `Bool feasible);
        ("reward", `Float reward);
        ( "resource_scope",
          `String
            "Hardcaml graph before synthesis; not LUT/DSP usage or silicon area"
        );
        ( "timing_scope",
          `String
            "simulation cycles with unlimited input bandwidth; no clock \
             frequency, host transfer or FPGA measurement" );
        ( "correctness",
          `String
            "bit-exact against independent integer model on corpus; \
             quantization error against float64" );
        ("rtl", `String (design_id ^ ".v"));
      ]
  in
  write (prefix ^ ".json") (Yojson.Basic.pretty_to_string json ^ "\n");
  let outputs =
    `List
      (List.map
         (fun (o : D.observation) ->
           `Assoc
             [
               ("raw", `Int o.raw);
               ("value", `Float o.value);
               ("reference", `Float o.reference);
               ("cycle", `Int o.cycle);
             ])
         observations)
  in
  write (prefix ^ ".outputs.json") (Yojson.Basic.to_string outputs ^ "\n");
  json

let run t directory =
  let path = Filename.concat directory "measurements.jsonl" in
  if Sys.file_exists path then
    invalid_arg "refusing to overwrite existing hardware experiment";
  write
    (Filename.concat directory "space.json")
    (Yojson.Basic.pretty_to_string (space t) ^ "\n");
  let out = open_out path in
  let rows = ref [] in
  Fun.protect
    ~finally:(fun () -> close_out out)
    (fun () ->
      List.iter
        (fun d ->
          let json = evaluate t d directory in
          rows := json :: !rows;
          output_string out (Yojson.Basic.to_string json ^ "\n");
          flush out;
          Printf.printf "Hardcaml checked %s\n%!" (D.name d))
        (designs t));
  let open Yojson.Basic.Util in
  let feasible =
    List.filter (fun j -> j |> member "feasible" |> to_bool) !rows
  in
  let ranked =
    List.sort
      (fun a b ->
        Float.compare
          (b |> member "reward" |> to_float)
          (a |> member "reward" |> to_float))
      feasible
  in
  let best = match ranked with [] -> `Null | row :: _ -> row in
  write
    (Filename.concat directory "summary.json")
    (Yojson.Basic.pretty_to_string
       (`Assoc
         [
           ("designs", `Int (List.length !rows));
           ("feasible_designs", `Int (List.length feasible));
           ("best", best);
           ( "method",
             `String "exhaustive configured grid, not a Bayesian optimizer run"
           );
         ])
    ^ "\n")
