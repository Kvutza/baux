type t = {
  batch_sizes : int list;
  workers : int list;
  block_pairs : int list;
  samples : int;
  dimension : int;
  quota_ms : int;
  seed : int;
  functions : string list;
}

let decode ?samples_override document =
  let open Otoml in
  let fail message = invalid_arg ("Experiment TOML: " ^ message) in
  let fields =
    match document with
    | TomlTable fields -> fields
    | _ -> fail "expected a table"
  in
  let allowed =
    [
      "version";
      "experiment";
      "batch_sizes";
      "workers";
      "block_pairs";
      "samples";
      "dimension";
      "quota_ms";
      "seed";
      "functions";
    ]
  in
  List.iter
    (fun (key, _) ->
      if not (List.mem key allowed) then fail ("unknown field " ^ key))
    fields;
  let field key =
    match List.assoc_opt key fields with
    | Some value -> value
    | None -> fail ("missing field " ^ key)
  in
  let integer key value =
    match value with
    | TomlInteger n -> n
    | _ -> fail (key ^ " must be an integer")
  in
  let number key = integer key (field key) in
  let optional key default =
    match List.assoc_opt key fields with
    | None -> default
    | Some v -> integer key v
  in
  let positive key n = if n < 1 then fail (key ^ " must be positive") else n in
  let numbers key default =
    let value =
      match List.assoc_opt key fields with
      | None -> default
      | Some (TomlArray xs) -> List.map (integer key) xs
      | Some (TomlInteger n) when key = "workers" -> [ n ]
      | Some _ -> fail (key ^ " must be an integer array")
    in
    if value = [] then fail (key ^ " must not be empty");
    let value = List.map (positive key) value in
    if List.length (List.sort_uniq compare value) <> List.length value then
      fail (key ^ " must not contain duplicates");
    value
  in
  let version = number "version" in
  if version <> 1 && version <> 2 then fail "unsupported version";
  (match field "experiment" with
  | TomlString "oxcaml-catalogue" -> ()
  | _ -> fail "unknown experiment");
  let functions =
    match List.assoc_opt "functions" fields with
    | None -> List.map (fun (d : Problem.description) -> d.id) Problem.catalogue
    | Some (TomlArray xs) ->
        List.map
          (function
            | TomlString id ->
                ignore (Problem.create id);
                id
            | _ -> fail "functions must contain strings")
          xs
    | Some _ -> fail "functions must be a string array"
  in
  if
    functions = []
    || List.length functions
       <> List.length (List.sort_uniq String.compare functions)
  then fail "functions must be nonempty and unique";
  let samples =
    match samples_override with None -> number "samples" | Some n -> n
  in
  {
    batch_sizes = numbers "batch_sizes" [];
    workers = numbers "workers" [];
    block_pairs = numbers "block_pairs" [ 128 ];
    samples = positive "samples" samples;
    dimension = positive "dimension" (number "dimension");
    quota_ms = positive "quota_ms" (optional "quota_ms" 20);
    seed = optional "seed" 20261107;
    functions;
  }

let of_string ?samples_override text =
  match Otoml.Parser.from_string_result text with
  | Ok document -> decode ?samples_override document
  | Error message -> invalid_arg ("Experiment TOML: " ^ message)

let load ?samples_override path =
  match Otoml.Parser.from_file_result path with
  | Ok document -> decode ?samples_override document
  | Error message -> invalid_arg ("Experiment TOML: " ^ message)

let to_string t =
  let ints xs = "[" ^ String.concat ", " (List.map string_of_int xs) ^ "]" in
  let strings xs =
    "[" ^ String.concat ", " (List.map (Printf.sprintf "%S") xs) ^ "]"
  in
  Printf.sprintf
    "version = 2\n\
     experiment = \"oxcaml-catalogue\"\n\
     batch_sizes = %s\n\
     workers = %s\n\
     block_pairs = %s\n\
     samples = %d\n\
     dimension = %d\n\
     quota_ms = %d\n\
     seed = %d\n\
     functions = %s\n"
    (ints t.batch_sizes) (ints t.workers) (ints t.block_pairs) t.samples
    t.dimension t.quota_ms t.seed (strings t.functions)
