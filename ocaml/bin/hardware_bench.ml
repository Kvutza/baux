open Baux_hardware

let run = function
  | [ "flow-profile"; width; depth; heads ] ->
      let model =
        {
          Flow_model.levels = 20;
          steps = 32;
          width = int_of_string width;
          depth = int_of_string depth;
          heads = int_of_string heads;
          expansion = 4;
        }
      in
      let design =
        {
          Flow_core.bits = 28;
          lanes = 8;
          banks = 8;
          pipeline = true;
          queue = 4;
        }
      in
      let result = Flow_plan.profile model design 1 1 in
      let result =
        match result with
        | `Assoc fields ->
            `Assoc (("design", `String (Flow_core.name design)) :: fields)
        | _ -> result
      in
      print_endline (Yojson.Basic.to_string result)
  | [ "flow-worker"; capacity ] -> Flow_worker.serve (int_of_string capacity)
  | [ "matvec-worker"; capacity ] ->
      Matvec_search.serve (int_of_string capacity)
  | [ "worker"; capacity ] -> Search.serve (int_of_string capacity)
  | [ "space"; source ] ->
      Experiment.load source |> Experiment.space |> Yojson.Basic.to_string
      |> print_endline
  | [ "resolve"; source; destination ] ->
      Experiment.write destination
        (Experiment.to_string (Experiment.load source))
  | [ "experiment"; source; directory ] ->
      Experiment.run (Experiment.load source) directory
  | [ "evaluate"; source; bits; lanes; pipeline; directory ] ->
      let config = Experiment.load source in
      let design =
        {
          Distance.dimension = config.dimension;
          bits = int_of_string bits;
          lanes = int_of_string lanes;
          pipeline = bool_of_string pipeline;
        }
      in
      let result = Experiment.evaluate config design directory in
      print_endline (Yojson.Basic.to_string result)
  | _ ->
      invalid_arg
        "Usage: worker CACHE_CAPACITY | space CONFIG | resolve CONFIG OUTPUT | \
         experiment CONFIG OUTPUT_DIR | evaluate CONFIG BITS LANES PIPELINE \
         OUTPUT_DIR"

let () =
  try run (List.tl (Array.to_list Sys.argv))
  with Invalid_argument message | Failure message | Sys_error message ->
    prerr_endline message;
    exit 2
