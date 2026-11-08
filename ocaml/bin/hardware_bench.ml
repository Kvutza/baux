open Baux_hardware

let run = function
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
        "Usage: space CONFIG | resolve CONFIG OUTPUT | experiment CONFIG \
         OUTPUT_DIR | evaluate CONFIG BITS LANES PIPELINE OUTPUT_DIR"

let () =
  try run (List.tl (Array.to_list Sys.argv))
  with Invalid_argument message | Failure message | Sys_error message ->
    prerr_endline message;
    exit 2
