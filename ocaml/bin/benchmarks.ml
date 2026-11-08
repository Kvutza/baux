open Baux_benchmarks
module Iarray = Stdlib_stable.Iarray
module F = Stdlib_upstream_compatible.Float_u

let fields s =
  String.split_on_char ' ' (String.map (function '\t' -> ' ' | c -> c) s)
  |> List.filter (fun s -> s <> "")

let point fields = Iarray.of_list (List.map float_of_string fields)
let print_value x = Printf.printf "%.17g\n" (F.to_float x)

let run = function
  | "resolve" :: source :: destination :: overrides ->
      let samples_override =
        match overrides with
        | [] -> None
        | [ n ] -> Some (int_of_string n)
        | _ -> invalid_arg "resolve CONFIG OUTPUT [SAMPLES]"
      in
      let config = Experiment_config.load ?samples_override source in
      let out = open_out destination in
      Fun.protect
        ~finally:(fun () -> close_out out)
        (fun () -> output_string out (Experiment_config.to_string config))
  | [ "experiment"; source; output ] ->
      Measure.run (Experiment_config.load source) output
  | [ "list" ] ->
      List.iter
        (fun (d : Problem.description) ->
          let t = Problem.create d.id in
          Printf.printf "%s\t%d\t%s\n" d.id (Problem.dimension t) d.source)
        Problem.catalogue
  | "eval" :: id :: dimension :: coordinates ->
      let t = Problem.create ~dimension:(int_of_string dimension) id in
      print_value (Problem.evaluate t (point coordinates))
  | "batch" :: id :: dimension :: workers :: backend ->
      let backend =
        match backend with
        | [] | [ "scalar" ] -> Batch.Scalar
        | [ "simd" ] -> Batch.Simd
        | _ -> invalid_arg "Backend must be scalar or simd"
      in
      let t = Problem.create ~dimension:(int_of_string dimension) id in
      let workers = int_of_string workers in
      if workers < 1 then invalid_arg "Worker count must be positive";
      let rec read acc =
        match input_line stdin with
        | line -> read (point (fields line) :: acc)
        | exception End_of_file -> Iarray.of_list (List.rev acc)
      in
      let points = read [] in
      let emit values =
        for i = 0 to Iarray.length values - 1 do
          print_value (Iarray.get values i)
        done
      in
      if workers = 1 then
        emit (Batch.evaluate ~backend Parallel.sequential t points)
      else
        Parallel_scheduler.with_parallel ~max_workers:workers (fun parallel ->
            emit (Batch.evaluate ~backend parallel t points))
  | _ ->
      invalid_arg
        "Usage: list | eval ID DIM X... | batch ID DIM WORKERS [scalar|simd] \
         (rows on stdin) | resolve CONFIG OUTPUT [SAMPLES] | experiment CONFIG \
         OUTPUT_DIR"

let () =
  try run (List.tl (Array.to_list Sys.argv))
  with Invalid_argument message | Failure message ->
    prerr_endline message;
    exit 2
