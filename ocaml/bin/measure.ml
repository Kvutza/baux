open Baux_benchmarks
module I = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u
module B = Core_bench.Bench

type measurement = {
  seconds : float;
  low : float option;
  high : float option;
  samples : int;
}

type choice = {
  backend : Batch.backend;
  workers : int;
  block : int;
  workspace : Batch.workspace;
}

let backend_name = function Batch.Scalar -> "scalar" | Simd -> "simd"

let median xs =
  let a = Array.of_list xs in
  Array.sort Float.compare a;
  let n = Array.length a in
  if n mod 2 = 0 then (a.((n / 2) - 1) +. a.(n / 2)) /. 2. else a.(n / 2)

let number out n =
  if Float.is_finite n then Printf.fprintf out "%.17g" n
  else output_string out "null"

let optional out = function
  | None -> output_string out "null"
  | Some n -> number out n

external now : unit -> float# = "baux_simd_bytecode" "baux_monotonic_seconds"
[@@noalloc]

let seconds () = U.to_float (now ())

let measure config output name workers action =
  let collect (parallel @ local) =
    for _ = 1 to 3 do
      action parallel
    done;
    let started = seconds () in
    let quota = float_of_int config.Experiment_config.quota_ms /. 1000. in
    let rec sample index runs rows =
      if index >= 12 && seconds () -. started >= quota then List.rev rows
      else
        let before = Gc.quick_stat () in
        let start = seconds () in
        for _ = 1 to runs do
          action parallel
        done;
        let elapsed = seconds () -. start in
        let after = Gc.quick_stat () in
        let row = Core_bench_internals.Measurement_sample.create () in
        row.runs <- runs;
        row.nanos <- Core.Int63.of_float (elapsed *. 1e9);
        row.minor_allocated <-
          int_of_float (after.minor_words -. before.minor_words);
        row.major_allocated <-
          int_of_float (after.major_words -. before.major_words);
        row.promoted <-
          int_of_float (after.promoted_words -. before.promoted_words);
        row.minor_collections <-
          after.minor_collections - before.minor_collections;
        row.major_collections <-
          after.major_collections - before.major_collections;
        row.compactions <- after.compactions - before.compactions;
        sample (index + 1)
          (min 1000000 (max (runs + 1) (runs * 5 / 4)))
          (row :: rows)
    in
    let rows = sample 0 1 [] in
    let largest_run =
      List.fold_left
        (fun n r -> max n r.Core_bench_internals.Measurement_sample.runs)
        0 rows
    in
    Core_bench_internals.Measurement.create ~name ~test_name:name
      ~file_name:"measure.ml" ~module_name:"Baux" ~largest_run
      ~samples:(Array.of_list rows)
  in
  let m =
    if workers = 1 then collect Parallel.sequential
    else Parallel_scheduler.with_parallel ~max_workers:workers collect
  in
  let measurements = [ m ] in
  match measurements with
  | [ m ] ->
      B.Measurement.save m
        ~filename:(Filename.concat output (name ^ ".samples"));
      let analysis =
        B.analyze m
          ~analysis_configs:
            [
              B.Analysis_config.with_error_estimation ~bootstrap_trials:100
                B.Analysis_config.nanos_vs_runs;
            ]
      in
      let result = Core.Or_error.ok_exn analysis in
      let out = open_out (Filename.concat output (name ^ ".sexp")) in
      Fun.protect
        ~finally:(fun () -> close_out out)
        (fun () ->
          output_string out
            (Core.Sexp.to_string_hum (B.Analysis_result.sexp_of_t result)));
      let regression = (B.Analysis_result.regressions result).(0) in
      let coefficient =
        (B.Analysis_result.Regression.coefficients regression).(0)
      in
      let seconds = B.Analysis_result.Coefficient.estimate coefficient /. 1e9 in
      if (not (Float.is_finite seconds)) || seconds <= 0. then
        failwith ("Invalid timing: " ^ name);
      let low, high =
        match B.Analysis_result.Coefficient.ci95 coefficient with
        | None -> (None, None)
        | Some ci ->
            ( Some (B.Analysis_result.Ci95.left_endpoint ci /. 1e9),
              Some (B.Analysis_result.Ci95.right_endpoint ci /. 1e9) )
      in
      { seconds; low; high; samples = B.Analysis_result.sample_count result }
  | _ -> failwith "Expected one Core_bench measurement"

let run config output =
  let rec mkdir path =
    if not (Sys.file_exists path) then (
      mkdir (Filename.dirname path);
      Sys.mkdir path 0o755)
  in
  mkdir output;
  if Sys.file_exists (Filename.concat output "measurements.jsonl") then
    invalid_arg "Experiment output already contains measurements";
  let out = open_out (Filename.concat output "measurements.jsonl") in
  let profiles = open_out (Filename.concat output "schedule.jsonl") in
  Fun.protect
    ~finally:(fun () ->
      close_out out;
      close_out profiles)
    (fun () ->
      List.iter
        (fun id ->
          let description = Problem.describe (Problem.create id) in
          let dimension =
            match description.dimensions with
            | Fixed n -> n
            | At_least n -> max n config.Experiment_config.dimension
            | Multiple_of n -> n * max 1 ((config.dimension + n - 1) / n)
          in
          let problem = Problem.create ~dimension id in
          let bounds = Problem.bounds problem in
          let coordinate seed row col =
            let low, high = I.get bounds col in
            let r =
              ((seed mod 1009) + (row mod 1009 * 193) + (col mod 1009 * 317))
              mod 1009
            in
            U.of_float (low +. ((high -. low) *. float_of_int r /. 1008.))
          in
          List.iter
            (fun count ->
              let make block =
                Batch.init ~block_pairs:block problem ~length:count
                  ~f:(coordinate config.seed)
              in
              let choices =
                List.concat_map
                  (fun block ->
                    let workspace = make block in
                    List.concat_map
                      (fun workers ->
                        List.map
                          (fun backend ->
                            { backend; workers; block; workspace })
                          [ Batch.Scalar; Simd ])
                      config.workers)
                  config.block_pairs
              in
              let check choice seed =
                Batch.refill choice.workspace ~f:(coordinate seed);
                Batch.run Parallel.sequential choice.backend choice.workspace;
                for row = 0 to count - 1 do
                  let point =
                    I.init dimension (fun col ->
                        U.to_float (coordinate seed row col))
                  in
                  let expected = U.to_float (Problem.evaluate problem point) in
                  let actual = U.to_float (Batch.get choice.workspace row) in
                  if
                    (not (Float.is_finite expected && Float.is_finite actual))
                    || Float.abs (expected -. actual)
                       > (if id = "bukin6" then 1e-6 else 2e-10)
                         *. Float.max 1. (Float.abs expected)
                  then failwith ("Experiment parity failure: " ^ id)
                done
              in
              let emit phase choice repeat action =
                let name =
                  Printf.sprintf "%s-n%d-%s-%s-w%d-b%d-r%d" id count phase
                    (backend_name choice.backend)
                    choice.workers choice.block repeat
                in
                let m = measure config output name choice.workers action in
                Printf.fprintf out
                  "{\"id\":%S,\"dimension\":%d,\"batch_size\":%d,\"phase\":%S,\"backend\":%S,\"workers\":%d,\"block_pairs\":%d,\"repeat\":%d,\"seconds\":"
                  id dimension count phase
                  (backend_name choice.backend)
                  choice.workers choice.block repeat;
                number out m.seconds;
                output_string out ",\"ci95_low\":";
                optional out m.low;
                output_string out ",\"ci95_high\":";
                optional out m.high;
                Printf.fprintf out ",\"sample_count\":%d,\"raw\":%S}\n"
                  m.samples (name ^ ".samples");
                flush out;
                m.seconds
              in
              let scored =
                List.map
                  (fun choice ->
                    check choice config.seed;
                    (choice, ref []))
                  choices
              in
              for repeat = 1 to config.samples do
                let order =
                  if repeat mod 2 = 0 then List.rev scored else scored
                in
                List.iter
                  (fun (choice, times) ->
                    let seconds =
                      emit "calibration" choice repeat (fun parallel ->
                          Batch.run parallel choice.backend choice.workspace)
                    in
                    times := seconds :: !times)
                  order
              done;
              let sorted =
                List.sort
                  (fun (_, a) (_, b) -> Float.compare (median !a) (median !b))
                  scored
              in
              let chosen, _ = List.hd sorted in
              let baseline =
                {
                  backend = Batch.Scalar;
                  workers = 1;
                  block = 128;
                  workspace = make 128;
                }
              in
              let seed = config.seed + 1 in
              check chosen seed;
              check baseline seed;
              let confirmation choice phase =
                List.init config.samples (fun repeat ->
                    emit phase choice (repeat + 1) (fun parallel ->
                        Batch.run parallel choice.backend choice.workspace))
                |> median
              in
              let baseline_seconds = confirmation baseline "baseline" in
              let chosen_seconds = confirmation chosen "confirmation" in
              let pipeline =
                List.init config.samples (fun repeat ->
                    emit "packed_pipeline" chosen (repeat + 1) (fun parallel ->
                        Batch.refill chosen.workspace ~f:(coordinate seed);
                        Batch.run parallel chosen.backend chosen.workspace;
                        ignore
                          (Sys.opaque_identity
                             (U.to_float
                                (Batch.fold_results chosen.workspace ~init:#0.
                                   ~f:(fun _ a b -> U.add a b))))))
                |> median
              in
              let direct =
                List.init config.samples (fun repeat ->
                    emit "direct_pipeline" baseline (repeat + 1)
                      (fun parallel ->
                        let points =
                          I.init count (fun row ->
                              I.init dimension (fun col ->
                                  U.to_float (coordinate seed row col)))
                        in
                        let values =
                          Batch.evaluate ~backend:Scalar parallel problem points
                        in
                        let rec sum i acc =
                          if i = count then acc
                          else sum (i + 1) (U.add acc (I.get values i))
                        in
                        ignore (Sys.opaque_identity (U.to_float (sum 0 #0.)))))
                |> median
              in
              Printf.fprintf profiles
                "{\"id\":%S,\"dimension\":%d,\"batch_size\":%d,\"backend\":%S,\"workers\":%d,\"block_pairs\":%d,\"baseline_seconds\":%.17g,\"selected_seconds\":%.17g,\"confirmed_speedup\":%.17g,\"packed_pipeline_seconds\":%.17g,\"direct_pipeline_seconds\":%.17g,\"pipeline_speedup\":%.17g}\n"
                id dimension count
                (backend_name chosen.backend)
                chosen.workers chosen.block baseline_seconds chosen_seconds
                (baseline_seconds /. chosen_seconds)
                pipeline direct (direct /. pipeline);
              flush profiles;
              Printf.printf
                "%s batch %d: %s, %d workers, %d pairs/block; confirmation \
                 %.2fx, pipeline %.2fx\n\
                 %!"
                id count
                (backend_name chosen.backend)
                chosen.workers chosen.block
                (baseline_seconds /. chosen_seconds)
                (direct /. pipeline))
            config.batch_sizes)
        config.functions)
