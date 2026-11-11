open Bigarray
open Baux_program

type 'a state = { parameters : 'a; residuals : 'a }

let record item =
  map
    ~host:
      ( (fun s -> (s.parameters, s.residuals)),
        fun (parameters, residuals) -> { parameters; residuals } )
    ~resident:
      ( (fun s -> (s.parameters, s.residuals)),
        fun (parameters, residuals) -> { parameters; residuals } )
    (pair item item)

let floats values = genarray_of_array1 (Array1.of_array float32 c_layout values)

let assertvalues actual expected =
  let actual = array1_of_genarray actual in
  Array.iteri
    (fun i v ->
      if Array1.get actual i <> v then failwith "native tensor differs")
    expected

let rejects f =
  match f () with
  | () -> failwith "invalid operation succeeded"
  | (exception Invalid_argument _) | (exception Failure _) -> ()

let exercise device path compiler =
  let leaf = tensor float32 [| 4 |] in
  let inputs = pair leaf leaf and outputs = record leaf in
  let load () =
    match compiler with
    | `Tvm -> load_tvm ~library:path ~entry:"main" ~device ~inputs ~outputs
    | `Xla ->
        let source = In_channel.with_open_bin path In_channel.input_all in
        compile_xla ~source ~device ~inputs ~outputs
    | `Iree ->
        let bytecode =
          In_channel.with_open_bin path In_channel.input_all |> Bytes.of_string
        in
        load ~bytecode ~entry:"round.main" ~device ~inputs ~outputs
  in
  let program = load () in
  let weights = floats [| 1.; -2.; 3.; 0. |] in
  let noise = floats [| 0.25; -0.5; 0.; 2. |] in
  let resident = put program (weights, noise) in
  Genarray.fill weights 99.;
  let first = run program resident in
  let pending = submit program (first.parameters, snd resident) in
  ignore (ready pending);
  Gc.full_major ();
  let result = wait pending |> read outputs in
  assertvalues result.parameters [| 1.5; -3.; 3.; 4. |];
  assertvalues result.residuals [| 2.25; 9.; 9.; 16. |];
  assertvalues (read outputs first).parameters [| 1.25; -2.5; 3.; 2. |];
  rejects (fun () -> ignore (wait pending));
  let foreign = put (load ()) (weights, noise) in
  rejects (fun () -> ignore (run program foreign));
  rejects (fun () -> ignore (put program (floats [| 1. |], noise)));
  (* Only the returned pending call retains this session and its input buffers. *)
  let detached =
    let local = load () in
    submit local (put local (noise, noise))
  in
  Gc.full_major ();
  Gc.compact ();
  assertvalues (read outputs (wait detached)).parameters [| 0.5; -1.; 0.; 4. |];
  let calls =
    Array.init
      (max 1 (min 2 (Domain.recommended_domain_count () - 1)))
      (fun _ -> submit program resident)
  in
  (* This isolated test owns a bounded number of domains; no Multicore pool exists. *)
  let domains =
    Array.map
      (fun (call : _ pending) ->
        (Domain.Safe.spawn [@alert "-do_not_spawn_domains"]) (fun () ->
            wait call))
      calls
  in
  Array.iter
    (fun d ->
      assertvalues (read outputs (Domain.join d)).parameters
        [| 1.25; -2.5; 3.; 2. |])
    domains;
  rejects (fun () ->
      ignore
        (match compiler with
        | `Tvm ->
            load_tvm ~library:path ~entry:"missing" ~device ~inputs ~outputs
        | `Xla -> compile_xla ~source:"invalid" ~device ~inputs ~outputs
        | `Iree ->
            Baux_program.load
              ~bytecode:(Bytes.of_string "invalid")
              ~entry:"round.main" ~device ~inputs ~outputs));
  let label =
    match compiler with
    | `Tvm -> "TVM CPU"
    | `Xla -> "XLA CPU"
    | `Iree -> ( match device with Cpu -> "CPU" | Metal -> "Metal")
  in
  Printf.printf
    "OxCaml %s: structured rounds, ownership, GC, domains, rejection passed\n%!"
    label

let () =
  exercise Cpu (Sys.getenv "BAUX_VMFB_CPU") `Iree;
  exercise Metal (Sys.getenv "BAUX_VMFB_METAL") `Iree;
  exercise Cpu (Sys.getenv "BAUX_STABLEHLO") `Xla;
  let model = create_model ~seed:42L in
  if model_parameters model <> 265_623_320 || model_rows model <> 8192 then
    failwith "resident model metadata differs";
  let tokens = Genarray.create int32 c_layout [| model_rows model |] in
  let labels = Genarray.create int32 c_layout [| model_rows model |] in
  let mask = Genarray.create int8_unsigned c_layout [| model_rows model |] in
  Genarray.fill tokens 0l;
  Genarray.fill labels 0l;
  Genarray.fill mask 1;
  let call = submit_model model ~tokens ~labels ~mask in
  ignore (ready call);
  let observation = wait call |> read_model model in
  if
    Genarray.dims observation.token_losses <> [| 8192 |]
    || Genarray.dims observation.sequence_nlls <> [| 2 |]
  then failwith "resident model output shape differs";
  let sequence_nlls = array1_of_genarray observation.sequence_nlls in
  if not (Float.is_finite sequence_nlls.{0} && Float.is_finite sequence_nlls.{1})
  then failwith "resident model produced nonfinite observations";
  Printf.printf "OxCaml FBT/PISA: initialized model forward passed\n%!"
