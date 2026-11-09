open Baux_hardware
module N = Flow_numeric
module B = Flow_backend
module C = Flow_core
module M = Flow_model

let check condition message = if not condition then failwith message

let close a b tolerance =
  check (abs_float (a -. b) <= tolerance) "flow numerical mismatch"

let primitive d =
  let integer = B.create ~period:7 B.Integer d
  and rtl = B.create ~period:7 B.Rtl d in
  let a =
    [|
      -16.;
      -8.;
      -4.1;
      -0.5;
      0.;
      0.0001;
      0.01;
      0.5;
      1.;
      1.9;
      2.;
      3.;
      7.;
      8.;
      16.;
      300.;
      1000.;
    |]
  in
  let b =
    Array.init (Array.length a) (fun i -> if i mod 2 = 0 then 0.25 else -0.5)
  in
  List.iter
    (fun op ->
      let x = B.binary integer op a b and y = B.binary rtl op a b in
      check (x = y) "primitive integer/RTL mismatch")
    [ N.Add; N.Sub; N.Mul; N.Max; N.Exp; N.Inv; N.Rsqrt; N.Gelu; N.Silu ];
  close (B.dot integer a b) (B.dot rtl a b) 0.;
  check (integer.stats.cycles = rtl.stats.cycles) "RTL timing mismatch";
  check
    (rtl.stats.checked > 0 && rtl.stats.stalls > 0)
    "RTL did not exercise backpressure"

let c =
  { M.levels = 2; steps = 2; width = 4; depth = 2; heads = 2; expansion = 2 }

let d = { C.bits = 28; lanes = 4; banks = 4; pipeline = true; queue = 2 }

let weights c =
  Array.mapi (fun i x -> x +. (0.04 *. sin (float_of_int i))) (M.initial c 17)

let model_checks () =
  let example = Flow_data.synthetic c ~seed:3 ~regime:2 in
  let weights = weights c in
  let integer = B.create B.Integer d
  and rtl = B.create B.Rtl d
  and reference = B.create B.Float d in
  let li, xi =
    Flow_sampling.training integer c weights example 7 ~dropout:false
  in
  let lr, xr = Flow_sampling.training rtl c weights example 7 ~dropout:false in
  let lf, _ =
    Flow_sampling.training reference c weights example 7 ~dropout:false
  in
  check (xi = xr && li = lr) "complete transformer RTL parity";
  close li lf 0.01;
  List.iter
    (fun method_ ->
      let run backend =
        Flow_sampling.generate backend c weights ~condition:example.condition
          ~noise:(Flow_data.noise c 9) ~method_ ~steps:2 ~guidance:1.5
      in
      let xi, ni = run integer and xr, nr = run rtl in
      check (xi = xr && ni = nr) "complete guided sampler RTL parity";
      check
        (ni = 4 * match method_ with "euler" -> 1 | "heun" -> 2 | _ -> 4)
        "NFE accounting")
    [ "euler"; "heun"; "rk4" ];
  check
    (integer.stats.cycles = rtl.stats.cycles)
    "complete model cycle accounting"

let representation () =
  let snapshots =
    Array.init c.M.steps (fun i ->
        {
          Flow_data.prices = [| 101. +. float_of_int i; 99. +. float_of_int i |];
          volumes = [| 100.; 200. |];
        })
  in
  let encoded = Flow_data.encode c ~tick:0.5 ~previous:99. snapshots in
  let decoded = Flow_data.decode c ~tick:0.5 ~previous:99. encoded in
  Array.iteri
    (fun i s ->
      Array.iteri
        (fun l p -> close p snapshots.(i).prices.(l) 1e-9)
        s.Flow_data.prices;
      Array.iteri (fun l v -> close v snapshots.(i).volumes.(l) 1e-9) s.volumes)
    decoded

let () =
  List.iter
    (fun bits ->
      List.iter
        (fun lanes ->
          List.iter
            (fun banks ->
              List.iter
                (fun pipeline ->
                  List.iter
                    (fun queue ->
                      primitive { C.bits; lanes; banks; pipeline; queue })
                    [ 1; 4 ])
                [ false; true ])
            [ 1; 4 ])
        [ 1; 4 ])
    [ 20; 28 ];
  representation ();
  model_checks ();
  let full = { c with M.levels = 20; steps = 32; width = 8; heads = 2 } in
  let example = Flow_data.synthetic full ~seed:29 ~regime:203 in
  let backend = B.create B.Integer d in
  let (loss, _), profile =
    Search.measure (fun () ->
        Flow_sampling.training backend full (weights full) example 13
          ~dropout:false)
  in
  check (Float.is_finite loss) "full 640-token model failed";
  let open Yojson.Basic.Util in
  let plan = Flow_plan.profile full d 1 1 in
  check
    (to_int (member "kernel_cycles" plan) = backend.stats.cycles)
    "static whole-model plan mismatch";
  check
    (to_int (member "scalar_operations" plan) = backend.stats.operations)
    "planned operation count mismatch";

  Printf.printf
    "FlowLOB: primitives, complete transformer, Euler/Heun/RK4, guidance and \
     640-token graph passed; loss=%g profile=%s counters=%s\n\
     %!"
    loss
    (Yojson.Basic.to_string profile)
    (Yojson.Basic.to_string (B.json backend))

let () =
  let c =
    {
      M.levels = 20;
      steps = 32;
      width = 768;
      depth = 9;
      heads = 12;
      expansion = 4;
    }
  in
  check (M.parameters c = 98066690) "large model schema count";
  Printf.printf "FlowLOB large reconstruction profile: %s\n%!"
    (Yojson.Basic.to_string (M.profile c))

let () =
  let open Yojson.Basic.Util in
  let evaluate = Flow_worker.create 1 in
  let coordinates =
    Array.init (M.parameters c) (fun i -> 0.2 *. sin (float_of_int i))
  in
  let base =
    [
      ("version", `Int 1);
      ("model", Flow_worker.model_json c);
      ("bits", `Int 28);
      ("lanes", `Int 4);
      ("banks", `Int 4);
      ("pipeline", `Bool true);
      ("queue", `Int 2);
      ("backend", `String "integer");
      ("ready_period", `Int 3);
      ("initial_seed", `Int 17);
      ("radius", `Float 0.1);
      ("samples", `Int 1);
      ("window", `Int 0);
      ("data_seed", `Int 9);
      ("target_seed", `Int 101);
      ("split", `String "test");
      ("solver", `String "heun");
      ("solver_steps", `Int 2);
      ("rollout_windows", `Int 2);
      ("guidance", `Float 1.5);
    ]
  in
  let values a = `List (Array.to_list (Array.map (fun x -> `Float x) a)) in
  let request extra weights =
    `Assoc (extra @ [ ("weights", values weights) ] @ base)
  in
  let first = evaluate (request [] coordinates) in
  let again = evaluate (request [] coordinates) in
  check (member "loss" first = member "loss" again) "flow cache changed loss";
  check (member "cache_hit" again = `Bool true) "flow backend cache miss";
  let generated =
    evaluate (request [ ("op", `String "generate") ] coordinates)
  in
  check
    (member "neural_function_evaluations" generated = `Int 16)
    "rollout NFE mismatch";
  check
    (List.length (generated |> member "rollout" |> to_list) = 2)
    "missing autoregressive rollout";
  let checkpoint = Filename.temp_file "baux-flow" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove checkpoint)
    (fun () ->
      Yojson.Basic.to_file checkpoint (member "checkpoint" generated);
      let replay =
        evaluate
          (request
             [ ("checkpoint", `String checkpoint) ]
             (Array.make (M.parameters c) 0.))
      in
      check
        (member "loss" first = member "loss" replay)
        "checkpoint replay changed weights");
  let snapshots =
    `List
      (List.init c.steps (fun i ->
           `Assoc
             [
               ( "prices",
                 values [| 101. +. float_of_int i; 99. +. float_of_int i |] );
               ("volumes", values [| 100.; 200. |]);
             ]))
  in
  let entry =
    `Assoc
      [
        ("tick_size", `Float 1.);
        ("previous_mid", `Float 100.);
        ("time_of_day", `Float 0.5);
        ("past", snapshots);
        ("future", snapshots);
      ]
  in
  let dataset = Filename.temp_file "baux-lob" ".json" in
  Fun.protect
    ~finally:(fun () -> Sys.remove dataset)
    (fun () ->
      Yojson.Basic.to_file dataset
        (`Assoc
          [
            ("version", `Int 1);
            ("train", `List [ entry ]);
            ("validation", `List [ entry ]);
            ("test", `List [ entry ]);
          ]);
      let loaded = Flow_data.load c dataset "test" 0 in
      check (Array.length loaded.target = M.tokens c) "raw LOB dataset loader";
      ignore (evaluate (request [ ("dataset", `String dataset) ] coordinates)));
  Printf.printf
    "FlowLOB checkpoint, dataset, cache and autoregressive rollout replay passed\n\
     %!"
