open Baux_hardware

let () =
  let c =
    {
      Flow_model.levels = 20;
      steps = 32;
      width = 4;
      depth = 1;
      heads = 1;
      expansion = 2;
    }
  in
  let d =
    { Flow_core.bits = 28; lanes = 4; banks = 8; pipeline = true; queue = 4 }
  in
  let weights =
    Array.mapi
      (fun i x -> x +. (0.04 *. sin (float_of_int i)))
      (Flow_model.initial c 19)
  in
  let example = Flow_data.synthetic c ~seed:29 ~regime:203 in
  let run mode =
    let backend = Flow_backend.create mode d in
    let (loss, outputs), profile =
      Search.measure (fun () ->
          Flow_sampling.training backend c weights example 37 ~dropout:false)
    in
    (loss, outputs, backend.stats, profile)
  in
  let li, xi, si, pi = run Flow_backend.Integer in
  let open Yojson.Basic.Util in
  let plan = Flow_plan.profile c d 1 1 in
  if to_int (member "kernel_cycles" plan) <> si.cycles then
    failwith "full-grid static plan mismatch";

  Printf.printf "640-token integer execution: %s\n%!"
    (Yojson.Basic.to_string pi);
  let lr, xr, sr, pr = run Flow_backend.Rtl in
  if xi <> xr || li <> lr || si.cycles <> sr.cycles || sr.checked <> sr.outputs
  then failwith "full token-grid RTL parity failure";
  Printf.printf
    "640-token RTL execution: cycles=%d checked_words=%d profile=%s\n%!"
    sr.cycles sr.checked
    (Yojson.Basic.to_string pr)
