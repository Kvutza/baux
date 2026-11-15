(* admin/ui/oxcaml_eval.ml
   DOM-free Js_of_ocaml evaluator bundle (assets/oxcaml_eval.js) exposing all 56
   canonical OCaml benchmark formulas on globalThis.BauxOxcaml for Web Workers
   and Pyodide zero-copy TypedArray crossings.

   NEVER references window or document; safe in dedicated Web Workers and Node.
*)

open Js_of_ocaml

let version_string = "1.0.0"

let raise_js_error (msg : string) : 'a =
  let err =
    Js.Unsafe.new_obj
      (Js.Unsafe.pure_js_expr "Error")
      [| Js.Unsafe.inject (Js.string msg) |]
  in
  Js.Js_error.raise_ (Js.Js_error.of_error err)

let is_null_or_undef (v : Js.Unsafe.any) : bool =
  Js.Unsafe.strict_equals v (Js.Unsafe.inject Js.undefined)
  || Js.Unsafe.strict_equals v (Js.Unsafe.inject Js.null)

let parse_string_arg label (v : Js.Unsafe.any) : string =
  if Js.to_string (Js.typeof (Js.Unsafe.coerce v)) <> "string" then
    raise_js_error (label ^ " must be a string")
  else Js.to_string (Js.Unsafe.coerce v)

let parse_int_arg label (v : Js.Unsafe.any) : int =
  if Js.to_string (Js.typeof (Js.Unsafe.coerce v)) <> "number" then
    raise_js_error (label ^ " must be an integer")
  else
    let f = Js.to_float (Js.Unsafe.coerce v) in
    if (not (Float.is_finite f)) || Float.floor f <> f then
      raise_js_error (label ^ " must be a finite integer")
    else int_of_float f

let parse_opt_dim (v : Js.Unsafe.any) : int =
  if is_null_or_undef v then 0 else parse_int_arg "dim" v

let bounds_to_js (bounds : (float * float) array) =
  let n = Array.length bounds in
  let outer = new%js Js.array_length n in
  for i = 0 to n - 1 do
    let lo, hi = bounds.(i) in
    let pair = new%js Js.array_length 2 in
    Js.array_set pair 0 (Js.float lo);
    Js.array_set pair 1 (Js.float hi);
    Js.array_set outer i pair
  done;
  outer

let spec_to_js_obj (sp : Bench_formulas.spec) ~dim
    (bounds : (float * float) array) =
  let dd = Bench_formulas.default_dim sp.rule in
  Js.Unsafe.obj
    [|
      ("id", Js.Unsafe.inject (Js.string sp.id));
      ("label", Js.Unsafe.inject (Js.string sp.label));
      ( "family",
        Js.Unsafe.inject (Js.string (Bench_formulas.family_slug sp.family)) );
      ("rule", Js.Unsafe.inject (Js.string (Bench_formulas.rule_slug sp.rule)));
      ("dim", Js.Unsafe.inject dim);
      ("dimension", Js.Unsafe.inject dim);
      ("default_dim", Js.Unsafe.inject dd);
      ( "scalable",
        Js.Unsafe.inject (Js.bool (Bench_formulas.is_scalable sp.rule)) );
      ("min_dim", Js.Unsafe.inject (Bench_formulas.min_dim sp.rule));
      ("dim_step", Js.Unsafe.inject (Bench_formulas.dim_step sp.rule));
      ("source", Js.Unsafe.inject (Js.string sp.source));
      ("bounds", Js.Unsafe.inject (bounds_to_js bounds));
      ("domain", Js.Unsafe.inject (bounds_to_js bounds));
    |]

let js_version _ = Js.Unsafe.inject (Js.string version_string)

let js_catalogue _ =
  let n = Array.length Bench_formulas.catalogue in
  let out = new%js Js.array_length n in
  for i = 0 to n - 1 do
    let sp = Bench_formulas.catalogue.(i) in
    let dd = Bench_formulas.default_dim sp.rule in
    let bounds = sp.domain dd in
    Js.array_set out i (spec_to_js_obj sp ~dim:dd bounds)
  done;
  Js.Unsafe.inject out

let js_describe name_any dim_any =
  let id = parse_string_arg "name" name_any in
  let dim = parse_opt_dim dim_any in
  if dim < 0 then raise_js_error (Printf.sprintf "Invalid dim %d for %s" dim id);
  try
    let desc = Bench_formulas.describe ~dim id in
    Js.Unsafe.inject (spec_to_js_obj desc.spec ~dim:desc.dim desc.bounds)
  with Invalid_argument msg -> raise_js_error msg

let js_eval_one name_any dim_any pt_any =
  let id = parse_string_arg "name" name_any in
  let req_dim = parse_opt_dim dim_any in
  if req_dim < 0 then
    raise_js_error (Printf.sprintf "Invalid dim %d for %s" req_dim id);
  if is_null_or_undef pt_any then raise_js_error "point must be an array";
  let len =
    parse_int_arg "point.length" (Js.Unsafe.get pt_any (Js.string "length"))
  in
  let target_dim = if req_dim > 0 then req_dim else len in
  if len <> target_dim then
    raise_js_error
      (Printf.sprintf "Point dimension mismatch for %s: expected %d, got %d"
         id target_dim len);
  let x = Array.make target_dim 0.0 in
  for i = 0 to target_dim - 1 do
    let raw = Js.Unsafe.get pt_any i in
    if Js.to_string (Js.typeof raw) <> "number" then
      raise_js_error
        (Printf.sprintf "Coordinate %d for %s must be a finite number" i id);
    let v = Js.to_float raw in
    if not (Float.is_finite v) then
      raise_js_error
        (Printf.sprintf "Non-finite coordinate at index %d for %s" i id);
    x.(i) <- v
  done;
  try
    let y = Bench_formulas.eval_one ~dim:target_dim id x in
    Js.Unsafe.inject (Js.float y)
  with Invalid_argument msg -> raise_js_error msg

let js_eval_batch name_any dim_any n_any flat_any out_any =
  let id = parse_string_arg "name" name_any in
  let dim = parse_int_arg "dim" dim_any in
  let n = parse_int_arg "n" n_any in
  if dim <= 0 then
    raise_js_error (Printf.sprintf "Invalid dim %d for %s" dim id);
  if n < 0 then raise_js_error (Printf.sprintf "Invalid batch size %d" n);
  if is_null_or_undef flat_any then
    raise_js_error "flatFloat64Array must not be null or undefined";
  let in_len =
    parse_int_arg "flatFloat64Array.length"
      (Js.Unsafe.get flat_any (Js.string "length"))
  in
  if n > 0 && dim > max_int / n then
    raise_js_error "Batch dimensions overflow integer bounds";
  let needed = n * dim in
  if in_len < needed then
    raise_js_error
      (Printf.sprintf
         "flatFloat64Array length %d is smaller than n*dim=%d" in_len needed);
  let flat_ta : Typed_array.float64Array Js.t = Js.Unsafe.coerce flat_any in
  let out_ta : Typed_array.float64Array Js.t =
    if is_null_or_undef out_any then new%js Typed_array.float64Array n
    else
      let out_len =
        parse_int_arg "outFloat64Array.length"
          (Js.Unsafe.get out_any (Js.string "length"))
      in
      if out_len < n then
        raise_js_error
          (Printf.sprintf "outFloat64Array length %d is smaller than n=%d"
             out_len n);
      Js.Unsafe.coerce out_any
  in
  (try
     Bench_formulas.eval_batch_indexed id ~dim ~count:n
       ~get:(fun i j ->
         Js.to_float (Typed_array.unsafe_get flat_ta ((i * dim) + j)))
       ~set:(fun i y -> Typed_array.set out_ta i (Js.float y))
   with Invalid_argument msg -> raise_js_error msg);
  Js.Unsafe.inject out_ta

let api =
  Js.Unsafe.obj
    [|
      ("version", Js.Unsafe.inject (Js.Unsafe.callback js_version));
      ("catalogue", Js.Unsafe.inject (Js.Unsafe.callback js_catalogue));
      ("describe", Js.Unsafe.inject (Js.Unsafe.callback js_describe));
      ("evalOne", Js.Unsafe.inject (Js.Unsafe.callback js_eval_one));
      ("evalBatch", Js.Unsafe.inject (Js.Unsafe.callback js_eval_batch));
    |]

let () =
  Js.Unsafe.set Js.Unsafe.global (Js.string "BauxOxcaml") api;
  Js.export "BauxOxcaml" api
