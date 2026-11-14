open Baux_benchmarks
module Iarray = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u

let family_name = function
  | Problem.Multimodal -> "multimodal"
  | Bowl -> "bowl"
  | Plate -> "plate"
  | Valley -> "valley"
  | Ridge -> "ridge"
  | Other -> "other"

let dimension_rule_info = function
  | Problem.Fixed n -> ("fixed", n, false)
  | At_least n -> ("at_least", max 2 n, true)
  | Multiple_of n -> ("multiple_of", n, true)

let bounds_to_py_list (bounds : (float * float) iarray) : Py_obj.t @ unique =
  let d = Iarray.length bounds in
  let outer = Py_obj.List.create d in
  for i = 0 to d - 1 do
    let lo, hi = Iarray.get bounds i in
    let pair = Py_obj.List.create 2 in
    Py_obj.List.set_steal (borrow_ pair) 0 (Py_obj.of_float lo);
    Py_obj.List.set_steal (borrow_ pair) 1 (Py_obj.of_float hi);
    Py_obj.List.set_steal (borrow_ outer) i pair
  done;
  outer

let problem_to_dict (prob : Problem.t) : Py_obj.t @ unique =
  let desc = Problem.describe prob in
  let rule_kind, default_dim, scalable = dimension_rule_info desc.dimensions in
  let dict = Py_obj.Dict.create () in
  Py_obj.Dict.set_string (borrow_ dict) "id" (Py_obj.of_string desc.id);
  Py_obj.Dict.set_string
    (borrow_ dict)
    "family"
    (Py_obj.of_string (family_name desc.family));
  Py_obj.Dict.set_string
    (borrow_ dict)
    "dimension"
    (Py_obj.of_int (Problem.dimension prob));
  Py_obj.Dict.set_string
    (borrow_ dict)
    "default_dim"
    (Py_obj.of_int default_dim);
  Py_obj.Dict.set_string (borrow_ dict) "rule" (Py_obj.of_string rule_kind);
  Py_obj.Dict.set_string (borrow_ dict) "scalable" (Py_obj.of_bool scalable);
  Py_obj.Dict.set_string (borrow_ dict) "source" (Py_obj.of_string desc.source);
  Py_obj.Dict.set_string
    (borrow_ dict)
    "bounds"
    (bounds_to_py_list (Problem.bounds prob));
  dict

let catalogue () : Py_obj.t @ unique =
  let entries = Iarray.of_list Problem.catalogue in
  let count = Iarray.length entries in
  let out = Py_obj.List.create count in
  for idx = 0 to count - 1 do
    let desc = Iarray.get entries idx in
    let prob = Problem.create desc.id in
    Py_obj.List.set_steal (borrow_ out) idx (problem_to_dict prob)
  done;
  out

let describe (id_obj @ local) (dim_obj @ local) : Py_obj.t @ unique =
  let id = Py_obj.to_string id_obj in
  let dim = Py_obj.to_int dim_obj in
  let prob =
    if dim <= 0 then Problem.create id else Problem.create ~dimension:dim id
  in
  problem_to_dict prob

let eval_one (id_obj @ local) (coords_obj @ local) : Py_obj.t @ unique =
  let id = Py_obj.to_string id_obj in
  let pt = Py_obj.to_f64_iarray coords_obj in
  let prob = Problem.create ~dimension:(Iarray.length pt) id in
  let y = U.to_float (Problem.evaluate prob pt) in
  Py_obj.of_float y

let parse_backend = function
  | "scalar" -> Batch.Scalar
  | "simd" -> Batch.Simd
  | other -> invalid_arg ("Unknown backend (expected scalar or simd): " ^ other)

let schedulers = Array.make 65 None

let scheduler_for workers : _ @ portable =
  let w = min 64 (max 2 workers) in
  match schedulers.(w) with
  | Some { Modes.Portable.portable = s } -> s
  | None ->
      let s = Parallel_scheduler.scheduler ~max_workers:w () in
      schedulers.(w) <- Some { Modes.Portable.portable = s };
      s

let eval_batch (id_obj @ local) (matrix_obj @ local) (backend_obj @ local)
    (workers_obj @ local) (block_pairs_obj @ local) : Py_obj.t @ unique =
  let id = Py_obj.to_string id_obj in
  let backend = parse_backend (Py_obj.to_string backend_obj) in
  let workers = Py_obj.to_int workers_obj in
  let block_pairs = Py_obj.to_int block_pairs_obj in
  if workers < 1 then invalid_arg "workers must be >= 1";
  if block_pairs < 1 then invalid_arg "block_pairs must be >= 1";
  let count, workspace =
    Py_obj.F64_buffer.with_matrix matrix_obj (fun view ->
        let rows = Py_obj.F64_buffer.rows view in
        let cols = Py_obj.F64_buffer.cols view in
        let prob = Problem.create ~dimension:cols id in
        let ws =
          Batch.init ~block_pairs prob ~length:rows ~f:(fun r c ->
              Py_obj.F64_buffer.get view r c)
        in
        (rows, ws))
  in
  if workers = 1 then Batch.run Parallel.sequential backend workspace
  else
    Parallel_scheduler.parallel (scheduler_for workers) (fun parallel _ ->
        Batch.run parallel backend workspace);
  let out_buf = Py_obj.F64_buffer.create_output count in
  for i = 0 to count - 1 do
    Py_obj.F64_buffer.set_output (borrow_ out_buf) i (Batch.get workspace i)
  done;
  out_buf

let refcount_probe (borrowed @ local) : Py_obj.t @ unique =
  let rc_borrowed = Py_obj.refcount borrowed in
  let rc_during_owned =
    Py_obj.with_release
      (fun () -> Py_obj.newref borrowed)
      (fun temp_local -> Py_obj.refcount temp_local)
  in
  let rc_after_release = Py_obj.refcount borrowed in
  let report = Py_obj.Dict.create () in
  Py_obj.Dict.set_string
    (borrow_ report)
    "borrowed_rc"
    (Py_obj.of_int rc_borrowed);
  Py_obj.Dict.set_string
    (borrow_ report)
    "owned_rc"
    (Py_obj.of_int rc_during_owned);
  Py_obj.Dict.set_string
    (borrow_ report)
    "after_release_rc"
    (Py_obj.of_int rc_after_release);
  report

let register_all () =
  Py_obj.Def.register0 catalogue 0;
  Py_obj.Def.register2 describe 0;
  Py_obj.Def.register2 eval_one 1;
  Py_obj.Def.register5 eval_batch 0;
  Py_obj.Def.register1 refcount_probe 0
