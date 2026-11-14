(** Type-safe modal OxCaml-to-Python benchmark interface over the 56-function
    [Baux_benchmarks] suite. *)

val catalogue : unit -> Py_obj.t @ unique
val describe : Py_obj.t @ local -> Py_obj.t @ local -> Py_obj.t @ unique
val eval_one : Py_obj.t @ local -> Py_obj.t @ local -> Py_obj.t @ unique

val eval_batch :
  Py_obj.t @ local ->
  Py_obj.t @ local ->
  Py_obj.t @ local ->
  Py_obj.t @ local ->
  Py_obj.t @ local ->
  Py_obj.t @ unique

val refcount_probe : Py_obj.t @ local -> Py_obj.t @ unique
val register_all : unit -> unit
