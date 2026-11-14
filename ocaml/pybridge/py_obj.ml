module Iarray = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u

type t

external newref : t @ local -> t @ unique = "baux_py_newref"
external decref : t @ local unique -> unit = "baux_py_decref"

let with_release (acquire @ local) (body @ local) =
  let handle = acquire () in
  match body (borrow_ handle) with
  | result ->
      decref handle;
      result
  | exception exn ->
      decref handle;
      raise exn

external to_string : t @ local -> string = "baux_py_to_string"
external to_int : t @ local -> int = "baux_py_to_int"
external to_float : t @ local -> float = "baux_py_to_float"
external to_f64_array : t @ local -> float array = "baux_py_to_f64_array"
external refcount : t @ local -> int = "baux_py_refcount"

let to_f64_iarray (x @ local) = Iarray.of_array (to_f64_array x)

external of_float : float -> t @ unique = "baux_py_of_float"
external of_int : int -> t @ unique = "baux_py_of_int"
external of_bool : bool -> t @ unique = "baux_py_of_bool"
external of_string : string @ local -> t @ unique = "baux_py_of_string"

module List = struct
  external create : int -> t @ unique = "baux_py_list_create"

  external set_steal : t @ local -> int -> t @ unique -> unit
    = "baux_py_list_set_steal"
end

module Dict = struct
  external create : unit -> t @ unique = "baux_py_dict_create"

  external set_string : t @ local -> string @ local -> t @ unique -> unit
    = "baux_py_dict_set_string"
end

module F64_buffer = struct
  type raw_handle : immediate
  type view = { handle : raw_handle; rows : int; cols : int }

  external acquire_matrix : t @ local -> raw_handle * int * int
    = "baux_py_acquire_matrix"

  external release_matrix : raw_handle -> unit = "baux_py_release_matrix"

  external get_cell : raw_handle -> int -> int -> float# @@ portable
    = "baux_py_matrix_get_byte" "baux_py_matrix_get"
  [@@noalloc]

  let rows v = v.rows
  let cols v = v.cols
  let[@inline always] get v r c = get_cell v.handle r c

  let with_matrix (obj @ local) (f @ local) =
    let handle, rows, cols = acquire_matrix obj in
    let view = { handle; rows; cols } in
    match f view with
    | res ->
        release_matrix handle;
        res
    | exception exn ->
        release_matrix handle;
        raise exn

  external create_output : int -> t @ unique = "baux_py_output_create"

  external set_output_raw : t @ local -> int -> float# -> unit
    = "baux_py_output_set_byte" "baux_py_output_set"
  [@@noalloc]

  let[@inline always] set_output (buf @ local) idx v = set_output_raw buf idx v
end

module Def = struct
  external set_slot0 : int -> (unit -> t) -> unit = "baux_py_set_slot0"
  external set_slot1 : int -> (t -> t) -> unit = "baux_py_set_slot1"
  external set_slot2 : int -> (t -> t -> t) -> unit = "baux_py_set_slot2"

  external set_slot5 : int -> (t -> t -> t -> t -> t -> t) -> unit
    = "baux_py_set_slot5"

  let register0 (f : unit -> t @ unique) slot = set_slot0 slot (fun () -> f ())

  let register1 (f : t @ local -> t @ unique) slot =
    set_slot1 slot (fun a -> f a)

  let register2 (f : t @ local -> t @ local -> t @ unique) slot =
    set_slot2 slot (fun a b -> f a b)

  let register5
      (f :
        t @ local ->
        t @ local ->
        t @ local ->
        t @ local ->
        t @ local ->
        t @ unique) slot =
    set_slot5 slot (fun a b c d e -> f a b c d e)
end
