module Iarray = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u

type t
(** Modal Python object wrapper with compile-time borrow ([local]) and
    single-owner steal/release ([unique]) tracking. *)

val newref : t @ local -> t @ unique
(** Increment Python's reference count to promote a borrowed local reference
    into an owned unique handle. *)

val decref : t @ local unique -> unit
(** Deterministically decrement Python's reference count and consume uniqueness
    so subsequent uses fail at compile time. *)

val with_release :
  (unit -> t @ unique) @ local -> (t @ local -> 'a) @ local -> 'a
(** Exception-safe scoped release for unique Python objects, borrowing the
    handle locally for [body] and invoking [decref] on every exit path. *)

val to_string : t @ local -> string
(** Read-only inspection of borrowed Python values without [Py_INCREF]. *)

val to_int : t @ local -> int
val to_float : t @ local -> float
val to_f64_iarray : t @ local -> float iarray
val refcount : t @ local -> int

val of_float : float -> t @ unique
(** Construct fresh single-owner ([ob_refcnt = 1]) Python values. *)

val of_int : int -> t @ unique
val of_bool : bool -> t @ unique
val of_string : string @ local -> t @ unique

module List : sig
  val create : int -> t @ unique

  val set_steal : t @ local -> int -> t @ unique -> unit
  (** Borrows the target list ([borrow_ lst]) and steals ownership of the
      element reference ([PyList_SetItem]). *)
end

module Dict : sig
  val create : unit -> t @ unique

  val set_string : t @ local -> string @ local -> t @ unique -> unit
  (** Borrows the target dictionary ([borrow_ dict]) and consumes the unique
      value reference after insertion. *)
end

module F64_buffer : sig
  type view : value mod contended portable

  val rows : view -> int @@ portable
  val cols : view -> int @@ portable
  val get : view -> int -> int -> float# @@ portable

  val with_matrix : t @ local -> (view -> 'a) @ local -> 'a
  (** Borrow a contiguous 2D C-order [float64] buffer (e.g. [numpy.ndarray])
      without incrementing its reference count; the view is invalidated and
      [PyBuffer_Release] runs before [with_matrix] returns. *)

  val create_output : int -> t @ unique
  (** Allocate an unaliased [N * 8]-byte Python [bytearray] ([ob_refcnt = 1]). *)

  val set_output : t @ local -> int -> float# -> unit
  (** Write an unboxed [float#] into a borrowed output buffer ([borrow_ buf]). *)
end

module Def : sig
  val register0 : (unit -> t @ unique) -> int -> unit
  val register1 : (t @ local -> t @ unique) -> int -> unit
  val register2 : (t @ local -> t @ local -> t @ unique) -> int -> unit

  val register5 :
    (t @ local ->
    t @ local ->
    t @ local ->
    t @ local ->
    t @ local ->
    t @ unique) ->
    int ->
    unit
end
