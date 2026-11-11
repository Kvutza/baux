@@ portable

open Bigarray
(** Typed structures over the shared Rust runtime. Transfers are explicit.
    Native sessions are serialized; [submit] queues work without blocking the
    calling domain. Device results stay resident until [read]. *)

type dtype = Bool | I32 | I64 | U32 | U64 | F16 | BF16 | F32 | F64
type device = Cpu | Metal
type ('a, 'b) buffer : value mod contended portable
type ('host, 'resident) layout : value mod contended portable

val tensor :
  ?dtype:dtype ->
  ('a, 'b) kind ->
  int array ->
  (('a, 'b, c_layout) Genarray.t, ('a, 'b) buffer) layout
(** Float and signed integer kinds infer their dtype. Unsigned, Boolean and
    half-precision values use an explicit dtype with the corresponding raw
    Bigarray storage kind. Shapes are copied and checked at every boundary. *)

val unit : (unit, unit) layout
val pair : ('a, 'b) layout -> ('c, 'd) layout -> ('a * 'c, 'b * 'd) layout
val array : int -> ('a, 'b) layout -> ('a array, 'b array) layout

val map :
  host:('a -> 'b) * ('b -> 'a) @ portable ->
  resident:('c -> 'd) * ('d -> 'c) @ portable ->
  ('b, 'd) layout ->
  ('a, 'c) layout
(** Isomorphisms preserve user records without runtime reflection. *)

type ('host, 'resident, 'output, 'result) t : value mod contended portable

val load :
  bytecode:bytes ->
  entry:string ->
  device:device ->
  inputs:('a, 'b) layout ->
  outputs:('c, 'd) layout ->
  ('a, 'b, 'c, 'd) t
(** Load a compiled VMFB artifact. [entry] is its qualified module.function
    name. CPU and Metal artifacts must be compiled for the selected device. *)

val load_tvm :
  library:string ->
  entry:string ->
  device:device ->
  inputs:('a, 'b) layout ->
  outputs:('c, 'd) layout ->
  ('a, 'b, 'c, 'd) t
(** Load a trusted TVM shared library for CPU execution. The library remains
    loaded while any buffer or pending call uses it. *)

val compile_xla :
  source:string ->
  device:device ->
  inputs:('a, 'b) layout ->
  outputs:('c, 'd) layout ->
  ('a, 'b, 'c, 'd) t
(** Compile StableHLO with a [main] entry using the native XLA CPU plugin. *)

val put : ('a, 'b, 'c, 'd) t -> 'a -> 'b
val run : ('a, 'b, 'c, 'd) t -> 'b -> 'd
val read : ('a, 'b) layout -> 'b -> 'a

type 'result pending : value mod contended portable

val submit : ('a, 'b, 'c, 'd) t -> 'b -> 'd pending
(** Pending calls retain their executable and input buffers, including if the
    caller drops its references. Dropping a pending call does not cancel it. *)

val ready : 'a pending -> bool

val wait : 'a pending -> 'a
(** Consume completion exactly once. Blocking releases the OCaml runtime lock. *)

type model : value mod contended portable
type model_backend = Native | Tvm

type model_output = {
  token_losses : (float, float32_elt) buffer;
  sequence_nlls : (float, float32_elt) buffer;
}

type model_observation = {
  token_losses : (float, float32_elt, c_layout) Genarray.t;
  sequence_nlls : (float, float32_elt, c_layout) Genarray.t;
}

val create_model : seed:int64 -> model
(** Initialize the fixed-shape 265M-parameter FBT/PISA MHC4 model on Metal. *)

val create_on : model_backend -> seed:int64 -> model
(** Select direct native execution or TVM Relax, preserving the same Metal kernels. *)

val model_parameters : model -> int
val model_rows : model -> int

val submit_model :
  model ->
  tokens:(int32, int32_elt, c_layout) Genarray.t ->
  labels:(int32, int32_elt, c_layout) Genarray.t ->
  mask:(int, int8_unsigned_elt, c_layout) Genarray.t ->
  model_output pending
(** Copy one complete batch, lease resident parameters and queue its forward
    pass. The model shape is checked before native execution. *)

val read_model : model -> model_output -> model_observation
(** Copy both resident observations to host Bigarrays. *)
