@@ portable

type backend = Scalar | Simd
type workspace : value mod portable

val create : ?block_pairs:int -> Problem.t -> length:int -> workspace
(** Allocate reusable packed input and output blocks. Refill before running. *)

val init :
  ?block_pairs:int ->
  Problem.t ->
  length:int ->
  f:(int -> int -> float#) @ portable ->
  workspace

val refill : workspace -> f:(int -> int -> float#) @ portable -> unit
(** Generate coordinates directly into existing packed storage. Invalid values
    invalidate the workspace until a successful refill. Results become stale. *)

val fold_results :
  workspace ->
  init:float# ->
  f:(int -> float# -> float# -> float#) @ portable ->
  float#
(** Consume results without allocating a snapshot; callback receives index,
    accumulator and value. The workspace retains ownership of its buffers. *)

val prepare : ?block_pairs:int -> Problem.t -> Problem.point iarray -> workspace
(** Validate and pack immutable inputs once; allocate independent mutable output
    blocks. The workspace is reusable and must be accessed without contention. *)

val length : workspace @ contended -> int

val run : Parallel.t @ local -> backend -> workspace -> unit
(** Reuses output storage. Parallel tasks receive exclusive access to each block
    through the parallel-array capsule interface. Scheduling may allocate. *)

val get : workspace -> int -> float#

val snapshot : workspace -> float# iarray
(** Copy results into an immutable array that survives later workspace reuse. *)

val evaluate :
  ?backend:backend ->
  Parallel.t @ local ->
  Problem.t ->
  Problem.point iarray ->
  float# iarray
(** Convenience operation including validation and result allocation. Defaults
    to direct scalar evaluation. [Simd] additionally packs inputs; use [prepare]
    and [run] to amortize that cost across repeated calls. *)
