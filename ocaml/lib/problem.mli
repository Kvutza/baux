@@ portable

type point = float iarray
type dimensions = Fixed of int | At_least of int | Multiple_of of int
type family = Multimodal | Bowl | Plate | Valley | Ridge | Other

type description = private {
  id : string;
  family : family;
  dimensions : dimensions;
  source : string;
}

type t : value mod contended portable

val catalogue : description list

val create : ?dimension:int -> string -> t
(** Unknown identifiers and incompatible dimensions raise [Invalid_argument].
    Parameterized functions use the source's documented default parameters. *)

val describe : t -> description
val dimension : t -> int

val bounds : t -> (float * float) iarray
(** Recommended search domain; evaluation does not clamp inputs to it. *)

val evaluate : t -> point -> float#
(** Deterministic minimization objective. Requires the configured dimension and
    finite coordinates. Arithmetic overflow is reported as its IEEE result. *)

val within_bounds : t -> point -> bool

module Prepared : sig
  type problem := t
  type t : value mod contended portable

  val create : problem -> point iarray -> t
  (** Validates once and packs pairs of candidates, coordinate by coordinate. An
      odd final candidate is duplicated into the unused lane. *)

  val length : t -> int
  val pairs : t -> int
  val scalar : t -> int -> float#

  val simd : t -> int -> float64x2#
  (** Evaluate one packed pair. Indices are checked. *)
end

module Packed : sig
  val scalar : t -> floatarray -> offset:int -> float#

  val simd : t -> floatarray -> offset:int -> float64x2#
  (** Low-level evaluation of previously validated interleaved coordinates.
      Coordinate i starts at offset + 2*i. Array accesses are bounds checked.
      Scalar evaluates one lane; SIMD evaluates both lanes. *)
end
