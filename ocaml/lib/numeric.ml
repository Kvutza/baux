module Iarray = Stdlib_stable.Iarray
module U = Stdlib_upstream_compatible.Float_u

type input =
  | Point of float iarray @@ global
  | Packed of { data : floatarray @@ global; offset : int; dimension : int }

module type%template S = sig @@ portable
  type t : k mod everything

  val constant : float# -> t [@@zero_alloc]
  val of_float : float @ local -> t [@@zero_alloc]
  val of_int : int -> t [@@zero_alloc]
  val get : input @ local -> int -> t [@@zero_alloc]
  val dimension : input @ local -> int [@@zero_alloc]
  val add : t -> t -> t [@@zero_alloc]
  val sub : t -> t -> t [@@zero_alloc]
  val mul : t -> t -> t [@@zero_alloc]
  val div : t -> t -> t [@@zero_alloc]
  val pow : t -> t -> t [@@zero_alloc]
  val neg : t -> t [@@zero_alloc]
  val abs : t -> t [@@zero_alloc]
  val sqrt : t -> t [@@zero_alloc]
  val sin : t -> t [@@zero_alloc]
  val cos : t -> t [@@zero_alloc]
  val exp : t -> t [@@zero_alloc]
  val log : t -> t [@@zero_alloc]
end
[@@kind k = (float64, vec128)]

module Scalar = struct
  include U

  let[@zero_alloc] [@inline always] of_float (x @ local) = U.of_float x
  let[@zero_alloc] [@inline always] of_int x = U.of_int x
  let[@zero_alloc] [@inline always] add x y = U.add x y
  let[@zero_alloc] [@inline always] sub x y = U.sub x y
  let[@zero_alloc] [@inline always] mul x y = U.mul x y
  let[@zero_alloc] [@inline always] div x y = U.div x y
  let[@zero_alloc] [@inline always] pow x y = U.pow x y
  let[@zero_alloc] [@inline always] neg x = U.neg x
  let[@zero_alloc] [@inline always] abs x = U.abs x
  let[@zero_alloc] [@inline always] sqrt x = U.sqrt x
  let[@zero_alloc] [@inline always] sin x = U.sin x
  let[@zero_alloc] [@inline always] cos x = U.cos x
  let[@zero_alloc] [@inline always] exp x = U.exp x
  let[@zero_alloc] [@inline always] log x = U.log x
  let[@zero_alloc] [@inline always] constant (x : float#) = x

  let[@zero_alloc] [@inline always] get (x @ local) i =
    match x with
    | Point p -> of_float (Iarray.get p i) [@nontail]
    | Packed p ->
        of_float (Float.Array.get p.data (p.offset + (2 * i))) [@nontail]

  let[@zero_alloc] [@inline always] dimension (x @ local) =
    match x with Point p -> Iarray.length p | Packed p -> p.dimension
end

module Neon = struct
  type t = float64x2#

  external add_native : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_add"
  [@@noalloc] [@@builtin]

  external sub_native : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_sub"
  [@@noalloc] [@@builtin]

  external mul_native : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_mul"
  [@@noalloc] [@@builtin]

  external div_native : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_div"
  [@@noalloc] [@@builtin]

  external sqrt_native : t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_sqrt"
  [@@noalloc] [@@builtin]

  external low_of : float# -> t @@ portable
    = "baux_simd_bytecode" "caml_float64x2_low_of_float"
  [@@noalloc] [@@builtin]

  external low : t -> float# @@ portable
    = "baux_simd_bytecode" "caml_float64x2_low_to_float"
  [@@noalloc] [@@builtin]

  external zip_low : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_zip1"
  [@@noalloc] [@@builtin]

  external zip_high : t -> t -> t @@ portable
    = "baux_simd_bytecode" "caml_neon_float64x2_zip2"
  [@@noalloc] [@@builtin]

  external load : float iarray -> int -> t @@ portable
    = "%caml_floatarray_get128#"

  external load_packed : floatarray -> int -> t @@ portable
    = "%caml_floatarray_get128#"

  external store : floatarray -> int -> t -> unit @@ portable
    = "%caml_floatarray_set128#"

  external abs_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_abs"
  [@@noalloc]

  external neg_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_neg"
  [@@noalloc]

  external sin_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_sin"
  [@@noalloc]

  external cos_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_cos"
  [@@noalloc]

  external exp_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_exp"
  [@@noalloc]

  external log_native : t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_log"
  [@@noalloc]

  external pow_native : t -> t -> t @@ portable
    = "baux_simd_bytecode" "baux_neon_pow"
  [@@noalloc]

  let[@zero_alloc] [@inline always] add x y = add_native x y
  let[@zero_alloc] [@inline always] sub x y = sub_native x y
  let[@zero_alloc] [@inline always] mul x y = mul_native x y
  let[@zero_alloc] [@inline always] div x y = div_native x y
  let[@zero_alloc] [@inline always] pow x y = pow_native x y
  let[@zero_alloc] [@inline always] neg x = neg_native x
  let[@zero_alloc] [@inline always] abs x = abs_native x
  let[@zero_alloc] [@inline always] sqrt x = sqrt_native x
  let[@zero_alloc] [@inline always] sin x = sin_native x
  let[@zero_alloc] [@inline always] cos x = cos_native x
  let[@zero_alloc] [@inline always] exp x = exp_native x
  let[@zero_alloc] [@inline always] log x = log_native x

  let[@zero_alloc] [@inline always] constant x =
    let v = low_of x in
    zip_low v v

  let[@zero_alloc] [@inline always] high x = low (zip_high x x)

  let[@zero_alloc] [@inline always] of_float (x @ local) =
    constant (U.of_float x)

  let[@zero_alloc] [@inline always] of_int x = constant (U.of_int x)

  let[@zero_alloc] [@inline always] get (x @ local) i =
    match x with
    | Point p -> load p (2 * i)
    | Packed p -> load_packed p.data (p.offset + (2 * i))

  let[@zero_alloc] [@inline always] dimension (x @ local) =
    match x with Point p -> Iarray.length p / 2 | Packed p -> p.dimension
end
