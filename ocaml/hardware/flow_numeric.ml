type op = Add | Sub | Mul | Max | Exp | Inv | Rsqrt | Gelu | Silu | Dot

let code = function
  | Add -> 0
  | Sub -> 1
  | Mul -> 2
  | Max -> 3
  | Exp -> 4
  | Inv -> 5
  | Rsqrt -> 6
  | Gelu -> 7
  | Silu -> 8
  | Dot -> 9

let frac bits = bits - 12
let scale bits = 1 lsl frac bits
let bound bits = (1 lsl (bits - 1)) - 1
let clamp bits x = max (-(1 lsl (bits - 1))) (min (bound bits) x)

let quantize bits x =
  if not (Float.is_finite x) then invalid_arg "nonfinite flow tensor";
  int_of_float
    (max
       (float_of_int (-(1 lsl (bits - 1))))
       (min
          (float_of_int (bound bits))
          (Float.round (x *. float_of_int (scale bits)))))

let real bits x = float_of_int x /. float_of_int (scale bits)

let exact op a b =
  match op with
  | Add -> a +. b
  | Sub -> a -. b
  | Mul | Dot -> a *. b
  | Max -> max a b
  | Exp -> exp (min 0. a)
  | Inv -> 1. /. max 1e-6 a
  | Rsqrt -> 1. /. sqrt (max 1e-6 a)
  | Gelu ->
      0.5 *. a
      *. (1. +. tanh (sqrt (2. /. Float.pi) *. (a +. (0.044715 *. (a ** 3.)))))
  | Silu -> a /. (1. +. exp (-.a))

let table bits op =
  Array.init 257 (fun i ->
      let x = (float_of_int i /. 16.) -. if op = Exp then 16. else 8. in
      quantize bits (exact op x 0.))

let build_inverse_table bits square odd =
  Array.init 256 (fun i ->
      let x = float_of_int (max 128 i) /. 128. *. if odd then 2. else 1. in
      quantize bits (if square then 1. /. sqrt x else 1. /. x))

let inverse_cache = Hashtbl.create 32

let inverse_table bits square odd =
  let key = (bits, square, odd) in
  match Hashtbl.find_opt inverse_cache key with
  | Some table -> table
  | None ->
      let table = build_inverse_table bits square odd in
      Hashtbl.add inverse_cache key table;
      table

let tables bits =
  Array.init 10 (fun i ->
      match i with
      | 4 -> table bits Exp
      | 7 -> table bits Gelu
      | 8 -> table bits Silu
      | _ -> [||])

let floor_half n = if n >= 0 then n / 2 else (n - 1) / 2
let shift x n = if n >= 0 then x lsl n else x asr -n

let inverse bits square a =
  if a <= 0 then bound bits
  else
    let rec highest n b = if n < 2 then b else highest (n lsr 1) (b + 1) in
    let b = highest a 0 in
    let index = shift a (7 - b) in
    let exponent = b - frac bits in
    let odd = square && exponent mod 2 <> 0 in
    let coefficient = inverse_table bits square odd in
    let power = if square then floor_half exponent else exponent in
    clamp bits (shift coefficient.(index) (-power))

let unary bits tables op a =
  match op with
  | Inv -> inverse bits false a
  | Rsqrt -> inverse bits true a
  | Exp | Gelu | Silu ->
      let radius = (if op = Exp then 16 else 8) * scale bits in
      let upper = if op = Exp then 0 else radius in
      if a < -radius then 0
      else if a > upper then if op = Exp then scale bits else a
      else tables.(code op).((a + radius) asr (frac bits - 4))
  | _ -> invalid_arg "not a unary operation"

let binary bits op a b =
  clamp bits
    (match op with
    | Add -> a + b
    | Sub -> a - b
    | Mul | Dot -> (a * b) asr frac bits
    | Max -> max a b
    | _ -> invalid_arg "not a binary operation")
