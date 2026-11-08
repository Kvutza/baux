module B = Flow_backend
module N = Flow_numeric

type config = {
  levels : int;
  steps : int;
  width : int;
  depth : int;
  heads : int;
  expansion : int;
}

let validate c =
  if
    c.levels < 2 || c.levels > 20
    || c.levels mod 2 <> 0
    || c.steps < 1 || c.steps > 32 || c.width < 4 || c.width > 1024
    || c.width mod 2 <> 0
    || c.depth < 1 || c.depth > 32 || c.heads < 1
    || c.width mod c.heads <> 0
    || c.expansion < 1 || c.expansion > 4
  then invalid_arg "invalid FlowLOB model shape"

let tokens c = c.levels * c.steps

type tensor = {
  name : string;
  rows : int;
  cols : int;
  offset : int;
  length : int;
}

let schema c =
  validate c;
  let offset = ref 0 and fields = ref [] in
  let add name rows cols =
    let length = rows * cols in
    fields := { name; rows; cols; offset = !offset; length } :: !fields;
    offset := !offset + length
  in
  let linear name rows cols =
    add (name ^ ".weight") rows cols;
    add (name ^ ".bias") rows 1
  in
  linear "input" c.width 9;
  add "level" c.levels c.width;
  add "position" c.steps c.width;
  linear "time.0" c.width c.width;
  linear "time.1" c.width c.width;
  for layer = 0 to c.depth - 1 do
    let prefix = Printf.sprintf "block.%d." layer in
    linear (prefix ^ "modulation") (6 * c.width) c.width;
    linear (prefix ^ "qkv") (3 * c.width) c.width;
    linear (prefix ^ "attention") c.width c.width;
    linear (prefix ^ "ff.0") (c.expansion * c.width) c.width;
    linear (prefix ^ "ff.1") c.width (c.expansion * c.width)
  done;
  linear "final.modulation" (2 * c.width) c.width;
  linear "output" 2 c.width;
  List.rev !fields

let parameters c = List.fold_left (fun n t -> n + t.length) 0 (schema c)

let initial c seed =
  let rng = Random.State.make [| seed; 9173 |] in
  let weights = Array.make (parameters c) 0. in
  List.iter
    (fun t ->
      let bias = Filename.check_suffix t.name ".bias" in
      let zero =
        bias || t.name = "output.weight"
        || String.ends_with ~suffix:"modulation.weight" t.name
      in
      if not zero then
        for i = 0 to t.length - 1 do
          weights.(t.offset + i) <-
            (Random.State.float rng 2. -. 1.) *. sqrt (3. /. float_of_int t.cols)
        done)
    (schema c);
  weights

let add b x y = B.binary b N.Add x y
let mul b x y = B.binary b N.Mul x y
let constant n x = Array.make n x
let scale b x factor = mul b x (constant (Array.length x) factor)

let linear b schema weights name inputs =
  let w = List.find (fun t -> t.name = name ^ ".weight") schema in
  let bias = List.find (fun t -> t.name = name ^ ".bias") schema in
  let rows =
    Array.init w.rows (fun row ->
        Array.sub weights (w.offset + (row * w.cols)) w.cols)
  in
  Array.map
    (fun x ->
      if Array.length x <> w.cols then invalid_arg ("linear shape " ^ name);
      let y = Array.map (fun row -> B.dot b row x) rows in
      add b y (Array.sub weights bias.offset bias.length))
    inputs

let norm b x =
  let n = Array.length x in
  let mean = B.dot b x (constant n (1. /. float_of_int n)) in
  let centered = B.binary b N.Sub x (constant n mean) in
  let variance =
    B.dot b (mul b centered centered) (constant n (1. /. float_of_int n))
  in
  let epsilon =
    if b.B.mode = B.Float then 1e-5
    else max 1e-5 (1. /. float_of_int (N.scale b.design.bits))
  in
  let inverse = B.unary b N.Rsqrt [| B.scalar b N.Add variance epsilon |] in
  scale b centered inverse.(0)

let modulate b x shift factor =
  add b (mul b (norm b x) (add b (constant (Array.length x) 1.) factor)) shift

let time_embedding c t =
  Array.init c.width (fun i ->
      let frequency =
        exp
          (-.log 10000.
          *. float_of_int (i mod (c.width / 2))
          /. float_of_int (c.width / 2))
      in
      if i < c.width / 2 then cos (t *. frequency) else sin (t *. frequency))

let softmax b x =
  let maximum = B.max b x in
  let exponents =
    B.unary b N.Exp (B.binary b N.Sub x (constant (Array.length x) maximum))
  in
  let inverse = (B.unary b N.Inv [| B.sum b exponents |]).(0) in
  scale b exponents inverse

let attention b c qkv =
  let count = Array.length qkv and size = c.width / c.heads in
  let result = Array.init count (fun _ -> Array.make c.width 0.) in
  for head = 0 to c.heads - 1 do
    let q = Array.map (fun x -> Array.sub x (head * size) size) qkv in
    let k =
      Array.map (fun x -> Array.sub x (c.width + (head * size)) size) qkv
    in
    let v =
      Array.init size (fun d ->
          Array.map (fun x -> x.((2 * c.width) + (head * size) + d)) qkv)
    in
    for token = 0 to count - 1 do
      let scores = Array.map (fun key -> B.dot b q.(token) key) k in
      let probabilities =
        softmax b (scale b scores (1. /. sqrt (float_of_int size)))
      in
      for d = 0 to size - 1 do
        result.(token).((head * size) + d) <- B.dot b probabilities v.(d)
      done
    done
  done;
  result

let block b c schema weights layer embedding x =
  let prefix = Printf.sprintf "block.%d." layer in
  let mods =
    (linear b schema weights (prefix ^ "modulation")
       [| B.unary b N.Silu embedding |]).(0)
  in
  let part i = Array.sub mods (i * c.width) c.width in
  let h = Array.map (fun x -> modulate b x (part 0) (part 1)) x in
  let qkv = linear b schema weights (prefix ^ "qkv") h in
  let h = linear b schema weights (prefix ^ "attention") (attention b c qkv) in
  let x = Array.mapi (fun i x -> add b x (mul b (part 2) h.(i))) x in
  let h = Array.map (fun x -> modulate b x (part 3) (part 4)) x in
  let h =
    linear b schema weights (prefix ^ "ff.0") h |> Array.map (B.unary b N.Gelu)
  in
  let h = linear b schema weights (prefix ^ "ff.1") h in
  Array.mapi (fun i x -> add b x (mul b (part 5) h.(i))) x

let forward b c weights ~state ~condition ~time =
  if
    Array.length weights <> parameters c
    || Array.length state <> tokens c
    || Array.length condition <> tokens c
    || not (Float.is_finite time)
  then invalid_arg "FlowLOB tensor/model mismatch";
  Array.iter
    (fun x ->
      if Array.length x <> 2 then invalid_arg "state needs two channels")
    state;
  Array.iter
    (fun x ->
      if Array.length x <> 7 then invalid_arg "condition needs seven channels")
    condition;
  let schema = schema c in
  let x = Array.mapi (fun i x -> Array.append x condition.(i)) state in
  let x = linear b schema weights "input" x in
  let level = List.find (fun t -> t.name = "level") schema
  and position = List.find (fun t -> t.name = "position") schema in
  let x =
    Array.mapi
      (fun i x ->
        add b x
          (add b
             (Array.sub weights
                (level.offset + (i mod c.levels * c.width))
                c.width)
             (Array.sub weights
                (position.offset + (i / c.levels * c.width))
                c.width)))
      x
  in
  let embedding =
    linear b schema weights "time.0" [| time_embedding c time |]
    |> Array.map (B.unary b N.Silu)
    |> linear b schema weights "time.1"
  in
  let x = ref x in
  for layer = 0 to c.depth - 1 do
    x := block b c schema weights layer embedding.(0) !x
  done;
  let mods =
    (linear b schema weights "final.modulation"
       [| B.unary b N.Silu embedding.(0) |]).(0)
  in
  let x =
    Array.map
      (fun x ->
        modulate b x (Array.sub mods 0 c.width) (Array.sub mods c.width c.width))
      !x
  in
  linear b schema weights "output" x

let profile c =
  let n = tokens c and d = c.width in
  let macs =
    (n * 9 * d)
    + (2 * d * d)
    + c.depth
      * ((6 * d * d) + (n * (4 + (2 * c.expansion)) * d * d) + (2 * n * n * d))
    + (2 * d * d)
    + (n * 2 * d)
  in
  `Assoc
    [
      ("parameters", `Int (parameters c));
      ("tokens", `Int n);
      ("float64_parameter_bytes", `Int (8 * parameters c));
      ("dense_and_attention_mac_terms_per_nfe", `Int macs);
      ( "scope",
        `String
          "architecture accounting only; no model execution or FPGA timing \
           measurement" );
    ]
