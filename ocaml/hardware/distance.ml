open Hardcaml

type design = { dimension : int; bits : int; lanes : int; pipeline : bool }

let validate d =
  if d.dimension < 1 || d.dimension > 1024 then
    invalid_arg "dimension must be in 1..1024";
  if d.bits < 4 || d.bits > 20 then invalid_arg "bits must be in 4..20";
  if d.lanes < 1 || d.lanes > 32 || d.lanes > d.dimension then
    invalid_arg "lanes must be in 1..min(dimension,32)"

let rec ceil_log2 n = if n <= 1 then 0 else 1 + ceil_log2 ((n + 1) / 2)
let output_bits d = (2 * (d.bits + 1)) + ceil_log2 d.dimension
let scale d = float_of_int (1 lsl (d.bits - 2))

let quantize d x =
  if (not (Float.is_finite x)) || x < -1. || x > 1. then
    invalid_arg "coordinates must be finite and in [-1,1]";
  int_of_float (Float.round (x *. scale d))

let name d =
  Printf.sprintf "distance_d%d_b%d_l%d_p%d" d.dimension d.bits d.lanes
    (if d.pipeline then 1 else 0)

(* Each valid beat carries [lanes] signed fixed-point coordinate pairs.
   First/last delimit ceil(dimension/lanes) beats; the final unused lanes
   must have x=y=0. There is no backpressure. Bubbles preserve the accumulator,
   clear aborts pending work, and valid_out qualifies the registered result.
   Products and the accumulator retain all bits; only inputs are quantized. *)
let create d =
  validate d;
  let open Signal in
  let clock = input "clock" 1 and clear = input "clear" 1 in
  let spec = Reg_spec.create ~clock ~clear () in
  let valid = input "valid" 1 in
  let first = input "first" 1 and last = input "last" 1 in
  let products =
    List.init d.lanes (fun lane ->
        let x = input (Printf.sprintf "x%d" lane) d.bits in
        let y = input (Printf.sprintf "y%d" lane) d.bits in
        let delta =
          sresize x ~width:(d.bits + 1) -: sresize y ~width:(d.bits + 1)
        in
        let square = delta *+ delta in
        if d.pipeline then reg spec ~enable:valid square else square)
  in
  let valid, first, last =
    if d.pipeline then (reg spec valid, reg spec first, reg spec last)
    else (valid, first, last)
  in
  let width = output_bits d in
  let rec sum = function
    | [] -> zero width
    | [ x ] -> uresize x ~width
    | xs ->
        let rec pairs = function
          | a :: b :: rest ->
              (uresize a ~width +: uresize b ~width) :: pairs rest
          | xs -> xs
        in
        sum (pairs xs)
  in
  let chunk = sum products in
  let accumulator =
    reg_fb spec ~enable:valid ~width ~f:(fun previous ->
        mux2 first chunk (previous +: chunk))
  in
  Circuit.create_exn ~name:(name d)
    [
      output "distance" accumulator;
      output "valid_out" (reg spec (valid &: last));
    ]

let rtl circuit =
  Rtl.create Verilog [ circuit ] |> Rtl.full_hierarchy |> Rope.to_string

type sample = { x : float array; y : float array }
type observation = { raw : int; value : float; reference : float; cycle : int }

(* Includes exact extrema, equality, alternating signs, near-zero and random
   inputs. Every design in a run sees the same floating-point corpus. *)
let corpus ~dimension ~samples ~seed =
  if dimension < 1 || dimension > 1024 || samples < 8 || samples > 10000 then
    invalid_arg "corpus requires dimension 1..1024 and samples 8..10000";
  let rng = Random.State.make [| seed |] in
  Array.init samples (fun row ->
      let coordinate i =
        match row with
        | 0 -> (0., 0.)
        | 1 -> (1., -1.)
        | 2 -> (-1., 1.)
        | 3 -> (0.25, 0.25)
        | 4 -> if i mod 2 = 0 then (1., -1.) else (-1., 1.)
        | 5 -> (0.0001, -0.0001)
        | _ -> (Random.State.float rng 2. -. 1., Random.State.float rng 2. -. 1.)
      in
      let pairs = Array.init dimension coordinate in
      { x = Array.map fst pairs; y = Array.map snd pairs })

let reference sample =
  Array.fold_left ( +. ) 0.
    (Array.mapi
       (fun i x ->
         let delta = x -. sample.y.(i) in
         delta *. delta)
       sample.x)

let integer_reference d sample =
  Array.fold_left ( + ) 0
    (Array.mapi
       (fun i x ->
         let delta = quantize d x - quantize d sample.y.(i) in
         delta * delta)
       sample.x)

let simulate ?sim ?(bubbles = false) d circuit samples =
  validate d;
  Array.iter
    (fun s ->
      if Array.length s.x <> d.dimension || Array.length s.y <> d.dimension then
        invalid_arg "sample dimension mismatch";
      Array.iter (fun x -> ignore (quantize d x)) s.x;
      Array.iter (fun x -> ignore (quantize d x)) s.y)
    samples;
  let sim =
    match sim with Some sim -> sim | None -> Cyclesim.create circuit
  in
  let port = Cyclesim.in_port sim in
  let clear = port "clear" and valid = port "valid" in
  let first = port "first" and last = port "last" in
  let xs = Array.init d.lanes (fun i -> port (Printf.sprintf "x%d" i)) in
  let ys = Array.init d.lanes (fun i -> port (Printf.sprintf "y%d" i)) in
  let result = Cyclesim.out_port sim "distance" in
  let ready = Cyclesim.out_port sim "valid_out" in
  Cyclesim.reset sim;
  clear := Bits.vdd;
  Cyclesim.cycle sim;
  clear := Bits.gnd;
  let cycles = ref 0 and next = ref 0 and observed = ref [] in
  let tick () =
    Cyclesim.cycle sim;
    incr cycles;
    if Bits.to_unsigned_int !ready <> 0 then (
      if !next >= Array.length samples then failwith "unexpected circuit output";
      let sample = samples.(!next) in
      let raw = Bits.to_unsigned_int !result in
      let expected = integer_reference d sample in
      if raw <> expected then
        failwith
          (Printf.sprintf "%s sample %d: RTL=%d reference=%d" (name d) !next raw
             expected);
      let value = float_of_int raw /. (scale d *. scale d) in
      observed :=
        { raw; value; reference = reference sample; cycle = !cycles }
        :: !observed;
      incr next)
  in
  let chunks = (d.dimension + d.lanes - 1) / d.lanes in
  Array.iteri
    (fun row sample ->
      for chunk = 0 to chunks - 1 do
        if bubbles && (row + chunk) mod 3 = 0 then (
          valid := Bits.gnd;
          tick ());
        valid := Bits.vdd;
        first := Bits.of_int_trunc ~width:1 (if chunk = 0 then 1 else 0);
        last := Bits.of_int_trunc ~width:1 (if chunk = chunks - 1 then 1 else 0);
        Array.iteri
          (fun lane x ->
            let i = (chunk * d.lanes) + lane in
            let q a = if i < d.dimension then quantize d a.(i) else 0 in
            x := Bits.of_int_trunc ~width:d.bits (q sample.x);
            ys.(lane) := Bits.of_int_trunc ~width:d.bits (q sample.y))
          xs;
        tick ()
      done)
    samples;
  valid := Bits.gnd;
  if d.pipeline then tick ();
  let completion_cycles = !cycles in
  (* Flush twice to detect duplicate or stale valid pulses. *)
  tick ();
  tick ();
  if !next <> Array.length samples then failwith "circuit dropped outputs";
  (List.rev !observed, completion_cycles)

let utilization circuit =
  Circuit_utilization.create circuit |> Circuit_utilization.sexp_of_t

let resource_bits circuit category =
  let open Base.Sexp in
  let field key = function
    | List fields ->
        List.find_map
          (function
            | List [ Atom k; value ] when String.equal k key -> Some value
            | _ -> None)
          fields
    | Atom _ -> None
  in
  match
    Option.bind (field category (utilization circuit)) (field "total_bits")
  with
  | Some (Atom n) -> int_of_string n
  | _ -> 0

let evaluate d samples circuit =
  let observations, cycles = simulate d circuit samples in
  let max_error =
    List.fold_left
      (fun worst o ->
        max worst
          (abs_float (o.value -. o.reference) /. float_of_int d.dimension))
      0. observations
  in
  (observations, cycles, max_error)
