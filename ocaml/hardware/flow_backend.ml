open Hardcaml
module N = Flow_numeric
module C = Flow_core

type mode = Float | Integer | Rtl | Verilator

let mode = function
  | "float" -> Float
  | "integer" -> Integer
  | "rtl" -> Rtl
  | "verilator" -> Verilator
  | _ -> invalid_arg "unknown flow backend"

type stats = {
  mutable cycles : int;
  mutable operands : int;
  mutable operations : int;
  mutable outputs : int;
  mutable stalls : int;
  mutable kernels : int;
  mutable checked : int;
}

type t = {
  design : C.design;
  mode : mode;
  period : int;
  stats : stats;
  sim : Cyclesim.t_port_list option;
  tables : int array array;
  estimate : dot:bool -> int -> int * int;
}

let reset_sim = function
  | None -> ()
  | Some sim ->
      Cyclesim.in_port sim "valid" := Bits.gnd;
      Cyclesim.in_port sim "out_ready" := Bits.gnd;
      Cyclesim.in_port sim "clear" := Bits.vdd;
      Cyclesim.cycle sim;
      Cyclesim.in_port sim "clear" := Bits.gnd

let create ?(period = 1) mode design =
  C.validate design;
  if period < 1 || period > 1000 then invalid_arg "invalid memory ready period";
  let sim =
    match mode with
    | Rtl -> Some (Cyclesim.create (C.create design))
    | Verilator -> Some (Hardcaml_verilator.create (C.create design))
    | Float | Integer -> None
  in
  reset_sim sim;
  {
    design;
    mode;
    period;
    tables = N.tables design.bits;
    estimate = Flow_schedule.memo design period;
    stats =
      {
        cycles = 1;
        operands = 0;
        operations = 0;
        outputs = 0;
        stalls = 0;
        kernels = 0;
        checked = 0;
      };
    sim;
  }

let reset_stats t =
  reset_sim t.sim;
  let s = t.stats in
  s.cycles <- 1;
  s.operands <- 0;
  s.operations <- 0;
  s.outputs <- 0;
  s.stalls <- 0;
  s.kernels <- 0;
  s.checked <- 0

let json t =
  let s = t.stats in
  `Assoc
    [
      ( "backend",
        `String
          (match t.mode with
          | Float -> "float"
          | Integer -> "integer"
          | Rtl -> "rtl"
          | Verilator -> "verilator") );
      ("cycles", `Int s.cycles);
      ("operand_words", `Int s.operands);
      ("scalar_operations", `Int s.operations);
      ("output_words", `Int s.outputs);
      ("stall_cycles", `Int s.stalls);
      ("kernels", `Int s.kernels);
      ("rtl_checked_words", `Int s.checked);
    ]

let calculate t op a b =
  if t.mode = Float then Array.mapi (fun i x -> N.exact op x b.(i)) a
  else
    Array.mapi
      (fun i x ->
        let x = N.quantize t.design.bits x in
        let raw =
          match op with
          | N.Exp | N.Inv | N.Rsqrt | N.Gelu | N.Silu ->
              N.unary t.design.bits t.tables op x
          | _ -> N.binary t.design.bits op x (N.quantize t.design.bits b.(i))
        in
        N.real t.design.bits raw)
      a

(* The host submits kernels to a streaming datapath. External SRAM is modelled
   as bank-limited transfers; no vendor DDR, PCIe or FPGA clock is assumed. *)
let execute t op a b expected =
  let d = t.design in
  let n = Array.length a in
  if n = 0 || n <> Array.length b || n > 65536 then
    invalid_arg "invalid flow kernel shape";
  let dot = op = N.Dot in
  let chunks = Matvec.ceil_div n d.lanes in
  let count = if dot then 1 else chunks in
  let s = t.stats in
  s.kernels <- s.kernels + 1;
  s.operands <- s.operands + (2 * n);
  s.operations <- s.operations + n;
  s.outputs <- s.outputs + Array.length expected;
  if t.mode = Integer then (
    let elapsed, stalls = t.estimate ~dot n in
    s.cycles <- s.cycles + elapsed;
    s.stalls <- s.stalls + stalls;
    expected)
  else
    let produced = Array.copy expected in
    let get, set, tick =
      match t.sim with
      | None -> ((fun _ -> 0), (fun _ _ -> ()), fun () -> ())
      | Some sim ->
          let inputs =
            List.map
              (fun (name, p) -> (name, (p, Bits.width !p)))
              (Cyclesim.in_ports sim)
          in
          let outputs =
            List.map
              (fun (name, p) -> (name, p))
              (Cyclesim.out_ports ~clock_edge:Before sim)
          in
          let get key =
            let value = !(List.assoc key outputs) in
            Bits.to_int_trunc
              (if key.[0] = 'y' then Bits.sresize value ~width:63 else value)
          in
          let set key value =
            let p, w = List.assoc key inputs in
            p := Bits.of_int_trunc ~width:w value
          in
          (get, set, fun () -> Cyclesim.cycle sim)
    in
    let read_cost chunk =
      Matvec.ceil_div (2 * min d.lanes (n - (chunk * d.lanes))) d.banks
    in
    let queued = Queue.create () in
    let stage = ref None in
    let sent = ref 0 and received = ref 0 and gap = ref (read_cost 0) in
    let elapsed = ref 0 and loaded = ref (-1) in
    set "clear" 0;
    set "op" (N.code op);
    while !received < count do
      let consumer = !elapsed mod t.period = 0 in
      let out_valid = not (Queue.is_empty queued) in
      let pop = out_valid && consumer in
      let space = Queue.length queued < d.queue || pop in
      let ready = if d.pipeline then !stage = None || space else space in
      let input_valid = !sent < chunks && !gap = 0 in
      if t.sim <> None && input_valid && !loaded <> !sent then (
        for lane = 0 to d.lanes - 1 do
          let i = (!sent * d.lanes) + lane in
          set
            (Printf.sprintf "a%d" lane)
            (if i < n then N.quantize d.bits a.(i) else 0);
          set
            (Printf.sprintf "b%d" lane)
            (if i < n then N.quantize d.bits b.(i) else 0)
        done;
        set "first" (Bool.to_int (!sent = 0));
        set "last" (Bool.to_int (!sent = chunks - 1));
        loaded := !sent);
      set "valid" (Bool.to_int input_valid);
      set "out_ready" (Bool.to_int consumer);
      tick ();
      (match t.sim with
      | None -> ()
      | Some _ ->
          if
            get "ready" <> Bool.to_int ready
            || get "out_valid" <> Bool.to_int out_valid
          then failwith "flow handshake disagrees with independent scheduler");
      if pop then (
        let chunk = Queue.take queued in
        let width = if dot then 1 else min d.lanes (n - (chunk * d.lanes)) in
        for lane = 0 to width - 1 do
          let i = if dot then 0 else (chunk * d.lanes) + lane in
          match t.sim with
          | None -> ()
          | Some _ ->
              let actual = get (Printf.sprintf "y%d" lane) in
              if actual <> N.quantize d.bits expected.(i) then
                failwith
                  (Printf.sprintf
                     "flow RTL mismatch op=%d lane=%d actual=%d expected=%d"
                     (N.code op) lane actual
                     (N.quantize d.bits expected.(i)));
              produced.(i) <- N.real d.bits actual;
              s.checked <- s.checked + 1
        done;
        incr received);
      if out_valid && not consumer then s.stalls <- s.stalls + 1;
      let completed =
        input_valid && ready && ((not dot) || !sent = chunks - 1)
      in
      if d.pipeline then (
        (match !stage with
        | Some chunk when space -> Queue.add chunk queued
        | _ -> ());
        if ready then stage := if completed then Some !sent else None)
      else if completed then Queue.add !sent queued;
      if input_valid && ready then (
        incr sent;
        if !sent < chunks then gap := read_cost !sent)
      else if !gap > 0 then decr gap;
      incr elapsed
    done;
    set "valid" 0;
    set "out_ready" 0;
    s.cycles <- s.cycles + !elapsed;
    produced

let binary t op a b =
  let result = calculate t op a b in
  if t.mode = Float then result else execute t op a b result

let unary t op a = binary t op a (Array.make (Array.length a) 0.)

let dot t a b =
  let products = calculate t N.Dot a b in
  let total = Array.fold_left ( +. ) 0. products in
  let result =
    if t.mode = Float then total
    else N.real t.design.bits (N.quantize t.design.bits total)
  in
  if t.mode = Float then result else (execute t N.Dot a b [| result |]).(0)

let scalar t op a b = (binary t op [| a |] [| b |]).(0)
let sum t a = dot t a (Array.make (Array.length a) 1.)

let max t a =
  let rec reduce a =
    if Array.length a = 1 then a.(0)
    else
      let n = Matvec.ceil_div (Array.length a) 2 in
      let left = Array.init n (fun i -> a.(2 * i)) in
      let right =
        Array.init n (fun i -> a.(min ((2 * i) + 1) (Array.length a - 1)))
      in
      reduce (binary t N.Max left right)
  in
  reduce a
