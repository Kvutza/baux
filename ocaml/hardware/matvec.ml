open Hardcaml

type design = {
  rows : int;
  cols : int;
  bits : int;
  lanes : int;
  banks : int;
  pipeline : bool;
  queue : int;
}

let ceil_div n d = (n + d - 1) / d
let log2 = Distance.ceil_log2
let index_bits n = max 1 (log2 n)
let power2 n = n > 0 && n land (n - 1) = 0

let validate d =
  if d.rows < 1 || d.rows > 256 || d.cols < 1 || d.cols > 4096 then
    invalid_arg "matrix shape must be in rows 1..256, columns 1..4096";
  if d.bits < 4 || d.bits > 20 then invalid_arg "precision must be in 4..20";
  List.iter
    (fun (name, n) ->
      if (not (power2 n)) || n > 32 then
        invalid_arg (name ^ " must be a power of two in 1..32"))
    [ ("lanes", d.lanes); ("banks", d.banks); ("queue", d.queue) ]

let name d =
  Printf.sprintf "matvec_r%d_c%d_b%d_l%d_m%d_p%d_q%d" d.rows d.cols d.bits
    d.lanes d.banks
    (if d.pipeline then 1 else 0)
    d.queue

let words d = ceil_div d.cols d.banks
let tiles d = ceil_div d.cols d.lanes
let passes d = ceil_div d.lanes d.banks
let output_bits d = (2 * d.bits) + log2 d.cols
let scale d = float_of_int (1 lsl (d.bits - 2))

let quantize d x =
  if (not (Float.is_finite x)) || x < -1. || x > 1. then
    invalid_arg "matrix and input entries must be finite and in [-1,1]";
  int_of_float (Float.round (x *. scale d))

let sum result_width xs =
  let open Signal in
  let rec tree = function
    | [] -> zero result_width
    | [ x ] -> sresize x ~width:result_width
    | xs ->
        let rec pairs = function
          | a :: b :: rest ->
              (sresize a ~width:result_width +: sresize b ~width:result_width)
              :: pairs rest
          | xs -> xs
        in
        tree (pairs xs)
  in
  tree xs

type fifo = {
  data : Signal.t;
  valid : Signal.t;
  push : Signal.t;
  used : Signal.t;
}

let fifo spec ~depth ~clear ~offer ~value ~ready =
  let open Signal in
  let open Always in
  let head = Variable.reg spec ~width:(index_bits depth) in
  let tail = Variable.reg spec ~width:(index_bits depth) in
  let count = Variable.reg spec ~width:(index_bits (depth + 1)) in
  let valid = count.value <>:. 0 &: ~:clear in
  let pop = valid &: ready in
  let push = offer &: (count.value <:. depth |: pop) &: ~:clear in
  let storage =
    List.init depth (fun i ->
        reg spec ~enable:(push &: (head.value ==:. i)) value)
  in
  let advance v = if depth = 1 then v <--. 0 else v <-- v.value +:. 1 in
  compile
    [
      when_ push [ advance head ];
      when_ pop [ advance tail ];
      when_ (push &: ~:pop) [ count <-- count.value +:. 1 ];
      when_ (pop &: ~:push) [ count <-- count.value -:. 1 ];
    ];
  { data = mux tail.value storage; valid; push; used = count.value }

(* One write and one registered read per bank. Programming is accepted only
   while idle with an empty output FIFO. All matrix/input words must be loaded
   before start; clear aborts computation and queued outputs, but retains RAM. *)
let create d =
  validate d;
  let open Signal in
  let open Always in
  let clock = input "clock" 1 and clear = input "clear" 1 in
  let spec = Reg_spec.create ~clock ~clear () in
  let state = Variable.reg spec ~width:3 in
  let row = Variable.reg spec ~width:(index_bits d.rows) in
  let tile = Variable.reg spec ~width:(index_bits (tiles d)) in
  let pass = Variable.reg spec ~width:(index_bits (passes d)) in
  let accumulator = Variable.reg spec ~width:(output_bits d) in
  let idle = state.value ==:. 0 and gather = state.value ==:. 1 in
  let multiply = state.value ==:. 2 and emit = state.value ==:. 4 in
  let out_ready = input "out_ready" 1 in
  let queued =
    fifo spec ~depth:d.queue ~clear ~offer:emit
      ~value:(concat_msb [ row.value; accumulator.value ])
      ~ready:out_ready
  in
  let available = idle &: (queued.used ==:. 0) &: ~:clear in
  let w_valid = input "w_valid" 1 and x_valid = input "x_valid" 1 in
  let w_row = input "w_row" (index_bits d.rows) in
  let w_word = input "w_word" (index_bits (words d)) in
  let x_word = input "x_word" (index_bits (words d)) in
  let below signal limit =
    if 1 lsl Signal.width signal <= limit then vdd else signal <:. limit
  in
  let w_ready = available &: below w_row d.rows &: below w_word (words d) in
  let x_ready = available &: below x_word (words d) in
  let start_ready = available &: ~:w_valid &: ~:x_valid in
  let start = input "start" 1 &: start_ready in
  let col_width = index_bits (d.cols + d.lanes + d.banks) in
  let base =
    uresize tile.value ~width:col_width
    *: of_int_trunc ~width:col_width d.lanes
    +: uresize pass.value ~width:col_width
       *: of_int_trunc ~width:col_width (min d.lanes d.banks)
    |> fun x -> uresize x ~width:col_width
  in
  let word =
    srl base ~by:(log2 d.banks) |> fun x ->
    uresize x ~width:(index_bits (words d))
  in
  let word = mux2 (base <:. d.cols) word (zero (index_bits (words d))) in
  let memory_width = index_bits (d.rows * words d) in
  let address r w =
    uresize
      (uresize
         (uresize r ~width:memory_width
         *: of_int_trunc ~width:memory_width (words d))
         ~width:memory_width
      +: uresize w ~width:memory_width)
      ~width:memory_width
  in
  let weights, inputs =
    List.init d.banks (fun bank ->
        let ram size write_address write_enable write_data read_address =
          memory size
            ~write_port:
              {
                Write_port.write_clock = clock;
                write_address;
                write_enable;
                write_data;
              }
            ~read_address
        in
        ( ram
            (d.rows * words d)
            (address w_row w_word) (w_valid &: w_ready)
            (input (Printf.sprintf "w%d" bank) d.bits)
            (address row.value word),
          ram (words d) x_word (x_valid &: x_ready)
            (input (Printf.sprintf "x%d" bank) d.bits)
            word ))
    |> List.split
  in
  let operands =
    List.init d.lanes (fun lane ->
        let ports = min d.lanes d.banks in
        let col = base +:. (lane mod ports) in
        let bank =
          if d.banks = 1 then zero 1 else uresize col ~width:(log2 d.banks)
        in
        let capture = gather &: (pass.value ==:. lane / ports) in
        let read memories =
          reg spec ~enable:capture
            (mux2 (col <:. d.cols) (mux bank memories) (zero d.bits))
        in
        (read weights, read inputs))
  in
  let products =
    List.map
      (fun (w, x) ->
        let product = w *+ x in
        if d.pipeline then reg spec ~enable:multiply product else product)
      operands
  in
  let total = sum (output_bits d) products in
  let accumulate =
    [
      accumulator <-- mux2 (tile.value ==:. 0) total (accumulator.value +: total);
      if_
        (tile.value ==:. tiles d - 1)
        [ state <--. 4 ]
        [ tile <-- tile.value +:. 1; pass <--. 0; state <--. 1 ];
    ]
  in
  compile
    [
      when_ start
        [
          state <--. 1; row <--. 0; tile <--. 0; pass <--. 0; accumulator <--. 0;
        ];
      when_ gather
        [
          if_
            (pass.value ==:. passes d - 1)
            [ state <--. 2 ]
            [ pass <-- pass.value +:. 1 ];
        ];
      when_ multiply (if d.pipeline then [ state <--. 3 ] else accumulate);
      when_ (state.value ==:. 3) accumulate;
      when_ queued.push
        [
          if_
            (row.value ==:. d.rows - 1)
            [ state <--. 0 ]
            [
              row <-- row.value +:. 1;
              tile <--. 0;
              pass <--. 0;
              accumulator <--. 0;
              state <--. 1;
            ];
        ];
    ];
  let result = select queued.data ~high:(output_bits d - 1) ~low:0 in
  let result_row =
    select queued.data ~high:(Signal.width queued.data - 1) ~low:(output_bits d)
  in
  Circuit.create_exn ~name:(name d)
    [
      output "w_ready" w_ready;
      output "x_ready" x_ready;
      output "start_ready" start_ready;
      output "busy" (~:idle |: queued.valid);
      output "out_valid" queued.valid;
      output "out_value" result;
      output "out_row" result_row;
      output "out_last" (result_row ==:. d.rows - 1);
      output "queue_used" queued.used;
      output "read_cycle" gather;
      output "bank_wait" (gather &: (pass.value <>:. 0));
      output "mac_cycle" (multiply |: (state.value ==:. 3));
      output "fifo_stall" (emit &: ~:(queued.push));
    ]

let rtl circuit = Distance.rtl circuit
