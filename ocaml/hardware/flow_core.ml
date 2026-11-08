open Hardcaml
module N = Flow_numeric

type design = {
  bits : int;
  lanes : int;
  banks : int;
  pipeline : bool;
  queue : int;
}

let validate d =
  if d.bits < 20 || d.bits > 28 then invalid_arg "flow precision must be 20..28";
  List.iter
    (fun n ->
      if (not (Matvec.power2 n)) || n > 32 then
        invalid_arg "flow lanes, banks and queue must be powers of two in 1..32")
    [ d.lanes; d.banks; d.queue ]

let name d =
  Printf.sprintf "flow_core_b%d_l%d_m%d_p%d_q%d" d.bits d.lanes d.banks
    (Bool.to_int d.pipeline) d.queue

let saturate bits x =
  let open Signal in
  let w = width x in
  let lo = of_int_trunc ~width:w (-(1 lsl (bits - 1))) in
  let hi = of_int_trunc ~width:w (N.bound bits) in
  uresize (mux2 (x <+ hi) (mux2 (x <+ lo) lo x) hi) ~width:bits

let lookup bits address values =
  (Signal.rom ~read_addresses:[| address |]
     (Array.map (Bits.of_int_trunc ~width:bits) values)).(0)

let nonlinear bits op a =
  let open Signal in
  let f = N.frac bits in
  let radius = (if op = N.Exp then 16 else 8) * N.scale bits in
  let offset = sresize a ~width:(bits + 1) +:. radius in
  let index = select offset ~high:(f + 4) ~low:(f - 4) in
  let index = mux2 (index >:. 256) (zero 9) index in
  let y = lookup bits index (N.table bits op) in
  let upper = if op = N.Exp then 0 else radius in
  mux2
    (a <+ of_int_trunc ~width:bits (-radius))
    (zero bits)
    (mux2
       (a >+ of_int_trunc ~width:bits upper)
       (if op = N.Exp then of_int_trunc ~width:bits (N.scale bits) else a)
       y)

let inverse bits square a =
  let open Signal in
  let f = N.frac bits in
  let index = ref (zero 8)
  and select_bit = ref (zero (Matvec.index_bits bits)) in
  for b = 0 to bits - 2 do
    let candidate = if b >= 7 then srl a ~by:(b - 7) else sll a ~by:(7 - b) in
    let set = bit a ~pos:b in
    index := mux2 set (uresize candidate ~width:8) !index;
    select_bit :=
      mux2 set (of_int_trunc ~width:(Matvec.index_bits bits) b) !select_bit
  done;
  let even = lookup bits !index (N.inverse_table bits square false) in
  let odd = lookup bits !index (N.inverse_table bits square true) in
  let values =
    List.init (bits - 1) (fun b ->
        let exponent = b - f in
        let coefficient = if square && exponent mod 2 <> 0 then odd else even in
        let power = if square then N.floor_half exponent else exponent in
        let wide = uresize coefficient ~width:(2 * bits) in
        saturate bits
          (if power >= 0 then srl wide ~by:power else sll wide ~by:(-power)))
  in
  mux2
    (a <=+ zero bits)
    (of_int_trunc ~width:bits (N.bound bits))
    (mux !select_bit values)

let create d =
  validate d;
  let open Signal in
  let open Always in
  let clock = input "clock" 1 and clear = input "clear" 1 in
  let spec = Reg_spec.create ~clock ~clear () in
  let valid = input "valid" 1 and ready = wire 1 in
  let op = input "op" 4 in
  let first = input "first" 1 and last = input "last" 1 in
  let dot = op ==:. N.code N.Dot in
  let acc = Variable.reg spec ~width:(d.bits + 16) in
  let products = ref [] in
  let values =
    List.init d.lanes (fun lane ->
        let a = input (Printf.sprintf "a%d" lane) d.bits in
        let b = input (Printf.sprintf "b%d" lane) d.bits in
        let product = saturate d.bits (sra (a *+ b) ~by:(N.frac d.bits)) in
        products := product :: !products;
        mux op
          [
            saturate d.bits
              (sresize a ~width:(d.bits + 1) +: sresize b ~width:(d.bits + 1));
            saturate d.bits
              (sresize a ~width:(d.bits + 1) -: sresize b ~width:(d.bits + 1));
            product;
            mux2 (a >+ b) a b;
            nonlinear d.bits N.Exp a;
            inverse d.bits false a;
            inverse d.bits true a;
            nonlinear d.bits N.Gelu a;
            nonlinear d.bits N.Silu a;
            zero d.bits;
          ])
  in
  let total = Matvec.sum (d.bits + 16) !products in
  let accumulated = mux2 first total (acc.value +: total) in
  let accept = valid &: ready &: ~:clear in
  compile [ when_ (accept &: dot) [ acc <-- accumulated ] ];
  let completed = accept &: (~:dot |: last) in
  let packet =
    mux2 dot
      (uresize (saturate d.bits accumulated) ~width:(d.bits * d.lanes))
      (concat_lsb values)
  in
  let output_ready = input "out_ready" 1 in
  let stage_valid = Variable.reg spec ~width:1 in
  let stage_data = Variable.reg spec ~width:(d.bits * d.lanes) in
  let queued =
    Matvec.fifo spec ~depth:d.queue ~clear
      ~offer:(if d.pipeline then stage_valid.value else completed)
      ~value:(if d.pipeline then stage_data.value else packet)
      ~ready:output_ready
  in
  let space = queued.used <:. d.queue |: output_ready in
  let input_ready =
    (if d.pipeline then ~:(stage_valid.value) |: space else space) &: ~:clear
  in
  assign ready input_ready;
  if d.pipeline then
    compile
      [
        when_ input_ready
          [
            stage_valid <-- completed; when_ completed [ stage_data <-- packet ];
          ];
      ];
  Circuit.create_exn ~name:(name d)
    ([
       output "ready" input_ready;
       output "out_valid" queued.valid;
       output "queue_used" queued.used;
     ]
    @ List.init d.lanes (fun i ->
          output (Printf.sprintf "y%d" i)
            (select queued.data
               ~high:(((i + 1) * d.bits) - 1)
               ~low:(i * d.bits))))
