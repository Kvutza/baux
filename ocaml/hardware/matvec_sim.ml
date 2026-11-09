open Hardcaml
module M = Matvec

type schedule = {
  ready_period : int;
  ready_cycles : int;
  initial_stall : int;
  load_gap : int;
}

let validate s =
  if
    s.ready_period < 1 || s.ready_period > 10000 || s.ready_cycles < 1
    || s.ready_cycles > s.ready_period
    || s.initial_stall < 0 || s.initial_stall > 1000000 || s.load_gap < 0
    || s.load_gap > 10000
  then invalid_arg "invalid consumer or programming schedule"

type counts = {
  mutable reset : int;
  mutable weights : int;
  mutable inputs : int;
  mutable compute : int;
  mutable reads : int;
  mutable bank_wait : int;
  mutable mac : int;
  mutable fifo_stall : int;
  mutable output_stall : int;
  mutable queue_peak : int;
}

let counts () =
  {
    reset = 0;
    weights = 0;
    inputs = 0;
    compute = 0;
    reads = 0;
    bank_wait = 0;
    mac = 0;
    fifo_stall = 0;
    output_stall = 0;
    queue_peak = 0;
  }

let cycles c = c.reset + c.weights + c.inputs + c.compute

let to_json c =
  `Assoc
    (List.map
       (fun (k, v) -> (k, `Int v))
       [
         ("reset", c.reset);
         ("weight_load", c.weights);
         ("input_load", c.inputs);
         ("compute", c.compute);
         ("reads", c.reads);
         ("bank_wait", c.bank_wait);
         ("mac", c.mac);
         ("fifo_stall", c.fifo_stall);
         ("output_stall", c.output_stall);
         ("queue_peak", c.queue_peak);
       ])

let integer_reference d weights input =
  Array.init d.M.rows (fun row ->
      let total = ref 0 in
      for col = 0 to d.cols - 1 do
        total :=
          !total
          + M.quantize d weights.((row * d.cols) + col)
            * M.quantize d input.(col)
      done;
      !total)

let reference d weights input =
  Array.init d.M.rows (fun row ->
      let total = ref 0. in
      for col = 0 to d.cols - 1 do
        total := !total +. (weights.((row * d.cols) + col) *. input.(col))
      done;
      !total)

let simulate d sim schedule weights samples =
  M.validate d;
  validate schedule;
  if Array.length weights <> d.M.rows * d.cols then
    invalid_arg "matrix weight dimension mismatch";
  Array.iter (fun x -> ignore (M.quantize d x)) weights;
  Array.iter
    (fun input ->
      if Array.length input <> d.cols then
        invalid_arg "input dimension mismatch";
      Array.iter (fun x -> ignore (M.quantize d x)) input)
    samples;
  let ports = Cyclesim.in_ports sim in
  List.iter (fun (_, value) -> value := Bits.zero (Bits.width !value)) ports;
  let set key value =
    let port = List.assoc key ports in
    port := Bits.of_int_trunc ~width:(Bits.width !port) value
  in
  let get key = !(Cyclesim.out_port ~clock_edge:Before sim key) in
  let int key = Bits.to_int_trunc (get key) in
  let c = counts () in
  let tick () =
    Cyclesim.cycle sim;
    c.reads <- c.reads + int "read_cycle";
    c.bank_wait <- c.bank_wait + int "bank_wait";
    c.mac <- c.mac + int "mac_cycle";
    c.fifo_stall <- c.fifo_stall + int "fifo_stall";
    c.queue_peak <- max c.queue_peak (int "queue_used")
  in
  set "clear" 1;
  tick ();
  c.reset <- 1;
  set "clear" 0;
  let load ~prefix ~row values =
    if prefix = "w_" then set "w_row" row;
    for word = 0 to M.words d - 1 do
      set (prefix ^ "word") word;
      for bank = 0 to d.banks - 1 do
        let col = (word * d.banks) + bank in
        let v = if col < d.cols then M.quantize d values.(col) else 0 in
        set (Printf.sprintf "%s%d" (String.sub prefix 0 1) bank) v
      done;
      set (prefix ^ "valid") 1;
      tick ();
      if int (prefix ^ "ready") <> 1 then failwith "programming rejected";
      set (prefix ^ "valid") 0;
      for _ = 1 to schedule.load_gap do
        tick ()
      done
    done
  in
  for row = 0 to d.rows - 1 do
    load ~prefix:"w_" ~row (Array.sub weights (row * d.cols) d.cols)
  done;
  c.weights <- d.rows * M.words d * (1 + schedule.load_gap);
  let observations =
    Array.map
      (fun input ->
        load ~prefix:"x_" ~row:0 input;
        c.inputs <- c.inputs + (M.words d * (1 + schedule.load_gap));
        let expected = integer_reference d weights input in
        let result = Array.make d.rows 0. in
        set "start" 1;
        set "out_ready" 0;
        tick ();
        c.compute <- c.compute + 1;
        if int "start_ready" <> 1 then failwith "start rejected";
        set "start" 0;
        let received = ref 0 and elapsed = ref 0 and held = ref None in
        let limit =
          schedule.initial_stall
          + d.rows
            * ((M.tiles d * (M.passes d + 3)) + 3)
            * schedule.ready_period
          + 10
        in
        while !received < d.rows do
          if !elapsed > limit then failwith "matrix simulator timed out";
          let ready =
            !elapsed >= schedule.initial_stall
            && (!elapsed - schedule.initial_stall) mod schedule.ready_period
               < schedule.ready_cycles
          in
          set "out_ready" (if ready then 1 else 0);
          tick ();
          c.compute <- c.compute + 1;
          let valid = int "out_valid" = 1 in
          let raw =
            Bits.to_int_trunc (Bits.sresize (get "out_value") ~width:63)
          in
          let tag = int "out_row" in
          let last = int "out_last" in
          (match !held with
          | Some previous when (not valid) || previous <> (raw, tag, last) ->
              failwith "output changed while backpressured"
          | _ -> ());
          held := if valid && not ready then Some (raw, tag, last) else None;
          if valid && not ready then c.output_stall <- c.output_stall + 1;
          if valid && ready then (
            if tag <> !received || last <> if tag = d.rows - 1 then 1 else 0
            then failwith "matrix output ordering violation";
            if raw <> expected.(tag) then
              failwith
                (Printf.sprintf "integer mismatch %s row %d: %d <> %d"
                   (M.name d) tag raw expected.(tag));
            result.(tag) <- float_of_int raw /. (M.scale d *. M.scale d);
            incr received);
          incr elapsed
        done;
        set "out_ready" 0;
        result)
      samples
  in
  (observations, c)
