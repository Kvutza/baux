module M = Flow_model
module C = Flow_core

let profile c d period samples =
  M.validate c;
  C.validate d;
  let estimate = Flow_schedule.memo d period in
  let cycles = ref 1
  and operations = ref 0
  and operands = ref 0
  and outputs = ref 0
  and kernels = ref 0 in
  let emit ?(dot = false) repeat n =
    let cost, _ = estimate ~dot n in
    cycles := !cycles + (repeat * cost);
    operations := !operations + (repeat * n);
    operands := !operands + (repeat * 2 * n);
    outputs := !outputs + (repeat * if dot then 1 else n);
    kernels := !kernels + repeat
  in
  let linear count rows cols =
    emit ~dot:true (count * rows) cols;
    emit count rows
  in
  let width = c.M.width and tokens = M.tokens c in
  let norm count =
    emit ~dot:true (2 * count) width;
    emit (3 * count) width;
    emit (2 * count) 1
  in
  let modulate count =
    norm count;
    emit (3 * count) width
  in
  linear (samples * tokens) width 9;
  emit (2 * samples * tokens) width;
  linear samples width width;
  emit samples width;
  linear samples width width;
  for _layer = 1 to c.depth do
    emit samples width;
    linear samples (6 * width) width;
    modulate (samples * tokens);
    linear (samples * tokens) (3 * width) width;
    let rows = samples * c.heads * tokens in
    emit ~dot:true (rows * tokens) (width / c.heads);
    emit rows tokens;
    let rec maximum n =
      if n > 1 then (
        let size = Matvec.ceil_div n 2 in
        emit rows size;
        maximum size)
    in
    maximum tokens;
    emit (3 * rows) tokens;
    emit ~dot:true rows tokens;
    emit rows 1;
    emit ~dot:true (samples * tokens * width) tokens;
    linear (samples * tokens) width width;
    emit (2 * samples * tokens) width;
    modulate (samples * tokens);
    linear (samples * tokens) (c.expansion * width) width;
    emit (samples * tokens) (c.expansion * width);
    linear (samples * tokens) width (c.expansion * width);
    emit (2 * samples * tokens) width
  done;
  emit samples width;
  linear samples (2 * width) width;
  modulate (samples * tokens);
  linear (samples * tokens) 2 width;
  let load = Matvec.ceil_div (M.parameters c) d.banks in
  let io = Matvec.ceil_div (tokens * 9 * samples) d.banks in
  `Assoc
    [
      ("cycles", `Int (!cycles + load + io));
      ("kernel_cycles", `Int !cycles);
      ("weight_load_cycles", `Int load);
      ("input_load_cycles", `Int io);
      ("scalar_operations", `Int !operations);
      ("operand_words", `Int !operands);
      ("output_words", `Int !outputs);
      ("kernels", `Int !kernels);
      ("model", M.profile c);
      ( "scope",
        `String "static integer/RTL-kernel schedule; not physical timing" );
    ]
