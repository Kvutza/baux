module C = Flow_core

let cycles d period ~dot n =
  let chunks = Matvec.ceil_div n d.C.lanes in
  let outputs = if dot then 1 else chunks in
  let read_cost chunk =
    Matvec.ceil_div (2 * min d.lanes (n - (chunk * d.lanes))) d.banks
  in
  let queue = ref 0 and stage = ref false in
  let sent = ref 0 and received = ref 0 and gap = ref (read_cost 0) in
  let elapsed = ref 0 and stalls = ref 0 in
  while !received < outputs do
    let consumer = !elapsed mod period = 0 in
    let pop = !queue > 0 && consumer in
    let space = !queue < d.queue || pop in
    let ready = if d.pipeline then (not !stage) || space else space in
    let valid = !sent < chunks && !gap = 0 in
    if !queue > 0 && not consumer then incr stalls;
    if pop then (
      decr queue;
      incr received);
    let complete = valid && ready && ((not dot) || !sent = chunks - 1) in
    if d.pipeline then (
      if !stage && space then incr queue;
      if ready then stage := complete)
    else if complete then incr queue;
    if valid && ready then (
      incr sent;
      if !sent < chunks then gap := read_cost !sent)
    else if !gap > 0 then decr gap;
    incr elapsed
  done;
  (!elapsed, !stalls)

let memo d period =
  let cache = Hashtbl.create 64 in
  fun ~dot n ->
    match Hashtbl.find_opt cache (dot, n) with
    | Some value -> value
    | None ->
        let value = cycles d period ~dot n in
        Hashtbl.add cache (dot, n) value;
        value
