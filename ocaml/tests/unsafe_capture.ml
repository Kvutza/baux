let counter = ref 0
let unsafe : (unit -> unit) @ portable = fun () -> incr counter
