open Baux_benchmarks
let unsafe (workspace : Batch.workspace) : (unit -> unit) @ portable =
  fun () -> ignore (Batch.get workspace 0 : float#)
