open Baux_pybridge

let stashed : Py_obj.t option ref = ref None
let unsafe () = Py_obj.Def.register1 (fun arg -> stashed := Some arg; Py_obj.of_int 0) 0
