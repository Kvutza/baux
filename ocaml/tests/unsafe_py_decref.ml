open Baux_pybridge

let unsafe () : Py_obj.t @ unique =
  let fresh = Py_obj.of_float 1.0 in
  Py_obj.decref fresh;
  fresh
