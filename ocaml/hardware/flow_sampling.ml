module B = Flow_backend
module M = Flow_model
module N = Flow_numeric

let combine b x terms =
  let flat x = Array.concat (Array.to_list x) in
  let result =
    List.fold_left
      (fun value (factor, term) ->
        B.binary b N.Add value (M.scale b (flat term) factor))
      (flat x) terms
  in
  Array.init (Array.length x) (fun i -> Array.sub result (2 * i) 2)

let generate b c weights ~condition ~noise ~method_ ~steps ~guidance =
  if
    steps < 1 || steps > 1000
    || (not (List.mem method_ [ "euler"; "heun"; "rk4" ]))
    || (not (Float.is_finite guidance))
    || guidance < 0. || guidance > 10.
  then invalid_arg "invalid FlowLOB sampler";
  let nfe = ref 0 in
  let field x time =
    incr nfe;
    let conditional = M.forward b c weights ~state:x ~condition ~time in
    if guidance = 1. then conditional
    else (
      incr nfe;
      let unconditional =
        M.forward b c weights ~state:x
          ~condition:(Array.make (M.tokens c) (Array.make 7 0.))
          ~time
      in
      combine b unconditional
        [ (guidance, conditional); (-.guidance, unconditional) ])
  in
  let x = ref noise and dt = 1. /. float_of_int steps in
  for step = 0 to steps - 1 do
    let time = float_of_int step *. dt in
    let k1 = field !x time in
    x :=
      match method_ with
      | "euler" -> combine b !x [ (dt, k1) ]
      | "heun" ->
          let k2 = field (combine b !x [ (dt, k1) ]) (time +. dt) in
          combine b !x [ (dt *. 0.5, k1); (dt *. 0.5, k2) ]
      | _ ->
          let k2 =
            field (combine b !x [ (dt *. 0.5, k1) ]) (time +. (dt *. 0.5))
          in
          let k3 =
            field (combine b !x [ (dt *. 0.5, k2) ]) (time +. (dt *. 0.5))
          in
          let k4 = field (combine b !x [ (dt, k3) ]) (time +. dt) in
          combine b !x
            [ (dt /. 6., k1); (dt /. 3., k2); (dt /. 3., k3); (dt /. 6., k4) ]
  done;
  (!x, !nfe)

let training b c weights example seed ~dropout =
  let rng = Random.State.make [| seed; 721 |] in
  let noise = Flow_data.noise c seed in
  let time = Random.State.float rng 1. in
  let state =
    Array.mapi
      (fun i x ->
        Array.mapi
          (fun j x ->
            ((1. -. time) *. x) +. (time *. example.Flow_data.target.(i).(j)))
          x)
      noise
  in
  let target =
    Array.mapi
      (fun i x -> Array.mapi (fun j x -> x -. noise.(i).(j)) x)
      example.target
  in
  let condition =
    if dropout && Random.State.bool rng then
      Array.make (M.tokens c) (Array.make 7 0.)
    else example.condition
  in
  let output = M.forward b c weights ~state ~condition ~time in
  (Flow_data.mse output target, output)
