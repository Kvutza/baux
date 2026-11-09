module M = Flow_model

type example = {
  target : float array array;
  condition : float array array;
  tick : float;
  previous_mid : float;
}

type snapshot = { prices : float array; volumes : float array }

let mid snapshot =
  let n = Array.length snapshot.prices in
  0.5 *. (snapshot.prices.((n / 2) - 1) +. snapshot.prices.(n / 2))

let validate levels snapshot =
  if
    Array.length snapshot.prices <> levels
    || Array.length snapshot.volumes <> levels
  then invalid_arg "LOB level count mismatch";
  Array.iteri
    (fun i x ->
      if
        (not (Float.is_finite x))
        || x <= 0.
        || (i > 0 && x >= snapshot.prices.(i - 1))
      then
        invalid_arg
          "LOB prices must be positive and strictly descending outer ask to \
           outer bid")
    snapshot.prices;
  Array.iter
    (fun x ->
      if (not (Float.is_finite x)) || x < 0. then
        invalid_arg "invalid LOB volume")
    snapshot.volumes

let encode c ~tick ~previous snapshots =
  if
    Array.length snapshots <> c.M.steps
    || (not (Float.is_finite tick))
    || tick <= 0.
    || not (Float.is_finite previous)
  then invalid_arg "invalid LOB window";
  Array.iter (validate c.levels) snapshots;
  Array.init (M.tokens c) (fun i ->
      let time = i / c.levels and level = i mod c.levels in
      let s = snapshots.(time) in
      let price =
        if level = 0 then
          (mid s -. if time = 0 then previous else mid snapshots.(time - 1))
          /. tick
        else (s.prices.(level - 1) -. s.prices.(level)) /. tick
      in
      [| price /. 8.; log1p s.volumes.(level) /. 12. |])

let decode c ~tick ~previous target =
  if Array.length target <> M.tokens c then
    invalid_arg "invalid generated LOB window";
  let current = ref previous in
  Array.init c.M.steps (fun time ->
      let gaps =
        Array.init c.levels (fun l ->
            target.((time * c.levels) + l).(0) *. 8. *. tick)
      in
      current := !current +. gaps.(0);
      let above = ref (gaps.(c.levels / 2) *. 0.5) in
      for l = 1 to (c.levels / 2) - 1 do
        above := !above +. gaps.(l)
      done;
      let price = ref (!current +. !above) in
      let prices =
        Array.init c.levels (fun l ->
            if l > 0 then price := !price -. gaps.(l);
            !price)
      in
      let volumes =
        Array.init c.levels (fun l ->
            expm1 (target.((time * c.levels) + l).(1) *. 12.))
      in
      { prices; volumes })

let example c ~tick ~previous ~time_of_day past future =
  if (not (Float.is_finite time_of_day)) || time_of_day < 0. || time_of_day > 1.
  then invalid_arg "time_of_day must be in [0,1]";
  let history = encode c ~tick ~previous past in
  let target = encode c ~tick ~previous:(mid past.(c.steps - 1)) future in
  let returns = Array.init c.steps (fun t -> target.(t * c.levels).(0)) in
  let mean = Array.fold_left ( +. ) 0. returns /. float_of_int c.steps in
  let volatility =
    sqrt
      (Array.fold_left (fun s x -> s +. ((x -. mean) ** 2.)) 0. returns
      /. float_of_int c.steps)
  in
  let condition =
    Array.init (M.tokens c) (fun i ->
        let s = future.(i / c.levels) in
        let ask = ref 0. and bid = ref 0. in
        Array.iteri
          (fun l v ->
            if l < c.levels / 2 then ask := !ask +. v else bid := !bid +. v)
          s.volumes;
        let liquidity =
          log1p ((!ask +. !bid) /. float_of_int c.levels) /. 12.
        in
        let imbalance = (!bid -. !ask) /. max 1e-9 (!bid +. !ask) in
        [|
          history.(i).(0);
          history.(i).(1);
          time_of_day;
          mean;
          volatility;
          liquidity;
          imbalance;
        |])
  in
  { target; condition; tick; previous_mid = mid past.(c.steps - 1) }

let synthetic c ~seed ~regime =
  let rng = Random.State.make [| seed; regime; 4981 |] in
  let midpoint = ref 10000. in
  let drift = float_of_int ((regime mod 3) - 1) *. 0.15 in
  let volatility = 0.2 +. (float_of_int (regime mod 5) *. 0.15) in
  let snapshot () =
    midpoint :=
      !midpoint +. drift +. ((Random.State.float rng 2. -. 1.) *. volatility);
    let spread = 1. +. Random.State.float rng 1. in
    let prices =
      Array.init c.M.levels (fun l ->
          !midpoint
          +.
          if l < c.levels / 2 then
            (spread *. 0.5) +. float_of_int ((c.levels / 2) - 1 - l)
          else (-.spread *. 0.5) -. float_of_int (l - (c.levels / 2)))
    in
    let volumes =
      Array.init c.levels (fun l ->
          (500. +. Random.State.float rng 1000.)
          *. if l < c.levels / 2 = (regime mod 2 = 0) then 1.5 else 0.7)
    in
    { prices; volumes }
  in
  let past = Array.init c.steps (fun _ -> snapshot ()) in
  let future = Array.init c.steps (fun _ -> snapshot ()) in
  example c ~tick:1. ~previous:10000. ~time_of_day:0.5 past future

let gaussian rng =
  sqrt (-2. *. log (max 1e-12 (Random.State.float rng 1.)))
  *. cos (2. *. Float.pi *. Random.State.float rng 1.)

let noise c seed =
  let rng = Random.State.make [| seed; 4109 |] in
  Array.init (M.tokens c) (fun _ -> Array.init 2 (fun _ -> gaussian rng))

let mse a b =
  let total = ref 0. in
  Array.iteri
    (fun i x ->
      Array.iteri (fun j x -> total := !total +. ((x -. b.(i).(j)) ** 2.)) x)
    a;
  !total /. float_of_int (2 * Array.length a)

let distance a b =
  let x = Array.concat (Array.to_list a)
  and y = Array.concat (Array.to_list b) in
  Array.sort Float.compare x;
  Array.sort Float.compare y;
  Array.fold_left ( +. ) 0. (Array.mapi (fun i x -> abs_float (x -. y.(i))) x)
  /. float_of_int (Array.length x)

let metrics c generated target =
  let violations = ref 0 in
  Array.iteri
    (fun i x ->
      if
        (i mod c.M.levels <> 0 && x.(0) <= 0.)
        || x.(1) < 0.
        || x.(1) > log Float.max_float /. 12.
      then incr violations)
    generated;
  let channel j x = Array.map (fun row -> [| row.(j) |]) x in
  `Assoc
    [
      ("price_w1", `Float (distance (channel 0 generated) (channel 0 target)));
      ( "log_volume_w1",
        `Float (distance (channel 1 generated) (channel 1 target)) );
      ( "invalid_token_fraction",
        `Float (float_of_int !violations /. float_of_int (M.tokens c)) );
    ]

let load c filename split sample_index =
  let open Yojson.Basic.Util in
  let document = Yojson.Basic.from_file filename in
  if document |> member "version" |> to_int <> 1 then
    invalid_arg "unknown LOB dataset version";
  let entries = document |> member split |> to_list |> Array.of_list in
  if Array.length entries = 0 then invalid_arg "empty LOB dataset split";
  let e = entries.(sample_index mod Array.length entries) in
  let snapshots key =
    e |> member key |> to_list
    |> List.map (fun x ->
           let array key =
             x |> member key |> to_list |> List.map to_number |> Array.of_list
           in
           { prices = array "prices"; volumes = array "volumes" })
    |> Array.of_list
  in
  example c
    ~tick:(e |> member "tick_size" |> to_number)
    ~previous:(e |> member "previous_mid" |> to_number)
    ~time_of_day:(e |> member "time_of_day" |> to_number)
    (snapshots "past") (snapshots "future")
