(* admin/ui/bench_formulas.ml
   Pure OCaml mathematical definitions, domain rules, catalogue metadata, and
   scalar/batch evaluators for all 56 baux synthetic optimization benchmark
   functions (matching ocaml/lib/problem.ml and ocaml/lib/formulas.ml).
   Zero DOM / js_of_ocaml dependencies so it can be linked by both baux_ui.js
   and the DOM-free worker bundle oxcaml_eval.js.
*)

type family =
  | Multimodal
  | Bowl
  | Plate
  | Valley
  | Ridge
  | Other

type dim_rule =
  | Fixed of int
  | At_least of int
  | Multiple_of of int

type spec = {
  id : string;
  label : string;
  family : family;
  rule : dim_rule;
  source : string;
  domain : int -> (float * float) array;
  eval : float array -> float;
}

type problem_description = {
  spec : spec;
  dim : int;
  bounds : (float * float) array;
}

let family_slug = function
  | Multimodal -> "multimodal"
  | Bowl -> "bowl"
  | Plate -> "plate"
  | Valley -> "valley"
  | Ridge -> "ridge"
  | Other -> "other"

let family_title = function
  | Multimodal -> "Multimodal"
  | Bowl -> "Bowl"
  | Plate -> "Plate"
  | Valley -> "Valley"
  | Ridge -> "Ridge"
  | Other -> "Standard BO"

let rule_slug = function
  | Fixed _ -> "fixed"
  | At_least _ -> "at_least"
  | Multiple_of _ -> "multiple_of"

let default_dim = function
  | Fixed n | Multiple_of n -> n
  | At_least n -> Int.max 2 n

let is_scalable = function
  | Fixed _ -> false
  | At_least _ | Multiple_of _ -> true

let min_dim = function
  | Fixed n | At_least n | Multiple_of n -> n

let dim_step = function
  | Fixed _ -> 0
  | At_least _ -> 1
  | Multiple_of n -> n

let valid_dim rule d =
  d > 0
  &&
  match rule with
  | Fixed n -> d = n
  | At_least n -> d >= n
  | Multiple_of n -> d mod n = 0

(* --- Exact Boxed-Float Formulas Matching ocaml/lib/formulas.ml --- *)

module Formulas = struct
  let[@inline] sq x = x *. x

  let powi x n =
    let rec loop x n acc =
      if n = 0 then acc
      else
        let acc = if n land 1 = 1 then acc *. x else acc in
        if n = 1 then acc else loop (sq x) (n lsr 1) acc
    in
    loop x n 1.0

  let sum n f =
    let acc = ref 0.0 in
    for i = 0 to n - 1 do
      acc := !acc +. f i
    done;
    !acc

  let product n f =
    let acc = ref 1.0 in
    for i = 0 to n - 1 do
      acc := !acc *. f i
    done;
    !acc

  let sphere (x : float array) =
    sum (Array.length x) (fun i -> sq x.(i))

  let rosenbrock (x : float array) =
    sum (Array.length x - 1) (fun i ->
        (100.0 *. sq (x.(i + 1) -. sq x.(i))) +. sq (1.0 -. x.(i)))

  let ackley (x : float array) =
    let n = Array.length x in
    let d = float_of_int n in
    (-20.0 *. Float.exp (-0.2 *. Float.sqrt (sphere x /. d)))
    -. Float.exp
         (sum n (fun i -> Float.cos (2.0 *. Float.pi *. x.(i))) /. d)
    +. 20.0
    +. Float.exp 1.0

  let rastrigin (x : float array) =
    let n = Array.length x in
    (10.0 *. float_of_int n)
    +. sum n (fun i ->
           sq x.(i) -. (10.0 *. Float.cos (2.0 *. Float.pi *. x.(i))))

  let griewank (x : float array) =
    let n = Array.length x in
    1.0
    +. (sphere x /. 4000.0)
    -. product n (fun i ->
           Float.cos (x.(i) /. Float.sqrt (float_of_int (i + 1))))

  let schwefel (x : float array) =
    let n = Array.length x in
    (418.9829 *. float_of_int n)
    -. sum n (fun i -> x.(i) *. Float.sin (Float.sqrt (Float.abs x.(i))))

  let levy (x : float array) =
    let n = Array.length x in
    let w i = 1.0 +. ((x.(i) -. 1.0) /. 4.0) in
    let last = w (n - 1) in
    sq (Float.sin (Float.pi *. w 0))
    +. sum (n - 1) (fun i ->
           let wi = w i in
           sq (wi -. 1.0)
           *. (1.0 +. (10.0 *. sq (Float.sin ((Float.pi *. wi) +. 1.0)))))
    +. (sq (last -. 1.0)
       *. (1.0 +. sq (Float.sin (2.0 *. Float.pi *. last))))

  let sum_squares (x : float array) =
    sum (Array.length x) (fun i -> float_of_int (i + 1) *. sq x.(i))

  let rotated_hyperellipsoid (x : float array) =
    let n = Array.length x in
    sum n (fun i -> float_of_int (n - i) *. sq x.(i))

  let sum_powers (x : float array) =
    sum (Array.length x) (fun i -> powi (Float.abs x.(i)) (i + 2))

  let trid (x : float array) =
    let n = Array.length x in
    sum n (fun i -> sq (x.(i) -. 1.0))
    -. sum (n - 1) (fun i -> x.(i) *. x.(i + 1))

  let zakharov (x : float array) =
    let n = Array.length x in
    let a = sum n (fun i -> 0.5 *. float_of_int (i + 1) *. x.(i)) in
    sphere x +. sq a +. sq (sq a)

  let dixon_price (x : float array) =
    let n = Array.length x in
    sq (x.(0) -. 1.0)
    +. sum (n - 1) (fun i ->
           float_of_int (i + 2) *. sq ((2.0 *. sq x.(i + 1)) -. x.(i)))

  let michalewicz (x : float array) =
    let n = Array.length x in
    -.sum n (fun i ->
          Float.sin x.(i)
          *. powi
               (Float.sin (float_of_int (i + 1) *. sq x.(i) /. Float.pi))
               20)

  let styblinski_tang (x : float array) =
    0.5
    *. sum (Array.length x) (fun i ->
           let v = x.(i) in
           sq (sq v) -. (16.0 *. sq v) +. (5.0 *. v))

  let perm_zero (x : float array) =
    let n = Array.length x in
    sum n (fun k ->
        let power = k + 1 in
        sq
          (sum n (fun i ->
               let j = float_of_int (i + 1) in
               (j +. 10.0) *. (powi x.(i) power -. powi (1.0 /. j) power))))

  let perm (x : float array) =
    let n = Array.length x in
    sum n (fun k ->
        let power = k + 1 in
        sq
          (sum n (fun i ->
               let j = float_of_int (i + 1) in
               (powi j power +. 0.5) *. (powi (x.(i) /. j) power -. 1.0))))

  let powell (x : float array) =
    sum (Array.length x / 4) (fun block ->
        let i = 4 * block in
        let a = x.(i) and b = x.(i + 1) in
        let c = x.(i + 2) and d = x.(i + 3) in
        sq (a +. (10.0 *. b))
        +. (5.0 *. sq (c -. d))
        +. sq (sq (b -. (2.0 *. c)))
        +. (10.0 *. sq (sq (a -. d))))

  let bukin6 (x : float array) =
    (100.0 *. Float.sqrt (Float.abs (x.(1) -. (0.01 *. sq x.(0)))))
    +. (0.01 *. Float.abs (x.(0) +. 10.0))

  let cross_in_tray (x : float array) =
    let a = x.(0) and b = x.(1) in
    -0.0001
    *. Float.pow
         (Float.abs
            (Float.sin a *. Float.sin b
            *. Float.exp
                 (Float.abs (100.0 -. (Float.sqrt (sq a +. sq b) /. Float.pi)))
            )
         +. 1.0)
         0.1

  let drop_wave (x : float array) =
    let r2 = sphere x in
    -.(1.0 +. Float.cos (12.0 *. Float.sqrt r2)) /. (2.0 +. (0.5 *. r2))

  let eggholder (x : float array) =
    let a = x.(0) and b = x.(1) +. 47.0 in
    (-.b *. Float.sin (Float.sqrt (Float.abs ((a /. 2.0) +. b))))
    -. (a *. Float.sin (Float.sqrt (Float.abs (a -. b))))

  let gramacy_lee (x : float array) =
    let a = x.(0) in
    (Float.sin (10.0 *. Float.pi *. a) /. (2.0 *. a)) +. sq (sq (a -. 1.0))

  let holder_table (x : float array) =
    let a = x.(0) and b = x.(1) in
    -.Float.abs
        (Float.sin a *. Float.cos b
        *. Float.exp
             (Float.abs (1.0 -. (Float.sqrt (sq a +. sq b) /. Float.pi))))

  let langermann_centers =
    [| [| 3.; 5. |]; [| 5.; 2. |]; [| 2.; 1. |]; [| 1.; 4. |]; [| 7.; 9. |] |]

  let langermann_weights = [| 1.; 2.; 5.; 2.; 3. |]

  let langermann (x : float array) =
    sum 5 (fun i ->
        let c = langermann_centers.(i) in
        let r2 = sq (x.(0) -. c.(0)) +. sq (x.(1) -. c.(1)) in
        langermann_weights.(i)
        *. Float.exp (-.r2 /. Float.pi)
        *. Float.cos (Float.pi *. r2))

  let levy13 (x : float array) =
    let a = x.(0) and b = x.(1) in
    sq (Float.sin (3.0 *. Float.pi *. a))
    +. (sq (a -. 1.0) *. (1.0 +. sq (Float.sin (3.0 *. Float.pi *. b))))
    +. (sq (b -. 1.0) *. (1.0 +. sq (Float.sin (2.0 *. Float.pi *. b))))

  let schaffer2 (x : float array) =
    let a = sq x.(0) and b = sq x.(1) in
    0.5
    +. ((sq (Float.sin (a -. b)) -. 0.5)
       /. sq (1.0 +. (0.001 *. (a +. b))))

  let schaffer4 (x : float array) =
    let a = sq x.(0) and b = sq x.(1) in
    0.5
    +. ((sq (Float.cos (Float.sin (Float.abs (a -. b)))) -. 0.5)
       /. sq (1.0 +. (0.001 *. (a +. b))))

  let shubert (x : float array) =
    product 2 (fun j ->
        sum 5 (fun i ->
            let k = float_of_int (i + 1) in
            k *. Float.cos (((k +. 1.0) *. x.(j)) +. k)))

  let bohachevsky kind (x : float array) =
    let a = x.(0) and b = x.(1) in
    let u = 3.0 *. Float.pi *. a and v = 4.0 *. Float.pi *. b in
    sq a +. (2.0 *. sq b)
    +.
    match kind with
    | 1 -> 0.7 -. (0.3 *. Float.cos u) -. (0.4 *. Float.cos v)
    | 2 -> 0.3 -. (0.3 *. Float.cos u *. Float.cos v)
    | _ -> 0.3 -. (0.3 *. Float.cos (u +. v))

  let booth (x : float array) =
    let a = x.(0) and b = x.(1) in
    sq (a +. (2.0 *. b) -. 7.0) +. sq ((2.0 *. a) +. b -. 5.0)

  let matyas (x : float array) =
    (0.26 *. sphere x) -. (0.48 *. x.(0) *. x.(1))

  let mccormick (x : float array) =
    let a = x.(0) and b = x.(1) in
    Float.sin (a +. b) +. sq (a -. b) -. (1.5 *. a) +. (2.5 *. b) +. 1.0

  let power_sum_targets = [| 8.; 18.; 44.; 114. |]

  let power_sum (x : float array) =
    sum 4 (fun k ->
        sq (sum 4 (fun i -> powi x.(i) (k + 1)) -. power_sum_targets.(k)))

  let three_hump_camel (x : float array) =
    let a = x.(0) and b = x.(1) in
    (2.0 *. sq a) -. (1.05 *. sq (sq a)) +. (powi a 6 /. 6.0)
    +. (a *. b) +. sq b

  let six_hump_camel (x : float array) =
    let a = x.(0) and b = x.(1) in
    ((4.0 -. (2.1 *. sq a) +. (sq (sq a) /. 3.0)) *. sq a)
    +. (a *. b)
    +. ((-4.0 +. (4.0 *. sq b)) *. sq b)

  let de_jong5 (x : float array) =
    1.0
    /. (0.002
       +. sum 25 (fun i ->
              let a = float_of_int (16 * ((i mod 5) - 2)) in
              let b = float_of_int (16 * ((i / 5) - 2)) in
              1.0
              /. (float_of_int (i + 1)
                 +. powi (x.(0) -. a) 6
                 +. powi (x.(1) -. b) 6)))

  let easom (x : float array) =
    let a = x.(0) and b = x.(1) in
    -.(Float.cos a *. Float.cos b)
    *. Float.exp (-.(sq (a -. Float.pi) +. sq (b -. Float.pi)))

  let beale (x : float array) =
    let a = x.(0) and b = x.(1) in
    sq (1.5 -. a +. (a *. b))
    +. sq (2.25 -. a +. (a *. sq b))
    +. sq (2.625 -. a +. (a *. powi b 3))

  let branin_at a b =
    sq
      (b
      -. (5.1 *. sq a /. (4.0 *. sq Float.pi))
      +. (5.0 *. a /. Float.pi)
      -. 6.0)
    +. (10.0 *. (1.0 -. (1.0 /. (8.0 *. Float.pi))) *. Float.cos a)
    +. 10.0

  let branin (x : float array) = branin_at x.(0) x.(1)

  let colville (x : float array) =
    let a = x.(0) and b = x.(1) and c = x.(2) and d = x.(3) in
    (100.0 *. sq (sq a -. b))
    +. sq (a -. 1.0)
    +. sq (c -. 1.0)
    +. (90.0 *. sq (sq c -. d))
    +. (10.1 *. (sq (b -. 1.0) +. sq (d -. 1.0)))
    +. (19.8 *. (b -. 1.0) *. (d -. 1.0))

  let forrester (x : float array) =
    let a = x.(0) in
    sq ((6.0 *. a) -. 2.0) *. Float.sin ((12.0 *. a) -. 4.0)

  let goldstein_price_at a b =
    (1.0
    +. sq (a +. b +. 1.0)
       *. (19.0 -. (14.0 *. a) +. (3.0 *. sq a) -. (14.0 *. b)
          +. (6.0 *. a *. b) +. (3.0 *. sq b)))
    *. (30.0
       +. sq ((2.0 *. a) -. (3.0 *. b))
          *. (18.0 -. (32.0 *. a) +. (12.0 *. sq a) +. (48.0 *. b)
             -. (36.0 *. a *. b) +. (27.0 *. sq b)))

  let goldstein_price (x : float array) = goldstein_price_at x.(0) x.(1)

  let hart_alpha = [| 1.; 1.2; 3.; 3.2 |]

  let hart3_a =
    [|
      [| 3.; 10.; 30. |];
      [| 0.1; 10.; 35. |];
      [| 3.; 10.; 30. |];
      [| 0.1; 10.; 35. |];
    |]

  let hart3_p =
    [|
      [| 3689.; 1170.; 2673. |];
      [| 4699.; 4387.; 7470. |];
      [| 1091.; 8732.; 5547. |];
      [| 381.; 5743.; 8828. |];
    |]

  let hart6_a =
    [|
      [| 10.; 3.; 17.; 3.5; 1.7; 8. |];
      [| 0.05; 10.; 17.; 0.1; 8.; 14. |];
      [| 3.; 3.5; 1.7; 10.; 17.; 8. |];
      [| 17.; 8.; 0.05; 10.; 0.1; 14. |];
    |]

  let hart6_p =
    [|
      [| 1312.; 1696.; 5569.; 124.; 8283.; 5886. |];
      [| 2329.; 4135.; 8307.; 3736.; 1004.; 9991. |];
      [| 2348.; 1451.; 3522.; 2883.; 3047.; 6650. |];
      [| 4047.; 8828.; 8732.; 5743.; 1091.; 381. |];
    |]

  let hartmann (x : float array) =
    let d = Array.length x in
    let a, p = if d = 3 then (hart3_a, hart3_p) else (hart6_a, hart6_p) in
    -.sum 4 (fun i ->
          hart_alpha.(i)
          *. Float.exp
               (-.sum d (fun j ->
                      a.(i).(j) *. sq (x.(j) -. (0.0001 *. p.(i).(j))))))

  let hartmann4 (x : float array) = (1.1 +. hartmann x) /. 0.839

  let shekel_c =
    [|
      [| 4.; 4.; 4.; 4. |];
      [| 1.; 1.; 1.; 1. |];
      [| 8.; 8.; 8.; 8. |];
      [| 6.; 6.; 6.; 6. |];
      [| 3.; 7.; 3.; 7. |];
      [| 2.; 9.; 2.; 9. |];
      [| 5.; 3.; 5.; 3. |];
      [| 8.; 1.; 8.; 1. |];
      [| 6.; 2.; 6.; 2. |];
      [| 7.; 3.6; 7.; 3.6 |];
    |]

  let shekel_b = [| 0.1; 0.2; 0.2; 0.4; 0.4; 0.6; 0.3; 0.7; 0.5; 0.5 |]

  let shekel (x : float array) =
    -.sum 10 (fun i ->
          1.0
          /. (shekel_b.(i)
             +. sum 4 (fun j -> sq (x.(j) -. shekel_c.(i).(j)))))

  let sphere6_rescaled (x : float array) =
    (sum 6 (fun i -> powi 2.0 (i + 1) *. sq x.(i)) -. 1745.0) /. 899.0

  let rosenbrock4_rescaled (x : float array) =
    let at i = (15.0 *. x.(i)) -. 5.0 in
    (sum 3 (fun i ->
         (100.0 *. sq (at (i + 1) -. sq (at i))) +. sq (1.0 -. at i))
    -. 382700.0)
    /. 375500.0

  let branin_rescaled (x : float array) =
    (branin_at ((15.0 *. x.(0)) -. 5.0) (15.0 *. x.(1)) -. 54.81) /. 51.95

  let branin_modified (x : float array) = branin x +. (5.0 *. x.(0))

  let forrester_low_fidelity (x : float array) =
    (0.5 *. forrester x) +. (10.0 *. (x.(0) -. 0.5)) +. 5.0

  let goldstein_price_rescaled (x : float array) =
    (Float.log
       (goldstein_price_at ((4.0 *. x.(0)) -. 2.0) ((4.0 *. x.(1)) -. 2.0))
    -. 8.693)
    /. 2.427

  let hartmann6_rescaled (x : float array) = (hartmann x -. 2.58) /. 1.94
end

let uniform low high n = Array.make n (low, high)
let fixed arr _n = Array.copy arr

let mk id label family rule page domain eval =
  {
    id;
    label;
    family;
    rule;
    source = "https://www.sfu.ca/~ssurjano/" ^ page ^ ".html";
    domain;
    eval;
  }

let catalogue : spec array =
  let any = At_least 1 and two = Fixed 2 in
  [|
    mk "ackley" "Ackley" Multimodal any "ackley" (uniform (-32.768) 32.768)
      Formulas.ackley;
    mk "bukin6" "Bukin N.6" Multimodal two "bukin6"
      (fixed [| (-15., -5.); (-3., 3.) |])
      Formulas.bukin6;
    mk "cross-in-tray" "Cross-in-Tray" Multimodal two "crossit"
      (uniform (-10.) 10.) Formulas.cross_in_tray;
    mk "drop-wave" "Drop-Wave" Multimodal two "drop" (uniform (-5.12) 5.12)
      Formulas.drop_wave;
    mk "eggholder" "Eggholder" Multimodal two "egg" (uniform (-512.) 512.)
      Formulas.eggholder;
    mk "gramacy-lee" "Gramacy & Lee (1D)" Multimodal (Fixed 1) "grlee12"
      (uniform 0.5 2.5) Formulas.gramacy_lee;
    mk "griewank" "Griewank" Multimodal any "griewank" (uniform (-600.) 600.)
      Formulas.griewank;
    mk "holder-table" "Holder Table" Multimodal two "holder"
      (uniform (-10.) 10.) Formulas.holder_table;
    mk "langermann" "Langermann" Multimodal two "langer" (uniform 0. 10.)
      Formulas.langermann;
    mk "levy" "Levy" Multimodal any "levy" (uniform (-10.) 10.) Formulas.levy;
    mk "levy13" "Levy N.13" Multimodal two "levy13" (uniform (-10.) 10.)
      Formulas.levy13;
    mk "rastrigin" "Rastrigin" Multimodal any "rastr" (uniform (-5.12) 5.12)
      Formulas.rastrigin;
    mk "schaffer2" "Schaffer N.2" Multimodal two "schaffer2"
      (uniform (-100.) 100.) Formulas.schaffer2;
    mk "schaffer4" "Schaffer N.4" Multimodal two "schaffer4"
      (uniform (-100.) 100.) Formulas.schaffer4;
    mk "schwefel" "Schwefel" Multimodal any "schwef" (uniform (-500.) 500.)
      Formulas.schwefel;
    mk "shubert" "Shubert" Multimodal two "shubert" (uniform (-10.) 10.)
      Formulas.shubert;
    mk "bohachevsky1" "Bohachevsky N.1" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> Formulas.bohachevsky 1 x);
    mk "bohachevsky2" "Bohachevsky N.2" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> Formulas.bohachevsky 2 x);
    mk "bohachevsky3" "Bohachevsky N.3" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> Formulas.bohachevsky 3 x);
    mk "perm-zero" "Perm 0, d, β" Bowl any "perm0db"
      (fun n -> uniform (-.float_of_int n) (float_of_int n) n)
      Formulas.perm_zero;
    mk "rotated-hyperellipsoid" "Rotated Hyper-Ellipsoid" Bowl any "rothyp"
      (uniform (-65.536) 65.536) Formulas.rotated_hyperellipsoid;
    mk "sphere" "Sphere" Bowl any "spheref" (uniform (-5.12) 5.12)
      Formulas.sphere;
    mk "sum-powers" "Sum of Different Powers" Bowl any "sumpow"
      (uniform (-1.) 1.) Formulas.sum_powers;
    mk "sum-squares" "Sum Squares" Bowl any "sumsqu" (uniform (-10.) 10.)
      Formulas.sum_squares;
    mk "trid" "Trid" Bowl (At_least 2) "trid"
      (fun n ->
        let d = float_of_int n in
        uniform (-.d *. d) (d *. d) n)
      Formulas.trid;
    mk "booth" "Booth" Plate two "booth" (uniform (-10.) 10.) Formulas.booth;
    mk "matyas" "Matyas" Plate two "matya" (uniform (-10.) 10.) Formulas.matyas;
    mk "mccormick" "McCormick" Plate two "mccorm"
      (fixed [| (-1.5, 4.); (-3., 4.) |])
      Formulas.mccormick;
    mk "power-sum" "Power Sum (4D)" Plate (Fixed 4) "powersum" (uniform 0. 4.)
      Formulas.power_sum;
    mk "zakharov" "Zakharov" Plate any "zakharov" (uniform (-5.) 10.)
      Formulas.zakharov;
    mk "three-hump-camel" "Three-Hump Camel" Valley two "camel3"
      (uniform (-5.) 5.) Formulas.three_hump_camel;
    mk "six-hump-camel" "Six-Hump Camel" Valley two "camel6"
      (fixed [| (-3., 3.); (-2., 2.) |])
      Formulas.six_hump_camel;
    mk "dixon-price" "Dixon-Price" Valley (At_least 2) "dixonpr"
      (uniform (-10.) 10.) Formulas.dixon_price;
    mk "rosenbrock" "Rosenbrock" Valley (At_least 2) "rosen" (uniform (-5.) 10.)
      Formulas.rosenbrock;
    mk "de-jong5" "De Jong N.5" Ridge two "dejong5" (uniform (-65.536) 65.536)
      Formulas.de_jong5;
    mk "easom" "Easom" Ridge two "easom" (uniform (-100.) 100.) Formulas.easom;
    mk "michalewicz" "Michalewicz" Ridge any "michal" (uniform 0. Float.pi)
      Formulas.michalewicz;
    mk "beale" "Beale" Other two "beale" (uniform (-4.5) 4.5) Formulas.beale;
    mk "branin" "Branin" Other two "branin"
      (fixed [| (-5., 10.); (0., 15.) |])
      Formulas.branin;
    mk "colville" "Colville (4D)" Other (Fixed 4) "colville"
      (uniform (-10.) 10.) Formulas.colville;
    mk "forrester" "Forrester (1D)" Other (Fixed 1) "forretal08"
      (uniform 0. 1.) Formulas.forrester;
    mk "goldstein-price" "Goldstein-Price" Other two "goldpr" (uniform (-2.) 2.)
      Formulas.goldstein_price;
    mk "hartmann3" "Hartmann (3D)" Other (Fixed 3) "hart3" (uniform 0. 1.)
      Formulas.hartmann;
    mk "hartmann4" "Hartmann (4D)" Other (Fixed 4) "hart4" (uniform 0. 1.)
      Formulas.hartmann4;
    mk "hartmann6" "Hartmann (6D)" Other (Fixed 6) "hart6" (uniform 0. 1.)
      Formulas.hartmann;
    mk "perm" "Perm d, β" Other any "permdb"
      (fun n -> uniform (-.float_of_int n) (float_of_int n) n)
      Formulas.perm;
    mk "powell" "Powell" Other (Multiple_of 4) "powell" (uniform (-4.) 5.)
      Formulas.powell;
    mk "shekel" "Shekel (4D)" Other (Fixed 4) "shekel" (uniform 0. 10.)
      Formulas.shekel;
    mk "styblinski-tang" "Styblinski-Tang" Other any "stybtang"
      (uniform (-5.) 5.) Formulas.styblinski_tang;
    mk "sphere6-rescaled" "Sphere-6 (Rescaled)" Bowl (Fixed 6) "spheref"
      (uniform 0. 1.) Formulas.sphere6_rescaled;
    mk "rosenbrock4-rescaled" "Rosenbrock-4 (Rescaled)" Valley (Fixed 4)
      "rosen" (uniform 0. 1.) Formulas.rosenbrock4_rescaled;
    mk "branin-rescaled" "Branin (Rescaled)" Other two "branin" (uniform 0. 1.)
      Formulas.branin_rescaled;
    mk "branin-modified" "Branin (Modified)" Other two "branin"
      (fixed [| (-5., 10.); (0., 15.) |])
      Formulas.branin_modified;
    mk "forrester-low-fidelity" "Forrester Low-Fi (1D)" Other (Fixed 1)
      "forretal08" (uniform 0. 1.) Formulas.forrester_low_fidelity;
    mk "goldstein-price-rescaled" "Goldstein-Price (Rescaled)" Other two
      "goldpr" (uniform 0. 1.) Formulas.goldstein_price_rescaled;
    mk "hartmann6-rescaled" "Hartmann-6 (Rescaled)" Other (Fixed 6) "hart6"
      (uniform 0. 1.) Formulas.hartmann6_rescaled;
  |]

let spec_index : (string, spec) Hashtbl.t =
  let tbl = Hashtbl.create (Array.length catalogue * 2) in
  Array.iter (fun (sp : spec) -> Hashtbl.replace tbl sp.id sp) catalogue;
  tbl

let find_spec id = Hashtbl.find_opt spec_index id

let require_spec id =
  match find_spec id with
  | Some sp -> sp
  | None -> invalid_arg ("Unknown OxCaml benchmark: " ^ id)

let resolve_dim (sp : spec) (dim : int) =
  let d = if dim <= 0 then default_dim sp.rule else dim in
  if not (valid_dim sp.rule d) then
    invalid_arg (Printf.sprintf "Invalid dimension %d for %s" d sp.id);
  d

let describe ?(dim = 0) id =
  let sp = require_spec id in
  let target_dim = resolve_dim sp dim in
  { spec = sp; dim = target_dim; bounds = sp.domain target_dim }

let validate_point (sp : spec) (dim : int) (x : float array) =
  if Array.length x <> dim then
    invalid_arg
      (Printf.sprintf "Point dimension mismatch for %s: expected %d, got %d"
         sp.id dim (Array.length x));
  for j = 0 to dim - 1 do
    if not (Float.is_finite x.(j)) then
      invalid_arg
        (Printf.sprintf "Non-finite coordinate at index %d for %s" j sp.id)
  done

let eval_one ?dim id (x : float array) =
  let sp = require_spec id in
  let target_dim =
    match dim with
    | None ->
        let d = Array.length x in
        if not (valid_dim sp.rule d) then
          invalid_arg (Printf.sprintf "Invalid dimension %d for %s" d sp.id);
        d
    | Some d ->
        if d <= 0 || not (valid_dim sp.rule d) then
          invalid_arg (Printf.sprintf "Invalid dimension %d for %s" d sp.id);
        d
  in
  validate_point sp target_dim x;
  sp.eval x

let eval_batch_indexed id ~dim ~count ~get ~set =
  let sp = require_spec id in
  if dim <= 0 || not (valid_dim sp.rule dim) then
    invalid_arg (Printf.sprintf "Invalid dimension %d for %s" dim sp.id);
  if count < 0 then
    invalid_arg (Printf.sprintf "Invalid batch count %d for %s" count sp.id);
  let buf = Array.make dim 0.0 in
  for i = 0 to count - 1 do
    for j = 0 to dim - 1 do
      let v = get i j in
      if not (Float.is_finite v) then
        invalid_arg
          (Printf.sprintf "Non-finite coordinate at (%d, %d) for %s" i j sp.id);
      buf.(j) <- v
    done;
    set i (sp.eval buf)
  done

let eval_batch ?dim id (points : float array array) =
  let sp = require_spec id in
  let count = Array.length points in
  let target_dim =
    match dim with
    | Some d -> d
    | None -> if count > 0 then Array.length points.(0) else default_dim sp.rule
  in
  if target_dim <= 0 || not (valid_dim sp.rule target_dim) then
    invalid_arg (Printf.sprintf "Invalid dimension %d for %s" target_dim sp.id);
  let out = Array.make count 0.0 in
  for i = 0 to count - 1 do
    validate_point sp target_dim points.(i);
    out.(i) <- sp.eval points.(i)
  done;
  out

let formula_latex = function
  | "ackley" ->
      {tex|f(x) = -20\exp\!\left(-0.2\sqrt{\tfrac{1}{d}\sum_{i=1}^d x_i^2}\right) - \exp\!\left(\tfrac{1}{d}\sum_{i=1}^d \cos(2\pi x_i)\right) + 20 + e|tex}
  | "bukin6" ->
      {tex|f(x) = 100\sqrt{|x_2 - 0.01 x_1^2|} + 0.01|x_1 + 10||tex}
  | "cross-in-tray" ->
      {tex|f(x) = -10^{-4}\!\left(\left|\sin(x_1)\sin(x_2)\exp\!\left(\left|100 - \tfrac{\sqrt{x_1^2 + x_2^2}}{\pi}\right|\right)\right| + 1\right)^{0.1}|tex}
  | "drop-wave" ->
      {tex|f(x) = -\frac{1 + \cos\!\left(12\sqrt{x_1^2 + x_2^2}\right)}{2 + 0.5(x_1^2 + x_2^2)}|tex}
  | "eggholder" ->
      {tex|f(x) = -(x_2 + 47)\sin\!\left(\sqrt{\left|\tfrac{x_1}{2} + x_2 + 47\right|}\right) - x_1\sin\!\left(\sqrt{|x_1 - (x_2 + 47)|}\right)|tex}
  | "gramacy-lee" ->
      {tex|f(x) = \frac{\sin(10\pi x)}{2x} + (x - 1)^4|tex}
  | "griewank" ->
      {tex|f(x) = 1 + \frac{1}{4000}\sum_{i=1}^d x_i^2 - \prod_{i=1}^d \cos\!\left(\frac{x_i}{\sqrt{i}}\right)|tex}
  | "holder-table" ->
      {tex|f(x) = -\left|\sin(x_1)\cos(x_2)\exp\!\left(\left|1 - \tfrac{\sqrt{x_1^2 + x_2^2}}{\pi}\right|\right)\right||tex}
  | "langermann" ->
      {tex|f(x) = \sum_{i=1}^5 c_i \exp\!\left(-\tfrac{\|x - A_i\|^2}{\pi}\right)\cos\!\left(\pi\|x - A_i\|^2\right)|tex}
  | "levy" ->
      {tex|f(x) = \sin^2(\pi w_1) + \sum_{i=1}^{d-1}(w_i - 1)^2\left[1 + 10\sin^2(\pi w_i + 1)\right] + (w_d - 1)^2\left[1 + \sin^2(2\pi w_d)\right],\;\; w_i = 1 + \tfrac{x_i - 1}{4}|tex}
  | "levy13" ->
      {tex|f(x) = \sin^2(3\pi x_1) + (x_1 - 1)^2\left[1 + \sin^2(3\pi x_2)\right] + (x_2 - 1)^2\left[1 + \sin^2(2\pi x_2)\right]|tex}
  | "rastrigin" ->
      {tex|f(x) = 10d + \sum_{i=1}^d \left[x_i^2 - 10\cos(2\pi x_i)\right]|tex}
  | "schaffer2" ->
      {tex|f(x) = 0.5 + \frac{\sin^2(x_1^2 - x_2^2) - 0.5}{\left[1 + 0.001(x_1^2 + x_2^2)\right]^2}|tex}
  | "schaffer4" ->
      {tex|f(x) = 0.5 + \frac{\cos^2(\sin(|x_1^2 - x_2^2|)) - 0.5}{\left[1 + 0.001(x_1^2 + x_2^2)\right]^2}|tex}
  | "schwefel" ->
      {tex|f(x) = 418.9829\,d - \sum_{i=1}^d x_i\sin\!\left(\sqrt{|x_i|}\right)|tex}
  | "shubert" ->
      {tex|f(x) = \left(\sum_{k=1}^5 k\cos((k+1)x_1 + k)\right)\left(\sum_{k=1}^5 k\cos((k+1)x_2 + k)\right)|tex}
  | "bohachevsky1" ->
      {tex|f(x) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1) - 0.4\cos(4\pi x_2) + 0.7|tex}
  | "bohachevsky2" ->
      {tex|f(x) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1)\cos(4\pi x_2) + 0.3|tex}
  | "bohachevsky3" ->
      {tex|f(x) = x_1^2 + 2x_2^2 - 0.3\cos(3\pi x_1 + 4\pi x_2) + 0.3|tex}
  | "perm-zero" ->
      {tex|f(x) = \sum_{k=1}^d \left(\sum_{j=1}^d (j + 10)\!\left(x_j^k - \tfrac{1}{j^k}\right)\right)^2|tex}
  | "rotated-hyperellipsoid" ->
      {tex|f(x) = \sum_{i=1}^d (d - i + 1)\,x_i^2|tex}
  | "sphere" ->
      {tex|f(x) = \sum_{i=1}^d x_i^2|tex}
  | "sum-powers" ->
      {tex|f(x) = \sum_{i=1}^d |x_i|^{i+1}|tex}
  | "sum-squares" ->
      {tex|f(x) = \sum_{i=1}^d i\,x_i^2|tex}
  | "trid" ->
      {tex|f(x) = \sum_{i=1}^d (x_i - 1)^2 - \sum_{i=2}^d x_i x_{i-1}|tex}
  | "booth" ->
      {tex|f(x) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2|tex}
  | "matyas" ->
      {tex|f(x) = 0.26(x_1^2 + x_2^2) - 0.48 x_1 x_2|tex}
  | "mccormick" ->
      {tex|f(x) = \sin(x_1 + x_2) + (x_1 - x_2)^2 - 1.5 x_1 + 2.5 x_2 + 1|tex}
  | "power-sum" ->
      {tex|f(x) = \sum_{k=1}^4 \left(\sum_{i=1}^4 x_i^k - b_k\right)^2,\;\; b=(8,18,44,114)|tex}
  | "zakharov" ->
      {tex|f(x) = \sum_{i=1}^d x_i^2 + \left(\sum_{i=1}^d 0.5 i\, x_i\right)^2 + \left(\sum_{i=1}^d 0.5 i\, x_i\right)^4|tex}
  | "three-hump-camel" ->
      {tex|f(x) = 2x_1^2 - 1.05 x_1^4 + \tfrac{x_1^6}{6} + x_1 x_2 + x_2^2|tex}
  | "six-hump-camel" ->
      {tex|f(x) = \left(4 - 2.1x_1^2 + \tfrac{x_1^4}{3}\right)x_1^2 + x_1 x_2 + (-4 + 4x_2^2)x_2^2|tex}
  | "dixon-price" ->
      {tex|f(x) = (x_1 - 1)^2 + \sum_{i=2}^d i\,(2x_i^2 - x_{i-1})^2|tex}
  | "rosenbrock" ->
      {tex|f(x) = \sum_{i=1}^{d-1}\left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\right]|tex}
  | "de-jong5" ->
      {tex|f(x) = \left(0.002 + \sum_{i=1}^{25}\frac{1}{i + (x_1 - a_{1i})^6 + (x_2 - a_{2i})^6}\right)^{-1}|tex}
  | "easom" ->
      {tex|f(x) = -\cos(x_1)\cos(x_2)\exp\!\left(-(x_1 - \pi)^2 - (x_2 - \pi)^2\right)|tex}
  | "michalewicz" ->
      {tex|f(x) = -\sum_{i=1}^d \sin(x_i)\sin^{20}\!\left(\frac{i\,x_i^2}{\pi}\right)|tex}
  | "beale" ->
      {tex|f(x) = (1.5 - x_1 + x_1 x_2)^2 + (2.25 - x_1 + x_1 x_2^2)^2 + (2.625 - x_1 + x_1 x_2^3)^2|tex}
  | "branin" ->
      {tex|f(x) = \left(x_2 - \tfrac{5.1}{4\pi^2}x_1^2 + \tfrac{5}{\pi}x_1 - 6\right)^2 + 10\left(1 - \tfrac{1}{8\pi}\right)\cos(x_1) + 10|tex}
  | "colville" ->
      {tex|f(x) = 100(x_1^2 - x_2)^2 + (x_1 - 1)^2 + (x_3 - 1)^2 + 90(x_3^2 - x_4)^2 + 10.1\left[(x_2 - 1)^2 + (x_4 - 1)^2\right] + 19.8(x_2 - 1)(x_4 - 1)|tex}
  | "forrester" ->
      {tex|f(x) = (6x - 2)^2\sin(12x - 4)|tex}
  | "goldstein-price" ->
      {tex|f(x) = \left[1 + (x_1+x_2+1)^2(19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2)\right]\left[30 + (2x_1 - 3x_2)^2(18 - 32x_1 + 12x_1^2 + 48x_2 - 36x_1x_2 + 27x_2^2)\right]|tex}
  | "hartmann3" ->
      {tex|f(x) = -\sum_{i=1}^4 \alpha_i \exp\!\left(-\sum_{j=1}^3 A_{ij}(x_j - P_{ij})^2\right),\;\; \alpha=(1,1.2,3,3.2)|tex}
  | "hartmann4" ->
      {tex|f(x) = \frac{1.1 - \sum_{i=1}^4 \alpha_i \exp\!\left(-\sum_{j=1}^4 A_{ij}(x_j - P_{ij})^2\right)}{0.839}|tex}
  | "hartmann6" ->
      {tex|f(x) = -\sum_{i=1}^4 \alpha_i \exp\!\left(-\sum_{j=1}^6 A_{ij}(x_j - P_{ij})^2\right),\;\; \alpha=(1,1.2,3,3.2)|tex}
  | "perm" ->
      {tex|f(x) = \sum_{k=1}^d \left(\sum_{j=1}^d (j^k + 0.5)\left[\left(\tfrac{x_j}{j}\right)^k - 1\right]\right)^2|tex}
  | "powell" ->
      {tex|f(x) = \sum_{i=1}^{d/4}\left[(x_{4i-3} + 10x_{4i-2})^2 + 5(x_{4i-1} - x_{4i})^2 + (x_{4i-2} - 2x_{4i-1})^4 + 10(x_{4i-3} - x_{4i})^4\right]|tex}
  | "shekel" ->
      {tex|f(x) = -\sum_{i=1}^{10}\left(\beta_i + \sum_{j=1}^4 (x_j - C_{ij})^2\right)^{-1}|tex}
  | "styblinski-tang" ->
      {tex|f(x) = \frac{1}{2}\sum_{i=1}^d \left(x_i^4 - 16x_i^2 + 5x_i\right)|tex}
  | "sphere6-rescaled" ->
      {tex|f(x) = \frac{\sum_{i=1}^6 2^i x_i^2 - 1745}{899}|tex}
  | "rosenbrock4-rescaled" ->
      {tex|f(x) = \frac{\sum_{i=1}^3\left[100(u_{i+1} - u_i^2)^2 + (1 - u_i)^2\right] - 382700}{375500},\;\; u_i = 15x_i - 5|tex}
  | "branin-rescaled" ->
      {tex|f(x) = \frac{f_{\mathrm{Branin}}(15x_1 - 5,\, 15x_2) - 54.81}{51.95}|tex}
  | "branin-modified" ->
      {tex|f(x) = f_{\mathrm{Branin}}(x_1, x_2) + 5x_1|tex}
  | "forrester-low-fidelity" ->
      {tex|f(x) = 0.5\,f_{\mathrm{Forrester}}(x) + 10(x - 0.5) + 5|tex}
  | "goldstein-price-rescaled" ->
      {tex|f(x) = \frac{\ln f_{\mathrm{GP}}(4x_1 - 2,\, 4x_2 - 2) - 8.693}{2.427}|tex}
  | "hartmann6-rescaled" ->
      {tex|f(x) = \frac{f_{\mathrm{Hart6}}(x) - 2.58}{1.94}|tex}
  | _ -> {tex|f(x)|tex}
