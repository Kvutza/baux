(* admin/ui/bench_suite.ml
   Pure OCaml implementation of all 56 BAUX synthetic optimization benchmark
   functions (matching ocaml/lib/problem.ml and ocaml/lib/formulas.ml) plus a
   deterministic in-browser multi-region TuRBO + Epistemic Nearest-Neighbor
   (ENN) optimizer supporting Euclidean, Conformal, Hyperbolic, and Wasserstein
   local-scale geometries across canonical, D=16, and D=64 regimes.
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

let default_dim = function
  | Fixed n | Multiple_of n -> n
  | At_least n -> Int.max 2 n

let is_scalable = function
  | Fixed _ -> false
  | At_least _ | Multiple_of _ -> true

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

let find_spec id =
  let found = ref None in
  for i = 0 to Array.length catalogue - 1 do
    if catalogue.(i).id = id then found := Some catalogue.(i)
  done;
  !found

(* --- Deterministic SplitMix64 PRNG (Works Identically in Native & JSOO) --- *)

module Rng = struct
  type t = { mutable state : int64 }

  let hash_string s =
    let h = ref 0x9e3779b97f4a7c15L in
    for i = 0 to String.length s - 1 do
      let c = Int64.of_int (Char.code s.[i]) in
      h := Int64.mul (Int64.logxor !h c) 0xbf58476d1ce4e5b9L;
      h := Int64.logxor !h (Int64.shift_right_logical !h 31)
    done;
    !h

  let create ~seed ~tag =
    let s0 =
      Int64.add (Int64.mul (Int64.of_int seed) 0x9e3779b97f4a7c15L)
        (hash_string tag)
    in
    { state = Int64.logor s0 1L }

  let next_u64 t =
    let z = Int64.add t.state 0x9e3779b97f4a7c15L in
    t.state <- z;
    let z =
      Int64.mul (Int64.logxor z (Int64.shift_right_logical z 30))
        0xbf58476d1ce4e5b9L
    in
    let z =
      Int64.mul (Int64.logxor z (Int64.shift_right_logical z 27))
        0x94d049bb133111ebL
    in
    Int64.logxor z (Int64.shift_right_logical z 31)

  let uniform01 t =
    let u53 = Int64.shift_right_logical (next_u64 t) 11 in
    Int64.to_float u53 *. (1.0 /. 9007199254740992.0)

  let int t bound =
    if bound <= 1 then 0
    else
      let r = Int64.to_int (Int64.shift_right_logical (next_u64 t) 33) in
      (r land 0x3fffffff) mod bound
end

(* --- Live Multi-Region TuRBO + Metric ENN Optimizer --- *)

type metric =
  | Hyperbolic
  | Wasserstein
  | Conformal
  | Euclidean

type arm =
  | Enn_multi of metric
  | Turbo_zero
  | Enn_single

let arm_of_key = function
  | "enn_h" -> Enn_multi Hyperbolic
  | "enn_w" -> Enn_multi Wasserstein
  | "enn_c" -> Enn_multi Conformal
  | "enn_e" -> Enn_multi Euclidean
  | "turbo0" -> Turbo_zero
  | "enn_r1" -> Enn_single
  | _ -> Enn_multi Hyperbolic

let key_of_arm = function
  | Enn_multi Hyperbolic -> "enn_h"
  | Enn_multi Wasserstein -> "enn_w"
  | Enn_multi Conformal -> "enn_c"
  | Enn_multi Euclidean -> "enn_e"
  | Turbo_zero -> "turbo0"
  | Enn_single -> "enn_r1"

let label_of_arm = function
  | Enn_multi Hyperbolic -> "TuRBO-ENN · Hyperbolic (R=4)"
  | Enn_multi Wasserstein -> "TuRBO-ENN · Wasserstein (R=4)"
  | Enn_multi Conformal -> "TuRBO-ENN · Conformal (R=4)"
  | Enn_multi Euclidean -> "TuRBO-ENN · Euclidean (R=4)"
  | Turbo_zero -> "TuRBO-0 · Random (R=4)"
  | Enn_single -> "Single-Region ENN (R=1)"

type region = {
  mutable center_u : float array;
  mutable best_y : float;
  mutable length : float;
  mutable succ : int;
  mutable fail : int;
  mutable visits : int;
}

type enn_model = {
  n : int;
  dim : int;
  u_pts : float array array;
  y_norm : float array;
  y_center : float;
  y_scale : float;
  scales : float array;
  mean_scale : float;
  k : int;
  metric : metric;
}

type posterior = {
  mean : float;
  sigma : float;
  lcb : float;
  neighbors : int array;
}

type session = {
  spec : spec;
  dim : int;
  bounds : (float * float) array;
  arm : arm;
  seed : int;
  rng : Rng.t;
  n_init : int;
  mutable xs_u : float array array;
  mutable ys : float array;
  mutable best_trace : float array;
  mutable best_idx : int;
  mutable last_region : int;
  regions : region array;
  mutable cached_model : enn_model option;
}

let clamp lo hi v = if v < lo then lo else if v > hi then hi else v

let u_to_x (bounds : (float * float) array) (u : float array) =
  let d = Array.length bounds in
  Array.init d (fun j ->
      let lo, hi = bounds.(j) in
      lo +. (clamp 0.0 1.0 u.(j) *. (hi -. lo)))

let safe_eval (spec : spec) (bounds : (float * float) array) (u : float array) =
  let x = u_to_x bounds u in
  let raw = spec.eval x in
  if Float.is_nan raw then 1.0e30
  else if raw > 1.0e30 then 1.0e30
  else if raw < -1.0e30 then -1.0e30
  else raw

(* Rank-preserving asinh compression keeps extreme polynomials (Perm, Rosenbrock,
   Goldstein-Price) numerically well-conditioned during ENN distance weighting. *)
let compress_y y center scale =
  let z = (y -. center) /. scale in
  Float.asinh z

let decompress_y z center scale = center +. (scale *. Float.sinh z)

let sq_dist (a : float array) (b : float array) =
  let s = ref 0.0 in
  for j = 0 to Array.length a - 1 do
    let d = a.(j) -. b.(j) in
    s := !s +. (d *. d)
  done;
  !s

let kth_smallest (arr : float array) k =
  let n = Array.length arr in
  if n = 0 then 0.1
  else
    let copy = Array.copy arr in
    Array.sort Float.compare copy;
    let idx = Int.max 0 (Int.min (n - 1) k) in
    copy.(idx)

let fit_enn (u_pts : float array array) (ys : float array) (dim : int)
    (metric : metric) : enn_model =
  let n = Array.length ys in
  let sorted_y = Array.copy ys in
  Array.sort Float.compare sorted_y;
  let y_med = sorted_y.(n / 2) in
  let q25 = sorted_y.(Int.max 0 (n / 4)) in
  let q75 = sorted_y.(Int.min (n - 1) ((3 * n) / 4)) in
  let iqr = q75 -. q25 in
  let span = sorted_y.(n - 1) -. sorted_y.(0) in
  let y_scale =
    if iqr > 1.0e-9 then iqr /. 1.349
    else if span > 1.0e-9 then span *. 0.5
    else 1.0
  in
  let y_norm = Array.init n (fun i -> compress_y ys.(i) y_med y_scale) in
  let ks = Int.max 1 (Int.min (n - 1) 4) in
  let root_d = Float.sqrt (float_of_int (Int.max 1 dim)) in
  let min_ell = 0.015 *. root_d in
  let scales = Array.make n min_ell in
  let sum_ell = ref 0.0 in
  for i = 0 to n - 1 do
    let dists = Array.make n 0.0 in
    for j = 0 to n - 1 do
      dists.(j) <-
        (if i = j then Float.infinity
         else Float.sqrt (sq_dist u_pts.(i) u_pts.(j)))
    done;
    let ell = Float.max min_ell (kth_smallest dists (ks - 1)) in
    scales.(i) <- ell;
    sum_ell := !sum_ell +. ell
  done;
  let mean_scale = !sum_ell /. float_of_int (Int.max 1 n) in
  let k = Int.max 2 (Int.min 8 n) in
  {
    n;
    dim;
    u_pts;
    y_norm;
    y_center = y_med;
    y_scale;
    scales;
    mean_scale;
    k;
    metric;
  }

let metric_distance (m : enn_model) (d2 : float) (ell_u : float) (ell_j : float)
    =
  match m.metric with
  | Euclidean -> Float.sqrt d2 /. Float.max 1.0e-6 m.mean_scale
  | Conformal -> Float.sqrt d2 /. Float.sqrt (ell_u *. ell_j)
  | Wasserstein ->
      let dl = ell_u -. ell_j in
      Float.sqrt (d2 +. (dl *. dl)) /. Float.max 1.0e-6 m.mean_scale
  | Hyperbolic ->
      let dl = ell_u -. ell_j in
      let arg = 1.0 +. ((d2 +. (dl *. dl)) /. (2.0 *. ell_u *. ell_j)) in
      Float.acosh (Float.max 1.0 arg)

let predict_enn ?(beta = 1.65) (m : enn_model) (u : float array) : posterior =
  let n = m.n in
  let raw_d2 = Array.make n 0.0 in
  let raw_d = Array.make n 0.0 in
  for j = 0 to n - 1 do
    let d2 = sq_dist u m.u_pts.(j) in
    raw_d2.(j) <- d2;
    raw_d.(j) <- Float.sqrt d2
  done;
  let ks = Int.max 1 (Int.min (n - 1) 4) in
  let min_ell = 0.015 *. Float.sqrt (float_of_int (Int.max 1 m.dim)) in
  let ell_u = Float.max min_ell (kth_smallest raw_d (ks - 1)) in
  let k = m.k in
  let best_idx = Array.make k (-1) in
  let best_dist = Array.make k Float.infinity in
  for j = 0 to n - 1 do
    let md = metric_distance m raw_d2.(j) ell_u m.scales.(j) in
    if md < best_dist.(k - 1) then begin
      let pos = ref (k - 1) in
      while !pos > 0 && md < best_dist.(!pos - 1) do
        best_dist.(!pos) <- best_dist.(!pos - 1);
        best_idx.(!pos) <- best_idx.(!pos - 1);
        decr pos
      done;
      best_dist.(!pos) <- md;
      best_idx.(!pos) <- j
    end
  done;
  let nugget2 = 1.0e-4 in
  let w_sum = ref 0.0 in
  let wy_sum = ref 0.0 in
  for r = 0 to k - 1 do
    let j = best_idx.(r) in
    if j >= 0 then begin
      let d = best_dist.(r) in
      let w = 1.0 /. ((d *. d) +. nugget2) in
      w_sum := !w_sum +. w;
      wy_sum := !wy_sum +. (w *. m.y_norm.(j))
    end
  done;
  let mu_z = if !w_sum > 0.0 then !wy_sum /. !w_sum else 0.0 in
  let var_loc = ref 0.0 in
  for r = 0 to k - 1 do
    let j = best_idx.(r) in
    if j >= 0 then begin
      let d = best_dist.(r) in
      let w = 1.0 /. ((d *. d) +. nugget2) in
      let diff = m.y_norm.(j) -. mu_z in
      var_loc := !var_loc +. (w *. diff *. diff)
    end
  done;
  let loc_term = if !w_sum > 0.0 then !var_loc /. !w_sum else 0.25 in
  let geom_term = if !w_sum > 0.0 then 1.0 /. !w_sum else 1.0 in
  let nearest_d = if k > 0 then best_dist.(0) else 1.0 in
  let taper = Float.min 1.0 (nearest_d /. 0.08) in
  let sig_z =
    Float.sqrt ((0.55 *. geom_term) +. (0.45 *. taper *. taper *. loc_term))
  in
  let mean = decompress_y mu_z m.y_center m.y_scale in
  let hi = decompress_y (mu_z +. sig_z) m.y_center m.y_scale in
  let lo = decompress_y (mu_z -. sig_z) m.y_center m.y_scale in
  let sigma = Float.max 1.0e-9 (0.5 *. Float.abs (hi -. lo)) in
  let lcb = decompress_y (mu_z -. (beta *. sig_z)) m.y_center m.y_scale in
  let valid_neighbors =
    best_idx |> Array.to_list |> List.filter (fun idx -> idx >= 0)
    |> Array.of_list
  in
  { mean; sigma; lcb; neighbors = valid_neighbors }

let ensure_model (s : session) : enn_model =
  match s.cached_model with
  | Some m when m.n = Array.length s.ys -> m
  | _ ->
      let metric =
        match s.arm with
        | Enn_multi met -> met
        | Turbo_zero | Enn_single -> Hyperbolic
      in
      let m = fit_enn s.xs_u s.ys s.dim metric in
      s.cached_model <- Some m;
      m

let make_lhs_design (rng : Rng.t) ~n_pts ~dim =
  let pts = Array.init n_pts (fun _ -> Array.make dim 0.5) in
  for j = 0 to dim - 1 do
    let perm = Array.init n_pts (fun i -> i) in
    for i = n_pts - 1 downto 1 do
      let k = Rng.int rng (i + 1) in
      let tmp = perm.(i) in
      perm.(i) <- perm.(k);
      perm.(k) <- tmp
    done;
    for i = 0 to n_pts - 1 do
      let u = (float_of_int perm.(i) +. Rng.uniform01 rng) /. float_of_int n_pts in
      pts.(i).(j) <- clamp 0.0 1.0 u
    done
  done;
  pts

let init_regions (arm : arm) (xs_u : float array array) (ys : float array) =
  let n_reg =
    match arm with Enn_multi _ | Turbo_zero -> 4 | Enn_single -> 1
  in
  let n_obs = Array.length ys in
  let order = Array.init n_obs (fun i -> i) in
  Array.sort (fun a b -> Float.compare ys.(a) ys.(b)) order;
  (* Spread multi-region centers across distinct well-separated initial points *)
  let chosen = Array.make n_reg order.(0) in
  let used = Array.make n_obs false in
  used.(order.(0)) <- true;
  for r = 1 to n_reg - 1 do
    let best_cand = ref order.(Int.min (n_obs - 1) r) in
    let best_score = ref neg_infinity in
    for idx = 0 to n_obs - 1 do
      let cand = order.(idx) in
      if not used.(cand) then begin
        let min_d = ref Float.infinity in
        for prev = 0 to r - 1 do
          let d = Float.sqrt (sq_dist xs_u.(cand) xs_u.(chosen.(prev))) in
          if d < !min_d then min_d := d
        done;
        let rank_bonus = 1.0 -. (float_of_int idx /. float_of_int n_obs) in
        let score = !min_d +. (0.28 *. rank_bonus) in
        if score > !best_score then begin
          best_score := score;
          best_cand := cand
        end
      end
    done;
    chosen.(r) <- !best_cand;
    used.(!best_cand) <- true
  done;
  let init_len = if n_reg = 1 then 0.52 else 0.36 in
  Array.init n_reg (fun r ->
      let idx = chosen.(r) in
      {
        center_u = Array.copy xs_u.(idx);
        best_y = ys.(idx);
        length = init_len;
        succ = 0;
        fail = 0;
        visits = 1;
      })

let create_session ?(seed = 1) ?(warm_steps = 24) ~(spec : spec) ~(dim : int)
    ~(arm : arm) () : session =
  let bounds = spec.domain dim in
  (* Common-random-number initial LHS design depends ONLY on (seed, spec.id, dim) *)
  let init_rng = Rng.create ~seed ~tag:(Printf.sprintf "init:%s:%d" spec.id dim) in
  let n_init = Int.max 8 (Int.min 16 ((2 * dim) + 4)) in
  let xs_u = make_lhs_design init_rng ~n_pts:n_init ~dim in
  let ys = Array.init n_init (fun i -> safe_eval spec bounds xs_u.(i)) in
  let best_trace = Array.make n_init ys.(0) in
  let best_idx = ref 0 in
  let running = ref ys.(0) in
  for i = 0 to n_init - 1 do
    if ys.(i) < !running then begin
      running := ys.(i);
      best_idx := i
    end;
    best_trace.(i) <- !running
  done;
  let step_rng =
    Rng.create ~seed
      ~tag:(Printf.sprintf "step:%s:%d:%s" spec.id dim (key_of_arm arm))
  in
  let regions = init_regions arm xs_u ys in
  let sess =
    {
      spec;
      dim;
      bounds;
      arm;
      seed;
      rng = step_rng;
      n_init;
      xs_u;
      ys;
      best_trace;
      best_idx = !best_idx;
      last_region = 0;
      regions;
      cached_model = None;
    }
  in
  for _ = 1 to warm_steps do
    (* Forward declaration via helper below *)
    ignore sess
  done;
  sess

let raasp_candidate ?(full = false) (rng : Rng.t) (center : float array)
    (length : float) =
  let d = Array.length center in
  let cand = Array.copy center in
  let p_pert =
    if full || d <= 4 then 1.0
    else
      let block = Float.sqrt (float_of_int d) +. 4.0 in
      Float.min 1.0 (block /. float_of_int d)
  in
  let any_perturbed = ref false in
  let half = 0.5 *. length in
  for j = 0 to d - 1 do
    if Rng.uniform01 rng <= p_pert then begin
      any_perturbed := true;
      let lo = Float.max 0.0 (center.(j) -. half) in
      let hi = Float.min 1.0 (center.(j) +. half) in
      cand.(j) <- lo +. (Rng.uniform01 rng *. (hi -. lo))
    end
  done;
  if not !any_perturbed then begin
    let j = Rng.int rng d in
    let lo = Float.max 0.0 (center.(j) -. half) in
    let hi = Float.min 1.0 (center.(j) +. half) in
    cand.(j) <- lo +. (Rng.uniform01 rng *. (hi -. lo))
  end;
  cand

let select_region (s : session) =
  let n_reg = Array.length s.regions in
  if n_reg <= 1 then 0
  else
    let total_v =
      Array.fold_left (fun acc r -> acc + r.visits) 0 s.regions
      |> Int.max 1 |> float_of_int
    in
    let y_vals = Array.map (fun r -> r.best_y) s.regions in
    let y_lo = Array.fold_left Float.min Float.infinity y_vals in
    let y_hi = Array.fold_left Float.max Float.neg_infinity y_vals in
    let span = Float.max 1.0e-9 (y_hi -. y_lo) in
    let best_r = ref 0 in
    let best_ucb = ref Float.infinity in
    for r = 0 to n_reg - 1 do
      let reg = s.regions.(r) in
      let norm_y = (reg.best_y -. y_lo) /. span in
      let bonus =
        0.42
        *. Float.sqrt (Float.log (total_v +. 1.0) /. float_of_int reg.visits)
      in
      let score = norm_y -. bonus in
      if score < !best_ucb then begin
        best_ucb := score;
        best_r := r
      end
    done;
    !best_r

let restart_region (s : session) (reg : region) =
  let best_cand = ref (Array.init s.dim (fun _ -> Rng.uniform01 s.rng)) in
  let best_min_d = ref (-1.0) in
  for _ = 1 to 20 do
    let c = Array.init s.dim (fun _ -> Rng.uniform01 s.rng) in
    let min_d = ref Float.infinity in
    for i = 0 to Array.length s.xs_u - 1 do
      let d2 = sq_dist c s.xs_u.(i) in
      if d2 < !min_d then min_d := d2
    done;
    if !min_d > !best_min_d then begin
      best_min_d := !min_d;
      best_cand := c
    end
  done;
  let y_new = safe_eval s.spec s.bounds !best_cand in
  reg.center_u <- !best_cand;
  reg.best_y <- y_new;
  reg.length <- (if Array.length s.regions = 1 then 0.50 else 0.36);
  reg.succ <- 0;
  reg.fail <- 0;
  (!best_cand, y_new)

let step_once (s : session) =
  let r_idx = select_region s in
  s.last_region <- r_idx;
  let reg = s.regions.(r_idx) in
  reg.visits <- reg.visits + 1;
  let cand_u, y_val =
    match s.arm with
    | Turbo_zero ->
        let c = raasp_candidate s.rng reg.center_u reg.length in
        (c, safe_eval s.spec s.bounds c)
    | Enn_multi _ | Enn_single ->
        let m = ensure_model s in
        let n_pool = if s.dim >= 16 then 56 else 40 in
        let best_c = ref (raasp_candidate s.rng reg.center_u reg.length) in
        let best_lcb = ref Float.infinity in
        for p = 0 to n_pool - 1 do
          let c =
            if p = 0 then !best_c
            else if p = n_pool - 1 && Array.length s.regions > 1 then
              Array.init s.dim (fun _ -> Rng.uniform01 s.rng)
            else
              raasp_candidate ~full:(p mod 3 = 1) s.rng reg.center_u reg.length
          in
          let post = predict_enn ~beta:1.75 m c in
          if post.lcb < !best_lcb then begin
            best_lcb := post.lcb;
            best_c := c
          end
        done;
        (!best_c, safe_eval s.spec s.bounds !best_c)
  in
  let tol = 1.0e-4 *. Float.max 1.0 (Float.abs reg.best_y) in
  if y_val < reg.best_y -. tol then begin
    reg.center_u <- Array.copy cand_u;
    reg.best_y <- y_val;
    reg.succ <- reg.succ + 1;
    reg.fail <- 0
  end
  else begin
    if y_val < reg.best_y then begin
      reg.center_u <- Array.copy cand_u;
      reg.best_y <- y_val
    end;
    reg.succ <- 0;
    reg.fail <- reg.fail + 1
  end;
  let fail_tol = Int.max 4 (Int.min 10 (2 + (s.dim / 4))) in
  if reg.succ >= 3 then begin
    reg.length <- Float.min 0.80 (reg.length *. 1.8);
    reg.succ <- 0
  end
  else if reg.fail >= fail_tol then begin
    reg.length <- reg.length *. 0.5;
    reg.fail <- 0
  end;
  let final_u, final_y =
    if reg.length < 0.018 then restart_region s reg else (cand_u, y_val)
  in
  let old_n = Array.length s.ys in
  let next_xs = Array.make (old_n + 1) final_u in
  Array.blit s.xs_u 0 next_xs 0 old_n;
  let next_ys = Array.make (old_n + 1) final_y in
  Array.blit s.ys 0 next_ys 0 old_n;
  let prev_best = s.ys.(s.best_idx) in
  let new_best =
    if final_y < prev_best then begin
      s.best_idx <- old_n;
      final_y
    end
    else prev_best
  in
  let next_trace = Array.make (old_n + 1) new_best in
  Array.blit s.best_trace 0 next_trace 0 old_n;
  s.xs_u <- next_xs;
  s.ys <- next_ys;
  s.best_trace <- next_trace;
  s.cached_model <- None

let step_many (s : session) n_steps =
  for _ = 1 to n_steps do
    step_once s
  done

let init_session ?(seed = 1) ?(warm_steps = 24) ~(spec : spec) ~(dim : int)
    ~(arm : arm) () : session =
  let s = create_session ~seed ~warm_steps:0 ~spec ~dim ~arm () in
  step_many s warm_steps;
  s
