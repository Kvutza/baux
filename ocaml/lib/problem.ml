module Iarray = Stdlib_stable.Iarray
module F = Formulas.Scalar
module V = Formulas.Simd

type point = float iarray
type dimensions = Fixed of int | At_least of int | Multiple_of of int
type family = Multimodal | Bowl | Plate | Valley | Ridge | Other

type description = {
  id : string;
  family : family;
  dimensions : dimensions;
  source : string;
}

type entry = {
  description : description;
  domain : int -> (float * float) iarray @@ portable;
  formula : Numeric.input @ local -> float# @@ portable;
  vector : Numeric.input @ local -> float64x2# @@ portable;
}

type t = { entry : entry; dimension : int; bounds : (float * float) iarray }

let uniform (low : float) (high : float) : (int -> (float * float) iarray)
    @ portable =
 fun n -> Iarray.init n (fun _ -> (low, high))

let fixed (bounds : (float * float) iarray) : (int -> (float * float) iarray)
    @ portable =
 fun _ -> bounds

let entry id family dimensions page domain formula vector =
  {
    description =
      {
        id;
        family;
        dimensions;
        source = "https://www.sfu.ca/~ssurjano/" ^ page ^ ".html";
      };
    domain;
    formula;
    vector;
  }

let entries =
  let any = At_least 1 and two = Fixed 2 in
  [
    entry "ackley" Multimodal any "ackley" (uniform (-32.768) 32.768) F.ackley
      V.ackley;
    entry "bukin6" Multimodal two "bukin6"
      (fixed [: (-15., -5.); (-3., 3.) :])
      F.bukin6 V.bukin6;
    entry "cross-in-tray" Multimodal two "crossit" (uniform (-10.) 10.)
      F.cross_in_tray V.cross_in_tray;
    entry "drop-wave" Multimodal two "drop" (uniform (-5.12) 5.12) F.drop_wave
      V.drop_wave;
    entry "eggholder" Multimodal two "egg" (uniform (-512.) 512.) F.eggholder
      V.eggholder;
    entry "gramacy-lee" Multimodal (Fixed 1) "grlee12" (uniform 0.5 2.5)
      F.gramacy_lee V.gramacy_lee;
    entry "griewank" Multimodal any "griewank" (uniform (-600.) 600.) F.griewank
      V.griewank;
    entry "holder-table" Multimodal two "holder" (uniform (-10.) 10.)
      F.holder_table V.holder_table;
    entry "langermann" Multimodal two "langer" (uniform 0. 10.) F.langermann
      V.langermann;
    entry "levy" Multimodal any "levy" (uniform (-10.) 10.) F.levy V.levy;
    entry "levy13" Multimodal two "levy13" (uniform (-10.) 10.) F.levy13
      V.levy13;
    entry "rastrigin" Multimodal any "rastr" (uniform (-5.12) 5.12) F.rastrigin
      V.rastrigin;
    entry "schaffer2" Multimodal two "schaffer2" (uniform (-100.) 100.)
      F.schaffer2 V.schaffer2;
    entry "schaffer4" Multimodal two "schaffer4" (uniform (-100.) 100.)
      F.schaffer4 V.schaffer4;
    entry "schwefel" Multimodal any "schwef" (uniform (-500.) 500.) F.schwefel
      V.schwefel;
    entry "shubert" Multimodal two "shubert" (uniform (-10.) 10.) F.shubert
      V.shubert;
    entry "bohachevsky1" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> F.bohachevsky 1 x)
      (fun x -> V.bohachevsky 1 x);
    entry "bohachevsky2" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> F.bohachevsky 2 x)
      (fun x -> V.bohachevsky 2 x);
    entry "bohachevsky3" Bowl two "boha" (uniform (-100.) 100.)
      (fun x -> F.bohachevsky 3 x)
      (fun x -> V.bohachevsky 3 x);
    entry "perm-zero" Bowl any "perm0db"
      (fun n -> uniform (-.float_of_int n) (float_of_int n) n)
      F.perm_zero V.perm_zero;
    entry "rotated-hyperellipsoid" Bowl any "rothyp" (uniform (-65.536) 65.536)
      F.rotated_hyperellipsoid V.rotated_hyperellipsoid;
    entry "sphere" Bowl any "spheref" (uniform (-5.12) 5.12) F.sphere V.sphere;
    entry "sum-powers" Bowl any "sumpow" (uniform (-1.) 1.) F.sum_powers
      V.sum_powers;
    entry "sum-squares" Bowl any "sumsqu" (uniform (-10.) 10.) F.sum_squares
      V.sum_squares;
    entry "trid" Bowl (At_least 2) "trid"
      (fun n ->
        let d = float_of_int n in
        uniform (-.d *. d) (d *. d) n)
      F.trid V.trid;
    entry "booth" Plate two "booth" (uniform (-10.) 10.) F.booth V.booth;
    entry "matyas" Plate two "matya" (uniform (-10.) 10.) F.matyas V.matyas;
    entry "mccormick" Plate two "mccorm"
      (fixed [: (-1.5, 4.); (-3., 4.) :])
      F.mccormick V.mccormick;
    entry "power-sum" Plate (Fixed 4) "powersum" (uniform 0. 4.) F.power_sum
      V.power_sum;
    entry "zakharov" Plate any "zakharov" (uniform (-5.) 10.) F.zakharov
      V.zakharov;
    entry "three-hump-camel" Valley two "camel3" (uniform (-5.) 5.)
      F.three_hump_camel V.three_hump_camel;
    entry "six-hump-camel" Valley two "camel6"
      (fixed [: (-3., 3.); (-2., 2.) :])
      F.six_hump_camel V.six_hump_camel;
    entry "dixon-price" Valley (At_least 2) "dixonpr" (uniform (-10.) 10.)
      F.dixon_price V.dixon_price;
    entry "rosenbrock" Valley (At_least 2) "rosen" (uniform (-5.) 10.)
      F.rosenbrock V.rosenbrock;
    entry "de-jong5" Ridge two "dejong5" (uniform (-65.536) 65.536) F.de_jong5
      V.de_jong5;
    entry "easom" Ridge two "easom" (uniform (-100.) 100.) F.easom V.easom;
    entry "michalewicz" Ridge any "michal" (uniform 0. Float.pi) F.michalewicz
      V.michalewicz;
    entry "beale" Other two "beale" (uniform (-4.5) 4.5) F.beale V.beale;
    entry "branin" Other two "branin"
      (fixed [: (-5., 10.); (0., 15.) :])
      F.branin V.branin;
    entry "colville" Other (Fixed 4) "colville" (uniform (-10.) 10.) F.colville
      V.colville;
    entry "forrester" Other (Fixed 1) "forretal08" (uniform 0. 1.) F.forrester
      V.forrester;
    entry "goldstein-price" Other two "goldpr" (uniform (-2.) 2.)
      F.goldstein_price V.goldstein_price;
    entry "hartmann3" Other (Fixed 3) "hart3" (uniform 0. 1.) F.hartmann
      V.hartmann;
    entry "hartmann4" Other (Fixed 4) "hart4" (uniform 0. 1.) F.hartmann4
      V.hartmann4;
    entry "hartmann6" Other (Fixed 6) "hart6" (uniform 0. 1.) F.hartmann
      V.hartmann;
    entry "perm" Other any "permdb"
      (fun n -> uniform (-.float_of_int n) (float_of_int n) n)
      F.perm V.perm;
    entry "powell" Other (Multiple_of 4) "powell" (uniform (-4.) 5.) F.powell
      V.powell;
    entry "shekel" Other (Fixed 4) "shekel" (uniform 0. 10.) F.shekel V.shekel;
    entry "styblinski-tang" Other any "stybtang" (uniform (-5.) 5.)
      F.styblinski_tang V.styblinski_tang;
    entry "sphere6-rescaled" Bowl (Fixed 6) "spheref" (uniform 0. 1.)
      F.sphere6_rescaled V.sphere6_rescaled;
    entry "rosenbrock4-rescaled" Valley (Fixed 4) "rosen" (uniform 0. 1.)
      F.rosenbrock4_rescaled V.rosenbrock4_rescaled;
    entry "branin-rescaled" Other two "branin" (uniform 0. 1.) F.branin_rescaled
      V.branin_rescaled;
    entry "branin-modified" Other two "branin"
      (fixed [: (-5., 10.); (0., 15.) :])
      F.branin_modified V.branin_modified;
    entry "forrester-low-fidelity" Other (Fixed 1) "forretal08" (uniform 0. 1.)
      F.forrester_low_fidelity V.forrester_low_fidelity;
    entry "goldstein-price-rescaled" Other two "goldpr" (uniform 0. 1.)
      F.goldstein_price_rescaled V.goldstein_price_rescaled;
    entry "hartmann6-rescaled" Other (Fixed 6) "hart6" (uniform 0. 1.)
      F.hartmann6_rescaled V.hartmann6_rescaled;
  ]

let catalogue = List.map (fun e -> e.description) entries

let create ?dimension id =
  let entry =
    match List.find_opt (fun e -> e.description.id = id) entries with
    | Some e -> e
    | None -> invalid_arg ("Unknown benchmark: " ^ id)
  in
  let rule = entry.description.dimensions in
  let dimension =
    match (dimension, rule) with
    | Some n, _ -> n
    | None, (Fixed n | Multiple_of n) -> n
    | None, At_least n -> max 2 n
  in
  let valid =
    dimension > 0
    &&
    match rule with
    | Fixed n -> dimension = n
    | At_least n -> dimension >= n
    | Multiple_of n -> dimension mod n = 0
  in
  if not valid then invalid_arg ("Invalid dimension for " ^ id);
  { entry; dimension; bounds = entry.domain dimension }

let describe t = t.entry.description
let dimension t = t.dimension
let bounds t = t.bounds

let validate t x =
  if Iarray.length x <> t.dimension then invalid_arg "Point dimension mismatch";
  Iarray.iter
    (fun v ->
      if not (Float.is_finite v) then invalid_arg "Non-finite coordinate")
    x

let evaluate t x =
  validate t x;
  t.entry.formula (stack_ Numeric.Point x) [@nontail]

let within_bounds t x =
  if Iarray.length x <> t.dimension then false
  else
    let rec loop i =
      if i = t.dimension then true
      else
        let low, high = Iarray.get t.bounds i in
        let v = Iarray.get x i in
        Float.is_finite v && low <= v && v <= high && loop (i + 1)
    in
    loop 0

module Prepared = struct
  type problem = t
  type t = { problem : problem; points : point iarray; pairs : point iarray }

  let create problem points =
    Iarray.iter (validate problem) points;
    let count = Iarray.length points in
    if problem.dimension > Sys.max_array_length / 2 then
      invalid_arg "SIMD point dimension exceeds array capacity";
    let pairs =
      Iarray.init
        ((count / 2) + (count mod 2))
        (fun pair ->
          let left = Iarray.get points (2 * pair) in
          let right = Iarray.get points (min ((2 * pair) + 1) (count - 1)) in
          Iarray.init (2 * problem.dimension) (fun coordinate ->
              Iarray.get
                (if coordinate mod 2 = 0 then left else right)
                (coordinate / 2)))
    in
    { problem; points; pairs }

  let length t = Iarray.length t.points
  let pairs t = Iarray.length t.pairs

  let scalar t i =
    t.problem.entry.formula
      (stack_ Numeric.Point (Iarray.get t.points i)) [@nontail]

  let simd t i =
    t.problem.entry.vector
      (stack_ Numeric.Point (Iarray.get t.pairs i)) [@nontail]
end

module Packed = struct
  let scalar t data ~offset =
    t.entry.formula
      (stack_ Numeric.Packed { data; offset; dimension = t.dimension })
    [@nontail]

  let simd t data ~offset =
    t.entry.vector
      (stack_ Numeric.Packed { data; offset; dimension = t.dimension })
    [@nontail]
end
