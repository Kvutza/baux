module Iarray = Stdlib_stable.Iarray

module%template Make (N : Numeric.S [@kind k]) = struct
  open N

  let[@inline always] ( +. ) x y = add x y
  let[@inline always] ( -. ) x y = sub x y
  let[@inline always] ( *. ) x y = mul x y
  let[@inline always] ( /. ) x y = div x y
  let[@inline always] ( ** ) x y = pow x y
  let[@inline always] square x = mul x x

  let[@zero_alloc] [@inline always] powi x n =
    let rec loop x n acc =
      if n = 0 then acc
      else
        let acc = if n land 1 = 1 then acc *. x else acc in
        if n = 1 then acc else loop (square x) (n lsr 1) acc
    in
    loop x n (constant #1.)

  let[@inline always] pi () = constant (Numeric.U.of_float Float.pi)
  let[@inline always] coeff x i = constant (Numeric.U.of_float (Iarray.get x i))

  let[@inline always] sum n ~(f : (int -> t) @ local) =
    let rec loop i acc = if i = n then acc else loop (i + 1) (add acc (f i)) in
    loop 0 (constant #0.) [@nontail]

  let[@inline always] product n ~(f : (int -> t) @ local) =
    let rec loop i acc = if i = n then acc else loop (i + 1) (mul acc (f i)) in
    loop 0 (constant #1.) [@nontail]

  let[@zero_alloc] sphere (x @ local) =
    let (f @ local) i = square (get x i) in
    sum (dimension x) ~f [@nontail]

  let[@zero_alloc] rosenbrock (x @ local) =
    sum
      (dimension x - 1)
      ~f:(stack_ fun i ->
        (constant #100. *. square (get x (i + 1) -. square (get x i)))
        +. square (constant #1. -. get x i)) [@nontail]

  let[@zero_alloc] ackley (x @ local) =
    let n = dimension x in
    let d = of_int n in
    (neg (constant #20.) *. exp (neg (constant #0.2) *. sqrt (sphere x /. d)))
    -. exp
         ((sum n ~f:(stack_ fun i -> cos (constant #2. *. pi () *. get x i))
          [@nontail])
         /. d)
    +. constant #20.
    +. exp (constant #1.)

  let[@zero_alloc] rastrigin (x @ local) =
    (constant #10. *. of_int (dimension x))
    +. (sum (dimension x) ~f:(stack_ fun i ->
            square (get x i)
            -. (constant #10. *. cos (constant #2. *. pi () *. get x i)))
       [@nontail])

  let[@zero_alloc] griewank (x @ local) =
    constant #1.
    +. (sphere x /. constant #4000.)
    -. (product (dimension x) ~f:(stack_ fun i ->
            cos (get x i /. sqrt (of_int (i + 1))))
       [@nontail])

  let[@zero_alloc] schwefel (x @ local) =
    (constant #418.9829 *. of_int (dimension x))
    -. (sum (dimension x) ~f:(stack_ fun i ->
            get x i *. sin (sqrt (abs (get x i))))
       [@nontail])

  let[@zero_alloc] levy (x @ local) =
    let w i = constant #1. +. ((get x i -. constant #1.) /. constant #4.) in
    let last = w (dimension x - 1) in
    square (sin (pi () *. w 0))
    +. (sum
          (dimension x - 1)
          ~f:(stack_ fun i ->
            square (w i -. constant #1.)
            *. (constant #1.
               +. constant #10.
                  *. square (sin ((pi () *. w i) +. constant #1.)))) [@nontail])
    +. square (last -. constant #1.)
       *. (constant #1. +. square (sin (constant #2. *. pi () *. last)))

  let[@zero_alloc] sum_squares (x @ local) =
    sum (dimension x) ~f:(stack_ fun i -> of_int (i + 1) *. square (get x i))
    [@nontail]

  let[@zero_alloc] rotated_hyperellipsoid (x @ local) =
    sum (dimension x) ~f:(stack_ fun i ->
        of_int (dimension x - i) *. square (get x i))
    [@nontail]

  let[@zero_alloc] sum_powers (x @ local) =
    sum (dimension x) ~f:(stack_ fun i -> powi (abs (get x i)) (i + 2))
    [@nontail]

  let[@zero_alloc] trid (x @ local) =
    (sum (dimension x) ~f:(stack_ fun i -> square (get x i -. constant #1.))
    [@nontail])
    -. (sum
          (dimension x - 1)
          ~f:(stack_ fun i -> get x i *. get x (i + 1)) [@nontail])

  let[@zero_alloc] zakharov (x @ local) =
    let a =
      sum (dimension x) ~f:(stack_ fun i ->
          constant #0.5 *. of_int (i + 1) *. get x i)
      [@nontail]
    in
    sphere x +. square a +. square (square a)

  let[@zero_alloc] dixon_price (x @ local) =
    square (get x 0 -. constant #1.)
    +. (sum
          (dimension x - 1)
          ~f:(stack_ fun i ->
            of_int (i + 2)
            *. square ((constant #2. *. square (get x (i + 1))) -. get x i))
        [@nontail])

  let[@zero_alloc] michalewicz (x @ local) =
    neg
      (sum (dimension x) ~f:(stack_ fun i ->
           sin (get x i)
           *. powi (sin (of_int (i + 1) *. square (get x i) /. pi ())) 20)
      [@nontail])

  let[@zero_alloc] styblinski_tang (x @ local) =
    constant #0.5
    *. (sum (dimension x) ~f:(stack_ fun i ->
            let v = get x i in
            square (square v)
            -. (constant #16. *. square v)
            +. (constant #5. *. v))
       [@nontail])

  let[@zero_alloc] perm_zero (x @ local) =
    let n = dimension x in
    sum n ~f:(stack_ fun k ->
        let power = k + 1 in
        square
          (sum n ~f:(stack_ fun i ->
               let j = of_int (i + 1) in
               (j +. constant #10.)
               *. (powi (get x i) power -. powi (constant #1. /. j) power))
          [@nontail]))
    [@nontail]

  let[@zero_alloc] perm (x @ local) =
    let n = dimension x in
    sum n ~f:(stack_ fun k ->
        let power = k + 1 in
        square
          (sum n ~f:(stack_ fun i ->
               let j = of_int (i + 1) in
               (powi j power +. constant #0.5)
               *. (powi (get x i /. j) power -. constant #1.))
          [@nontail]))
    [@nontail]

  let[@zero_alloc] powell (x @ local) =
    sum
      (dimension x / 4)
      ~f:(stack_ fun block ->
        let i = 4 * block in
        let a = get x i and b = get x (i + 1) in
        let c = get x (i + 2) and d = get x (i + 3) in
        square (a +. (constant #10. *. b))
        +. (constant #5. *. square (c -. d))
        +. square (square (b -. (constant #2. *. c)))
        +. (constant #10. *. square (square (a -. d)))) [@nontail]

  let[@zero_alloc] bukin6 (x @ local) =
    constant #100.
    *. sqrt (abs (get x 1 -. (constant #0.01 *. square (get x 0))))
    +. (constant #0.01 *. abs (get x 0 +. constant #10.))

  let[@zero_alloc] cross_in_tray (x @ local) =
    let a = get x 0 and b = get x 1 in
    neg (constant #0.0001)
    *. (abs
          (sin a *. sin b
          *. exp
               (abs (constant #100. -. (sqrt (square a +. square b) /. pi ())))
          )
       +. constant #1.)
       ** constant #0.1

  let[@zero_alloc] drop_wave (x @ local) =
    let r2 = sphere x in
    neg (constant #1. +. cos (constant #12. *. sqrt r2))
    /. (constant #2. +. (constant #0.5 *. r2))

  let[@zero_alloc] eggholder (x @ local) =
    let a = get x 0 and b = get x 1 +. constant #47. in
    (neg b *. sin (sqrt (abs ((a /. constant #2.) +. b))))
    -. (a *. sin (sqrt (abs (a -. b))))

  let[@zero_alloc] gramacy_lee (x @ local) =
    let a = get x 0 in
    (sin (constant #10. *. pi () *. a) /. (constant #2. *. a))
    +. square (square (a -. constant #1.))

  let[@zero_alloc] holder_table (x @ local) =
    let a = get x 0 and b = get x 1 in
    neg
      (abs
         (sin a *. cos b
         *. exp (abs (constant #1. -. (sqrt (square a +. square b) /. pi ())))))

  let[@zero_alloc] langermann (x @ local) =
    let centers =
      [: [: 3.; 5. :]; [: 5.; 2. :]; [: 2.; 1. :]; [: 1.; 4. :]; [: 7.; 9. :] :]
    in
    let weights = [: 1.; 2.; 5.; 2.; 3. :] in
    sum 5 ~f:(stack_ fun i ->
        let center = Iarray.get centers i in
        let r2 =
          sum 2 ~f:(stack_ fun j -> square (get x j -. coeff center j))
          [@nontail]
        in
        coeff weights i *. exp (neg r2 /. pi ()) *. cos (pi () *. r2))
    [@nontail]

  let[@zero_alloc] levy13 (x @ local) =
    let a = get x 0 and b = get x 1 in
    square (sin (constant #3. *. pi () *. a))
    +. square (a -. constant #1.)
       *. (constant #1. +. square (sin (constant #3. *. pi () *. b)))
    +. square (b -. constant #1.)
       *. (constant #1. +. square (sin (constant #2. *. pi () *. b)))

  let[@zero_alloc] schaffer2 (x @ local) =
    let a = square (get x 0) and b = square (get x 1) in
    constant #0.5
    +. (square (sin (a -. b)) -. constant #0.5)
       /. square (constant #1. +. (constant #0.001 *. (a +. b)))

  let[@zero_alloc] schaffer4 (x @ local) =
    let a = square (get x 0) and b = square (get x 1) in
    constant #0.5
    +. (square (cos (sin (abs (a -. b)))) -. constant #0.5)
       /. square (constant #1. +. (constant #0.001 *. (a +. b)))

  let[@zero_alloc] shubert (x @ local) =
    product 2 ~f:(stack_ fun j ->
        sum 5 ~f:(stack_ fun i ->
            let k = of_int (i + 1) in
            k *. cos (((k +. constant #1.) *. get x j) +. k))
        [@nontail])
    [@nontail]

  let[@zero_alloc] bohachevsky kind (x @ local) =
    let a = get x 0 and b = get x 1 in
    let u = constant #3. *. pi () *. a and v = constant #4. *. pi () *. b in
    square a
    +. (constant #2. *. square b)
    +.
    match kind with
    | 1 -> constant #0.7 -. (constant #0.3 *. cos u) -. (constant #0.4 *. cos v)
    | 2 -> constant #0.3 -. (constant #0.3 *. cos u *. cos v)
    | _ -> constant #0.3 -. (constant #0.3 *. cos (u +. v))

  let[@zero_alloc] booth (x @ local) =
    let a = get x 0 and b = get x 1 in
    square (a +. (constant #2. *. b) -. constant #7.)
    +. square ((constant #2. *. a) +. b -. constant #5.)

  let[@zero_alloc] matyas (x @ local) =
    (constant #0.26 *. sphere x) -. (constant #0.48 *. get x 0 *. get x 1)

  let[@zero_alloc] mccormick (x @ local) =
    let a = get x 0 and b = get x 1 in
    sin (a +. b)
    +. square (a -. b)
    -. (constant #1.5 *. a)
    +. (constant #2.5 *. b)
    +. constant #1.

  let[@zero_alloc] power_sum (x @ local) =
    let targets = [: 8.; 18.; 44.; 114. :] in
    sum 4 ~f:(stack_ fun k ->
        square
          ((sum 4 ~f:(stack_ fun i -> powi (get x i) (k + 1)) [@nontail])
          -. coeff targets k))
    [@nontail]

  let[@zero_alloc] three_hump_camel (x @ local) =
    let a = get x 0 and b = get x 1 in
    (constant #2. *. square a)
    -. (constant #1.05 *. square (square a))
    +. (powi a 6 /. constant #6.)
    +. (a *. b) +. square b

  let[@zero_alloc] six_hump_camel (x @ local) =
    let a = get x 0 and b = get x 1 in
    (constant #4.
    -. (constant #2.1 *. square a)
    +. (square (square a) /. constant #3.))
    *. square a
    +. (a *. b)
    +. ((neg (constant #4.) +. (constant #4. *. square b)) *. square b)

  let[@zero_alloc] de_jong5 (x @ local) =
    constant #1.
    /. (constant #0.002
       +. (sum 25 ~f:(stack_ fun i ->
               let a = of_int (16 * ((i mod 5) - 2)) in
               let b = of_int (16 * ((i / 5) - 2)) in
               constant #1.
               /. (of_int (i + 1)
                  +. powi (get x 0 -. a) 6
                  +. powi (get x 1 -. b) 6))
          [@nontail]))

  let[@zero_alloc] easom (x @ local) =
    let a = get x 0 and b = get x 1 in
    neg (cos a *. cos b)
    *. exp (neg (square (a -. pi ()) +. square (b -. pi ())))

  let[@zero_alloc] beale (x @ local) =
    let a = get x 0 and b = get x 1 in
    square (constant #1.5 -. a +. (a *. b))
    +. square (constant #2.25 -. a +. (a *. square b))
    +. square (constant #2.625 -. a +. (a *. powi b 3))

  let[@zero_alloc] branin_at a b =
    square
      (b
      -. (constant #5.1 *. square a /. (constant #4. *. square (pi ())))
      +. (constant #5. *. a /. pi ())
      -. constant #6.)
    +. constant #10.
       *. (constant #1. -. (constant #1. /. (constant #8. *. pi ())))
       *. cos a
    +. constant #10.

  let[@zero_alloc] branin (x @ local) = branin_at (get x 0) (get x 1)

  let[@zero_alloc] colville (x @ local) =
    let a = get x 0 and b = get x 1 in
    let c = get x 2 and d = get x 3 in
    (constant #100. *. square (square a -. b))
    +. square (a -. constant #1.)
    +. square (c -. constant #1.)
    +. (constant #90. *. square (square c -. d))
    +. constant #10.1
       *. (square (b -. constant #1.) +. square (d -. constant #1.))
    +. (constant #19.8 *. (b -. constant #1.) *. (d -. constant #1.))

  let[@zero_alloc] forrester (x @ local) =
    let a = get x 0 in
    square ((constant #6. *. a) -. constant #2.)
    *. sin ((constant #12. *. a) -. constant #4.)

  let[@zero_alloc] goldstein_price_at a b =
    (constant #1.
    +. square (a +. b +. constant #1.)
       *. (constant #19.
          -. (constant #14. *. a)
          +. (constant #3. *. square a)
          -. (constant #14. *. b)
          +. (constant #6. *. a *. b)
          +. (constant #3. *. square b)))
    *. (constant #30.
       +. square ((constant #2. *. a) -. (constant #3. *. b))
          *. (constant #18.
             -. (constant #32. *. a)
             +. (constant #12. *. square a)
             +. (constant #48. *. b)
             -. (constant #36. *. a *. b)
             +. (constant #27. *. square b)))

  let[@zero_alloc] goldstein_price (x @ local) =
    goldstein_price_at (get x 0) (get x 1)

  let[@zero_alloc] hartmann (x @ local) =
    let alpha = [: 1.; 1.2; 3.; 3.2 :] in
    let a, p =
      if dimension x = 3 then
        ( [:
            [: 3.; 10.; 30. :];
            [: 0.1; 10.; 35. :];
            [: 3.; 10.; 30. :];
            [: 0.1; 10.; 35. :];
          :],
          [:
            [: 3689.; 1170.; 2673. :];
            [: 4699.; 4387.; 7470. :];
            [: 1091.; 8732.; 5547. :];
            [: 381.; 5743.; 8828. :];
          :] )
      else
        ( [:
            [: 10.; 3.; 17.; 3.5; 1.7; 8. :];
            [: 0.05; 10.; 17.; 0.1; 8.; 14. :];
            [: 3.; 3.5; 1.7; 10.; 17.; 8. :];
            [: 17.; 8.; 0.05; 10.; 0.1; 14. :];
          :],
          [:
            [: 1312.; 1696.; 5569.; 124.; 8283.; 5886. :];
            [: 2329.; 4135.; 8307.; 3736.; 1004.; 9991. :];
            [: 2348.; 1451.; 3522.; 2883.; 3047.; 6650. :];
            [: 4047.; 8828.; 8732.; 5743.; 1091.; 381. :];
          :] )
    in
    neg
      (sum 4 ~f:(stack_ fun i ->
           coeff alpha i
           *. exp
                (neg
                   (sum (dimension x) ~f:(stack_ fun j ->
                        coeff (Iarray.get a i) j
                        *. square
                             (get x j
                             -. (constant #0.0001 *. coeff (Iarray.get p i) j)))
                   [@nontail])))
      [@nontail])

  let[@zero_alloc] hartmann4 (x @ local) =
    (constant #1.1 +. hartmann x) /. constant #0.839

  let[@zero_alloc] shekel (x @ local) =
    let c =
      [:
        [: 4.; 4.; 4.; 4. :];
        [: 1.; 1.; 1.; 1. :];
        [: 8.; 8.; 8.; 8. :];
        [: 6.; 6.; 6.; 6. :];
        [: 3.; 7.; 3.; 7. :];
        [: 2.; 9.; 2.; 9. :];
        [: 5.; 3.; 5.; 3. :];
        [: 8.; 1.; 8.; 1. :];
        [: 6.; 2.; 6.; 2. :];
        [: 7.; 3.6; 7.; 3.6 :];
      :]
    in
    let b = [: 0.1; 0.2; 0.2; 0.4; 0.4; 0.6; 0.3; 0.7; 0.5; 0.5 :] in
    neg
      (sum 10 ~f:(stack_ fun i ->
           constant #1.
           /. (coeff b i
              +. (sum 4 ~f:(stack_ fun j ->
                      square (get x j -. coeff (Iarray.get c i) j))
                 [@nontail])))
      [@nontail])

  let[@zero_alloc] sphere6_rescaled (x @ local) =
    ((sum 6 ~f:(stack_ fun i -> powi (constant #2.) (i + 1) *. square (get x i))
     [@nontail])
    -. constant #1745.)
    /. constant #899.

  let[@zero_alloc] rosenbrock4_rescaled (x @ local) =
    let at i = (constant #15. *. get x i) -. constant #5. in
    ((sum 3 ~f:(stack_ fun i ->
          (constant #100. *. square (at (i + 1) -. square (at i)))
          +. square (constant #1. -. at i))
     [@nontail])
    -. constant #382700.)
    /. constant #375500.

  let[@zero_alloc] branin_rescaled (x @ local) =
    (branin_at
       ((constant #15. *. get x 0) -. constant #5.)
       (constant #15. *. get x 1)
    -. constant #54.81)
    /. constant #51.95

  let[@zero_alloc] branin_modified (x @ local) =
    branin x +. (constant #5. *. get x 0)

  let[@zero_alloc] forrester_low_fidelity (x @ local) =
    (constant #0.5 *. forrester x)
    +. (constant #10. *. (get x 0 -. constant #0.5))
    +. constant #5.

  let[@zero_alloc] goldstein_price_rescaled (x @ local) =
    (log
       (goldstein_price_at
          ((constant #4. *. get x 0) -. constant #2.)
          ((constant #4. *. get x 1) -. constant #2.))
    -. constant #8.693)
    /. constant #2.427

  let[@zero_alloc] hartmann6_rescaled (x @ local) =
    (hartmann x -. constant #2.58) /. constant #1.94
end
[@@kind k = (float64, vec128)]

module Scalar = Make [@kind float64] (Numeric.Scalar)
module Simd = Make [@kind vec128] (Numeric.Neon)
