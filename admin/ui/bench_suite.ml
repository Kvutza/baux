(* admin/ui/bench_suite.ml
   Pure OCaml implementation of all 56 baux synthetic optimization benchmark
   functions (delegated to Bench_formulas, matching ocaml/lib/problem.ml and
   ocaml/lib/formulas.ml) plus a deterministic in-browser multi-region TuRBO +
   Epistemic Nearest-Neighbor (ENN) optimizer supporting Euclidean, Conformal,
   Hyperbolic, and Wasserstein local-scale geometries across canonical, D=16,
   and D=64 regimes.
*)

type family = Bench_formulas.family =
  | Multimodal
  | Bowl
  | Plate
  | Valley
  | Ridge
  | Other

type dim_rule = Bench_formulas.dim_rule =
  | Fixed of int
  | At_least of int
  | Multiple_of of int

type spec = Bench_formulas.spec = {
  id : string;
  label : string;
  family : family;
  rule : dim_rule;
  source : string;
  domain : int -> (float * float) array;
  eval : float array -> float;
}

let family_slug = Bench_formulas.family_slug
let family_title = Bench_formulas.family_title
let default_dim = Bench_formulas.default_dim
let is_scalable = Bench_formulas.is_scalable
let valid_dim = Bench_formulas.valid_dim

module Formulas = Bench_formulas.Formulas

let uniform = Bench_formulas.uniform
let fixed = Bench_formulas.fixed
let mk = Bench_formulas.mk
let catalogue = Bench_formulas.catalogue
let find_spec = Bench_formulas.find_spec

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
  ucb : float;
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
  if Float.is_nan raw then 1.0e290
  else if raw > 1.0e290 then 1.0e290
  else if raw < -1.0e290 then -1.0e290
  else raw

(* Asymptotic log formula for |z| > 1e8 avoids intermediate z*z overflow in float64. *)
let safe_asinh z =
  let az = Float.abs z in
  if az > 1.0e8 then Float.copy_sign (Float.log (2.0 *. az)) z
  else Float.asinh z

(* Rank-preserving asinh compression keeps extreme polynomials (Perm, Rosenbrock,
   Goldstein-Price) numerically well-conditioned during ENN distance weighting. *)
let compress_y y center scale =
  let z = (y -. center) /. scale in
  safe_asinh z

let decompress_y z center scale =
  let zc = clamp (-700.0) 700.0 z in
  let raw = center +. (scale *. Float.sinh zc) in
  if Float.is_nan raw then 1.0e290
  else clamp (-1.0e290) 1.0e290 raw

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

let predict_enn ?(beta = 2.0) (m : enn_model) (u : float array) : posterior =
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
  let ucb = -.decompress_y (mu_z -. (beta *. sig_z)) m.y_center m.y_scale in
  let valid_neighbors =
    best_idx |> Array.to_list |> List.filter (fun idx -> idx >= 0)
    |> Array.of_list
  in
  { mean; sigma; ucb; neighbors = valid_neighbors }

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
  ignore warm_steps;
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
        let best_ucb = ref Float.neg_infinity in
        for p = 0 to n_pool - 1 do
          let c =
            if p = 0 then !best_c
            else if p = n_pool - 1 && Array.length s.regions > 1 then
              Array.init s.dim (fun _ -> Rng.uniform01 s.rng)
            else
              raasp_candidate ~full:(p mod 3 = 1) s.rng reg.center_u reg.length
          in
          let post = predict_enn ~beta:2.0 m c in
          if post.ucb > !best_ucb then begin
            best_ucb := post.ucb;
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
