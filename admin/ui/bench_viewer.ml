(* admin/ui/bench_viewer.ml
   Interactive 56-Function OxCaml Benchmark Suite Explorer (WebGL1 + Canvas2D).
   Pure DOM construction (zero innerHTML / outerHTML / eval_string).
*)

[@@@warning "-32-27-33"]

open Js_of_ocaml
open Dom_html

let jf = Js.float
let tf = Js.to_float
let js = Js.string

class type wheel_ev = object
  inherit mouseEvent
  method deltaY : Js.number_t Js.readonly_prop
end

class type ctx2d_ext = object
  inherit canvasRenderingContext2D
  method setLineDash : Js.number_t Js.js_array Js.t -> unit Js.meth
  method imageSmoothingEnabled : bool Js.t Js.prop
end

type field_mode =
  | Field_sigma
  | Field_mean
  | Field_lcb
  | Field_true

type dim_regime =
  | Reg_canonical
  | Reg_d16
  | Reg_d64

(* --- String & Formatting Utilities --- *)

let string_contains haystack needle =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  if n_len = 0 then true
  else if n_len > h_len then false
  else
    let rec loop i j =
      if j = n_len then true
      else if i + j >= h_len then false
      else if haystack.[i + j] = needle.[j] then loop i (j + 1)
      else loop (i + 1) 0
    in
    loop 0 0

let fmt_num v =
  if Float.is_nan v || not (Float.is_finite v) then "n/a"
  else
    let av = Float.abs v in
    if av < 1.0e-12 then "0.000"
    else if av >= 1.0e5 || av < 1.0e-3 then Printf.sprintf "%.2e" v
    else if av >= 100.0 then Printf.sprintf "%.1f" v
    else if av >= 10.0 then Printf.sprintf "%.2f" v
    else Printf.sprintf "%.4f" v

let fmt_axis v =
  if Float.is_nan v || not (Float.is_finite v) then "0"
  else
    let av = Float.abs v in
    if av < 1.0e-12 then "0"
    else if av >= 1.0e4 || av < 1.0e-2 then Printf.sprintf "%.1e" v
    else if Float.abs (v -. Float.round v) < 1.0e-6 then
      Printf.sprintf "%.0f" v
    else if av >= 10.0 then Printf.sprintf "%.1f" v
    else Printf.sprintf "%.2f" v

let option_label_of_spec (sp : Bench_suite.spec) =
  let d0 = Bench_suite.default_dim sp.rule in
  if Bench_suite.is_scalable sp.rule then
    let tag =
      match sp.rule with
      | Bench_suite.At_least 1 -> "1D\u{2013}D"
      | Bench_suite.At_least k -> Printf.sprintf "%dD\u{2013}D" k
      | Bench_suite.Multiple_of k -> Printf.sprintf "%dkD" k
      | Bench_suite.Fixed d -> Printf.sprintf "%dD" d
    in
    Printf.sprintf "%s (%dD \u{00B7} [%s] scalable)" sp.label d0 tag
  else
    let tag_check = Printf.sprintf "(%dD)" d0 in
    if string_contains sp.label tag_check then sp.label
    else Printf.sprintf "%s (%dD)" sp.label d0

let rule_summary (sp : Bench_suite.spec) active_dim =
  match sp.rule with
  | Bench_suite.Fixed d -> Printf.sprintf "Fixed %dD" d
  | Bench_suite.At_least k ->
      Printf.sprintf "At_least %dD (active D=%d)" k active_dim
  | Bench_suite.Multiple_of k ->
      Printf.sprintf "Multiple_of %dD (active D=%d)" k active_dim

let domain_summary (bounds : (float * float) array) =
  let d = Array.length bounds in
  if d = 0 then "[]"
  else
    let lo0, hi0 = bounds.(0) in
    let all_same = ref true in
    for j = 1 to d - 1 do
      let lo, hi = bounds.(j) in
      if Float.abs (lo -. lo0) > 1.0e-9 || Float.abs (hi -. hi0) > 1.0e-9 then
        all_same := false
    done;
    if !all_same then
      if d = 1 then Printf.sprintf "[%s, %s]" (fmt_axis lo0) (fmt_axis hi0)
      else Printf.sprintf "[%s, %s]^%d" (fmt_axis lo0) (fmt_axis hi0) d
    else if d = 2 then
      let lo1, hi1 = bounds.(1) in
      Printf.sprintf "[%s, %s] \u{00D7} [%s, %s]"
        (fmt_axis lo0) (fmt_axis hi0) (fmt_axis lo1) (fmt_axis hi1)
    else
      Printf.sprintf "[%s, %s] \u{00D7} \u{2026} (%dD)"
        (fmt_axis lo0) (fmt_axis hi0) d

(* --- Pure DOM Helpers --- *)

let is_dark_theme () =
  match
    Js.Opt.to_option (document##.documentElement##getAttribute (js "data-theme"))
  with
  | Some s -> Js.to_string s = "dark"
  | None -> false

let set_text (el : #Dom.node Js.t) txt =
  el##.textContent := Js.some (js txt)

let clear_children (el : #Dom.node Js.t) =
  while Js.Opt.test el##.firstChild do
    match Js.Opt.to_option el##.firstChild with
    | Some child -> ignore (el##removeChild child)
    | None -> ()
  done

let toggle_class (el : #element Js.t) cls active =
  if active then el##.classList##add (js cls)
  else el##.classList##remove (js cls)

let mk_div ?cls () =
  let d = createDiv document in
  (match cls with Some c -> d##.className := js c | None -> ());
  d

let mk_span ?cls ?txt () =
  let s = createSpan document in
  (match cls with Some c -> s##.className := js c | None -> ());
  (match txt with Some t -> set_text s t | None -> ());
  s

let mk_strong txt =
  let st = createStrong document in
  set_text st txt;
  st

let mk_btn ?cls txt =
  let b = createButton ~_type:(js "button") document in
  (match cls with Some c -> b##.className := js c | None -> ());
  set_text b txt;
  b

let add_click el f =
  let h =
    handler (fun _ev ->
        f ();
        Js._true)
  in
  ignore (addEventListener el Event.click h Js._false)

let set_line_dash (ctx : canvasRenderingContext2D Js.t) (arr : float array) =
  let ext : ctx2d_ext Js.t = Js.Unsafe.coerce ctx in
  ext##setLineDash (Js.array (Array.map jf arr))

let set_smoothing (ctx : canvasRenderingContext2D Js.t) flag =
  let ext : ctx2d_ext Js.t = Js.Unsafe.coerce ctx in
  ext##.imageSmoothingEnabled := Js.bool flag

let device_dpr () =
  let raw = tf window##.devicePixelRatio in
  if Float.is_nan raw || raw <= 0.0 then 1.0 else Float.min 2.0 raw

let sync_canvas_dpi (canvas : canvasElement Js.t) : float * float * float =
  let dpr = device_dpr () in
  let rect = canvas##getBoundingClientRect in
  let cw = Float.max 120.0 (tf rect##.width) in
  let ch = Float.max 72.0 (tf rect##.height) in
  let pw = Int.max 1 (int_of_float (Float.round (cw *. dpr))) in
  let ph = Int.max 1 (int_of_float (Float.round (ch *. dpr))) in
  if canvas##.width <> pw then canvas##.width := pw;
  if canvas##.height <> ph then canvas##.height := ph;
  (cw, ch, dpr)

(* --- Scientific Palettes --- *)

let clamp01 t = if t < 0.0 then 0.0 else if t > 1.0 then 1.0 else t

let lerp a b u = a +. ((b -. a) *. u)

let ramp_stops (stops : (float * float * float * float) array) t =
  let u = clamp01 t in
  let n = Array.length stops in
  let idx = ref (n - 2) in
  for i = 0 to n - 2 do
    let s0, _, _, _ = stops.(i) in
    let s1, _, _, _ = stops.(i + 1) in
    if u >= s0 && u <= s1 then idx := i
  done;
  let s0, r0, g0, b0 = stops.(!idx) in
  let s1, r1, g1, b1 = stops.(!idx + 1) in
  let loc = if s1 > s0 then (u -. s0) /. (s1 -. s0) else 0.0 in
  (lerp r0 r1 loc, lerp g0 g1 loc, lerp b0 b1 loc)

let surface_stops =
  [|
    (0.00, 0.094, 0.110, 0.345);
    (0.22, 0.110, 0.361, 0.541);
    (0.45, 0.071, 0.580, 0.518);
    (0.68, 0.463, 0.761, 0.345);
    (0.86, 0.918, 0.737, 0.180);
    (1.00, 0.992, 0.910, 0.408);
  |]

let sigma_stops =
  [|
    (0.00, 0.075, 0.106, 0.227);
    (0.28, 0.263, 0.220, 0.557);
    (0.55, 0.616, 0.180, 0.541);
    (0.80, 0.902, 0.384, 0.243);
    (1.00, 0.988, 0.827, 0.302);
  |]

let surface_rgb t = ramp_stops surface_stops t
let sigma_rgb t = ramp_stops sigma_stops t

let rgb_bytes (r, g, b) =
  let to_u8 c = Int.max 0 (Int.min 255 (int_of_float (Float.round (c *. 255.0)))) in
  (to_u8 r, to_u8 g, to_u8 b)

let region_rgba r_idx is_active is_dark =
  let alpha = if is_active then 0.96 else 0.62 in
  match (r_idx mod 4, is_dark) with
  | 0, true -> (0.220, 0.741, 0.973, alpha)
  | 0, false -> (0.008, 0.518, 0.780, alpha)
  | 1, true -> (0.984, 0.749, 0.141, alpha)
  | 1, false -> (0.851, 0.467, 0.024, alpha)
  | 2, true -> (0.753, 0.518, 0.988, alpha)
  | 2, false -> (0.486, 0.227, 0.929, alpha)
  | _, true -> (0.204, 0.827, 0.600, alpha)
  | _, false -> (0.020, 0.588, 0.412, alpha)

let region_hex r_idx is_dark =
  match (r_idx mod 4, is_dark) with
  | 0, true -> "#38bdf8"
  | 0, false -> "#0284c7"
  | 1, true -> "#fbbf24"
  | 1, false -> "#d97706"
  | 2, true -> "#c084fc"
  | 2, false -> "#7c3aed"
  | _, true -> "#34d399"
  | _, false -> "#059669"

(* --- Asinh-Stabilized Field Scaling --- *)

type norm_scale = {
  use_asinh : bool;
  anchor : float;
  scale : float;
  z_min : float;
  z_max : float;
  raw_min : float;
  raw_max : float;
}

let build_norm_scale (vals : float array) : norm_scale =
  let n = Array.length vals in
  if n = 0 then
    {
      use_asinh = false;
      anchor = 0.0;
      scale = 1.0;
      z_min = 0.0;
      z_max = 1.0;
      raw_min = 0.0;
      raw_max = 1.0;
    }
  else
    let sorted = Array.copy vals in
    Array.sort Float.compare sorted;
    let raw_min = sorted.(0) in
    let raw_max = sorted.(n - 1) in
    let q10 = sorted.(Int.max 0 (n / 10)) in
    let q85 = sorted.(Int.min (n - 1) ((85 * n) / 100)) in
    let lin_span = raw_max -. raw_min in
    let core_span = Float.max 1.0e-9 (q85 -. q10) in
    let use_asinh = lin_span > 7.5 *. core_span && lin_span > 1.0e-6 in
    let anchor = q10 in
    let scale = Float.max 1.0e-9 (0.35 *. core_span) in
    let xf v =
      if use_asinh then Float.asinh ((v -. anchor) /. scale) else v
    in
    let z0 = xf raw_min in
    let z1 = xf raw_max in
    let z_max = if z1 -. z0 < 1.0e-12 then z0 +. 1.0 else z1 in
    { use_asinh; anchor; scale; z_min = z0; z_max; raw_min; raw_max }

let eval_norm (ns : norm_scale) v =
  let z =
    if ns.use_asinh then Float.asinh ((v -. ns.anchor) /. ns.scale) else v
  in
  clamp01 ((z -. ns.z_min) /. Float.max 1.0e-12 (ns.z_max -. ns.z_min))

(* --- 44 x 44 3D Surface Cache --- *)

let grid_3d = 44
let y_base = -0.62

type surface_cache = {
  s_key : string;
  s_dim : int;
  s_heights : float array array;
  s_tnorm : float array array;
  s_normals : (float * float * float) array array;
  s_scale : norm_scale;
}

let slice_signature (s : Bench_suite.session) =
  let best_u = s.xs_u.(s.best_idx) in
  if s.dim <= 2 then Printf.sprintf "%s:%d" s.spec.id s.dim
  else
    let buf = Buffer.create 64 in
    Buffer.add_string buf (Printf.sprintf "%s:%d:" s.spec.id s.dim);
    for c = 2 to Int.min (s.dim - 1) 12 do
      Buffer.add_string buf (Printf.sprintf "%.3f," best_u.(c))
    done;
    if s.dim > 13 then
      Buffer.add_string buf (Printf.sprintf "#%d" s.best_idx);
    Buffer.contents buf

let world_xz_of_u dim u1 u2 =
  let xw = (u1 -. 0.5) *. 2.0 in
  let zw =
    if dim = 1 then (u2 -. 0.5) *. 1.24 else (u2 -. 0.5) *. 2.0
  in
  (xw, zw)

let compute_surface (s : Bench_suite.session) : surface_cache =
  let n = grid_3d in
  let best_u = s.xs_u.(s.best_idx) in
  let u_ probe1 probe2 =
    let arr = Array.copy best_u in
    arr.(0) <- probe1;
    if s.dim >= 2 then arr.(1) <- probe2;
    arr
  in
  let raw = Array.make_matrix n n 0.0 in
  let flat = Array.make (n * n) 0.0 in
  let k = ref 0 in
  for i = 0 to n - 1 do
    let u1 = float_of_int i /. float_of_int (n - 1) in
    if s.dim = 1 then begin
      let y1 = Bench_suite.safe_eval s.spec s.bounds (u_ u1 0.5) in
      for j = 0 to n - 1 do
        raw.(i).(j) <- y1;
        flat.(!k) <- y1;
        incr k
      done
    end
    else
      for j = 0 to n - 1 do
        let u2 = float_of_int j /. float_of_int (n - 1) in
        let y = Bench_suite.safe_eval s.spec s.bounds (u_ u1 u2) in
        raw.(i).(j) <- y;
        flat.(!k) <- y;
        incr k
      done
  done;
  let sc = build_norm_scale flat in
  let heights = Array.make_matrix n n 0.0 in
  let tnorm = Array.make_matrix n n 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      let t = eval_norm sc raw.(i).(j) in
      tnorm.(i).(j) <- t;
      heights.(i).(j) <- -0.50 +. (1.08 *. t)
    done
  done;
  let normals = Array.make_matrix n n (0.0, 1.0, 0.0) in
  let step_x = 2.0 /. float_of_int (n - 1) in
  let step_z = (if s.dim = 1 then 1.24 else 2.0) /. float_of_int (n - 1) in
  for i = 0 to n - 1 do
    let i0 = Int.max 0 (i - 1) in
    let i1 = Int.min (n - 1) (i + 1) in
    let dx = float_of_int (i1 - i0) *. step_x in
    for j = 0 to n - 1 do
      let j0 = Int.max 0 (j - 1) in
      let j1 = Int.min (n - 1) (j + 1) in
      let dz = float_of_int (j1 - j0) *. step_z in
      let dh_dx = (heights.(i1).(j) -. heights.(i0).(j)) /. dx in
      let dh_dz = (heights.(i).(j1) -. heights.(i).(j0)) /. dz in
      let nx = -.dh_dx in
      let ny = 1.0 in
      let nz = -.dh_dz in
      let inv = 1.0 /. Float.sqrt ((nx *. nx) +. (ny *. ny) +. (nz *. nz)) in
      normals.(i).(j) <- (nx *. inv, ny *. inv, nz *. inv)
    done
  done;
  {
    s_key = slice_signature s;
    s_dim = s.dim;
    s_heights = heights;
    s_tnorm = tnorm;
    s_normals = normals;
    s_scale = sc;
  }

let bilerp_surface (sc : surface_cache) u1 u2 =
  let n = grid_3d in
  let fx = clamp01 u1 *. float_of_int (n - 1) in
  let fy = clamp01 u2 *. float_of_int (n - 1) in
  let i0 = Int.min (n - 2) (int_of_float fx) in
  let j0 = Int.min (n - 2) (int_of_float fy) in
  let tx = fx -. float_of_int i0 in
  let ty = fy -. float_of_int j0 in
  let h00 = sc.s_heights.(i0).(j0) in
  let h10 = sc.s_heights.(i0 + 1).(j0) in
  let h01 = sc.s_heights.(i0).(j0 + 1) in
  let h11 = sc.s_heights.(i0 + 1).(j0 + 1) in
  lerp (lerp h00 h10 tx) (lerp h01 h11 tx) ty

(* --- 4x4 Matrix Math for WebGL1 & Fallback --- *)

let mat4_mul (a : float array) (b : float array) =
  let out = Array.make 16 0.0 in
  for col = 0 to 3 do
    for row = 0 to 3 do
      let sum = ref 0.0 in
      for k = 0 to 3 do
        sum := !sum +. (a.((k * 4) + row) *. b.((col * 4) + k))
      done;
      out.((col * 4) + row) <- !sum
    done
  done;
  out

let mat4_perspective fovy aspect near far =
  let f = 1.0 /. Float.tan (0.5 *. fovy) in
  let nf = 1.0 /. (near -. far) in
  let out = Array.make 16 0.0 in
  out.(0) <- f /. Float.max 0.2 aspect;
  out.(5) <- f;
  out.(10) <- (far +. near) *. nf;
  out.(11) <- -1.0;
  out.(14) <- 2.0 *. far *. near *. nf;
  out

let mat4_look_at (ex, ey, ez) (cx, cy, cz) (ux, uy, uz) =
  let zx = ex -. cx and zy = ey -. cy and zz = ez -. cz in
  let lz = Float.max 1.0e-9 (Float.sqrt ((zx *. zx) +. (zy *. zy) +. (zz *. zz))) in
  let zx = zx /. lz and zy = zy /. lz and zz = zz /. lz in
  let xx = (uy *. zz) -. (uz *. zy) in
  let xy = (uz *. zx) -. (ux *. zz) in
  let xz = (ux *. zy) -. (uy *. zx) in
  let lx = Float.max 1.0e-9 (Float.sqrt ((xx *. xx) +. (xy *. xy) +. (xz *. xz))) in
  let xx = xx /. lx and xy = xy /. lx and xz = xz /. lx in
  let yx = (zy *. xz) -. (zz *. xy) in
  let yy = (zz *. xx) -. (zx *. xz) in
  let yz = (zx *. xy) -. (zy *. xx) in
  let out = Array.make 16 0.0 in
  out.(0) <- xx;
  out.(4) <- xy;
  out.(8) <- xz;
  out.(12) <- -.((xx *. ex) +. (xy *. ey) +. (xz *. ez));
  out.(1) <- yx;
  out.(5) <- yy;
  out.(9) <- yz;
  out.(13) <- -.((yx *. ex) +. (yy *. ey) +. (yz *. ez));
  out.(2) <- zx;
  out.(6) <- zy;
  out.(10) <- zz;
  out.(14) <- -.((zx *. ex) +. (zy *. ey) +. (zz *. ez));
  out.(15) <- 1.0;
  out

let f32_of_array (arr : float array) (len : int) : Typed_array.float32Array Js.t =
  let ta = new%js Typed_array.float32Array len in
  for i = 0 to len - 1 do
    Typed_array.set ta i (jf arr.(i))
  done;
  ta

(* --- Growable Float Buffer Builder --- *)

type fbuf = {
  mutable data : float array;
  mutable len : int;
}

let fbuf_create cap = { data = Array.make (Int.max 64 cap) 0.0; len = 0 }
let fbuf_clear b = b.len <- 0

let fbuf_ensure b extra =
  let needed = b.len + extra in
  let cur = Array.length b.data in
  if needed > cur then begin
    let next_cap = Int.max needed (cur * 2) in
    let next = Array.make next_cap 0.0 in
    Array.blit b.data 0 next 0 b.len;
    b.data <- next
  end

let fbuf_push3 b x y z =
  fbuf_ensure b 3;
  let i = b.len in
  b.data.(i) <- x;
  b.data.(i + 1) <- y;
  b.data.(i + 2) <- z;
  b.len <- i + 3

let fbuf_push4 b r g bl a =
  fbuf_ensure b 4;
  let i = b.len in
  b.data.(i) <- r;
  b.data.(i + 1) <- g;
  b.data.(i + 2) <- bl;
  b.data.(i + 3) <- a;
  b.len <- i + 4

let push_line pos_b col_b (x0, y0, z0) (x1, y1, z1) (r, g, b, a) =
  fbuf_push3 pos_b x0 y0 z0;
  fbuf_push3 pos_b x1 y1 z1;
  fbuf_push4 col_b r g b a;
  fbuf_push4 col_b r g b a

let push_octahedron pos_b norm_b col_b (cx, cy, cz) rad (r, g, b, a) =
  let top = (cx, cy +. (rad *. 1.25), cz) in
  let bot = (cx, cy -. (rad *. 1.25), cz) in
  let ring =
    [|
      (cx +. rad, cy, cz);
      (cx, cy, cz +. rad);
      (cx -. rad, cy, cz);
      (cx, cy, cz -. rad);
    |]
  in
  let add_tri (ax, ay, az) (bx, by, bz) (dx, dy, dz) =
    let ux = bx -. ax and uy = by -. ay and uz = bz -. az in
    let vx = dx -. ax and vy = dy -. ay and vz = dz -. az in
    let nx = (uy *. vz) -. (uz *. vy) in
    let ny = (uz *. vx) -. (ux *. vz) in
    let nz = (ux *. vy) -. (uy *. vx) in
    let inv =
      1.0 /. Float.max 1.0e-9 (Float.sqrt ((nx *. nx) +. (ny *. ny) +. (nz *. nz)))
    in
    let nx = nx *. inv and ny = ny *. inv and nz = nz *. inv in
    fbuf_push3 pos_b ax ay az;
    fbuf_push3 pos_b bx by bz;
    fbuf_push3 pos_b dx dy dz;
    for _ = 1 to 3 do
      fbuf_push3 norm_b nx ny nz;
      fbuf_push4 col_b r g b a
    done
  in
  for k = 0 to 3 do
    let p0 = ring.(k) in
    let p1 = ring.((k + 1) mod 4) in
    add_tri top p1 p0;
    add_tri bot p0 p1
  done

(* --- WebGL1 Context State --- *)

type gl_bundle = {
  gl : WebGL.renderingContext Js.t;
  prog : WebGL.program Js.t;
  a_pos : int;
  a_norm : int;
  a_col : int;
  u_mvp : [ `mat4 ] WebGL.uniformLocation Js.t;
  u_cam : [ `vec3 ] WebGL.uniformLocation Js.t;
  u_unlit : Js.number_t WebGL.uniformLocation Js.t;
  surf_pos_buf : WebGL.buffer Js.t;
  surf_norm_buf : WebGL.buffer Js.t;
  surf_col_buf : WebGL.buffer Js.t;
  mutable surf_uploaded_key : string;
  mutable surf_vert_count : int;
  dyn_pos_buf : WebGL.buffer Js.t;
  dyn_norm_buf : WebGL.buffer Js.t;
  dyn_col_buf : WebGL.buffer Js.t;
  line_pos_buf : WebGL.buffer Js.t;
  line_col_buf : WebGL.buffer Js.t;
}

let vs_source =
  "attribute vec3 aPos;\n\
   attribute vec3 aNorm;\n\
   attribute vec4 aCol;\n\
   uniform mat4 uMVP;\n\
   uniform vec3 uCamPos;\n\
   uniform float uUnlit;\n\
   varying vec4 vCol;\n\
   varying vec3 vNorm;\n\
   varying vec3 vView;\n\
   varying float vUnlit;\n\
   void main() {\n\
     vCol = aCol;\n\
     vNorm = aNorm;\n\
     vView = uCamPos - aPos;\n\
     vUnlit = uUnlit;\n\
     gl_Position = uMVP * vec4(aPos, 1.0);\n\
   }"

let fs_source =
  "precision mediump float;\n\
   varying vec4 vCol;\n\
   varying vec3 vNorm;\n\
   varying vec3 vView;\n\
   varying float vUnlit;\n\
   void main() {\n\
     if (vUnlit > 0.5) {\n\
       gl_FragColor = vCol;\n\
       return;\n\
     }\n\
     vec3 n = normalize(vNorm);\n\
     vec3 v = normalize(vView);\n\
     if (dot(n, v) < 0.0) n = -n;\n\
     vec3 l1 = normalize(vec3(0.52, 0.86, 0.44));\n\
     vec3 l2 = normalize(vec3(-0.62, 0.38, -0.58));\n\
     float d1 = max(dot(n, l1), 0.0);\n\
     float d2 = max(dot(n, l2), 0.0) * 0.32;\n\
     vec3 h = normalize(l1 + v);\n\
     float spec = pow(max(dot(n, h), 0.0), 28.0) * 0.20;\n\
     float hemi = 0.35 + 0.10 * n.y;\n\
     vec3 rgb = vCol.rgb * (hemi + 0.63 * d1 + d2) + vec3(spec);\n\
     gl_FragColor = vec4(min(vec3(1.0), rgb), vCol.a);\n\
   }"

let init_webgl (canvas : canvasElement Js.t) : gl_bundle option =
  try
    let attrs = WebGL.defaultContextAttributes in
    attrs##.antialias := Js._true;
    attrs##.depth := Js._true;
    attrs##.alpha := Js._false;
    match Js.Opt.to_option (WebGL.getContextWithAttributes canvas attrs) with
    | None -> None
    | Some gl ->
        let compile stype src =
          let sh = gl##createShader stype in
          gl##shaderSource sh (js src);
          gl##compileShader sh;
          if Js.to_bool (gl##getShaderParameter sh gl##._COMPILE_STATUS_) then
            Some sh
          else None
        in
        (match
           ( compile gl##._VERTEX_SHADER_ vs_source,
             compile gl##._FRAGMENT_SHADER_ fs_source )
         with
        | Some vs, Some fs ->
            let prog = gl##createProgram in
            gl##attachShader prog vs;
            gl##attachShader prog fs;
            gl##bindAttribLocation prog 0 (js "aPos");
            gl##bindAttribLocation prog 1 (js "aNorm");
            gl##bindAttribLocation prog 2 (js "aCol");
            gl##linkProgram prog;
            if not (Js.to_bool (gl##getProgramParameter prog gl##._LINK_STATUS_))
            then None
            else
              Some
                {
                  gl;
                  prog;
                  a_pos = gl##getAttribLocation prog (js "aPos");
                  a_norm = gl##getAttribLocation prog (js "aNorm");
                  a_col = gl##getAttribLocation prog (js "aCol");
                  u_mvp = gl##getUniformLocation prog (js "uMVP");
                  u_cam = gl##getUniformLocation prog (js "uCamPos");
                  u_unlit = gl##getUniformLocation prog (js "uUnlit");
                  surf_pos_buf = gl##createBuffer;
                  surf_norm_buf = gl##createBuffer;
                  surf_col_buf = gl##createBuffer;
                  surf_uploaded_key = "";
                  surf_vert_count = 0;
                  dyn_pos_buf = gl##createBuffer;
                  dyn_norm_buf = gl##createBuffer;
                  dyn_col_buf = gl##createBuffer;
                  line_pos_buf = gl##createBuffer;
                  line_col_buf = gl##createBuffer;
                }
        | _ -> None)
  with _ -> None

(* --- 48 x 48 2D Field Offscreen Heatmap Cache --- *)

let grid_2d = 48
let tex_2d = 144

type map2d_cache = {
  m_key : string;
  m_field : field_mode;
  m_tnorm : float array array;
  m_scale : norm_scale;
  m_canvas : canvasElement Js.t;
}

let build_map2d_cache (s : Bench_suite.session) (field : field_mode)
    (off_canvas : canvasElement Js.t) : map2d_cache =
  let n = grid_2d in
  let best_u = s.xs_u.(s.best_idx) in
  let m = Bench_suite.ensure_model s in
  let raw = Array.make_matrix n n 0.0 in
  let flat = Array.make (n * n) 0.0 in
  let probe = Array.copy best_u in
  let k = ref 0 in
  for i = 0 to n - 1 do
    let u1 = float_of_int i /. float_of_int (n - 1) in
    probe.(0) <- u1;
    for j = 0 to n - 1 do
      let u2 = float_of_int j /. float_of_int (n - 1) in
      if s.dim >= 2 then probe.(1) <- u2;
      let v =
        match field with
        | Field_true -> Bench_suite.safe_eval s.spec s.bounds probe
        | Field_sigma -> (Bench_suite.predict_enn m probe).sigma
        | Field_mean -> (Bench_suite.predict_enn m probe).mean
        | Field_lcb -> (Bench_suite.predict_enn m probe).lcb
      in
      raw.(i).(j) <- v;
      flat.(!k) <- v;
      incr k
    done
  done;
  let sc = build_norm_scale flat in
  let tnorm = Array.make_matrix n n 0.0 in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      tnorm.(i).(j) <- eval_norm sc raw.(i).(j)
    done
  done;
  if off_canvas##.width <> tex_2d then off_canvas##.width := tex_2d;
  if off_canvas##.height <> tex_2d then off_canvas##.height := tex_2d;
  let ctx = off_canvas##getContext _2d_ in
  let img = ctx##createImageData tex_2d tex_2d in
  let pix = img##.data in
  for py = 0 to tex_2d - 1 do
    let u2 = 1.0 -. (float_of_int py /. float_of_int (tex_2d - 1)) in
    let fy = clamp01 u2 *. float_of_int (n - 1) in
    let j0 = Int.min (n - 2) (int_of_float fy) in
    let ty = fy -. float_of_int j0 in
    for px = 0 to tex_2d - 1 do
      let u1 = float_of_int px /. float_of_int (tex_2d - 1) in
      let fx = clamp01 u1 *. float_of_int (n - 1) in
      let i0 = Int.min (n - 2) (int_of_float fx) in
      let tx = fx -. float_of_int i0 in
      let t =
        lerp
          (lerp tnorm.(i0).(j0) tnorm.(i0 + 1).(j0) tx)
          (lerp tnorm.(i0).(j0 + 1) tnorm.(i0 + 1).(j0 + 1) tx)
          ty
      in
      let rgb =
        match field with
        | Field_sigma -> sigma_rgb t
        | Field_mean | Field_lcb | Field_true -> surface_rgb t
      in
      let r8, g8, b8 = rgb_bytes rgb in
      let base = (py * tex_2d + px) * 4 in
      pixel_set pix base r8;
      pixel_set pix (base + 1) g8;
      pixel_set pix (base + 2) b8;
      pixel_set pix (base + 3) 255
    done
  done;
  ctx##putImageData img (jf 0.0) (jf 0.0);
  let key =
    Printf.sprintf "%s:%d:%s:%d:%d:%d" s.spec.id s.dim
      (Bench_suite.key_of_arm s.arm)
      (Array.length s.ys) s.seed
      (match field with
      | Field_sigma -> 0
      | Field_mean -> 1
      | Field_lcb -> 2
      | Field_true -> 3)
  in
  { m_key = key; m_field = field; m_tnorm = tnorm; m_scale = sc; m_canvas = off_canvas }

(* Draw marching-squares iso-contours over a [box_x, box_y, box_w, box_h] plot *)
let draw_contours (ctx : canvasRenderingContext2D Js.t)
    (tnorm : float array array) box_x box_y box_w box_h =
  let n = Array.length tnorm in
  let levels = [| 0.14; 0.28; 0.42; 0.56; 0.70; 0.84 |] in
  let to_xy u1 u2 =
    (box_x +. (u1 *. box_w), box_y +. ((1.0 -. u2) *. box_h))
  in
  let inv = 1.0 /. float_of_int (n - 1) in
  ctx##save;
  ctx##.strokeStyle := js "rgba(255, 255, 255, 0.26)";
  ctx##.lineWidth := jf 0.85;
  Array.iter
    (fun lv ->
      ctx##beginPath;
      for i = 0 to n - 2 do
        let u1_0 = float_of_int i *. inv in
        let u1_1 = float_of_int (i + 1) *. inv in
        for j = 0 to n - 2 do
          let u2_0 = float_of_int j *. inv in
          let u2_1 = float_of_int (j + 1) *. inv in
          let v00 = tnorm.(i).(j) in
          let v10 = tnorm.(i + 1).(j) in
          let v11 = tnorm.(i + 1).(j + 1) in
          let v01 = tnorm.(i).(j + 1) in
          let idx =
            (if v00 >= lv then 1 else 0)
            lor (if v10 >= lv then 2 else 0)
            lor (if v11 >= lv then 4 else 0)
            lor if v01 >= lv then 8 else 0
          in
          if idx <> 0 && idx <> 15 then begin
            let interp va vb =
              let d = vb -. va in
              if Float.abs d < 1.0e-9 then 0.5 else clamp01 ((lv -. va) /. d)
            in
            let e_bot () = to_xy (lerp u1_0 u1_1 (interp v00 v10)) u2_0 in
            let e_right () = to_xy u1_1 (lerp u2_0 u2_1 (interp v10 v11)) in
            let e_top () = to_xy (lerp u1_0 u1_1 (interp v01 v11)) u2_1 in
            let e_left () = to_xy u1_0 (lerp u2_0 u2_1 (interp v00 v01)) in
            let seg (ax, ay) (bx, by) =
              ctx##moveTo (jf ax) (jf ay);
              ctx##lineTo (jf bx) (jf by)
            in
            match idx with
            | 1 | 14 -> seg (e_left ()) (e_bot ())
            | 2 | 13 -> seg (e_bot ()) (e_right ())
            | 3 | 12 -> seg (e_left ()) (e_right ())
            | 4 | 11 -> seg (e_right ()) (e_top ())
            | 5 ->
                seg (e_left ()) (e_top ());
                seg (e_bot ()) (e_right ())
            | 6 | 9 -> seg (e_bot ()) (e_top ())
            | 7 | 8 -> seg (e_left ()) (e_top ())
            | 10 ->
                seg (e_left ()) (e_bot ());
                seg (e_right ()) (e_top ())
            | _ -> ()
          end
        done
      done;
      ctx##stroke)
    levels;
  ctx##restore

(* --- Viewer Mount & Interactive Controller --- *)

let max_auto_evals = 240

let mount_viewer (root : element Js.t) =
  if
    Js.Opt.to_option (root##getAttribute (js "data-oxb-mounted"))
    |> Option.map Js.to_string = Some "1"
  then ()
  else begin
    root##setAttribute (js "data-oxb-mounted") (js "1");
    clear_children root;

    let default_spec =
      match Bench_suite.find_spec "eggholder" with
      | Some sp -> sp
      | None -> Bench_suite.catalogue.(0)
    in

    (* Mutable UI State *)
    let cur_family : Bench_suite.family option ref = ref None in
    let cur_spec = ref default_spec in
    let cur_regime = ref Reg_canonical in
    let cur_arm = ref (Bench_suite.Enn_multi Bench_suite.Hyperbolic) in
    let cur_field = ref Field_sigma in
    let cur_seed = ref 1 in
    let cur_session =
      ref
        (Bench_suite.init_session ~seed:1 ~warm_steps:24 ~spec:default_spec
           ~dim:(Bench_suite.default_dim default_spec.rule)
           ~arm:!cur_arm ())
    in

    (* Camera & Hover State *)
    let cam_yaw = ref (-0.68) in
    let cam_pitch = ref 0.56 in
    let cam_dist = ref 3.10 in
    let hover_u : (float * float) option ref = ref None in
    let play_timer : interval_id option ref = ref None in

    (* Caches & Scratch Buffers *)
    let surf_cache : surface_cache option ref = ref None in
    let map_cache : map2d_cache option ref = ref None in
    let off_canvas = createCanvas document in
    let surf_pos_b = fbuf_create 36000 in
    let surf_norm_b = fbuf_create 36000 in
    let surf_col_b = fbuf_create 48000 in
    let dyn_pos_b = fbuf_create 18000 in
    let dyn_norm_b = fbuf_create 18000 in
    let dyn_col_b = fbuf_create 24000 in
    let line_pos_b = fbuf_create 8000 in
    let line_col_b = fbuf_create 10000 in

    (* --- Build Outer Layout --- *)
    let outer = mk_div ~cls:"tcv-cad-outer oxb-cad-outer" () in
    let sidebar = document##createElement (js "aside") in
    sidebar##.className := js "tcv-side-controls oxb-side-controls";
    sidebar##setAttribute (js "aria-label") (js "Benchmark Suite Controls");

    let title_bar = mk_div ~cls:"tcv-hud-badge tcv-ctrl-title oxb-ctrl-title" () in
    Dom.appendChild title_bar (mk_strong "BENCHMARK SUITE (56)");
    Dom.appendChild sidebar title_bar;

    (* 1. Family Filter Pills + Function Select *)
    let sec_fn = mk_div ~cls:"tcv-ctrl-sec oxb-ctrl-fn" () in
    let hdr_fn = mk_div ~cls:"tcv-ctrl-hdr" () in
    let fn_count_span = mk_span ~txt:"56 functions" () in
    Dom.appendChild hdr_fn (mk_span ~txt:"Function Family" ());
    Dom.appendChild hdr_fn fn_count_span;
    Dom.appendChild sec_fn hdr_fn;

    let fam_pills_wrap = mk_div ~cls:"oxb-fam-pills" () in
    let families : (string * Bench_suite.family option) array =
      [|
        ("All (56)", None);
        ("Multimodal", Some Bench_suite.Multimodal);
        ("Bowl", Some Bench_suite.Bowl);
        ("Plate", Some Bench_suite.Plate);
        ("Valley", Some Bench_suite.Valley);
        ("Ridge", Some Bench_suite.Ridge);
        ("Standard BO", Some Bench_suite.Other);
      |]
    in
    let fam_btns =
      Array.map
        (fun (lbl, fam_opt) ->
          let b = mk_btn ~cls:"oxb-pill-btn" lbl in
          Dom.appendChild fam_pills_wrap b;
          (b, fam_opt))
        families
    in
    Dom.appendChild sec_fn fam_pills_wrap;

    let fn_select = createSelect document in
    fn_select##.className := js "tcv-select oxb-fn-select";
    fn_select##setAttribute (js "aria-label") (js "Benchmark Function");
    Dom.appendChild sec_fn fn_select;
    Dom.appendChild sidebar sec_fn;

    (* 2. Dimension Regime Selector *)
    let sec_dim = mk_div ~cls:"tcv-ctrl-sec oxb-ctrl-dim" () in
    let hdr_dim = mk_div ~cls:"tcv-ctrl-hdr" () in
    let dim_active_lbl = mk_strong "D = 2" in
    Dom.appendChild hdr_dim (mk_span ~txt:"Dimension Regime" ());
    Dom.appendChild hdr_dim dim_active_lbl;
    Dom.appendChild sec_dim hdr_dim;

    let dim_row = mk_div ~cls:"tcv-preset-row oxb-dim-row" () in
    let btn_dim_can = mk_btn ~cls:"tcv-preset-btn active" "Canonical" in
    let btn_dim_16 = mk_btn ~cls:"tcv-preset-btn" "D = 16" in
    let btn_dim_64 = mk_btn ~cls:"tcv-preset-btn" "D = 64" in
    Dom.appendChild dim_row btn_dim_can;
    Dom.appendChild dim_row btn_dim_16;
    Dom.appendChild dim_row btn_dim_64;
    Dom.appendChild sec_dim dim_row;
    Dom.appendChild sidebar sec_dim;

    (* 3. Optimizer / Geometry Arm Selector *)
    let sec_arm = mk_div ~cls:"tcv-ctrl-sec oxb-ctrl-arm" () in
    let hdr_arm = mk_div ~cls:"tcv-ctrl-hdr" () in
    Dom.appendChild hdr_arm (mk_span ~txt:"Optimizer / Geometry" ());
    Dom.appendChild hdr_arm (mk_span ~txt:"Regions" ());
    Dom.appendChild sec_arm hdr_arm;

    let arm_group = mk_div ~cls:"tcv-pol-group oxb-arm-group" () in
    let arm_specs : (string * string * string * Bench_suite.arm) array =
      [|
        ( "ENN \u{00B7} Hyperbolic",
          "R=4",
          "#b45309",
          Bench_suite.Enn_multi Bench_suite.Hyperbolic );
        ( "ENN \u{00B7} Wasserstein",
          "R=4",
          "#1d4ed8",
          Bench_suite.Enn_multi Bench_suite.Wasserstein );
        ( "ENN \u{00B7} Conformal",
          "R=4",
          "#7e22ce",
          Bench_suite.Enn_multi Bench_suite.Conformal );
        ( "ENN \u{00B7} Euclidean",
          "R=4",
          "#0f766e",
          Bench_suite.Enn_multi Bench_suite.Euclidean );
        ("TuRBO-0 (R=4)", "Rand", "#475569", Bench_suite.Turbo_zero);
        ("ENN Single (R=1)", "R=1", "#dc2626", Bench_suite.Enn_single);
      |]
    in
    let arm_btns =
      Array.map
        (fun (lbl, tag_r, dot_col, arm_val) ->
          let b = createButton ~_type:(js "button") document in
          b##.className := js "tcv-pol-btn oxb-arm-btn";
          let left_sp = mk_span ~cls:"oxb-arm-lbl" () in
          let dot = mk_span ~cls:"tcv-pol-dot" () in
          dot##.style##.backgroundColor := js dot_col;
          Dom.appendChild left_sp dot;
          Dom.appendChild left_sp (document##createTextNode (js lbl));
          let right_sp = mk_span ~cls:"oxb-arm-tag" ~txt:tag_r () in
          Dom.appendChild b left_sp;
          Dom.appendChild b right_sp;
          Dom.appendChild arm_group b;
          (b, arm_val))
        arm_specs
    in
    Dom.appendChild sec_arm arm_group;
    Dom.appendChild sidebar sec_arm;

    (* 4. 2D Surrogate Field Selector Pills *)
    let sec_field = mk_div ~cls:"tcv-ctrl-sec oxb-ctrl-field" () in
    let hdr_field = mk_div ~cls:"tcv-ctrl-hdr" () in
    Dom.appendChild hdr_field (mk_span ~txt:"2D Surrogate Field" ());
    Dom.appendChild hdr_field (mk_span ~txt:"Pane (b)" ());
    Dom.appendChild sec_field hdr_field;

    let field_row = mk_div ~cls:"oxb-field-row" () in
    let field_specs =
      [|
        ("Epistemic \u{03C3}_e", Field_sigma);
        ("Posterior \u{03BC}", Field_mean);
        ("LCB \u{03B1}(x)", Field_lcb);
        ("True f(x)", Field_true);
      |]
    in
    let field_btns =
      Array.map
        (fun (lbl, fm) ->
          let b = mk_btn ~cls:"tcv-preset-btn oxb-field-btn" lbl in
          Dom.appendChild field_row b;
          (b, fm))
        field_specs
    in
    Dom.appendChild sec_field field_row;
    Dom.appendChild sidebar sec_field;

    (* 5. Live Step & Playback Controls *)
    let sec_play = mk_div ~cls:"tcv-ctrl-sec oxb-ctrl-play" () in
    let hdr_play = mk_div ~cls:"tcv-ctrl-hdr" () in
    let budget_lbl = mk_span ~txt:"Budget 240" () in
    Dom.appendChild hdr_play (mk_span ~txt:"Live Trajectory" ());
    Dom.appendChild hdr_play budget_lbl;
    Dom.appendChild sec_play hdr_play;

    let status_badge = mk_div ~cls:"tcv-shot-status is-done oxb-status-badge" () in
    Dom.appendChild sec_play status_badge;

    let play_grid = mk_div ~cls:"oxb-play-grid" () in
    let btn_play = mk_btn ~cls:"tcv-pol-btn oxb-play-btn" "\u{25B6} Play" in
    let btn_step1 = mk_btn ~cls:"tcv-preset-btn oxb-step-btn" "+1 Step" in
    let btn_step10 = mk_btn ~cls:"tcv-preset-btn oxb-step-btn" "+10 Steps" in
    let btn_reset = mk_btn ~cls:"tcv-preset-btn oxb-step-btn" "\u{21BA} Reset" in
    let btn_seed = mk_btn ~cls:"tcv-preset-btn oxb-step-btn" "Seed 1" in
    Dom.appendChild play_grid btn_play;
    Dom.appendChild play_grid btn_step1;
    Dom.appendChild play_grid btn_step10;
    Dom.appendChild play_grid btn_reset;
    Dom.appendChild play_grid btn_seed;
    Dom.appendChild sec_play play_grid;
    Dom.appendChild sidebar sec_play;

    (* --- Main Dual-Pane Shell --- *)
    let shell = mk_div ~cls:"tcv-cad-shell oxb-cad-shell" () in
    let grid = mk_div ~cls:"tcv-grid oxb-grid" () in

    (* Pane (a): 3D Objective Surface *)
    let pane_a = mk_div ~cls:"tcv-pane oxb-pane-a" () in
    let badge_a = mk_div ~cls:"tcv-hud-badge" () in
    let badge_a_left = mk_span () in
    Dom.appendChild badge_a_left (mk_strong "(a) ");
    let badge_a_title = mk_span ~txt:"3D Objective Surface f(x\u{2081}, x\u{2082})" () in
    Dom.appendChild badge_a_left badge_a_title;
    let badge_a_hint =
      mk_span ~cls:"tcv-touch-hint" ~txt:"Drag to orbit \u{00B7} Scroll to zoom" ()
    in
    Dom.appendChild badge_a badge_a_left;
    Dom.appendChild badge_a badge_a_hint;

    let wrap_a = mk_div ~cls:"tcv-canvas-wrap oxb-webgl-wrap" () in
    let gl_canvas = createCanvas document in
    gl_canvas##.className := js "oxb-webgl-canvas";
    Dom.appendChild wrap_a gl_canvas;
    Dom.appendChild pane_a badge_a;
    Dom.appendChild pane_a wrap_a;

    (* Pane (b): Live ENN Surrogate & Trust Regions *)
    let pane_b = mk_div ~cls:"tcv-pane oxb-pane-b" () in
    let badge_b = mk_div ~cls:"tcv-hud-badge" () in
    let badge_b_left = mk_span () in
    Dom.appendChild badge_b_left (mk_strong "(b) ");
    let badge_b_title = mk_span ~txt:"Live ENN Surrogate & Trust Regions" () in
    Dom.appendChild badge_b_left badge_b_title;
    let badge_b_sub = mk_span ~cls:"tcv-touch-hint oxb-pane-b-sub" ~txt:"\u{03C3}_e(x)" () in
    Dom.appendChild badge_b badge_b_left;
    Dom.appendChild badge_b badge_b_sub;

    let wrap_b = mk_div ~cls:"tcv-canvas-wrap oxb-map-wrap" () in
    let map_canvas = createCanvas document in
    map_canvas##.className := js "oxb-map-canvas";
    Dom.appendChild wrap_b map_canvas;
    Dom.appendChild pane_b badge_b;
    Dom.appendChild pane_b wrap_b;

    Dom.appendChild grid pane_a;
    Dom.appendChild grid pane_b;
    Dom.appendChild shell grid;

    (* Synchronized Convergence Strip *)
    let trace_stage = mk_div ~cls:"oxb-trace-stage" () in
    let trace_hdr = mk_div ~cls:"oxb-trace-hdr" () in
    let trace_hdr_left = mk_span () in
    Dom.appendChild trace_hdr_left (mk_strong "(c) ");
    Dom.appendChild trace_hdr_left
      (mk_span ~txt:"Anytime Best-So-Far Objective min_{t \u{2264} n} f(x_t)" ());
    let trace_hdr_right = mk_span ~cls:"oxb-trace-sub" () in
    Dom.appendChild trace_hdr trace_hdr_left;
    Dom.appendChild trace_hdr trace_hdr_right;
    let trace_wrap = mk_div ~cls:"oxb-trace-wrap" () in
    let trace_canvas = createCanvas document in
    trace_canvas##.className := js "oxb-trace-canvas";
    Dom.appendChild trace_wrap trace_canvas;
    Dom.appendChild trace_stage trace_hdr;
    Dom.appendChild trace_stage trace_wrap;
    Dom.appendChild shell trace_stage;

    (* Metadata & Hover Telemetry Deck *)
    let meta_strip = mk_div ~cls:"tcv-legend-strip oxb-meta-strip" () in
    let meta_chips = mk_div ~cls:"tcv-legend-items oxb-meta-chips" () in
    let chip_domain = mk_span ~cls:"tcv-leg-chip oxb-meta-chip" () in
    let chip_rule = mk_span ~cls:"tcv-leg-chip oxb-meta-chip" () in
    let chip_family = mk_span ~cls:"tcv-leg-chip oxb-meta-chip" () in
    let chip_regions = mk_span ~cls:"tcv-leg-chip oxb-meta-chip" () in
    let src_link = createA document in
    src_link##.className := js "oxb-src-link";
    src_link##.target := js "_blank";
    src_link##.rel := js "noopener noreferrer";
    set_text src_link "SFU Ref \u{2197}";
    Dom.appendChild meta_chips chip_domain;
    Dom.appendChild meta_chips chip_rule;
    Dom.appendChild meta_chips chip_family;
    Dom.appendChild meta_chips chip_regions;
    Dom.appendChild meta_chips src_link;
    Dom.appendChild meta_strip meta_chips;
    Dom.appendChild shell meta_strip;

    let hover_bar = mk_div ~cls:"baux-scrub-bar oxb-hover-bar" () in
    Dom.appendChild shell hover_bar;

    Dom.appendChild outer sidebar;
    Dom.appendChild outer shell;
    Dom.appendChild root outer;

    (* Initialize WebGL1 (or fallback cleanly to 2D isometric painter) *)
    let gl_opt = init_webgl gl_canvas in

    (* --- Helper: Resolve Active Dimension --- *)
    let resolve_dim (sp : Bench_suite.spec) reg =
      if not (Bench_suite.is_scalable sp.rule) then
        Bench_suite.default_dim sp.rule
      else
        match reg with
        | Reg_canonical -> Bench_suite.default_dim sp.rule
        | Reg_d16 ->
            if Bench_suite.valid_dim sp.rule 16 then 16
            else Bench_suite.default_dim sp.rule
        | Reg_d64 ->
            if Bench_suite.valid_dim sp.rule 64 then 64
            else Bench_suite.default_dim sp.rule
    in

    (* --- Stop / Start Playback Timer --- *)
    let stop_playback () =
      match !play_timer with
      | None -> ()
      | Some tid ->
          window##clearInterval tid;
          play_timer := None;
          set_text btn_play "\u{25B6} Play";
          toggle_class btn_play "active" false
    in

    (* --- Ensure Cached 3D Surface --- *)
    let get_surface (s : Bench_suite.session) =
      let sig_key = slice_signature s in
      match !surf_cache with
      | Some sc when sc.s_key = sig_key -> sc
      | _ ->
          let sc = compute_surface s in
          surf_cache := Some sc;
          sc
    in

    (* --- Render Pane (a): WebGL1 or 2D Isometric Fallback --- *)
    let render_pane_a () =
      let s = !cur_session in
      let sc = get_surface s in
      let is_dark = is_dark_theme () in
      let cw, ch, _dpr = sync_canvas_dpi gl_canvas in
      let pw = gl_canvas##.width in
      let ph = gl_canvas##.height in
      let aspect = cw /. Float.max 1.0 ch in
      let cy = Float.cos !cam_yaw and sy = Float.sin !cam_yaw in
      let cp = Float.cos !cam_pitch and sp = Float.sin !cam_pitch in
      let eye =
        ( !cam_dist *. cp *. sy,
          !cam_dist *. sp,
          !cam_dist *. cp *. cy )
      in
      let view = mat4_look_at eye (0.0, -0.05, 0.0) (0.0, 1.0, 0.0) in
      let proj = mat4_perspective (40.0 *. Float.pi /. 180.0) aspect 0.15 20.0 in
      let mvp = mat4_mul proj view in

      match gl_opt with
      | Some b ->
          let gl = b.gl in
          gl##viewport 0 0 pw ph;
          if is_dark then gl##clearColor (jf 0.051) (jf 0.067) (jf 0.090) (jf 1.0)
          else gl##clearColor (jf 0.980) (jf 0.984) (jf 0.992) (jf 1.0);
          gl##clear (gl##._COLOR_BUFFER_BIT_ lor gl##._DEPTH_BUFFER_BIT_);
          gl##enable gl##._DEPTH_TEST_;
          gl##depthFunc gl##._LEQUAL;
          gl##enable gl##._BLEND;
          gl##blendFunc gl##._SRC_ALPHA_ gl##._ONE_MINUS_SRC_ALPHA_;
          gl##useProgram b.prog;

          let mvp_ta = f32_of_array mvp 16 in
          gl##uniformMatrix4fv_typed b.u_mvp Js._false mvp_ta;
          let ex, ey, ez = eye in
          gl##uniform3f b.u_cam (jf ex) (jf ey) (jf ez);

          (* Upload static-per-slice surface triangles only when key changes *)
          if b.surf_uploaded_key <> sc.s_key then begin
            fbuf_clear surf_pos_b;
            fbuf_clear surf_norm_b;
            fbuf_clear surf_col_b;
            let n = grid_3d in
            let inv = 1.0 /. float_of_int (n - 1) in
            let push_grid_vert gi gj =
              let u1 = float_of_int gi *. inv in
              let u2 = float_of_int gj *. inv in
              let xw, zw = world_xz_of_u s.dim u1 u2 in
              let yw = sc.s_heights.(gi).(gj) in
              let nx, ny, nz = sc.s_normals.(gi).(gj) in
              let r, g, bl = surface_rgb sc.s_tnorm.(gi).(gj) in
              fbuf_push3 surf_pos_b xw yw zw;
              fbuf_push3 surf_norm_b nx ny nz;
              fbuf_push4 surf_col_b r g bl 0.97
            in
            for i = 0 to n - 2 do
              for j = 0 to n - 2 do
                push_grid_vert i j;
                push_grid_vert (i + 1) j;
                push_grid_vert (i + 1) (j + 1);
                push_grid_vert i j;
                push_grid_vert (i + 1) (j + 1);
                push_grid_vert i (j + 1)
              done
            done;
            b.surf_vert_count <- surf_pos_b.len / 3;
            b.surf_uploaded_key <- sc.s_key;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_pos_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array surf_pos_b.data surf_pos_b.len)
              gl##._STATIC_DRAW_;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_norm_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array surf_norm_b.data surf_norm_b.len)
              gl##._STATIC_DRAW_;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_col_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array surf_col_b.data surf_col_b.len)
              gl##._STATIC_DRAW_
          end;

          (* Draw lit 3D objective surface *)
          gl##uniform1f b.u_unlit (jf 0.0);
          if b.a_pos >= 0 then begin
            gl##enableVertexAttribArray b.a_pos;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_pos_buf;
            gl##vertexAttribPointer b.a_pos 3 gl##._FLOAT Js._false 0 0
          end;
          if b.a_norm >= 0 then begin
            gl##enableVertexAttribArray b.a_norm;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_norm_buf;
            gl##vertexAttribPointer b.a_norm 3 gl##._FLOAT Js._false 0 0
          end;
          if b.a_col >= 0 then begin
            gl##enableVertexAttribArray b.a_col;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.surf_col_buf;
            gl##vertexAttribPointer b.a_col 4 gl##._FLOAT Js._false 0 0
          end;
          gl##drawArrays gl##._TRIANGLES 0 b.surf_vert_count;

          (* Build dynamic octahedron markers: samples + incumbent x_star *)
          fbuf_clear dyn_pos_b;
          fbuf_clear dyn_norm_b;
          fbuf_clear dyn_col_b;
          let n_obs = Array.length s.ys in
          for idx = 0 to n_obs - 1 do
            let u = s.xs_u.(idx) in
            let u1 = u.(0) in
            let u2 = if s.dim >= 2 then u.(1) else 0.5 in
            let xw, zw = world_xz_of_u s.dim u1 u2 in
            let y_surf = bilerp_surface sc u1 u2 in
            if idx = s.best_idx then
              push_octahedron dyn_pos_b dyn_norm_b dyn_col_b
                (xw, y_surf +. 0.032, zw) 0.044 (0.98, 0.22, 0.42, 1.0)
            else if idx = n_obs - 1 && idx >= s.n_init then
              push_octahedron dyn_pos_b dyn_norm_b dyn_col_b
                (xw, y_surf +. 0.020, zw) 0.026 (0.14, 0.86, 0.98, 0.96)
            else
              let pt_col =
                if idx < s.n_init then
                  if is_dark then (0.72, 0.78, 0.86, 0.82)
                  else (0.28, 0.34, 0.44, 0.85)
                else if is_dark then (0.96, 0.98, 1.0, 0.92)
                else (0.09, 0.11, 0.16, 0.92)
              in
              push_octahedron dyn_pos_b dyn_norm_b dyn_col_b
                (xw, y_surf +. 0.015, zw) 0.017 pt_col
          done;
          let dyn_verts = dyn_pos_b.len / 3 in
          if dyn_verts > 0 then begin
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.dyn_pos_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array dyn_pos_b.data dyn_pos_b.len)
              gl##._DYNAMIC_DRAW_;
            gl##vertexAttribPointer b.a_pos 3 gl##._FLOAT Js._false 0 0;

            gl##bindBuffer gl##._ARRAY_BUFFER_ b.dyn_norm_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array dyn_norm_b.data dyn_norm_b.len)
              gl##._DYNAMIC_DRAW_;
            if b.a_norm >= 0 then
              gl##vertexAttribPointer b.a_norm 3 gl##._FLOAT Js._false 0 0;

            gl##bindBuffer gl##._ARRAY_BUFFER_ b.dyn_col_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array dyn_col_b.data dyn_col_b.len)
              gl##._DYNAMIC_DRAW_;
            if b.a_col >= 0 then
              gl##vertexAttribPointer b.a_col 4 gl##._FLOAT Js._false 0 0;

            gl##drawArrays gl##._TRIANGLES 0 dyn_verts
          end;

          (* Build unlit line overlays: pedestal, surface isolines, 3D trust boxes, incumbent stem *)
          fbuf_clear line_pos_b;
          fbuf_clear line_col_b;
          let ped_col =
            if is_dark then (0.30, 0.36, 0.45, 0.55)
            else (0.55, 0.60, 0.68, 0.55)
          in
          let grid_col =
            if is_dark then (0.20, 0.25, 0.32, 0.38)
            else (0.76, 0.80, 0.86, 0.48)
          in
          let z_ext = if s.dim = 1 then 0.62 else 1.0 in
          for g = 0 to 4 do
            let u = float_of_int g *. 0.25 in
            let xw = (u -. 0.5) *. 2.0 in
            let zw = (u -. 0.5) *. (2.0 *. z_ext) in
            let c = if g = 0 || g = 4 then ped_col else grid_col in
            push_line line_pos_b line_col_b (xw, y_base, -.z_ext)
              (xw, y_base, z_ext) c;
            push_line line_pos_b line_col_b (-1.0, y_base, zw)
              (1.0, y_base, zw) c
          done;

          (* Subtle surface grid lines every 4 vertices *)
          let surf_wire =
            if is_dark then (1.0, 1.0, 1.0, 0.13) else (0.05, 0.08, 0.14, 0.13)
          in
          let n = grid_3d in
          let inv = 1.0 /. float_of_int (n - 1) in
          let lift = 0.0035 in
          for i = 0 to n - 1 do
            if i mod 4 = 0 || i = n - 1 then
              for j = 0 to n - 2 do
                let x0, z0 =
                  world_xz_of_u s.dim (float_of_int i *. inv)
                    (float_of_int j *. inv)
                in
                let x1, z1 =
                  world_xz_of_u s.dim (float_of_int i *. inv)
                    (float_of_int (j + 1) *. inv)
                in
                push_line line_pos_b line_col_b
                  (x0, sc.s_heights.(i).(j) +. lift, z0)
                  (x1, sc.s_heights.(i).(j + 1) +. lift, z1)
                  surf_wire
              done
          done;
          for j = 0 to n - 1 do
            if j mod 4 = 0 || j = n - 1 then
              for i = 0 to n - 2 do
                let x0, z0 =
                  world_xz_of_u s.dim (float_of_int i *. inv)
                    (float_of_int j *. inv)
                in
                let x1, z1 =
                  world_xz_of_u s.dim
                    (float_of_int (i + 1) *. inv)
                    (float_of_int j *. inv)
                in
                push_line line_pos_b line_col_b
                  (x0, sc.s_heights.(i).(j) +. lift, z0)
                  (x1, sc.s_heights.(i + 1).(j) +. lift, z1)
                  surf_wire
              done
          done;

          (* 3D Trust-Region Bounding Boxes projected onto pedestal + surface *)
          Array.iteri
            (fun r_idx (reg : Bench_suite.region) ->
              let half = 0.5 *. reg.length in
              let u1_lo = clamp01 (reg.center_u.(0) -. half) in
              let u1_hi = clamp01 (reg.center_u.(0) +. half) in
              let u2_lo, u2_hi =
                if s.dim >= 2 then
                  ( clamp01 (reg.center_u.(1) -. half),
                    clamp01 (reg.center_u.(1) +. half) )
                else (0.12, 0.88)
              in
              let rc = region_rgba r_idx (r_idx = s.last_region) is_dark in
              let x0, z0 = world_xz_of_u s.dim u1_lo u2_lo in
              let x1, z1 = world_xz_of_u s.dim u1_hi u2_hi in
              (* Base footprint *)
              push_line line_pos_b line_col_b (x0, y_base +. 0.002, z0)
                (x1, y_base +. 0.002, z0) rc;
              push_line line_pos_b line_col_b (x1, y_base +. 0.002, z0)
                (x1, y_base +. 0.002, z1) rc;
              push_line line_pos_b line_col_b (x1, y_base +. 0.002, z1)
                (x0, y_base +. 0.002, z1) rc;
              push_line line_pos_b line_col_b (x0, y_base +. 0.002, z1)
                (x0, y_base +. 0.002, z0) rc;
              (* Vertical corner posts to surface *)
              let corners =
                [| (u1_lo, u2_lo); (u1_hi, u2_lo); (u1_hi, u2_hi); (u1_lo, u2_hi) |]
              in
              Array.iter
                (fun (cu1, cu2) ->
                  let cx, cz = world_xz_of_u s.dim cu1 cu2 in
                  let cy_s = bilerp_surface sc cu1 cu2 +. 0.012 in
                  push_line line_pos_b line_col_b (cx, y_base, cz) (cx, cy_s, cz)
                    rc)
                corners;
              (* Surface-hugging perimeter *)
              let segs = 8 in
              let trace_edge (a1, a2) (b1, b2) =
                for k = 0 to segs - 1 do
                  let t0 = float_of_int k /. float_of_int segs in
                  let t1 = float_of_int (k + 1) /. float_of_int segs in
                  let p1_0 = lerp a1 b1 t0 and p2_0 = lerp a2 b2 t0 in
                  let p1_1 = lerp a1 b1 t1 and p2_1 = lerp a2 b2 t1 in
                  let wx0, wz0 = world_xz_of_u s.dim p1_0 p2_0 in
                  let wx1, wz1 = world_xz_of_u s.dim p1_1 p2_1 in
                  let wy0 = bilerp_surface sc p1_0 p2_0 +. 0.012 in
                  let wy1 = bilerp_surface sc p1_1 p2_1 +. 0.012 in
                  push_line line_pos_b line_col_b (wx0, wy0, wz0)
                    (wx1, wy1, wz1) rc
                done
              in
              trace_edge (u1_lo, u2_lo) (u1_hi, u2_lo);
              trace_edge (u1_hi, u2_lo) (u1_hi, u2_hi);
              trace_edge (u1_hi, u2_hi) (u1_lo, u2_hi);
              trace_edge (u1_lo, u2_hi) (u1_lo, u2_lo))
            s.regions;

          (* Global incumbent x* vertical drop stem + base crosshair *)
          let bu = s.xs_u.(s.best_idx) in
          let bu1 = bu.(0) in
          let bu2 = if s.dim >= 2 then bu.(1) else 0.5 in
          let bx, bz = world_xz_of_u s.dim bu1 bu2 in
          let by_s = bilerp_surface sc bu1 bu2 in
          let stem_col = (0.99, 0.24, 0.44, 0.98) in
          push_line line_pos_b line_col_b (bx, y_base, bz)
            (bx, by_s +. 0.18, bz) stem_col;
          push_line line_pos_b line_col_b (bx -. 0.07, y_base +. 0.003, bz)
            (bx +. 0.07, y_base +. 0.003, bz) stem_col;
          push_line line_pos_b line_col_b (bx, y_base +. 0.003, bz -. 0.07)
            (bx, y_base +. 0.003, bz +. 0.07) stem_col;

          (* Coupled 2D -> 3D cursor probe needle when hovering Pane (b) *)
          (match !hover_u with
          | None -> ()
          | Some (hu1, hu2) ->
              let hx, hz =
                world_xz_of_u s.dim hu1 (if s.dim = 1 then 0.5 else hu2)
              in
              let hy = bilerp_surface sc hu1 (if s.dim = 1 then 0.5 else hu2) in
              let hcol =
                if is_dark then (0.35, 0.92, 1.0, 0.92)
                else (0.03, 0.55, 0.82, 0.92)
              in
              push_line line_pos_b line_col_b (hx, y_base, hz)
                (hx, hy +. 0.14, hz) hcol);

          let line_verts = line_pos_b.len / 3 in
          if line_verts > 0 then begin
            gl##uniform1f b.u_unlit (jf 1.0);
            if b.a_norm >= 0 then begin
              gl##disableVertexAttribArray b.a_norm;
              gl##vertexAttrib3f b.a_norm (jf 0.0) (jf 1.0) (jf 0.0)
            end;
            gl##bindBuffer gl##._ARRAY_BUFFER_ b.line_pos_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array line_pos_b.data line_pos_b.len)
              gl##._DYNAMIC_DRAW_;
            gl##vertexAttribPointer b.a_pos 3 gl##._FLOAT Js._false 0 0;

            gl##bindBuffer gl##._ARRAY_BUFFER_ b.line_col_buf;
            gl##bufferData gl##._ARRAY_BUFFER_
              (f32_of_array line_col_b.data line_col_b.len)
              gl##._DYNAMIC_DRAW_;
            if b.a_col >= 0 then
              gl##vertexAttribPointer b.a_col 4 gl##._FLOAT Js._false 0 0;

            gl##drawArrays gl##._LINES 0 line_verts
          end
      | None ->
          (* Graceful 2D Isometric Painter Fallback when WebGL1 is unavailable *)
          let ctx = gl_canvas##getContext _2d_ in
          let dpr = device_dpr () in
          ctx##save;
          ctx##setTransform (jf dpr) (jf 0.0) (jf 0.0) (jf dpr) (jf 0.0) (jf 0.0);
          ctx##.fillStyle := js (if is_dark then "#0d1117" else "#f8fafc");
          ctx##fillRect (jf 0.0) (jf 0.0) (jf cw) (jf ch);
          let project (x, y, z) =
            let cx =
              (mvp.(0) *. x) +. (mvp.(4) *. y) +. (mvp.(8) *. z) +. mvp.(12)
            in
            let cy =
              (mvp.(1) *. x) +. (mvp.(5) *. y) +. (mvp.(9) *. z) +. mvp.(13)
            in
            let cz =
              (mvp.(2) *. x) +. (mvp.(6) *. y) +. (mvp.(10) *. z) +. mvp.(14)
            in
            let cw_p =
              Float.max 0.05
                ((mvp.(3) *. x) +. (mvp.(7) *. y) +. (mvp.(11) *. z) +. mvp.(15))
            in
            let sx = (0.5 +. (0.5 *. (cx /. cw_p))) *. cw in
            let sy = (0.5 -. (0.5 *. (cy /. cw_p))) *. ch in
            (sx, sy, cz /. cw_p)
          in
          let step_ iso = 2 in
          let n = grid_3d in
          let inv = 1.0 /. float_of_int (n - 1) in
          let quads = ref [] in
          let i = ref 0 in
          while !i < n - 1 do
            let i1 = Int.min (n - 1) (!i + step_ 0) in
            let j = ref 0 in
            while !j < n - 1 do
              let j1 = Int.min (n - 1) (!j + step_ 0) in
              let pt gi gj =
                let xw, zw =
                  world_xz_of_u s.dim (float_of_int gi *. inv)
                    (float_of_int gj *. inv)
                in
                project (xw, sc.s_heights.(gi).(gj), zw)
              in
              let p00 = pt !i !j in
              let p10 = pt i1 !j in
              let p11 = pt i1 j1 in
              let p01 = pt !i j1 in
              let _, _, z00 = p00 and _, _, z11 = p11 in
              let depth = 0.5 *. (z00 +. z11) in
              let t_avg =
                0.25
                *. (sc.s_tnorm.(!i).(!j) +. sc.s_tnorm.(i1).(!j)
                  +. sc.s_tnorm.(i1).(j1) +. sc.s_tnorm.(!i).(j1))
              in
              quads := (depth, p00, p10, p11, p01, t_avg) :: !quads;
              j := j1
            done;
            i := i1
          done;
          let sorted_q =
            List.sort (fun (da, _, _, _, _, _) (db, _, _, _, _, _) ->
                Float.compare db da)
              !quads
          in
          List.iter
            (fun (_, (x0, y0, _), (x1, y1, _), (x2, y2, _), (x3, y3, _), t) ->
              let r8, g8, b8 = rgb_bytes (surface_rgb t) in
              ctx##.fillStyle := js (Printf.sprintf "rgb(%d,%d,%d)" r8 g8 b8);
              ctx##.strokeStyle := js "rgba(255,255,255,0.12)";
              ctx##.lineWidth := jf 0.5;
              ctx##beginPath;
              ctx##moveTo (jf x0) (jf y0);
              ctx##lineTo (jf x1) (jf y1);
              ctx##lineTo (jf x2) (jf y2);
              ctx##lineTo (jf x3) (jf y3);
              ctx##closePath;
              ctx##fill;
              ctx##stroke)
            sorted_q;
          (* Incumbent marker in fallback *)
          let bu = s.xs_u.(s.best_idx) in
          let bu1 = bu.(0) in
          let bu2 = if s.dim >= 2 then bu.(1) else 0.5 in
          let bx, bz = world_xz_of_u s.dim bu1 bu2 in
          let by_s = bilerp_surface sc bu1 bu2 in
          let sx0, sy0, _ = project (bx, y_base, bz) in
          let sx1, sy1, _ = project (bx, by_s +. 0.14, bz) in
          ctx##.strokeStyle := js "#f43f5e";
          ctx##.lineWidth := jf 2.0;
          ctx##beginPath;
          ctx##moveTo (jf sx0) (jf sy0);
          ctx##lineTo (jf sx1) (jf sy1);
          ctx##stroke;
          ctx##.fillStyle := js "#f43f5e";
          ctx##beginPath;
          ctx##arc (jf sx1) (jf sy1) (jf 4.5) (jf 0.0) (jf (2.0 *. Float.pi))
            Js._false;
          ctx##fill;
          ctx##restore
    in

    (* --- Store active 2D plot rectangle for mouse/touch hit-testing --- *)
    let map_box = ref (44.0, 18.0, 220.0, 220.0) in

    (* --- Render Pane (b): 2D Surrogate Heatmap/Contours OR 1D GP/ENN Curve --- *)
    let render_pane_b () =
      let s = !cur_session in
      let is_dark = is_dark_theme () in
      let cw, ch, dpr = sync_canvas_dpi map_canvas in
      let ctx = map_canvas##getContext _2d_ in
      ctx##save;
      ctx##setTransform (jf dpr) (jf 0.0) (jf 0.0) (jf dpr) (jf 0.0) (jf 0.0);
      ctx##.fillStyle := js (if is_dark then "#0d1117" else "#ffffff");
      ctx##fillRect (jf 0.0) (jf 0.0) (jf cw) (jf ch);

      let fg_ink = if is_dark then "#e6edf3" else "#111111" in
      let muted_ink = if is_dark then "#8b949e" else "#475569" in
      let border_ink = if is_dark then "#30363d" else "#334155" in
      let m = Bench_suite.ensure_model s in

      if s.dim = 1 then begin
        (* 1D Curve Mode (Forrester, Gramacy & Lee) *)
        let pad_l = 46.0 and pad_r = 16.0 and pad_t = 18.0 and pad_b = 34.0 in
        let bw = Float.max 80.0 (cw -. pad_l -. pad_r) in
        let bh = Float.max 80.0 (ch -. pad_t -. pad_b) in
        map_box := (pad_l, pad_t, bw, bh);

        let n_pts = 160 in
        let f_arr = Array.make n_pts 0.0 in
        let mu_arr = Array.make n_pts 0.0 in
        let lo_arr = Array.make n_pts 0.0 in
        let hi_arr = Array.make n_pts 0.0 in
        let lcb_arr = Array.make n_pts 0.0 in
        let y_min = ref Float.infinity in
        let y_max = ref Float.neg_infinity in
        let note v =
          if Float.is_finite v then begin
            if v < !y_min then y_min := v;
            if v > !y_max then y_max := v
          end
        in
        for i = 0 to n_pts - 1 do
          let u1 = float_of_int i /. float_of_int (n_pts - 1) in
          let u_vec = [| u1 |] in
          let fv = Bench_suite.safe_eval s.spec s.bounds u_vec in
          let post = Bench_suite.predict_enn m u_vec in
          f_arr.(i) <- fv;
          mu_arr.(i) <- post.mean;
          lo_arr.(i) <- post.mean -. (2.0 *. post.sigma);
          hi_arr.(i) <- post.mean +. (2.0 *. post.sigma);
          lcb_arr.(i) <- post.lcb;
          note fv;
          note lo_arr.(i);
          note hi_arr.(i)
        done;
        let span = Float.max 1.0e-6 (!y_max -. !y_min) in
        let y0 = !y_min -. (0.08 *. span) in
        let y1 = !y_max +. (0.10 *. span) in
        let x_of_u u = pad_l +. (clamp01 u *. bw) in
        let y_of_v v =
          pad_t +. ((1.0 -. clamp01 ((v -. y0) /. (y1 -. y0))) *. bh)
        in

        (* Plot background & grid *)
        ctx##.fillStyle := js (if is_dark then "#0f1520" else "#f8fafc");
        ctx##fillRect (jf pad_l) (jf pad_t) (jf bw) (jf bh);
        ctx##.strokeStyle :=
          js (if is_dark then "rgba(148,163,184,0.14)" else "rgba(15,23,42,0.08)");
        ctx##.lineWidth := jf 0.8;
        for g = 1 to 3 do
          let gy = pad_t +. (float_of_int g *. 0.25 *. bh) in
          ctx##beginPath;
          ctx##moveTo (jf pad_l) (jf gy);
          ctx##lineTo (jf (pad_l +. bw)) (jf gy);
          ctx##stroke
        done;

        ctx##save;
        ctx##beginPath;
        ctx##rect (jf pad_l) (jf pad_t) (jf bw) (jf bh);
        ctx##clip;

        (* Shaded mu +/- 2 sigma_e band *)
        ctx##.fillStyle :=
          js
            (if is_dark then "rgba(56, 189, 248, 0.20)"
             else "rgba(2, 132, 199, 0.16)");
        ctx##beginPath;
        for i = 0 to n_pts - 1 do
          let u1 = float_of_int i /. float_of_int (n_pts - 1) in
          let px = x_of_u u1 and py = y_of_v hi_arr.(i) in
          if i = 0 then ctx##moveTo (jf px) (jf py) else ctx##lineTo (jf px) (jf py)
        done;
        for i = n_pts - 1 downto 0 do
          let u1 = float_of_int i /. float_of_int (n_pts - 1) in
          ctx##lineTo (jf (x_of_u u1)) (jf (y_of_v lo_arr.(i)))
        done;
        ctx##closePath;
        ctx##fill;

        (* True f(x1) dashed curve *)
        set_line_dash ctx [| 5.0; 3.0 |];
        ctx##.strokeStyle := js (if is_dark then "#94a3b8" else "#475569");
        ctx##.lineWidth := jf 1.4;
        ctx##beginPath;
        for i = 0 to n_pts - 1 do
          let u1 = float_of_int i /. float_of_int (n_pts - 1) in
          let px = x_of_u u1 and py = y_of_v f_arr.(i) in
          if i = 0 then ctx##moveTo (jf px) (jf py) else ctx##lineTo (jf px) (jf py)
        done;
        ctx##stroke;

        (* Optional LCB curve *)
        if !cur_field = Field_lcb then begin
          set_line_dash ctx [| 2.0; 2.5 |];
          ctx##.strokeStyle := js (if is_dark then "#fbbf24" else "#d97706");
          ctx##.lineWidth := jf 1.4;
          ctx##beginPath;
          for i = 0 to n_pts - 1 do
            let u1 = float_of_int i /. float_of_int (n_pts - 1) in
            let px = x_of_u u1 and py = y_of_v lcb_arr.(i) in
            if i = 0 then ctx##moveTo (jf px) (jf py)
            else ctx##lineTo (jf px) (jf py)
          done;
          ctx##stroke
        end;

        (* Posterior mean mu(x1) solid curve *)
        set_line_dash ctx [||];
        ctx##.strokeStyle := js (if is_dark then "#38bdf8" else "#0284c7");
        ctx##.lineWidth := jf 2.0;
        ctx##beginPath;
        for i = 0 to n_pts - 1 do
          let u1 = float_of_int i /. float_of_int (n_pts - 1) in
          let px = x_of_u u1 and py = y_of_v mu_arr.(i) in
          if i = 0 then ctx##moveTo (jf px) (jf py) else ctx##lineTo (jf px) (jf py)
        done;
        ctx##stroke;

        (* Active 1D trust-region interval bars near bottom *)
        Array.iteri
          (fun r_idx (reg : Bench_suite.region) ->
            let half = 0.5 *. reg.length in
            let rx0 = x_of_u (clamp01 (reg.center_u.(0) -. half)) in
            let rx1 = x_of_u (clamp01 (reg.center_u.(0) +. half)) in
            let ry = pad_t +. bh -. 6.0 -. (float_of_int r_idx *. 5.0) in
            ctx##.strokeStyle := js (region_hex r_idx is_dark);
            ctx##.lineWidth := jf (if r_idx = s.last_region then 2.8 else 1.8);
            ctx##beginPath;
            ctx##moveTo (jf rx0) (jf ry);
            ctx##lineTo (jf rx1) (jf ry);
            ctx##stroke)
          s.regions;

        (* Hover vertical crosshair & K-NN neighbor chords *)
        (match !hover_u with
        | None -> ()
        | Some (hu1, _) ->
            let post = Bench_suite.predict_enn m [| hu1 |] in
            let hx = x_of_u hu1 in
            let hy = y_of_v post.mean in
            set_line_dash ctx [| 3.0; 3.0 |];
            ctx##.strokeStyle :=
              js (if is_dark then "rgba(250,204,21,0.75)" else "rgba(217,119,6,0.75)");
            ctx##.lineWidth := jf 1.1;
            Array.iter
              (fun nb ->
                if nb >= 0 && nb < Array.length s.ys then begin
                  let nx = x_of_u s.xs_u.(nb).(0) in
                  let ny = y_of_v s.ys.(nb) in
                  ctx##beginPath;
                  ctx##moveTo (jf hx) (jf hy);
                  ctx##lineTo (jf nx) (jf ny);
                  ctx##stroke
                end)
              post.neighbors;
            ctx##.strokeStyle :=
              js (if is_dark then "rgba(226,232,240,0.65)" else "rgba(15,23,42,0.55)");
            ctx##beginPath;
            ctx##moveTo (jf hx) (jf pad_t);
            ctx##lineTo (jf hx) (jf (pad_t +. bh));
            ctx##stroke;
            set_line_dash ctx [||]);

        (* Evaluated observations & global incumbent *)
        for idx = 0 to Array.length s.ys - 1 do
          let px = x_of_u s.xs_u.(idx).(0) in
          let py = y_of_v s.ys.(idx) in
          if idx = s.best_idx then begin
            ctx##.fillStyle := js "#f43f5e";
            ctx##.strokeStyle := js "#ffffff";
            ctx##.lineWidth := jf 1.6;
            ctx##beginPath;
            ctx##arc (jf px) (jf py) (jf 5.0) (jf 0.0) (jf (2.0 *. Float.pi))
              Js._false;
            ctx##fill;
            ctx##stroke
          end
          else begin
            ctx##.fillStyle := js fg_ink;
            ctx##beginPath;
            ctx##arc (jf px) (jf py) (jf 2.7) (jf 0.0) (jf (2.0 *. Float.pi))
              Js._false;
            ctx##fill
          end
        done;
        ctx##restore;

        (* Frame & ticks *)
        ctx##.strokeStyle := js border_ink;
        ctx##.lineWidth := jf 1.1;
        ctx##strokeRect (jf pad_l) (jf pad_t) (jf bw) (jf bh);
        let lo0, hi0 = s.bounds.(0) in
        ctx##.fillStyle := js muted_ink;
        ctx##.font := js "400 9.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
        ctx##.textAlign := js "center";
        for t = 0 to 4 do
          let u = float_of_int t *. 0.25 in
          let xv = lo0 +. (u *. (hi0 -. lo0)) in
          ctx##fillText (js (fmt_axis xv)) (jf (x_of_u u)) (jf (pad_t +. bh +. 13.0))
        done;
        ctx##.font := js "italic 11px 'Latin Modern Roman', Georgia, serif";
        ctx##.fillStyle := js fg_ink;
        ctx##fillText (js "x\u{2081}") (jf (pad_l +. (0.5 *. bw)))
          (jf (pad_t +. bh +. 27.0));
        ctx##.font := js "400 9.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
        ctx##.fillStyle := js muted_ink;
        ctx##.textAlign := js "right";
        ctx##fillText (js (fmt_axis y1)) (jf (pad_l -. 5.0)) (jf (pad_t +. 8.0));
        ctx##fillText
          (js (fmt_axis (0.5 *. (y0 +. y1))))
          (jf (pad_l -. 5.0))
          (jf (pad_t +. (0.5 *. bh) +. 3.0));
        ctx##fillText (js (fmt_axis y0)) (jf (pad_l -. 5.0))
          (jf (pad_t +. bh -. 2.0))
      end
      else begin
        (* 2D Heatmap + Contours + Trust Regions + Active K-NN Hover Graph *)
        let pad_l = 44.0 and pad_r = 56.0 and pad_t = 16.0 and pad_b = 34.0 in
        let avail_w = Float.max 80.0 (cw -. pad_l -. pad_r) in
        let avail_h = Float.max 80.0 (ch -. pad_t -. pad_b) in
        let side = Float.min avail_w (avail_h *. 1.12) in
        let bw = side in
        let bh = Float.min avail_h side in
        let bx = pad_l +. (0.5 *. (avail_w -. bw)) in
        let by = pad_t +. (0.5 *. (avail_h -. bh)) in
        map_box := (bx, by, bw, bh);

        let want_key =
          Printf.sprintf "%s:%d:%s:%d:%d:%d" s.spec.id s.dim
            (Bench_suite.key_of_arm s.arm)
            (Array.length s.ys) s.seed
            (match !cur_field with
            | Field_sigma -> 0
            | Field_mean -> 1
            | Field_lcb -> 2
            | Field_true -> 3)
        in
        let mc =
          match !map_cache with
          | Some c when c.m_key = want_key -> c
          | _ ->
              let built = build_map2d_cache s !cur_field off_canvas in
              map_cache := Some built;
              built
        in

        set_smoothing ctx true;
        ctx##drawImage_fromCanvasWithSize mc.m_canvas (jf bx) (jf by) (jf bw)
          (jf bh);
        draw_contours ctx mc.m_tnorm bx by bw bh;

        let x_of_u1 u1 = bx +. (clamp01 u1 *. bw) in
        let y_of_u2 u2 = by +. ((1.0 -. clamp01 u2) *. bh) in

        ctx##save;
        ctx##beginPath;
        ctx##rect (jf bx) (jf by) (jf bw) (jf bh);
        ctx##clip;

        (* Active TuRBO Trust-Region Rectangles *)
        Array.iteri
          (fun r_idx (reg : Bench_suite.region) ->
            let half = 0.5 *. reg.length in
            let u1_lo = clamp01 (reg.center_u.(0) -. half) in
            let u1_hi = clamp01 (reg.center_u.(0) +. half) in
            let u2_lo = clamp01 (reg.center_u.(1) -. half) in
            let u2_hi = clamp01 (reg.center_u.(1) +. half) in
            let rx = x_of_u1 u1_lo in
            let ry = y_of_u2 u2_hi in
            let rw = Float.max 3.0 (x_of_u1 u1_hi -. rx) in
            let rh = Float.max 3.0 (y_of_u2 u2_lo -. ry) in
            let col_hex = region_hex r_idx is_dark in
            ctx##.fillStyle := js "rgba(255, 255, 255, 0.05)";
            ctx##fillRect (jf rx) (jf ry) (jf rw) (jf rh);
            ctx##.strokeStyle := js col_hex;
            if r_idx = s.last_region then begin
              set_line_dash ctx [||];
              ctx##.lineWidth := jf 1.9
            end
            else begin
              set_line_dash ctx [| 4.0; 2.8 |];
              ctx##.lineWidth := jf 1.3
            end;
            ctx##strokeRect (jf rx) (jf ry) (jf rw) (jf rh);
            set_line_dash ctx [||];
            if rw > 22.0 && rh > 14.0 then begin
              ctx##.fillStyle := js col_hex;
              ctx##.font :=
                js "700 8.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
              ctx##.textAlign := js "left";
              ctx##fillText
                (js (Printf.sprintf "R%d" (r_idx + 1)))
                (jf (rx +. 3.0)) (jf (ry +. 9.5))
            end)
          s.regions;

        (* Active Hover K-NN Stencil Rays & Neighbor Rings *)
        let active_nbs : int array =
          match !hover_u with
          | None -> [||]
          | Some (hu1, hu2) ->
              let probe = Array.copy s.xs_u.(s.best_idx) in
              probe.(0) <- hu1;
              probe.(1) <- hu2;
              let post = Bench_suite.predict_enn m probe in
              let cx_p = x_of_u1 hu1 and cy_p = y_of_u2 hu2 in
              set_line_dash ctx [| 3.5; 2.5 |];
              ctx##.strokeStyle := js "rgba(255, 255, 255, 0.88)";
              ctx##.lineWidth := jf 1.25;
              Array.iter
                (fun nb ->
                  if nb >= 0 && nb < Array.length s.ys then begin
                    let nx = x_of_u1 s.xs_u.(nb).(0) in
                    let ny = y_of_u2 s.xs_u.(nb).(1) in
                    ctx##beginPath;
                    ctx##moveTo (jf cx_p) (jf cy_p);
                    ctx##lineTo (jf nx) (jf ny);
                    ctx##stroke
                  end)
                post.neighbors;
              set_line_dash ctx [||];
              ctx##.strokeStyle := js "#ffffff";
              ctx##.lineWidth := jf 1.4;
              ctx##beginPath;
              ctx##arc (jf cx_p) (jf cy_p) (jf 4.2) (jf 0.0)
                (jf (2.0 *. Float.pi)) Js._false;
              ctx##stroke;
              post.neighbors
        in
        let is_nb idx =
          let found = ref false in
          for r = 0 to Array.length active_nbs - 1 do
            if active_nbs.(r) = idx then found := true
          done;
          !found
        in

        (* Evaluated Sample Points + Global Incumbent Star *)
        let n_obs = Array.length s.ys in
        for idx = 0 to n_obs - 1 do
          if idx <> s.best_idx then begin
            let px = x_of_u1 s.xs_u.(idx).(0) in
            let py = y_of_u2 s.xs_u.(idx).(1) in
            if is_nb idx then begin
              ctx##.strokeStyle := js "#fde047";
              ctx##.lineWidth := jf 2.0;
              ctx##beginPath;
              ctx##arc (jf px) (jf py) (jf 5.2) (jf 0.0) (jf (2.0 *. Float.pi))
                Js._false;
              ctx##stroke
            end;
            if idx < s.n_init then begin
              ctx##.fillStyle := js "rgba(15, 23, 42, 0.65)";
              ctx##.strokeStyle := js "rgba(255, 255, 255, 0.85)";
              ctx##.lineWidth := jf 1.1;
              ctx##beginPath;
              ctx##arc (jf px) (jf py) (jf 2.4) (jf 0.0) (jf (2.0 *. Float.pi))
                Js._false;
              ctx##fill;
              ctx##stroke
            end
            else begin
              let is_latest = idx = n_obs - 1 in
              ctx##.fillStyle := js (if is_latest then "#38bdf8" else "#ffffff");
              ctx##.strokeStyle := js "#0f172a";
              ctx##.lineWidth := jf 1.0;
              ctx##beginPath;
              ctx##arc (jf px) (jf py) (jf (if is_latest then 3.6 else 2.6))
                (jf 0.0) (jf (2.0 *. Float.pi)) Js._false;
              ctx##fill;
              ctx##stroke
            end
          end
        done;

        (* Incumbent Star Marker x* *)
        let bx_s = x_of_u1 s.xs_u.(s.best_idx).(0) in
        let by_s = y_of_u2 s.xs_u.(s.best_idx).(1) in
        ctx##.fillStyle := js "#f43f5e";
        ctx##.strokeStyle := js "#ffffff";
        ctx##.lineWidth := jf 1.6;
        ctx##beginPath;
        for v = 0 to 9 do
          let ang =
            (-.0.5 *. Float.pi) +. (float_of_int v *. Float.pi /. 5.0)
          in
          let rad = if v mod 2 = 0 then 6.8 else 3.1 in
          let sx = bx_s +. (rad *. Float.cos ang) in
          let sy = by_s +. (rad *. Float.sin ang) in
          if v = 0 then ctx##moveTo (jf sx) (jf sy) else ctx##lineTo (jf sx) (jf sy)
        done;
        ctx##closePath;
        ctx##fill;
        ctx##stroke;
        ctx##restore;

        (* Plot Frame & Domain Ticks *)
        ctx##.strokeStyle := js border_ink;
        ctx##.lineWidth := jf 1.1;
        ctx##strokeRect (jf bx) (jf by) (jf bw) (jf bh);

        let lo0, hi0 = s.bounds.(0) in
        let lo1, hi1 = s.bounds.(1) in
        ctx##.fillStyle := js muted_ink;
        ctx##.font := js "400 9px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
        ctx##.textAlign := js "center";
        for t = 0 to 2 do
          let u = float_of_int t *. 0.5 in
          let xv = lo0 +. (u *. (hi0 -. lo0)) in
          let px = x_of_u1 u in
          ctx##beginPath;
          ctx##moveTo (jf px) (jf (by +. bh));
          ctx##lineTo (jf px) (jf (by +. bh +. 3.5));
          ctx##stroke;
          ctx##fillText (js (fmt_axis xv)) (jf px) (jf (by +. bh +. 13.0))
        done;
        ctx##.textAlign := js "right";
        for t = 0 to 2 do
          let u = float_of_int t *. 0.5 in
          let yv = lo1 +. (u *. (hi1 -. lo1)) in
          let py = y_of_u2 u in
          ctx##beginPath;
          ctx##moveTo (jf (bx -. 3.5)) (jf py);
          ctx##lineTo (jf bx) (jf py);
          ctx##stroke;
          ctx##fillText (js (fmt_axis yv)) (jf (bx -. 5.0)) (jf (py +. 3.0))
        done;
        ctx##.fillStyle := js fg_ink;
        ctx##.font := js "italic 11px 'Latin Modern Roman', Georgia, serif";
        ctx##.textAlign := js "center";
        ctx##fillText
          (js (if s.dim > 2 then "x\u{2081} (slice @ x*)" else "x\u{2081}"))
          (jf (bx +. (0.5 *. bw)))
          (jf (by +. bh +. 26.5));
        ctx##save;
        ctx##translate (jf (bx -. 31.0)) (jf (by +. (0.5 *. bh)));
        ctx##rotate (jf (-.0.5 *. Float.pi));
        ctx##fillText (js "x\u{2082}") (jf 0.0) (jf 0.0);
        ctx##restore;

        (* Right Colorbar *)
        let cb_x = bx +. bw +. 10.0 in
        let cb_w = 10.0 in
        let grad = ctx##createLinearGradient (jf cb_x) (jf (by +. bh)) (jf cb_x) (jf by) in
        for st = 0 to 8 do
          let u = float_of_int st /. 8.0 in
          let rgb =
            match !cur_field with
            | Field_sigma -> sigma_rgb u
            | Field_mean | Field_lcb | Field_true -> surface_rgb u
          in
          let r8, g8, b8 = rgb_bytes rgb in
          grad##addColorStop (jf u) (js (Printf.sprintf "rgb(%d,%d,%d)" r8 g8 b8))
        done;
        ctx##.fillStyle_gradient := grad;
        ctx##fillRect (jf cb_x) (jf by) (jf cb_w) (jf bh);
        ctx##.strokeStyle := js border_ink;
        ctx##.lineWidth := jf 0.9;
        ctx##strokeRect (jf cb_x) (jf by) (jf cb_w) (jf bh);
        ctx##.fillStyle := js muted_ink;
        ctx##.font := js "400 8.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
        ctx##.textAlign := js "left";
        ctx##fillText (js (fmt_axis mc.m_scale.raw_max)) (jf (cb_x +. cb_w +. 3.0))
          (jf (by +. 7.0));
        ctx##fillText (js (fmt_axis mc.m_scale.raw_min)) (jf (cb_x +. cb_w +. 3.0))
          (jf (by +. bh))
      end;
      ctx##restore
    in

    (* --- Render Synchronized Convergence Strip (.oxb-trace-canvas) --- *)
    let render_trace () =
      let s = !cur_session in
      let is_dark = is_dark_theme () in
      let cw, ch, dpr = sync_canvas_dpi trace_canvas in
      let ctx = trace_canvas##getContext _2d_ in
      ctx##save;
      ctx##setTransform (jf dpr) (jf 0.0) (jf 0.0) (jf dpr) (jf 0.0) (jf 0.0);
      ctx##.fillStyle := js (if is_dark then "#0d1117" else "#ffffff");
      ctx##fillRect (jf 0.0) (jf 0.0) (jf cw) (jf ch);

      let pad_l = 56.0 and pad_r = 18.0 and pad_t = 10.0 and pad_b = 20.0 in
      let bw = Float.max 100.0 (cw -. pad_l -. pad_r) in
      let bh = Float.max 36.0 (ch -. pad_t -. pad_b) in
      let n_obs = Array.length s.best_trace in
      let max_x = Int.max max_auto_evals n_obs in
      let sc = build_norm_scale s.best_trace in

      let x_of_step i =
        pad_l
        +. (float_of_int i /. float_of_int (Int.max 1 (max_x - 1)) *. bw)
      in
      let y_of_val v = pad_t +. ((1.0 -. eval_norm sc v) *. bh) in

      ctx##.fillStyle := js (if is_dark then "#0f1520" else "#f8fafc");
      ctx##fillRect (jf pad_l) (jf pad_t) (jf bw) (jf bh);

      (* Gridlines *)
      ctx##.strokeStyle :=
        js (if is_dark then "rgba(148,163,184,0.14)" else "rgba(15,23,42,0.08)");
      ctx##.lineWidth := jf 0.8;
      ctx##beginPath;
      ctx##moveTo (jf pad_l) (jf (pad_t +. (0.5 *. bh)));
      ctx##lineTo (jf (pad_l +. bw)) (jf (pad_t +. (0.5 *. bh)));
      ctx##stroke;

      (* Vertical n_init LHS -> BO boundary *)
      let init_x = x_of_step (s.n_init - 1) in
      set_line_dash ctx [| 3.5; 2.5 |];
      ctx##.strokeStyle :=
        js (if is_dark then "rgba(251,191,36,0.65)" else "rgba(180,83,9,0.55)");
      ctx##.lineWidth := jf 1.1;
      ctx##beginPath;
      ctx##moveTo (jf init_x) (jf pad_t);
      ctx##lineTo (jf init_x) (jf (pad_t +. bh));
      ctx##stroke;
      set_line_dash ctx [||];
      ctx##.fillStyle := js (if is_dark then "#fbbf24" else "#b45309");
      ctx##.font := js "500 8.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
      ctx##.textAlign := js "left";
      ctx##fillText
        (js (Printf.sprintf "LHS (%d) \u{2192} BO" s.n_init))
        (jf (init_x +. 4.0)) (jf (pad_t +. 10.0));

      (* Faint individual evaluation scatter dots *)
      ctx##.fillStyle :=
        js (if is_dark then "rgba(148,163,184,0.34)" else "rgba(71,85,105,0.30)");
      for i = 0 to n_obs - 1 do
        let px = x_of_step i in
        let py = y_of_val s.ys.(i) in
        ctx##beginPath;
        ctx##arc (jf px) (jf py) (jf 1.6) (jf 0.0) (jf (2.0 *. Float.pi))
          Js._false;
        ctx##fill
      done;

      (* Monotone best-so-far trace + subtle area fill *)
      let line_col = if is_dark then "#38bdf8" else "#0284c7" in
      if n_obs > 0 then begin
        ctx##.fillStyle :=
          js
            (if is_dark then "rgba(56, 189, 248, 0.12)"
             else "rgba(2, 132, 199, 0.10)");
        ctx##beginPath;
        ctx##moveTo (jf (x_of_step 0)) (jf (pad_t +. bh));
        for i = 0 to n_obs - 1 do
          ctx##lineTo (jf (x_of_step i)) (jf (y_of_val s.best_trace.(i)))
        done;
        ctx##lineTo (jf (x_of_step (n_obs - 1))) (jf (pad_t +. bh));
        ctx##closePath;
        ctx##fill;

        ctx##.strokeStyle := js line_col;
        ctx##.lineWidth := jf 2.0;
        ctx##beginPath;
        for i = 0 to n_obs - 1 do
          let px = x_of_step i and py = y_of_val s.best_trace.(i) in
          if i = 0 then ctx##moveTo (jf px) (jf py)
          else ctx##lineTo (jf px) (jf py)
        done;
        ctx##stroke;

        let last_x = x_of_step (n_obs - 1) in
        let last_y = y_of_val s.best_trace.(n_obs - 1) in
        ctx##.fillStyle := js "#f43f5e";
        ctx##beginPath;
        ctx##arc (jf last_x) (jf last_y) (jf 3.3) (jf 0.0)
          (jf (2.0 *. Float.pi)) Js._false;
        ctx##fill
      end;

      (* Border & Axis Labels *)
      ctx##.strokeStyle := js (if is_dark then "#30363d" else "#cbd5e1");
      ctx##.lineWidth := jf 1.0;
      ctx##strokeRect (jf pad_l) (jf pad_t) (jf bw) (jf bh);

      let muted_ink = if is_dark then "#8b949e" else "#475569" in
      ctx##.fillStyle := js muted_ink;
      ctx##.font := js "400 8.5px 'IBM Plex Mono', 'Latin Modern Mono', monospace";
      ctx##.textAlign := js "right";
      ctx##fillText (js (fmt_num sc.raw_max)) (jf (pad_l -. 5.0))
        (jf (pad_t +. 8.0));
      ctx##fillText (js (fmt_num sc.raw_min)) (jf (pad_l -. 5.0))
        (jf (pad_t +. bh));

      ctx##.textAlign := js "center";
      List.iter
        (fun st ->
          if st <= max_x then
            ctx##fillText (js (string_of_int st))
              (jf (x_of_step (Int.max 0 (st - 1))))
              (jf (pad_t +. bh +. 12.5)))
        [ 1; 60; 120; 180; max_x ];
      ctx##restore
    in

    (* --- Update Hover Readout & Status Telemetry --- *)
    let update_hover_readout () =
      let s = !cur_session in
      let best_y = s.ys.(s.best_idx) in
      let best_x = Bench_suite.u_to_x s.bounds s.xs_u.(s.best_idx) in
      let x_star_str =
        if s.dim = 1 then Printf.sprintf "x* = %.4f" best_x.(0)
        else if s.dim = 2 then
          Printf.sprintf "x* = (%.3f, %.3f)" best_x.(0) best_x.(1)
        else
          Printf.sprintf "x* = (%.3f, %.3f, \u{2026})" best_x.(0) best_x.(1)
      in
      match !hover_u with
      | None ->
          set_text hover_bar
            (Printf.sprintf
               "%s  \u{00B7}  Incumbent %s  \u{00B7}  f(x*) = %s  \u{00B7}  Hover pane (b) to inspect local ENN posterior & active K-NN stencil"
               (Bench_suite.label_of_arm s.arm)
               x_star_str (fmt_num best_y))
      | Some (hu1, hu2) ->
          let probe = Array.copy s.xs_u.(s.best_idx) in
          probe.(0) <- hu1;
          if s.dim >= 2 then probe.(1) <- hu2;
          let m = Bench_suite.ensure_model s in
          let post = Bench_suite.predict_enn m probe in
          let f_val = Bench_suite.safe_eval s.spec s.bounds probe in
          let x_cur = Bench_suite.u_to_x s.bounds probe in
          let coord_str =
            if s.dim = 1 then Printf.sprintf "x\u{2081} = %.4f" x_cur.(0)
            else Printf.sprintf "x = (%.3f, %.3f)" x_cur.(0) x_cur.(1)
          in
          set_text hover_bar
            (Printf.sprintf
               "Cursor %s  \u{00B7}  True f(x) = %s  \u{00B7}  ENN \u{03BC}(x) = %s  \u{00B7}  \u{03C3}_e(x) = %s  \u{00B7}  LCB \u{03B1}(x) = %s  \u{00B7}  Stencil K = %d neighbors"
               coord_str (fmt_num f_val) (fmt_num post.mean) (fmt_num post.sigma)
               (fmt_num post.lcb)
               (Array.length post.neighbors))
    in

    let sync_ui_chrome () =
      let s = !cur_session in
      let sp = s.spec in
      let scalable = Bench_suite.is_scalable sp.rule in

      (* Family pills *)
      Array.iter
        (fun (btn, fam_opt) -> toggle_class btn "active" (fam_opt = !cur_family))
        fam_btns;

      (* Dimension regime buttons *)
      set_text dim_active_lbl (Printf.sprintf "D = %d" s.dim);
      let ok16 = scalable && Bench_suite.valid_dim sp.rule 16 in
      let ok64 = scalable && Bench_suite.valid_dim sp.rule 64 in
      btn_dim_16##.disabled := Js.bool (not ok16);
      btn_dim_64##.disabled := Js.bool (not ok64);
      toggle_class btn_dim_16 "is-disabled" (not ok16);
      toggle_class btn_dim_64 "is-disabled" (not ok64);
      toggle_class btn_dim_can "active" (!cur_regime = Reg_canonical || not scalable);
      toggle_class btn_dim_16 "active" (ok16 && !cur_regime = Reg_d16);
      toggle_class btn_dim_64 "active" (ok64 && !cur_regime = Reg_d64);

      (* Arm buttons *)
      Array.iter
        (fun (btn, arm_v) -> toggle_class btn "active" (arm_v = !cur_arm))
        arm_btns;

      (* Field buttons *)
      Array.iter
        (fun (btn, fm) -> toggle_class btn "active" (fm = !cur_field))
        field_btns;

      (* Status & seed *)
      let n_obs = Array.length s.ys in
      let best_y = s.ys.(s.best_idx) in
      set_text status_badge
        (Printf.sprintf "Evals: %d  \u{00B7}  Best f(x*): %s" n_obs
           (fmt_num best_y));
      set_text btn_seed (Printf.sprintf "Seed %d" !cur_seed);

      (* HUD Badges *)
      if s.dim = 1 then begin
        set_text badge_a_title "3D Objective Ribbon f(x\u{2081})";
        set_text badge_a_hint "1D Curve Extrusion \u{00B7} Drag to orbit";
        set_text badge_b_sub "1D \u{03BC}(x\u{2081}) \u{00B1} 2\u{03C3}_e"
      end
      else if s.dim = 2 then begin
        set_text badge_a_title "3D Objective Surface f(x\u{2081}, x\u{2082})";
        set_text badge_a_hint "Drag to orbit \u{00B7} Scroll to zoom";
        let f_lbl =
          match !cur_field with
          | Field_sigma -> "Epistemic \u{03C3}_e(x)"
          | Field_mean -> "Posterior \u{03BC}(x)"
          | Field_lcb -> "LCB \u{03B1}(x)"
          | Field_true -> "True f(x)"
        in
        set_text badge_b_sub f_lbl
      end
      else begin
        set_text badge_a_title "3D Objective Surface f(x\u{2081}, x\u{2082})";
        set_text badge_a_hint
          (Printf.sprintf "(x\u{2081}, x\u{2082}) slice @ x* (D=%d)" s.dim);
        let f_lbl =
          match !cur_field with
          | Field_sigma -> Printf.sprintf "\u{03C3}_e slice (D=%d)" s.dim
          | Field_mean -> Printf.sprintf "\u{03BC} slice (D=%d)" s.dim
          | Field_lcb -> Printf.sprintf "LCB slice (D=%d)" s.dim
          | Field_true -> Printf.sprintf "f(x) slice (D=%d)" s.dim
        in
        set_text badge_b_sub f_lbl
      end;

      (* Trace header & Metadata chips *)
      let init_y = s.best_trace.(Int.max 0 (s.n_init - 1)) in
      set_text trace_hdr_right
        (Printf.sprintf "Init best: %s \u{2192} Current: %s (Seed %d)"
           (fmt_num init_y) (fmt_num best_y) s.seed);
      set_text chip_domain
        (Printf.sprintf "Domain: %s" (domain_summary s.bounds));
      set_text chip_rule (Printf.sprintf "Rule: %s" (rule_summary sp s.dim));
      set_text chip_family
        (Printf.sprintf "Family: %s" (Bench_suite.family_title sp.family));
      let lens =
        Array.to_list s.regions
        |> List.map (fun (r : Bench_suite.region) -> Printf.sprintf "%.2f" r.length)
        |> String.concat ", "
      in
      set_text chip_regions (Printf.sprintf "Regions L=[%s]" lens);
      if String.starts_with ~prefix:"https://" sp.source then
        src_link##.href := js sp.source
      else src_link##.href := js "https://www.sfu.ca/~ssurjano/optimization.html";

      update_hover_readout ()
    in

    let render_all () =
      sync_ui_chrome ();
      render_pane_a ();
      render_pane_b ();
      render_trace ()
    in

    (* --- Rebuild Session Helper --- *)
    let reset_session ?(warm_steps = 24) () =
      if not (Bench_suite.is_scalable (!cur_spec).rule) then
        cur_regime := Reg_canonical;
      let d = resolve_dim !cur_spec !cur_regime in
      hover_u := None;
      surf_cache := None;
      map_cache := None;
      cur_session :=
        Bench_suite.init_session ~seed:!cur_seed ~warm_steps ~spec:!cur_spec
          ~dim:d ~arm:!cur_arm ();
      render_all ()
    in

    (* --- Populate Function <select> Matching Family --- *)
    let rebuild_fn_options () =
      let matching =
        Array.to_list Bench_suite.catalogue
        |> List.filter (fun (sp : Bench_suite.spec) ->
               match !cur_family with
               | None -> true
               | Some fam -> sp.family = fam)
      in
      set_text fn_count_span
        (Printf.sprintf "%d %s" (List.length matching)
           (if List.length matching = 1 then "fn" else "fns"));
      clear_children fn_select;
      let has_cur =
        List.exists (fun (sp : Bench_suite.spec) -> sp.id = (!cur_spec).id) matching
      in
      if not has_cur then (
        match matching with sp0 :: _ -> cur_spec := sp0 | [] -> ());
      List.iter
        (fun (sp : Bench_suite.spec) ->
          let opt = createOption document in
          opt##.value := js sp.id;
          set_text opt (option_label_of_spec sp);
          if sp.id = (!cur_spec).id then opt##.selected := Js._true;
          Dom.appendChild fn_select opt)
        matching;
      fn_select##.value := js (!cur_spec).id;
      has_cur
    in
    ignore (rebuild_fn_options ());

    (* --- Wire Sidebar Controls --- *)
    Array.iter
      (fun (btn, fam_opt) ->
        add_click btn (fun () ->
            if !cur_family <> fam_opt then begin
              stop_playback ();
              cur_family := fam_opt;
              let kept = rebuild_fn_options () in
              if kept then sync_ui_chrome () else reset_session ~warm_steps:24 ()
            end))
      fam_btns;

    let on_fn_change =
      handler (fun _ev ->
          let id = Js.to_string fn_select##.value in
          (match Bench_suite.find_spec id with
          | Some sp when sp.id <> (!cur_spec).id ->
              stop_playback ();
              cur_spec := sp;
              reset_session ~warm_steps:24 ()
          | _ -> ());
          Js._true)
    in
    ignore (addEventListener fn_select Event.change on_fn_change Js._false);

    let bind_regime btn reg =
      add_click btn (fun () ->
          if not (Js.to_bool btn##.disabled) && !cur_regime <> reg then begin
            stop_playback ();
            cur_regime := reg;
            reset_session ~warm_steps:24 ()
          end)
    in
    bind_regime btn_dim_can Reg_canonical;
    bind_regime btn_dim_16 Reg_d16;
    bind_regime btn_dim_64 Reg_d64;

    Array.iter
      (fun (btn, arm_v) ->
        add_click btn (fun () ->
            if !cur_arm <> arm_v then begin
              stop_playback ();
              cur_arm := arm_v;
              reset_session ~warm_steps:24 ()
            end))
      arm_btns;

    Array.iter
      (fun (btn, fm) ->
        add_click btn (fun () ->
            if !cur_field <> fm then begin
              cur_field := fm;
              sync_ui_chrome ();
              render_pane_b ()
            end))
      field_btns;

    (* Playback & Step Buttons *)
    let start_playback () =
      stop_playback ();
      if Array.length (!cur_session).ys >= max_auto_evals then
        reset_session ~warm_steps:0 ();
      set_text btn_play "\u{23F8} Pause";
      toggle_class btn_play "active" true;
      let cb =
        Js.wrap_callback (fun () ->
            let s = !cur_session in
            if Array.length s.ys >= max_auto_evals then stop_playback ()
            else begin
              Bench_suite.step_once s;
              render_all ();
              if Array.length s.ys >= max_auto_evals then stop_playback ()
            end)
      in
      play_timer := Some (window##setInterval cb (jf 140.0))
    in

    add_click btn_play (fun () ->
        match !play_timer with
        | Some _ -> stop_playback ()
        | None -> start_playback ());

    add_click btn_step1 (fun () ->
        stop_playback ();
        Bench_suite.step_once !cur_session;
        render_all ());

    add_click btn_step10 (fun () ->
        stop_playback ();
        Bench_suite.step_many !cur_session 10;
        render_all ());

    add_click btn_reset (fun () ->
        stop_playback ();
        reset_session ~warm_steps:0 ());

    add_click btn_seed (fun () ->
        stop_playback ();
        cur_seed := (!cur_seed mod 5) + 1;
        reset_session ~warm_steps:24 ());

    (* --- 3D Orbit Mouse & Touch Handlers on Pane (a) --- *)
    let dragging = ref false in
    let last_xy = ref (0.0, 0.0) in

    let on_gl_down =
      handler (fun (ev : mouseEvent Js.t) ->
          dragging := true;
          last_xy := (tf ev##.clientX, tf ev##.clientY);
          Js._true)
    in
    let on_win_move =
      handler (fun (ev : mouseEvent Js.t) ->
          if !dragging then begin
            let x = tf ev##.clientX and y = tf ev##.clientY in
            let lx, ly = !last_xy in
            last_xy := (x, y);
            cam_yaw := !cam_yaw -. ((x -. lx) *. 0.0085);
            cam_pitch :=
              Bench_suite.clamp 0.08 1.46 (!cam_pitch +. ((y -. ly) *. 0.0075));
            render_pane_a ()
          end;
          Js._true)
    in
    let on_win_up =
      handler (fun (_ev : mouseEvent Js.t) ->
          dragging := false;
          Js._true)
    in
    let on_gl_dblclick =
      handler (fun (_ev : mouseEvent Js.t) ->
          cam_yaw := -0.68;
          cam_pitch := 0.56;
          cam_dist := 3.10;
          render_pane_a ();
          Js._true)
    in
    let wheel_typ : mouseEvent Js.t Event.typ = Event.make "wheel" in
    let on_gl_wheel =
      handler (fun (ev : mouseEvent Js.t) ->
          Dom.preventDefault ev;
          let wev : wheel_ev Js.t = Js.Unsafe.coerce ev in
          let dy = tf wev##.deltaY in
          let factor = if dy > 0.0 then 1.07 else 0.935 in
          cam_dist := Bench_suite.clamp 1.85 5.20 (!cam_dist *. factor);
          render_pane_a ();
          Js._false)
    in
    let touch_dragging = ref false in
    let on_gl_touchstart =
      handler (fun (ev : touchEvent Js.t) ->
          (match Js.Optdef.to_option (ev##.touches##item 0) with
          | Some t when ev##.touches##.length = 1 ->
              touch_dragging := true;
              last_xy := (tf t##.clientX, tf t##.clientY)
          | _ -> touch_dragging := false);
          Js._true)
    in
    let on_gl_touchmove =
      handler (fun (ev : touchEvent Js.t) ->
          (match Js.Optdef.to_option (ev##.touches##item 0) with
          | Some t when !touch_dragging && ev##.touches##.length = 1 ->
              Dom.preventDefault ev;
              let x = tf t##.clientX and y = tf t##.clientY in
              let lx, ly = !last_xy in
              last_xy := (x, y);
              cam_yaw := !cam_yaw -. ((x -. lx) *. 0.009);
              cam_pitch :=
                Bench_suite.clamp 0.08 1.46
                  (!cam_pitch +. ((y -. ly) *. 0.008));
              render_pane_a ()
          | _ -> ());
          Js._true)
    in
    let on_gl_touchend =
      handler (fun (_ev : touchEvent Js.t) ->
          touch_dragging := false;
          Js._true)
    in

    ignore (addEventListener gl_canvas Event.mousedown on_gl_down Js._false);
    ignore (addEventListener window Event.mousemove on_win_move Js._false);
    ignore (addEventListener window Event.mouseup on_win_up Js._false);
    ignore (addEventListener gl_canvas Event.dblclick on_gl_dblclick Js._false);
    ignore
      (addEventListenerWithOptions gl_canvas wheel_typ ~passive:Js._false
         on_gl_wheel);
    ignore
      (addEventListenerWithOptions gl_canvas Event.touchstart ~passive:Js._true
         on_gl_touchstart);
    ignore
      (addEventListenerWithOptions gl_canvas Event.touchmove ~passive:Js._false
         on_gl_touchmove);
    ignore
      (addEventListenerWithOptions gl_canvas Event.touchend ~passive:Js._true
         on_gl_touchend);

    (* --- 2D Surrogate Hover & Touch Handlers on Pane (b) --- *)
    let update_hover_client client_x client_y =
      let rect = map_canvas##getBoundingClientRect in
      let lx = client_x -. tf rect##.left in
      let ly = client_y -. tf rect##.top in
      let bx, by, bw, bh = !map_box in
      if lx >= bx && lx <= bx +. bw && ly >= by && ly <= by +. bh then begin
        let u1 = clamp01 ((lx -. bx) /. Float.max 1.0 bw) in
        let u2 = clamp01 (1.0 -. ((ly -. by) /. Float.max 1.0 bh)) in
        hover_u := Some (u1, u2);
        update_hover_readout ();
        render_pane_b ();
        render_pane_a ()
      end
      else if !hover_u <> None then begin
        hover_u := None;
        update_hover_readout ();
        render_pane_b ();
        render_pane_a ()
      end
    in
    let on_map_move =
      handler (fun (ev : mouseEvent Js.t) ->
          update_hover_client (tf ev##.clientX) (tf ev##.clientY);
          Js._true)
    in
    let on_map_leave =
      handler (fun (_ev : mouseEvent Js.t) ->
          if !hover_u <> None then begin
            hover_u := None;
            update_hover_readout ();
            render_pane_b ();
            render_pane_a ()
          end;
          Js._true)
    in
    let on_map_touch =
      handler (fun (ev : touchEvent Js.t) ->
          (match Js.Optdef.to_option (ev##.touches##item 0) with
          | Some t -> update_hover_client (tf t##.clientX) (tf t##.clientY)
          | None -> ());
          Js._true)
    in
    ignore (addEventListener map_canvas Event.mousemove on_map_move Js._false);
    ignore
      (addEventListener map_canvas
         (Event.make "mouseleave" : mouseEvent Js.t Event.typ)
         on_map_leave Js._false);
    ignore
      (addEventListenerWithOptions map_canvas Event.touchstart ~passive:Js._true
         on_map_touch);
    ignore
      (addEventListenerWithOptions map_canvas Event.touchmove ~passive:Js._true
         on_map_touch);

    (* --- Theme Toggle & Resize Listeners --- *)
    let schedule_rerender ms =
      let cb = Js.wrap_callback (fun () -> render_all ()) in
      ignore (window##setTimeout cb (jf ms))
    in
    List.iter
      (fun id ->
        match Dom_html.getElementById_opt id with
        | Some el -> add_click el (fun () -> schedule_rerender 16.0)
        | None -> ())
      [ "theme-toggle-btn"; "theme-toggle" ];

    let on_resize =
      handler (fun _ev ->
          render_all ();
          Js._true)
    in
    ignore (addEventListener window Event.resize on_resize Js._false);

    render_all ();
    schedule_rerender 40.0
  end

let init () =
  match Dom_html.getElementById_opt "ocaml-bench-simulation" with
  | Some root -> mount_viewer root
  | None -> ()
