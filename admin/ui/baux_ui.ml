[@@@warning "-32-27"]
open Js_of_ocaml
open Dom_html

let jf = Js.float
let js = Js.string

(* --- Canvas helpers --- *)
let fill_rect ctx x y w h = ctx##fillRect (jf x) (jf y) (jf w) (jf h)
let stroke_rect ctx x y w h = ctx##strokeRect (jf x) (jf y) (jf w) (jf h)
let arc ctx x y r a1 a2 = ctx##arc (jf x) (jf y) (jf r) (jf a1) (jf a2) Js._false
let move_to ctx x y = ctx##moveTo (jf x) (jf y)
let line_to ctx x y = ctx##lineTo (jf x) (jf y)

let query_all sel =
  let nl = document##querySelectorAll (js sel) in
  let len = nl##.length in
  let acc = ref [] in
  for i = len - 1 downto 0 do
    match Js.Opt.to_option (nl##item i) with
    | Some el -> acc := el :: !acc
    | None -> ()
  done;
  !acc

let add_click el f =
  let h = handler (fun _ev -> f (); Js._true) in
  let _ = addEventListener el Event.click h Js._false in
  ()

let set_timeout f ms =
  let cb = Js.wrap_callback f in
  let _ = window##setTimeout cb (jf ms) in
  ()

(* --- Optimization Objectives (higher is better) --- *)
let ackley x y =
  let t1 = -20.0 *. exp (-0.2 *. sqrt (0.5 *. ((x *. x) +. (y *. y)))) in
  let t2 = -.exp (0.5 *. (cos (2.0 *. Float.pi *. x) +. cos (2.0 *. Float.pi *. y))) in
  -.(t1 +. t2 +. 20.0 +. exp 1.0)

let rosenbrock x y =
  let term1 = (1.0 -. x) ** 2.0 in
  let term2 = 100.0 *. ((y -. (x *. x)) ** 2.0) in
  -.(term1 +. term2) /. 20.0

let sphere x y =
  -.((x *. x) +. (y *. y))

type point = { x : float; y : float; val_ : float }

type sim_state = {
  mutable func_name : string;
  mutable range : float;
  mutable points : point list;
  mutable incumbent : point option;
  mutable tr_cx : float;
  mutable tr_cy : float;
  mutable tr_len : float;
  mutable succ_count : int;
  mutable fail_count : int;
  succ_tol : int;
  fail_tol : int;
  mutable round : int;
  mutable candidates : point list;
  mutable status : string;
  mutable running : bool;
  mutable pulse_anim : float;
  mutable pulse_color : string;
}

let eval_func name x y =
  match name with
  | "rosenbrock" -> rosenbrock x y
  | "sphere" -> sphere x y
  | _ -> ackley x y

let clamp min_val max_val v =
  if v < min_val then min_val else if v > max_val then max_val else v

let to_canvas sim w h x y =
  let r = sim.range in
  let cx = ((x +. r) /. (2.0 *. r)) *. w in
  let cy = ((r -. y) /. (2.0 *. r)) *. h in
  (cx, cy)

let reset_sim sim =
  sim.range <- (if sim.func_name = "rosenbrock" then 2.0 else 2.5);
  sim.points <- [];
  sim.candidates <- [];
  sim.round <- 0;
  sim.succ_count <- 0;
  sim.fail_count <- 0;
  sim.tr_len <- 1.0;
  sim.status <- "Ready";

  let x0 = (Random.float 1.0 -. 0.5) *. sim.range *. 1.4 in
  let y0 = (Random.float 1.0 -. 0.5) *. sim.range *. 1.4 in
  let val0 = eval_func sim.func_name x0 y0 in
  let init_pt = { x = x0; y = y0; val_ = val0 } in
  sim.points <- [init_pt];
  sim.incumbent <- Some init_pt;
  sim.tr_cx <- x0;
  sim.tr_cy <- y0

let step_sim sim =
  sim.round <- sim.round + 1;
  let cx = sim.tr_cx in
  let cy = sim.tr_cy in
  let l = sim.tr_len in

  let batch = ref [] in
  let best_in_batch = ref None in

  for _ = 1 to 4 do
    let rx = (Random.float 1.0 -. 0.5) *. l in
    let ry = (Random.float 1.0 -. 0.5) *. l in
    let px = clamp (-.sim.range) sim.range (cx +. rx) in
    let py = clamp (-.sim.range) sim.range (cy +. ry) in
    let v = eval_func sim.func_name px py in
    let pt = { x = px; y = py; val_ = v } in
    batch := pt :: !batch;
    sim.points <- pt :: sim.points;

    match !best_in_batch with
    | None -> best_in_batch := Some pt
    | Some cur -> if v > cur.val_ then best_in_batch := Some pt
  done;

  sim.candidates <- !batch;

  match (!best_in_batch, sim.incumbent) with
  | (Some cand, Some inc) ->
      if cand.val_ > inc.val_ then begin
        sim.incumbent <- Some cand;
        sim.tr_cx <- cand.x;
        sim.tr_cy <- cand.y;
        sim.succ_count <- sim.succ_count + 1;
        sim.fail_count <- 0;
        sim.pulse_color <- "green";
        sim.pulse_anim <- 1.0;

        if sim.succ_count >= sim.succ_tol then begin
          sim.tr_len <- min (sim.tr_len *. 1.4) 2.0;
          sim.succ_count <- 0;
          sim.status <- "Expanded (Success)"
        end else
          sim.status <- "Improvement!"
      end else begin
        sim.fail_count <- sim.fail_count + 1;
        sim.succ_count <- 0;
        sim.pulse_color <- "amber";
        sim.pulse_anim <- 0.8;

        if sim.fail_count >= sim.fail_tol then begin
          sim.tr_len <- max (sim.tr_len *. 0.5) 0.08;
          sim.fail_count <- 0;
          sim.status <- "Contracted (Stalled)";

          if sim.tr_len <= 0.09 then begin
            sim.tr_len <- 1.0;
            sim.status <- "Restarted"
          end
        end else
          sim.status <- "No improvement"
      end
  | _ -> ()

let draw_sim sim canvas ctx =
  let w = float_of_int canvas##.width in
  let h = float_of_int canvas##.height in
  let is_dark =
    document##.documentElement##getAttribute (js "data-theme")
    |> Js.Opt.to_option
    |> (function Some s -> Js.to_string s = "dark" | None -> false)
  in

  (* Background *)
  ctx##.fillStyle := js (if is_dark then "#0f1319" else "#1a2130");
  fill_rect ctx 0.0 0.0 w h;

  (* Grid lines *)
  ctx##.strokeStyle := js "rgba(255, 255, 255, 0.05)";
  ctx##.lineWidth := jf 1.0;
  ctx##beginPath;
  let r = sim.range in
  let step = 1.0 in
  let rec draw_grid i =
    if i <= r then begin
      let (gx0, gy0) = to_canvas sim w h i (-.r) in
      let (gx1, gy1) = to_canvas sim w h i r in
      move_to ctx gx0 gy0;
      line_to ctx gx1 gy1;
      let (hx0, hy0) = to_canvas sim w h (-.r) i in
      let (hx1, hy1) = to_canvas sim w h r i in
      move_to ctx hx0 hy0;
      line_to ctx hx1 hy1;
      draw_grid (i +. step)
    end
  in
  draw_grid (-.r);
  ctx##stroke;

  (* Optimum target *)
  let (opt_x, opt_y) = to_canvas sim w h 0.0 0.0 in
  ctx##.strokeStyle := js "rgba(255, 255, 255, 0.15)";
  ctx##beginPath;
  arc ctx opt_x opt_y 6.0 0.0 (2.0 *. Float.pi);
  ctx##stroke;

  (* Evaluated historical points *)
  List.iter (fun pt ->
    let (px, py) = to_canvas sim w h pt.x pt.y in
    ctx##.fillStyle := js "rgba(100, 160, 240, 0.5)";
    ctx##beginPath;
    arc ctx px py 3.5 0.0 (2.0 *. Float.pi);
    ctx##fill
  ) sim.points;

  (* Trust Region Bounding Box *)
  let cx = sim.tr_cx in
  let cy = sim.tr_cy in
  let l = sim.tr_len in
  let (rx0, ry0) = to_canvas sim w h (cx -. (l /. 2.0)) (cy +. (l /. 2.0)) in
  let (rx1, ry1) = to_canvas sim w h (cx +. (l /. 2.0)) (cy -. (l /. 2.0)) in
  let rw = rx1 -. rx0 in
  let rh = ry1 -. ry0 in

  let (box_stroke, box_fill) =
    if sim.pulse_anim > 0.0 then begin
      let alpha = sim.pulse_anim in
      sim.pulse_anim <- sim.pulse_anim *. 0.92;
      if sim.pulse_color = "green" then
        (Printf.sprintf "rgba(16, 185, 129, %.2f)" (0.4 +. (0.6 *. alpha)),
         Printf.sprintf "rgba(16, 185, 129, %.2f)" (0.05 +. (0.15 *. alpha)))
      else
        (Printf.sprintf "rgba(245, 158, 11, %.2f)" (0.4 +. (0.6 *. alpha)),
         Printf.sprintf "rgba(245, 158, 11, %.2f)" (0.05 +. (0.15 *. alpha)))
    end else
      ("rgba(56, 189, 248, 0.7)", "rgba(56, 189, 248, 0.08)")
  in

  ctx##.fillStyle := js box_fill;
  fill_rect ctx rx0 ry0 rw rh;
  ctx##.strokeStyle := js box_stroke;
  ctx##.lineWidth := jf 2.0;
  stroke_rect ctx rx0 ry0 rw rh;

  (* Candidate beams from incumbent *)
  (match sim.incumbent with
  | Some inc ->
      let (ix, iy) = to_canvas sim w h inc.x inc.y in
      List.iter (fun c ->
        let (cx, cy) = to_canvas sim w h c.x c.y in
        ctx##.strokeStyle := js "rgba(56, 189, 248, 0.25)";
        ctx##.lineWidth := jf 1.0;
        ctx##beginPath;
        move_to ctx ix iy;
        line_to ctx cx cy;
        ctx##stroke;

        ctx##.fillStyle := js "#38bdf8";
        ctx##beginPath;
        arc ctx cx cy 4.0 0.0 (2.0 *. Float.pi);
        ctx##fill
      ) sim.candidates
  | None -> ());

  (* Incumbent gold target *)
  (match sim.incumbent with
  | Some inc ->
      let (ix, iy) = to_canvas sim w h inc.x inc.y in
      ctx##.strokeStyle := js "#fbbf24";
      ctx##.lineWidth := jf 2.0;
      ctx##beginPath;
      arc ctx ix iy 7.0 0.0 (2.0 *. Float.pi);
      ctx##stroke;

      ctx##.fillStyle := js "#f59e0b";
      ctx##beginPath;
      arc ctx ix iy 3.5 0.0 (2.0 *. Float.pi);
      ctx##fill
  | None -> ())

let update_telemetry sim =
  (match Js.Opt.to_option (document##getElementById (js "stat-round")) with
  | Some el -> el##.innerText := js (string_of_int sim.round)
  | None -> ());
  (match Js.Opt.to_option (document##getElementById (js "stat-reward")) with
  | Some el ->
      let s = match sim.incumbent with Some inc -> Printf.sprintf "%.3f" inc.val_ | None -> "-" in
      el##.innerText := js s
  | None -> ());
  (match Js.Opt.to_option (document##getElementById (js "stat-length")) with
  | Some el -> el##.innerText := js (Printf.sprintf "%.3f" sim.tr_len)
  | None -> ());
  (match Js.Opt.to_option (document##getElementById (js "stat-status")) with
  | Some el ->
      el##.innerText := js sim.status;
      let cls =
        if String.length sim.status >= 8 && String.sub sim.status 0 8 = "Expanded" then "sim-stat-val status-good"
        else if String.length sim.status >= 10 && String.sub sim.status 0 10 = "Contracted" then "sim-stat-val status-warn"
        else "sim-stat-val"
      in
      el##.className := js cls
  | None -> ())

(* --- Init Simulator --- *)
let init_simulator () =
  match (Js.Opt.to_option (document##getElementById (js "sim-canvas")),
         Js.Opt.to_option (document##getElementById (js "sim-play-btn")),
         Js.Opt.to_option (document##getElementById (js "sim-step-btn")),
         Js.Opt.to_option (document##getElementById (js "sim-reset-btn")),
         Js.Opt.to_option (document##getElementById (js "sim-func-select"))) with
  | (Some canvas_raw, Some play_btn, Some step_btn, Some reset_btn, Some func_sel) ->
      let canvas = match Js.Opt.to_option (CoerceTo.canvas canvas_raw) with Some c -> c | None -> assert false in
      let ctx = canvas##getContext _2d_ in
      let func_select = match Js.Opt.to_option (CoerceTo.select func_sel) with Some s -> s | None -> assert false in

      let sim = {
        func_name = "ackley";
        range = 2.5;
        points = [];
        incumbent = None;
        tr_cx = 0.0;
        tr_cy = 0.0;
        tr_len = 1.0;
        succ_count = 0;
        fail_count = 0;
        succ_tol = 3;
        fail_tol = 4;
        round = 0;
        candidates = [];
        status = "Ready";
        running = false;
        pulse_anim = 0.0;
        pulse_color = "green";
      } in

      let redraw () =
        draw_sim sim canvas ctx;
        update_telemetry sim
      in

      reset_sim sim;
      redraw ();

      let set_play_text running =
        (match Js.Opt.to_option (document##getElementById (js "sim-play-text")) with
        | Some el -> el##.innerText := js (if running then "Pause" else "Run")
        | None -> ());
        (match Js.Opt.to_option (document##getElementById (js "sim-play-icon")) with
        | Some el -> el##.innerHTML := js (if running then "&#10074;&#10074;" else "&#9654;")
        | None -> ())
      in

      add_click play_btn (fun () ->
        sim.running <- not sim.running;
        set_play_text sim.running;
        if sim.running then play_btn##.classList##remove (js "primary")
        else play_btn##.classList##add (js "primary")
      );

      add_click step_btn (fun () ->
        if sim.running then begin
          sim.running <- false;
          set_play_text false;
          play_btn##.classList##add (js "primary")
        end;
        step_sim sim;
        redraw ()
      );

      add_click reset_btn (fun () ->
        reset_sim sim;
        redraw ()
      );

      let on_change = handler (fun _ev ->
        let sel_val = Js.to_string func_select##.value in
        sim.func_name <- sel_val;
        reset_sim sim;
        redraw ();
        Js._true
      ) in
      let _ = addEventListener func_select Event.change on_change Js._false in

      let last_step = ref 0.0 in
      let rec anim_loop (timestamp : Js.number Js.t) =
        let t = Js.to_float timestamp in
        if sim.running && (t -. !last_step > 420.0) then begin
          step_sim sim;
          redraw ();
          last_step := t
        end else if sim.pulse_anim > 0.05 then begin
          draw_sim sim canvas ctx
        end;
        let cb = Js.wrap_callback anim_loop in
        let _ = window##requestAnimationFrame cb in ()
      in
      let cb = Js.wrap_callback anim_loop in
      let _ = window##requestAnimationFrame cb in ()
  | _ -> ()

(* --- Copy Buttons --- *)
let init_copy_buttons () =
  let buttons = query_all ".copy-btn" in
  List.iter (fun btn ->
    add_click btn (fun () ->
      let text =
        match Js.Opt.to_option (btn##closest (js ".install-box")) with
        | Some _ ->
            (match Js.Opt.to_option (document##getElementById (js "install-cmd-text")) with
            | Some el -> Js.to_string el##.innerText
            | None -> "")
        | None ->
            (match Js.Opt.to_option (btn##closest (js ".code-panel")) with
            | Some panel ->
                (match Js.Opt.to_option (panel##querySelector (js "code")) with
                | Some c -> Js.to_string c##.innerText
                | None -> Js.to_string panel##.innerText)
            | None ->
                (match Js.Opt.to_option (btn##closest (js "pre")) with
                | Some pre ->
                    (match Js.Opt.to_option (pre##querySelector (js "code")) with
                    | Some c -> Js.to_string c##.innerText
                    | None -> Js.to_string pre##.innerText)
                | None -> ""))
      in
      try
        let clip = Js.Unsafe.global##.navigator##.clipboard in
        let _ = clip##writeText (js text) in
        btn##.innerText := js "Copied \u{2713}";
        btn##.classList##add (js "copied");
        set_timeout (fun () ->
          btn##.innerText := js "Copy";
          btn##.classList##remove (js "copied")
        ) 2000.0
      with _ -> ()
    )
  ) buttons

(* --- Code Hub Tabs --- *)
let init_code_tabs () =
  let tab_btns = query_all ".code-tab-btn" in
  let panels = query_all ".code-panel" in
  List.iter (fun btn ->
    add_click btn (fun () ->
      let target_id =
        btn##getAttribute (js "data-tab")
        |> Js.Opt.to_option
        |> (function Some s -> Js.to_string s | None -> "")
      in
      List.iter (fun b -> b##.classList##remove (js "active")) tab_btns;
      List.iter (fun p -> p##.classList##remove (js "active")) panels;
      btn##.classList##add (js "active");
      match Js.Opt.to_option (document##getElementById (js target_id)) with
      | Some p -> p##.classList##add (js "active")
      | None -> ()
    )
  ) tab_btns

(* --- Install Switcher --- *)
let init_install_switcher () =
  let pills = query_all ".install-pill" in
  let install_cmd = document##getElementById (js "install-cmd-text") in
  let install_details = document##getElementById (js "install-details") in

  let set_content cmd badge desc =
    (match Js.Opt.to_option install_cmd with
    | Some el -> el##.innerText := js cmd
    | None -> ());
    (match Js.Opt.to_option install_details with
    | Some el ->
        el##.innerHTML := js (Printf.sprintf "<span class=\"install-badge %s\">%s</span><span>%s</span>" badge badge desc)
    | None -> ())
  in

  List.iter (fun pill ->
    add_click pill (fun () ->
      List.iter (fun p -> p##.classList##remove (js "active")) pills;
      pill##.classList##add (js "active");
      let plat =
        pill##getAttribute (js "data-platform")
        |> Js.Opt.to_option
        |> (function Some s -> Js.to_string s | None -> "")
      in
      match plat with
      | "linux" ->
          set_content "pip install baux" "badge-cuda" "Pre-compiled manylinux wheel with Tensor Core acceleration for NVIDIA GPUs."
      | "cpu" ->
          set_content "pip install baux" "badge-cpu" "Pure-Rust native fallback using AVX-512 and NEON SIMD routines."
      | _ ->
          set_content "pip install baux" "badge-metal" "Pre-compiled wheel with unified memory acceleration for Apple Silicon (M1\u{2013}M4)."
    )
  ) pills

(* --- Theme Toggle --- *)
let init_theme_toggle () =
  match (Js.Opt.to_option (document##getElementById (js "theme-toggle-btn")),
         Js.Opt.to_option (document##getElementById (js "theme-toggle-label"))) with
  | (Some btn, Some label) ->
      add_click btn (fun () ->
        let cur =
          document##.documentElement##getAttribute (js "data-theme")
          |> Js.Opt.to_option
          |> (function Some s -> Js.to_string s | None -> "light")
        in
        let nxt = if cur = "light" then "dark" else "light" in
        document##.documentElement##setAttribute (js "data-theme") (js nxt);
        label##.innerText := js (if nxt = "light" then "Dark" else "Light")
      )
  | _ -> ()

(* --- Main Entrypoint --- *)
let () =
  let on_load = handler (fun _ev ->
    init_theme_toggle ();
    init_copy_buttons ();
    init_code_tabs ();
    init_install_switcher ();
    init_simulator ();
    Js._true
  ) in
  window##.onload := on_load
