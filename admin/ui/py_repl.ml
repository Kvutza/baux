(* admin/ui/py_repl.ml
   Interactive in-browser Python REPL & inline runnable snippet controller.
   Talks to self-hosted stripped Pyodide via dedicated Web Worker (assets/repl_worker.js).
   Zero innerHTML / outerHTML / eval_string; pure DOM / createElementNS construction.
*)

[@@@warning "-32-27-33"]

open Js_of_ocaml
open Dom_html

let js = Js.string
let to_s = Js.to_string
let svg_ns = "http://www.w3.org/2000/svg"

type preset = {
  id : string;
  label : string;
  description : string;
  code : string;
}

let presets : preset list =
  [
    {
      id = "quickstart";
      label = "Quickstart: 2D TuRBO-ENN";
      description = "Run 8 ask/tell iterations of TuRBO-ENN on a shifted sphere.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])\n"
        ^ "state = baux.init(bounds, baux.turbo_enn(num_init=4), seed=0)\n\n"
        ^ "for step in range(8):\n"
        ^ "    state, batch = baux.ask(state, 2)\n"
        ^ "    y = -np.sum((batch.points - np.array([0.8, -0.5])) ** 2, axis=-1, keepdims=True)\n"
        ^ "    state, _ = baux.tell(state, batch, y)\n"
        ^ "    print(f\"step {step + 1}: n={state.num_observations} best_y={state.best_y:.5f}\")\n\n"
        ^ "print(\"best_x:\", np.round(state.best_x, 4))\n";
    };
    {
      id = "pytree";
      label = "Structured PyTree + Log & Categorical";
      description = "Mixed continuous, log-scale, and categorical search space via PyTreeSpec.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "spec = baux.pytree_bounds(\n"
        ^ "    {\n"
        ^ "        \"lr\": baux.Log(1e-4, 1e-1),\n"
        ^ "        \"depth\": (),\n"
        ^ "        \"opt\": baux.Categorical([\"adamw\", \"muon\", \"sgd\"]),\n"
        ^ "    },\n"
        ^ "    low={\"depth\": 2.0},\n"
        ^ "    high={\"depth\": 12.0},\n"
        ^ ")\n"
        ^ "state = baux.init(spec, baux.turbo_enn(num_init=4), seed=7)\n\n"
        ^ "for _ in range(8):\n"
        ^ "    state, batch = baux.ask(state, 2)\n"
        ^ "    scores = []\n"
        ^ "    for p in batch.params:\n"
        ^ "        opt_bonus = 0.25 if p[\"opt\"] == \"muon\" else 0.0\n"
        ^ "        lr_pen = (np.log10(float(p[\"lr\"])) + 2.3) ** 2\n"
        ^ "        d_pen = 0.05 * (float(p[\"depth\"]) - 6.0) ** 2\n"
        ^ "        scores.append([opt_bonus - lr_pen - d_pen])\n"
        ^ "    state, _ = baux.tell(state, batch, np.array(scores))\n\n"
        ^ "print(\"best_params:\", state.best_params)\n"
        ^ "print(\"best_y:\", round(float(state.best_y), 6))\n";
    };
    {
      id = "oxcaml_ackley";
      label = "OxCaml 16D Ackley Benchmark";
      description = "Evaluate the compiled-formula OxCaml 16D Ackley benchmark under TuRBO-ENN vs TuRBO-Zero.";
      code =
        "import numpy as np\n"
        ^ "import baux\n"
        ^ "from baux.benchmarks import oxcaml\n\n"
        ^ "desc = oxcaml.describe(\"ackley\", 16)\n"
        ^ "bounds = np.array(desc[\"bounds\"], dtype=float)\n"
        ^ "state = baux.init(bounds, baux.turbo_enn(num_init=6), seed=42)\n\n"
        ^ "for step in range(10):\n"
        ^ "    state, batch = baux.ask(state, 4)\n"
        ^ "    # Negate because baux maximizes objective values\n"
        ^ "    y = -oxcaml.eval_batch(\"ackley\", batch.points).reshape(-1, 1)\n"
        ^ "    state, _ = baux.tell(state, batch, y)\n\n"
        ^ "print(f\"Catalogue functions available: {len(oxcaml.catalogue())}\")\n"
        ^ "print(f\"Best Ackley-16D value (max -f): {state.best_y:.5f} after {state.num_observations} evals\")\n";
    };
    {
      id = "enn_draws";
      label = "Functional ENN Posterior & Draws";
      description = "Fit Epistemic Nearest Neighbors, decompose epistemic vs aleatoric variance, and draw function samples.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "rng = np.random.default_rng(7)\n"
        ^ "basin_x = 0.6 + 0.04 * rng.normal(size=(24, 6))\n"
        ^ "shell_x = rng.uniform(-1.5, 1.5, size=(24, 6))\n"
        ^ "train_x = np.vstack([basin_x, shell_x])\n"
        ^ "signal = np.exp(-1.5 * np.sum((train_x - 0.6) ** 2, axis=1, keepdims=True))\n"
        ^ "train_yvar = np.vstack([np.full((24, 1), 5e-4), np.full((24, 1), 5e-2)])\n"
        ^ "train_y = signal + np.sqrt(train_yvar) * rng.normal(size=(48, 1))\n\n"
        ^ "k_fit, k_draw = baux.split(baux.key(11), 2)\n"
        ^ "enn_state = baux.with_geometry(baux.enn_init(train_x, train_y, train_yvar, scale_x=True), \"hyperbolic\", local_k=4)\n"
        ^ "params = baux.enn_fit(enn_state, k=8, infer_aleatoric=False, key=k_fit)\n"
        ^ "queries = np.vstack([np.full((1, 6), 0.600), np.full((1, 6), 0.601), np.full((1, 6), 2.400)])\n"
        ^ "post = baux.enn_posterior(enn_state, queries, params, flags=baux.enn.PosteriorFlags(observation_noise=True))\n"
        ^ "draws, idx = baux.enn_sample(enn_state, queries, params, key=k_draw, num_draws=64)\n\n"
        ^ "corr = float(np.corrcoef(draws[0, 0], draws[1, 0])[0, 1])\n"
        ^ "print(f\"basin se_epi={post.se_epi[0, 0]:.4f} vs far se_epi={post.se_epi[2, 0]:.4f}\")\n"
        ^ "print(f\"neighbor draw correlation (0.600 vs 0.601): {corr:.4f}\")\n";
    };
    {
      id = "checkpoint";
      label = "Checkpoints: freeze / thaw / async";
      description = "Freeze optimizer state to binary bytes, verify BLAKE3 digest, thaw cleanly, and test cancel/complete.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "domain = np.array([[-2.0, 2.0], [-2.0, 2.0]])\n"
        ^ "blank = baux.init(domain, baux.turbo_enn(num_init=2), seed=123)\n"
        ^ "state, b1 = baux.ask(blank, 2)\n"
        ^ "state, _ = baux.tell(state, b1, -np.sum(b1.points ** 2, axis=-1, keepdims=True))\n\n"
        ^ "blob = baux.freeze(state)\n"
        ^ "restored = baux.thaw(blank, blob)\n"
        ^ "assert restored.epoch == state.epoch and restored.digest == state.digest\n"
        ^ "print(f\"Frozen checkpoint bytes={len(blob)} digest={state.digest[:16]}...\")\n"
        ^ "print(\"Reproducibility matches:\", np.array_equal(baux.ask(state, 2)[1].points, baux.ask(restored, 2)[1].points))\n";
    };
    {
      id = "turbo_gp";
      label = "TuRBO-1 GP vs Simplex Manifold";
      description = "Run Matern-5/2 Exact GP surrogate with Simplex + Periodic parameters in browser WebAssembly.";
      code =
        "import numpy as np\n"
        ^ "import baux\n\n"
        ^ "spec = baux.pytree_bounds({\n"
        ^ "    \"weights\": baux.Simplex(4),\n"
        ^ "    \"phase\": baux.Periodic(0.0, 2.0 * np.pi),\n"
        ^ "})\n"
        ^ "state = baux.init(spec, baux.turbo_one(num_init=4), seed=11)\n\n"
        ^ "for _ in range(7):\n"
        ^ "    state, batch = baux.ask(state, 2)\n"
        ^ "    ys = []\n"
        ^ "    for item in batch.params:\n"
        ^ "        w = np.asarray(item[\"weights\"])\n"
        ^ "        y = -float(np.sum((w - np.array([0.5, 0.3, 0.15, 0.05])) ** 2)) + 0.1 * float(np.cos(item[\"phase\"] - 1.2))\n"
        ^ "        ys.append([y])\n"
        ^ "    state, _ = baux.tell(state, batch, np.array(ys))\n\n"
        ^ "print(\"capabilities[0]:\", baux.capabilities()[0])\n"
        ^ "print(\"best_y:\", round(float(state.best_y), 6), \"best_params:\", state.best_params)\n";
    };
  ]

let find_preset id =
  List.find_opt (fun p -> p.id = id) presets

let trim_s s = String.trim s

let contains_sub hay needle =
  let hl = String.length hay in
  let nl = String.length needle in
  if nl = 0 then true
  else if nl > hl then false
  else
    let rec loop i =
      if i > hl - nl then false
      else if String.sub hay i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let get_base_url () =
  match Js.Opt.to_option (document##.documentElement##getAttribute (js "data-base-url")) with
  | Some v -> to_s v
  | None -> "./"

let mk ?cls ?txt tag =
  let e = document##createElement (js tag) in
  (match cls with Some c -> e##.className := js c | None -> ());
  (match txt with Some t -> e##.textContent := Js.some (js t) | None -> ());
  e

let set_text (e : #Dom.node Js.t) (s : string) =
  e##.textContent := Js.some (js s)

let svg_el tag attrs =
  let e : element Js.t =
    Js.Unsafe.meth_call document "createElementNS"
      [| Js.Unsafe.inject (js svg_ns); Js.Unsafe.inject (js tag) |]
  in
  List.iter (fun (k, v) -> e##setAttribute (js k) (js v)) attrs;
  e

let clear_children (parent : #Dom.node Js.t) =
  let rec loop () =
    match Js.Opt.to_option parent##.firstChild with
    | None -> ()
    | Some ch ->
        ignore (parent##removeChild ch);
        loop ()
  in
  loop ()

(* ---------------------------------------------------------------------------
   Singleton lazy Web Worker kernel controller
   --------------------------------------------------------------------------- *)

type kernel_state = Idle | Booting of string | Ready | Running | Interrupted of string

let worker_ref : Js.Unsafe.any option ref = ref None
let kernel_state_ref : kernel_state ref = ref Idle
let status_listeners : (kernel_state -> unit) list ref = ref []
let pending_reqs : (int, Js.Unsafe.any -> unit) Hashtbl.t = Hashtbl.create 16
let next_req_id = ref 1

let notify_status st =
  kernel_state_ref := st;
  List.iter (fun fn -> fn st) !status_listeners

let add_status_listener fn =
  status_listeners := fn :: !status_listeners;
  fn !kernel_state_ref

let interrupt_kernel () =
  (match !worker_ref with
  | Some w ->
      ignore (Js.Unsafe.meth_call w "terminate" [||]);
      worker_ref := None
  | None -> ());
  let callbacks = Hashtbl.fold (fun _ v acc -> v :: acc) pending_reqs [] in
  Hashtbl.clear pending_reqs;
  List.iter
    (fun cb ->
      let err_obj =
        Js.Unsafe.obj
          [|
            ("ok", Js.Unsafe.inject Js._false);
            ("stdout", Js.Unsafe.inject (js ""));
            ("stderr", Js.Unsafe.inject (js "Kernel interrupted by user. Click Run to start a fresh kernel."));
            ("incomplete", Js.Unsafe.inject Js._false);
          |]
      in
      cb err_obj)
    callbacks;
  notify_status (Interrupted "Kernel interrupted")

let ensure_worker () =
  match !worker_ref with
  | Some w -> w
  | None ->
      let worker_ctor = Js.Unsafe.get Dom_html.window (js "Worker") in
      let url = get_base_url () ^ "assets/repl_worker.js" in
      let w = Js.Unsafe.new_obj worker_ctor [| Js.Unsafe.inject (js url) |] in
      notify_status (Booting "Starting Web Worker...");
      let onmsg =
        Dom.handler (fun ev ->
            let data = Js.Unsafe.get ev (js "data") in
            let typ =
              if Js.Optdef.test (Js.Unsafe.get data (js "type")) then
                to_s (Js.Unsafe.get data (js "type"))
              else ""
            in
            if typ = "status" then (
              let detail = to_s (Js.Unsafe.get data (js "detail")) in
              let stage = to_s (Js.Unsafe.get data (js "stage")) in
              if stage = "ready" then notify_status Ready
              else notify_status (Booting detail))
            else (
              let id_any = Js.Unsafe.get data (js "id") in
              if Js.Opt.test id_any then (
                let id_int = int_of_float (Js.float_of_number id_any) in
                match Hashtbl.find_opt pending_reqs id_int with
                | Some cb ->
                    Hashtbl.remove pending_reqs id_int;
                    if !kernel_state_ref = Running then notify_status Ready;
                    cb data
                | None -> ()));
            Js._true)
      in
      let onerr =
        Dom.handler (fun ev ->
            let msg =
              if Js.Optdef.test (Js.Unsafe.get ev (js "message")) then
                to_s (Js.Unsafe.get ev (js "message"))
              else "Worker error"
            in
            notify_status (Interrupted ("Worker failed: " ^ msg));
            Js._true)
      in
      Js.Unsafe.set w (js "onmessage") onmsg;
      Js.Unsafe.set w (js "onerror") onerr;
      worker_ref := Some w;
      w

let send_worker typ fields cb =
  let w = ensure_worker () in
  let req_id = !next_req_id in
  next_req_id := req_id + 1;
  Hashtbl.add pending_reqs req_id cb;
  if !kernel_state_ref = Ready then notify_status Running;
  let payload =
    Js.Unsafe.obj
      (Array.of_list
         (("type", Js.Unsafe.inject (js typ))
         :: ("id", Js.Unsafe.inject req_id)
         :: fields))
  in
  ignore (Js.Unsafe.meth_call w "postMessage" [| Js.Unsafe.inject payload |])

let warm_kernel () =
  send_worker "init" [] (fun _ -> ())

let status_text = function
  | Idle -> "Idle (loads ~16 MB WASM on first Run)"
  | Booting d -> d
  | Ready -> "Python 3.12 + NumPy + BAUX Ready"
  | Running -> "Executing Python in Web Worker..."
  | Interrupted r -> r

(* ---------------------------------------------------------------------------
   Live OptimizerState Summary & SVG Convergence Trajectory Renderer
   --------------------------------------------------------------------------- *)

let render_mini_svg_chart (y_vals : float list) (best_vals : float list) =
  let n = List.length best_vals in
  let w = 420.0 in
  let h = 164.0 in
  let pl = 46.0 in
  let pr = 14.0 in
  let pt = 14.0 in
  let pb = 26.0 in
  let pw = w -. pl -. pr in
  let ph = h -. pt -. pb in
  let svg =
    svg_el "svg"
      [
        ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
        ("class", "py-traj-svg");
        ("role", "img");
        ("aria-label", "Cumulative best objective trajectory");
      ]
  in
  if n = 0 then svg
  else
    let all_vals = y_vals @ best_vals in
    let lo = List.fold_left min (List.hd all_vals) all_vals in
    let hi = List.fold_left max (List.hd all_vals) all_vals in
    let span = if hi -. lo < 1e-9 then 1.0 else hi -. lo in
    let lo_p = lo -. 0.06 *. span in
    let hi_p = hi +. 0.06 *. span in
    let sp = hi_p -. lo_p in
    let x_px i =
      if n <= 1 then pl +. 0.5 *. pw
      else pl +. (float_of_int i /. float_of_int (n - 1)) *. pw
    in
    let y_px v = pt +. (1.0 -. ((v -. lo_p) /. sp)) *. ph in
    (* Grid lines & Y ticks *)
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let yy = pt +. (1.0 -. frac) *. ph in
      let v = lo_p +. frac *. sp in
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.1f" pl);
             ("x2", Printf.sprintf "%.1f" (w -. pr));
             ("y1", Printf.sprintf "%.1f" yy);
             ("y2", Printf.sprintf "%.1f" yy);
             ("class", "py-svg-grid");
           ]);
      let lbl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" (pl -. 6.0));
            ("y", Printf.sprintf "%.1f" (yy +. 3.5));
            ("text-anchor", "end");
            ("class", "py-svg-tick");
          ]
      in
      lbl##.textContent := Js.some (js (Printf.sprintf "%.2g" v));
      Dom.appendChild svg lbl
    done;
    (* Axes *)
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" (w -. pr));
           ("y1", Printf.sprintf "%.1f" (h -. pb));
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" pl);
           ("y1", Printf.sprintf "%.1f" pt);
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    (* Raw observations points *)
    List.iteri
      (fun i v ->
        Dom.appendChild svg
          (svg_el "circle"
             [
               ("cx", Printf.sprintf "%.2f" (x_px i));
               ("cy", Printf.sprintf "%.2f" (y_px v));
               ("r", "2.4");
               ("class", "py-svg-obs-dot");
             ]))
      y_vals;
    (* Best-so-far polyline *)
    let pts =
      String.concat " "
        (List.mapi
           (fun i v -> Printf.sprintf "%.2f,%.2f" (x_px i) (y_px v))
           best_vals)
    in
    Dom.appendChild svg
      (svg_el "polyline"
         [
           ("fill", "none");
           ("points", pts);
           ("class", "py-svg-best-line");
         ]);
    let last_x = x_px (n - 1) in
    let last_y = y_px (List.nth best_vals (n - 1)) in
    Dom.appendChild svg
      (svg_el "circle"
         [
           ("cx", Printf.sprintf "%.2f" last_x);
           ("cy", Printf.sprintf "%.2f" last_y);
           ("r", "3.8");
           ("class", "py-svg-best-dot");
         ]);
    let xlbl =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (pl +. 0.5 *. pw));
          ("y", Printf.sprintf "%.1f" (h -. 5.0));
          ("text-anchor", "middle");
          ("class", "py-svg-tick");
        ]
    in
    xlbl##.textContent := Js.some (js (Printf.sprintf "Evaluations (1 .. %d)" n));
    Dom.appendChild svg xlbl;
    svg

let is_defined (v : Js.Unsafe.any) : bool =
  Js.Optdef.test (Obj.magic v : _ Js.optdef) && Js.Opt.test (Obj.magic v : _ Js.opt)

let json_num_list (raw : Js.Unsafe.any) =
  if not (is_defined raw) then []
  else
    let arr : Js.Unsafe.any Js.js_array Js.t = Obj.magic raw in
    let len = arr##.length in
    let out = ref [] in
    for i = 0 to len - 1 do
      let item = Js.array_get arr i in
      match Js.Optdef.to_option item with
      | Some v when is_defined v -> out := Js.float_of_number (Obj.magic v) :: !out
      | _ -> ()
    done;
    List.rev !out

let opt_str obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (to_s (Obj.magic v)) else None

let opt_int obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (int_of_float (Js.float_of_number (Obj.magic v))) else None

let opt_float obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (Js.float_of_number (Obj.magic v)) else None

let render_state_panel (panel : element Js.t) (state_raw : Js.Unsafe.any) =
  clear_children (panel :> Dom.node Js.t);
  if not (is_defined state_raw) then (
    let hint =
      mk ~cls:"py-inspector-empty"
        ~txt:
          "Run a script that creates an OptimizerState (for example state = baux.init(...); state, _ = baux.tell(...)) to inspect live convergence trajectories, trust-region widths, and recent evaluations."
        "p"
    in
    Dom.appendChild panel hint)
  else
    let hdr = mk ~cls:"py-inspector-subhdr" "div" in
    let vname = Option.value (opt_str state_raw "var") ~default:"state" in
    Dom.appendChild hdr (mk ~cls:"py-inspector-var" ~txt:("Live variable: " ^ vname) "span");
    Dom.appendChild panel hdr;

    let metrics = mk ~cls:"py-metrics-grid" "div" in
    let add_metric label value =
      let card = mk ~cls:"py-metric-card" "div" in
      Dom.appendChild card (mk ~cls:"py-metric-label" ~txt:label "span");
      Dom.appendChild card (mk ~cls:"py-metric-val" ~txt:value "strong");
      Dom.appendChild metrics card
    in
    add_metric "Observations"
      (string_of_int (Option.value (opt_int state_raw "num_observations") ~default:0));
    add_metric "Epoch" (string_of_int (Option.value (opt_int state_raw "epoch") ~default:0));
    add_metric "Regions" (string_of_int (Option.value (opt_int state_raw "num_regions") ~default:1));
    add_metric "TR length"
      (match opt_float state_raw "tr_length" with Some f -> Printf.sprintf "%.4g" f | None -> "-");
    add_metric "Best y"
      (match opt_float state_raw "best_y" with Some f -> Printf.sprintf "%.5f" f | None -> "-");
    add_metric "Pending" (string_of_int (Option.value (opt_int state_raw "pending_count") ~default:0));
    Dom.appendChild panel metrics;

    (match opt_str state_raw "best_params" with
    | Some bp ->
        let bwrap = mk ~cls:"py-best-params" "div" in
        Dom.appendChild bwrap (mk ~cls:"py-metric-label" ~txt:"best_params" "div");
        Dom.appendChild bwrap (mk ~cls:"py-best-code" ~txt:bp "code");
        Dom.appendChild panel bwrap
    | None -> (
        let bx = json_num_list (Js.Unsafe.get state_raw (js "best_x")) in
        if bx <> [] then (
          let bwrap = mk ~cls:"py-best-params" "div" in
          Dom.appendChild bwrap (mk ~cls:"py-metric-label" ~txt:"best_x" "div");
          let fmt = "[" ^ String.concat ", " (List.map (Printf.sprintf "%.4f") bx) ^ "]" in
          Dom.appendChild bwrap (mk ~cls:"py-best-code" ~txt:fmt "code");
          Dom.appendChild panel bwrap)));

    let y_hist = json_num_list (Js.Unsafe.get state_raw (js "y_history")) in
    let best_hist = json_num_list (Js.Unsafe.get state_raw (js "best_history")) in
    if best_hist <> [] then (
      let cwrap = mk ~cls:"py-chart-card" "div" in
      let clbl = mk ~cls:"py-metric-label" ~txt:"Cumulative best objective" "div" in
      Dom.appendChild cwrap clbl;
      Dom.appendChild cwrap (render_mini_svg_chart y_hist best_hist);
      Dom.appendChild panel cwrap);

    let recent_raw = Js.Unsafe.get state_raw (js "recent") in
    if Js.Optdef.test recent_raw then (
      let rows : Js.Unsafe.any Js.js_array Js.t = Obj.magic recent_raw in
      if rows##.length > 0 then (
        let tbl = mk ~cls:"py-recent-table" "table" in
        let thead = mk "thead" in
        let hr = mk "tr" in
        List.iter (fun h -> Dom.appendChild hr (mk ~txt:h "th")) [ "#"; "y"; "Best"; "x[:4]" ];
        Dom.appendChild thead hr;
        Dom.appendChild tbl thead;
        let tbody = mk "tbody" in
        for i = rows##.length - 1 downto 0 do
          match Js.Optdef.to_option (Js.array_get rows i) with
          | None -> ()
          | Some r ->
              let tr = mk "tr" in
              let obs_n = string_of_int (Option.value (opt_int r "obs") ~default:0) in
              let yv = match opt_float r "y" with Some f -> Printf.sprintf "%.4f" f | None -> "-" in
              let bv = match opt_float r "best_so_far" with Some f -> Printf.sprintf "%.4f" f | None -> "-" in
              let xs = json_num_list (Js.Unsafe.get r (js "x")) in
              let xs_cut =
                if List.length xs <= 4 then xs
                else [ List.nth xs 0; List.nth xs 1; List.nth xs 2; List.nth xs 3 ]
              in
              let xstr = "[" ^ String.concat ", " (List.map (Printf.sprintf "%.2f") xs_cut) ^ "]" in
              List.iter (fun cell -> Dom.appendChild tr (mk ~txt:cell "td")) [ obs_n; yv; bv; xstr ];
              Dom.appendChild tbody tr
        done;
        Dom.appendChild tbl tbody;
        Dom.appendChild panel tbl))

(* ---------------------------------------------------------------------------
   Full Playground Workbench UI (#py-playground-root)
   --------------------------------------------------------------------------- *)

let append_console_chunk (out_pre : element Js.t) cls text =
  if String.length text > 0 then (
    let sp = mk ~cls ~txt:text "span" in
    Dom.appendChild out_pre sp;
    out_pre##.scrollTop := Js.float (float_of_int out_pre##.scrollHeight))

let initial_playground_code () =
  let stored =
    try
      let ss = Js.Unsafe.get Dom_html.window (js "sessionStorage") in
      let raw = Js.Unsafe.meth_call ss "getItem" [| Js.Unsafe.inject (js "baux_playground_code") |] in
      if Js.Opt.test raw then (
        ignore (Js.Unsafe.meth_call ss "removeItem" [| Js.Unsafe.inject (js "baux_playground_code") |]);
        Some (to_s raw))
      else None
    with _ -> None
  in
  match stored with
  | Some code -> (None, code)
  | None ->
      let first = List.hd presets in
      (Some first.id, first.code)

let mount_playground (root : element Js.t) =
  clear_children (root :> Dom.node Js.t);
  let shell = mk ~cls:"py-pg-app" "div" in

  (* Top Toolbar *)
  let toolbar = mk ~cls:"py-pg-toolbar" "div" in
  let select_wrap = mk ~cls:"py-pg-select-wrap" "div" in
  let lbl = mk ~cls:"py-pg-label" ~txt:"Preset" "label" in
  let sel = createSelect document in
  sel##.className := js "py-pg-select";
  sel##setAttribute (js "aria-label") (js "Example preset");
  List.iter
    (fun p ->
      let opt = createOption document in
      opt##.value := js p.id;
      opt##.textContent := Js.some (js p.label);
      Dom.appendChild sel opt)
    presets;
  Dom.appendChild select_wrap lbl;
  Dom.appendChild select_wrap sel;

  let btns = mk ~cls:"py-pg-actions" "div" in
  let run_btn = mk ~cls:"py-btn py-btn-primary" ~txt:"Run Script" "button" in
  run_btn##setAttribute (js "type") (js "button");
  let reset_btn = mk ~cls:"py-btn" ~txt:"Reset / Interrupt" "button" in
  reset_btn##setAttribute (js "type") (js "button");
  let clear_btn = mk ~cls:"py-btn" ~txt:"Clear Console" "button" in
  clear_btn##setAttribute (js "type") (js "button");
  let copy_btn = mk ~cls:"py-btn" ~txt:"Copy" "button" in
  copy_btn##setAttribute (js "type") (js "button");
  List.iter (fun b -> Dom.appendChild btns b) [ run_btn; reset_btn; clear_btn; copy_btn ];

  let status_bar = mk ~cls:"py-pg-status-row" "div" in
  let status_dot = mk ~cls:"py-status-dot" "span" in
  let status_lbl = mk ~cls:"py-status-text" ~txt:"Idle" "span" in
  let timing_lbl = mk ~cls:"py-timing-badge" ~txt:"" "span" in
  Dom.appendChild status_bar status_dot;
  Dom.appendChild status_bar status_lbl;
  Dom.appendChild status_bar timing_lbl;

  Dom.appendChild toolbar select_wrap;
  Dom.appendChild toolbar btns;

  let desc_bar = mk ~cls:"py-pg-desc" ~txt:(List.hd presets).description "p" in

  (* Main Split Grid *)
  let grid = mk ~cls:"py-pg-grid" "div" in
  let left_col = mk ~cls:"py-pg-left" "div" in
  let right_col = mk ~cls:"py-pg-right" "div" in

  (* Editor Box *)
  let ed_hdr = mk ~cls:"py-pane-hdr" "div" in
  Dom.appendChild ed_hdr (mk ~txt:"Script Editor (Ctrl/Cmd + Enter to Run)" "span");
  let editor = createTextarea document in
  editor##.className := js "py-pg-editor";
  editor##setAttribute (js "spellcheck") (js "false");
  editor##setAttribute (js "aria-label") (js "Python script editor");
  editor##.rows := 14;

  let (init_preset_id, init_code) = initial_playground_code () in
  editor##.value := js init_code;
  (match init_preset_id with
  | Some pid -> sel##.value := js pid
  | None -> desc_bar##.textContent := Js.some (js "Loaded snippet from documentation page."));

  (* Output + Interactive >>> Console *)
  let con_hdr = mk ~cls:"py-pane-hdr" "div" in
  Dom.appendChild con_hdr (mk ~txt:"Interactive Python 3.12 REPL & Output" "span");
  let console_pre = mk ~cls:"py-pg-console" "pre" in
  append_console_chunk console_pre "py-out-muted"
    "Click 'Run Script' or type Python expressions below (Up/Down for history).\n";

  let prompt_row = createForm document in
  prompt_row##.className := js "py-repl-prompt-row";
  let prompt_glyph = mk ~cls:"py-repl-glyph" ~txt:">>>" "span" in
  let repl_input = createInput ~_type:(js "text") document in
  repl_input##.className := js "py-repl-input";
  repl_input##setAttribute (js "placeholder") (js "Type Python expression (e.g. state.best_y, dir(baux))...");
  repl_input##setAttribute (js "spellcheck") (js "false");
  repl_input##setAttribute (js "autocomplete") (js "off");
  repl_input##setAttribute (js "aria-label") (js "Interactive Python prompt");
  let repl_send = mk ~cls:"py-btn py-btn-sm" ~txt:"Eval" "button" in
  repl_send##setAttribute (js "type") (js "submit");
  Dom.appendChild prompt_row prompt_glyph;
  Dom.appendChild prompt_row repl_input;
  Dom.appendChild prompt_row repl_send;

  Dom.appendChild left_col ed_hdr;
  Dom.appendChild left_col editor;
  Dom.appendChild left_col con_hdr;
  Dom.appendChild left_col console_pre;
  Dom.appendChild left_col prompt_row;

  (* Inspector Pane *)
  let insp_hdr = mk ~cls:"py-pane-hdr" "div" in
  Dom.appendChild insp_hdr (mk ~txt:"Live BAUX OptimizerState Inspector" "span");
  let insp_body = mk ~cls:"py-inspector-body" "div" in
  render_state_panel insp_body (Js.Unsafe.inject Js.undefined);
  Dom.appendChild right_col insp_hdr;
  Dom.appendChild right_col insp_body;

  Dom.appendChild grid left_col;
  Dom.appendChild grid right_col;

  Dom.appendChild shell toolbar;
  Dom.appendChild shell status_bar;
  Dom.appendChild shell desc_bar;
  Dom.appendChild shell grid;
  Dom.appendChild root shell;

  add_status_listener (fun st ->
      status_lbl##.textContent := Js.some (js (status_text st));
      let cls =
        match st with
        | Ready -> "py-status-dot is-ready"
        | Booting _ | Running -> "py-status-dot is-busy"
        | Interrupted _ -> "py-status-dot is-err"
        | Idle -> "py-status-dot"
      in
      status_dot##.className := js cls);

  let run_script () =
    let src = to_s editor##.value in
    clear_children (console_pre :> Dom.node Js.t);
    append_console_chunk console_pre "py-out-muted" ">>> Running script...\n";
    timing_lbl##.textContent := Js.some (js "");
    send_worker "exec"
      [
        ("code", Js.Unsafe.inject (js src));
        ("resetNamespace", Js.Unsafe.inject Js._true);
      ]
      (fun res ->
        let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
        let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
        let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
        let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
        let dt_s =
          if Js.Optdef.test dt_raw then Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw) else ""
        in
        timing_lbl##.textContent := Js.some (js dt_s);
        clear_children (console_pre :> Dom.node Js.t);
        if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
        if err_s <> "" then append_console_chunk console_pre "py-out-stderr" err_s;
        if out_s = "" && err_s = "" && ok then
          append_console_chunk console_pre "py-out-muted" "(completed with no printed output)\n";
        set_text prompt_glyph ">>>";
        render_state_panel insp_body (Js.Unsafe.get res (js "state")))
  in

  let repl_history : string list ref = ref [] in
  let hist_idx : int ref = ref (-1) in

  let push_repl_line line =
    let glyph_now = to_s (Js.Opt.get prompt_glyph##.textContent (fun () -> js ">>>")) in
    append_console_chunk console_pre "py-out-cmd" (glyph_now ^ " " ^ line ^ "\n");
    if trim_s line <> "" then (
      repl_history := line :: !repl_history;
      hist_idx := -1);
    send_worker "repl" [ ("line", Js.Unsafe.inject (js line)) ] (fun res ->
        let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
        let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
        let inc = Js.to_bool (Js.Unsafe.get res (js "incomplete")) in
        let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
        if Js.Optdef.test dt_raw then
          timing_lbl##.textContent := Js.some (js (Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw)));
        if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
        if err_s <> "" then append_console_chunk console_pre "py-out-stderr" err_s;
        set_text prompt_glyph (if inc then "..." else ">>>");
        render_state_panel insp_body (Js.Unsafe.get res (js "state")))
  in

  sel##.onchange :=
    Dom.handler (fun _ ->
        (match find_preset (to_s sel##.value) with
        | Some p ->
            editor##.value := js p.code;
            desc_bar##.textContent := Js.some (js p.description)
        | None -> ());
        Js._true);

  run_btn##.onclick :=
    Dom.handler (fun _ ->
        run_script ();
        Js._true);

  reset_btn##.onclick :=
    Dom.handler (fun _ ->
        interrupt_kernel ();
        set_text prompt_glyph ">>>";
        timing_lbl##.textContent := Js.some (js "");
        warm_kernel ();
        Js._true);

  clear_btn##.onclick :=
    Dom.handler (fun _ ->
        clear_children (console_pre :> Dom.node Js.t);
        append_console_chunk console_pre "py-out-muted" "Console cleared.\n";
        Js._true);

  copy_btn##.onclick :=
    Dom.handler (fun _ ->
        let clip = Js.Unsafe.get Dom_html.window##.navigator (js "clipboard") in
        if is_defined clip then (
          ignore (Js.Unsafe.meth_call clip "writeText" [| Js.Unsafe.inject editor##.value |]);
          copy_btn##.textContent := Js.some (js "Copied");
          ignore
            (Dom_html.window##setTimeout
               (Js.wrap_callback (fun () -> copy_btn##.textContent := Js.some (js "Copy")))
               (Js.float 1200.0)));
        Js._true);

  editor##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let ctrl_or_meta = Js.to_bool ev##.ctrlKey || Js.to_bool ev##.metaKey in
        if ctrl_or_meta && key = "Enter" then (
          Dom.preventDefault ev;
          run_script ();
          Js._false)
        else if key = "Tab" then (
          Dom.preventDefault ev;
          let st = editor##.selectionStart in
          let en = editor##.selectionEnd in
          let v = to_s editor##.value in
          let left = String.sub v 0 st in
          let right = String.sub v en (String.length v - en) in
          editor##.value := js (left ^ "    " ^ right);
          editor##.selectionStart := st + 4;
          editor##.selectionEnd := st + 4;
          Js._false)
        else Js._true);

  prompt_row##.onsubmit :=
    Dom.handler (fun ev ->
        Dom.preventDefault ev;
        let line = to_s repl_input##.value in
        repl_input##.value := js "";
        push_repl_line line;
        Js._false);

  repl_input##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let hlen = List.length !repl_history in
        if key = "ArrowUp" && hlen > 0 then (
          Dom.preventDefault ev;
          let next_i = min (hlen - 1) (!hist_idx + 1) in
          hist_idx := next_i;
          repl_input##.value := js (List.nth !repl_history next_i);
          Js._false)
        else if key = "ArrowDown" then (
          Dom.preventDefault ev;
          if !hist_idx <= 0 then (
            hist_idx := -1;
            repl_input##.value := js "")
          else (
            let next_i = !hist_idx - 1 in
            hist_idx := next_i;
            repl_input##.value := js (List.nth !repl_history next_i));
          Js._false)
        else Js._true);

  (* Automatically warm runtime on the dedicated Playground page *)
  warm_kernel ()

(* ---------------------------------------------------------------------------
   Inline Doc Code Block Runner (.code-wrap inside article)
   --------------------------------------------------------------------------- *)

let is_runnable_baux_code code =
  (contains_sub code "import baux" || contains_sub code "from baux")
  && (not (contains_sub code "..."))
  && (not (contains_sub code "your_"))
  && (not (contains_sub code "oxcaml_step"))

let enhance_inline_blocks () =
  let wraps = document##querySelectorAll (js "article .code-wrap") in
  let len = wraps##.length in
  for i = 0 to len - 1 do
    match Js.Opt.to_option (wraps##item i) with
    | None -> ()
    | Some wrap ->
        let inside_pg =
          Js.Opt.test
            (Js.Unsafe.meth_call wrap "closest" [| Js.Unsafe.inject (js "#py-playground-root") |])
        in
        if (not inside_pg)
           && not (Js.Opt.test (wrap##querySelector (js ".code-run")))
        then
          match Js.Opt.to_option (wrap##querySelector (js "pre")) with
          | None -> ()
          | Some pre ->
              let orig_code = trim_s (to_s (Js.Opt.get pre##.textContent (fun () -> js ""))) in
              if is_runnable_baux_code orig_code then (
                wrap##.className := js (to_s wrap##.className ^ " has-py-run");
                let actions = mk ~cls:"code-actions" "div" in
                let run_btn = mk ~cls:"code-run" ~txt:"Run" "button" in
                run_btn##setAttribute (js "type") (js "button");
                run_btn##setAttribute (js "title") (js "Run live in browser Python REPL");
                let pg_btn = mk ~cls:"code-pg" ~txt:"Playground" "button" in
                pg_btn##setAttribute (js "type") (js "button");
                pg_btn##setAttribute (js "title") (js "Open in interactive Playground");
                Dom.appendChild actions run_btn;
                Dom.appendChild actions pg_btn;
                (match Js.Opt.to_option (wrap##querySelector (js ".code-copy")) with
                | Some copy_btn -> Dom.appendChild actions copy_btn
                | None -> ());
                Dom.appendChild wrap actions;

                let drawer_ref : element Js.t option ref = ref None in
                let editor_ref : textAreaElement Js.t option ref = ref None in
                let out_ref : element Js.t option ref = ref None in
                let meta_ref : element Js.t option ref = ref None in
                let chart_ref : element Js.t option ref = ref None in

                let ensure_drawer () =
                  match !drawer_ref with
                  | Some d ->
                      d##removeAttribute (js "hidden");
                      d
                  | None ->
                      let d = mk ~cls:"py-inline-drawer" "div" in
                      let bar = mk ~cls:"py-inline-bar" "div" in
                      let ed_toggle = mk ~cls:"py-btn py-btn-sm" ~txt:"Edit" "button" in
                      ed_toggle##setAttribute (js "type") (js "button");
                      let rerun_btn = mk ~cls:"py-btn py-btn-sm py-btn-primary" ~txt:"Run" "button" in
                      rerun_btn##setAttribute (js "type") (js "button");
                      let close_btn = mk ~cls:"py-btn py-btn-sm" ~txt:"Close" "button" in
                      close_btn##setAttribute (js "type") (js "button");
                      let meta = mk ~cls:"py-inline-meta" ~txt:"" "span" in
                      List.iter (fun x -> Dom.appendChild bar x) [ rerun_btn; ed_toggle; close_btn; meta ];

                      let ed = createTextarea document in
                      ed##.className := js "py-inline-editor";
                      ed##setAttribute (js "spellcheck") (js "false");
                      ed##setAttribute (js "hidden") (js "hidden");
                      ed##.rows := max 5 (min 14 (List.length (String.split_on_char '\n' orig_code) + 1));
                      ed##.value := js (orig_code ^ "\n");

                      let out = mk ~cls:"py-inline-out" "pre" in
                      let ch = mk ~cls:"py-inline-state" "div" in
                      Dom.appendChild d bar;
                      Dom.appendChild d ed;
                      Dom.appendChild d out;
                      Dom.appendChild d ch;
                      Dom.appendChild wrap d;
                      drawer_ref := Some d;
                      editor_ref := Some ed;
                      out_ref := Some out;
                      meta_ref := Some meta;
                      chart_ref := Some ch;

                      ed_toggle##.onclick :=
                        Dom.handler (fun _ ->
                            if Js.to_bool (ed##hasAttribute (js "hidden")) then (
                              ed##removeAttribute (js "hidden");
                              set_text ed_toggle "Hide Editor")
                            else (
                              ed##setAttribute (js "hidden") (js "hidden");
                              set_text ed_toggle "Edit");
                            Js._true);
                      close_btn##.onclick :=
                        Dom.handler (fun _ ->
                            d##setAttribute (js "hidden") (js "hidden");
                            Js._true);
                      rerun_btn##.onclick :=
                        Dom.handler (fun _ ->
                            let code = to_s ed##.value in
                            set_text meta "Running...";
                            clear_children (out :> Dom.node Js.t);
                            send_worker "exec"
                              [
                                ("code", Js.Unsafe.inject (js code));
                                ("resetNamespace", Js.Unsafe.inject Js._true);
                              ]
                              (fun res ->
                                let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
                                let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
                                let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
                                let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
                                let dt_s =
                                  if Js.Optdef.test dt_raw then
                                    Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw)
                                  else ""
                                in
                                set_text meta
                                  ((if ok then "OK" else "Error")
                                  ^ if dt_s <> "" then " (" ^ dt_s ^ ")" else "");
                                clear_children (out :> Dom.node Js.t);
                                if out_s <> "" then append_console_chunk out "py-out-stdout" out_s;
                                if err_s <> "" then append_console_chunk out "py-out-stderr" err_s;
                                if out_s = "" && err_s = "" && ok then
                                  append_console_chunk out "py-out-muted"
                                    "Assertions passed (no stdout). Try Edit -> print(...) to inspect values.\n";
                                render_state_panel ch (Js.Unsafe.get res (js "state")));
                            Js._true);
                      d
                in

                let trigger_run () =
                  let _d = ensure_drawer () in
                  match (!editor_ref, !out_ref, !meta_ref, !chart_ref) with
                  | Some ed, Some out, Some meta, Some ch ->
                      let src = to_s ed##.value in
                      set_text meta (status_text !kernel_state_ref);
                      clear_children (out :> Dom.node Js.t);
                      append_console_chunk out "py-out-muted" "Running in WebAssembly Python worker...\n";
                      let unsub_ref = ref false in
                      add_status_listener (fun st ->
                          if (not !unsub_ref) && !kernel_state_ref <> Ready then
                            set_text meta (status_text st));
                      send_worker "exec"
                        [
                          ("code", Js.Unsafe.inject (js src));
                          ("resetNamespace", Js.Unsafe.inject Js._true);
                        ]
                        (fun res ->
                          unsub_ref := true;
                          let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
                          let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
                          let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
                          let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
                          let dt_s =
                            if Js.Optdef.test dt_raw then
                              Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw)
                            else ""
                          in
                          set_text meta
                            ((if ok then "OK" else "Error")
                            ^ if dt_s <> "" then " (" ^ dt_s ^ ")" else "");
                          clear_children (out :> Dom.node Js.t);
                          if out_s <> "" then append_console_chunk out "py-out-stdout" out_s;
                          if err_s <> "" then append_console_chunk out "py-out-stderr" err_s;
                          if out_s = "" && err_s = "" && ok then
                            append_console_chunk out "py-out-muted"
                              "Assertions passed (no stdout). Click Edit -> print(...) to inspect values.\n";
                          render_state_panel ch (Js.Unsafe.get res (js "state")))
                  | _ -> ()
                in

                run_btn##.onclick :=
                  Dom.handler (fun _ ->
                      trigger_run ();
                      Js._true);

                pg_btn##.onclick :=
                  Dom.handler (fun _ ->
                      let src =
                        match !editor_ref with
                        | Some ed -> to_s ed##.value
                        | None -> orig_code
                      in
                      (try
                         let ss = Js.Unsafe.get Dom_html.window (js "sessionStorage") in
                         ignore
                           (Js.Unsafe.meth_call ss "setItem"
                              [|
                                Js.Unsafe.inject (js "baux_playground_code");
                                Js.Unsafe.inject (js (src ^ "\n"));
                              |])
                       with _ -> ());
                      Dom_html.window##.location##.href := js (get_base_url () ^ "playground.html");
                      Js._true))
  done

let init () : unit =
  (match Js.Opt.to_option (document##getElementById (js "py-playground-root")) with
  | Some root -> mount_playground root
  | None -> ());
  enhance_inline_blocks ()
