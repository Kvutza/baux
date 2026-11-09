(* admin/ui/baux_ui.ml
   Compiled via Dune & Js_of_ocaml (opam switch 5.2.0+ox)
   Baux API Interactive Function Explorer & Subsystem Architecture Graph
*)

[@@@warning "-32-27"]
open Js_of_ocaml
open Dom_html

let jf = Js.float
let js = Js.string

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

(* --- Copy to Clipboard Buttons --- *)
let init_copy_buttons () =
  let buttons = query_all ".copy-btn" in
  List.iter (fun btn_raw ->
    let btn : htmlElement Js.t = Js.Unsafe.coerce btn_raw in
    add_click btn_raw (fun () ->
      let parent = btn##.parentElement in
      let text =
        match Js.Opt.to_option parent with
        | Some p ->
            let p_elem : htmlElement Js.t = Js.Unsafe.coerce p in
            (match Js.Opt.to_option (p_elem##querySelector (js "code")) with
            | Some c ->
                let c_elem : htmlElement Js.t = Js.Unsafe.coerce c in
                Js.to_string c_elem##.innerText
            | None ->
                (match Js.Opt.to_option (p_elem##querySelector (js ".install-cmd-text")) with
                | Some c ->
                    let c_elem : htmlElement Js.t = Js.Unsafe.coerce c in
                    Js.to_string c_elem##.innerText
                | None -> Js.to_string p_elem##.innerText))
        | None -> ""
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

(* --- Theme Toggle --- *)
let init_theme_toggle () =
  match (Js.Opt.to_option (document##getElementById (js "theme-toggle-btn")),
         Js.Opt.to_option (document##getElementById (js "theme-toggle-label"))) with
  | (Some btn, Some label) ->
      let label_html : htmlElement Js.t = Js.Unsafe.coerce label in
      add_click btn (fun () ->
        let cur =
          document##.documentElement##getAttribute (js "data-theme")
          |> Js.Opt.to_option
          |> (function Some s -> Js.to_string s | None -> "light")
        in
        let nxt = if cur = "light" then "dark" else "light" in
        document##.documentElement##setAttribute (js "data-theme") (js nxt);
        label_html##.innerText := js (if nxt = "light" then "Dark" else "Light")
      )
  | _ -> ()

(* --- BAUX API Explorer in OCaml --- *)
type api_symbol = {
  name : string;
  kind : string;
  sig_str : string;
  doc : string;
}

type api_module = {
  id : string;
  title : string;
  desc : string;
  symbols : api_symbol list;
}

type graph_node = {
  g_id : string;
  g_label : string;
  g_desc : string;
  g_deps : string list;
}

let baux_modules : api_module list = [
  {
    id = "core";
    title = "Functional Transitions";
    desc = "Pure functional state transitions, execution folds, and parallel execution.";
    symbols = [
      { name = "init"; kind = "function";
        sig_str = "(domain, config=None, *, pipe=None, key=None, seed=None, low=-1.0, high=1.0) -> OptimizerState";
        doc = "Initialize an immutable OptimizerState from bounds or a parameter PyTree." };
      { name = "ask"; kind = "function";
        sig_str = "(state, num_arms=None, *, n=None) -> tuple[OptimizerState, AskBatch]";
        doc = "Pure proposal transition generating candidate points and updated state." };
      { name = "tell"; kind = "function";
        sig_str = "(state, x, y, *, y_var=None) -> tuple[OptimizerState, np.ndarray]";
        doc = "Pure observation transition incorporating evaluation outcomes and updating posterior buffers." };
      { name = "step"; kind = "function";
        sig_str = "(state, objective, *, num_arms=1, use_pytree=False) -> tuple[OptimizerState, np.ndarray]";
        doc = "Single atomic ask -> evaluate -> tell state transition." };
      { name = "scan"; kind = "function";
        sig_str = "(state, key=None, steps=0, eval_fn=None, num_arms=1) -> tuple[OptimizerState, Key, list[dict]]";
        doc = "Folds optimization rounds deterministically over state and splittable PRNG key." };
      { name = "pmap"; kind = "function";
        sig_str = "(state, key, branches, branch_fn) -> list[Any]";
        doc = "Forks state snapshot and splits PRNG keys across parallel worker threads." };
      { name = "complete"; kind = "function";
        sig_str = "(state, ids, y, *, y_var=None) -> OptimizerState";
        doc = "Asynchronous regional completion by AskId ticket." };
      { name = "cancel"; kind = "function";
        sig_str = "(state, ids) -> OptimizerState";
        doc = "Pure transition discarding pending regional candidate tickets." };
      { name = "digest"; kind = "function";
        sig_str = "(state) -> str";
        doc = "BLAKE3 digest of the optimizer state." };
      { name = "freeze"; kind = "function";
        sig_str = "(state) -> bytes";
        doc = "Serializes state to bytes." };
      { name = "thaw"; kind = "function";
        sig_str = "(state, data: bytes) -> OptimizerState";
        doc = "Reconstructs state from bytes." };
      { name = "OptimizerState"; kind = "class";
        sig_str = "Immutable optimization state record";
        doc = "Exposes .best_x, .best_y, .best_params, .x_obs, .y_obs, .num_observations, .epoch, .tr_length, and .key." };
      { name = "AskBatch"; kind = "class";
        sig_str = "(ids, points, spec=None, weights=None)";
        doc = "Proposal container providing candidate IDs, points, decoded parameters, and coordinate tensors." };
      { name = "AskId"; kind = "class";
        sig_str = "(value: int)";
        doc = "Epoch-guarded ticket for asynchronous regional proposals." };
    ];
  };
  {
    id = "keys";
    title = "PRNG Key Algebra";
    desc = "Deterministic, splittable 128-bit pseudo-random key generation and folding.";
    symbols = [
      { name = "key"; kind = "function";
        sig_str = "(source=None) -> Key";
        doc = "Constructs splittable 128-bit key from integer seed, byte buffer, array, or state." };
      { name = "split"; kind = "function";
        sig_str = "(k, n=2) -> tuple[Key, ...]";
        doc = "Deterministically splits a parent key into n independent subkeys." };
      { name = "fold"; kind = "function";
        sig_str = "(k, tag: int) -> Key";
        doc = "Deterministically folds an integer domain tag into key without altering parent sequence." };
      { name = "Key"; kind = "class";
        sig_str = "128-bit splittable PRNG key";
        doc = "128-bit (u64, u64) SplitMix64 key algebra seeding deterministic ChaCha12 and optimizer streams." };
    ];
  };
  {
    id = "transforms";
    title = "Pipeline Transforms";
    desc = "Composable optimization pipeline combinators adhering to the Transform protocol.";
    symbols = [
      { name = "chain"; kind = "function";
        sig_str = "(*steps) -> OptimizerConfig";
        doc = "Composes Transform stages into an OptimizerConfig." };
      { name = "lhd"; kind = "function";
        sig_str = "(num_init=10, *, only=False) -> Transform";
        doc = "Latin Hypercube Design initialization stage." };
      { name = "sobol"; kind = "function";
        sig_str = "(num_candidates=None, *, per_arm=None, num_pert=None) -> Transform";
        doc = "Sobol sequence quasi-random candidate generator." };
      { name = "raasp"; kind = "function";
        sig_str = "(num_pert=20, *, num_candidates=None, per_arm=None, driver=RAASPDriver.ORIG) -> Transform";
        doc = "Random Axis-Aligned Subspace Perturbation proposal sampling." };
      { name = "uniform_pert"; kind = "function";
        sig_str = "(num_candidates=None, *, per_arm=None, num_pert=None) -> Transform";
        doc = "Uniform candidate perturbation sampler." };
      { name = "enn"; kind = "function";
        sig_str = "(*, k=10, num_samples=100, scale_x=False, index=NeighborIndex.FLAT, geometry='euclidean', local_k=4) -> Transform";
        doc = "Attaches Epistemic Nearest Neighbors surrogate to optimization pipeline." };
      { name = "gp"; kind = "function";
        sig_str = "() -> Transform";
        doc = "Gaussian Process surrogate transform." };
      { name = "zero"; kind = "function";
        sig_str = "() -> Transform";
        doc = "Zero baseline surrogate transform." };
      { name = "ucb"; kind = "function";
        sig_str = "(beta=2.0) -> Transform";
        doc = "Upper Confidence Bound acquisition function." };
      { name = "thompson"; kind = "function";
        sig_str = "() -> Transform";
        doc = "Thompson sampling acquisition via coherent posterior function draws." };
      { name = "pareto"; kind = "function";
        sig_str = "() -> Transform";
        doc = "Multi-objective Pareto-front non-dominated arm selector." };
      { name = "random_acq"; kind = "function";
        sig_str = "() -> Transform";
        doc = "Uniform random acquisition selector." };
      { name = "tr_bounds"; kind = "function";
        sig_str = "(*, length_init=0.8, length_min=0.5**7, length_max=1.6) -> Transform";
        doc = "Configures trust-region length bounds and expansion/contraction factors." };
      { name = "regions"; kind = "function";
        sig_str = "(count=4, *, sharing='ellipsoid_ball', allocation='incumbent_ucb1') -> Transform";
        doc = "Configures multi-region TuRBO/MORBO portfolio coordination." };
      { name = "with_geometry"; kind = "function";
        sig_str = "(target, geometry: str, local_k=4)";
        doc = "Equips surrogate target with conformal, hyperbolic, or Wasserstein geometry." };
      { name = "stein"; kind = "function";
        sig_str = "(target, frame=None, *, rank=None)";
        doc = "Attaches polar Stiefel subspace projection with orthogonal complement shrinkage." };
      { name = "calibrate"; kind = "function";
        sig_str = "(target, affine=None, *, alpha=None)";
        doc = "Attaches affine location-scale and split-conformal residual scaling (valid on exchangeable held-out splits)." };
      { name = "Transform"; kind = "class";
        sig_str = "Pipeline step protocol exposing __call__(cfg) -> cfg and pipe operator |";
        doc = "Base protocol for all composable optimization transforms." };
      { name = "OptimizerConfig"; kind = "class";
        sig_str = "Full declarative specification of surrogate, sampler, acquisition, and regions";
        doc = "Immutable configuration object passed to optimizer init." };
    ];
  };
  {
    id = "charts";
    title = "Domain Charts, PyTree & Codomain";
    desc = "Structured parameter charts, PyTree flattening, and multi-output codomain operators.";
    symbols = [
      { name = "pytree_bounds"; kind = "function";
        sig_str = "(tree, low=-1.0, high=1.0) -> PyTreeSpec";
        doc = "Compiles nested Python dictionaries, lists, and tuples into flat coordinate specifications." };
      { name = "Log"; kind = "class";
        sig_str = "(low: float, high: float)";
        doc = "Scale-isometric logarithmic chart on positive multiplicative reals [a, b] (0 < a < b)." };
      { name = "Periodic"; kind = "class";
        sig_str = "(low=0.0, high=2*pi)";
        doc = "Circular trigonometric embedding on [a, b) into [0, 1]^2 monotone in shortest circular angle." };
      { name = "Simplex"; kind = "class";
        sig_str = "(dim: int)";
        doc = "Square-root spherical coordinates on the probability simplex Delta^{d-1} (1/4 Fisher-Rao pullback)." };
      { name = "categorical"; kind = "function";
        sig_str = "(categories: list[Any], *, encoding='auto') -> Categorical";
        doc = "Constructs exact one-hot (C <= 15) or Rademacher JL codebook (C > 15) for discrete choices." };
      { name = "Categorical"; kind = "class";
        sig_str = "Categorical coordinate embedding specification";
        doc = "Manages discrete category codebook and minimum Euclidean distance retraction." };
      { name = "Stiefel"; kind = "class";
        sig_str = "(loadings: np.ndarray)";
        doc = "Codomain operator: polar orthonormal projection onto St(r, M) for subspace shrinkage (used with baux.stein)." };
      { name = "Affine"; kind = "class";
        sig_str = "(scale: np.ndarray, shift: np.ndarray, conformal_q=1.0)";
        doc = "Codomain operator: affine location-scale and split-conformal multiplier (used with baux.calibrate)." };
      { name = "ravel"; kind = "function";
        sig_str = "(tree, spec=None) -> np.ndarray";
        doc = "Flattens structured parameter container into normalized 1D coordinate array." };
      { name = "unflatten"; kind = "function";
        sig_str = "(flat: np.ndarray, spec: PyTreeSpec) -> Any";
        doc = "Isomorphically reconstructs nested parameter container from coordinate array." };
      { name = "unflatten_batch"; kind = "function";
        sig_str = "(batch: np.ndarray, spec: PyTreeSpec) -> list[Any]";
        doc = "Decodes batch of coordinate vectors into list of parameter PyTrees." };
      { name = "unflatten_stacked"; kind = "function";
        sig_str = "(stacked: np.ndarray, spec: PyTreeSpec) -> Any";
        doc = "Decodes stacked coordinate tensor into PyTree with array leaves." };
      { name = "PyTreeSpec"; kind = "class";
        sig_str = "Compiled domain manifold layout";
        doc = "Stores coordinate dimensions, bound vectors, and leaf chart retractions." };
    ];
  };
  {
    id = "surrogate";
    title = "Functional ENN Surrogate";
    desc = "Epistemic Nearest Neighbor surrogate initialization, likelihood fitting, and posterior sampling.";
    symbols = [
      { name = "enn_init"; kind = "function";
        sig_str = "(train_x, train_y, train_yvar=None, *, scale_x=False, **options) -> ENNState";
        doc = "Constructs immutable ENN surrogate state from observation matrices." };
      { name = "enn_append"; kind = "function";
        sig_str = "(state, x, y, y_var=None) -> ENNState";
        doc = "Copy-on-write observation addition returning new ENNState." };
      { name = "enn_fit"; kind = "function";
        sig_str = "(state, k=10, *, num_candidates=100, num_samples=10, infer_aleatoric=True, key=None, warm=None) -> ENNParams";
        doc = "Estimates optimal scale parameters by leave-one-out log-pseudolikelihood." };
      { name = "enn_posterior"; kind = "function";
        sig_str = "(state, x, params, *, flags=None) -> ENNNormal";
        doc = "Evaluates posterior mean, total variance, epistemic variance, and aleatoric variance." };
      { name = "enn_sample"; kind = "function";
        sig_str = "(state, x, params, *, key=None, num_draws=1, seeds=None, flags=None) -> tuple[np.ndarray, np.ndarray]";
        doc = "Draws deterministic coherent function samples over a query batch (smooth within order-K Voronoi cells)." };
      { name = "ENNState"; kind = "class";
        sig_str = "Immutable surrogate buffer state";
        doc = "Holds train_x, train_y, pairwise metric providers, and local scale caches." };
      { name = "ENNParams"; kind = "class";
        sig_str = "(c_e, s_0, lengthscales=None)";
        doc = "Surrogate hyperparameters fitted via pseudolikelihood." };
      { name = "ENN"; kind = "class";
        sig_str = "Stateful wrapper around functional ENN routines";
        doc = "Object-oriented interface exposing fit, predict, and sample." };
    ];
  };
  {
    id = "adapters";
    title = "Hardware & Adapters";
    desc = "Device capability enumeration, stateful wrappers, and Optuna integration.";
    symbols = [
      { name = "capabilities"; kind = "function";
        sig_str = "(*, probe=False) -> tuple[BackendCapabilities, ...]";
        doc = "Reports compiled and active hardware accelerators (Metal, CUDA, CPU SIMD)." };
      { name = "create_optimizer"; kind = "function";
        sig_str = "(*, bounds, config, rng) -> Optimizer";
        doc = "Instantiates stateful object-oriented optimizer wrapper." };
      { name = "Optimizer"; kind = "class";
        sig_str = "Stateful optimizer harness";
        doc = "Exposes ask(num_arms), tell(x, y), posterior_mu(x), and telemetry()." };
      { name = "turbo_enn"; kind = "function";
        sig_str = "() -> OptimizerConfig";
        doc = "Default TuRBO-1 trust-region search configuration backed by ENN surrogate." };
      { name = "turbo_one"; kind = "function";
        sig_str = "() -> OptimizerConfig";
        doc = "Single-region trust-region configuration." };
      { name = "turbo_zero"; kind = "function";
        sig_str = "() -> OptimizerConfig";
        doc = "Trust-region baseline configuration with zero surrogate." };
      { name = "lhd_only"; kind = "function";
        sig_str = "() -> OptimizerConfig";
        doc = "Quasi-random Latin Hypercube sampling configuration." };
      { name = "BackendCapabilities"; kind = "class";
        sig_str = "Hardware device capability record";
        doc = "Provides device availability, compilation flags, and kernel operation support." };
    ];
  };
]

let baux_graph : graph_node list = [
  { g_id = "domain"; g_label = "Domain Space (PyTree, Manifolds)";
    g_desc = "Section-retraction charts (Log, Periodic, Simplex, Categorical) compiled to coordinates.";
    g_deps = [] };
  { g_id = "state"; g_label = "Optimizer State (S)";
    g_desc = "Immutable state snapshot: incumbent, observation buffers (.x_obs, .y_obs), epoch, and PRNG key.";
    g_deps = ["domain"; "keys"] };
  { g_id = "keys"; g_label = "PRNG Key Algebra";
    g_desc = "Splittable 128-bit SplitMix64 keys for deterministic execution and parallel branches.";
    g_deps = [] };
  { g_id = "pipeline"; g_label = "Transform Pipeline";
    g_desc = "Composable search operators: initialization, perturbation, surrogate, and acquisition.";
    g_deps = ["domain"] };
  { g_id = "surrogate_eng"; g_label = "ENN Surrogate Engine";
    g_desc = "Epistemic Nearest Neighbors with conformal, Poincare half-space, and transport dissimilarities.";
    g_deps = ["hardware"] };
  { g_id = "hardware"; g_label = "Hardware Kernels";
    g_desc = "Vectorized distance sweeps and posterior draws across CPU SIMD, Metal, and CUDA.";
    g_deps = [] };
  { g_id = "ask"; g_label = "Proposal Transition (ask)";
    g_desc = "Pure function generating candidate arm batches via RAASP / Sobol / Pareto.";
    g_deps = ["state"; "pipeline"; "surrogate_eng"] };
  { g_id = "tell"; g_label = "Observation Transition (tell)";
    g_desc = "Pure function incorporating target observations and updating surrogate posterior.";
    g_deps = ["state"; "surrogate_eng"] };
  { g_id = "rollouts"; g_label = "Rollouts (scan / pmap)";
    g_desc = "Deterministic trajectory folding and multi-threaded parallel mapping.";
    g_deps = ["ask"; "tell"; "keys"] };
  { g_id = "adapters"; g_label = "Adapters (Optuna, Harness)";
    g_desc = "Stateful wrappers and Optuna BaseSampler compatibility.";
    g_deps = ["state"; "ask"; "tell"] };
]

let html_escape s =
  let len = String.length s in
  let buf = Buffer.create len in
  for i = 0 to len - 1 do
    match s.[i] with
    | '&' -> Buffer.add_string buf "&amp;"
    | '<' -> Buffer.add_string buf "&lt;"
    | '>' -> Buffer.add_string buf "&gt;"
    | '"' -> Buffer.add_string buf "&quot;"
    | '\'' -> Buffer.add_string buf "&#x27;"
    | c -> Buffer.add_char buf c
  done;
  Buffer.contents buf

let string_contains haystack needle =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  if n_len = 0 then true
  else if n_len > h_len then false
  else
    let rec check i j =
      if j = n_len then true
      else if i + j >= h_len then false
      else if haystack.[i + j] = needle.[j] then check i (j + 1)
      else check (i + 1) 0
    in
    check 0 0

let render_api_explorer () =
  match Js.Opt.to_option (document##getElementById (js "baux-api-explorer")) with
  | None -> ()
  | Some root ->
      let active_mod = ref "core" in
      let search_query = ref "" in
      let selected_node_id = ref "" in

      let shell_html = Printf.sprintf "
        <div class=\"api-explorer-header\">
          <div class=\"api-view-tabs\" role=\"tablist\">
            <button type=\"button\" class=\"api-tab active\" id=\"tab-btn-fn\" data-view=\"functions\">Function &amp; Class Explorer</button>
            <button type=\"button\" class=\"api-tab\" id=\"tab-btn-graph\" data-view=\"graph\">Architecture &amp; Pipeline Graph</button>
          </div>
          <div class=\"api-search-box\">
            <input type=\"text\" id=\"api-search-input\" value=\"%s\" placeholder=\"Filter functions, classes, transforms... (e.g. ask, pytree, enn, ucb)\" />
          </div>
        </div>

        <div id=\"api-view-functions\" class=\"api-panel active\">
          <div class=\"api-explorer-layout\">
            <nav id=\"api-module-nav\" class=\"api-module-nav\"></nav>
            <main class=\"api-symbol-details\">
              <div id=\"api-details-header\" class=\"api-details-header\"></div>
              <div id=\"api-cards-list\" class=\"api-cards-list\"></div>
            </main>
          </div>
        </div>

        <div id=\"api-view-graph\" class=\"api-panel\">
          <div class=\"api-graph-container\">
            <div class=\"api-graph-intro\">Interactive Subsystem Graph. Click any component below to view dependency contracts and execution flows:</div>
            <div id=\"api-graph-grid\" class=\"graph-grid\"></div>
            <div id=\"api-node-inspector\" class=\"api-node-inspector\"></div>
          </div>
        </div>
      " (html_escape !search_query) in
      root##.innerHTML := js shell_html;

      let activate_tab view =
        match (Js.Opt.to_option (document##getElementById (js "tab-btn-fn")),
               Js.Opt.to_option (document##getElementById (js "tab-btn-graph")),
               Js.Opt.to_option (document##getElementById (js "api-view-functions")),
               Js.Opt.to_option (document##getElementById (js "api-view-graph"))) with
        | (Some b_fn, Some b_gr, Some p_fn, Some p_gr) ->
            if view = "graph" then begin
              b_gr##.classList##add (js "active");
              b_fn##.classList##remove (js "active");
              p_gr##.classList##add (js "active");
              p_fn##.classList##remove (js "active")
            end else begin
              b_fn##.classList##add (js "active");
              b_gr##.classList##remove (js "active");
              p_fn##.classList##add (js "active");
              p_gr##.classList##remove (js "active")
            end
        | _ -> ()
      in

      let rec render_functions () =
        let q = String.lowercase_ascii !search_query in

        (* Generate Module Nav Buttons *)
        let nav_buf = Buffer.create 1024 in
        List.iter (fun m ->
          let is_act = m.id = !active_mod && !search_query = "" in
          let cnt = List.length m.symbols in
          Buffer.add_string nav_buf
            (Printf.sprintf "<button type=\"button\" class=\"api-mod-btn %s\" data-mod=\"%s\"><span class=\"mod-title\">%s</span><span class=\"mod-badge\">%d</span></button>"
               (if is_act then "active" else "") (html_escape m.id) (html_escape m.title) cnt)
        ) baux_modules;

        (* Generate Symbol Cards *)
        let cards_buf = Buffer.create 4096 in
        let header_desc = ref "" in
        let sym_count = ref 0 in

        if !search_query <> "" then begin
          List.iter (fun m ->
            List.iter (fun s ->
              let s_name = String.lowercase_ascii s.name in
              let s_doc = String.lowercase_ascii s.doc in
              if string_contains s_name q || string_contains s_doc q then begin
                incr sym_count;
                let badge = if s.kind = "class" then "<span class=\"badge class\">class</span>" else "<span class=\"badge fn\">function</span>" in
                Buffer.add_string cards_buf
                  (Printf.sprintf "<article class=\"api-card\"><div class=\"api-card-header\"><span class=\"api-symbol-name\">%s</span>%s<span class=\"badge mod\">%s</span></div><div class=\"api-card-signature\"><code>baux.%s%s</code></div><div class=\"api-card-doc\">%s</div></article>"
                     (html_escape s.name) badge (html_escape m.title) (html_escape s.name) (html_escape s.sig_str) (html_escape s.doc))
              end
            ) m.symbols
          ) baux_modules;
          header_desc := Printf.sprintf "Found %d symbols matching \"%s\"" !sym_count (html_escape !search_query)
        end else begin
          let current_m =
            match List.find_opt (fun m -> m.id = !active_mod) baux_modules with
            | Some m -> m
            | None -> List.hd baux_modules
          in
          header_desc := Printf.sprintf "<strong>%s</strong> &mdash; %s"
            (html_escape current_m.title) (html_escape current_m.desc);
          List.iter (fun s ->
            incr sym_count;
            let badge = if s.kind = "class" then "<span class=\"badge class\">class</span>" else "<span class=\"badge fn\">function</span>" in
            Buffer.add_string cards_buf
              (Printf.sprintf "<article class=\"api-card\"><div class=\"api-card-header\"><span class=\"api-symbol-name\">%s</span>%s</div><div class=\"api-card-signature\"><code>baux.%s%s</code></div><div class=\"api-card-doc\">%s</div></article>"
                 (html_escape s.name) badge (html_escape s.name) (html_escape s.sig_str) (html_escape s.doc))
          ) current_m.symbols
        end;

        (match Js.Opt.to_option (document##getElementById (js "api-module-nav")) with
        | Some el -> el##.innerHTML := js (Buffer.contents nav_buf)
        | None -> ());
        (match Js.Opt.to_option (document##getElementById (js "api-details-header")) with
        | Some el -> el##.innerHTML := js !header_desc
        | None -> ());
        (match Js.Opt.to_option (document##getElementById (js "api-cards-list")) with
        | Some el -> el##.innerHTML := js (Buffer.contents cards_buf)
        | None -> ());

        (* Bind Module Nav Buttons *)
        let nav_buttons = query_all "#api-module-nav .api-mod-btn" in
        List.iter (fun btn ->
          add_click btn (fun () ->
            let mid =
              btn##getAttribute (js "data-mod")
              |> Js.Opt.to_option
              |> (function Some s -> Js.to_string s | None -> "core")
            in
            active_mod := mid;
            search_query := "";
            (match Js.Opt.to_option (document##getElementById (js "api-search-input")) with
            | Some inp ->
                let inp_elem : inputElement Js.t = Js.Unsafe.coerce inp in
                inp_elem##.value := js ""
            | None -> ());
            render_functions ()
          )
        ) nav_buttons
      in

      let rec render_graph () =
        (* Generate Architecture Graph Nodes *)
        let graph_buf = Buffer.create 2048 in
        List.iter (fun n ->
          let is_sel = n.g_id = !selected_node_id in
          let sel_cls = if is_sel then "selected" else "" in
          let deps_str =
            if n.g_deps <> [] then
              Printf.sprintf "<div class=\"node-deps\">Inputs: %s</div>"
                (String.concat ", " (List.map html_escape n.g_deps))
            else ""
          in
          Buffer.add_string graph_buf
            (Printf.sprintf "<div class=\"graph-node-box %s\" data-node=\"%s\"><div class=\"node-title\">%s</div><div class=\"node-preview\">%s</div>%s</div>"
               sel_cls (html_escape n.g_id) (html_escape n.g_label) (html_escape n.g_desc) deps_str)
        ) baux_graph;

        (* Inspector card for selected node *)
        let inspector_html =
          if !selected_node_id = "" then
            "<div class=\"node-placeholder\">Click any component box above to inspect its execution contracts and pipeline flow.</div>"
          else
            try
              let n = List.find (fun x -> x.g_id = !selected_node_id) baux_graph in
              let inputs_li =
                if n.g_deps = [] then "<li><em>None (Root Leaf Component)</em></li>"
                else String.concat "" (List.map (fun d -> Printf.sprintf "<li><code>%s</code></li>" (html_escape d)) n.g_deps)
              in
              let down_nodes = List.filter (fun x -> List.mem n.g_id x.g_deps) baux_graph in
              let down_li =
                if down_nodes = [] then "<li><em>End Terminal Component</em></li>"
                else String.concat "" (List.map (fun x -> Printf.sprintf "<li><code>%s</code></li>" (html_escape x.g_label)) down_nodes)
              in
              Printf.sprintf "<div class=\"node-detail-card\"><div class=\"node-detail-header\"><h3>%s</h3><span class=\"badge comp\">Subsystem</span></div><p class=\"node-detail-desc\">%s</p><div class=\"node-contracts-grid\"><div class=\"contract-col\"><strong>Prerequisite Components (Inputs):</strong><ul>%s</ul></div><div class=\"contract-col\"><strong>Dependent Subsystems (Downstream):</strong><ul>%s</ul></div></div></div>"
                (html_escape n.g_label) (html_escape n.g_desc) inputs_li down_li
            with _ -> ""
        in

        (match Js.Opt.to_option (document##getElementById (js "api-graph-grid")) with
        | Some el -> el##.innerHTML := js (Buffer.contents graph_buf)
        | None -> ());
        (match Js.Opt.to_option (document##getElementById (js "api-node-inspector")) with
        | Some el -> el##.innerHTML := js inspector_html
        | None -> ());

        (* Bind Architecture Graph Boxes *)
        let graph_boxes = query_all "#api-graph-grid .graph-node-box" in
        List.iter (fun box ->
          add_click box (fun () ->
            let nid =
              box##getAttribute (js "data-node")
              |> Js.Opt.to_option
              |> (function Some s -> Js.to_string s | None -> "")
            in
            selected_node_id := nid;
            render_graph ();
            activate_tab "graph"
          )
        ) graph_boxes
      in

      (* Bind View Switcher Tabs *)
      (match (Js.Opt.to_option (document##getElementById (js "tab-btn-fn")),
              Js.Opt.to_option (document##getElementById (js "tab-btn-graph"))) with
      | (Some b_fn, Some b_gr) ->
          add_click b_fn (fun () -> activate_tab "functions");
          add_click b_gr (fun () -> activate_tab "graph")
      | _ -> ());

      (* Bind Persistent Search Input *)
      (match Js.Opt.to_option (document##getElementById (js "api-search-input")) with
      | Some inp ->
          let inp_elem : inputElement Js.t = Js.Unsafe.coerce inp in
          let on_input = handler (fun _ev ->
            let val_ = Js.to_string inp_elem##.value in
            search_query := val_;
            if val_ <> "" then activate_tab "functions";
            render_functions ();
            Js._true
          ) in
          let _ = addEventListener inp (Event.make "input") on_input Js._false in
          ()
      | None -> ());

      render_functions ();
      render_graph ()

(* --- Tokamak 3D WebGL CAD & Benchmark Explorer (arXiv:2505.00663 Fig. 8-9 + 10-Seed TORAX 2.0.0) --- *)
let mount_tokamak_cad () =
  let boot_fn = Js.Unsafe.js_expr {|
(function() {
  function escHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#x27;");
  }

  // Exact 25-vertex TCV vacuum vessel inner contour extracted from arXiv:2505.00663 Fig. 9 (state0_refs.pdf)
  var VESSEL_INNER_RZ = [
    [0.6240,  0.0000], [0.6241,  0.7091], [0.6475,  0.7270], [0.6662,  0.7500],
    [0.9733,  0.7498], [0.9907,  0.7252], [1.1289,  0.5599], [1.1360,  0.5351],
    [1.1360, -0.3282], [1.1402, -0.3460], [1.0869, -0.3764], [1.0707, -0.3994],
    [1.0586, -0.4599], [1.0829, -0.5808], [1.0967, -0.5905], [1.1165, -0.5669],
    [1.1289, -0.5598], [0.9907, -0.7252], [0.9733, -0.7498], [0.8506, -0.7500],
    [0.6662, -0.7500], [0.6475, -0.7270], [0.6241, -0.7091], [0.6240, -0.4933]
  ];

  // Outer octagonal TCV vacuum vessel shell (arXiv:2505.00663 Fig. 8)
  var VESSEL_OUTER_RZ = [
    [0.592, -0.725], [0.592,  0.725], [0.654,  0.792], [0.988,  0.792],
    [1.174,  0.555], [1.174, -0.575], [0.988, -0.792], [0.654, -0.792]
  ];

  var VESSEL_WALL_IN_RZ = [
    [0.624, -0.708], [0.624,  0.708], [0.666,  0.748], [0.973,  0.748],
    [1.134,  0.535], [1.134, -0.558], [0.973, -0.748], [0.666, -0.748]
  ];

  var TCV_CAD = {
    id: "tcv",
    name: "TCV (Tokamak à Configuration Variable)",
    shortName: "TCV (EPFL)",
    R0: 0.88, a: 0.25, B0: 1.43, ipNominal: 1.49,
    dist: 5.55,
    rMin: 0.55, rMax: 1.25, zMin: -0.85, zMax: 0.85,
    rTicks: [0.65, 0.90, 1.10],
    zTicks: [-0.6, 0.0, 0.6],
    vessel_outer: VESSEL_OUTER_RZ,
    vessel_inner: VESSEL_WALL_IN_RZ,
    first_wall: VESSEL_INNER_RZ,
    coils: [
      {name: "OH1-Top", r: 0.325, z: 0.885, w: 0.100, h: 0.090},
      {name: "OH1-Bot", r: 0.325, z: -0.885, w: 0.100, h: 0.090},
      {name: "OH2-Top", r: 0.455, z: 0.910, w: 0.080, h: 0.100},
      {name: "OH2-Bot", r: 0.455, z: -0.910, w: 0.080, h: 0.100},
      {name: "E1", r: 0.480, z: -0.672, w: 0.096, h: 0.176},
      {name: "E2", r: 0.480, z: -0.480, w: 0.096, h: 0.176},
      {name: "E3", r: 0.480, z: -0.288, w: 0.096, h: 0.176},
      {name: "E4", r: 0.480, z: -0.096, w: 0.096, h: 0.176},
      {name: "E5", r: 0.480, z:  0.096, w: 0.096, h: 0.176},
      {name: "E6", r: 0.480, z:  0.288, w: 0.096, h: 0.176},
      {name: "E7", r: 0.480, z:  0.480, w: 0.096, h: 0.176},
      {name: "E8", r: 0.480, z:  0.672, w: 0.096, h: 0.176},
      {name: "F1", r: 1.296, z: -0.720, w: 0.143, h: 0.160},
      {name: "F2", r: 1.296, z: -0.518, w: 0.143, h: 0.160},
      {name: "F3", r: 1.296, z: -0.316, w: 0.143, h: 0.160},
      {name: "F4", r: 1.296, z: -0.114, w: 0.143, h: 0.160},
      {name: "F5", r: 1.296, z:  0.088, w: 0.143, h: 0.160},
      {name: "F6", r: 1.296, z:  0.290, w: 0.143, h: 0.160},
      {name: "F7", r: 1.296, z:  0.492, w: 0.143, h: 0.160},
      {name: "F8", r: 1.296, z:  0.694, w: 0.143, h: 0.160}
    ]
  };


  var ARC_CAD = {
    id: "arc",
    name: "ARC (Affordable Robust Compact)",
    shortName: "ARC (MIT / CFS)",
    R0: 3.30, a: 1.13, B0: 9.20, ipNominal: 7.80,
    dist: 22.8,
    rMin: 0.60, rMax: 6.60, zMin: -4.60, zMax: 4.60,
    rTicks: [1.5, 3.0, 4.5, 6.0],
    zTicks: [-4.0, -2.0, 0.0, 2.0, 4.0],
    vessel_outer: [[3.330,0.000], [3.330,-0.859], [3.581,-2.068], [3.370,-2.260], [3.370,-2.386], [3.696,-2.549], [4.284,-3.528], [4.722,-4.180], [5.400,-4.180], [5.400,-3.210], [4.973,-3.406], [4.298,-2.367], [4.834,-1.904], [5.299,-1.357], [5.750,-0.794], [5.910,-0.315], [5.910,0.000], [5.910,0.315], [5.750,0.794], [5.299,1.357], [4.834,1.904], [4.298,2.367], [4.973,3.406], [5.400,3.210], [5.400,4.180], [4.722,4.180], [4.284,3.528], [3.696,2.549], [3.370,2.386], [3.370,2.260], [3.581,2.068], [3.330,0.859], [3.330,0.000]],
    vessel_inner: [[3.405,0.000], [3.405,-0.852], [3.664,-2.095], [3.445,-2.293], [3.445,-2.339], [3.749,-2.491], [4.347,-3.488], [4.762,-4.105], [5.325,-4.105], [5.325,-3.327], [4.946,-3.501], [4.200,-2.353], [4.781,-1.851], [5.242,-1.310], [5.683,-0.757], [5.835,-0.302], [5.835,0.000], [5.835,0.302], [5.683,0.757], [5.242,1.310], [4.781,1.851], [4.200,2.353], [4.946,3.501], [5.325,3.327], [5.325,4.105], [4.762,4.105], [4.347,3.488], [3.749,2.491], [3.445,2.339], [3.445,2.293], [3.664,2.095], [3.405,0.852], [3.405,0.000]],
    first_wall: [[3.415,0.000], [3.415,-0.851], [3.675,-2.098], [3.455,-2.298], [3.455,-2.333], [3.756,-2.484], [4.356,-3.483], [4.767,-4.095], [5.315,-4.095], [5.315,-3.342], [4.942,-3.514], [4.187,-2.351], [4.774,-1.843], [5.234,-1.303], [5.675,-0.752], [5.825,-0.301], [5.825,0.000], [5.825,0.301], [5.675,0.752], [5.234,1.303], [4.774,1.843], [4.187,2.351], [4.942,3.514], [5.315,3.342], [5.315,4.095], [4.767,4.095], [4.356,3.483], [3.756,2.484], [3.455,2.333], [3.455,2.298], [3.675,2.098], [3.415,0.851], [3.415,0.000]],
    coils: [
    {name: "cs1u", r: 1.105, z: 0.750, w: 0.756, h: 1.387},
    {name: "cs1l", r: 1.105, z: -0.750, w: 0.756, h: 1.387},
    {name: "cs2u", r: 1.105, z: 1.900, w: 0.756, h: 0.687},
    {name: "cs2l", r: 1.105, z: -1.900, w: 0.756, h: 0.687},
    {name: "cs3u", r: 1.105, z: 2.700, w: 0.756, h: 0.687},
    {name: "cs3l", r: 1.105, z: -2.700, w: 0.756, h: 0.687},
    {name: "pf1u", r: 1.400, z: 3.900, w: 0.590, h: 0.590},
    {name: "pf1l", r: 1.400, z: -3.900, w: 0.590, h: 0.590},
    {name: "pf2u", r: 1.900, z: 4.750, w: 0.690, h: 0.690},
    {name: "pf2l", r: 1.900, z: -4.750, w: 0.690, h: 0.690},
    {name: "pf3u", r: 6.200, z: 2.950, w: 0.489, h: 0.489},
    {name: "pf3l", r: 6.200, z: -2.950, w: 0.489, h: 0.489},
    {name: "pf4u", r: 6.200, z: 2.250, w: 0.489, h: 0.489},
    {name: "pf4l", r: 6.200, z: -2.250, w: 0.489, h: 0.489},
    {name: "pf5u", r: 6.950, z: 1.600, w: 0.389, h: 0.389},
    {name: "pf5l", r: 6.950, z: -1.600, w: 0.389, h: 0.389}
  ]
  };


  var ITER_CAD = {
    id: "iter",
    name: "ITER (International Thermonuclear Experimental Reactor)",
    shortName: "ITER (Cadarache)",
    R0: 6.20, a: 2.00, B0: 5.30, ipNominal: 15.00,
    dist: 39.5,
    rMin: 1.10, rMax: 9.60, zMin: -6.00, zMax: 6.50,
    rTicks: [2.0, 4.0, 6.0, 8.0],
    zTicks: [-4.0, -2.0, 0.0, 2.0, 4.0],
    vessel_outer: [[6.716,-3.493], [7.732,-2.704], [8.350,-1.787], [8.756,-0.870], [8.844,1.085], [8.720,2.135], [8.356,2.915], [7.851,3.630], [7.002,4.376], [5.353,4.984], [4.535,5.170], [3.925,4.787], [3.725,4.035], [3.705,3.022], [3.705,2.007], [3.705,0.990], [3.705,-0.926], [3.705,-1.941], [3.705,-2.954], [3.542,-2.986], [3.715,-2.996], [3.814,-3.027], [3.905,-3.078], [3.984,-3.146], [4.048,-3.228], [4.094,-3.322], [4.120,-3.422], [4.126,-3.527], [4.110,-3.630], [4.074,-3.728], [3.762,-4.367], [3.788,-4.333], [4.091,-4.356], [4.245,-4.193], [4.328,-4.167], [4.414,-4.156], [4.500,-4.162], [4.584,-4.184], [4.663,-4.220], [4.733,-4.271], [4.793,-4.334], [4.853,-4.432], [4.872,-4.711], [5.157,-4.995], [5.164,-5.024], [5.164,-4.373], [5.169,-4.243], [5.189,-4.146], [5.237,-4.025], [5.308,-3.917], [5.399,-3.825], [5.508,-3.753], [5.628,-3.704], [5.755,-3.680], [6.715,-3.679], [6.813,-3.695]],
    vessel_inner: [[6.486,-3.263], [7.502,-2.474], [8.120,-1.557], [8.526,-0.640], [8.614,0.855], [8.490,1.905], [8.126,2.685], [7.622,3.400], [6.772,4.146], [5.553,4.754], [4.735,4.940], [4.125,4.557], [3.925,3.805], [3.905,2.792], [3.905,1.777], [3.905,0.760], [3.905,-0.696], [3.905,-1.711], [3.905,-2.724], [3.742,-2.756], [3.915,-2.766], [4.014,-2.797], [4.105,-2.848], [4.184,-2.916], [4.248,-2.998], [4.294,-3.092], [4.321,-3.192], [4.326,-3.297], [4.310,-3.400], [4.274,-3.498], [3.962,-4.138], [3.988,-4.103], [4.292,-4.127], [4.445,-3.963], [4.528,-3.937], [4.614,-3.926], [4.700,-3.932], [4.784,-3.954], [4.863,-3.990], [4.933,-4.041], [4.993,-4.104], [5.053,-4.202], [5.072,-4.481], [5.357,-4.765], [5.364,-4.794], [5.364,-4.144], [5.369,-4.013], [5.389,-3.916], [5.437,-3.795], [5.508,-3.687], [5.599,-3.595], [5.708,-3.523], [5.828,-3.474], [5.955,-3.450], [6.485,-3.449], [6.583,-3.465]],
    first_wall: [[6.266,-3.043], [7.282,-2.254], [7.900,-1.337], [8.306,-0.420], [8.394,0.635], [8.270,1.685], [7.906,2.465], [7.402,3.180], [6.552,3.926], [5.753,4.534], [4.935,4.720], [4.325,4.337], [4.125,3.585], [4.105,2.572], [4.105,1.557], [4.105,0.540], [4.105,-0.476], [4.105,-1.491], [4.105,-2.504], [3.942,-2.536], [4.115,-2.546], [4.214,-2.577], [4.305,-2.628], [4.384,-2.696], [4.448,-2.778], [4.494,-2.872], [4.521,-2.973], [4.526,-3.077], [4.510,-3.180], [4.474,-3.277], [4.162,-3.917], [4.188,-3.883], [4.492,-3.906], [4.645,-3.743], [4.728,-3.717], [4.814,-3.706], [4.900,-3.712], [4.984,-3.733], [5.063,-3.770], [5.133,-3.821], [5.193,-3.884], [5.253,-3.982], [5.272,-4.261], [5.557,-4.545], [5.564,-4.574], [5.564,-3.924], [5.569,-3.793], [5.589,-3.696], [5.637,-3.575], [5.708,-3.467], [5.800,-3.375], [5.908,-3.303], [6.028,-3.254], [6.155,-3.230], [6.265,-3.229], [6.363,-3.245]],
    coils: [
    {name: "Central Solenoid 3U (CS3U)", r: 1.687, z: 5.464, w: 0.740, h: 2.093},
    {name: "Central Solenoid 2U (CS2U)", r: 1.687, z: 3.278, w: 0.740, h: 2.093},
    {name: "Central Solenoid 1U (CS1U)", r: 1.687, z: 1.092, w: 0.740, h: 2.093},
    {name: "Central Solenoid 1L (CS1L)", r: 1.687, z: -1.072, w: 0.740, h: 2.093},
    {name: "Central Solenoid 2L (CS2L)", r: 1.687, z: -3.258, w: 0.740, h: 2.093},
    {name: "Central Solenoid 3L (CS3L)", r: 1.687, z: -5.444, w: 0.740, h: 2.093},
    {name: "Poloidal Field Coil 1 (PF1)", r: 3.943, z: 7.574, w: 0.959, h: 0.984},
    {name: "Poloidal Field Coil 2 (PF2)", r: 8.285, z: 6.540, w: 0.580, h: 0.715},
    {name: "Poloidal Field Coil 3 (PF3)", r: 11.992, z: 3.275, w: 0.696, h: 0.954},
    {name: "Poloidal Field Coil 4 (PF4)", r: 11.963, z: -2.234, w: 0.638, h: 0.954},
    {name: "Poloidal Field Coil 5 (PF5)", r: 8.391, z: -6.727, w: 0.812, h: 0.954},
    {name: "Poloidal Field Coil 6 (PF6)", r: 4.334, z: -7.466, w: 1.559, h: 1.107}
  ]
  };


  var D3D_CAD = {
    id: "d3d",
    name: "DIII-D National Fusion Facility",
    shortName: "DIII-D (General Atomics)",
    R0: 1.67, a: 0.67, B0: 2.17, ipNominal: 1.60,
    dist: 10.1,
    rMin: 0.55, rMax: 2.50, zMin: -1.70, zMax: 1.70,
    rTicks: [0.8, 1.3, 1.8, 2.3],
    zTicks: [-1.2, -0.6, 0.0, 0.6, 1.2],
    vessel_outer: [[0.876,-0.160], [0.876,1.124], [0.876,1.128], [0.876,1.161], [0.876,1.179], [0.876,1.237], [0.876,1.230], [0.876,1.256], [0.876,1.273], [0.876,1.298], [0.876,1.307], [0.872,1.325], [0.861,1.377], [0.889,1.377], [0.902,1.322], [0.906,1.322], [0.916,1.323], [0.957,1.325], [0.968,1.326], [0.976,1.326], [0.994,1.329], [1.008,1.332], [1.022,1.336], [1.041,1.343], [1.042,1.343], [1.045,1.345], [1.050,1.348], [1.055,1.351], [1.061,1.356], [1.069,1.362], [1.075,1.368], [1.082,1.374], [1.088,1.381], [1.094,1.391], [1.099,1.398], [1.102,1.404], [1.108,1.414], [1.118,1.438], [1.123,1.450], [1.140,1.491], [1.140,1.507], [1.140,1.508], [1.170,1.508], [1.188,1.508], [1.221,1.508], [1.240,1.508], [1.279,1.508], [1.279,1.470], [1.232,1.470], [1.232,1.452], [1.468,1.255], [1.507,1.237], [1.945,1.237], [2.230,1.200], [2.288,1.153], [2.405,0.869], [2.483,0.679], [2.537,0.549], [2.512,0.560], [2.511,0.497], [2.511,0.365], [2.514,0.228], [2.514,-0.160], [2.514,-0.228], [2.513,-0.365], [2.511,-0.497], [2.513,-0.560], [2.537,-0.549], [2.294,-1.133], [1.946,-1.334], [1.928,-1.371], [1.928,-1.410], [1.842,-1.410], [1.232,-1.410], [1.232,-1.489], [1.280,-1.489], [1.280,-1.523], [1.133,-1.523], [1.013,-1.523], [0.876,-1.383], [0.876,-1.383], [0.876,-0.990], [0.876,-0.960], [0.876,-0.575], [0.876,-0.560], [0.876,-0.161], [0.876,-0.160]],
    vessel_inner: [[0.946,-0.080], [0.946,1.044], [0.946,1.048], [0.946,1.081], [0.946,1.099], [0.946,1.157], [0.946,1.150], [0.946,1.176], [0.946,1.193], [0.946,1.218], [0.946,1.227], [0.942,1.245], [0.931,1.297], [0.959,1.297], [0.972,1.242], [0.976,1.242], [0.986,1.243], [1.027,1.244], [1.038,1.246], [1.046,1.246], [1.064,1.249], [1.078,1.252], [1.092,1.256], [1.111,1.263], [1.112,1.264], [1.115,1.265], [1.120,1.268], [1.125,1.271], [1.131,1.276], [1.139,1.282], [1.145,1.288], [1.152,1.294], [1.158,1.301], [1.164,1.311], [1.169,1.318], [1.172,1.324], [1.178,1.334], [1.188,1.358], [1.193,1.370], [1.210,1.411], [1.210,1.427], [1.210,1.428], [1.240,1.428], [1.258,1.428], [1.291,1.428], [1.310,1.428], [1.349,1.428], [1.349,1.390], [1.302,1.390], [1.302,1.372], [1.538,1.175], [1.577,1.157], [1.865,1.157], [2.150,1.120], [2.208,1.073], [2.325,0.789], [2.403,0.599], [2.457,0.469], [2.432,0.480], [2.431,0.417], [2.431,0.285], [2.434,0.148], [2.434,-0.080], [2.434,-0.148], [2.433,-0.285], [2.431,-0.417], [2.433,-0.480], [2.457,-0.469], [2.214,-1.053], [1.866,-1.254], [1.848,-1.291], [1.848,-1.330], [1.762,-1.330], [1.302,-1.330], [1.302,-1.409], [1.350,-1.409], [1.350,-1.443], [1.203,-1.443], [1.083,-1.443], [0.946,-1.303], [0.946,-1.303], [0.946,-0.910], [0.946,-0.880], [0.946,-0.495], [0.946,-0.480], [0.946,-0.081], [0.946,-0.080]],
    first_wall: [[1.016,0.000], [1.016,0.964], [1.016,0.968], [1.016,1.001], [1.016,1.019], [1.016,1.077], [1.016,1.070], [1.016,1.096], [1.016,1.113], [1.016,1.138], [1.016,1.147], [1.012,1.165], [1.001,1.217], [1.029,1.217], [1.042,1.162], [1.046,1.162], [1.056,1.163], [1.097,1.165], [1.108,1.166], [1.116,1.166], [1.134,1.169], [1.148,1.172], [1.162,1.176], [1.181,1.183], [1.182,1.183], [1.185,1.185], [1.190,1.188], [1.195,1.191], [1.201,1.196], [1.209,1.202], [1.215,1.208], [1.222,1.214], [1.228,1.221], [1.234,1.231], [1.239,1.238], [1.242,1.244], [1.248,1.254], [1.258,1.278], [1.263,1.290], [1.280,1.331], [1.280,1.347], [1.280,1.348], [1.310,1.348], [1.328,1.348], [1.361,1.348], [1.380,1.348], [1.419,1.348], [1.419,1.310], [1.372,1.310], [1.372,1.292], [1.608,1.095], [1.647,1.077], [1.785,1.077], [2.070,1.040], [2.128,0.993], [2.245,0.709], [2.323,0.519], [2.377,0.389], [2.352,0.400], [2.351,0.337], [2.351,0.205], [2.354,0.068], [2.354,0.000], [2.354,-0.068], [2.353,-0.205], [2.351,-0.337], [2.353,-0.400], [2.377,-0.389], [2.134,-0.973], [1.786,-1.174], [1.768,-1.211], [1.768,-1.250], [1.682,-1.250], [1.372,-1.250], [1.372,-1.329], [1.420,-1.329], [1.420,-1.363], [1.273,-1.363], [1.153,-1.363], [1.016,-1.223], [1.016,-1.223], [1.016,-0.830], [1.016,-0.800], [1.016,-0.415], [1.016,-0.400], [1.016,-0.001], [1.016,0.000]],
    coils: [
    {name: "OH0", r: 0.688, z: 0.038, w: 0.059, h: 0.074},
    {name: "OH1", r: 0.757, z: 0.038, w: 0.059, h: 0.074},
    {name: "OH2", r: 0.879, z: 1.712, w: 0.114, h: 0.029},
    {name: "OH3", r: 0.879, z: -1.713, w: 0.114, h: 0.029},
    {name: "OH4", r: 0.953, z: -1.945, w: 0.059, h: 0.056},
    {name: "OH5", r: 0.953, z: 1.946, w: 0.059, h: 0.056},
    {name: "PF6", r: 0.861, z: 0.168, w: 0.051, h: 0.321},
    {name: "PF7", r: 0.861, z: 0.508, w: 0.051, h: 0.321},
    {name: "PF8", r: 0.863, z: 0.849, w: 0.051, h: 0.321},
    {name: "PF9", r: 0.861, z: 1.190, w: 0.051, h: 0.321},
    {name: "PF13", r: 1.252, z: 1.602, w: 0.235, h: 0.085},
    {name: "PF14", r: 1.689, z: 1.587, w: 0.169, h: 0.133},
    {name: "PF15", r: 0.861, z: -0.174, w: 0.051, h: 0.321},
    {name: "PF16", r: 0.861, z: -0.513, w: 0.051, h: 0.321},
    {name: "PF17", r: 0.861, z: -0.854, w: 0.051, h: 0.321},
    {name: "PF18", r: 0.863, z: -1.196, w: 0.051, h: 0.321},
    {name: "PF22", r: 1.252, z: -1.603, w: 0.235, h: 0.085},
    {name: "PF23", r: 1.689, z: -1.578, w: 0.169, h: 0.133}
  ]
  };


  var MACHINES = {
    tcv: TCV_CAD,
    arc: ARC_CAD,
    iter: ITER_CAD,
    d3d: D3D_CAD
  };

  // Exact 32 shape_70166 reference LCFS control points extracted from arXiv:2505.00663 Fig. 9 (state0_refs.pdf)
  var SHAPE_70166_REFS = [
    [0.6250,  0.1943], [0.6299,  0.1413], [0.6393,  0.0889], [0.6551,  0.0380],
    [0.6803, -0.0088], [0.7187, -0.0458], [0.7685, -0.0641], [0.8217, -0.0638],
    [0.8731, -0.0502], [0.9215, -0.0281], [0.9663,  0.0005], [1.0070,  0.0346],
    [1.0428,  0.0741], [1.0721,  0.1186], [1.0926,  0.1679], [1.1021,  0.2205],
    [1.0989,  0.2737], [1.0837,  0.3250], [1.0586,  0.3720], [1.0258,  0.4140],
    [0.9875,  0.4509], [0.9446,  0.4824], [0.8981,  0.5079], [0.8481,  0.5262],
    [0.7955,  0.5340], [0.7429,  0.5257], [0.6981,  0.4973], [0.6664,  0.4545],
    [0.6464,  0.4051], [0.6340,  0.3533], [0.6270,  0.3006], [0.6241,  0.2475]
  ];
  var SHAPE_70166_AXIS = [0.8788, 0.2339];
  var SHAPE_70166_LIMIT = [0.6240, 0.2339];

  // Verified 10-Seed x 10,000-Round Paired TORAX 2.0.0 TuRBO-zero Benchmark Data (D = 160,531, 93->256->256->256->19)
  var ROUNDS = [0, 300, 1000, 2000, 3000, 5000, 8000, 10000];
  var BENCH = {
    shared: {
      name: "Shared TuRBO-zero (ellipsoid-ball)",
      short: "Shared TuRBO",
      color: "#d97706",
      accent: "#f59e0b",
      mean: [5.97, 53.36, 70.77, 81.33, 83.25, 87.65, 90.52, 91.57],
      sem:  [1.13,  5.90,  5.95,  2.63,  2.60,  1.81,  0.65,  0.55],
      seeds10k: [91.13, 91.80, 87.78, 90.24, 91.99, 92.76, 90.46, 92.86, 93.37, 93.26],
      dR: 0.0015, dZ: -0.002, dKappa: -0.01, dTri: 0.00,
      comps: [0.952, 0.964, 0.892, 0.948, 0.885],
      te0: 14.8, ti0: 12.1, ne0: 1.02, ip: 1.49, q95: 3.24,
      coils: [0.62, 0.58, 0.44, 0.21, -0.18, -0.45, -0.68, -0.52, 0.49, 0.64, 0.35, -0.12, -0.38, -0.61, 0.72, 0.55, -0.42, -0.41, 0.14]
    },
    indep: {
      name: "Independent TuRBO-zero (origin-only)",
      short: "Indep. TuRBO",
      color: "#2563eb",
      accent: "#3b82f6",
      mean: [5.97, 52.38, 71.39, 80.29, 80.93, 86.28, 88.60, 89.41],
      sem:  [1.13,  5.13,  4.74,  3.08,  2.97,  1.14,  0.81,  0.97],
      seeds10k: [89.42, 90.95, 85.61, 93.36, 92.40, 88.44, 83.77, 89.61, 88.16, 92.36],
      dR: 0.0045, dZ: -0.006, dKappa: -0.03, dTri: -0.02,
      comps: [0.924, 0.941, 0.865, 0.928, 0.860],
      te0: 14.2, ti0: 11.6, ne0: 0.99, ip: 1.47, q95: 3.19,
      coils: [0.55, 0.51, 0.39, 0.18, -0.15, -0.41, -0.62, -0.48, 0.44, 0.58, 0.31, -0.15, -0.34, -0.55, 0.66, 0.49, -0.39, -0.44, 0.19]
    },
    init: {
      name: "Initial Random Policy θ₀ (Round 0)",
      short: "Initial θ₀",
      color: "#dc2626",
      accent: "#ef4444",
      mean: [5.97, 5.97, 5.97, 5.97, 5.97, 5.97, 5.97, 5.97],
      sem:  [1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13],
      seeds10k: [3.95, 6.14, 1.59, 8.59, 7.93, 12.63, 1.37, 7.33, 6.74, 3.46],
      dR: -0.048, dZ: -0.135, dKappa: -0.34, dTri: -0.25,
      comps: [0.062, 0.115, 0.084, 0.190, 0.045],
      te0: 3.2, ti0: 2.6, ne0: 0.45, ip: 0.72, q95: 2.12,
      coils: [-0.12, 0.08, -0.19, 0.24, -0.05, 0.14, -0.22, 0.11, -0.18, 0.09, 0.21, -0.15, 0.07, -0.13, 0.16, -0.09, -0.82, 0.44, -0.68]
    }
  };

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function evalState(polKey, round, seedIdx) {
    var pol = Object.prototype.hasOwnProperty.call(BENCH, polKey) ? BENCH[polKey] : BENCH.shared;
    var r = Math.max(0, Math.min(10000, round));
    var idx = 0;
    while (idx < ROUNDS.length - 1 && ROUNDS[idx + 1] <= r) idx++;
    var t = (idx >= ROUNDS.length - 1) ? 1.0 : (r - ROUNDS[idx]) / (ROUNDS[idx + 1] - ROUNDS[idx]);

    var scoreMean = lerp(pol.mean[idx], pol.mean[Math.min(ROUNDS.length - 1, idx + 1)], t);
    var semVal = lerp(pol.sem[idx], pol.sem[Math.min(ROUNDS.length - 1, idx + 1)], t);
    var score = scoreMean;
    if (seedIdx >= 0 && seedIdx < pol.seeds10k.length) {
      var finalScore = pol.seeds10k[seedIdx];
      var seedScale = (finalScore - pol.mean[0]) / Math.max(1e-3, pol.mean[pol.mean.length - 1] - pol.mean[0]);
      score = pol.mean[0] + (scoreMean - pol.mean[0]) * seedScale;
    }

    var progress = Math.min(1.0, Math.max(0.0, (scoreMean - pol.mean[0]) / Math.max(1e-3, 91.57 - pol.mean[0])));
    var initPol = BENCH.init;

    var dR = lerp(initPol.dR, pol.dR, progress);
    var dZ = lerp(initPol.dZ, pol.dZ, progress);
    var dKappa = lerp(initPol.dKappa, pol.dKappa, progress);
    var dTri = lerp(initPol.dTri, pol.dTri, progress);
    var te0 = lerp(initPol.te0, pol.te0, progress);
    var ti0 = lerp(initPol.ti0, pol.ti0, progress);
    var ne0 = lerp(initPol.ne0, pol.ne0, progress);
    var ip = lerp(initPol.ip, pol.ip, progress);
    var q95 = lerp(initPol.q95, pol.q95, progress);

    var comps = [];
    for (var c = 0; c < 5; c++) {
      comps.push(lerp(initPol.comps[c], pol.comps[c], progress));
    }

    var coils = [];
    for (var ci = 0; ci < initPol.coils.length; ci++) {
      coils.push(lerp(initPol.coils[ci], pol.coils[ci], progress));
    }

    var lcfsErrMm = lerp(48.2, (polKey === "shared" ? 2.1 : (polKey === "indep" ? 6.4 : 48.2)), progress);

    return {
      polKey: polKey,
      policy: pol,
      round: r,
      seedIdx: seedIdx,
      score: score,
      sem: semVal,
      dR: dR,
      dZ: dZ,
      dKappa: dKappa,
      dTri: dTri,
      te0: te0,
      ti0: ti0,
      ne0: ne0,
      ip: ip,
      q95: q95,
      comps: comps,
      coils: coils,
      lcfsErrMm: lcfsErrMm
    };
  }

  function plasmaContour(rho, st, nPts, mach) {
    mach = mach || TCV_CAD;
    var pts = [];
    var R0 = mach.R0 + (mach.id === "tcv" ? st.dR * 0.7 : 0.0);
    var Z0 = (mach.id === "tcv" ? (-0.045 + st.dZ * 0.8) : (mach.id === "iter" ? 0.35 : 0.0));
    var a = (mach.a * 0.86) * rho;
    var baseKappa = (mach.id === "tcv") ? (1.58 + st.dKappa * 0.45) : (mach.id === "arc" ? 1.78 : (mach.id === "iter" ? 1.82 : 1.75));
    var kappa = baseKappa * (0.78 + 0.22 * rho);
    var deltaU = ((mach.id === "tcv" ? (0.36 + st.dTri * 0.4) : 0.38)) * rho;
    var deltaL = ((mach.id === "tcv" ? (0.52 + st.dTri * 0.4) : 0.48)) * rho;
    var xptPull = (rho > 0.55) ? Math.pow((rho - 0.55) / 0.45, 2.2) * (mach.a * 0.32) : 0.0;
    for (var i = 0; i < nPts; i++) {
      var th = (2.0 * Math.PI * i) / nPts;
      var s = Math.sin(th);
      var c = Math.cos(th);
      var d = (s >= 0.0) ? deltaU : deltaL;
      var r = R0 + a * Math.cos(th + Math.asin(Math.max(-0.95, Math.min(0.95, d))) * s);
      var z = Z0 + kappa * a * s;
      var dTh = Math.atan2(Math.sin(th + 1.78), Math.cos(th + 1.78));
      var cusp = Math.exp(-(dTh * dTh) / 0.16);
      r -= xptPull * 0.42 * cusp;
      z -= xptPull * 1.05 * cusp;
      pts.push([r, z]);
    }
    var rx = R0 - mach.a * 0.42;
    var zx = Z0 - kappa * a - xptPull * 0.8;
    return { pts: pts, R0: R0, Z0: Z0, Rx: rx, Zx: zx };
  }

  // Deform the exact 32-point shape_70166 reference contour from Fig. 9 according to policy state
  function plasmaContourFig9(rho, st) {
    var R0 = SHAPE_70166_AXIS[0] + st.dR;
    var Z0 = SHAPE_70166_AXIS[1] + st.dZ;
    var kScale = 1.0 + st.dKappa * 0.55;
    var rScale = 1.0 + st.dTri * 0.25;
    var pts = [];
    for (var i = 0; i < SHAPE_70166_REFS.length; i++) {
      var refR = SHAPE_70166_REFS[i][0];
      var refZ = SHAPE_70166_REFS[i][1];
      var dR = (refR - SHAPE_70166_AXIS[0]) * rho * rScale;
      var dZ = (refZ - SHAPE_70166_AXIS[1]) * rho * kScale;
      pts.push([R0 + dR, Z0 + dZ]);
    }
    return { pts: pts, R0: R0, Z0: Z0 };
  }

  var VS_CAD = [
    "attribute vec3 aPos;",
    "attribute vec3 aNorm;",
    "attribute vec4 aCol;",
    "uniform mat4 uMVP;",
    "uniform mat4 uModel;",
    "uniform vec3 uCamPos;",
    "uniform float uUnlit;",
    "varying vec4 vCol;",
    "varying vec3 vNorm;",
    "varying vec3 vViewDir;",
    "varying vec3 vWorldPos;",
    "varying float vUnlit;",
    "void main() {",
    "  vec4 wp = uModel * vec4(aPos, 1.0);",
    "  vWorldPos = wp.xyz;",
    "  vNorm = normalize(mat3(uModel) * aNorm);",
    "  vViewDir = normalize(uCamPos - wp.xyz);",
    "  vCol = aCol;",
    "  vUnlit = uUnlit;",
    "  gl_Position = uMVP * wp;",
    "}"
  ].join("\n");

  var FS_CAD = [
    "precision mediump float;",
    "varying vec4 vCol;",
    "varying vec3 vNorm;",
    "varying vec3 vViewDir;",
    "varying vec3 vWorldPos;",
    "varying float vUnlit;",
    "void main() {",
    "  if (vUnlit > 0.5) {",
    "    gl_FragColor = vCol;",
    "    return;",
    "  }",
    "  vec3 n = normalize(vNorm);",
    "  vec3 v = normalize(vViewDir);",
    "  vec3 l1 = normalize(vec3(0.55, 0.85, 0.45));",
    "  vec3 l2 = normalize(vec3(-0.65, 0.35, -0.65));",
    "  float diff1 = max(dot(n, l1), 0.0);",
    "  float diff2 = max(dot(n, l2), 0.0) * 0.35;",
    "  vec3 h1 = normalize(l1 + v);",
    "  float spec = pow(max(dot(n, h1), 0.0), 32.0) * 0.28;",
    "  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.15;",
    "  vec3 ambient = vec3(0.32, 0.34, 0.37);",
    "  vec3 lit = vCol.rgb * (ambient + diff1 * 0.65 + diff2) + vec3(spec + fres);",
    "  gl_FragColor = vec4(lit, vCol.a);",
    "}"
  ].join("\n");

  function mat4Identity() {
    return new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1
    ]);
  }

  function mat4Mul(a, b) {
    var out = new Float32Array(16);
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < 4; j++) {
        var sum = 0.0;
        for (var k = 0; k < 4; k++) {
          sum += a[k * 4 + i] * b[j * 4 + k];
        }
        out[j * 4 + i] = sum;
      }
    }
    return out;
  }

  function mat4Perspective(fovy, aspect, near, far) {
    var f = 1.0 / Math.tan(fovy / 2.0);
    var nf = 1.0 / (near - far);
    var out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) * nf;
    out[11] = -1.0;
    out[14] = 2.0 * far * near * nf;
    return out;
  }

  function mat4LookAt(eye, center, up) {
    var ex = eye[0], ey = eye[1], ez = eye[2];
    var cx = center[0], cy = center[1], cz = center[2];
    var ux = up[0], uy = up[1], uz = up[2];
    var z0 = ex - cx, z1 = ey - cy, z2 = ez - cz;
    var lenZ = Math.hypot(z0, z1, z2) || 1.0;
    z0 /= lenZ; z1 /= lenZ; z2 /= lenZ;
    var x0 = uy * z2 - uz * z1;
    var x1 = uz * z0 - ux * z2;
    var x2 = ux * z1 - uy * z0;
    var lenX = Math.hypot(x0, x1, x2) || 1.0;
    x0 /= lenX; x1 /= lenX; x2 /= lenX;
    var y0 = z1 * x2 - z2 * x1;
    var y1 = z2 * x0 - z0 * x2;
    var y2 = z0 * x1 - z1 * x0;
    var out = new Float32Array(16);
    out[0] = x0; out[4] = x1; out[8] = x2; out[12] = -(x0 * ex + x1 * ey + x2 * ez);
    out[1] = y0; out[5] = y1; out[9] = y2; out[13] = -(y0 * ex + y1 * ey + y2 * ez);
    out[2] = z0; out[6] = z1; out[10] = z2; out[14] = -(z0 * ex + z1 * ey + z2 * ez);
    out[3] = 0.0; out[7] = 0.0; out[11] = 0.0; out[15] = 1.0;
    return out;
  }

  function cylToXYZ(R, Z, phi, off) {
    off = off || 0.0;
    var c = Math.cos(phi);
    var s = Math.sin(phi);
    return [R * c - off * s, Z, R * s + off * c];
  }

  // Build the complete 3D CAD mesh buffers for active tokamak geometry
  function buildTokamakCADBuffers(gl, st, opts) {
    var mKey = opts.machine || "tcv";
    var mach = Object.prototype.hasOwnProperty.call(MACHINES, mKey) ? MACHINES[mKey] : MACHINES.tcv;
    var pos = [];
    var norm = [];
    var col = [];
    var linePos = [];
    var lineCol = [];

    function pushVert(p, n, c) {
      pos.push(p[0], p[1], p[2]);
      norm.push(n[0], n[1], n[2]);
      col.push(c[0], c[1], c[2], c[3]);
    }
    function pushTri(p1, p2, p3, n1, n2, n3, c1, c2, c3) {
      pushVert(p1, n1, c1);
      pushVert(p2, n2, c2 || c1);
      pushVert(p3, n3, c3 || c1);
    }
    function pushLine(p1, p2, c) {
      linePos.push(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      lineCol.push(c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3]);
    }

    function addCutawayFan(polyRZ, centerRZ, phi, signNorm, dn, colorFn, edgeCol) {
      var nx = -Math.sin(phi) * signNorm;
      var nz =  Math.cos(phi) * signNorm;
      var n = [nx * 0.55, 0.35, nz * 0.55 - 0.50];
      var pc = cylToXYZ(centerRZ[0], centerRZ[1], phi, dn);
      var cc = (typeof colorFn === "function") ? colorFn(centerRZ[0], centerRZ[1]) : colorFn;
      for (var i = 0; i < polyRZ.length; i++) {
        var a = polyRZ[i];
        var b = polyRZ[(i + 1) % polyRZ.length];
        var pa = cylToXYZ(a[0], a[1], phi, dn);
        var pb = cylToXYZ(b[0], b[1], phi, dn);
        var ca = (typeof colorFn === "function") ? colorFn(a[0], a[1]) : colorFn;
        var cb = (typeof colorFn === "function") ? colorFn(b[0], b[1], i) : colorFn;
        if (signNorm > 0) {
          pushTri(pc, pa, pb, n, n, n, cc, ca, cb);
        } else {
          pushTri(pc, pb, pa, n, n, n, cc, cb, ca);
        }
        if (edgeCol) {
          var paE = cylToXYZ(a[0], a[1], phi, dn + signNorm * 0.0012);
          var pbE = cylToXYZ(b[0], b[1], phi, dn + signNorm * 0.0012);
          pushLine(paE, pbE, edgeCol);
        }
      }
    }

    function addRevolvedSolid(polyRZ, phi0, phi1, nSeg, sideCol, capCol, edgeCol, capCenter) {
      var m = polyRZ.length;
      var dPhi = (phi1 - phi0) / nSeg;
      for (var s = 0; s < nSeg; s++) {
        var a0 = phi0 + s * dPhi;
        var a1 = a0 + dPhi;
        var c0 = Math.cos(a0), s0 = Math.sin(a0);
        var c1 = Math.cos(a1), s1 = Math.sin(a1);
        for (var i = 0; i < m; i++) {
          var v0 = polyRZ[i];
          var v1 = polyRZ[(i + 1) % m];
          var dr = v1[0] - v0[0];
          var dz = v1[1] - v0[1];
          var len = Math.hypot(dr, dz) || 1.0;
          var nr = dz / len;
          var nz = -dr / len;
          var n0 = [nr * c0, nz, nr * s0];
          var n1 = [nr * c1, nz, nr * s1];
          var p00 = [v0[0] * c0, v0[1], v0[0] * s0];
          var p10 = [v1[0] * c0, v1[1], v1[0] * s0];
          var p11 = [v1[0] * c1, v1[1], v1[0] * s1];
          var p01 = [v0[0] * c1, v0[1], v0[0] * s1];
          pushTri(p00, p10, p11, n0, n0, n1, sideCol, sideCol, sideCol);
          pushTri(p00, p11, p01, n0, n1, n1, sideCol, sideCol, sideCol);
        }
      }
      if (capCol) {
        var ctr = capCenter || [
          0.5 * (polyRZ[0][0] + polyRZ[Math.floor(m / 2)][0]),
          0.5 * (polyRZ[0][1] + polyRZ[Math.floor(m / 2)][1])
        ];
        addCutawayFan(polyRZ, ctr, phi0, -1.0, 0.0, capCol, edgeCol);
        addCutawayFan(polyRZ, ctr, phi1,  1.0, 0.0, capCol, edgeCol);
      }
    }

    function addHollowShell(outerRZ, innerRZ, phi0, phi1, nSeg, outerCol, innerCol, rimCol) {
      var m = Math.min(outerRZ.length, innerRZ.length);
      var dPhi = (phi1 - phi0) / nSeg;
      for (var s = 0; s < nSeg; s++) {
        var a0 = phi0 + s * dPhi;
        var a1 = a0 + dPhi;
        var c0 = Math.cos(a0), s0 = Math.sin(a0);
        var c1 = Math.cos(a1), s1 = Math.sin(a1);
        for (var i = 0; i < m; i++) {
          var o0 = outerRZ[i], o1 = outerRZ[(i + 1) % m];
          var dr = o1[0] - o0[0], dz = o1[1] - o0[1];
          var len = Math.hypot(dr, dz) || 1.0;
          var nr = dz / len, nz = -dr / len;
          var n0 = [nr * c0, nz, nr * s0], n1 = [nr * c1, nz, nr * s1];
          pushTri([o0[0]*c0, o0[1], o0[0]*s0], [o1[0]*c0, o1[1], o1[0]*s0], [o1[0]*c1, o1[1], o1[0]*s1], n0, n0, n1, outerCol, outerCol, outerCol);
          pushTri([o0[0]*c0, o0[1], o0[0]*s0], [o1[0]*c1, o1[1], o1[0]*s1], [o0[0]*c1, o0[1], o0[0]*s1], n0, n1, n1, outerCol, outerCol, outerCol);

          var k0 = innerRZ[i], k1 = innerRZ[(i + 1) % m];
          var ni0 = [-nr * c0, -nz, -nr * s0], ni1 = [-nr * c1, -nz, -nr * s1];
          pushTri([k0[0]*c0, k0[1], k0[0]*s0], [k1[0]*c1, k1[1], k1[0]*s1], [k1[0]*c0, k1[1], k1[0]*s0], ni0, ni1, ni0, innerCol, innerCol, innerCol);
          pushTri([k0[0]*c0, k0[1], k0[0]*s0], [k0[0]*c1, k0[1], k0[0]*s1], [k1[0]*c1, k1[1], k1[0]*s1], ni0, ni1, ni1, innerCol, innerCol, innerCol);
        }
      }
      if (rimCol) {
        var nUp = [0.0, 0.35, -0.94];
        for (var endIdx = 0; endIdx < 2; endIdx++) {
          var phi = (endIdx === 0) ? phi0 : phi1;
          var signN = (endIdx === 0) ? -1.0 : 1.0;
          for (var i = 0; i < m; i++) {
            var o0 = outerRZ[i], o1 = outerRZ[(i + 1) % m];
            var k0 = innerRZ[i], k1 = innerRZ[(i + 1) % m];
            var po0 = cylToXYZ(o0[0], o0[1], phi, 0.002 * signN);
            var po1 = cylToXYZ(o1[0], o1[1], phi, 0.002 * signN);
            var pk1 = cylToXYZ(k1[0], k1[1], phi, 0.002 * signN);
            var pk0 = cylToXYZ(k0[0], k0[1], phi, 0.002 * signN);
            pushTri(po0, po1, pk1, nUp, nUp, nUp, rimCol, rimCol, rimCol);
            pushTri(po0, pk1, pk0, nUp, nUp, nUp, rimCol, rimCol, rimCol);
          }
        }
      }
    }

    function addToroidalBoxRing(rIn, rOut, zBot, zTop, phi0, phi1, nSeg, sideCol, capCol) {
      var poly = [
        [rIn, zBot], [rOut, zBot], [rOut, zTop], [rIn, zTop]
      ];
      addRevolvedSolid(poly, phi0, phi1, nSeg, sideCol, capCol, null, [0.5 * (rIn + rOut), 0.5 * (zBot + zTop)]);
    }

    var deg = Math.PI / 180.0;
    var scaleCut = (opts.cutawayDeg || 56.0) / 56.0;
    var phiInner0  = (-90.0 + 30.0 * scaleCut) * deg;
    var phiInner1  = (270.0 - 30.0 * scaleCut) * deg;
    var phiOuter0  = (-90.0 + 44.0 * scaleCut) * deg;
    var phiOuter1  = (270.0 - 44.0 * scaleCut) * deg;
    var phiVessel0 = (-90.0 + 58.0 * scaleCut) * deg;
    var phiVessel1 = (270.0 - 58.0 * scaleCut) * deg;

    // 1. Render All Poloidal Field & Solenoid Magnetic Coils
    if (opts.showCoils && mach.coils) {
      for (var ci = 0; ci < mach.coils.length; ci++) {
        var c = mach.coils[ci];
        var rIn = Math.max(0.05, c.r - c.w * 0.5);
        var rOut = c.r + c.w * 0.5;
        var zBot = c.z - c.h * 0.5;
        var zTop = c.z + c.h * 0.5;
        var isInner = (c.r < mach.R0);
        var isCS = c.name && (c.name.indexOf("cs") !== -1 || c.name.indexOf("CS") !== -1 || c.name.indexOf("OH") !== -1 || c.name.indexOf("Central") !== -1);
        var act = (mach.id === "tcv" && ci < st.coils.length) ? Math.abs(st.coils[ci]) : 0.35;
        var cBody, cCap;
        if (isCS) {
          cBody = [0.42 + 0.06 * act, 0.27 + 0.05 * act, 0.48 + 0.07 * act, 1.0];
          cCap  = [0.56 + 0.07 * act, 0.37 + 0.06 * act, 0.63 + 0.07 * act, 1.0];
        } else if (isInner) {
          cBody = [0.24 + 0.07 * act, 0.37 + 0.10 * act, 0.54 + 0.12 * act, 1.0];
          cCap  = [0.32 + 0.08 * act, 0.47 + 0.11 * act, 0.66 + 0.12 * act, 1.0];
        } else {
          cBody = [0.68 + 0.10 * act, 0.47 + 0.09 * act, 0.14 + 0.04 * act, 1.0];
          cCap  = [0.82 + 0.10 * act, 0.58 + 0.09 * act, 0.20 + 0.04 * act, 1.0];
        }
        var p0 = isInner ? phiInner0 : phiOuter0;
        var p1 = isInner ? phiInner1 : phiOuter1;
        addToroidalBoxRing(rIn, rOut, zBot, zTop, p0, p1, 48, cBody, cCap);
      }
    }

    // 2. Hollow 3D Vacuum Vessel Shell (Exterior Steel, Charcoal Interior, White Cutaway Rim)
    var vesselShellCol = [0.60, 0.62, 0.65, 1.0];
    var vesselInnerCol = [0.15, 0.16, 0.17, 1.0];
    var whiteRimCol    = [0.97, 0.98, 1.00, 1.0];
    addHollowShell(mach.vessel_outer, mach.vessel_inner, phiVessel0, phiVessel1, 64, vesselShellCol, vesselInnerCol, whiteRimCol);

    // 3. Cutaway Faces & Protruding Plasma Torus
    var eq = plasmaContour(1.0, st, 64, mach);

    function buildCutawayFace(phiVesselPlane, signN, isLeftProtruding) {
      var phiSolPlane = isLeftProtruding ? (phiVesselPlane - 9.5 * deg) : phiVesselPlane;
      var phiPlasmaPlane = isLeftProtruding ? (phiVesselPlane - 13.5 * deg) : phiVesselPlane;

      // Layer A: White vessel cutaway rim
      addCutawayFan(
        mach.vessel_outer, [mach.R0, 0.0], phiVesselPlane, signN, 0.001 * signN,
        whiteRimCol, [0.16, 0.17, 0.19, 0.85]
      );

      // Layer B: Dark interior first-wall cavity
      addCutawayFan(
        mach.first_wall, [mach.R0, 0.0], phiVesselPlane, signN, 0.004 * signN,
        [0.15, 0.16, 0.17, 1.0], [0.93, 0.95, 0.97, 0.95]
      );

      // Layer C: 3D Scrape-Off Layer & Plasma Torus jutting forward on left
      if (isLeftProtruding) {
        addRevolvedSolid(
          eq.pts, phiPlasmaPlane, phiVesselPlane, 18,
          [0.95, 0.52, 0.09, 1.0], null, null, [eq.R0, eq.Z0]
        );
      }

      // Layer D: Nested Plasma Flux Surfaces at phiPlasmaPlane
      var fluxBands = [
        { rho: 1.00, col: [0.98, 0.56, 0.10, 1.0], edge: isLeftProtruding ? null : [0.30, 0.88, 0.94, 0.95] },
        { rho: 0.80, col: [1.00, 0.70, 0.22, 1.0], edge: null },
        { rho: 0.60, col: [1.00, 0.82, 0.42, 1.0], edge: null },
        { rho: 0.42, col: [1.00, 0.92, 0.68, 1.0], edge: null },
        { rho: 0.22, col: [1.00, 0.98, 0.92, 1.0], edge: null }
      ];
      for (var b = 0; b < fluxBands.length; b++) {
        var band = fluxBands[b];
        var cData = plasmaContour(band.rho, st, 56, mach);
        addCutawayFan(
          cData.pts, [cData.R0, cData.Z0], phiPlasmaPlane, signN, (0.008 + 0.0018 * b) * signN,
          band.col, band.edge
        );
      }

      // Layer E: Cyan Divertor Strike Lines
      var pX = cylToXYZ(eq.Rx, eq.Zx, phiVesselPlane, 0.0085 * signN);
      var pLegIn = cylToXYZ(eq.Rx - mach.a * 0.18, eq.Zx - mach.a * 0.35, phiVesselPlane, 0.0085 * signN);
      var pLegOut = cylToXYZ(eq.Rx + mach.a * 0.32, eq.Zx - mach.a * 0.32, phiVesselPlane, 0.0085 * signN);
      var cyanLine = [0.30, 0.88, 0.94, 1.0];
      pushLine(pX, pLegIn, cyanLine);
      pushLine(pX, pLegOut, cyanLine);
    }

    buildCutawayFace(phiVessel0, -1.0, true);
    buildCutawayFace(phiVessel1,  1.0, false);

    // 4. Helical Magnetic Field Lines
    if (opts.showFieldLines) {
      var nLines = 3;
      for (var l = 0; l < nLines; l++) {
        var th0 = (2.0 * Math.PI * l) / nLines;
        var prevPt = null;
        var steps = 120;
        var pStart = phiVessel0;
        var pEnd = phiVessel1 - 11.0 * deg;
        for (var k = 0; k <= steps; k++) {
          var phi = pEnd + (pStart - pEnd) * (k / steps);
          var th = th0 + st.q95 * (phi - pEnd);
          var idx = ((Math.floor((th / (2.0 * Math.PI)) * eq.pts.length) % eq.pts.length) + eq.pts.length) % eq.pts.length;
          var rz = eq.pts[idx];
          var pt = cylToXYZ(rz[0] * 1.003, rz[1], phi, 0.0);
          if (prevPt) {
            pushLine(prevPt, pt, [1.0, 0.92, 0.55, 0.90]);
          }
          prevPt = pt;
        }
      }
    }

    function makeBuf(arr) {
      var b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
      return b;
    }

    return {
      posBuf: makeBuf(pos),
      normBuf: makeBuf(norm),
      colBuf: makeBuf(col),
      triCount: pos.length / 3,
      linePosBuf: makeBuf(linePos),
      lineColBuf: makeBuf(lineCol),
      lineCount: linePos.length / 3
    };
  }

  // --- Center Panel: Poloidal Equilibrium Cross Section & Flux Surfaces ---
  function renderFigure9Canvas(canvas, st, mach) {
    mach = mach || MACHINES.tcv;
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    var w = Math.max(1, Math.round(rect.width)) || 320;
    var h = Math.max(1, Math.round(rect.height)) || 280;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    var ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);

    var isDark = document.documentElement.getAttribute("data-theme") === "dark";
    ctx.fillStyle = isDark ? "#0d1117" : "#ffffff";
    ctx.fillRect(0, 0, w, h);

    var compact = w < 420;
    var ultraCompact = w < 330;
    var padTop = compact ? 24 : 30;
    var padBot = compact ? 30 : 38;
    var padLeft = ultraCompact ? 33 : (compact ? 38 : 44);
    var padRight = ultraCompact ? 7 : (compact ? 10 : 14);
    var gap = ultraCompact ? 10 : (compact ? 14 : 18);
    var availH = Math.max(80, h - padTop - padBot);
    var availW = Math.max(60, (w - padLeft - padRight - gap) * 0.5);

    var rMin = mach.rMin, rMax = mach.rMax, zMin = mach.zMin, zMax = mach.zMax;
    var dR = rMax - rMin, dZ = zMax - zMin;
    var geomAspect = dZ / dR;
    var boxH = Math.min(availH, availW * geomAspect);
    var boxW = boxH / geomAspect;
    var totalW = 2 * boxW + gap;
    var startX = padLeft + Math.floor((w - padLeft - padRight - totalW) * 0.5);
    var startY = padTop + Math.floor((availH - boxH) * 0.5);

    function rzToXY(boxX, r, z) {
      var x = boxX + ((r - rMin) / dR) * boxW;
      var y = startY + ((zMax - z) / dZ) * boxH;
      return [x, y];
    }

    var tickFont = ultraCompact
      ? "400 8px 'Latin Modern Mono', monospace"
      : (compact ? "400 8.8px 'Latin Modern Mono', monospace" : "400 9.5px 'Latin Modern Mono', monospace");
    var axisFont = compact
      ? "italic 10px 'Latin Modern Roman', Georgia, serif"
      : "italic 11px 'Latin Modern Roman', Georgia, serif";
    var titleFont = ultraCompact
      ? "700 10px 'Latin Modern Roman', Georgia, serif"
      : (compact ? "700 11.2px 'Latin Modern Roman', Georgia, serif" : "700 13px 'Latin Modern Roman', Georgia, serif");

    function drawBoxTicks(boxX, showZLabels) {
      ctx.save();
      ctx.strokeStyle = isDark ? "#8b949e" : "#111111";
      ctx.fillStyle = isDark ? "#8b949e" : "#222222";
      ctx.font = tickFont;
      ctx.lineWidth = 0.9;

      var rTicks = mach.rTicks || [mach.R0];
      ctx.textAlign = "center";
      for (var i = 0; i < rTicks.length; i++) {
        if (ultraCompact && rTicks.length >= 4 && (i % 2 === 1)) continue;
        var pt = rzToXY(boxX, rTicks[i], zMin);
        ctx.beginPath();
        ctx.moveTo(pt[0], startY + boxH);
        ctx.lineTo(pt[0], startY + boxH + 3.5);
        ctx.stroke();
        ctx.fillText(rTicks[i].toFixed(1), pt[0], startY + boxH + (compact ? 11.5 : 13));
      }

      ctx.font = axisFont;
      ctx.fillText("R [m]", boxX + boxW * 0.5, startY + boxH + (compact ? 23.5 : 28));

      var zTicks = mach.zTicks || [0.0];
      ctx.textAlign = "right";
      ctx.font = tickFont;
      for (var j = 0; j < zTicks.length; j++) {
        var pz = rzToXY(boxX, rMin, zTicks[j]);
        ctx.beginPath();
        ctx.moveTo(boxX - 3.5, pz[1]);
        ctx.lineTo(boxX, pz[1]);
        ctx.stroke();
        if (showZLabels) {
          ctx.fillText((zTicks[j] > 0 ? "+" : "") + zTicks[j].toFixed(1), boxX - 4.5, pz[1] + 3);
        }
      }

      if (showZLabels) {
        ctx.save();
        ctx.translate(boxX - (ultraCompact ? 24 : (compact ? 28 : 32)), startY + boxH * 0.5);
        ctx.rotate(-Math.PI * 0.5);
        ctx.font = axisFont;
        ctx.textAlign = "center";
        ctx.fillText("Z [m]", 0, 0);
        ctx.restore();
      }
      ctx.restore();
    }

    function drawCoils(boxX) {
      if (!mach.coils) return;
      ctx.save();
      for (var ci = 0; ci < mach.coils.length; ci++) {
        var c = mach.coils[ci];
        var pTL = rzToXY(boxX, c.r - c.w * 0.5, c.z + c.h * 0.5);
        var pBR = rzToXY(boxX, c.r + c.w * 0.5, c.z - c.h * 0.5);
        var cW = Math.max(2, pBR[0] - pTL[0]);
        var cH = Math.max(2, pBR[1] - pTL[1]);
        ctx.fillStyle = (c.r < mach.R0) ? "#4a5568" : "#d97706";
        ctx.fillRect(pTL[0], pTL[1], cW, cH);
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 0.8;
        ctx.strokeRect(pTL[0], pTL[1], cW, cH);
      }
      ctx.restore();
    }

    function drawMachineProfiles(boxX) {
      ctx.save();
      // Vessel Outer Shell
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (var i = 0; i < mach.vessel_outer.length; i++) {
        var p = rzToXY(boxX, mach.vessel_outer[i][0], mach.vessel_outer[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();

      // First Wall / Divertor Limiter
      ctx.strokeStyle = "#00008b";
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 2]);
      ctx.beginPath();
      for (var i = 0; i < mach.first_wall.length; i++) {
        var p = rzToXY(boxX, mach.first_wall[i][0], mach.first_wall[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Left Plot: References & Machine Geometry
    var leftX = startX;
    ctx.fillStyle = isDark ? "#e6edf3" : "#111111";
    ctx.font = titleFont;
    ctx.textAlign = "center";
    ctx.fillText("CAD & Target", leftX + boxW * 0.5, startY - (compact ? 7 : 9));

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(leftX, startY, boxW, boxH);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1.1;
    ctx.strokeRect(leftX, startY, boxW, boxH);
    drawBoxTicks(leftX, true);

    ctx.save();
    ctx.beginPath();
    ctx.rect(leftX, startY, boxW, boxH);
    ctx.clip();
    drawCoils(leftX);
    drawMachineProfiles(leftX);

    // References target dots (for TCV) or nominal LCFS (for others)
    if (mach.id === "tcv") {
      ctx.strokeStyle = "#0022ff";
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      for (var i = 0; i < SHAPE_70166_REFS.length; i++) {
        var p = rzToXY(leftX, SHAPE_70166_REFS[i][0], SHAPE_70166_REFS[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();

      ctx.fillStyle = "#0022ff";
      for (var i = 0; i < SHAPE_70166_REFS.length; i++) {
        var p = rzToXY(leftX, SHAPE_70166_REFS[i][0], SHAPE_70166_REFS[i][1]);
        ctx.beginPath();
        ctx.arc(p[0], p[1], 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Achieved LCFS boundary
    var eqTarget = plasmaContour(1.0, st, 64, mach);
    ctx.save();
    ctx.strokeStyle = st.policy.color;
    ctx.lineWidth = 1.6;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    for (var i = 0; i < eqTarget.pts.length; i++) {
      var p = rzToXY(leftX, eqTarget.pts[i][0], eqTarget.pts[i][1]);
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
    ctx.restore();

    // Right Plot: Poloidal Flux Surfaces ψ(R, Z)
    var rightX = leftX + boxW + gap;
    ctx.fillStyle = isDark ? "#e6edf3" : "#111111";
    ctx.font = titleFont;
    ctx.textAlign = "center";
    ctx.fillText(compact ? "Flux ψ(R, Z)" : "Poloidal Flux ψ(R, Z)", rightX + boxW * 0.5, startY - (compact ? 7 : 9));

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(rightX, startY, boxW, boxH);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1.1;
    ctx.strokeRect(rightX, startY, boxW, boxH);
    drawBoxTicks(rightX, false);

    ctx.save();
    ctx.beginPath();
    ctx.rect(rightX, startY, boxW, boxH);
    ctx.clip();
    drawCoils(rightX);
    drawMachineProfiles(rightX);

    var fluxLevels = [
      { rho: 1.00, col: "#00008b", width: 1.4 },
      { rho: 0.82, col: "#2563eb", width: 1.1 },
      { rho: 0.65, col: "#0284c7", width: 1.0 },
      { rho: 0.48, col: "#0d9488", width: 0.95 },
      { rho: 0.32, col: "#d97706", width: 0.95 },
      { rho: 0.16, col: "#ea580c", width: 0.95 }
    ];
    for (var f = 0; f < fluxLevels.length; f++) {
      var fl = fluxLevels[f];
      var fData = plasmaContour(fl.rho, st, 60, mach);
      ctx.strokeStyle = fl.col;
      ctx.lineWidth = fl.width;
      ctx.beginPath();
      for (var i = 0; i < fData.pts.length; i++) {
        var p = rzToXY(rightX, fData.pts[i][0], fData.pts[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();
    }

    var pAxis = rzToXY(rightX, eqTarget.R0, eqTarget.Z0);
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(pAxis[0], pAxis[1], 3.0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    ctx.restore();
  }

  // --- Main Interactive Figure Mounting & Controls ---
  function mountTokamakCADViewer(root) {
    if (!root) return;
    if (root.getAttribute("data-tcv-mounted") === "1") return;
    root.setAttribute("data-tcv-mounted", "1");

    if (!document.getElementById("baux-tokamak-cad-styles")) {
      var stEl = document.createElement("style");
      stEl.id = "baux-tokamak-cad-styles";
      stEl.textContent = [
        ".tcv-cad-outer { position: relative; margin: 1.75rem 0 2.4rem 0; width: 100%; max-width: 100%; box-sizing: border-box; overflow: visible; }",
        ".tcv-cad-shell { width: 100%; max-width: 100%; box-sizing: border-box; overflow: hidden; border-top: 1.5px solid var(--hairline-strong, #111111); border-bottom: 1.5px solid var(--hairline-strong, #111111); border-left: 1px solid var(--hairline, #dddddd); border-right: 1px solid var(--hairline, #dddddd); background: var(--bg-canvas, #ffffff); font-family: var(--font-serif, 'Latin Modern Roman', Georgia, serif); color: var(--fg-primary, #111111); }",
        "html[data-theme='dark'] .tcv-cad-shell { background: #0d1117; border-color: #30363d; border-top-color: #e6edf3; border-bottom-color: #e6edf3; color: #e6edf3; }",
        ".tcv-side-controls { position: absolute; left: calc(100% + 22px); top: 0; width: 232px; box-sizing: border-box; border-top: 1.5px solid var(--hairline-strong, #111111); border-bottom: 1.5px solid var(--hairline-strong, #111111); border-left: 1px solid var(--hairline, #dddddd); border-right: 1px solid var(--hairline, #dddddd); background: var(--bg-canvas, #ffffff); font-family: var(--font-serif, 'Latin Modern Roman', Georgia, serif); color: var(--fg-primary, #111111); display: flex; flex-direction: column; }",
        "html[data-theme='dark'] .tcv-side-controls { background: #0d1117; border-color: #30363d; border-top-color: #e6edf3; border-bottom-color: #e6edf3; color: #e6edf3; }",
        ".tcv-ctrl-sec { padding: 8px 12px; border-bottom: 1px solid var(--hairline, #dddddd); display: flex; flex-direction: column; gap: 5px; }",
        "html[data-theme='dark'] .tcv-ctrl-sec { border-color: #30363d; }",
        ".tcv-ctrl-sec:last-child { border-bottom: none; }",
        ".tcv-ctrl-hdr { font-family: var(--font-sc, 'Latin Modern Roman Caps', serif); font-size: 11px; letter-spacing: 0.05em; color: var(--fg-muted, #555555); display: flex; justify-content: space-between; align-items: baseline; gap: 6px; }",
        "html[data-theme='dark'] .tcv-ctrl-hdr { color: #8b949e; }",
        ".tcv-select { width: 100%; font-family: var(--font-serif, Georgia, serif); font-size: 12px; padding: 4px 6px; border-radius: 2px; border: 1px solid var(--hairline-mid, #bbbbbb); background: var(--bg-canvas, #ffffff); color: inherit; cursor: pointer; box-sizing: border-box; }",
        ".tcv-select.mono { font-family: var(--font-mono, monospace); font-size: 11.5px; }",
        "html[data-theme='dark'] .tcv-select { background: #161b22; border-color: #30363d; color: #c9d1d9; }",
        ".tcv-pol-group { display: flex; flex-direction: column; gap: 5px; }",
        ".tcv-pol-btn { width: 100%; text-align: left; padding: 4.5px 8px; font-family: var(--font-serif, 'Latin Modern Roman', Georgia, serif); font-size: 12px; font-weight: 400; border-radius: 2px; border: 1px solid var(--hairline-mid, #bbbbbb); background: var(--bg-canvas, #ffffff); color: var(--fg-secondary, #222222); cursor: pointer; display: flex; justify-content: space-between; align-items: baseline; gap: 4px; transition: all 0.12s; box-sizing: border-box; }",
        ".tcv-pol-btn:hover { border-color: var(--hairline-strong, #111111); color: var(--fg-primary, #111111); background: var(--bg-surface, #fafafa); }",
        ".tcv-pol-btn.active { background: var(--hairline-strong, #111111); color: #ffffff; border-color: var(--hairline-strong, #111111); font-weight: 700; }",
        "html[data-theme='dark'] .tcv-pol-btn { background: #161b22; border-color: #30363d; color: #c9d1d9; }",
        "html[data-theme='dark'] .tcv-pol-btn.active { background: #f0f6fc; color: #0d1117; border-color: #f0f6fc; }",
        ".tcv-round-row { display: flex; flex-direction: column; gap: 5px; }",
        ".tcv-layers-row { display: flex; flex-direction: column; gap: 5px; }",
        ".tcv-side-controls input[type='range'] { accent-color: #111111; width: 100%; height: 22px; cursor: pointer; margin: 0; }",
        "html[data-theme='dark'] .tcv-side-controls input[type='range'] { accent-color: #f0f6fc; }",
        ".tcv-side-controls input[type='checkbox'] { accent-color: #111111; width: 15px; height: 15px; cursor: pointer; margin: 0; flex-shrink: 0; }",
        "html[data-theme='dark'] .tcv-side-controls input[type='checkbox'] { accent-color: #f0f6fc; }",
        ".tcv-chk-row { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--fg-secondary, #222222); cursor: pointer; user-select: none; }",
        "html[data-theme='dark'] .tcv-chk-row { color: #c9d1d9; }",
        ".tcv-grid { display: grid; grid-template-columns: 1.16fr 1fr; width: 100%; }",
        ".tcv-pane { position: relative; min-width: 0; overflow: hidden; border-right: 1px solid var(--hairline, #dddddd); min-height: 420px; display: flex; flex-direction: column; }",
        "html[data-theme='dark'] .tcv-pane { border-color: #30363d; }",
        ".tcv-pane:nth-child(2n) { border-right: none; }",
        ".tcv-subtable-cell { border-top: 1px solid var(--hairline, #dddddd); border-right: 1px solid var(--hairline, #dddddd); padding: 10px 14px; background: var(--bg-canvas, #ffffff); display: flex; flex-direction: column; justify-content: center; gap: 6px; min-width: 0; }",
        "html[data-theme='dark'] .tcv-subtable-cell { background: #0d1117; border-color: #30363d; }",
        ".tcv-subtable-cell:last-child { border-right: none; }",
        ".tcv-subtable-hdr { font-family: var(--font-sc, 'Latin Modern Roman Caps', serif); font-size: 11.5px; letter-spacing: 0.05em; color: var(--fg-muted, #555555); border-bottom: 1px solid var(--hairline-mid, #bbbbbb); padding-bottom: 4px; display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }",
        "html[data-theme='dark'] .tcv-subtable-hdr { color: #8b949e; border-color: #30363d; }",
        ".tcv-kv-list { display: flex; flex-direction: column; gap: 3.5px; }",
        ".tcv-kv-row { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; font-size: 12px; }",
        ".tcv-kv-lbl { color: var(--fg-secondary, #333333); min-width: 0; }",
        "html[data-theme='dark'] .tcv-kv-lbl { color: #8b949e; }",
        ".tcv-kv-val { font-family: var(--font-mono, 'Latin Modern Mono', monospace); font-size: 11.5px; font-weight: 500; color: var(--fg-primary, #111111); text-align: right; white-space: nowrap; }",
        "html[data-theme='dark'] .tcv-kv-val { color: #e6edf3; }",
        ".tcv-wpo-stack { display: flex; flex-direction: column; gap: 4px; }",
        ".tcv-wpo-item { display: grid; grid-template-columns: minmax(112px, 140px) 1fr 46px; align-items: center; gap: 8px; font-size: 12px; }",
        ".tcv-wpo-lbl { color: var(--fg-secondary, #333333); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        "html[data-theme='dark'] .tcv-wpo-lbl { color: #8b949e; }",
        ".tcv-wpo-track { position: relative; height: 5px; background: var(--hairline-light, #eeeeee); border-radius: 2px; overflow: hidden; }",
        "html[data-theme='dark'] .tcv-wpo-track { background: #21262d; }",
        ".tcv-wpo-fill { height: 100%; transition: width 0.15s ease-out; }",
        ".tcv-wpo-val { font-family: var(--font-mono, 'Latin Modern Mono', monospace); font-size: 11px; text-align: right; color: var(--fg-primary, #111111); }",
        "html[data-theme='dark'] .tcv-wpo-val { color: #e6edf3; }",
        ".tcv-canvas-wrap { position: relative; flex: 1 1 0; min-height: 380px; width: 100%; }",
        ".tcv-canvas-wrap canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; display: block; }",
        "#tcv-webgl-canvas { touch-action: none; -webkit-user-select: none; user-select: none; cursor: grab; }",
        "#tcv-webgl-canvas:active { cursor: grabbing; }",
        ".tcv-hud-badge { padding: 6px 12px; border-bottom: 1px solid var(--hairline, #dddddd); font-size: 12.5px; font-family: var(--font-serif, 'Latin Modern Roman', Georgia, serif); background: var(--bg-surface, #fafafa); color: var(--fg-primary, #111111); display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }",
        "html[data-theme='dark'] .tcv-hud-badge { background: #161b22; border-color: #30363d; color: #e6edf3; }",
        ".tcv-touch-hint { font-family: var(--font-sc, 'Latin Modern Roman Caps', serif); font-size: 10.5px; letter-spacing: 0.04em; color: var(--fg-muted, #666666); white-space: nowrap; }",
        "html[data-theme='dark'] .tcv-touch-hint { color: #8b949e; }",
        ".tcv-fig-caption { padding: 9px 14px; border-top: 1px solid var(--hairline-strong, #111111); background: var(--bg-surface, #fafafa); font-family: var(--font-serif, 'Latin Modern Roman', Georgia, serif); font-size: 12.5px; line-height: 1.48; color: var(--fg-secondary, #222222); }",
        "html[data-theme='dark'] .tcv-fig-caption { background: #161b22; border-color: #30363d; color: #c9d1d9; }",
        "@media (max-width: 1260px) {",
        "  .tcv-side-controls { position: static; width: 100%; margin-bottom: 10px; display: grid; grid-template-columns: 1fr 1fr; }",
        "  .tcv-ctrl-title { display: none; }",
        "  .tcv-ctrl-device { grid-column: 1; border-right: 1px solid var(--hairline, #dddddd); }",
        "  html[data-theme='dark'] .tcv-ctrl-device { border-right-color: #30363d; }",
        "  .tcv-ctrl-seed { grid-column: 2; }",
        "  .tcv-ctrl-policy { grid-column: 1 / -1; }",
        "  .tcv-pol-group { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }",
        "  .tcv-ctrl-round { grid-column: 1 / -1; }",
        "  .tcv-round-row { flex-direction: row; align-items: center; gap: 10px; }",
        "  .tcv-round-row input[type='range'] { flex: 1 1 auto; min-height: 34px; }",
        "  #tcv-play-btn { width: auto; flex: 0 0 auto; padding: 6px 12px; white-space: nowrap; }",
        "  .tcv-ctrl-layers { grid-column: 1 / -1; border-bottom: none; }",
        "  .tcv-layers-row { flex-direction: row; flex-wrap: wrap; align-items: center; gap: 8px 14px; }",
        "  .tcv-ctrl-layers .tcv-ctrl-hdr { display: none; }",
        "  #tcv-reset-cam { width: auto; margin-left: auto; margin-top: 0 !important; padding: 5px 10px; }",
        "}",
        "@media (max-width: 820px) {",
        "  .tcv-grid { grid-template-columns: 1fr; }",
        "  .tcv-pane { border-right: none; border-bottom: 1px solid var(--hairline, #dddddd); min-height: 0; }",
        "  html[data-theme='dark'] .tcv-pane { border-bottom-color: #30363d; }",
        "  .tcv-pane:nth-child(2) { border-bottom: none; }",
        "  .tcv-subtable-cell { border-right: none; }",
        "  #tcv-webgl-wrap { min-height: 290px; height: 290px; flex: none; }",
        "  #tcv-fig9-wrap { min-height: 300px; height: 300px; flex: none; }",
        "}",
        "@media (max-width: 640px) {",
        "  .tcv-cad-outer { margin: 1.15rem 0 1.75rem 0; }",
        "  .tcv-ctrl-sec { padding: 7px 10px; gap: 4px; }",
        "  .tcv-select, .tcv-select.mono { font-size: 16px; min-height: 38px; padding: 5px 6px; }",
        "  .tcv-pol-group { gap: 5px; }",
        "  .tcv-pol-btn { min-height: 38px; padding: 5px 6px; font-size: 11.5px; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 1px; }",
        "  #tcv-play-btn, #tcv-reset-cam { flex-direction: row; font-size: 12px; min-height: 36px; }",
        "  .tcv-chk-row { min-height: 34px; font-size: 12.5px; }",
        "  #tcv-webgl-wrap { min-height: 245px; height: 245px; }",
        "  #tcv-fig9-wrap { min-height: 255px; height: 255px; }",
        "  .tcv-hud-badge { padding: 5px 10px; font-size: 11.5px; }",
        "  .tcv-subtable-cell { padding: 9px 10px; }",
        "  .tcv-kv-row { font-size: 11.5px; }",
        "  .tcv-kv-val { font-size: 11px; }",
        "  .tcv-wpo-item { grid-template-columns: minmax(104px, 126px) 1fr 42px; gap: 6px; font-size: 11.5px; }",
        "  .tcv-fig-caption { padding: 8px 10px; font-size: 11.5px; }",
        "}",
        "@media (max-width: 345px) {",
        "  .tcv-side-controls { grid-template-columns: 1fr; }",
        "  .tcv-ctrl-device { grid-column: 1; border-right: none; }",
        "  .tcv-ctrl-seed { grid-column: 1; }",
        "  #tcv-webgl-wrap { min-height: 225px; height: 225px; }",
        "  #tcv-fig9-wrap { min-height: 235px; height: 235px; }",
        "}"
      ].join("\n");
      document.head.appendChild(stEl);
    }

    root.innerHTML = [
      '<div class="tcv-cad-outer">',
      '  <aside class="tcv-side-controls" aria-label="Controls">',
      '    <div class="tcv-hud-badge tcv-ctrl-title" style="padding:5px 12px;font-size:12px;"><strong>Controls</strong></div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-device">',
      '      <div class="tcv-ctrl-hdr"><span>Device</span><span>FUSE.jl</span></div>',
      '      <select id="tcv-machine-sel" class="tcv-select">',
      '        <option value="tcv" selected>TCV · EPFL</option>',
      '        <option value="arc">ARC · MIT/CFS</option>',
      '        <option value="iter">ITER · Cadarache</option>',
      '        <option value="d3d">DIII-D · GA</option>',
      '      </select>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-seed">',
      '      <div class="tcv-ctrl-hdr"><span>Seed</span><span><em>n</em> = 10</span></div>',
      '      <select id="tcv-seed-sel" class="tcv-select mono">',
      '        <option value="-1">Mean ± SEM</option>',
      '        <option value="0">Seed 1 (91.1)</option>',
      '        <option value="1">Seed 2 (91.8)</option>',
      '        <option value="2">Seed 3 (87.8)</option>',
      '        <option value="3">Seed 4 (90.2)</option>',
      '        <option value="4">Seed 5 (92.0)</option>',
      '        <option value="5">Seed 6 (92.8)</option>',
      '        <option value="6">Seed 7 (90.5)</option>',
      '        <option value="7">Seed 8 (92.9)</option>',
      '        <option value="8">Seed 9 (93.4)</option>',
      '        <option value="9">Seed 10 (93.3)</option>',
      '      </select>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-policy">',
      '      <div class="tcv-ctrl-hdr"><span>Policy</span><span>Return <em>J</em></span></div>',
      '      <div class="tcv-pol-group">',
      '        <button type="button" class="tcv-pol-btn active" data-pol="shared"><span>Shared TuRBO</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">91.6</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="indep"><span>Indep. TuRBO</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">89.4</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="init"><span>Initial θ₀</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">6.0</span></button>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-round">',
      '      <div class="tcv-ctrl-hdr"><span>Rollout Round</span><strong id="tcv-round-lbl" style="font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-primary,#111);">10000</strong></div>',
      '      <div class="tcv-round-row">',
      '        <input type="range" id="tcv-round-slider" min="0" max="10000" step="50" value="10000" aria-label="Rollout Round" />',
      '        <button type="button" id="tcv-play-btn" class="tcv-pol-btn" style="justify-content:center;font-weight:700;">▶ Play</button>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-layers">',
      '      <div class="tcv-ctrl-hdr"><span>Layers</span></div>',
      '      <div class="tcv-layers-row">',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-coils" checked /> Coils</label>',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-field" /> B-Field</label>',
      '        <button type="button" id="tcv-reset-cam" class="tcv-pol-btn" style="justify-content:center;margin-top:3px;">Reset View</button>',
      '      </div>',
      '    </div>',
      '  </aside>',
      '  <div class="tcv-cad-shell">',
      '    <div class="tcv-grid">',
      '      <div class="tcv-pane">',
      '        <div class="tcv-hud-badge"><span id="tcv-badge-a"><strong>(a)</strong> Cutaway</span><span class="tcv-touch-hint">Drag · Pinch</span></div>',
      '        <div class="tcv-canvas-wrap" id="tcv-webgl-wrap">',
      '          <canvas id="tcv-webgl-canvas"></canvas>',
      '        </div>',
      '      </div>',
      '      <div class="tcv-pane">',
      '        <div class="tcv-hud-badge"><span id="tcv-badge-b"><strong>(b)</strong> Miller Contour</span></div>',
      '        <div class="tcv-canvas-wrap" id="tcv-fig9-wrap">',
      '          <canvas id="tcv-fig9-canvas"></canvas>',
      '        </div>',
      '      </div>',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span id="tcv-subtable-title">FUSE.jl Geometry</span><span id="tcv-subtable-device">TCV (EPFL)</span></div>',
      '        <div class="tcv-kv-list" id="tcv-kv-list"></div>',
      '      </div>',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span>TORAX SmoothMax (α = −3, TuRBO-0)</span><span>Score</span></div>',
      '        <div class="tcv-wpo-stack" id="tcv-wpo-strip"></div>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-fig-caption" id="tcv-fig-caption">',
      '      <strong>Figure 1.</strong> Cross-device tokamak geometry and analytical Miller flux contours interpolated across the 10-seed, 10,000-round TORAX 2.0.0 regional-sharing benchmark (<em>D</em> = 160,531, surrogate-free TuRBO-zero). <em>(a)</em> Vacuum vessel, divertor, central solenoid, and poloidal field coils from FUSE.jl IMAS descriptions (Meneghini et al., 2024: TCV, ARC, ITER, DIII-D). <em>(b)</em> Analytical Miller flux surfaces ψ(<em>R</em>, <em>Z</em>) and coil cross-sections; recorded TORAX SmoothMax scores (adapted from Pfau et al., 2025) appear at right.',
      '    </div>',
      '  </div>',
      '</div>'
    ].join("\n");

    var glCanvas = root.querySelector("#tcv-webgl-canvas");
    var fig9Canvas = root.querySelector("#tcv-fig9-canvas");
    var kvList = root.querySelector("#tcv-kv-list");
    var wpoStrip = root.querySelector("#tcv-wpo-strip");
    var roundSlider = root.querySelector("#tcv-round-slider");
    var roundLbl = root.querySelector("#tcv-round-lbl");
    var seedSel = root.querySelector("#tcv-seed-sel");
    var machineSel = root.querySelector("#tcv-machine-sel");
    var playBtn = root.querySelector("#tcv-play-btn");
    var chkCoils = root.querySelector("#tcv-chk-coils");
    var chkField = root.querySelector("#tcv-chk-field");
    var resetCamBtn = root.querySelector("#tcv-reset-cam");
    var badgeA = root.querySelector("#tcv-badge-a");
    var badgeB = root.querySelector("#tcv-badge-b");
    var subtableDevice = root.querySelector("#tcv-subtable-device");

    var initialMach = "tcv";
    try {
      var qParams = new URLSearchParams(window.location.search);
      var qm = qParams.get("machine");
      if (qm && Object.prototype.hasOwnProperty.call(MACHINES, qm)) initialMach = qm;
    } catch (e) {}
    if (machineSel) machineSel.value = initialMach;

    var uiState = {
      machine: initialMach,
      policyKey: "shared",
      round: 10000,
      seedIdx: -1,
      showCoils: true,
      showFieldLines: false,
      cutawayDeg: 56.0,
      yaw: -Math.PI * 0.5,
      pitch: 0.46,
      dist: (Object.prototype.hasOwnProperty.call(MACHINES, initialMach) ? MACHINES[initialMach] : MACHINES.tcv).dist,
      playing: false
    };
    uiState.dist = (Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv).dist;

    var gl = glCanvas.getContext("webgl", { antialias: true, alpha: false, preserveDrawingBuffer: true });
    var prog = null;
    var cachedBuffers = null;
    var meshDirty = true;

    if (gl) {
      function compileShader(type, src) {
        var s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        return s;
      }
      prog = gl.createProgram();
      gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, VS_CAD));
      gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS_CAD));
      gl.linkProgram(prog);
      gl.useProgram(prog);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
    }

    function renderAll() {
      var mach = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      var st = evalState(uiState.policyKey, uiState.round, uiState.seedIdx);
      var jStr = st.score.toFixed(2) + (uiState.seedIdx < 0 ? (" ± " + st.sem.toFixed(2)) : (" (S" + (uiState.seedIdx + 1) + ")"));

      if (badgeA) badgeA.innerHTML = "<strong>(a)</strong> Cutaway — " + escHtml(mach.shortName);
      if (badgeB) badgeB.innerHTML = "<strong>(b)</strong> Miller Contour — " + escHtml(mach.shortName);
      if (subtableDevice) subtableDevice.textContent = mach.shortName;

      if (kvList) {
        var ipVal = mach.ipNominal.toFixed(2);
        var b0Val = mach.B0.toFixed(2);
        var aspectVal = (mach.R0 / Math.max(0.01, mach.a)).toFixed(2);
        var contourLbl = (mach.id === "tcv") ? "Miller Contour Δ" : "Scaled Template Δ";
        var contourVal = (mach.id === "tcv")
          ? (st.lcfsErrMm.toFixed(2) + " mm")
          : ((st.lcfsErrMm / 2.50).toFixed(2) + "% a");
        kvList.innerHTML = [
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">TORAX Return <em>J</em>(θ)</span><span class="tcv-kv-val" style="color:' + escHtml(st.policy.color) + ';">' + escHtml(jStr) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Nominal <em>I<sub>p</sub></em> (FUSE.jl)</span><span class="tcv-kv-val">' + escHtml(ipVal) + ' MA</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Toroidal Field <em>B</em><sub>0</sub></span><span class="tcv-kv-val">' + escHtml(b0Val) + ' T</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Aspect Ratio <em>R</em><sub>0</sub>/<em>a</em></span><span class="tcv-kv-val">' + escHtml(aspectVal) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Major / Minor Radius</span><span class="tcv-kv-val"><em>R</em>₀=' + escHtml(mach.R0.toFixed(2)) + ', <em>a</em>=' + escHtml(mach.a.toFixed(2)) + ' m</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">' + escHtml(contourLbl) + '</span><span class="tcv-kv-val">' + escHtml(contourVal) + '</span></div>'
        ].join("");
      }
      if (wpoStrip) {
        var cLabels = [
          "Miller Shape",
          "Plasma Current <em>I<sub>p</sub></em>",
          "Shaping Margin",
          "Actuators ∈ [−1, 1]",
          "Aux. Balance"
        ];
        var stripHtml = [];
        for (var ci = 0; ci < 5; ci++) {
          var pct = (st.comps[ci] * 100.0).toFixed(1) + "%";
          stripHtml.push(
            '<div class="tcv-wpo-item">' +
              '<span class="tcv-wpo-lbl">' + cLabels[ci] + '</span>' +
              '<div class="tcv-wpo-track"><div class="tcv-wpo-fill" style="width:' + escHtml(pct) + ';background:' + escHtml(st.policy.color) + ';"></div></div>' +
              '<span class="tcv-wpo-val">' + escHtml(pct) + '</span>' +
            '</div>'
          );
        }
        wpoStrip.innerHTML = stripHtml.join("");
      }

      // 1. Render 3D Tokamak Cutaway in WebGL
      if (gl && prog) {
        var dpr = window.devicePixelRatio || 1;
        var rect = glCanvas.getBoundingClientRect();
        var w = Math.max(1, Math.round(rect.width)) || 320;
        var h = Math.max(1, Math.round(rect.height)) || 260;
        if (glCanvas.width !== Math.round(w * dpr) || glCanvas.height !== Math.round(h * dpr)) {
          glCanvas.width = Math.round(w * dpr);
          glCanvas.height = Math.round(h * dpr);
        }
        gl.viewport(0, 0, glCanvas.width, glCanvas.height);
        var isDark = document.documentElement.getAttribute("data-theme") === "dark";
        if (isDark) gl.clearColor(0.06, 0.08, 0.11, 1.0);
        else gl.clearColor(0.985, 0.988, 0.992, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        if (meshDirty || !cachedBuffers) {
          if (cachedBuffers) {
            if (cachedBuffers.posBuf) gl.deleteBuffer(cachedBuffers.posBuf);
            if (cachedBuffers.normBuf) gl.deleteBuffer(cachedBuffers.normBuf);
            if (cachedBuffers.colBuf) gl.deleteBuffer(cachedBuffers.colBuf);
            if (cachedBuffers.linePosBuf) gl.deleteBuffer(cachedBuffers.linePosBuf);
            if (cachedBuffers.lineColBuf) gl.deleteBuffer(cachedBuffers.lineColBuf);
          }
          cachedBuffers = buildTokamakCADBuffers(gl, st, uiState);
          meshDirty = false;
        }

        var aspect = w / Math.max(1, h);
        var fitScale = (aspect < 1.08) ? (1.08 / Math.max(0.45, aspect)) : 1.0;
        var effDist = uiState.dist * fitScale;

        var eye = [
          effDist * Math.cos(uiState.pitch) * Math.cos(uiState.yaw),
          effDist * Math.sin(uiState.pitch),
          effDist * Math.cos(uiState.pitch) * Math.sin(uiState.yaw)
        ];
        var farPlane = Math.max(60.0, effDist * 3.5);
        var projMat = mat4Perspective(32.0 * Math.PI / 180.0, aspect, 0.2, farPlane);
        var viewMat = mat4LookAt(eye, [0.0, -0.04, 0.0], [0.0, 1.0, 0.0]);
        var mvp = mat4Mul(projMat, viewMat);
        var model = mat4Identity();

        gl.useProgram(prog);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uModel"), false, model);
        gl.uniform3fv(gl.getUniformLocation(prog, "uCamPos"), new Float32Array(eye));

        var aPos = gl.getAttribLocation(prog, "aPos");
        var aNorm = gl.getAttribLocation(prog, "aNorm");
        var aCol = gl.getAttribLocation(prog, "aCol");

        // Solid CAD triangles
        gl.uniform1f(gl.getUniformLocation(prog, "uUnlit"), 0.0);
        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.posBuf);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.normBuf);
        gl.enableVertexAttribArray(aNorm);
        gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.colBuf);
        gl.enableVertexAttribArray(aCol);
        gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.TRIANGLES, 0, cachedBuffers.triCount);

        // Crisp CAD feature-edge lines & helical field lines
        if (cachedBuffers.lineCount > 0) {
          gl.uniform1f(gl.getUniformLocation(prog, "uUnlit"), 1.0);
          gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.linePosBuf);
          gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
          gl.disableVertexAttribArray(aNorm);
          gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.lineColBuf);
          gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.LINES, 0, cachedBuffers.lineCount);
        }
      }

      // 2. Render Poloidal Equilibrium Cross Section & Flux Surfaces
      renderFigure9Canvas(fig9Canvas, st, mach);
    }

    // Bind Machine Dropdown
    machineSel.addEventListener("change", function() {
      uiState.machine = Object.prototype.hasOwnProperty.call(MACHINES, machineSel.value) ? machineSel.value : "tcv";
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      uiState.dist = m.dist;
      meshDirty = true;
      renderAll();
    });

    // Bind Policy Buttons
    var polBtns = root.querySelectorAll(".tcv-pol-btn[data-pol]");
    for (var i = 0; i < polBtns.length; i++) {
      (function(btn) {
        btn.addEventListener("click", function() {
          for (var j = 0; j < polBtns.length; j++) polBtns[j].classList.remove("active");
          btn.classList.add("active");
          var pk = btn.getAttribute("data-pol");
          uiState.policyKey = (pk && Object.prototype.hasOwnProperty.call(BENCH, pk)) ? pk : "shared";
          meshDirty = true;
          renderAll();
        });
      })(polBtns[i]);
    }

    roundSlider.addEventListener("input", function() {
      var r = parseInt(roundSlider.value, 10);
      uiState.round = Number.isFinite(r) ? Math.max(0, Math.min(10000, r)) : 0;
      roundLbl.textContent = String(uiState.round);
      meshDirty = true;
      renderAll();
    });

    seedSel.addEventListener("change", function() {
      var si = parseInt(seedSel.value, 10);
      uiState.seedIdx = (Number.isInteger(si) && si >= -1 && si < 10) ? si : -1;
      meshDirty = true;
      renderAll();
    });

    chkCoils.addEventListener("change", function() {
      uiState.showCoils = chkCoils.checked;
      meshDirty = true;
      renderAll();
    });

    chkField.addEventListener("change", function() {
      uiState.showFieldLines = chkField.checked;
      meshDirty = true;
      renderAll();
    });

    resetCamBtn.addEventListener("click", function() {
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      uiState.yaw = -Math.PI * 0.5;
      uiState.pitch = 0.46;
      uiState.dist = m.dist;
      renderAll();
    });

    // Play / Pause Rollout Animation
    var animTimer = null;
    playBtn.addEventListener("click", function() {
      uiState.playing = !uiState.playing;
      if (uiState.playing) {
        playBtn.textContent = "⏸ Pause";
        if (uiState.round >= 10000) uiState.round = 0;
        animTimer = setInterval(function() {
          uiState.round = Math.min(10000, uiState.round + 100);
          roundSlider.value = String(uiState.round);
          roundLbl.textContent = String(uiState.round);
          meshDirty = true;
          renderAll();
          if (uiState.round >= 10000) {
            uiState.playing = false;
            playBtn.textContent = "▶ Play";
            clearInterval(animTimer);
          }
        }, 65);
      } else {
        playBtn.textContent = "▶ Play";
        if (animTimer) clearInterval(animTimer);
      }
    });

    // 3D Orbit & Zoom Mouse + Touch Handlers on WebGL Canvas
    var dragging = false, lastX = 0, lastY = 0;
    glCanvas.addEventListener("mousedown", function(e) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener("mousemove", function(e) {
      if (!dragging) return;
      var dx = e.clientX - lastX;
      var dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      uiState.yaw += dx * 0.0085;
      uiState.pitch = Math.max(-1.1, Math.min(1.25, uiState.pitch + dy * 0.0085));
      renderAll();
    });
    window.addEventListener("mouseup", function() { dragging = false; });
    glCanvas.addEventListener("wheel", function(e) {
      e.preventDefault();
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      var zoomScale = m.dist / 5.85;
      var minDist = m.dist * 0.35;
      var maxDist = m.dist * 2.8;
      uiState.dist = Math.max(minDist, Math.min(maxDist, uiState.dist + e.deltaY * 0.0025 * zoomScale));
      renderAll();
    }, { passive: false });

    // Single-finger touch orbit, two-finger pinch zoom, and double-tap camera reset
    var touchMode = 0, tStartX = 0, tStartY = 0, tLastX = 0, tLastY = 0;
    var pinchSpan0 = 1.0, pinchDist0 = 5.85, lastTapMs = 0;
    glCanvas.addEventListener("touchstart", function(e) {
      if (e.touches.length === 1) {
        touchMode = 1;
        tStartX = tLastX = e.touches[0].clientX;
        tStartY = tLastY = e.touches[0].clientY;
      } else if (e.touches.length >= 2) {
        touchMode = 2;
        var dx = e.touches[1].clientX - e.touches[0].clientX;
        var dy = e.touches[1].clientY - e.touches[0].clientY;
        pinchSpan0 = Math.max(12.0, Math.hypot(dx, dy));
        pinchDist0 = uiState.dist;
      }
    }, { passive: true });

    glCanvas.addEventListener("touchmove", function(e) {
      if (touchMode === 1 && e.touches.length === 1) {
        e.preventDefault();
        var cx = e.touches[0].clientX;
        var cy = e.touches[0].clientY;
        var dx = cx - tLastX;
        var dy = cy - tLastY;
        tLastX = cx;
        tLastY = cy;
        uiState.yaw += dx * 0.010;
        uiState.pitch = Math.max(-1.1, Math.min(1.25, uiState.pitch + dy * 0.010));
        renderAll();
      } else if (touchMode === 2 && e.touches.length >= 2) {
        e.preventDefault();
        var pdx = e.touches[1].clientX - e.touches[0].clientX;
        var pdy = e.touches[1].clientY - e.touches[0].clientY;
        var span = Math.max(12.0, Math.hypot(pdx, pdy));
        var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
        uiState.dist = Math.max(m.dist * 0.35, Math.min(m.dist * 2.8, pinchDist0 * (pinchSpan0 / span)));
        renderAll();
      }
    }, { passive: false });

    glCanvas.addEventListener("touchend", function(e) {
      if (touchMode === 1 && e.touches.length === 0) {
        var moved = Math.hypot(tLastX - tStartX, tLastY - tStartY);
        var now = Date.now();
        if (moved < 10) {
          if (now - lastTapMs < 300) {
            var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
            uiState.yaw = -Math.PI * 0.5;
            uiState.pitch = 0.46;
            uiState.dist = m.dist;
            renderAll();
            lastTapMs = 0;
          } else {
            lastTapMs = now;
          }
        }
      }
      if (e.touches.length === 0) touchMode = 0;
      else if (e.touches.length === 1) {
        touchMode = 1;
        tLastX = e.touches[0].clientX;
        tLastY = e.touches[0].clientY;
      }
    }, { passive: true });
    glCanvas.addEventListener("touchcancel", function() { touchMode = 0; }, { passive: true });

    var cadShell = root.querySelector(".tcv-cad-shell");
    if (typeof ResizeObserver !== "undefined" && cadShell) {
      var ro = new ResizeObserver(function() { renderAll(); });
      ro.observe(cadShell);
    }
    window.addEventListener("resize", function() { renderAll(); });
    setTimeout(renderAll, 30);
    renderAll();
  }

  var autoEl = document.getElementById("tokamak-cad-simulation");
  if (autoEl) mountTokamakCADViewer(autoEl);
})
|} in
  let _ = Js.Unsafe.fun_call boot_fn [||] in
  ()

let init_tokamak_cad () =
  match Js.Opt.to_option (document##getElementById (js "tokamak-cad-simulation")) with
  | None -> ()
  | Some _ -> mount_tokamak_cad ()

(* --- Main Entrypoint --- *)
let () =
  let booted = ref false in
  let boot () =
    if not !booted then begin
      booted := true;
      init_theme_toggle ();
      init_copy_buttons ();
      render_api_explorer ();
      init_tokamak_cad ()
    end
  in
  let on_load = handler (fun _ev ->
    boot ();
    Js._true
  ) in
  window##.onload := on_load;
  let state = Js.to_string (Js.Unsafe.get document (js "readyState")) in
  if state <> "loading" then boot ()
