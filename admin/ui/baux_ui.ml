(* admin/ui/baux_ui.ml
   Compiled via Dune & Js_of_ocaml (opam switch 5.2.0+ox)
   BAUX Static API Reference Filter & Tokamak 3D WebGL CAD Explorer
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

let query_within (parent : element Js.t) sel =
  let nl = parent##querySelectorAll (js sel) in
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
      let parent = Js.Unsafe.get btn (js "parentElement") in
      let text =
        match Js.Opt.to_option parent with
        | Some p ->
            let p_elem : htmlElement Js.t = Js.Unsafe.coerce p in
            (match Js.Opt.to_option (p_elem##querySelector (js "code")) with
            | Some c ->
                let c_elem : htmlElement Js.t = Js.Unsafe.coerce c in
                Js.to_string c_elem##.innerText
            | None ->
                (match Js.Opt.to_option (p_elem##querySelector (js "pre")) with
                | Some pr ->
                    let pr_elem : htmlElement Js.t = Js.Unsafe.coerce pr in
                    Js.to_string pr_elem##.innerText
                | None ->
                    (match Js.Opt.to_option (p_elem##querySelector (js ".install-cmd-text")) with
                    | Some c ->
                        let c_elem : htmlElement Js.t = Js.Unsafe.coerce c in
                        Js.to_string c_elem##.innerText
                    | None -> Js.to_string p_elem##.innerText)))
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

(* --- Static Semantic API Reference Filter (Zero innerHTML) --- *)
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

let split_tokens s =
  let s = String.lowercase_ascii s in
  let len = String.length s in
  let acc = ref [] in
  let buf = Buffer.create 16 in
  for i = 0 to len - 1 do
    match s.[i] with
    | ' ' | '\t' | '\n' | '\r' ->
        if Buffer.length buf > 0 then begin
          acc := Buffer.contents buf :: !acc;
          Buffer.clear buf
        end
    | c -> Buffer.add_char buf c
  done;
  if Buffer.length buf > 0 then
    acc := Buffer.contents buf :: !acc;
  List.rev !acc

let get_attr (el : element Js.t) name =
  match Js.Opt.to_option (el##getAttribute (js name)) with
  | Some s -> Js.to_string s
  | None -> ""

let text_of_element (el : element Js.t) =
  let raw =
    match Js.Opt.to_option el##.textContent with
    | Some s -> Js.to_string s
    | None ->
        let h : htmlElement Js.t = Js.Unsafe.coerce el in
        Js.to_string h##.innerText
  in
  String.lowercase_ascii raw

let set_hidden (el : element Js.t) hidden =
  if hidden then el##.classList##add (js "api-hidden")
  else el##.classList##remove (js "api-hidden")

type api_entry_rec = {
  ent_el : element Js.t;
  ent_id : string;
  ent_symbol : string;
  ent_kind : string;
  ent_haystack : string;
  ent_has_example : bool;
  mutable ent_visible : bool;
}

type summary_row_rec = {
  row_el : element Js.t;
  row_symbol : string;
  row_haystack : string;
}

type summary_table_rec = {
  tbl_el : element Js.t;
  tbl_rows : summary_row_rec list;
}

type module_sec_rec = {
  sec_el : element Js.t;
  sec_entries : element Js.t list;
}

let init_api_reference () =
  match Js.Opt.to_option (document##getElementById (js "baux-api-filter")) with
  | None -> ()
  | Some root ->
      while Js.Opt.test root##.firstChild do
        match Js.Opt.to_option root##.firstChild with
        | Some child -> ignore (root##removeChild child)
        | None -> ()
      done;

      (* Collect summary tables & rows first so summary descriptions enrich entry search *)
      let row_text_by_sym : (string, string) Hashtbl.t = Hashtbl.create 96 in
      let raw_tables = query_all ".api-summary-table" in
      let tables : summary_table_rec list =
        List.map (fun tbl_el ->
          let trs = query_within tbl_el "tbody tr[data-symbol]" in
          let rows =
            List.map (fun row_el ->
              let sym = String.trim (get_attr row_el "data-symbol") in
              let row_txt = text_of_element row_el in
              let hay = (String.lowercase_ascii sym) ^ " " ^ row_txt in
              if sym <> "" then Hashtbl.replace row_text_by_sym sym row_txt;
              { row_el; row_symbol = sym; row_haystack = hay }
            ) trs
          in
          { tbl_el; tbl_rows = rows }
        ) raw_tables
      in

      (* Collect all .api-entry specification blocks once at boot *)
      let raw_entries = query_all ".api-entry" in
      let entries : api_entry_rec list =
        List.map (fun ent_el ->
          let id_attr = String.trim (get_attr ent_el "id") in
          let sym_attr = String.trim (get_attr ent_el "data-symbol") in
          let sym = if sym_attr <> "" then sym_attr else id_attr in
          let kind = String.lowercase_ascii (String.trim (get_attr ent_el "data-kind")) in
          let extra =
            match Hashtbl.find_opt row_text_by_sym sym with
            | Some t -> " " ^ t
            | None -> ""
          in
          let hay = (String.lowercase_ascii sym) ^ " " ^ (text_of_element ent_el) ^ extra in
          let pres = query_within ent_el "pre" in
          let has_example =
            List.exists (fun pr -> string_contains (text_of_element pr) "import baux") pres
          in
          List.iter (fun pr ->
            if string_contains (text_of_element pr) "import baux" then begin
              let is_wrapped =
                match Js.Opt.to_option pr##.parentNode with
                | Some pn ->
                    let p_el : element Js.t = Js.Unsafe.coerce pn in
                    (try Js.to_bool (p_el##.classList##contains (js "api-example-wrap"))
                     with _ -> false)
                | None -> false
              in
              if not is_wrapped then begin
                match Js.Opt.to_option pr##.parentNode with
                | Some parent_node ->
                    let wrap = createDiv document in
                    wrap##.className := js "api-example-wrap";
                    ignore (parent_node##insertBefore (wrap :> Dom.node Js.t) (Js.some (pr :> Dom.node Js.t)));
                    Dom.appendChild wrap pr;
                    let copy_btn = createButton ~_type:(js "button") document in
                    copy_btn##.className := js "copy-btn";
                    copy_btn##.textContent := Js.some (js "Copy");
                    add_click (copy_btn :> element Js.t) (fun () ->
                      let text =
                        match Js.Opt.to_option pr##.textContent with
                        | Some s -> Js.to_string s
                        | None ->
                            let h : htmlElement Js.t = Js.Unsafe.coerce pr in
                            Js.to_string h##.innerText
                      in
                      try
                        let clip = Js.Unsafe.global##.navigator##.clipboard in
                        let _ = clip##writeText (js text) in
                        copy_btn##.textContent := Js.some (js "Copied \u{2713}");
                        copy_btn##.classList##add (js "copied");
                        set_timeout (fun () ->
                          copy_btn##.textContent := Js.some (js "Copy");
                          copy_btn##.classList##remove (js "copied")
                        ) 2000.0
                      with _ -> ()
                    );
                    Dom.appendChild wrap copy_btn
                | None -> ()
              end
            end
          ) pres;
          {
            ent_el;
            ent_id = id_attr;
            ent_symbol = sym;
            ent_kind = kind;
            ent_haystack = hay;
            ent_has_example = has_example;
            ent_visible = true;
          }
        ) raw_entries
      in
      let total_count = List.length entries in

      (* Collect module section.block containers (excluding any ancestor containing #baux-api-filter) *)
      let is_ancestor_of_root (cand : element Js.t) =
        let cand_node : Dom.node Js.t = (cand :> Dom.node Js.t) in
        let rec climb cur_opt =
          match Js.Opt.to_option cur_opt with
          | None -> false
          | Some n ->
              if n == cand_node then true
              else climb n##.parentNode
        in
        climb (Js.some (root :> Dom.node Js.t))
      in
      let module_sections : module_sec_rec list =
        query_all "section.block"
        |> List.filter_map (fun sec_el ->
             if is_ancestor_of_root sec_el then None
             else
               let sec_entries = query_within sec_el ".api-entry" in
               if sec_entries = [] then None
               else Some { sec_el; sec_entries }
           )
      in

      (* Build filter bar strictly with typed DOM constructors — zero innerHTML *)
      let bar = createDiv document in
      bar##.className := js "api-filter-bar";

      let inp = createInput ~_type:(js "search") document in
      inp##.id := js "api-filter-input";
      inp##.className := js "api-filter-input";
      inp##.placeholder := js "Filter symbols or parameters\u{2026}  (press /)";
      inp##setAttribute (js "autocomplete") (js "off");
      inp##setAttribute (js "spellcheck") (js "false");
      inp##setAttribute (js "aria-label") (js "Filter API symbols");

      let pills_wrap = createDiv document in
      pills_wrap##.className := js "api-filter-pills";
      pills_wrap##setAttribute (js "role") (js "group");
      pills_wrap##setAttribute (js "aria-label") (js "Filter by symbol kind");

      let make_pill label kind_val is_active =
        let btn = createButton ~_type:(js "button") document in
        btn##.className := js (if is_active then "api-filter-pill active" else "api-filter-pill");
        btn##setAttribute (js "data-kind-filter") (js kind_val);
        btn##setAttribute (js "aria-pressed") (js (if is_active then "true" else "false"));
        btn##.textContent := Js.some (js label);
        Dom.appendChild pills_wrap btn;
        (kind_val, btn)
      in
      let pill_all = make_pill "All" "all" true in
      let pill_fn = make_pill "Functions" "function" false in
      let pill_cls = make_pill "Classes" "class" false in
      let pill_ex = make_pill "Examples" "examples" false in
      let pill_buttons = [pill_all; pill_fn; pill_cls; pill_ex] in

      let meta_wrap = createDiv document in
      meta_wrap##.className := js "api-filter-meta";

      let clear_btn = createButton ~_type:(js "button") document in
      clear_btn##.className := js "api-filter-clear api-hidden";
      clear_btn##.textContent := Js.some (js "Clear");

      let count_el = createSpan document in
      count_el##.className := js "api-filter-count";
      count_el##setAttribute (js "aria-live") (js "polite");

      Dom.appendChild meta_wrap clear_btn;
      Dom.appendChild meta_wrap count_el;

      Dom.appendChild bar inp;
      Dom.appendChild bar pills_wrap;
      Dom.appendChild bar meta_wrap;

      let active_kind = ref "all" in

      (* Build compact .api-section-jumps strip below .api-filter-bar *)
      let jumps_wrap = createDiv document in
      jumps_wrap##.className := js "api-section-jumps";
      let top_sections = query_all "article > section.block > .tree-node > section.block" in
      let jump_count = ref 0 in
      let reset_filter_ref = ref (fun () -> ()) in
      List.iter (fun sec_el ->
        let sec_entries = query_within sec_el ".api-entry" in
        if sec_entries <> [] then begin
          let h1_nodes = query_within sec_el "header h1" in
          let title =
            match h1_nodes with
            | h1 :: _ ->
                (match Js.Opt.to_option h1##.textContent with
                | Some s ->
                    let raw = Js.to_string s in
                    let b = Buffer.create (String.length raw) in
                    let len = String.length raw in
                    let rec loop i =
                      if i >= len then Buffer.contents b
                      else if Char.code raw.[i] = 0xC2 && i + 1 < len && Char.code raw.[i+1] = 0xA0 then begin
                        Buffer.add_char b ' ';
                        loop (i + 2)
                      end else begin
                        Buffer.add_char b raw.[i];
                        loop (i + 1)
                      end
                    in
                    String.trim (loop 0)
                | None -> "")
            | [] -> ""
          in
          if title <> "" then begin
            let chip = createButton ~_type:(js "button") document in
            chip##.className := js "api-jump-chip";
            chip##.textContent := Js.some (js title);
            add_click (chip :> element Js.t) (fun () ->
              let raw_q = String.trim (Js.to_string inp##.value) in
              if raw_q <> "" || !active_kind <> "all" then !reset_filter_ref ();
              try
                let opts = Js.Unsafe.obj [|
                  ("behavior", Js.Unsafe.inject (js "smooth"));
                  ("block", Js.Unsafe.inject (js "start"));
                |] in
                ignore ((Js.Unsafe.coerce sec_el)##scrollIntoView opts)
              with _ ->
                sec_el##scrollIntoView Js._true
            );
            Dom.appendChild jumps_wrap chip;
            incr jump_count
          end
        end
      ) top_sections;

      let empty_el = createDiv document in
      empty_el##.className := js "api-empty-state api-hidden";

      Dom.appendChild root bar;
      if !jump_count > 0 then Dom.appendChild root jumps_wrap;
      Dom.appendChild root empty_el;

      let sync_pills () =
        List.iter (fun (k, (btn : buttonElement Js.t)) ->
          let act = (k = !active_kind) in
          if act then btn##.classList##add (js "active")
          else btn##.classList##remove (js "active");
          btn##setAttribute (js "aria-pressed") (js (if act then "true" else "false"))
        ) pill_buttons
      in

      let sym_vis_tbl : (string, bool) Hashtbl.t = Hashtbl.create 96 in

      let apply_filter () =
        let raw_q = String.trim (Js.to_string inp##.value) in
        let tokens = split_tokens raw_q in
        let is_filtered = (tokens <> []) || (!active_kind <> "all") in

        Hashtbl.clear sym_vis_tbl;
        let visible_count = ref 0 in
        List.iter (fun ent ->
          let kind_ok =
            if !active_kind = "all" then true
            else if !active_kind = "examples" then ent.ent_has_example
            else ent.ent_kind = !active_kind
          in
          let tok_ok = List.for_all (fun tok -> string_contains ent.ent_haystack tok) tokens in
          let ok = kind_ok && tok_ok in
          ent.ent_visible <- ok;
          if ok then incr visible_count;
          if ent.ent_symbol <> "" then Hashtbl.replace sym_vis_tbl ent.ent_symbol ok;
          set_hidden ent.ent_el (not ok)
        ) entries;

        (* Sync summary tables & rows *)
        List.iter (fun tbl ->
          if tbl.tbl_rows <> [] then begin
            let vis_rows = ref 0 in
            List.iter (fun row ->
              let row_ok =
                match Hashtbl.find_opt sym_vis_tbl row.row_symbol with
                | Some ok -> ok
                | None ->
                    (!active_kind = "all")
                    && List.for_all (fun tok -> string_contains row.row_haystack tok) tokens
              in
              if row_ok then incr vis_rows;
              set_hidden row.row_el (not row_ok)
            ) tbl.tbl_rows;
            set_hidden tbl.tbl_el (!vis_rows = 0 && is_filtered)
          end
        ) tables;

        (* Hide module sections whose .api-entry elements all failed the filter *)
        List.iter (fun ms ->
          let any_vis =
            List.exists (fun el ->
              not (Js.to_bool (el##.classList##contains (js "api-hidden")))
            ) ms.sec_entries
          in
          set_hidden ms.sec_el ((not any_vis) && is_filtered)
        ) module_sections;

        (* Update live count & Clear button *)
        let count_label =
          if is_filtered then
            Printf.sprintf "%d of %d symbols" !visible_count total_count
          else
            Printf.sprintf "%d %s" total_count (if total_count = 1 then "symbol" else "symbols")
        in
        count_el##.textContent := Js.some (js count_label);
        set_hidden (clear_btn :> element Js.t) (raw_q = "" && !active_kind = "all");

        (* Empty state message via textContent only *)
        if total_count > 0 && !visible_count = 0 && is_filtered then begin
          let msg =
            if raw_q <> "" && !active_kind <> "all" then
              let k_lbl =
                if !active_kind = "class" then "class"
                else if !active_kind = "function" then "function"
                else if !active_kind = "examples" then "example"
                else "matching"
              in
              Printf.sprintf "No %s symbols match \"%s\"." k_lbl raw_q
            else if raw_q <> "" then
              Printf.sprintf "No symbols match \"%s\"." raw_q
            else if !active_kind = "examples" then
              "No symbols with runnable examples found."
            else
              "No matching symbols found."
          in
          empty_el##.textContent := Js.some (js msg);
          set_hidden (empty_el :> element Js.t) false
        end else begin
          empty_el##.textContent := Js.some (js "");
          set_hidden (empty_el :> element Js.t) true
        end
      in

      let reset_filter () =
        inp##.value := js "";
        active_kind := "all";
        sync_pills ();
        apply_filter ()
      in
      reset_filter_ref := reset_filter;

      (* Bind pill clicks *)
      List.iter (fun (k, (btn : buttonElement Js.t)) ->
        add_click (btn :> element Js.t) (fun () ->
          active_kind := k;
          sync_pills ();
          apply_filter ()
        )
      ) pill_buttons;

      (* Bind Clear button *)
      add_click (clear_btn :> element Js.t) (fun () ->
        reset_filter ();
        inp##focus
      );

      (* Bind live input & search events *)
      let on_input = handler (fun _ev ->
        apply_filter ();
        Js._true
      ) in
      let _ = addEventListener inp (Event.make "input") on_input Js._false in
      let _ = addEventListener inp (Event.make "search") on_input Js._false in

      (* Bind Enter / Escape inside #api-filter-input *)
      let on_inp_keydown = handler (fun (ev : keyboardEvent Js.t) ->
        let k =
          match Js.Optdef.to_option ev##.key with
          | Some s -> Js.to_string s
          | None -> ""
        in
        if k = "Escape" then begin
          Dom.preventDefault ev;
          reset_filter ();
          inp##blur
        end else if k = "Enter" then begin
          Dom.preventDefault ev;
          match List.find_opt (fun ent -> ent.ent_visible) entries with
          | Some first_ent ->
              (try
                 let opts = Js.Unsafe.obj [|
                   ("behavior", Js.Unsafe.inject (js "smooth"));
                   ("block", Js.Unsafe.inject (js "start"));
                 |] in
                 ignore ((Js.Unsafe.coerce first_ent.ent_el)##scrollIntoView opts)
               with _ ->
                 first_ent.ent_el##scrollIntoView Js._true)
          | None -> ()
        end;
        Js._true
      ) in
      let _ = addEventListener inp Event.keydown on_inp_keydown Js._false in

      (* Global '/' shortcut to focus #api-filter-input *)
      let is_editable_focused () =
        match Js.Opt.to_option document##.activeElement with
        | None -> false
        | Some active ->
            let tag = String.uppercase_ascii (Js.to_string active##.tagName) in
            let ce =
              let h : htmlElement Js.t = Js.Unsafe.coerce active in
              Js.to_bool (Js.Unsafe.get h (js "isContentEditable"))
            in
            tag = "INPUT" || tag = "TEXTAREA" || tag = "SELECT" || ce
      in
      let on_doc_keydown = handler (fun (ev : keyboardEvent Js.t) ->
        let k =
          match Js.Optdef.to_option ev##.key with
          | Some s -> Js.to_string s
          | None -> ""
        in
        let no_mods =
          not (Js.to_bool ev##.ctrlKey)
          && not (Js.to_bool ev##.metaKey)
          && not (Js.to_bool ev##.altKey)
        in
        if k = "/" && no_mods && not (is_editable_focused ()) then begin
          Dom.preventDefault ev;
          inp##focus;
          inp##select
        end;
        Js._true
      ) in
      let _ = addEventListener document Event.keydown on_doc_keydown Js._false in

      (* Reveal hidden entry if user navigates to a hash anchor while filtered *)
      let on_hashchange = handler (fun _ev ->
        let raw_hash = Js.to_string window##.location##.hash in
        let target_id =
          if String.length raw_hash > 1 && raw_hash.[0] = '#' then
            String.sub raw_hash 1 (String.length raw_hash - 1)
          else ""
        in
        if target_id <> "" then begin
          match List.find_opt (fun e -> e.ent_id = target_id && not e.ent_visible) entries with
          | Some hidden_ent ->
              reset_filter ();
              hidden_ent.ent_el##scrollIntoView Js._true
          | None -> ()
        end;
        Js._true
      ) in
      let _ = addEventListener window (Event.make "hashchange") on_hashchange Js._false in

      apply_filter ()

(* --- Central 3D WebGL CAD Simulation (FUSE.jl Tokamaks & Oceananigans.jl 3D Fluid Tanks) --- *)
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

  // Exact 25-vertex TCV vacuum vessel inner contour extracted from arXiv:2505.00663 Fig. 9
  var VESSEL_INNER_RZ = [
    [0.6240,  0.0000], [0.6241,  0.7091], [0.6475,  0.7270], [0.6662,  0.7500],
    [0.9733,  0.7498], [0.9907,  0.7252], [1.1289,  0.5599], [1.1360,  0.5351],
    [1.1360, -0.3282], [1.1402, -0.3460], [1.0869, -0.3764], [1.0707, -0.3994],
    [1.0586, -0.4599], [1.0829, -0.5808], [1.0967, -0.5905], [1.1165, -0.5669],
    [1.1289, -0.5598], [0.9907, -0.7252], [0.9733, -0.7498], [0.8506, -0.7500],
    [0.6662, -0.7500], [0.6475, -0.7270], [0.6241, -0.7091], [0.6240, -0.4933]
  ];

  var VESSEL_OUTER_RZ = [
    [1.174, -0.575], [1.174,  0.555], [0.988,  0.792], [0.654,  0.792],
    [0.592,  0.725], [0.592, -0.725], [0.654, -0.792], [0.988, -0.792]
  ];

  var VESSEL_WALL_IN_RZ = [
    [1.134, -0.558], [1.134,  0.535], [0.973,  0.748], [0.666,  0.748],
    [0.624,  0.708], [0.624, -0.708], [0.666, -0.748], [0.973, -0.748]
  ];

  var TCV_TF_OUT = [
    [1.52, -1.06], [1.52, 1.06], [0.36, 1.06], [0.36, -1.06]
  ];
  var TCV_TF_IN = [
    [1.40, -0.94], [1.40, 0.94], [0.53, 0.94], [0.53, -0.94]
  ];

  var TCV_CAD = {
    id: "tcv",
    kind: "tokamak",
    solver: "EPFL TokaGym + TORAX 2.0.0",
    name: "TCV (Tokamak à Configuration Variable)",
    shortName: "TCV · EPFL",
    R0: 0.88, Z0: 0.0, a: 0.25, B0: 1.43, ipNominal: 1.49, q95: 3.24, betaN: 1.85, pFusMW: 0.0,
    dist: 5.75,
    rMin: 0.30, rMax: 1.56, zMin: -1.12, zMax: 1.12,
    rTicks: [0.5, 0.9, 1.3],
    zTicks: [-0.8, 0.0, 0.8],
    n_tf: 16,
    tf_outer: TCV_TF_OUT,
    tf_inner: TCV_TF_IN,
    vessel_outer: VESSEL_OUTER_RZ,
    vessel_inner: VESSEL_WALL_IN_RZ,
    first_wall: VESSEL_INNER_RZ,
    coils: [
      {name: "OH1-Top", r: 0.325, z: 0.885, w: 0.100, h: 0.090, current_ma: 0.62},
      {name: "OH1-Bot", r: 0.325, z: -0.885, w: 0.100, h: 0.090, current_ma: 0.58},
      {name: "OH2-Top", r: 0.455, z: 0.910, w: 0.080, h: 0.100, current_ma: 0.44},
      {name: "OH2-Bot", r: 0.455, z: -0.910, w: 0.080, h: 0.100, current_ma: 0.21},
      {name: "E1", r: 0.480, z: -0.672, w: 0.096, h: 0.176, current_ma: -0.18},
      {name: "E2", r: 0.480, z: -0.480, w: 0.096, h: 0.176, current_ma: -0.45},
      {name: "E3", r: 0.480, z: -0.288, w: 0.096, h: 0.176, current_ma: -0.68},
      {name: "E4", r: 0.480, z: -0.096, w: 0.096, h: 0.176, current_ma: -0.52},
      {name: "E5", r: 0.480, z:  0.096, w: 0.096, h: 0.176, current_ma: 0.49},
      {name: "E6", r: 0.480, z:  0.288, w: 0.096, h: 0.176, current_ma: 0.64},
      {name: "E7", r: 0.480, z:  0.480, w: 0.096, h: 0.176, current_ma: 0.35},
      {name: "E8", r: 0.480, z:  0.672, w: 0.096, h: 0.176, current_ma: -0.12},
      {name: "F1", r: 1.296, z: -0.720, w: 0.143, h: 0.160, current_ma: -0.38},
      {name: "F2", r: 1.296, z: -0.518, w: 0.143, h: 0.160, current_ma: -0.61},
      {name: "F3", r: 1.296, z: -0.316, w: 0.143, h: 0.160, current_ma: 0.72},
      {name: "F4", r: 1.296, z: -0.114, w: 0.143, h: 0.160, current_ma: 0.55},
      {name: "F5", r: 1.296, z:  0.088, w: 0.143, h: 0.160, current_ma: -0.42},
      {name: "F6", r: 1.296, z:  0.290, w: 0.143, h: 0.160, current_ma: -0.41},
      {name: "F7", r: 1.296, z:  0.492, w: 0.143, h: 0.160, current_ma: 0.14},
      {name: "F8", r: 1.296, z:  0.694, w: 0.143, h: 0.160, current_ma: 0.22}
    ]
  };

  // Baseline parametric tokamak shells prior to async FUSE.jl `assets/baux_env.json` hydration
  function makeDLoop(rIn, rOut, zMax, nPts) {
    var out = [];
    for (var i = 0; i < nPts; i++) {
      var th = (2.0 * Math.PI * i) / nPts;
      var rm = 0.5 * (rIn + rOut);
      var a = 0.5 * (rOut - rIn);
      var r = rm + a * Math.cos(th + 0.36 * Math.sin(th));
      var z = zMax * Math.sin(th);
      out.push([r, z]);
    }
    return out;
  }

  var SPARC_CAD = {
    id: "sparc",
    kind: "tokamak",
    solver: "FUSE.jl 1.2.1 · TEQUILA + ActorPFactive",
    name: "SPARC (Commonwealth Fusion Systems)",
    shortName: "SPARC · CFS (FUSE.jl)",
    R0: 1.85, Z0: 0.0, a: 0.53, B0: 12.2, ipNominal: 8.70, q95: 3.00, betaN: 1.55, pFusMW: 140.0,
    dist: 12.8,
    rMin: 0.35, rMax: 3.90, zMin: -2.40, zMax: 2.40,
    rTicks: [1.0, 2.0, 3.0],
    zTicks: [-1.5, 0.0, 1.5],
    n_tf: 18,
    tf_outer: makeDLoop(0.70, 3.72, 2.15, 64),
    tf_inner: makeDLoop(1.04, 3.08, 1.82, 64),
    vessel_outer: makeDLoop(1.06, 2.72, 1.48, 64),
    vessel_inner: makeDLoop(1.13, 2.65, 1.40, 64),
    first_wall: makeDLoop(1.16, 2.62, 1.35, 64),
    coils: [
      {name: "CS1U", r: 0.56, z: 0.48, w: 0.24, h: 0.82, current_ma: -3.2},
      {name: "CS1L", r: 0.56, z: -0.48, w: 0.24, h: 0.82, current_ma: -3.2},
      {name: "CS2U", r: 0.56, z: 1.35, w: 0.24, h: 0.78, current_ma: -2.4},
      {name: "CS2L", r: 0.56, z: -1.35, w: 0.24, h: 0.78, current_ma: -2.4},
      {name: "PF1U", r: 1.15, z: 2.05, w: 0.30, h: 0.30, current_ma: 2.1},
      {name: "PF1L", r: 1.15, z: -2.05, w: 0.30, h: 0.30, current_ma: 2.1},
      {name: "PF3U", r: 3.35, z: 1.42, w: 0.28, h: 0.28, current_ma: -1.8},
      {name: "PF3L", r: 3.35, z: -1.42, w: 0.28, h: 0.28, current_ma: -1.8}
    ]
  };

  var ARC_CAD = {
    id: "arc",
    kind: "tokamak",
    solver: "FUSE.jl 1.2.1 · ARCv3a IMAS + TEQUILA",
    name: "ARCv3a (MIT PSFC / CFS Compact Pilot Plant)",
    shortName: "ARC · MIT/CFS (FUSE.jl)",
    R0: 4.62, Z0: 0.0, a: 1.18, B0: 11.4, ipNominal: 12.00, q95: 3.53, betaN: 2.25, pFusMW: 525.0,
    dist: 28.5,
    rMin: 0.65, rMax: 9.10, zMin: -5.40, zMax: 5.40,
    rTicks: [2.0, 4.5, 7.0],
    zTicks: [-4.0, -2.0, 0.0, 2.0, 4.0],
    n_tf: 18,
    tf_outer: makeDLoop(1.48, 8.85, 4.95, 64),
    tf_inner: makeDLoop(2.38, 7.68, 4.30, 64),
    vessel_outer: makeDLoop(3.31, 5.89, 3.85, 64),
    vessel_inner: makeDLoop(3.38, 5.81, 3.74, 64),
    first_wall: makeDLoop(3.39, 5.80, 3.68, 64),
    coils: [
      {name: "cs1u", r: 1.105, z: 0.750, w: 0.756, h: 1.387, current_ma: -8.2},
      {name: "cs1l", r: 1.105, z: -0.750, w: 0.756, h: 1.387, current_ma: -8.2},
      {name: "pf1u", r: 1.400, z: 3.900, w: 0.590, h: 0.590, current_ma: 5.1},
      {name: "pf1l", r: 1.400, z: -3.900, w: 0.590, h: 0.590, current_ma: 5.1},
      {name: "pf3u", r: 6.200, z: 2.950, w: 0.489, h: 0.489, current_ma: -3.8},
      {name: "pf3l", r: 6.200, z: -2.950, w: 0.489, h: 0.489, current_ma: -3.8}
    ]
  };

  var ITER_CAD = {
    id: "iter",
    kind: "tokamak",
    solver: "FUSE.jl 1.2.1 · ITER ODS + Equilibrium",
    name: "ITER (Q = 10 Burning Plasma Baseline)",
    shortName: "ITER · Cadarache (FUSE.jl)",
    R0: 6.20, Z0: 0.35, a: 2.00, B0: 5.30, ipNominal: 15.00, q95: 3.05, betaN: 1.80, pFusMW: 500.0,
    dist: 41.5,
    rMin: 1.10, rMax: 12.60, zMin: -8.20, zMax: 8.20,
    rTicks: [3.0, 6.0, 9.0, 12.0],
    zTicks: [-6.0, -3.0, 0.0, 3.0, 6.0],
    n_tf: 18,
    tf_outer: makeDLoop(2.65, 10.65, 6.80, 64),
    tf_inner: makeDLoop(3.52, 9.75, 5.90, 64),
    vessel_outer: makeDLoop(3.72, 8.84, 4.95, 64),
    vessel_inner: makeDLoop(3.92, 8.60, 4.72, 64),
    first_wall: makeDLoop(4.10, 8.38, 4.52, 64),
    coils: [
      {name: "CS3U", r: 1.687, z: 5.464, w: 0.740, h: 2.093, current_ma: -6.5},
      {name: "CS2U", r: 1.687, z: 3.278, w: 0.740, h: 2.093, current_ma: -9.8},
      {name: "CS1U", r: 1.687, z: 1.092, w: 0.740, h: 2.093, current_ma: -12.1},
      {name: "CS1L", r: 1.687, z: -1.072, w: 0.740, h: 2.093, current_ma: -12.1},
      {name: "CS2L", r: 1.687, z: -3.258, w: 0.740, h: 2.093, current_ma: -9.8},
      {name: "CS3L", r: 1.687, z: -5.444, w: 0.740, h: 2.093, current_ma: -6.5},
      {name: "PF1", r: 3.943, z: 7.574, w: 0.959, h: 0.984, current_ma: 4.8},
      {name: "PF2", r: 8.285, z: 6.540, w: 0.580, h: 0.715, current_ma: -2.6},
      {name: "PF3", r: 11.992, z: 3.275, w: 0.696, h: 0.954, current_ma: -4.2},
      {name: "PF4", r: 11.963, z: -2.234, w: 0.638, h: 0.954, current_ma: -3.9},
      {name: "PF5", r: 8.391, z: -6.727, w: 0.812, h: 0.954, current_ma: -5.4},
      {name: "PF6", r: 4.334, z: -7.466, w: 1.559, h: 1.107, current_ma: 11.2}
    ]
  };

  var D3D_CAD = {
    id: "d3d",
    kind: "tokamak",
    solver: "FUSE.jl 1.2.1 · DIII-D H-mode EFIT",
    name: "DIII-D National Fusion Facility (H-mode)",
    shortName: "DIII-D · GA (FUSE.jl)",
    R0: 1.67, Z0: 0.0, a: 0.61, B0: 2.17, ipNominal: 1.60, q95: 4.15, betaN: 2.10, pFusMW: 0.0,
    dist: 11.2,
    rMin: 0.40, rMax: 3.25, zMin: -2.15, zMax: 2.15,
    rTicks: [0.8, 1.6, 2.4, 3.0],
    zTicks: [-1.5, 0.0, 1.5],
    n_tf: 24,
    tf_outer: makeDLoop(0.52, 3.10, 2.02, 64),
    tf_inner: makeDLoop(0.76, 2.82, 1.78, 64),
    vessel_outer: makeDLoop(0.88, 2.52, 1.50, 64),
    vessel_inner: makeDLoop(0.95, 2.44, 1.42, 64),
    first_wall: makeDLoop(1.01, 2.36, 1.35, 64),
    coils: [
      {name: "PF6", r: 0.861, z: 0.168, w: 0.051, h: 0.321, current_ma: 0.25},
      {name: "PF7", r: 0.861, z: 0.508, w: 0.051, h: 0.321, current_ma: 0.31},
      {name: "PF13", r: 1.252, z: 1.602, w: 0.235, h: 0.085, current_ma: 0.42},
      {name: "PF14", r: 1.689, z: 1.587, w: 0.169, h: 0.133, current_ma: -0.28},
      {name: "PF22", r: 1.252, z: -1.603, w: 0.235, h: 0.085, current_ma: 0.48},
      {name: "PF23", r: 1.689, z: -1.578, w: 0.169, h: 0.133, current_ma: -0.32}
    ]
  };

  // Baseline Oceananigans.jl 3D Fluid Tank environments (hydrated asynchronously with exact PDE grids from assets/baux_env.json)
  function synthKH(nx, nz, billowAmp) {
    var b = [], u = [], w = [];
    for (var k = 0; k < nz; k++) {
      var z = -5.0 + 10.0 * (k + 0.5) / nz;
      for (var i = 0; i < nx; i++) {
        var x = -5.0 + 10.0 * (i + 0.5) / nx;
        var kx = (4.0 * Math.PI * x) / 10.0;
        var env = Math.exp(-0.5 * (z / 0.85) * (z / 0.85));
        var disp = billowAmp * env * Math.sin(kx - 0.55 * z);
        var zb = z - 1.35 * disp;
        b.push(0.025 * Math.tanh(zb / 0.32));
        u.push(Math.tanh(zb) - 0.45 * billowAmp * env * Math.cos(kx));
        w.push(0.55 * billowAmp * env * Math.cos(kx));
      }
    }
    return { b: b, u: u, w: w };
  }
  var kh0 = synthKH(32, 32, 0.85);
  var kh1 = synthKH(32, 32, 0.16);

  var OCEAN_KH = {
    id: "ocean_kh",
    kind: "ocean",
    solver: "Oceananigans.jl 0.117.2 · NonhydrostaticModel (RK3, Upwind-5)",
    name: "Stratified Kelvin–Helmholtz 3D Shear Channel",
    shortName: "KH Shear Tank (Oceananigans.jl)",
    nx: 32, ny: 4, nz: 32, dim: 8192,
    xMin: -5.0, xMax: 5.0, yMin: -2.5, yMax: 2.5, zMin: -5.0, zMax: 5.0,
    dist: 15.6, ri: 0.10, re: 5000.0,
    score_init: -1.00, score_opt: -0.34,
    actuators: [
      {name: "S-Act-1", x: -4.2, y: 0.0, z: -0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-2", x: -3.0, y: 0.0, z:  0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-3", x: -1.8, y: 0.0, z: -0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-4", x: -0.6, y: 0.0, z:  0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-5", x:  0.6, y: 0.0, z: -0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-6", x:  1.8, y: 0.0, z:  0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-7", x:  3.0, y: 0.0, z: -0.85, wx: 0.72, wy: 4.6, wz: 0.24},
      {name: "S-Act-8", x:  4.2, y: 0.0, z:  0.85, wx: 0.72, wy: 4.6, wz: 0.24}
    ],
    b_init: kh0.b, u_init: kh0.u, w_init: kh0.w,
    b_opt: kh1.b, u_opt: kh1.u, w_opt: kh1.w
  };

  function synthWM(nx, nz, matched) {
    var b = [], u = [], w = [];
    for (var k = 0; k < nz; k++) {
      var z = -128.0 + 128.0 * (k + 0.5) / nz;
      for (var i = 0; i < nx; i++) {
        var x = 1000.0 * (i + 0.5) / nx;
        var kx = (2.0 * Math.PI * x) / 1000.0;
        var mld = matched ? (-32.0 - 10.0 * Math.cos(kx)) : (-14.0 - 3.0 * Math.sin(kx));
        var strat = 0.01 * z;
        var mixed = (z > mld) ? (0.01 * mld + 0.02 * (z - mld) / 32.0) : strat;
        b.push(mixed);
        u.push((z > mld ? -0.18 : -0.02) * (1.0 + 0.45 * Math.cos(kx)) * (matched ? 1.0 : 0.25));
        w.push(0.0);
      }
    }
    return { b: b, u: u, w: w };
  }
  var wm0 = synthWM(24, 24, false);
  var wm1 = synthWM(24, 24, true);

  var OCEAN_WIND = {
    id: "ocean_wind",
    kind: "ocean",
    solver: "Oceananigans.jl 0.117.2 · HydrostaticFreeSurfaceModel (FPlane)",
    name: "3D Ocean Wind-Stress & Cooling Mixed-Layer Basin",
    shortName: "Wind-Mixing Basin (Oceananigans.jl)",
    nx: 24, ny: 4, nz: 24, dim: 192,
    xMin: 0.0, xMax: 1000.0, yMin: 0.0, yMax: 500.0, zMin: -128.0, zMax: 0.0,
    dist: 15.8, ri: 0.25, re: 12800.0,
    score_init: -0.92, score_opt: -0.04,
    actuators: [
      {name: "Flux-Plate-1", x: 80.0,  y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-2", x: 220.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-3", x: 360.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-4", x: 500.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-5", x: 640.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-6", x: 780.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5},
      {name: "Flux-Plate-7", x: 920.0, y: 250.0, z: -2.5, wx: 95.0, wy: 440.0, wz: 4.5}
    ],
    b_init: wm0.b, u_init: wm0.u, w_init: wm0.w,
    b_opt: wm1.b, u_opt: wm1.u, w_opt: wm1.w
  };

  var MACHINES = {
    tcv: TCV_CAD,
    sparc: SPARC_CAD,
    arc: ARC_CAD,
    iter: ITER_CAD,
    d3d: D3D_CAD,
    ocean_kh: OCEAN_KH,
    ocean_wind: OCEAN_WIND
  };

  var LIVE_RUNS = {
    fuseWorkerS: 5.83,
    oceanWorkerS: 0.24
  };

  // Exact 32 shape_70166 reference LCFS control points extracted from arXiv:2505.00663 Fig. 9
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

  // Verified 10-Seed x 10,000-Round Paired TORAX 2.0.0 TuRBO-zero Benchmark Data (D = 160,531)
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
      coils: [0.62, 0.58, 0.44, 0.21, -0.18, -0.45, -0.68, -0.52, 0.49, 0.64, 0.35, -0.12, -0.38, -0.61, 0.72, 0.55, -0.42, -0.41, 0.14, 0.22]
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
      coils: [0.55, 0.51, 0.39, 0.18, -0.15, -0.41, -0.62, -0.48, 0.44, 0.58, 0.31, -0.15, -0.34, -0.55, 0.66, 0.49, -0.39, -0.44, 0.19, 0.18]
    },
    init: {
      name: "Initial Unoptimized θ₀ (Round 0)",
      short: "Initial θ₀",
      color: "#dc2626",
      accent: "#ef4444",
      mean: [5.97, 5.97, 5.97, 5.97, 5.97, 5.97, 5.97, 5.97],
      sem:  [1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13, 1.13],
      seeds10k: [3.95, 6.14, 1.59, 8.59, 7.93, 12.63, 1.37, 7.33, 6.74, 3.46],
      dR: -0.048, dZ: -0.135, dKappa: -0.34, dTri: -0.25,
      comps: [0.062, 0.115, 0.084, 0.190, 0.045],
      te0: 3.2, ti0: 2.6, ne0: 0.45, ip: 0.72, q95: 2.12,
      coils: [-0.12, 0.08, -0.19, 0.24, -0.05, 0.14, -0.22, 0.11, -0.18, 0.09, 0.21, -0.15, 0.07, -0.13, 0.16, -0.09, -0.82, 0.44, -0.68, -0.31]
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
    if (polKey === "indep") progress *= 0.89;
    var initPol = BENCH.init;

    var dR = lerp(initPol.dR, pol.dR, progress);
    var dZ = lerp(initPol.dZ, pol.dZ, progress);
    var dKappa = lerp(initPol.dKappa, pol.dKappa, progress);
    var dTri = lerp(initPol.dTri, pol.dTri, progress);
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
      progress: progress,
      score: score,
      sem: semVal,
      dR: dR,
      dZ: dZ,
      dKappa: dKappa,
      dTri: dTri,
      q95: q95,
      comps: comps,
      coils: coils,
      lcfsErrMm: lcfsErrMm
    };
  }

  // Solved FUSE.jl TEQUILA LCFS contour (when mach.lcfs is present) or analytical Miller fallback
  function plasmaContour(rho, st, nPts, mach) {
    mach = mach || TCV_CAD;
    var scaleA = mach.a / 0.25;
    if (mach.lcfs && mach.lcfs.length >= 16) {
      var axR0 = (typeof mach.axisR === "number") ? mach.axisR : mach.R0;
      var axZ0 = (typeof mach.axisZ === "number") ? mach.axisZ : (mach.Z0 || 0.0);
      var curR0 = axR0 + st.dR * 0.45 * scaleA;
      var curZ0 = axZ0 + st.dZ * 0.45 * scaleA;
      var kS = (1.0 + st.dKappa * 0.35) * (0.84 + 0.16 * rho);
      var rS = 1.0 + st.dTri * 0.18;
      var outPts = [];
      var src = mach.lcfs;
      for (var j = 0; j < src.length; j++) {
        var dr = (src[j][0] - axR0) * rho * rS;
        var dz = (src[j][1] - axZ0) * rho * kS;
        outPts.push([curR0 + dr, curZ0 + dz]);
      }
      var rxS = (typeof mach.rx === "number") ? (curR0 + (mach.rx - axR0)) : (curR0 - mach.a * 0.40);
      var zxS = (typeof mach.zx === "number") ? (curZ0 + (mach.zx - axZ0) * kS) : (curZ0 - mach.a * 1.55);
      return { pts: outPts, R0: curR0, Z0: curZ0, Rx: rxS, Zx: zxS };
    }

    var pts = [];
    var R0 = mach.R0 + (mach.id === "tcv" ? st.dR * 0.7 : st.dR * 0.35 * scaleA);
    var Z0 = (mach.id === "tcv" ? (-0.045 + st.dZ * 0.8) : ((mach.Z0 || 0.0) + st.dZ * 0.35 * scaleA));
    var a = (mach.a * 0.86) * rho;
    var baseKappa = (mach.id === "tcv") ? (1.58 + st.dKappa * 0.45) : (1.78 + st.dKappa * 0.25);
    var kappa = baseKappa * (0.78 + 0.22 * rho);
    var deltaU = ((mach.id === "tcv" ? (0.36 + st.dTri * 0.4) : 0.38)) * rho;
    var deltaL = ((mach.id === "tcv" ? (0.52 + st.dTri * 0.4) : 0.48)) * rho;
    var xptPull = (rho > 0.55) ? Math.pow((rho - 0.55) / 0.45, 2.2) * (mach.a * 0.32) : 0.0;
    for (var i = 0; i < nPts; i++) {
      var th = (2.0 * Math.PI * i) / nPts;
      var s = Math.sin(th);
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
    "varying float vUnlit;",
    "void main() {",
    "  vec4 wp = uModel * vec4(aPos, 1.0);",
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
    "varying float vUnlit;",
    "void main() {",
    "  if (vUnlit > 0.5) {",
    "    gl_FragColor = vCol;",
    "    return;",
    "  }",
    "  vec3 n = normalize(vNorm);",
    "  vec3 v = normalize(vViewDir);",
    "  if (dot(n, v) < 0.0) n = -n;",
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
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }

  function mat4Mul(a, b) {
    var out = new Float32Array(16);
    for (var i = 0; i < 4; i++) {
      for (var j = 0; j < 4; j++) {
        var sum = 0.0;
        for (var k = 0; k < 4; k++) sum += a[k * 4 + i] * b[j * 4 + k];
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
    out[15] = 1.0;
    return out;
  }

  function cylToXYZ(R, Z, phi, off) {
    off = off || 0.0;
    var c = Math.cos(phi);
    var s = Math.sin(phi);
    return [R * c - off * s, Z, R * s + off * c];
  }

  // Thermal/haline scientific colormap t in [0, 1] -> [r, g, b, 1]
  function oceanPalette(t) {
    var u = Math.max(0.0, Math.min(1.0, t));
    if (u < 0.25) {
      var s = u / 0.25;
      return [lerp(0.07, 0.08, s), lerp(0.16, 0.42, s), lerp(0.38, 0.68, s), 1.0];
    } else if (u < 0.50) {
      var s2 = (u - 0.25) / 0.25;
      return [lerp(0.08, 0.18, s2), lerp(0.42, 0.72, s2), lerp(0.68, 0.72, s2), 1.0];
    } else if (u < 0.75) {
      var s3 = (u - 0.50) / 0.25;
      return [lerp(0.18, 0.92, s3), lerp(0.72, 0.74, s3), lerp(0.72, 0.24, s3), 1.0];
    } else {
      var s4 = (u - 0.75) / 0.25;
      return [lerp(0.92, 0.96, s4), lerp(0.74, 0.36, s4), lerp(0.24, 0.14, s4), 1.0];
    }
  }

  // Build 3D CAD mesh buffers for Oceananigans.jl 3D Stratified Shear Tank / Wind-Mixing Basin
  function buildOceanCADBuffers(gl, mach, st, opts) {
    var pos = [], norm = [], col = [], linePos = [], lineCol = [];
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
    function addBox3D(x0, x1, y0, y1, z0, z1, sideCol, topCol, edgeCol) {
      // WebGL coords: X = x, Y = z (vertical), Z = y (spanwise depth)
      var p000 = [x0, y0, z0], p100 = [x1, y0, z0], p110 = [x1, y1, z0], p010 = [x0, y1, z0];
      var p001 = [x0, y0, z1], p101 = [x1, y0, z1], p111 = [x1, y1, z1], p011 = [x0, y1, z1];
      var tc = topCol || sideCol;
      // Front (+Z), Back (-Z), Left (-X), Right (+X), Top (+Y), Bottom (-Y)
      pushTri(p001, p101, p111, [0,0,1], [0,0,1], [0,0,1], sideCol);
      pushTri(p001, p111, p011, [0,0,1], [0,0,1], [0,0,1], sideCol);
      pushTri(p100, p000, p010, [0,0,-1], [0,0,-1], [0,0,-1], sideCol);
      pushTri(p100, p010, p110, [0,0,-1], [0,0,-1], [0,0,-1], sideCol);
      pushTri(p000, p001, p011, [-1,0,0], [-1,0,0], [-1,0,0], sideCol);
      pushTri(p000, p011, p010, [-1,0,0], [-1,0,0], [-1,0,0], sideCol);
      pushTri(p101, p100, p110, [1,0,0], [1,0,0], [1,0,0], sideCol);
      pushTri(p101, p110, p111, [1,0,0], [1,0,0], [1,0,0], sideCol);
      pushTri(p011, p111, p110, [0,1,0], [0,1,0], [0,1,0], tc);
      pushTri(p011, p110, p010, [0,1,0], [0,1,0], [0,1,0], tc);
      pushTri(p000, p100, p101, [0,-1,0], [0,-1,0], [0,-1,0], sideCol);
      pushTri(p000, p101, p001, [0,-1,0], [0,-1,0], [0,-1,0], sideCol);
      if (edgeCol) {
        pushLine(p011, p111, edgeCol); pushLine(p111, p110, edgeCol);
        pushLine(p110, p010, edgeCol); pushLine(p010, p011, edgeCol);
      }
    }

    // Map physical tank box to canonical 3D CAD engineering envelope: X in [-4.6, 4.6], Y_vert in [-2.35, 2.35], Z_depth in [-2.3, 2.3]
    var XW = 4.6, YH = 2.35, ZD = 2.25, wt = 0.16;
    function mapX(xp) { return -XW + 2.0 * XW * ((xp - mach.xMin) / Math.max(1e-6, mach.xMax - mach.xMin)); }
    function mapY(zp) { return -YH + 2.0 * YH * ((zp - mach.zMin) / Math.max(1e-6, mach.zMax - mach.zMin)); }
    function mapZ(yp) { return -ZD + 2.0 * ZD * ((yp - mach.yMin) / Math.max(1e-6, mach.yMax - mach.yMin)); }

    var steelOuter = [0.55, 0.58, 0.63, 1.0];
    var rimWhite   = [0.96, 0.97, 0.99, 1.0];
    var rimEdge    = [0.16, 0.18, 0.22, 0.85];

    // 1. Cutaway Structural Tank Shell (Baseplate, Rear Wall, Left Wall, Half-Cut Right Flange)
    addBox3D(-XW - wt, XW + wt, -YH - wt, -YH, -ZD - wt, ZD + wt, steelOuter, [0.22, 0.25, 0.30, 1.0], rimEdge);
    addBox3D(-XW - wt, XW + wt, -YH, YH + 0.08, ZD, ZD + wt, steelOuter, rimWhite, rimEdge);
    addBox3D(-XW - wt, -XW, -YH, YH + 0.08, -ZD, ZD + wt, steelOuter, rimWhite, rimEdge);
    addBox3D(XW, XW + wt, -YH, -0.35 * YH, -ZD, ZD + wt, steelOuter, rimWhite, rimEdge);

    // Front transparent acrylic/glass CAD frame edges (facing viewer at Z = -ZD)
    var frameCol = [0.42, 0.52, 0.64, 0.85];
    pushLine([-XW, -YH, -ZD], [XW, -YH, -ZD], frameCol);
    pushLine([-XW, YH, -ZD], [XW, YH, -ZD], frameCol);
    pushLine([-XW, -YH, -ZD], [-XW, YH, -ZD], frameCol);
    pushLine([XW, -YH, -ZD], [XW, YH, -ZD], frameCol);
    pushLine([XW, -YH, ZD], [XW, YH, ZD], frameCol);
    pushLine([XW, YH, -ZD], [XW, YH, ZD], frameCol);

    // 2. Interpolate solved Oceananigans.jl PDE buoyancy & velocity fields across Rollout Round
    var nx = mach.nx || 32, nz = mach.nz || 32;
    var prog = st.progress;
    var bField = new Float32Array(nx * nz);
    var uField = new Float32Array(nx * nz);
    var wField = new Float32Array(nx * nz);
    var bMin = 1e9, bMax = -1e9;
    for (var idx = 0; idx < nx * nz; idx++) {
      var bv = lerp(mach.b_init[idx] || 0.0, mach.b_opt[idx] || 0.0, prog);
      var uv = lerp(mach.u_init[idx] || 0.0, mach.u_opt[idx] || 0.0, prog);
      var wv = lerp((mach.w_init && mach.w_init[idx]) || 0.0, (mach.w_opt && mach.w_opt[idx]) || 0.0, prog);
      bField[idx] = bv;
      uField[idx] = uv;
      wField[idx] = wv;
      if (bv < bMin) bMin = bv;
      if (bv > bMax) bMax = bv;
    }
    var bSpan = Math.max(1e-6, bMax - bMin);
    function bAt(i, k) {
      var ii = Math.max(0, Math.min(nx - 1, i));
      var kk = Math.max(0, Math.min(nz - 1, k));
      return bField[kk * nx + ii];
    }

    // 3. Rear Interior Fluid Cutaway Cross-Section + Right Cutaway Face
    var zBack = ZD - 0.02;
    for (var k = 0; k < nz - 1; k++) {
      var y0 = -YH + 2.0 * YH * (k / (nz - 1));
      var y1 = -YH + 2.0 * YH * ((k + 1) / (nz - 1));
      for (var i = 0; i < nx - 1; i++) {
        var x0 = -XW + 2.0 * XW * (i / (nx - 1));
        var x1 = -XW + 2.0 * XW * ((i + 1) / (nx - 1));
        var c00 = oceanPalette((bAt(i, k) - bMin) / bSpan);
        var c10 = oceanPalette((bAt(i + 1, k) - bMin) / bSpan);
        var c11 = oceanPalette((bAt(i + 1, k + 1) - bMin) / bSpan);
        var c01 = oceanPalette((bAt(i, k + 1) - bMin) / bSpan);
        var nFront = [0, 0, -1];
        pushTri([x0, y0, zBack], [x1, y1, zBack], [x1, y0, zBack], nFront, nFront, nFront, c00, c11, c10);
        pushTri([x0, y0, zBack], [x0, y1, zBack], [x1, y1, zBack], nFront, nFront, nFront, c00, c01, c11);
      }
    }

    // 4. Three 3D Phong-Lit Volumetric Isopycnal Material Surfaces across the 3D Tank
    var isoLevels = [0.28, 0.50, 0.72];
    var nSpanSeg = 14;
    for (var il = 0; il < isoLevels.length; il++) {
      var targetT = isoLevels[il];
      var targetB = bMin + targetT * bSpan;
      var isoCol = oceanPalette(targetT);
      // Extract height profile hIso[i] across x where b(i, k) crosses targetB
      var hIso = new Float32Array(nx);
      for (var i2 = 0; i2 < nx; i2++) {
        var bestK = (nz - 1) * targetT;
        var bestDiff = 1e9;
        for (var k2 = 0; k2 < nz - 1; k2++) {
          var v0 = bAt(i2, k2) - targetB;
          var v1 = bAt(i2, k2 + 1) - targetB;
          if ((v0 <= 0.0 && v1 >= 0.0) || (v0 >= 0.0 && v1 <= 0.0)) {
            var frac = Math.abs(v0) / Math.max(1e-9, Math.abs(v0 - v1));
            bestK = k2 + frac;
            bestDiff = 0.0;
          } else if (Math.abs(v0) < bestDiff) {
            bestDiff = Math.abs(v0);
            bestK = k2;
          }
        }
        hIso[i2] = -YH + 2.0 * YH * (bestK / Math.max(1, nz - 1));
      }

      for (var s = 0; s < nSpanSeg; s++) {
        var zA = -ZD * 0.92 + (1.84 * ZD) * (s / nSpanSeg);
        var zB = -ZD * 0.92 + (1.84 * ZD) * ((s + 1) / nSpanSeg);
        var spanModA = (1.0 - prog * 0.75) * 0.14 * Math.sin((2.0 * Math.PI * s) / nSpanSeg + il);
        var spanModB = (1.0 - prog * 0.75) * 0.14 * Math.sin((2.0 * Math.PI * (s + 1)) / nSpanSeg + il);
        for (var i3 = 0; i3 < nx - 1; i3++) {
          var xa = -XW + 2.0 * XW * (i3 / (nx - 1));
          var xb = -XW + 2.0 * XW * ((i3 + 1) / (nx - 1));
          var ya0 = Math.max(-YH + 0.05, Math.min(YH - 0.05, hIso[i3] + spanModA));
          var yb0 = Math.max(-YH + 0.05, Math.min(YH - 0.05, hIso[i3 + 1] + spanModA));
          var yb1 = Math.max(-YH + 0.05, Math.min(YH - 0.05, hIso[i3 + 1] + spanModB));
          var ya1 = Math.max(-YH + 0.05, Math.min(YH - 0.05, hIso[i3] + spanModB));
          var dyx = yb0 - ya0;
          var dx = xb - xa;
          var nrm = Math.hypot(dx, dyx) || 1.0;
          var nSurf = [-dyx / nrm, dx / nrm, 0.08];
          pushTri([xa, ya0, zA], [xb, yb0, zA], [xb, yb1, zB], nSurf, nSurf, nSurf, isoCol);
          pushTri([xa, ya0, zA], [xb, yb1, zB], [xa, ya1, zB], nSurf, nSurf, nSurf, isoCol);
          if (s === 0) {
            pushLine([xa, ya0, zA - 0.01], [xb, yb0, zA - 0.01], [0.96, 0.98, 1.0, 0.92]);
          }
        }
      }
    }

    // 5. 3D Actuator Manifolds / Surface Wind-Stress & Cooling Flux Vanes
    if (opts.showCoils && mach.actuators) {
      var actGain = 0.25 + 0.75 * prog;
      for (var ai = 0; ai < mach.actuators.length; ai++) {
        var ac = mach.actuators[ai];
        var ax0 = mapX(ac.x - 0.5 * ac.wx), ax1 = mapX(ac.x + 0.5 * ac.wx);
        var ay0 = mapY(ac.z - 0.5 * ac.wz), ay1 = mapY(ac.z + 0.5 * ac.wz);
        var az0 = Math.max(-ZD * 0.88, mapZ(ac.y - 0.5 * ac.wy));
        var az1 = Math.min( ZD * 0.88, mapZ(ac.y + 0.5 * ac.wy));
        var cAct = (ai % 2 === 0)
          ? [0.72 + 0.16 * actGain, 0.48 + 0.10 * actGain, 0.14, 1.0]
          : [0.24 + 0.08 * actGain, 0.42 + 0.14 * actGain, 0.64 + 0.12 * actGain, 1.0];
        var cTop = [Math.min(1.0, cAct[0] + 0.12), Math.min(1.0, cAct[1] + 0.12), Math.min(1.0, cAct[2] + 0.10), 1.0];
        addBox3D(ax0, ax1, ay0, ay1, az0, az1, cAct, cTop, rimEdge);
      }
    }

    // 6. 3D Velocity Streamlines Advected through Oceananigans.jl (u, w) Field
    if (opts.showFieldLines) {
      var zPlanes = [-ZD * 0.65, 0.0, ZD * 0.65];
      for (var zp = 0; zp < zPlanes.length; zp++) {
        for (var seedRow = 0; seedRow < 5; seedRow++) {
          var sx = -XW * 0.90;
          var sy = -YH * 0.72 + (1.44 * YH) * (seedRow / 4.0);
          var dir = (sy >= 0.0 || mach.id === "ocean_wind") ? 1.0 : -1.0;
          if (dir < 0.0) sx = XW * 0.90;
          var prev = [sx, sy, zPlanes[zp]];
          for (var step = 0; step < 36; step++) {
            var gi = Math.max(0, Math.min(nx - 1, Math.round(((sx + XW) / (2.0 * XW)) * (nx - 1))));
            var gk = Math.max(0, Math.min(nz - 1, Math.round(((sy + YH) / (2.0 * YH)) * (nz - 1))));
            var uu = uField[gk * nx + gi];
            var ww = wField[gk * nx + gi];
            var uScale = (mach.id === "ocean_wind") ? -2.2 : 0.28;
            var nxPt = sx + (Math.abs(uu) > 1e-4 ? uu * uScale : dir * 0.24);
            var nyPt = Math.max(-YH * 0.92, Math.min(YH * 0.92, sy + ww * 0.36));
            if (nxPt < -XW * 0.95 || nxPt > XW * 0.95) break;
            var cur = [nxPt, nyPt, zPlanes[zp]];
            pushLine(prev, cur, [1.0, 0.90, 0.45, 0.92]);
            prev = cur;
            sx = nxPt;
            sy = nyPt;
          }
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

  // Build the complete 3D CAD mesh buffers for active FUSE.jl / TCV tokamak geometry
  function buildTokamakCADBuffers(gl, st, opts) {
    var mKey = opts.machine || "tcv";
    var mach = Object.prototype.hasOwnProperty.call(MACHINES, mKey) ? MACHINES[mKey] : MACHINES.tcv;
    if (mach.kind === "ocean") {
      return buildOceanCADBuffers(gl, mach, st, opts);
    }
    var pos = [], norm = [], col = [], linePos = [], lineCol = [];

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
        if (signNorm > 0) pushTri(pc, pa, pb, n, n, n, cc, ca, cb);
        else pushTri(pc, pb, pa, n, n, n, cc, cb, ca);
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
        var a0 = phi0 + s * dPhi, a1 = a0 + dPhi;
        var c0 = Math.cos(a0), s0 = Math.sin(a0);
        var c1 = Math.cos(a1), s1 = Math.sin(a1);
        for (var i = 0; i < m; i++) {
          var v0 = polyRZ[i], v1 = polyRZ[(i + 1) % m];
          var dr = v1[0] - v0[0], dz = v1[1] - v0[1];
          var len = Math.hypot(dr, dz) || 1.0;
          var nr = dz / len, nz = -dr / len;
          var n0 = [nr * c0, nz, nr * s0], n1 = [nr * c1, nz, nr * s1];
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
        var a0 = phi0 + s * dPhi, a1 = a0 + dPhi;
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
        for (var endIdx = 0; endIdx < 2; endIdx++) {
          var phi = (endIdx === 0) ? phi0 : phi1;
          var signN = (endIdx === 0) ? -1.0 : 1.0;
          var nSide = [-Math.sin(phi) * signN, 0.12, Math.cos(phi) * signN];
          for (var i = 0; i < m; i++) {
            var o0 = outerRZ[i], o1 = outerRZ[(i + 1) % m];
            var k0 = innerRZ[i], k1 = innerRZ[(i + 1) % m];
            var po0 = cylToXYZ(o0[0], o0[1], phi, 0.002 * signN);
            var po1 = cylToXYZ(o1[0], o1[1], phi, 0.002 * signN);
            var pk1 = cylToXYZ(k1[0], k1[1], phi, 0.002 * signN);
            var pk0 = cylToXYZ(k0[0], k0[1], phi, 0.002 * signN);
            pushTri(po0, po1, pk1, nSide, nSide, nSide, rimCol, rimCol, rimCol);
            pushTri(po0, pk1, pk0, nSide, nSide, nSide, rimCol, rimCol, rimCol);
          }
        }
      }
    }

    function addToroidalBoxRing(rIn, rOut, zBot, zTop, phi0, phi1, nSeg, sideCol, capCol) {
      var poly = [[rIn, zBot], [rOut, zBot], [rOut, zTop], [rIn, zTop]];
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

    // 0. Discrete 3D Superconducting / Copper Toroidal Field (TF) D-Coils from FUSE.jl dd.build.layer
    if (opts.showTF && mach.tf_outer && mach.tf_inner && mach.tf_outer.length >= 4 && mach.tf_outer.length === mach.tf_inner.length) {
      var nTF = mach.n_tf || 18;
      var halfAng = (Math.PI / nTF) * 0.34;
      var tfCaseCol = (mach.id === "tcv" || mach.id === "d3d")
        ? [0.64, 0.39, 0.20, 1.0]
        : [0.34, 0.43, 0.54, 1.0];
      var tfSideCol = (mach.id === "tcv" || mach.id === "d3d")
        ? [0.76, 0.49, 0.26, 1.0]
        : [0.46, 0.56, 0.68, 1.0];
      for (var ti = 0; ti < nTF; ti++) {
        var phiC = -0.5 * Math.PI + (2.0 * Math.PI * ti) / nTF;
        // Leave front cutaway sector open so vessel cutaway & protruding 3D plasma torus stay unobstructed
        if (phiC > (phiVessel0 + 8.0 * deg) && phiC < (phiVessel1 - 8.0 * deg)) {
          addHollowShell(mach.tf_outer, mach.tf_inner, phiC - halfAng, phiC + halfAng, 2, tfCaseCol, tfCaseCol, tfSideCol);
        }
      }
    }

    // 1. Poloidal Field & Central Solenoid Coils (colored by solved FUSE.jl ActorPFactive currents / policy state)
    if (opts.showCoils && mach.coils) {
      for (var ci = 0; ci < mach.coils.length; ci++) {
        var c = mach.coils[ci];
        var rIn = Math.max(0.05, c.r - c.w * 0.5);
        var rOut = c.r + c.w * 0.5;
        var zBot = c.z - c.h * 0.5;
        var zTop = c.z + c.h * 0.5;
        var isInner = (c.r < mach.R0);
        var isCS = c.name && (c.name.indexOf("cs") !== -1 || c.name.indexOf("CS") !== -1 || c.name.indexOf("OH") !== -1 || c.name.indexOf("Central") !== -1);
        var baseCur = (typeof c.current_ma === "number" && Math.abs(c.current_ma) > 1e-4)
          ? Math.min(1.0, Math.abs(c.current_ma) / Math.max(1.0, mach.ipNominal * 0.65))
          : 0.42;
        var act = (mach.id === "tcv" && ci < st.coils.length) ? Math.abs(st.coils[ci]) : lerp(0.18, baseCur, st.progress);
        var cBody, cCap;
        if (isCS) {
          cBody = [0.42 + 0.08 * act, 0.27 + 0.05 * act, 0.48 + 0.09 * act, 1.0];
          cCap  = [0.56 + 0.09 * act, 0.37 + 0.06 * act, 0.63 + 0.09 * act, 1.0];
        } else if (isInner) {
          cBody = [0.24 + 0.08 * act, 0.37 + 0.11 * act, 0.54 + 0.14 * act, 1.0];
          cCap  = [0.32 + 0.09 * act, 0.47 + 0.12 * act, 0.66 + 0.14 * act, 1.0];
        } else {
          cBody = [0.68 + 0.12 * act, 0.47 + 0.10 * act, 0.14 + 0.04 * act, 1.0];
          cCap  = [0.82 + 0.12 * act, 0.58 + 0.10 * act, 0.20 + 0.04 * act, 1.0];
        }
        addToroidalBoxRing(rIn, rOut, zBot, zTop, isInner ? phiInner0 : phiOuter0, isInner ? phiInner1 : phiOuter1, 44, cBody, cCap);
      }
    }

    // 2. Hollow 3D Double-Wall Vacuum Vessel Shell
    var vesselShellCol = [0.60, 0.62, 0.65, 1.0];
    var vesselInnerCol = [0.15, 0.16, 0.17, 1.0];
    var whiteRimCol    = [0.97, 0.98, 1.00, 1.0];
    addHollowShell(mach.vessel_outer, mach.vessel_inner, phiVessel0, phiVessel1, 60, vesselShellCol, vesselInnerCol, whiteRimCol);

    // 3. Cutaway Faces & Protruding Solved TEQUILA / Miller Plasma Torus
    var eq = plasmaContour(1.0, st, 64, mach);

    function buildCutawayFace(phiVesselPlane, signN, isLeftProtruding) {
      var phiPlasmaPlane = isLeftProtruding ? (phiVesselPlane - 13.5 * deg) : phiVesselPlane;
      var dnScale = Math.max(1.0, mach.R0 * 0.55);

      addCutawayFan(
        mach.vessel_outer, [mach.R0, mach.Z0 || 0.0], phiVesselPlane, signN, 0.001 * dnScale * signN,
        whiteRimCol, [0.16, 0.17, 0.19, 0.85]
      );
      addCutawayFan(
        mach.first_wall, [mach.R0, mach.Z0 || 0.0], phiVesselPlane, signN, 0.004 * dnScale * signN,
        [0.15, 0.16, 0.17, 1.0], [0.93, 0.95, 0.97, 0.95]
      );

      if (isLeftProtruding) {
        addRevolvedSolid(
          eq.pts, phiPlasmaPlane, phiVesselPlane, 18,
          [0.95, 0.52, 0.09, 1.0], null, null, [eq.R0, eq.Z0]
        );
      }

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
          cData.pts, [cData.R0, cData.Z0], phiPlasmaPlane, signN, (0.008 + 0.0018 * b) * dnScale * signN,
          band.col, band.edge
        );
      }

      var pX = cylToXYZ(eq.Rx, eq.Zx, phiVesselPlane, 0.0085 * dnScale * signN);
      var pLegIn = cylToXYZ(eq.Rx - mach.a * 0.18, eq.Zx - mach.a * 0.28, phiVesselPlane, 0.0085 * dnScale * signN);
      var pLegOut = cylToXYZ(eq.Rx + mach.a * 0.28, eq.Zx - mach.a * 0.26, phiVesselPlane, 0.0085 * dnScale * signN);
      var cyanLine = [0.30, 0.88, 0.94, 1.0];
      pushLine(pX, pLegIn, cyanLine);
      pushLine(pX, pLegOut, cyanLine);
    }

    buildCutawayFace(phiVessel0, -1.0, true);
    buildCutawayFace(phiVessel1,  1.0, false);

    // 4. Helical q95 Magnetic Field Lines on LCFS
    if (opts.showFieldLines) {
      var qEff = (mach.q95 || 3.2) * (0.72 + 0.28 * st.progress);
      var nLines = 3;
      for (var l = 0; l < nLines; l++) {
        var th0 = (2.0 * Math.PI * l) / nLines;
        var prevPt = null;
        var steps = 120;
        var pStart = phiVessel0;
        var pEnd = phiVessel1 - 11.0 * deg;
        for (var k = 0; k <= steps; k++) {
          var phi = pEnd + (pStart - pEnd) * (k / steps);
          var th = th0 + qEff * (phi - pEnd);
          var idx = ((Math.floor((th / (2.0 * Math.PI)) * eq.pts.length) % eq.pts.length) + eq.pts.length) % eq.pts.length;
          var rz = eq.pts[idx];
          var pt = cylToXYZ(rz[0] * 1.003, rz[1], phi, 0.0);
          if (prevPt) pushLine(prevPt, pt, [1.0, 0.92, 0.55, 0.92]);
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

  // Render 2D Scientific Cross-Section for Oceananigans.jl 3D Fluid Environments
  function renderOcean2DCanvas(ctx, w, h, st, mach, isDark) {
    var compact = w < 420;
    var padTop = compact ? 24 : 30;
    var padBot = compact ? 30 : 38;
    var padLeft = compact ? 36 : 44;
    var padRight = compact ? 10 : 14;
    var gap = compact ? 12 : 18;
    var boxW = Math.max(60, Math.floor((w - padLeft - padRight - gap) * 0.5));
    var boxH = Math.min(Math.max(80, h - padTop - padBot), Math.round(boxW * 1.08));
    var startY = padTop + Math.floor((h - padTop - padBot - boxH) * 0.5);
    var leftX = padLeft;
    var rightX = leftX + boxW + gap;

    var nx = mach.nx || 32, nz = mach.nz || 32;
    var prog = st.progress;
    var bGrid = new Float32Array(nx * nz);
    var uGrid = new Float32Array(nx * nz);
    var bMin = 1e9, bMax = -1e9, uMax = 1e-6;
    for (var m = 0; m < nx * nz; m++) {
      var bv = lerp(mach.b_init[m] || 0.0, mach.b_opt[m] || 0.0, prog);
      var uv = lerp(mach.u_init[m] || 0.0, mach.u_opt[m] || 0.0, prog);
      bGrid[m] = bv;
      uGrid[m] = uv;
      if (bv < bMin) bMin = bv;
      if (bv > bMax) bMax = bv;
      if (Math.abs(uv) > uMax) uMax = Math.abs(uv);
    }
    var bSpan = Math.max(1e-6, bMax - bMin);

    var titleFont = compact ? "700 11px 'Latin Modern Roman', Georgia, serif" : "700 12.5px 'Latin Modern Roman', Georgia, serif";
    var tickFont = "400 9px 'Latin Modern Mono', monospace";
    var axisFont = "italic 10.5px 'Latin Modern Roman', Georgia, serif";

    function drawFieldBox(bx, titleStr, isVel, showZ) {
      ctx.fillStyle = isDark ? "#e6edf3" : "#111111";
      ctx.font = titleFont;
      ctx.textAlign = "center";
      ctx.fillText(titleStr, bx + boxW * 0.5, startY - 8);

      var cw = boxW / nx, ch = boxH / nz;
      for (var k = 0; k < nz; k++) {
        var py = startY + boxH - (k + 1) * ch;
        for (var i = 0; i < nx; i++) {
          var px = bx + i * cw;
          var idx = k * nx + i;
          if (!isVel) {
            var c = oceanPalette((bGrid[idx] - bMin) / bSpan);
            ctx.fillStyle = "rgb(" + Math.round(c[0]*255) + "," + Math.round(c[1]*255) + "," + Math.round(c[2]*255) + ")";
          } else {
            var un = Math.max(-1.0, Math.min(1.0, uGrid[idx] / uMax));
            var rC = un >= 0 ? 255 : Math.round(255 * (1 + un * 0.82));
            var gC = Math.round(255 * (1 - 0.65 * Math.abs(un)));
            var bC = un <= 0 ? 255 : Math.round(255 * (1 - un * 0.82));
            ctx.fillStyle = "rgb(" + rC + "," + gC + "," + bC + ")";
          }
          ctx.fillRect(px, py, Math.ceil(cw + 0.6), Math.ceil(ch + 0.6));
        }
      }

      // Overlay CAD actuators on left panel
      if (!isVel && mach.actuators) {
        ctx.fillStyle = "rgba(217, 119, 6, 0.88)";
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 0.8;
        for (var a = 0; a < mach.actuators.length; a++) {
          var ac = mach.actuators[a];
          var rx0 = bx + ((ac.x - 0.5 * ac.wx - mach.xMin) / (mach.xMax - mach.xMin)) * boxW;
          var rw  = Math.max(3, (ac.wx / (mach.xMax - mach.xMin)) * boxW);
          var ry0 = startY + ((mach.zMax - (ac.z + 0.5 * ac.wz)) / (mach.zMax - mach.zMin)) * boxH;
          var rh  = Math.max(3, (ac.wz / (mach.zMax - mach.zMin)) * boxH);
          ctx.fillRect(rx0, ry0, rw, rh);
          ctx.strokeRect(rx0, ry0, rw, rh);
        }
      }

      ctx.strokeStyle = isDark ? "#8b949e" : "#111111";
      ctx.lineWidth = 1.1;
      ctx.strokeRect(bx, startY, boxW, boxH);

      ctx.fillStyle = isDark ? "#8b949e" : "#222222";
      ctx.font = tickFont;
      ctx.textAlign = "center";
      var xm = 0.5 * (mach.xMin + mach.xMax);
      var xTicks = [mach.xMin, xm, mach.xMax];
      for (var ti = 0; ti < 3; ti++) {
        var tx = bx + (ti * 0.5) * boxW;
        ctx.fillText(xTicks[ti].toFixed(0), tx, startY + boxH + 12);
      }
      ctx.font = axisFont;
      ctx.fillText(mach.id === "ocean_wind" ? "x [m]" : "x / h", bx + boxW * 0.5, startY + boxH + 26);

      if (showZ) {
        ctx.font = tickFont;
        ctx.textAlign = "right";
        var zm = 0.5 * (mach.zMin + mach.zMax);
        var zTicks = [mach.zMin, zm, mach.zMax];
        for (var zj = 0; zj < 3; zj++) {
          var ty = startY + boxH - (zj * 0.5) * boxH;
          ctx.fillText(zTicks[zj].toFixed(0), bx - 4, ty + 3);
        }
        ctx.save();
        ctx.translate(bx - (compact ? 26 : 31), startY + boxH * 0.5);
        ctx.rotate(-Math.PI * 0.5);
        ctx.font = axisFont;
        ctx.textAlign = "center";
        ctx.fillText(mach.id === "ocean_wind" ? "z [m]" : "z / h", 0, 0);
        ctx.restore();
      }
    }

    drawFieldBox(leftX, compact ? "Buoyancy b(x,z)" : "CAD & Buoyancy b(x, z)", false, true);
    drawFieldBox(rightX, compact ? "Velocity u(x,z)" : "Velocity Shear u(x, z)", true, false);
  }

  // Right Panel: 2D Physical Cross-Section (FUSE.jl Tokamak Equilibrium OR Oceananigans.jl PDE Slice)
  function renderFigure9Canvas(canvas, st, mach, opts) {
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

    if (mach.kind === "ocean") {
      renderOcean2DCanvas(ctx, w, h, st, mach, isDark);
      ctx.restore();
      return;
    }

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
    var dR = Math.max(0.2, rMax - rMin), dZ = Math.max(0.2, zMax - zMin);
    var geomAspect = dZ / dR;
    var boxH = Math.min(availH, availW * geomAspect);
    var boxW = boxH / geomAspect;
    var totalW = 2 * boxW + gap;
    var startX = padLeft + Math.floor((w - padLeft - padRight - totalW) * 0.5);
    var startY = padTop + Math.floor((availH - boxH) * 0.5);

    function rzToXY(boxX, r, z) {
      return [
        boxX + ((r - rMin) / dR) * boxW,
        startY + ((zMax - z) / dZ) * boxH
      ];
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

    var rTicks = mach.rTicks;
    if (!rTicks || !rTicks.length) {
      rTicks = [rMin + 0.22 * dR, rMin + 0.52 * dR, rMin + 0.82 * dR];
    }
    var zTicks = mach.zTicks;
    if (!zTicks || !zTicks.length) {
      zTicks = [Math.round(zMin * 0.65 * 10) / 10, 0.0, Math.round(zMax * 0.65 * 10) / 10];
    }

    function drawBoxTicks(boxX, showZLabels) {
      ctx.save();
      ctx.strokeStyle = isDark ? "#8b949e" : "#111111";
      ctx.fillStyle = isDark ? "#8b949e" : "#222222";
      ctx.font = tickFont;
      ctx.lineWidth = 0.9;
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
      if (opts && opts.showTF && mach.tf_outer && mach.tf_outer.length >= 4) {
        ctx.save();
        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        for (var t0 = 0; t0 < mach.tf_outer.length; t0++) {
          var p0 = rzToXY(boxX, mach.tf_outer[t0][0], mach.tf_outer[t0][1]);
          if (t0 === 0) ctx.moveTo(p0[0], p0[1]); else ctx.lineTo(p0[0], p0[1]);
        }
        ctx.closePath();
        ctx.stroke();
        if (mach.tf_inner && mach.tf_inner.length >= 4) {
          ctx.beginPath();
          for (var t1 = 0; t1 < mach.tf_inner.length; t1++) {
            var p1 = rzToXY(boxX, mach.tf_inner[t1][0], mach.tf_inner[t1][1]);
            if (t1 === 0) ctx.moveTo(p1[0], p1[1]); else ctx.lineTo(p1[0], p1[1]);
          }
          ctx.closePath();
          ctx.stroke();
        }
        ctx.restore();
      }
      if (!mach.coils || (opts && !opts.showCoils)) return;
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
      ctx.strokeStyle = "#64748b";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (var i = 0; i < mach.vessel_outer.length; i++) {
        var p = rzToXY(boxX, mach.vessel_outer[i][0], mach.vessel_outer[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();

      ctx.strokeStyle = "#00008b";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 2]);
      ctx.beginPath();
      for (var j = 0; j < mach.first_wall.length; j++) {
        var pf = rzToXY(boxX, mach.first_wall[j][0], mach.first_wall[j][1]);
        if (j === 0) ctx.moveTo(pf[0], pf[1]); else ctx.lineTo(pf[0], pf[1]);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    var leftX = startX;
    ctx.fillStyle = isDark ? "#e6edf3" : "#111111";
    ctx.font = titleFont;
    ctx.textAlign = "center";
    ctx.fillText(mach.lcfs ? "FUSE CAD & LCFS" : "CAD & Target", leftX + boxW * 0.5, startY - (compact ? 7 : 9));

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

    var refPoly = (mach.id === "tcv") ? SHAPE_70166_REFS : mach.lcfs;
    if (refPoly && refPoly.length >= 8) {
      ctx.strokeStyle = "#0022ff";
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      for (var rIdx = 0; rIdx < refPoly.length; rIdx++) {
        var pr = rzToXY(leftX, refPoly[rIdx][0], refPoly[rIdx][1]);
        if (rIdx === 0) ctx.moveTo(pr[0], pr[1]); else ctx.lineTo(pr[0], pr[1]);
      }
      ctx.closePath();
      ctx.stroke();
      if (mach.id === "tcv") {
        ctx.fillStyle = "#0022ff";
        for (var dIdx = 0; dIdx < refPoly.length; dIdx++) {
          var pd = rzToXY(leftX, refPoly[dIdx][0], refPoly[dIdx][1]);
          ctx.beginPath();
          ctx.arc(pd[0], pd[1], 2.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    var eqTarget = plasmaContour(1.0, st, 64, mach);
    ctx.save();
    ctx.strokeStyle = st.policy.color;
    ctx.lineWidth = 1.65;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    for (var ei = 0; ei < eqTarget.pts.length; ei++) {
      var pe = rzToXY(leftX, eqTarget.pts[ei][0], eqTarget.pts[ei][1]);
      if (ei === 0) ctx.moveTo(pe[0], pe[1]); else ctx.lineTo(pe[0], pe[1]);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
    ctx.restore();

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
      for (var fi = 0; fi < fData.pts.length; fi++) {
        var pfl = rzToXY(rightX, fData.pts[fi][0], fData.pts[fi][1]);
        if (fi === 0) ctx.moveTo(pfl[0], pfl[1]); else ctx.lineTo(pfl[0], pfl[1]);
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

  function mountTokamakCADViewer(root) {
    if (!root) return;
    if (root.getAttribute("data-tcv-mounted") === "1") return;
    root.setAttribute("data-tcv-mounted", "1");

    root.innerHTML = [
      '<div class="tcv-cad-outer">',
      '  <aside class="tcv-side-controls" aria-label="Simulation Controls">',
      '    <div class="tcv-hud-badge tcv-ctrl-title" style="padding:5px 12px;font-size:12px;"><strong>Simulation Controls</strong></div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-device">',
      '      <div class="tcv-ctrl-hdr"><span>Environment</span><span id="tcv-builder-tag">FUSE.jl</span></div>',
      '      <select id="tcv-machine-sel" class="tcv-select">',
      '        <optgroup label="Tokamaks · FUSE.jl">',
      '          <option value="tcv" selected>TCV · EPFL</option>',
      '          <option value="sparc">SPARC · CFS (FUSE.jl)</option>',
      '          <option value="arc">ARC · MIT/CFS (FUSE.jl)</option>',
      '          <option value="iter">ITER · Cadarache (FUSE.jl)</option>',
      '          <option value="d3d">DIII-D · GA (FUSE.jl)</option>',
      '        </optgroup>',
      '        <optgroup label="Stratified Fluids · Oceananigans.jl">',
      '          <option value="ocean_kh">KH Shear Tank · 3D</option>',
      '          <option value="ocean_wind">Wind-Mixing Basin · 3D</option>',
      '        </optgroup>',
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
      '      <div class="tcv-ctrl-hdr"><span>CAD Layers</span></div>',
      '      <div class="tcv-layers-row">',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-coils" checked /> <span id="tcv-lbl-coils">PF/CS Coils</span></label>',
      '        <label class="tcv-chk-row" id="tcv-row-tf"><input type="checkbox" id="tcv-chk-tf" checked /> <span>TF Magnets</span></label>',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-field" checked /> <span id="tcv-lbl-field">B-Field / Flow</span></label>',
      '        <button type="button" id="tcv-reset-cam" class="tcv-pol-btn" style="justify-content:center;margin-top:3px;">Reset View</button>',
      '      </div>',
      '    </div>',
      '  </aside>',
      '  <div class="tcv-cad-shell">',
      '    <div class="tcv-grid">',
      '      <div class="tcv-pane">',
      '        <div class="tcv-hud-badge"><span id="tcv-badge-a"><strong>(a)</strong> 3D Cutaway CAD</span><span class="tcv-touch-hint">Drag · Pinch</span></div>',
      '        <div class="tcv-canvas-wrap" id="tcv-webgl-wrap">',
      '          <canvas id="tcv-webgl-canvas"></canvas>',
      '        </div>',
      '      </div>',
      '      <div class="tcv-pane">',
      '        <div class="tcv-hud-badge"><span id="tcv-badge-b"><strong>(b)</strong> Equilibrium Cross-Section</span></div>',
      '        <div class="tcv-canvas-wrap" id="tcv-fig9-wrap">',
      '          <canvas id="tcv-fig9-canvas"></canvas>',
      '        </div>',
      '      </div>',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span id="tcv-subtable-title">FUSE.jl 1.2.1 Environment</span><span id="tcv-subtable-device">TCV · EPFL</span></div>',
      '        <div class="tcv-kv-list" id="tcv-kv-list"></div>',
      '      </div>',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span id="tcv-obj-hdr">Multi-Component Objective (TuRBO-0)</span><span>Score</span></div>',
      '        <div class="tcv-wpo-stack" id="tcv-wpo-strip"></div>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-fig-caption" id="tcv-fig-caption">',
      '      <strong>Figure 1.</strong> 3D cutaway geometries, 2D equilibrium and field cross-sections, and closed-loop rollout trajectories across tokamak models exported from <code>FUSE.jl</code> (TCV, SPARC, ARC, ITER, DIII-D; Meneghini et al., 2024) and stratified fluid models from <code>Oceananigans.jl</code> (Ramadhan et al., 2020).',
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
    var chkTF = root.querySelector("#tcv-chk-tf");
    var chkField = root.querySelector("#tcv-chk-field");
    var rowTF = root.querySelector("#tcv-row-tf");
    var lblCoils = root.querySelector("#tcv-lbl-coils");
    var lblField = root.querySelector("#tcv-lbl-field");
    var resetCamBtn = root.querySelector("#tcv-reset-cam");
    var badgeA = root.querySelector("#tcv-badge-a");
    var badgeB = root.querySelector("#tcv-badge-b");
    var builderTag = root.querySelector("#tcv-builder-tag");
    var subtableTitle = root.querySelector("#tcv-subtable-title");
    var subtableDevice = root.querySelector("#tcv-subtable-device");
    var objHdr = root.querySelector("#tcv-obj-hdr");

    var initialMach = "tcv";
    try {
      var qParams = new URLSearchParams(window.location.search);
      var qm = qParams.get("machine");
      if (qm && Object.prototype.hasOwnProperty.call(MACHINES, qm)) {
        initialMach = qm;
      }
    } catch (e) {}
    if (machineSel) machineSel.value = initialMach;

    var initObj = Object.prototype.hasOwnProperty.call(MACHINES, initialMach) ? MACHINES[initialMach] : MACHINES.tcv;
    var uiState = {
      machine: initialMach,
      policyKey: "shared",
      round: 10000,
      seedIdx: -1,
      showCoils: true,
      showTF: true,
      showFieldLines: true,
      cutawayDeg: 56.0,
      yaw: initObj.kind === "ocean" ? -1.18 : -Math.PI * 0.5,
      pitch: initObj.kind === "ocean" ? 0.36 : 0.46,
      dist: initObj.dist,
      playing: false
    };

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
      var isOcean = (mach.kind === "ocean");
      var st = evalState(uiState.policyKey, uiState.round, uiState.seedIdx);

      if (builderTag) builderTag.textContent = isOcean ? "Oceananigans.jl" : "FUSE.jl";
      if (lblCoils) lblCoils.textContent = isOcean ? "3D Actuators" : "PF/CS Coils";
      if (lblField) lblField.textContent = isOcean ? "Streamlines" : "B-Field Lines";
      if (rowTF) rowTF.style.display = isOcean ? "none" : "inline-flex";

      if (badgeA) badgeA.innerHTML = "<strong>(a)</strong> 3D Cutaway CAD — " + escHtml(mach.shortName);
      if (badgeB) {
        badgeB.innerHTML = isOcean
          ? ("<strong>(b)</strong> PDE Cross-Section — " + escHtml(mach.shortName))
          : ("<strong>(b)</strong> Equilibrium & Flux — " + escHtml(mach.shortName));
      }
      if (subtableTitle) subtableTitle.textContent = isOcean ? "Oceananigans.jl 0.117.2" : "FUSE.jl 1.2.1 Geometry";
      if (subtableDevice) subtableDevice.textContent = mach.shortName;
      if (objHdr) {
        objHdr.textContent = isOcean
          ? "BAUX Regional Control Decomposition"
          : "TORAX / FUSE SmoothMax (α = −3)";
      }

      if (kvList) {
        if (isOcean) {
          var scCur = lerp(mach.score_init, mach.score_opt, st.progress);
          var suppPct = (100.0 * (1.0 - Math.abs(scCur) / Math.max(1e-6, Math.abs(mach.score_init)))).toFixed(1) + "%";
          var gridStr = mach.nx + "×" + mach.ny + "×" + mach.nz + " (D=" + mach.dim + ")";
          kvList.innerHTML = [
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Normalized PDE Objective</span><span class="tcv-kv-val" style="color:' + escHtml(st.policy.color) + ';">' + escHtml(scCur.toFixed(3)) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Perturbation Suppression</span><span class="tcv-kv-val">' + escHtml(suppPct) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">3D Grid & Control Dim</span><span class="tcv-kv-val">' + escHtml(gridStr) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Richardson / Reynolds</span><span class="tcv-kv-val">Ri=' + escHtml(mach.ri.toFixed(2)) + ', Re=' + escHtml(mach.re.toFixed(0)) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Domain Span</span><span class="tcv-kv-val">[' + escHtml(mach.xMin.toFixed(0)) + ',' + escHtml(mach.xMax.toFixed(0)) + ']×[' + escHtml(mach.zMin.toFixed(0)) + ',' + escHtml(mach.zMax.toFixed(0)) + ']</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Solver Step Time</span><span class="tcv-kv-val">' + escHtml(LIVE_RUNS.oceanWorkerS.toFixed(2)) + ' s</span></div>'
          ].join("");
        } else {
          var jStr = st.score.toFixed(2) + (uiState.seedIdx < 0 ? (" ± " + st.sem.toFixed(2)) : (" (S" + (uiState.seedIdx + 1) + ")"));
          var qVal = ((mach.q95 || 3.2) * (0.72 + 0.28 * st.progress)).toFixed(2);
          var nCoils = (mach.coils ? mach.coils.length : 0) + " PF + " + (mach.n_tf || 18) + " TF";
          var extraLbl = (mach.pFusMW && mach.pFusMW > 1.0) ? "Fusion Power <em>P</em><sub>fus</sub>" : "LCFS Contour Δ";
          var extraVal = (mach.pFusMW && mach.pFusMW > 1.0)
            ? ((mach.pFusMW * (0.25 + 0.75 * st.progress)).toFixed(1) + " MW")
            : (st.lcfsErrMm.toFixed(2) + " mm");
          kvList.innerHTML = [
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Return <em>J</em>(θ)</span><span class="tcv-kv-val" style="color:' + escHtml(st.policy.color) + ';">' + escHtml(jStr) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Plasma Current / <em>B</em><sub>0</sub></span><span class="tcv-kv-val">' + escHtml(mach.ipNominal.toFixed(2)) + ' MA · ' + escHtml(mach.B0.toFixed(2)) + ' T</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Safety Factor <em>q</em><sub>95</sub> / β<sub><em>N</em></sub></span><span class="tcv-kv-val">' + escHtml(qVal) + ' · ' + escHtml((mach.betaN || 1.8).toFixed(2)) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Radii <em>R</em><sub>0</sub>, <em>a</em></span><span class="tcv-kv-val">' + escHtml(mach.R0.toFixed(2)) + ' m, ' + escHtml(mach.a.toFixed(2)) + ' m</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Magnet CAD Coils</span><span class="tcv-kv-val">' + escHtml(nCoils) + '</span></div>',
            '<div class="tcv-kv-row"><span class="tcv-kv-lbl">' + extraLbl + '</span><span class="tcv-kv-val">' + escHtml(extraVal) + '</span></div>'
          ].join("");
        }
      }

      if (wpoStrip) {
        var cLabels = isOcean
          ? ["Billow / Misfit Energy", "Shear Stabilization", "Isopycnal Smoothness", "Control Effort L₂", "CFL Stability Margin"]
          : ["LCFS / TEQUILA Shape", "Plasma Current <em>I<sub>p</sub></em>", "PF Coil Margin", "Actuators ∈ [−1, 1]", "Safety Factor <em>q</em><sub>95</sub>"];
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
        var farPlane = Math.max(80.0, effDist * 3.8);
        var projMat = mat4Perspective(32.0 * Math.PI / 180.0, aspect, 0.2, farPlane);
        var viewMat = mat4LookAt(eye, [0.0, isOcean ? -0.10 : (mach.Z0 || -0.04) * 0.2, 0.0], [0.0, 1.0, 0.0]);
        var mvp = mat4Mul(projMat, viewMat);
        var model = mat4Identity();

        gl.useProgram(prog);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uModel"), false, model);
        gl.uniform3fv(gl.getUniformLocation(prog, "uCamPos"), new Float32Array(eye));

        var aPos = gl.getAttribLocation(prog, "aPos");
        var aNorm = gl.getAttribLocation(prog, "aNorm");
        var aCol = gl.getAttribLocation(prog, "aCol");

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

      renderFigure9Canvas(fig9Canvas, st, mach, uiState);
    }

    // Hydrate real FUSE.jl & Oceananigans.jl 3D CAD environments from assets/baux_env.json
    if (typeof window.fetch === "function") {
      window.fetch("assets/baux_env.json")
        .then(function(r) { return r.ok ? r.json() : null; })
        .then(function(data) {
          if (!data || typeof data !== "object") return;
          if (data.fuse && data.fuse.machines) {
            var fm = data.fuse.machines;
            var fKeys = ["sparc", "arc", "iter", "d3d"];
            for (var fk = 0; fk < fKeys.length; fk++) {
              var k = fKeys[fk];
              if (Object.prototype.hasOwnProperty.call(fm, k) && fm[k] && fm[k].vessel_outer && fm[k].vessel_outer.length >= 6) {
                MACHINES[k] = fm[k];
              }
            }
          }
          if (data.ocean && data.ocean.environments) {
            var oe = data.ocean.environments;
            var oKeys = ["ocean_kh", "ocean_wind"];
            for (var ok = 0; ok < oKeys.length; ok++) {
              var okName = oKeys[ok];
              if (Object.prototype.hasOwnProperty.call(oe, okName) && oe[okName] && oe[okName].b_init) {
                MACHINES[okName] = oe[okName];
              }
            }
          }
          if (Array.isArray(data.runs)) {
            for (var ri = 0; ri < data.runs.length; ri++) {
              var rec = data.runs[ri];
              if (rec && rec.metrics && typeof rec.metrics.worker_s === "number") {
                if (rec.task === "oceananigans" && rec.metrics.worker_s > 0.01) {
                  LIVE_RUNS.oceanWorkerS = rec.metrics.worker_s;
                } else if (rec.task === "fuse" && rec.metrics.worker_s > 0.01) {
                  LIVE_RUNS.fuseWorkerS = rec.metrics.worker_s;
                }
              }
            }
          }
          var activeM = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
          uiState.dist = activeM.dist;
          meshDirty = true;
          renderAll();
        })
        .catch(function() {});
    }

    machineSel.addEventListener("change", function() {
      var val = machineSel.value;
      uiState.machine = Object.prototype.hasOwnProperty.call(MACHINES, val) ? val : "tcv";
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      uiState.dist = m.dist;
      uiState.yaw = (m.kind === "ocean") ? -1.18 : -Math.PI * 0.5;
      uiState.pitch = (m.kind === "ocean") ? 0.36 : 0.46;
      meshDirty = true;
      renderAll();
    });

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

    if (chkTF) {
      chkTF.addEventListener("change", function() {
        uiState.showTF = chkTF.checked;
        meshDirty = true;
        renderAll();
      });
    }

    chkField.addEventListener("change", function() {
      uiState.showFieldLines = chkField.checked;
      meshDirty = true;
      renderAll();
    });

    resetCamBtn.addEventListener("click", function() {
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      uiState.yaw = (m.kind === "ocean") ? -1.18 : -Math.PI * 0.5;
      uiState.pitch = (m.kind === "ocean") ? 0.36 : 0.46;
      uiState.dist = m.dist;
      renderAll();
    });

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
      uiState.dist = Math.max(m.dist * 0.35, Math.min(m.dist * 2.8, uiState.dist + e.deltaY * 0.0025 * zoomScale));
      renderAll();
    }, { passive: false });

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
        uiState.yaw += (cx - tLastX) * 0.010;
        uiState.pitch = Math.max(-1.1, Math.min(1.25, uiState.pitch + (cy - tLastY) * 0.010));
        tLastX = cx;
        tLastY = cy;
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
            uiState.yaw = (m.kind === "ocean") ? -1.18 : -Math.PI * 0.5;
            uiState.pitch = (m.kind === "ocean") ? 0.36 : 0.46;
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
      init_api_reference ();
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
