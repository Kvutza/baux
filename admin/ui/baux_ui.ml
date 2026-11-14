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

(* --- Central 3D WebGL CAD Simulation (FUSE.jl Tokamaks) --- *)
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

  var MACHINES = {
    tcv: TCV_CAD,
    sparc: SPARC_CAD,
    arc: ARC_CAD,
    iter: ITER_CAD,
    d3d: D3D_CAD
  };

  var LIVE_RUNS = {
    fuseWorkerS: 5.83
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
  (function() {
    var n0 = SHAPE_70166_REFS.length;
    var up = [];
    for (var i = 0; i < n0; i++) {
      var p0 = SHAPE_70166_REFS[(i - 1 + n0) % n0];
      var p1 = SHAPE_70166_REFS[i];
      var p2 = SHAPE_70166_REFS[(i + 1) % n0];
      var p3 = SHAPE_70166_REFS[(i + 2) % n0];
      up.push([p1[0], p1[1]]);
      var mr = 0.5 * ((p1[0] + p2[0]) + 0.125 * ((p1[0] - p0[0]) + (p2[0] - p3[0])));
      var mz = 0.5 * ((p1[1] + p2[1]) + 0.125 * ((p1[1] - p0[1]) + (p2[1] - p3[1])));
      up.push([mr, mz]);
    }
    TCV_CAD.lcfs = up;
    TCV_CAD.axisR = SHAPE_70166_AXIS[0];
    TCV_CAD.axisZ = SHAPE_70166_AXIS[1];
    TCV_CAD.rx = 0.7685;
    TCV_CAD.zx = -0.0641;
  })();

  // Verified 10-Seed x 10,000-Round Paired TCV Rampdown Benchmark Data (toka_gym, D = 136,712)
  // Automatically upgraded to all 47 empirical knots when figure[data-bench-fig="tokamak-tokagym"] docks.
  var ROUNDS = [1, 2, 4, 8, 15, 36, 100, 250, 500, 1000, 3000, 6381, 10000];
  var INIT_ERR = 58.121;
  var INIT_COILS = [-0.14, 0.11, -0.22, 0.26, -0.08, 0.17, -0.25, 0.13, -0.21, 0.12, 0.24, -0.18, 0.09, -0.16, 0.19, -0.11, -0.84, 0.46, -0.71, -0.34];
  var TARGET_COILS = [0.64, 0.58, 0.44, 0.21, -0.22, -0.48, -0.72, -0.54, 0.51, 0.66, 0.36, -0.18, -0.42, -0.64, 0.74, 0.56, -0.46, -0.44, 0.22, 0.28];
  // Dynamic 400 ms rampdown coil schedules: OH swings +I -> -I for inductive dIp/dt < 0; E/F follow descending circularizing plasma
  var COILS_FLATTOP  = [ 0.64,  0.60,  0.48,  0.44, -0.18, -0.46, -0.68, -0.52,  0.48,  0.68,  0.42, -0.14, -0.38, -0.60,  0.72,  0.54, -0.44, -0.64, -0.36,  0.24];
  var COILS_TERMINAL = [-0.72, -0.68, -0.56, -0.52, -0.08, -0.26,  0.30,  0.48,  0.46,  0.12, -0.06, -0.04, -0.12, -0.24,  0.28, -0.58, -0.56, -0.14, -0.06, -0.04];
  var COILS_UNSTABLE = [-0.42,  0.54, -0.62,  0.48,  0.66,  0.78,  0.52, -0.44, -0.74, -0.82, -0.58,  0.36,  0.72,  0.68, -0.54, -0.62,  0.64,  0.76,  0.48, -0.42];
  var BENCH_ORDER = ["turbo0", "enn_w", "enn_e", "enn_h", "enn_c"];
  var BENCH = {
    turbo0: {
      name: "TuRBO-0 (R=10)",
      short: "TuRBO-0 (R=10)",
      color: "#9a3412",
      dot: "#c2410c",
      mean: [58.121, 20.330, 7.649, 6.078, 3.637, 2.666, 1.847, 1.419, 1.382, 1.382, 1.364, 1.332, 1.311],
      sem:  [31.392,  6.575, 2.305, 1.962, 0.681, 0.303, 0.109, 0.031, 0.036, 0.036, 0.029, 0.022, 0.016],
      seeds10k: [1.278, 1.320, 1.321, 1.191, 1.330, 1.341, 1.313, 1.382, 1.320, 1.316],
      dR: 0.0012, dZ: -0.0018, dKappa: -0.008, dTri: 0.000, q95: 3.24, coilBias: 0.00
    },
    enn_w: {
      name: "ENN · Wasserstein",
      short: "ENN · Wasserstein",
      color: "#0284c7",
      dot: "#0284c7",
      mean: [58.121, 51.187, 15.723, 7.883, 4.342, 2.664, 1.980, 1.679, 1.555, 1.528, 1.419, 1.382, 1.341],
      sem:  [31.392, 31.848,  5.789, 3.451, 1.417, 0.411, 0.137, 0.064, 0.041, 0.033, 0.018, 0.030, 0.030],
      seeds10k: [1.420, 1.219, 1.244, 1.348, 1.394, 1.383, 1.382, 1.192, 1.491, 1.336],
      dR: 0.0022, dZ: -0.0028, dKappa: -0.014, dTri: -0.006, q95: 3.22, coilBias: -0.02
    },
    enn_e: {
      name: "ENN · Euclidean",
      short: "ENN · Euclidean",
      color: "#0d9488",
      dot: "#0d9488",
      mean: [58.121, 21.280, 14.670, 8.663, 2.607, 1.923, 1.847, 1.683, 1.612, 1.580, 1.438, 1.382, 1.352],
      sem:  [31.392,  6.046,  3.732, 2.461, 0.332, 0.171, 0.152, 0.079, 0.062, 0.055, 0.024, 0.022, 0.023],
      seeds10k: [1.364, 1.389, 1.333, 1.303, 1.319, 1.454, 1.237, 1.429, 1.267, 1.422],
      dR: 0.0026, dZ: -0.0034, dKappa: -0.017, dTri: -0.009, q95: 3.21, coilBias: 0.02
    },
    enn_h: {
      name: "ENN · Hyperbolic",
      short: "ENN · Hyperbolic",
      color: "#7c3aed",
      dot: "#7c3aed",
      mean: [58.121, 51.187, 11.814, 7.710, 3.489, 2.272, 1.826, 1.582, 1.545, 1.497, 1.452, 1.389, 1.354],
      sem:  [31.392, 31.848,  4.602, 3.475, 0.924, 0.333, 0.139, 0.048, 0.036, 0.032, 0.027, 0.029, 0.035],
      seeds10k: [1.356, 1.274, 1.370, 1.496, 1.269, 1.332, 1.239, 1.429, 1.223, 1.552],
      dR: 0.0028, dZ: -0.0036, dKappa: -0.018, dTri: -0.010, q95: 3.21, coilBias: -0.03
    },
    enn_c: {
      name: "ENN · Conformal",
      short: "ENN · Conformal",
      color: "#c026d3",
      dot: "#c026d3",
      mean: [58.121, 21.280, 14.670, 9.020, 4.624, 2.686, 1.877, 1.660, 1.566, 1.527, 1.456, 1.388, 1.369],
      sem:  [31.392,  6.046,  3.732, 2.726, 1.429, 0.693, 0.157, 0.079, 0.035, 0.031, 0.031, 0.030, 0.030],
      seeds10k: [1.317, 1.348, 1.363, 1.591, 1.326, 1.481, 1.352, 1.288, 1.277, 1.351],
      dR: 0.0032, dZ: -0.0042, dKappa: -0.022, dTri: -0.012, q95: 3.20, coilBias: 0.03
    }
  };

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function signedCoilNorm(mach, c, ci, st) {
    if (mach.id === "tcv" && st && st.coils && ci < st.coils.length) {
      return Math.max(-1.0, Math.min(1.0, st.coils[ci]));
    }
    if (typeof c.current_ma === "number" && Math.abs(c.current_ma) > 1e-4) {
      var refI = Math.max(0.45, (mach.ipNominal || 2.0) * 0.62);
      return Math.max(-1.0, Math.min(1.0, c.current_ma / refI));
    }
    return (c.r < mach.R0) ? -0.45 : 0.45;
  }

  function signedCoilColors(val) {
    var mag = Math.min(1.0, Math.abs(val));
    var t = Math.pow(mag, 0.72);
    if (val < -0.03) {
      return {
        body: [lerp(0.24, 0.11, t), lerp(0.28, 0.32, t), lerp(0.35, 0.86, t), 1.0],
        cap:  [lerp(0.32, 0.22, t), lerp(0.37, 0.49, t), lerp(0.45, 0.98, t), 1.0]
      };
    } else if (val > 0.03) {
      return {
        body: [lerp(0.24, 0.85, t), lerp(0.28, 0.47, t), lerp(0.35, 0.05, t), 1.0],
        cap:  [lerp(0.32, 0.97, t), lerp(0.37, 0.62, t), lerp(0.45, 0.12, t), 1.0]
      };
    }
    return {
      body: [0.26, 0.30, 0.36, 1.0],
      cap:  [0.34, 0.39, 0.46, 1.0]
    };
  }

  function signedCoilHex(val, isDark) {
    var mag = Math.min(1.0, Math.abs(val));
    if (val < -0.04) {
      return (mag > 0.42) ? (isDark ? "#3b82f6" : "#1d4ed8") : (isDark ? "#2563eb" : "#3b82f6");
    } else if (val > 0.04) {
      return (mag > 0.42) ? (isDark ? "#f59e0b" : "#d97706") : (isDark ? "#d97706" : "#f59e0b");
    }
    return isDark ? "#475569" : "#64748b";
  }

  // Reference 400 ms (80 x 5 ms substep) TCV 170 kA -> 38 kA rampdown target trajectory (TokaGym 170kA.yaml)
  function targetRampState(shotU) {
    var u = Math.max(0.0, Math.min(1.0, (typeof shotU === "number" && Number.isFinite(shotU)) ? shotU : 0.55));
    var sPos = u * u * (3.0 - 2.0 * u);
    var axR0 = SHAPE_70166_AXIS[0];
    var axZ0 = SHAPE_70166_AXIS[1];
    var R0 = lerp(axR0, 0.7860, sPos);
    var Z0 = lerp(axZ0, -0.0350, sPos);
    var scaleR = lerp(1.00, 0.56, Math.pow(u, 0.92));
    var scaleK = lerp(1.00, 0.58, Math.pow(u, 0.88));
    var ipKA = lerp(170.0, 38.0, Math.pow(u, 0.86));
    var wtotKJ = lerp(160.0, 4.5, Math.pow(u, 0.72));
    var q95 = lerp(3.22, 5.38, Math.pow(u, 0.90));
    var pts = [];
    for (var i = 0; i < SHAPE_70166_REFS.length; i++) {
      var dr = (SHAPE_70166_REFS[i][0] - axR0) * scaleR;
      var dz = (SHAPE_70166_REFS[i][1] - axZ0) * scaleK;
      pts.push([R0 + dr, Z0 + dz]);
    }
    return {
      u: u,
      tauMs: Math.round(u * 400.0),
      R0: R0,
      Z0: Z0,
      scaleR: scaleR,
      scaleK: scaleK,
      ipKA: ipKA,
      wtotKJ: wtotKJ,
      q95: q95,
      pts: pts
    };
  }

  function evalState(polKey, round, seedIdx, machId, shotU) {
    var pol = Object.prototype.hasOwnProperty.call(BENCH, polKey) ? BENCH[polKey] : BENCH.turbo0;
    if (machId && machId !== "tcv") {
      return {
        polKey: polKey,
        policy: pol,
        round: 10000,
        seedIdx: -1,
        progress: 1.0,
        shotU: 0.0,
        tauMs: 0,
        err: pol.mean[pol.mean.length - 1],
        sem: pol.sem[pol.sem.length - 1],
        dR: 0.0,
        dZ: 0.0,
        dKappa: 0.0,
        dTri: 0.0,
        q95: 3.24,
        ipKA: 1490.0,
        wtotKJ: 160.0,
        emission: 1.0,
        disrupted: false,
        quench: 0.0,
        flare: 0.0,
        statusKind: "ok",
        statusText: "Stationary Equilibrium",
        coils: TARGET_COILS,
        lcfsErrMm: 0.0
      };
    }

    var r = Math.max(1, Math.min(10000, Math.round(round || 1)));
    var idx = 0;
    while (idx < ROUNDS.length - 1 && ROUNDS[idx + 1] <= r) idx++;
    var iNext = Math.min(ROUNDS.length - 1, idx + 1);
    var r0 = Math.max(1, ROUNDS[idx]);
    var r1 = Math.max(r0, ROUNDS[iNext]);
    var t = (r1 <= r0) ? 1.0 : (Math.log(r) - Math.log(r0)) / Math.max(1e-6, Math.log(r1) - Math.log(r0));
    t = Math.max(0.0, Math.min(1.0, t));

    var errMean = lerp(pol.mean[idx], pol.mean[iNext], t);
    var semVal = lerp(pol.sem[idx], pol.sem[iNext], t);
    var err = errMean;
    if (seedIdx >= 0 && seedIdx < pol.seeds10k.length) {
      var termMean = pol.mean[pol.mean.length - 1];
      var seedDelta = pol.seeds10k[seedIdx] - termMean;
      var logFrac = Math.log(r) / Math.log(10000.0);
      var initFactor = 1.0 + 0.14 * ((seedIdx % 5) - 2) * (1.0 - logFrac);
      err = Math.max(1.12, errMean * initFactor + seedDelta * (0.25 + 0.75 * logFrac));
    }

    var uLin = Math.max(0.0, Math.min(1.0, (INIT_ERR - err) / (INIT_ERR - 1.311)));
    var uLog = Math.max(0.0, Math.min(1.0, (Math.log(INIT_ERR) - Math.log(Math.max(1.15, err))) / (Math.log(INIT_ERR) - Math.log(1.311))));
    var progress = Math.max(0.0, Math.min(1.0, 0.30 * uLin + 0.70 * uLog));

    var targ = targetRampState(shotU);
    var u = targ.u;
    var curR0 = targ.R0;
    var curZ0 = targ.Z0;
    var curScaleR = targ.scaleR;
    var curScaleK = targ.scaleK;
    var ipKA = targ.ipKA;
    var wtotKJ = targ.wtotKJ;
    var q95 = targ.q95;
    var emission = 1.0;
    var disrupted = false;
    var quench = 0.0;
    var flare = 0.0;
    var statusKind = "ok";
    var statusText = "";
    var coils = [];

    if (err > 9.5) {
      // Untrained / early unstable regime: vertical growth rate runaway (VDE) -> lower-inboard wall strike & quench
      var sev = Math.max(0.0, Math.min(1.0, (err - 9.5) / (INIT_ERR - 9.5)));
      var uCrash = lerp(0.72, 0.42, Math.pow(sev, 0.65));
      if (u < uCrash) {
        var g = u / Math.max(0.05, uCrash);
        var vde = Math.pow(g, 2.15);
        curScaleR = lerp(targ.scaleR, 0.56, vde);
        curScaleK = lerp(targ.scaleK, 0.68, vde);
        var crashR0 = 0.6250 + 0.2547 * 0.56;
        var crashZ0 = -0.355;
        curR0 = lerp(targ.R0, crashR0, vde) + 0.010 * Math.sin(u * 28.0) * g;
        curZ0 = lerp(targ.Z0, crashZ0, vde) + 0.012 * Math.cos(u * 24.0) * g;
        curR0 = Math.max(0.6245 + 0.2547 * curScaleR, curR0);
        ipKA = lerp(170.0, targetRampState(uCrash).ipKA * 0.88, g);
        wtotKJ = lerp(160.0, targetRampState(uCrash).wtotKJ * 0.72, g);
        q95 = lerp(3.22, 2.08, vde);
        emission = lerp(1.00, 0.92, g);
        statusKind = (g > 0.52) ? "warn" : "ok";
        statusText = (g > 0.52)
          ? ("⚠ VDE Runaway · " + Math.round(ipKA) + " kA")
          : ("Flat-Top · " + Math.round(ipKA) + " kA · " + Math.round(wtotKJ) + " kJ");
      } else {
        disrupted = true;
        quench = Math.min(1.0, (u - uCrash) / 0.18);
        flare = (quench < 0.68) ? Math.sin((quench / 0.68) * Math.PI) : 0.0;
        curScaleR = 0.56 * (1.0 - 0.60 * quench);
        curScaleK = 0.68 * (1.0 - 0.44 * quench);
        curR0 = 0.6250 + 0.2547 * curScaleR;
        curZ0 = -0.355 - 0.055 * quench;
        var preCrashT = targetRampState(uCrash);
        ipKA = Math.max(0.0, (preCrashT.ipKA * 0.88) * Math.pow(1.0 - quench, 1.35));
        wtotKJ = Math.max(0.0, (preCrashT.wtotKJ * 0.72) * Math.pow(Math.max(0.0, 1.0 - quench * 1.8), 1.5));
        q95 = lerp(2.08, 1.42, quench);
        emission = Math.max(0.14, (1.0 - 0.84 * quench) + 0.78 * flare);
        statusKind = "crash";
        statusText = (quench < 0.92)
          ? ("⚡ Wall Strike · " + Math.round(ipKA) + " kA")
          : ("⚡ Disrupted at " + Math.round(uCrash * 400.0) + " ms");
      }
      for (var cu = 0; cu < COILS_FLATTOP.length; cu++) {
        var osc = 0.22 * Math.sin(u * 18.0 + cu * 0.9) * Math.min(1.0, u * 3.0);
        var cRaw = lerp(COILS_FLATTOP[cu], COILS_UNSTABLE[cu], Math.min(1.0, u * 2.1)) + osc;
        if (disrupted) cRaw *= (1.0 - 0.65 * quench);
        coils.push(Math.max(-1.0, Math.min(1.0, cRaw)));
      }
    } else if (err > 2.4) {
      // Mid-training regime: stabilized against VDE crash, but lags and oscillates off target
      var imp = (err - 2.4) / 7.1;
      var lagU = Math.max(0.0, u - 0.24 * imp * Math.sin(u * Math.PI * 0.85));
      var lagT = targetRampState(lagU);
      var wobEnv = Math.sin(Math.min(1.0, u * 2.2) * Math.PI * 0.5);
      var wobZ = (0.022 + 0.062 * imp) * Math.sin(u * 12.5 - 0.7) * wobEnv - 0.042 * imp * u;
      var wobR = (0.012 + 0.030 * imp) * Math.cos(u * 9.5) * wobEnv;
      curScaleR = lagT.scaleR * (1.0 + 0.09 * imp * u);
      curScaleK = lagT.scaleK * (1.0 + 0.12 * imp * u);
      curR0 = Math.max(0.6250 + 0.2547 * curScaleR, lagT.R0 + wobR);
      curZ0 = lagT.Z0 + wobZ;
      ipKA = targ.ipKA + (8.0 + 28.0 * imp) * u;
      wtotKJ = targ.wtotKJ + (2.5 + 14.0 * imp) * u;
      q95 = lerp(targ.q95, 2.75, 0.45 * imp * u);
      emission = lerp(1.00, 0.70, Math.pow(u, 0.85));
      statusKind = (u > 0.18) ? "warn" : "ok";
      statusText = (u >= 0.96)
        ? ("⚠ Drift · " + Math.round(ipKA) + " kA · " + Math.round(wtotKJ) + " kJ")
        : ((u < 0.04) ? "Flat-Top · 170 kA · 160 kJ" : ("⚠ Lagging · " + Math.round(ipKA) + " kA · " + Math.round(wtotKJ) + " kJ"));
      for (var cm = 0; cm < COILS_FLATTOP.length; cm++) {
        var cTargM = lerp(COILS_FLATTOP[cm], COILS_TERMINAL[cm], lagU);
        var cWob = 0.16 * imp * Math.sin(u * 13.0 + cm * 0.7) * wobEnv;
        coils.push(Math.max(-1.0, Math.min(1.0, cTargM + cWob)));
      }
    } else {
      // Converged closed-loop policy (t >= 100): smooth 170 kA -> 38 kA rampdown locked onto moving target
      var res = Math.max(0.0, (err - 1.18) / 1.22);
      var seedNudge = (seedIdx >= 0 && seedIdx < 10) ? ((seedIdx - 4.5) * 0.0007) : 0.0;
      var envW = Math.sin(u * Math.PI);
      curScaleR = targ.scaleR * (1.0 + 0.022 * res * envW);
      curScaleK = targ.scaleK * (1.0 + pol.dKappa * 0.6 + 0.028 * res * envW);
      curR0 = Math.max(
        0.6241 + 0.2547 * curScaleR,
        targ.R0 + (pol.dR * 1.8 + seedNudge + 0.0035 * res * Math.sin(u * 9.0)) * (0.20 + 0.80 * envW)
      );
      curZ0 = targ.Z0 + (pol.dZ * 1.8 - seedNudge + 0.005 * res * Math.cos(u * 11.0)) * (0.20 + 0.80 * envW);
      ipKA = targ.ipKA + 2.8 * res * u;
      wtotKJ = targ.wtotKJ + 0.5 * res * u;
      q95 = targ.q95 + (pol.q95 - 3.24) * 2.0;
      emission = lerp(1.00, 0.64, Math.pow(u, 0.85));
      statusKind = (u >= 0.96) ? "done" : "ok";
      statusText = (u >= 0.96)
        ? ("✓ Shutdown · " + Math.round(ipKA) + " kA · " + wtotKJ.toFixed(1) + " kJ")
        : ((u < 0.04) ? "Flat-Top · 170 kA · 160 kJ" : ("✓ Tracking · " + Math.round(ipKA) + " kA · " + Math.round(wtotKJ) + " kJ"));
      for (var cc = 0; cc < COILS_FLATTOP.length; cc++) {
        var bias = ((cc % 2 === 0) ? pol.coilBias : -pol.coilBias);
        var cVal = lerp(COILS_FLATTOP[cc], COILS_TERMINAL[cc] + bias, u);
        coils.push(Math.max(-1.0, Math.min(1.0, cVal)));
      }
    }

    var geomShiftMm = Math.hypot(curR0 - targ.R0, curZ0 - targ.Z0) * 1000.0;
    var shapeDiffMm = (Math.abs(curScaleR - targ.scaleR) * 230.0 + Math.abs(curScaleK - targ.scaleK) * 260.0) * 0.45;
    var baseMm = 2.1 + Math.max(0.0, (err - 1.311) * 3.8);
    var lcfsErrMm = (u < 0.01) ? baseMm : Math.max(1.8, geomShiftMm + shapeDiffMm + 0.55 * Math.min(8.0, baseMm));

    return {
      polKey: polKey,
      policy: pol,
      round: r,
      seedIdx: seedIdx,
      progress: progress,
      shotU: u,
      tauMs: targ.tauMs,
      targ: targ,
      curR0: curR0,
      curZ0: curZ0,
      scaleR: curScaleR,
      scaleK: curScaleK,
      ipKA: ipKA,
      wtotKJ: wtotKJ,
      emission: emission,
      disrupted: disrupted,
      quench: quench,
      flare: flare,
      statusKind: statusKind,
      statusText: statusText,
      err: err,
      sem: semVal,
      dR: curR0 - SHAPE_70166_AXIS[0],
      dZ: curZ0 - SHAPE_70166_AXIS[1],
      dKappa: curScaleK - 1.0,
      dTri: curScaleR - 1.0,
      q95: q95,
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
      var curR0 = (typeof st.curR0 === "number") ? st.curR0 : (axR0 + st.dR * 0.45 * scaleA);
      var curZ0 = (typeof st.curZ0 === "number") ? st.curZ0 : (axZ0 + st.dZ * 0.45 * scaleA);
      var rS = (typeof st.scaleR === "number") ? st.scaleR : (1.0 + st.dTri * 0.18);
      var kS = ((typeof st.scaleK === "number") ? st.scaleK : (1.0 + st.dKappa * 0.35)) * (0.84 + 0.16 * rho);
      var outPts = [];
      var src = mach.lcfs;
      for (var j = 0; j < src.length; j++) {
        var dr = (src[j][0] - axR0) * rho * rS;
        var dz = (src[j][1] - axZ0) * rho * kS;
        outPts.push([curR0 + dr, curZ0 + dz]);
      }
      var rxS = (typeof mach.rx === "number") ? (curR0 + (mach.rx - axR0) * rS) : (curR0 - mach.a * 0.40 * rS);
      var zxS = (typeof mach.zx === "number") ? (curZ0 + (mach.zx - axZ0) * kS) : (curZ0 - mach.a * 1.55 * kS);
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
    "  if (vUnlit > 0.75) {",
    "    gl_FragColor = vCol;",
    "    return;",
    "  }",
    "  vec3 n = normalize(vNorm);",
    "  vec3 v = normalize(vViewDir);",
    "  if (dot(n, v) < 0.0) n = -n;",
    "  vec3 l1 = normalize(vec3(0.55, 0.85, 0.45));",
    "  if (vUnlit > 0.25) {",
    "    float dp = max(dot(n, l1), 0.0);",
    "    float fp = pow(1.0 - max(dot(n, v), 0.0), 2.3) * 0.36;",
    "    vec3 eCol = min(vec3(1.0), vCol.rgb * (0.80 + 0.22 * dp) + vec3(fp * 0.52, fp * 0.32, fp * 0.68));",
    "    gl_FragColor = vec4(eCol, vCol.a);",
    "    return;",
    "  }",
    "  vec3 l2 = normalize(vec3(-0.65, 0.35, -0.65));",
    "  float diff1 = max(dot(n, l1), 0.0);",
    "  float diff2 = max(dot(n, l2), 0.0) * 0.35;",
    "  vec3 h1 = normalize(l1 + v);",
    "  float spec = pow(max(dot(n, h1), 0.0), 34.0) * 0.26;",
    "  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.14;",
    "  vec3 ambient = vec3(0.30, 0.32, 0.36);",
    "  vec3 lit = vCol.rgb * (ambient + diff1 * 0.66 + diff2) + vec3(spec + fres);",
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

  // Build the complete 3D CAD mesh buffers for active FUSE.jl / TCV tokamak geometry
  function buildTokamakCADBuffers(gl, st, opts) {
    var mKey = opts.machine || "tcv";
    var mach = Object.prototype.hasOwnProperty.call(MACHINES, mKey) ? MACHINES[mKey] : MACHINES.tcv;
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

    // 0. Discrete 3D Orthogonal Toroidal Field (TF) B_phi D-Coils (uniform cool anodized cryo-slate)
    if (opts.showTF && mach.tf_outer && mach.tf_inner && mach.tf_outer.length >= 4 && mach.tf_outer.length === mach.tf_inner.length) {
      var nTF = mach.n_tf || 18;
      var halfAng = (Math.PI / nTF) * 0.34;
      var tfCaseCol = [0.24, 0.31, 0.40, 1.0];
      var tfSideCol = [0.35, 0.44, 0.55, 1.0];
      for (var ti = 0; ti < nTF; ti++) {
        var phiC = -0.5 * Math.PI + (2.0 * Math.PI * ti) / nTF;
        if (phiC > (phiVessel0 + 8.0 * deg) && phiC < (phiVessel1 - 8.0 * deg)) {
          addHollowShell(mach.tf_outer, mach.tf_inner, phiC - halfAng, phiC + halfAng, 2, tfCaseCol, tfCaseCol, tfSideCol);
        }
      }
    }

    // 1. Poloidal Field & Central Solenoid Coils (diverging signed current polarity: -I cobalt <-> 0 slate <-> +I copper-amber)
    if (opts.showCoils && mach.coils) {
      for (var ci = 0; ci < mach.coils.length; ci++) {
        var c = mach.coils[ci];
        var rIn = Math.max(0.05, c.r - c.w * 0.5);
        var rOut = c.r + c.w * 0.5;
        var zBot = c.z - c.h * 0.5;
        var zTop = c.z + c.h * 0.5;
        var isInner = (c.r < mach.R0);
        var cVal = signedCoilNorm(mach, c, ci, st);
        var cc = signedCoilColors(cVal);
        addToroidalBoxRing(rIn, rOut, zBot, zTop, isInner ? phiInner0 : phiOuter0, isInner ? phiInner1 : phiOuter1, 44, cc.body, cc.cap);
      }
    }

    // 2. Recessed Matte Tungsten / Graphite Double-Wall Vacuum Vessel Stage
    var vesselShellCol = [0.34, 0.39, 0.46, 1.0];
    var vesselInnerCol = [0.12, 0.14, 0.18, 1.0];
    var cutRimCol      = [0.64, 0.70, 0.78, 1.0];
    addHollowShell(mach.vessel_outer, mach.vessel_inner, phiVessel0, phiVessel1, 60, vesselShellCol, vesselInnerCol, cutRimCol);

    function buildVesselCutaway(phiVesselPlane, signN) {
      var dnScale = Math.max(1.0, mach.R0 * 0.55);
      addCutawayFan(
        mach.vessel_outer, [mach.R0, mach.Z0 || 0.0], phiVesselPlane, signN, 0.001 * dnScale * signN,
        cutRimCol, [0.18, 0.21, 0.26, 0.90]
      );
      addCutawayFan(
        mach.first_wall, [mach.R0, mach.Z0 || 0.0], phiVesselPlane, signN, 0.004 * dnScale * signN,
        vesselInnerCol, [0.50, 0.57, 0.66, 0.92]
      );
    }
    buildVesselCutaway(phiVessel0, -1.0);
    buildVesselCutaway(phiVessel1,  1.0);

    var hwTriCount = pos.length / 3;

    // 3. Self-Luminous Volumetric D_alpha Plasma Torus Arms & Nested Poloidal Flux Surfaces
    var eq = plasmaContour(1.0, st, 72, mach);
    var phiLeftPlasma = phiVessel0 - 34.0 * scaleCut * deg;
    var phiRightPlasma = phiVessel1 + 22.0 * scaleCut * deg;
    var em = (typeof st.emission === "number") ? st.emission : 1.0;
    var isDis = !!st.disrupted;
    var fl = st.flare || 0.0;
    var qn = st.quench || 0.0;
    var dimQ = 1.0 - 0.55 * qn;

    var shellCol = isDis
      ? [lerp(0.42, 0.95, fl) * dimQ, lerp(0.11, 0.22, fl) * dimQ, lerp(0.32, 0.14, fl) * dimQ, 1.0]
      : [0.44 * (0.55 + 0.45 * em), 0.13 * (0.50 + 0.50 * em), 0.68 * (0.58 + 0.42 * em), 1.0];

    addRevolvedSolid(
      eq.pts, phiLeftPlasma, phiVessel0, 26,
      shellCol, null, null, [eq.R0, eq.Z0]
    );
    addRevolvedSolid(
      eq.pts, phiVessel1, phiRightPlasma, 18,
      shellCol, null, null, [eq.R0, eq.Z0]
    );

    function buildPlasmaCutaway(phiPlasmaPlane, signN) {
      var dnScale = Math.max(1.0, mach.R0 * 0.55);
      var coreLift = 0.52 + 0.48 * em;
      var fluxBands = isDis ? [
        { rho: 1.00, col: [lerp(0.52, 0.94, fl) * dimQ, lerp(0.12, 0.20, fl) * dimQ, 0.16 * dimQ, 1.0], edge: [0.98, 0.28, 0.24, 0.96] },
        { rho: 0.80, col: [lerp(0.66, 0.98, fl) * dimQ, lerp(0.15, 0.34, fl) * dimQ, 0.18 * dimQ, 1.0], edge: null },
        { rho: 0.60, col: [lerp(0.78, 1.00, fl) * dimQ, lerp(0.22, 0.50, fl) * dimQ, 0.22 * dimQ, 1.0], edge: null },
        { rho: 0.40, col: [lerp(0.86, 1.00, fl) * dimQ, lerp(0.36, 0.70, fl) * dimQ, lerp(0.30, 0.38, fl) * dimQ, 1.0], edge: null },
        { rho: 0.20, col: [lerp(0.92, 1.00, fl) * dimQ, lerp(0.54, 0.88, fl) * dimQ, lerp(0.46, 0.62, fl) * dimQ, 1.0], edge: null }
      ] : [
        { rho: 1.00, col: [0.44 * coreLift, 0.13 * coreLift, 0.68 * (0.65 + 0.35 * em), 1.0], edge: [0.18, 0.88, 0.99, 0.96] },
        { rho: 0.80, col: [0.66 * coreLift, 0.17 * coreLift, 0.78 * (0.68 + 0.32 * em), 1.0], edge: null },
        { rho: 0.60, col: [0.85 * coreLift, 0.26 * coreLift, 0.78 * (0.72 + 0.28 * em), 1.0], edge: null },
        { rho: 0.40, col: [lerp(0.78, 0.96, em), lerp(0.36, 0.55, em), lerp(0.78, 0.88, em), 1.0], edge: null },
        { rho: 0.20, col: [lerp(0.86, 0.99, em), lerp(0.58, 0.92, em), lerp(0.90, 0.99, em), 1.0], edge: null }
      ];
      for (var b = 0; b < fluxBands.length; b++) {
        var band = fluxBands[b];
        var cData = plasmaContour(band.rho, st, 64, mach);
        addCutawayFan(
          cData.pts, [cData.R0, cData.Z0], phiPlasmaPlane, signN, (0.008 + 0.0018 * b) * dnScale * signN,
          band.col, band.edge
        );
      }

      var legScale = (mach.id === "tcv") ? Math.max(0.18, 1.0 - 0.68 * (st.shotU || 0.0)) : 1.0;
      if (!isDis || qn < 0.35) {
        var pX = cylToXYZ(eq.Rx, eq.Zx, phiPlasmaPlane, 0.0085 * dnScale * signN);
        var pLegIn = cylToXYZ(eq.Rx - mach.a * 0.18 * legScale, eq.Zx - mach.a * 0.28 * legScale, phiPlasmaPlane, 0.0085 * dnScale * signN);
        var pLegOut = cylToXYZ(eq.Rx + mach.a * 0.28 * legScale, eq.Zx - mach.a * 0.26 * legScale, phiPlasmaPlane, 0.0085 * dnScale * signN);
        var cyanLine = isDis ? [0.98, 0.34, 0.24, 0.85] : [0.18, 0.88, 0.99, 1.0];
        pushLine(pX, pLegIn, cyanLine);
        pushLine(pX, pLegOut, cyanLine);
      }
    }

    buildPlasmaCutaway(phiLeftPlasma,  -1.0);
    buildPlasmaCutaway(phiRightPlasma,  1.0);

    // 3b. 3D Wall-Strike Thermal-Quench Arc along Inboard Limiter Tiles when Disrupted
    if (isDis && mach.id === "tcv") {
      var pArc0 = phiVessel1;
      var pArc1 = phiVessel0 + 2.0 * Math.PI;
      var arcSteps = 44;
      var arcAlpha = Math.max(0.32, 0.96 - 0.52 * qn);
      for (var ai = -1; ai <= 1; ai++) {
        var zStrike = eq.Z0 + ai * 0.032;
        var prevA = null;
        for (var ak = 0; ak <= arcSteps; ak++) {
          var phiA = pArc0 + (pArc1 - pArc0) * (ak / arcSteps);
          var ptA = cylToXYZ(0.627, zStrike, phiA, 0.0);
          if (prevA) {
            var spark = 0.5 + 0.5 * Math.sin(ak * 0.9 + (opts.fieldPhase || 0.0) * 6.0 + ai);
            pushLine(prevA, ptA, [1.0, lerp(0.22, 0.72, spark * fl), 0.16, arcAlpha]);
          }
          prevA = ptA;
        }
      }
    }

    // 4. Smooth Interpolated 3D Helical q95 Magnetic Field Lines with Flowing Alfvénic Pulses
    if (opts.showFieldLines && eq.pts.length >= 16) {
      var qEff = Math.max(1.4, (mach.id === "tcv") ? st.q95 : (mach.q95 || 3.2));
      var nLines = 8;
      var nPts = eq.pts.length;
      var pFront0 = phiVessel1 - 2.0 * deg;
      var pFront1 = phiVessel0 + 2.0 * Math.PI + 2.0 * deg;
      var pGap0 = phiRightPlasma;
      var pGap1 = phiLeftPlasma + 2.0 * Math.PI;
      var pitchRate = 2.0 / qEff;
      var fPhase = opts.fieldPhase || 0.0;
      for (var l = 0; l < nLines; l++) {
        var th0 = (2.0 * Math.PI * l) / nLines;
        var prevPt = null;
        var steps = 144;
        for (var k = 0; k <= steps; k++) {
          var phi = pFront0 + (pFront1 - pFront0) * (k / steps);
          var th = th0 + pitchRate * (phi - pFront0);
          var fIdx = (((th / (2.0 * Math.PI)) * nPts) % nPts + nPts) % nPts;
          var i0 = Math.floor(fIdx);
          var i1 = (i0 + 1) % nPts;
          var frac = fIdx - i0;
          var rI = lerp(eq.pts[i0][0], eq.pts[i1][0], frac);
          var zI = lerp(eq.pts[i0][1], eq.pts[i1][1], frac);
          var rOutP = eq.R0 + (rI - eq.R0) * 1.018;
          var zOutP = eq.Z0 + (zI - eq.Z0) * 1.018;
          var pt = cylToXYZ(rOutP, zOutP, phi, 0.0);
          if (prevPt) {
            var inGap = (phi > pGap0 && phi < pGap1);
            var wave = 0.5 + 0.5 * Math.sin((k / steps) * Math.PI * 6.0 - fPhase * 4.0 + l * 0.785);
            var pulse = Math.pow(wave, 5.0);
            if (isDis) {
              if (!inGap || (k % 3 === 0)) {
                pushLine(prevPt, pt, [lerp(0.86, 1.0, pulse), lerp(0.20, 0.58, pulse), 0.18, Math.max(0.22, (inGap ? 0.40 : 0.90) * (1.0 - 0.55 * qn))]);
              }
            } else if (!inGap) {
              pushLine(prevPt, pt, [lerp(0.14, 0.90, pulse), lerp(0.84, 0.99, pulse), 1.00, 0.98]);
            } else if (k % 2 === 0) {
              pushLine(prevPt, pt, [lerp(0.18, 0.88, pulse), lerp(0.86, 0.99, pulse), 1.00, lerp(0.42, 0.86, pulse)]);
            }
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
      hwTriCount: hwTriCount,
      triCount: pos.length / 3,
      linePosBuf: makeBuf(linePos),
      lineColBuf: makeBuf(lineCol),
      lineCount: linePos.length / 3
    };
  }

  // Right Panel: 2D Physical Cross-Section (FUSE.jl Tokamak Equilibrium)
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
        ctx.strokeStyle = isDark ? "#475569" : "#94a3b8";
        ctx.lineWidth = 1.05;
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
        var cVal = signedCoilNorm(mach, c, ci, st);
        ctx.fillStyle = signedCoilHex(cVal, isDark);
        ctx.fillRect(pTL[0], pTL[1], cW, cH);
        ctx.strokeStyle = isDark ? "#0d1219" : "#111111";
        ctx.lineWidth = 0.75;
        ctx.strokeRect(pTL[0], pTL[1], cW, cH);
      }
      ctx.restore();
    }

    function drawMachineProfiles(boxX) {
      ctx.save();
      ctx.strokeStyle = isDark ? "#64748b" : "#475569";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (var i = 0; i < mach.vessel_outer.length; i++) {
        var p = rzToXY(boxX, mach.vessel_outer[i][0], mach.vessel_outer[i][1]);
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.stroke();

      ctx.strokeStyle = isDark ? "#94a3b8" : "#334155";
      ctx.lineWidth = 1.35;
      ctx.setLineDash([5, 2.5]);
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

    var plotBg = isDark ? "#0d1219" : "#ffffff";
    var plotBorder = isDark ? "#30363d" : "#111111";
    var cyanTarget = isDark ? "#38bdf8" : "#0284c7";
    var magentaPlasma = isDark ? "#e879f9" : "#c026d3";

    var leftX = startX;
    ctx.fillStyle = isDark ? "#e6edf3" : "#111111";
    ctx.font = titleFont;
    ctx.textAlign = "center";
    ctx.fillText(mach.id === "tcv" ? "CAD & Target Shape" : "FUSE CAD & LCFS", leftX + boxW * 0.5, startY - (compact ? 7 : 9));

    ctx.fillStyle = plotBg;
    ctx.fillRect(leftX, startY, boxW, boxH);
    ctx.strokeStyle = plotBorder;
    ctx.lineWidth = 1.1;
    ctx.strokeRect(leftX, startY, boxW, boxH);
    drawBoxTicks(leftX, true);

    ctx.save();
    ctx.beginPath();
    ctx.rect(leftX, startY, boxW, boxH);
    ctx.clip();
    drawCoils(leftX);
    drawMachineProfiles(leftX);

    // Ghost τ = 0 ms flat-top outline + target descent trajectory guide on TCV
    if (mach.id === "tcv" && (st.shotU || 0.0) > 0.03) {
      ctx.save();
      ctx.strokeStyle = isDark ? "rgba(148,163,184,0.28)" : "rgba(100,116,139,0.28)";
      ctx.lineWidth = 1.0;
      ctx.setLineDash([2.5, 2.5]);
      ctx.beginPath();
      for (var gIdx = 0; gIdx < SHAPE_70166_REFS.length; gIdx++) {
        var pg = rzToXY(leftX, SHAPE_70166_REFS[gIdx][0], SHAPE_70166_REFS[gIdx][1]);
        if (gIdx === 0) ctx.moveTo(pg[0], pg[1]); else ctx.lineTo(pg[0], pg[1]);
      }
      ctx.closePath();
      ctx.stroke();

      var pAx0 = rzToXY(leftX, SHAPE_70166_AXIS[0], SHAPE_70166_AXIS[1]);
      var pAx1 = rzToXY(leftX, 0.7860, -0.0350);
      ctx.strokeStyle = isDark ? "rgba(56,189,248,0.38)" : "rgba(2,132,199,0.36)";
      ctx.lineWidth = 1.1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(pAx0[0], pAx0[1]);
      ctx.lineTo(pAx1[0], pAx1[1]);
      ctx.stroke();
      ctx.restore();
    }

    var refPoly = (mach.id === "tcv") ? ((st && st.targ && st.targ.pts) ? st.targ.pts : SHAPE_70166_REFS) : mach.lcfs;
    if (refPoly && refPoly.length >= 8) {
      ctx.strokeStyle = cyanTarget;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      for (var rIdx = 0; rIdx < refPoly.length; rIdx++) {
        var pr = rzToXY(leftX, refPoly[rIdx][0], refPoly[rIdx][1]);
        if (rIdx === 0) ctx.moveTo(pr[0], pr[1]); else ctx.lineTo(pr[0], pr[1]);
      }
      ctx.closePath();
      ctx.stroke();
      if (mach.id === "tcv") {
        ctx.fillStyle = cyanTarget;
        for (var dIdx = 0; dIdx < refPoly.length; dIdx++) {
          var pd = rzToXY(leftX, refPoly[dIdx][0], refPoly[dIdx][1]);
          ctx.beginPath();
          ctx.arc(pd[0], pd[1], 1.85, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    var eqTarget = plasmaContour(1.0, st, 64, mach);
    if (mach.id === "tcv" && st.disrupted) {
      var pStrike = rzToXY(leftX, 0.624, eqTarget.Z0);
      var glowR = Math.max(12, boxW * (0.18 + 0.10 * (st.flare || 0.0)));
      var grad = ctx.createRadialGradient(pStrike[0], pStrike[1], 1.5, pStrike[0], pStrike[1], glowR);
      var gAlpha = Math.max(0.28, 0.72 - 0.35 * (st.quench || 0.0));
      grad.addColorStop(0.0, "rgba(251,146,60," + gAlpha.toFixed(2) + ")");
      grad.addColorStop(0.45, "rgba(239,68,68," + (gAlpha * 0.65).toFixed(2) + ")");
      grad.addColorStop(1.0, "rgba(239,68,68,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(pStrike[0], pStrike[1], glowR, 0, Math.PI * 2);
      ctx.fill();

      var pWallTop = rzToXY(leftX, 0.624, eqTarget.Z0 + 0.14);
      var pWallBot = rzToXY(leftX, 0.624, eqTarget.Z0 - 0.14);
      ctx.strokeStyle = isDark ? "#fb923c" : "#dc2626";
      ctx.lineWidth = 2.8;
      ctx.beginPath();
      ctx.moveTo(pWallTop[0], pWallTop[1]);
      ctx.lineTo(pWallBot[0], pWallBot[1]);
      ctx.stroke();
    }

    ctx.save();
    ctx.strokeStyle = st.disrupted ? (isDark ? "#f87171" : "#dc2626") : magentaPlasma;
    ctx.lineWidth = st.disrupted ? 2.0 : 1.75;
    if (mach.id === "tcv") ctx.setLineDash([4, 2]);
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

    ctx.fillStyle = plotBg;
    ctx.fillRect(rightX, startY, boxW, boxH);
    ctx.strokeStyle = plotBorder;
    ctx.lineWidth = 1.1;
    ctx.strokeRect(rightX, startY, boxW, boxH);
    drawBoxTicks(rightX, false);

    ctx.save();
    ctx.beginPath();
    ctx.rect(rightX, startY, boxW, boxH);
    ctx.clip();
    drawCoils(rightX);
    drawMachineProfiles(rightX);

    var fluxLevels = st.disrupted ? (isDark ? [
      { rho: 1.00, col: "#ef4444", width: 1.45 },
      { rho: 0.84, col: "#f97316", width: 1.15 },
      { rho: 0.68, col: "#fb923c", width: 1.05 },
      { rho: 0.52, col: "#fca5a5", width: 1.00 },
      { rho: 0.36, col: "#fdba74", width: 0.95 },
      { rho: 0.18, col: "#fef2f2", width: 0.95 }
    ] : [
      { rho: 1.00, col: "#b91c1c", width: 1.45 },
      { rho: 0.84, col: "#dc2626", width: 1.15 },
      { rho: 0.68, col: "#ea580c", width: 1.05 },
      { rho: 0.52, col: "#ef4444", width: 1.00 },
      { rho: 0.36, col: "#f97316", width: 0.95 },
      { rho: 0.18, col: "#fb923c", width: 0.95 }
    ]) : (isDark ? [
      { rho: 1.00, col: "#38bdf8", width: 1.35 },
      { rho: 0.84, col: "#7e22ce", width: 1.15 },
      { rho: 0.68, col: "#a21caf", width: 1.05 },
      { rho: 0.52, col: "#d946ef", width: 1.00 },
      { rho: 0.36, col: "#f0abfc", width: 0.95 },
      { rho: 0.18, col: "#fae8ff", width: 0.95 }
    ] : [
      { rho: 1.00, col: "#0284c7", width: 1.35 },
      { rho: 0.84, col: "#6b21a8", width: 1.15 },
      { rho: 0.68, col: "#9333ea", width: 1.05 },
      { rho: 0.52, col: "#c026d3", width: 1.00 },
      { rho: 0.36, col: "#db2777", width: 0.95 },
      { rho: 0.18, col: "#ec4899", width: 0.95 }
    ]);
    ctx.save();
    if (st.disrupted) {
      ctx.globalAlpha = Math.max(0.32, 1.0 - 0.52 * (st.quench || 0.0));
    }
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
    ctx.restore();

    var pAxis = rzToXY(rightX, eqTarget.R0, eqTarget.Z0);
    ctx.fillStyle = st.disrupted ? (isDark ? "#fca5a5" : "#dc2626") : (isDark ? "#fdf4ff" : "#be185d");
    ctx.beginPath();
    ctx.arc(pAxis[0], pAxis[1], 2.8, 0, Math.PI * 2);
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
      '    <div class="tcv-hud-badge tcv-ctrl-title" style="padding:5px 12px;font-size:12px;"><strong>Simulation Console</strong></div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-device">',
      '      <div class="tcv-ctrl-hdr"><span>Environment</span><span id="tcv-builder-tag">toka_gym</span></div>',
      '      <select id="tcv-machine-sel" class="tcv-select">',
      '        <optgroup label="Closed-Loop Policy Search">',
      '          <option value="tcv" selected>TCV · EPFL (toka_gym)</option>',
      '        </optgroup>',
      '        <optgroup label="Stationary Equilibria · FUSE.jl">',
      '          <option value="sparc">SPARC · CFS (FUSE.jl)</option>',
      '          <option value="arc">ARC · MIT/CFS (FUSE.jl)</option>',
      '          <option value="iter">ITER · Cadarache (FUSE.jl)</option>',
      '          <option value="d3d">DIII-D · GA (FUSE.jl)</option>',
      '        </optgroup>',
      '      </select>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-policy" id="tcv-sec-policy">',
      '      <div class="tcv-ctrl-hdr"><span>Optimizer Arm</span><span>Loss (↓)</span></div>',
      '      <div class="tcv-pol-group">',
      '        <button type="button" class="tcv-pol-btn active" data-pol="turbo0"><span><span class="tcv-pol-dot" style="background:var(--bench-c-hero,#111);"></span>TuRBO-0 (R=10)</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">1.31</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="enn_w"><span><span class="tcv-pol-dot" style="background:var(--bench-c-enn1,#1d4ed8);"></span>ENN · Wasserstein</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">1.34</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="enn_e"><span><span class="tcv-pol-dot" style="background:var(--bench-c-enn2,#0f766e);"></span>ENN · Euclidean</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">1.35</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="enn_h"><span><span class="tcv-pol-dot" style="background:var(--bench-c-enn3,#b45309);"></span>ENN · Hyperbolic</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">1.35</span></button>',
      '        <button type="button" class="tcv-pol-btn" data-pol="enn_c"><span><span class="tcv-pol-dot" style="background:var(--bench-c-enn4,#7e22ce);"></span>ENN · Conformal</span><span style="font-family:var(--font-mono,monospace);font-size:11px;">1.37</span></button>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-shot" id="tcv-sec-shot">',
      '      <div class="tcv-ctrl-hdr"><span>Rampdown Movie</span><strong id="tcv-shot-lbl" style="font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-primary,#111);">τ = 220 ms</strong></div>',
      '      <div class="tcv-round-row">',
      '        <input type="range" id="tcv-shot-slider" min="0" max="400" step="2" value="220" aria-label="Rampdown Time (0 to 400 ms)" />',
      '        <button type="button" id="tcv-play-btn" class="tcv-pol-btn tcv-play-cta" style="justify-content:center;font-weight:700;">▶ Play 400 ms Shot</button>',
      '      </div>',
      '      <div class="tcv-shot-status is-ok" id="tcv-shot-status">✓ Tracking · Ip 92 kA · 56 kJ</div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-round" id="tcv-sec-round">',
      '      <div class="tcv-ctrl-hdr"><span>Policy Round</span><strong id="tcv-round-lbl" style="font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-primary,#111);">t = 10,000</strong></div>',
      '      <div class="tcv-round-row">',
      '        <input type="range" id="tcv-round-slider" min="0" max="1000" step="1" value="1000" aria-label="Evaluation Round (log scale)" />',
      '      </div>',
      '      <div class="tcv-preset-row" role="group" aria-label="Training Round Presets">',
      '        <button type="button" class="tcv-preset-btn" data-round-preset="1">t=1 VDE</button>',
      '        <button type="button" class="tcv-preset-btn" data-round-preset="10">t=10 Drift</button>',
      '        <button type="button" class="tcv-preset-btn active" data-round-preset="10000">10k Clean</button>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-seed" id="tcv-sec-seed">',
      '      <div class="tcv-ctrl-hdr"><span>Paired Seed</span><span><em>n</em> = 10</span></div>',
      '      <select id="tcv-seed-sel" class="tcv-select mono">',
      '        <option value="-1">Mean ± SEM (n = 10)</option>',
      '        <option value="0">Seed 1 (R=10: 1.28)</option>',
      '        <option value="1">Seed 2 (R=10: 1.32)</option>',
      '        <option value="2">Seed 3 (R=10: 1.32)</option>',
      '        <option value="3">Seed 4 (R=10: 1.19)</option>',
      '        <option value="4">Seed 5 (R=10: 1.33)</option>',
      '        <option value="5">Seed 6 (R=10: 1.34)</option>',
      '        <option value="6">Seed 7 (R=10: 1.31)</option>',
      '        <option value="7">Seed 8 (R=10: 1.38)</option>',
      '        <option value="8">Seed 9 (R=10: 1.32)</option>',
      '        <option value="9">Seed 10 (R=10: 1.32)</option>',
      '      </select>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-cutaway" id="tcv-sec-cutaway" style="display:none;">',
      '      <div class="tcv-ctrl-hdr"><span>Cutaway Sector</span><strong id="tcv-cut-lbl" style="font-family:var(--font-mono,monospace);font-size:11.5px;color:var(--fg-primary,#111);">56°</strong></div>',
      '      <div class="tcv-round-row">',
      '        <input type="range" id="tcv-cut-slider" min="28" max="84" step="1" value="56" aria-label="3D Cutaway Sector Angle" />',
      '      </div>',
      '    </div>',
      '    <div class="tcv-ctrl-sec tcv-ctrl-layers">',
      '      <div class="tcv-ctrl-hdr"><span>3D / 2D Layers</span></div>',
      '      <div class="tcv-layers-row">',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-coils" checked /> <span id="tcv-lbl-coils">PF/CS Coils (±I)</span></label>',
      '        <label class="tcv-chk-row" id="tcv-row-tf"><input type="checkbox" id="tcv-chk-tf" checked /> <span>TF Bφ Cage</span></label>',
      '        <label class="tcv-chk-row"><input type="checkbox" id="tcv-chk-field" checked /> <span id="tcv-lbl-field">Helical q95 Lines</span></label>',
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
      '        <div class="tcv-hud-badge"><span id="tcv-badge-b"><strong>(b)</strong> Equilibrium & Poloidal Flux</span></div>',
      '        <div class="tcv-canvas-wrap" id="tcv-fig9-wrap">',
      '          <canvas id="tcv-fig9-canvas"></canvas>',
      '        </div>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-legend-strip" aria-label="Physical Palette and Telemetry">',
      '      <span class="tcv-leg-chip"><span class="tcv-leg-swatch" style="background:linear-gradient(90deg,#6b21a8 0%,#d946ef 55%,#fdf4ff 100%);"></span>Dα Flux ψ(ρ̂)</span>',
      '      <span class="tcv-leg-chip"><span class="tcv-leg-line" style="background:#22d3ee;"></span>Helical q95 B-lines &amp; Target</span>',
      '      <span class="tcv-leg-chip"><span class="tcv-leg-swatch" style="background:linear-gradient(90deg,#1d4ed8 0%,#64748b 50%,#d97706 100%);"></span>PF/CS −I / +I Polarity</span>',
      '      <span class="tcv-leg-chip"><span class="tcv-leg-swatch" style="background:#475569;"></span>TF Bφ Cage &amp; Vessel</span>',
      '      <span class="tcv-legend-tele" id="tcv-legend-tele"></span>',
      '    </div>',
      '    <div class="tcv-bench-dock" id="tcv-bench-dock"></div>',
      '    <div class="tcv-fuse-deck" id="tcv-fuse-deck" style="display:none;">',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span id="tcv-subtable-title">FUSE.jl 1.2.1 Equilibrium</span><span id="tcv-subtable-device">SPARC · CFS</span></div>',
      '        <div class="tcv-kv-list" id="tcv-kv-list"></div>',
      '      </div>',
      '      <div class="tcv-subtable-cell">',
      '        <div class="tcv-subtable-hdr"><span id="tcv-obj-hdr">Active Magnetics &amp; Shaping</span><span>Solved</span></div>',
      '        <div class="tcv-kv-list" id="tcv-fuse-coils"></div>',
      '      </div>',
      '    </div>',
      '    <div class="tcv-fig-caption" id="tcv-fig-caption">',
      '      <strong>Figure 1.</strong> Coupled 3D cutaway magnetics (cryo-slate <em>B</em><sub>φ</sub> toroidal cage, diverging signed ±<em>I</em> poloidal shaping rings, volumetric <em>D</em><sub>α</sub> flux surfaces, and 3D helical <em>q</em><sub>95</sub> field lines), 2D equilibrium cross-sections, and 10,000-round closed-loop policy search trajectories across <em>n</em> = 10 paired seeds on <code>toka_gym</code> TCV rampdown (<em>D</em> = 136,712), alongside stationary equilibria exported from <code>FUSE.jl 1.2.1</code> (SPARC, ARC, ITER, DIII-D; Meneghini et al., 2024).',
      '    </div>',
      '  </div>',
      '</div>'
    ].join("\n");

    var glCanvas = root.querySelector("#tcv-webgl-canvas");
    var fig9Canvas = root.querySelector("#tcv-fig9-canvas");
    var kvList = root.querySelector("#tcv-kv-list");
    var fuseCoilsList = root.querySelector("#tcv-fuse-coils");
    var legendTele = root.querySelector("#tcv-legend-tele");
    var benchDock = root.querySelector("#tcv-bench-dock");
    var fuseDeck = root.querySelector("#tcv-fuse-deck");
    var figCaption = root.querySelector("#tcv-fig-caption");
    var secPolicy = root.querySelector("#tcv-sec-policy");
    var secShot = root.querySelector("#tcv-sec-shot");
    var secRound = root.querySelector("#tcv-sec-round");
    var secSeed = root.querySelector("#tcv-sec-seed");
    var secCutaway = root.querySelector("#tcv-sec-cutaway");
    var shotSlider = root.querySelector("#tcv-shot-slider");
    var shotLbl = root.querySelector("#tcv-shot-lbl");
    var shotStatus = root.querySelector("#tcv-shot-status");
    var roundSlider = root.querySelector("#tcv-round-slider");
    var roundLbl = root.querySelector("#tcv-round-lbl");
    var presetBtns = root.querySelectorAll(".tcv-preset-btn[data-round-preset]");
    var cutSlider = root.querySelector("#tcv-cut-slider");
    var cutLbl = root.querySelector("#tcv-cut-lbl");
    var seedSel = root.querySelector("#tcv-seed-sel");
    var machineSel = root.querySelector("#tcv-machine-sel");
    var playBtn = root.querySelector("#tcv-play-btn");
    var chkCoils = root.querySelector("#tcv-chk-coils");
    var chkTF = root.querySelector("#tcv-chk-tf");
    var chkField = root.querySelector("#tcv-chk-field");
    var rowTF = root.querySelector("#tcv-row-tf");
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
      policyKey: "turbo0",
      round: 10000,
      shotU: 0.55,
      fieldPhase: 0.0,
      regime: "full",
      seedIdx: -1,
      showCoils: true,
      showTF: true,
      showFieldLines: true,
      cutawayDeg: 56.0,
      yaw: -Math.PI * 0.5,
      pitch: 0.46,
      dist: initObj.dist,
      playing: false
    };
    var hasManualShotScrub = false;

    function formatRoundInt(r) {
      return String(Math.round(r)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    function roundToSliderVal(r, regime) {
      var logMin = (regime === "tail") ? 2.0 : 0.0;
      var logSpan = 4.0 - logMin;
      var lr = Math.log(Math.max(1.0, r)) / Math.LN10;
      var u = (lr - logMin) / logSpan;
      return Math.round(Math.max(0.0, Math.min(1.0, u)) * 1000.0);
    }

    function sliderValToRound(sVal, regime) {
      var logMin = (regime === "tail") ? 2.0 : 0.0;
      var logSpan = 4.0 - logMin;
      var u = Math.max(0.0, Math.min(1.0, sVal / 1000.0));
      var minR = (regime === "tail") ? 100 : 1;
      return Math.max(minR, Math.min(10000, Math.round(Math.pow(10.0, logMin + u * logSpan))));
    }

    // Dock figure[data-bench-fig="tokamak-tokagym"] into #tcv-bench-dock and wire two-way live coupling
    var dockedFig = document.querySelector('figure[data-bench-fig="tokamak-tokagym"]');
    var dockedSvgs = [];
    var dockedScrubBars = [];
    var dockedSeedDots = [];
    var armNameToKey = {
      "TuRBO-0 (R=10)": "turbo0",
      "ENN · Wasserstein": "enn_w",
      "ENN · Euclidean": "enn_e",
      "ENN · Hyperbolic": "enn_h",
      "ENN · Conformal": "enn_c"
    };

    if (dockedFig && benchDock) {
      dockedFig.setAttribute("data-bench-ready", "1");
      var origParent = dockedFig.parentElement;
      benchDock.appendChild(dockedFig);
      dockedFig.classList.add("is-docked");
      if (origParent && origParent.tagName === "SECTION" && origParent.textContent.replace(/\s+/g, "") === "") {
        origParent.style.display = "none";
      }
      var innerCap = dockedFig.querySelector("figcaption");
      if (innerCap) innerCap.style.display = "none";

      var benchHdrs = dockedFig.querySelectorAll(".baux-bench-hdr > span:first-child");
      if (benchHdrs.length >= 1) {
        benchHdrs[0].innerHTML = "<strong>(c)</strong> Anytime Tracking Error (↓)";
      }
      if (benchHdrs.length >= 2) {
        benchHdrs[1].innerHTML = "<strong>(d)</strong> Paired Differences vs. R=10 (<em>n</em> = 10)";
      }

      // Upgrade ROUNDS & BENCH from all 47 empirical knots in data-scrub
      var rawScrub = dockedFig.getAttribute("data-scrub") || "";
      var tildeIdx = rawScrub.indexOf("~");
      if (tildeIdx > 0) {
        var knotParts = rawScrub.slice(tildeIdx + 1).split(";");
        var parsedR = [];
        var parsedM = [[], [], [], [], []];
        var parsedS = [[], [], [], [], []];
        for (var kp = 0; kp < knotParts.length; kp++) {
          var kv = knotParts[kp].split(":");
          if (kv.length !== 2) continue;
          var kr = parseInt(kv[0], 10);
          var nums = kv[1].split(",");
          if (!Number.isFinite(kr) || nums.length < 10) continue;
          var okRow = true;
          for (var ni = 0; ni < 10; ni++) {
            if (!Number.isFinite(parseFloat(nums[ni]))) { okRow = false; break; }
          }
          if (!okRow) continue;
          parsedR.push(kr);
          for (var ai = 0; ai < 5; ai++) {
            parsedM[ai].push(parseFloat(nums[ai * 2]));
            parsedS[ai].push(parseFloat(nums[ai * 2 + 1]));
          }
        }
        if (parsedR.length >= 10) {
          ROUNDS = parsedR;
          for (var ak = 0; ak < BENCH_ORDER.length; ak++) {
            var bKey = BENCH_ORDER[ak];
            BENCH[bKey].mean = parsedM[ak];
            BENCH[bKey].sem = parsedS[ak];
          }
        }
      }

      var sBarNodes = dockedFig.querySelectorAll(".baux-scrub-bar");
      for (var sbi = 0; sbi < sBarNodes.length; sbi++) dockedScrubBars.push(sBarNodes[sbi]);

      var svgNodes = dockedFig.querySelectorAll("svg[data-view]");
      for (var svi = 0; svi < svgNodes.length; svi++) {
        var svView = (svgNodes[svi].getAttribute("data-view") || "").trim();
        if (svView === "full" || svView === "tail") dockedSvgs.push(svgNodes[svi]);
      }

      var dotNodes = dockedFig.querySelectorAll(".baux-seed-dot[data-seed]");
      for (var di = 0; di < dotNodes.length; di++) {
        dockedSeedDots.push({
          el: dotNodes[di],
          seed: parseInt(dotNodes[di].getAttribute("data-seed") || "-1", 10),
          arm: (dotNodes[di].getAttribute("data-arm") || "").trim(),
          summary: (dotNodes[di].getAttribute("data-summary") || "").trim()
        });
      }
    }

    function setActivePolBtn(polKey) {
      var btns = root.querySelectorAll(".tcv-pol-btn[data-pol]");
      for (var bi = 0; bi < btns.length; bi++) {
        if (btns[bi].getAttribute("data-pol") === polKey) btns[bi].classList.add("active");
        else btns[bi].classList.remove("active");
      }
    }

    function syncDockedBench(st) {
      if (!dockedFig) return;
      // 1. Position scrub cursor on active SVG
      for (var si = 0; si < dockedSvgs.length; si++) {
        var svgEl = dockedSvgs[si];
        var vName = svgEl.getAttribute("data-view");
        var lines = svgEl.querySelectorAll(".baux-scrub-line");
        if (vName !== uiState.regime) {
          for (var li0 = 0; li0 < lines.length; li0++) lines[li0].setAttribute("opacity", "0");
          continue;
        }
        var xLeft = parseFloat(svgEl.getAttribute("data-x-left") || "48");
        var xRight = parseFloat(svgEl.getAttribute("data-x-right") || "326");
        var logMin = parseFloat(svgEl.getAttribute("data-log-min") || "0");
        var logMax = parseFloat(svgEl.getAttribute("data-log-max") || "4");
        var lr = Math.log(Math.max(1.0, st.round)) / Math.LN10;
        if (lr < logMin - 0.01 || lr > logMax + 0.01) {
          for (var li1 = 0; li1 < lines.length; li1++) lines[li1].setAttribute("opacity", "0");
        } else {
          var u = Math.max(0.0, Math.min(1.0, (lr - logMin) / Math.max(1e-6, logMax - logMin)));
          var xPos = (xLeft + u * (xRight - xLeft)).toFixed(2);
          for (var li2 = 0; li2 < lines.length; li2++) {
            lines[li2].setAttribute("x1", xPos);
            lines[li2].setAttribute("x2", xPos);
            lines[li2].setAttribute("opacity", "1");
          }
        }
      }

      // 2. Highlight selected paired seed dots in (d)
      var matchedSeedSummary = "";
      for (var di = 0; di < dockedSeedDots.length; di++) {
        var d = dockedSeedDots[di];
        if (uiState.seedIdx < 0) {
          d.el.classList.remove("is-active");
          d.el.classList.remove("is-dimmed");
          d.el.setAttribute("r", "2.9");
        } else if (d.seed === uiState.seedIdx) {
          d.el.classList.add("is-active");
          d.el.classList.remove("is-dimmed");
          var dPol = Object.prototype.hasOwnProperty.call(armNameToKey, d.arm) ? armNameToKey[d.arm] : "";
          d.el.setAttribute("r", (dPol === uiState.policyKey) ? "4.8" : "4.1");
          if (dPol === uiState.policyKey && d.summary) matchedSeedSummary = d.summary;
        } else {
          d.el.classList.add("is-dimmed");
          d.el.classList.remove("is-active");
          d.el.setAttribute("r", "2.9");
        }
      }

      // 3. Format synchronized scrub readout bar across all 5 optimizer arms
      var barTxt = "";
      if (uiState.seedIdx >= 0 && st.round === 10000 && matchedSeedSummary) {
        barTxt = "Round t = 10,000  ·  " + matchedSeedSummary + "  ·  ΔLCFS: " + st.lcfsErrMm.toFixed(1) + " mm";
      } else {
        var segs = ["Round t = " + formatRoundInt(st.round)];
        if (uiState.seedIdx >= 0) segs.push("Seed " + (uiState.seedIdx + 1));
        for (var bi = 0; bi < BENCH_ORDER.length; bi++) {
          var bk = BENCH_ORDER[bi];
          var bSt = (bk === uiState.policyKey) ? st : evalState(bk, st.round, uiState.seedIdx, "tcv", uiState.shotU);
          if (uiState.seedIdx >= 0) {
            segs.push(BENCH[bk].short + ": " + bSt.err.toFixed(3));
          } else {
            segs.push(BENCH[bk].short + ": " + bSt.err.toFixed(3) + " ± " + bSt.sem.toFixed(3));
          }
        }
        barTxt = segs.join("  ·  ");
      }
      for (var sbi = 0; sbi < dockedScrubBars.length; sbi++) {
        dockedScrubBars[sbi].textContent = barTxt;
      }
    }

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

    function syncRoundPresets(r) {
      for (var pi = 0; pi < presetBtns.length; pi++) {
        var pr = parseInt(presetBtns[pi].getAttribute("data-round-preset") || "0", 10);
        var isMatch = (pr === 1 && r <= 2) || (pr === 10 && r >= 6 && r <= 20) || (pr === 10000 && r >= 5000);
        if (isMatch) presetBtns[pi].classList.add("active");
        else presetBtns[pi].classList.remove("active");
      }
    }

    function renderAll() {
      var isTCV = (uiState.machine === "tcv");
      var mach = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      var st = evalState(uiState.policyKey, uiState.round, uiState.seedIdx, uiState.machine, uiState.shotU);

      if (builderTag) builderTag.textContent = isTCV ? "toka_gym" : "FUSE.jl";
      if (rowTF) rowTF.style.display = "inline-flex";
      if (secPolicy) secPolicy.style.display = isTCV ? "" : "none";
      if (secShot) secShot.style.display = isTCV ? "" : "none";
      if (secRound) secRound.style.display = isTCV ? "" : "none";
      if (secSeed) secSeed.style.display = isTCV ? "" : "none";
      if (secCutaway) secCutaway.style.display = isTCV ? "none" : "";

      if (isTCV) {
        if (shotLbl) shotLbl.textContent = "τ = " + st.tauMs + " ms";
        if (shotSlider && Math.abs(parseFloat(shotSlider.value) - st.tauMs) >= 1.5) {
          shotSlider.value = String(st.tauMs);
        }
        if (shotStatus) {
          shotStatus.textContent = st.statusText;
          shotStatus.className = "tcv-shot-status is-" + st.statusKind;
        }
        syncRoundPresets(st.round);
      }

      var showDocked = isTCV && !!dockedFig;
      if (benchDock) benchDock.style.display = showDocked ? "" : "none";
      if (fuseDeck) fuseDeck.style.display = showDocked ? "none" : "";

      if (badgeA) {
        if (isTCV) {
          badgeA.innerHTML = "<strong>(a)</strong> 3D Cutaway · TCV <span class=\"tcv-badge-sub\">τ = " + escHtml(String(st.tauMs)) + " ms</span>";
        } else {
          badgeA.innerHTML = "<strong>(a)</strong> 3D Cutaway CAD — " + escHtml(mach.shortName);
        }
      }
      if (badgeB) {
        if (isTCV) {
          var bSub = st.disrupted
            ? ("⚡ Wall Strike · Ip " + Math.round(st.ipKA) + " kA")
            : ("Ip " + Math.round(st.ipKA) + " kA · Wtot " + (st.wtotKJ < 10 ? st.wtotKJ.toFixed(1) : Math.round(st.wtotKJ)) + " kJ");
          badgeB.innerHTML = "<strong>(b)</strong> Equilibrium &amp; Flux — <span class=\"tcv-badge-sub" + (st.disrupted ? " is-crash" : "") + "\">" + escHtml(bSub) + "</span>";
        } else {
          badgeB.innerHTML = "<strong>(b)</strong> Equilibrium &amp; Poloidal Flux — " + escHtml(mach.shortName);
        }
      }

      if (legendTele) {
        if (isTCV) {
          var lossStr = st.err.toFixed(3) + (uiState.seedIdx < 0 ? (" ± " + st.sem.toFixed(3)) : (" (S" + (uiState.seedIdx + 1) + ")"));
          legendTele.textContent = "τ = " + st.tauMs + " ms · t = " + formatRoundInt(st.round) + " · Loss " + lossStr + " · ΔLCFS " + st.lcfsErrMm.toFixed(1) + " mm · q95 " + st.q95.toFixed(2);
        } else {
          var pFusTxt = (mach.pFusMW && mach.pFusMW > 0.5) ? (" · Pfus " + mach.pFusMW.toFixed(0) + " MW") : "";
          legendTele.textContent = "Ip " + mach.ipNominal.toFixed(2) + " MA · B0 " + mach.B0.toFixed(2) + " T · q95 " + (mach.q95 || 3.2).toFixed(2) + pFusTxt;
        }
      }

      if (figCaption) {
        if (isTCV) {
          figCaption.innerHTML = '<strong>Figure 1.</strong> Coupled 3D cutaway magnetics (cryo-slate <em>B</em><sub>φ</sub> toroidal cage, diverging signed ±<em>I</em> poloidal shaping rings, volumetric <em>D</em><sub>α</sub> flux surfaces, and 3D helical <em>q</em><sub>95</sub> field lines), 2D equilibrium cross-sections, and 10,000-round closed-loop policy search trajectories across <em>n</em> = 10 paired seeds on <code>toka_gym</code> TCV rampdown (<em>D</em> = 136,712; Wang et al., 2025; Theiler et al., 2026).';
        } else {
          figCaption.innerHTML = '<strong>Figure 1.</strong> Stationary 3D cutaway CAD magnetics (discrete <em>B</em><sub>φ</sub> toroidal-field D-coils, signed ±<em>I</em> central-solenoid and poloidal-field shaping rings, double-wall vacuum vessel, and 3D helical <em>q</em><sub>95</sub> field lines) and solved 2D Grad–Shafranov equilibrium exported from <code>' + escHtml(mach.solver || "FUSE.jl 1.2.1") + '</code> for <strong>' + escHtml(mach.name || mach.shortName) + '</strong> (Meneghini et al., 2024).';
        }
      }

      if (!showDocked && kvList && fuseCoilsList) {
        if (subtableTitle) subtableTitle.textContent = isTCV ? "TCV Rampdown State" : "FUSE.jl 1.2.1 Equilibrium";
        if (subtableDevice) subtableDevice.textContent = mach.shortName;
        if (objHdr) objHdr.textContent = isTCV ? "Closed-Loop Tracking Telemetry" : "Active Magnetics & Shaping";

        var qDisp = isTCV ? st.q95.toFixed(2) : (mach.q95 || 3.2).toFixed(2);
        var nCoilsTxt = (mach.coils ? mach.coils.length : 0) + " PF/CS + " + (mach.n_tf || 18) + " TF";
        var sixthLbl = isTCV ? "LCFS Contour Residual" : ((mach.pFusMW && mach.pFusMW > 0.5) ? "Fusion Power <em>P</em><sub>fus</sub>" : "Solver Backend");
        var sixthVal = isTCV ? (st.lcfsErrMm.toFixed(2) + " mm") : ((mach.pFusMW && mach.pFusMW > 0.5) ? (mach.pFusMW.toFixed(1) + " MW") : "EFIT / FUSE.jl");

        kvList.innerHTML = [
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">' + (isTCV ? 'Tracking Error (↓)' : 'Device / Regime') + '</span><span class="tcv-kv-val">' + escHtml(isTCV ? (st.err.toFixed(3) + ' ± ' + st.sem.toFixed(3)) : mach.shortName) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Plasma Current / <em>B</em><sub>0</sub></span><span class="tcv-kv-val">' + escHtml(mach.ipNominal.toFixed(2)) + ' MA · ' + escHtml(mach.B0.toFixed(2)) + ' T</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Safety Factor <em>q</em><sub>95</sub> / β<sub><em>N</em></sub></span><span class="tcv-kv-val">' + escHtml(qDisp) + ' · ' + escHtml((mach.betaN || 1.8).toFixed(2)) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Radii <em>R</em><sub>0</sub>, <em>a</em></span><span class="tcv-kv-val">' + escHtml(mach.R0.toFixed(2)) + ' m, ' + escHtml(mach.a.toFixed(2)) + ' m</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Magnet CAD Coils</span><span class="tcv-kv-val">' + escHtml(nCoilsTxt) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">' + sixthLbl + '</span><span class="tcv-kv-val">' + escHtml(sixthVal) + '</span></div>'
        ].join("");

        var negCnt = 0, posCnt = 0, peakMA = 0.0;
        if (mach.coils) {
          for (var mci = 0; mci < mach.coils.length; mci++) {
            var cv = signedCoilNorm(mach, mach.coils[mci], mci, st);
            if (cv < -0.04) negCnt++;
            else if (cv > 0.04) posCnt++;
            var cma = Math.abs(mach.coils[mci].current_ma || 0.0);
            if (cma > peakMA) peakMA = cma;
          }
        }
        var aspectRat = (mach.R0 / Math.max(0.1, mach.a)).toFixed(2);
        var solverShort = (mach.solver || "FUSE.jl 1.2.1").replace(/^FUSE\.jl\s*[0-9.]*\s*·\s*/, "").replace(/\s+Equilibrium$/i, "");
        fuseCoilsList.innerHTML = [
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Solver Pipeline</span><span class="tcv-kv-val">' + escHtml(solverShort) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Aspect Ratio <em>R</em><sub>0</sub> / <em>a</em></span><span class="tcv-kv-val">' + escHtml(aspectRat) + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Pull (−<em>I</em>) / Push (+<em>I</em>) Coils</span><span class="tcv-kv-val">' + escHtml(negCnt + " cobalt · " + posCnt + " amber") + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">Peak Winding Current |<em>I</em><sub>coil</sub>|</span><span class="tcv-kv-val">' + escHtml(peakMA > 0.01 ? (peakMA.toFixed(2) + " MA") : "Normalized [−1, 1]") + '</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">LCFS Boundary Vertices</span><span class="tcv-kv-val">' + escHtml(String(mach.lcfs ? mach.lcfs.length : 64)) + ' pts</span></div>',
          '<div class="tcv-kv-row"><span class="tcv-kv-lbl">3D Toroidal Cutaway Sector</span><span class="tcv-kv-val">' + escHtml(Math.round(uiState.cutawayDeg) + "° open") + '</span></div>'
        ].join("");
      }

      if (showDocked) {
        syncDockedBench(st);
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
        if (isDark) gl.clearColor(0.051, 0.071, 0.098, 1.0);
        else gl.clearColor(0.976, 0.980, 0.988, 1.0);
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
        var viewMat = mat4LookAt(eye, [0.0, (mach.Z0 || -0.04) * 0.2, 0.0], [0.0, 1.0, 0.0]);
        var mvp = mat4Mul(projMat, viewMat);
        var model = mat4Identity();

        gl.useProgram(prog);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uMVP"), false, mvp);
        gl.uniformMatrix4fv(gl.getUniformLocation(prog, "uModel"), false, model);
        gl.uniform3fv(gl.getUniformLocation(prog, "uCamPos"), new Float32Array(eye));

        var aPos = gl.getAttribLocation(prog, "aPos");
        var aNorm = gl.getAttribLocation(prog, "aNorm");
        var aCol = gl.getAttribLocation(prog, "aCol");
        var uUnlitLoc = gl.getUniformLocation(prog, "uUnlit");

        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.posBuf);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.normBuf);
        gl.enableVertexAttribArray(aNorm);
        gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ARRAY_BUFFER, cachedBuffers.colBuf);
        gl.enableVertexAttribArray(aCol);
        gl.vertexAttribPointer(aCol, 4, gl.FLOAT, false, 0, 0);

        var hwCount = cachedBuffers.hwTriCount || cachedBuffers.triCount;
        gl.uniform1f(uUnlitLoc, 0.0);
        gl.drawArrays(gl.TRIANGLES, 0, hwCount);

        if (cachedBuffers.triCount > hwCount) {
          gl.uniform1f(uUnlitLoc, 0.5);
          gl.drawArrays(gl.TRIANGLES, hwCount, cachedBuffers.triCount - hwCount);
        }

        if (cachedBuffers.lineCount > 0) {
          gl.uniform1f(uUnlitLoc, 1.0);
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

    // Hydrate real FUSE.jl 3D CAD environments from assets/baux_env.json (preserving signed coil fallback currents)
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
                var prevM = MACHINES[k];
                var nextM = fm[k];
                if (prevM && prevM.coils && nextM.coils && nextM.coils.length > 0) {
                  var hasNonZero = false;
                  for (var ci0 = 0; ci0 < nextM.coils.length; ci0++) {
                    if (typeof nextM.coils[ci0].current_ma === "number" && Math.abs(nextM.coils[ci0].current_ma) > 1e-4) {
                      hasNonZero = true;
                      break;
                    }
                  }
                  if (!hasNonZero) {
                    for (var ci1 = 0; ci1 < nextM.coils.length; ci1++) {
                      var nc = nextM.coils[ci1];
                      var matchedCur = null;
                      for (var pi = 0; pi < prevM.coils.length; pi++) {
                        if (prevM.coils[pi].name && nc.name && prevM.coils[pi].name.toLowerCase() === nc.name.toLowerCase()) {
                          matchedCur = prevM.coils[pi].current_ma;
                          break;
                        }
                      }
                      if (matchedCur === null) {
                        var ipRef = nextM.ipNominal || prevM.ipNominal || 2.0;
                        matchedCur = (nc.r < (nextM.R0 || prevM.R0)) ? (-0.52 * ipRef) : ((Math.abs(nc.z) > (nextM.a || 1.0) * 1.15) ? (0.36 * ipRef) : (-0.32 * ipRef));
                      }
                      nc.current_ma = matchedCur;
                    }
                  }
                }
                MACHINES[k] = nextM;
              }
            }
          }
          if (Array.isArray(data.runs)) {
            for (var ri = 0; ri < data.runs.length; ri++) {
              var rec = data.runs[ri];
              if (rec && rec.metrics && typeof rec.metrics.worker_s === "number") {
                if (rec.task === "fuse" && rec.metrics.worker_s > 0.01) {
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

    var animTimer = null;
    var holdTicks = 0;
    function stopPlayback() {
      if (uiState.playing) {
        uiState.playing = false;
        if (playBtn) {
          playBtn.textContent = "▶ Play 400 ms Shot";
          playBtn.classList.remove("active");
        }
      }
      if (animTimer) {
        clearInterval(animTimer);
        animTimer = null;
      }
    }

    function setRegimeView(reg) {
      if (reg !== "full" && reg !== "tail") return;
      uiState.regime = reg;
      if (dockedFig) {
        var rBtns = dockedFig.querySelectorAll(".baux-regime-btn[data-regime]");
        for (var j = 0; j < rBtns.length; j++) {
          var act = (rBtns[j].getAttribute("data-regime") || "").trim() === reg;
          if (act) rBtns[j].classList.add("active");
          else rBtns[j].classList.remove("active");
        }
      }
      for (var sv = 0; sv < dockedSvgs.length; sv++) {
        if (dockedSvgs[sv].getAttribute("data-view") === reg) dockedSvgs[sv].removeAttribute("hidden");
        else dockedSvgs[sv].setAttribute("hidden", "hidden");
      }
    }

    function setRoundAndSync(newRound) {
      var minR = (uiState.regime === "tail") ? 100 : 1;
      uiState.round = Math.max(minR, Math.min(10000, Math.round(newRound)));
      if (roundSlider) roundSlider.value = String(roundToSliderVal(uiState.round, uiState.regime));
      if (roundLbl) roundLbl.textContent = "t = " + formatRoundInt(uiState.round);
      if (!uiState.playing && uiState.shotU < 0.08) {
        uiState.shotU = 0.55;
      }
      meshDirty = true;
      renderAll();
    }

    // Wire docked figure regime pills, SVG scrubbers, and paired seed dots
    if (dockedFig) {
      var regBtns = dockedFig.querySelectorAll(".baux-regime-btn[data-regime]");
      for (var rbi = 0; rbi < regBtns.length; rbi++) {
        (function(btn) {
          btn.addEventListener("click", function() {
            var reg = (btn.getAttribute("data-regime") || "").trim();
            if (reg !== "full" && reg !== "tail") return;
            setRegimeView(reg);
            if (reg === "tail" && uiState.round < 100) {
              uiState.round = 100;
            }
            setRoundAndSync(uiState.round);
          });
        })(regBtns[rbi]);
      }

      for (var svi2 = 0; svi2 < dockedSvgs.length; svi2++) {
        (function(svgEl) {
          var xLeft = parseFloat(svgEl.getAttribute("data-x-left") || "48");
          var xRight = parseFloat(svgEl.getAttribute("data-x-right") || "326");
          var logMin = parseFloat(svgEl.getAttribute("data-log-min") || "0");
          var logMax = parseFloat(svgEl.getAttribute("data-log-max") || "4");
          var vbParts = (svgEl.getAttribute("viewBox") || "0 0 432 272").trim().split(/\s+/);
          var vbW = parseFloat(vbParts[2] || "432");

          function scrubAtClientX(clientX) {
            var rect = svgEl.getBoundingClientRect();
            if (rect.width <= 1.0) return;
            var svgX = ((clientX - rect.left) / rect.width) * vbW;
            var u = Math.max(0.0, Math.min(1.0, (svgX - xLeft) / Math.max(1.0, xRight - xLeft)));
            var targetT = Math.pow(10.0, logMin + u * (logMax - logMin));
            var bestR = Math.round(targetT);
            var bestDist = Infinity;
            for (var ki = 0; ki < ROUNDS.length; ki++) {
              var dLog = Math.abs(Math.log(ROUNDS[ki]) - Math.log(Math.max(1.0, targetT)));
              if (dLog < bestDist) {
                bestDist = dLog;
                bestR = ROUNDS[ki];
              }
            }
            setRoundAndSync(bestR);
          }

          svgEl.addEventListener("mousemove", function(ev) {
            if (uiState.playing) return;
            scrubAtClientX(ev.clientX);
          });
          svgEl.addEventListener("click", function(ev) {
            scrubAtClientX(ev.clientX);
          });
          svgEl.addEventListener("touchstart", function(ev) {
            if (ev.touches.length > 0) {
              scrubAtClientX(ev.touches[0].clientX);
            }
          }, { passive: true });
          svgEl.addEventListener("touchmove", function(ev) {
            if (ev.touches.length > 0) {
              scrubAtClientX(ev.touches[0].clientX);
            }
          }, { passive: true });
        })(dockedSvgs[svi2]);
      }

      for (var doti = 0; doti < dockedSeedDots.length; doti++) {
        (function(dObj) {
          dObj.el.addEventListener("click", function(ev) {
            ev.stopPropagation();
            var armKey = Object.prototype.hasOwnProperty.call(armNameToKey, dObj.arm) ? armNameToKey[dObj.arm] : uiState.policyKey;
            if (uiState.seedIdx === dObj.seed && uiState.policyKey === armKey) {
              uiState.seedIdx = -1;
            } else {
              uiState.seedIdx = (dObj.seed >= 0 && dObj.seed < 10) ? dObj.seed : -1;
              uiState.policyKey = armKey;
              setActivePolBtn(armKey);
            }
            if (seedSel) seedSel.value = String(uiState.seedIdx);
            meshDirty = true;
            renderAll();
          });
        })(dockedSeedDots[doti]);
      }
    }

    machineSel.addEventListener("change", function() {
      stopPlayback();
      var val = machineSel.value;
      uiState.machine = Object.prototype.hasOwnProperty.call(MACHINES, val) ? val : "tcv";
      var m = Object.prototype.hasOwnProperty.call(MACHINES, uiState.machine) ? MACHINES[uiState.machine] : MACHINES.tcv;
      uiState.dist = m.dist;
      uiState.yaw = -Math.PI * 0.5;
      uiState.pitch = 0.46;
      meshDirty = true;
      renderAll();
    });

    var polBtns = root.querySelectorAll(".tcv-pol-btn[data-pol]");
    for (var i = 0; i < polBtns.length; i++) {
      (function(btn) {
        btn.addEventListener("click", function() {
          var pk = btn.getAttribute("data-pol");
          uiState.policyKey = (pk && Object.prototype.hasOwnProperty.call(BENCH, pk)) ? pk : "turbo0";
          setActivePolBtn(uiState.policyKey);
          meshDirty = true;
          renderAll();
        });
      })(polBtns[i]);
    }

    if (shotSlider) {
      shotSlider.addEventListener("input", function() {
        stopPlayback();
        hasManualShotScrub = true;
        var msVal = parseFloat(shotSlider.value);
        uiState.shotU = Number.isFinite(msVal) ? Math.max(0.0, Math.min(1.0, msVal / 400.0)) : 0.55;
        uiState.fieldPhase = uiState.shotU * 8.0;
        meshDirty = true;
        renderAll();
      });
    }

    roundSlider.addEventListener("input", function() {
      var sVal = parseInt(roundSlider.value, 10);
      var r = sliderValToRound(Number.isFinite(sVal) ? sVal : 1000, uiState.regime);
      setRoundAndSync(r);
    });

    for (var pbi = 0; pbi < presetBtns.length; pbi++) {
      (function(pBtn) {
        pBtn.addEventListener("click", function() {
          var pr = parseInt(pBtn.getAttribute("data-round-preset") || "10000", 10);
          if (pr < 100 && uiState.regime === "tail") {
            setRegimeView("full");
          }
          setRoundAndSync(pr);
        });
      })(presetBtns[pbi]);
    }

    if (cutSlider) {
      cutSlider.addEventListener("input", function() {
        var cv = parseFloat(cutSlider.value);
        uiState.cutawayDeg = Number.isFinite(cv) ? Math.max(25.0, Math.min(86.0, cv)) : 56.0;
        if (cutLbl) cutLbl.textContent = Math.round(uiState.cutawayDeg) + "°";
        meshDirty = true;
        renderAll();
      });
    }

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
      uiState.yaw = -Math.PI * 0.5;
      uiState.pitch = 0.46;
      uiState.dist = m.dist;
      uiState.cutawayDeg = 56.0;
      if (cutSlider) cutSlider.value = "56";
      if (cutLbl) cutLbl.textContent = "56°";
      meshDirty = true;
      renderAll();
    });

    playBtn.addEventListener("click", function() {
      if (uiState.playing) {
        stopPlayback();
        return;
      }
      uiState.playing = true;
      playBtn.textContent = "⏸ Pause Shot";
      playBtn.classList.add("active");
      if (!hasManualShotScrub || uiState.shotU >= 0.98) {
        uiState.shotU = 0.0;
      }
      hasManualShotScrub = true;
      holdTicks = 0;
      meshDirty = true;
      renderAll();
      animTimer = setInterval(function() {
        uiState.fieldPhase += 0.065;
        if (uiState.shotU < 1.0) {
          uiState.shotU = Math.min(1.0, uiState.shotU + 0.0086);
        } else {
          holdTicks++;
          if (holdTicks >= 26) {
            holdTicks = 0;
            uiState.shotU = 0.0;
          }
        }
        meshDirty = true;
        renderAll();
      }, 33);
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

    if (typeof MutationObserver !== "undefined" && document.documentElement) {
      var themeObs = new MutationObserver(function() { renderAll(); });
      themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    }

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

(* --- Interactive Benchmark Figure Controller ([data-bench-fig]) --- *)
type bench_seed_dot = {
  dot_el : element Js.t;
  dot_seed : string;
  dot_arm : string;
  dot_val : string;
  dot_summary : string;
}

let clamp_float v lo hi =
  Float.max lo (Float.min hi v)

let format_int_commas n =
  let neg = n < 0 in
  let s = Int.to_string (Int.abs n) in
  let len = String.length s in
  if len <= 3 then
    if neg then "-" ^ s else s
  else begin
    let buf = Buffer.create (len + (len - 1) / 3 + 1) in
    if neg then Buffer.add_char buf '-';
    for i = 0 to len - 1 do
      if i > 0 && (len - i) mod 3 = 0 then Buffer.add_char buf ',';
      Buffer.add_char buf s.[i]
    done;
    Buffer.contents buf
  end

let format_bench_metric v =
  if not (Float.is_finite v) then "\u{2014}"
  else
    let av = Float.abs v in
    if av = 0.0 then "0.000"
    else if av < 0.001 then Printf.sprintf "%.2e" v
    else if av < 1000.0 then Printf.sprintf "%.3f" v
    else Printf.sprintf "%.1f" v

let parse_finite_float raw =
  match float_of_string_opt (String.trim raw) with
  | Some v when Float.is_finite v -> Some v
  | _ -> None

let parse_viewbox_width raw =
  let normalized =
    String.map (function ',' | '\t' | '\n' | '\r' -> ' ' | c -> c) raw
  in
  let toks =
    String.split_on_char ' ' normalized
    |> List.filter (fun t -> t <> "")
  in
  match List.nth_opt toks 2 with
  | None -> None
  | Some w_str ->
      (match parse_finite_float w_str with
       | Some w when w > 0.0 -> Some w
       | _ -> None)

let parse_bench_scrub (raw : string) : (string array * (int * (float * float) array) array) option =
  let raw = String.trim raw in
  if raw = "" then None
  else
    match String.index_opt raw '~' with
    | None -> None
    | Some tilde_idx ->
        let header = String.sub raw 0 tilde_idx in
        let body = String.sub raw (tilde_idx + 1) (String.length raw - tilde_idx - 1) in
        let labels =
          String.split_on_char '|' header
          |> List.map String.trim
          |> List.filter (fun s -> s <> "")
          |> Array.of_list
        in
        if Array.length labels = 0 then None
        else
          let parse_knot seg =
            let seg = String.trim seg in
            if seg = "" then None
            else
              match String.index_opt seg ':' with
              | None -> None
              | Some colon_idx ->
                  let r_str = String.trim (String.sub seg 0 colon_idx) in
                  let vals_str =
                    String.sub seg (colon_idx + 1) (String.length seg - colon_idx - 1)
                  in
                  (match int_of_string_opt r_str with
                   | Some round when round > 0 ->
                       let raw_toks =
                         String.split_on_char ',' vals_str
                         |> List.map String.trim
                         |> List.filter (fun t -> t <> "")
                       in
                       let rec parse_vals acc = function
                         | [] -> Some (Array.of_list (List.rev acc))
                         | tok :: rest ->
                             (match parse_finite_float tok with
                              | Some v -> parse_vals (v :: acc) rest
                              | None -> None)
                       in
                       (match parse_vals [] raw_toks with
                        | Some nums when Array.length nums >= 2 && Array.length nums mod 2 = 0 ->
                            let n_pairs = Array.length nums / 2 in
                            let pairs =
                              Array.init n_pairs (fun i -> (nums.(2 * i), nums.(2 * i + 1)))
                            in
                            Some (round, pairs)
                        | _ -> None)
                   | _ -> None)
          in
          let knots =
            String.split_on_char ';' body
            |> List.filter_map parse_knot
            |> Array.of_list
          in
          if Array.length knots = 0 then None
          else Some (labels, knots)

let format_scrub_readout (labels : string array) (round : int) (pairs : (float * float) array) =
  let head = "Round t = " ^ format_int_commas round in
  let n_show = Int.min 3 (Int.min (Array.length labels) (Array.length pairs)) in
  if n_show <= 0 then head
  else
    let parts = ref [] in
    for i = n_show - 1 downto 0 do
      let (m, s) = pairs.(i) in
      let item =
        Printf.sprintf "%s: %s\u{00B1}%s"
          labels.(i)
          (format_bench_metric m)
          (format_bench_metric s)
      in
      parts := item :: !parts
    done;
    String.concat "  \u{00B7}  " (head :: !parts)

let is_node_within_el (target_el : element Js.t) (cand : element Js.t) =
  let cand_node : Dom.node Js.t = (cand :> Dom.node Js.t) in
  let rec climb cur_opt =
    match Js.Opt.to_option cur_opt with
    | None -> false
    | Some n ->
        if n == cand_node then true
        else climb n##.parentNode
  in
  climb (Js.some (target_el :> Dom.node Js.t))

let mount_bench_figure (fig_el : element Js.t) =
  if get_attr fig_el "data-bench-ready" <> "1" then begin
    fig_el##setAttribute (js "data-bench-ready") (js "1");

    let scrub_bars = query_within fig_el ".baux-scrub-bar" in
    let default_summary =
      let s = get_attr fig_el "data-summary" in
      if s <> "" then s
      else
        match scrub_bars with
        | bar :: _ ->
            (match Js.Opt.to_option bar##.textContent with
             | Some t -> Js.to_string t
             | None -> "")
        | [] -> ""
    in
    let set_scrub_bar txt =
      List.iter (fun bar ->
        bar##.textContent := Js.some (js txt)
      ) scrub_bars
    in

    let all_scrub_lines = query_within fig_el ".baux-scrub-line" in
    let hide_scrub_lines lines =
      List.iter (fun ln ->
        ln##setAttribute (js "opacity") (js "0")
      ) lines
    in

    (* Collect paired seed dots in Pane B *)
    let seed_dots : bench_seed_dot list =
      query_within fig_el ".baux-seed-dot[data-seed]"
      |> List.filter_map (fun dot_el ->
           let seed = String.trim (get_attr dot_el "data-seed") in
           if seed = "" then None
           else
             let arm = String.trim (get_attr dot_el "data-arm") in
             let v_raw = String.trim (get_attr dot_el "data-delta") in
             let v = if v_raw <> "" then v_raw else String.trim (get_attr dot_el "data-value") in
             let summary = String.trim (get_attr dot_el "data-summary") in
             Some {
               dot_el;
               dot_seed = seed;
               dot_arm = arm;
               dot_val = v;
               dot_summary = summary;
             })
    in

    let pinned_seed : (string * string) option ref = ref None in

    let apply_seed_highlight seed_id =
      List.iter (fun d ->
        if d.dot_seed = seed_id then begin
          d.dot_el##.classList##add (js "is-active");
          d.dot_el##.classList##remove (js "is-dimmed");
          d.dot_el##setAttribute (js "r") (js "4.3")
        end else begin
          d.dot_el##.classList##add (js "is-dimmed");
          d.dot_el##.classList##remove (js "is-active");
          d.dot_el##setAttribute (js "r") (js "2.9")
        end
      ) seed_dots
    in

    let clear_seed_classes () =
      List.iter (fun d ->
        d.dot_el##.classList##remove (js "is-active");
        d.dot_el##.classList##remove (js "is-dimmed");
        d.dot_el##setAttribute (js "r") (js "2.9")
      ) seed_dots
    in

    let reset_all_to_default () =
      pinned_seed := None;
      clear_seed_classes ();
      hide_scrub_lines all_scrub_lines;
      set_scrub_bar default_summary
    in

    (* 1. Regime Toggle Buttons (.baux-regime-btn[data-regime="full|tail"]) *)
    let regime_btns = query_within fig_el ".baux-regime-btn[data-regime]" in
    let traj_svgs =
      query_within fig_el "svg[data-view]"
      |> List.filter (fun svg_el ->
           let v = String.trim (get_attr svg_el "data-view") in
           v = "full" || v = "tail")
    in
    List.iter (fun btn ->
      add_click btn (fun () ->
        let target_regime = String.trim (get_attr btn "data-regime") in
        if target_regime = "full" || target_regime = "tail" then begin
          List.iter (fun sib ->
            let is_active = (String.trim (get_attr sib "data-regime") = target_regime) in
            if is_active then sib##.classList##add (js "active")
            else sib##.classList##remove (js "active");
            if Js.Opt.test (sib##getAttribute (js "aria-pressed")) then
              sib##setAttribute (js "aria-pressed") (js (if is_active then "true" else "false"))
          ) regime_btns;
          List.iter (fun svg_el ->
            let view = String.trim (get_attr svg_el "data-view") in
            if view = target_regime then
              svg_el##removeAttribute (js "hidden")
            else if view = "full" || view = "tail" then
              svg_el##setAttribute (js "hidden") (js "hidden")
          ) traj_svgs;
          reset_all_to_default ()
        end
      )
    ) regime_btns;

    (* 2. Interactive Trajectory Round Scrubber *)
    (match parse_bench_scrub (get_attr fig_el "data-scrub") with
     | None -> ()
     | Some (labels, knots) ->
         List.iter (fun svg_el ->
           let x_left_opt = parse_finite_float (get_attr svg_el "data-x-left") in
           let x_right_opt = parse_finite_float (get_attr svg_el "data-x-right") in
           let log_min_opt = parse_finite_float (get_attr svg_el "data-log-min") in
           let log_max_opt = parse_finite_float (get_attr svg_el "data-log-max") in
           let vb_w_opt = parse_viewbox_width (get_attr svg_el "viewBox") in
           match (x_left_opt, x_right_opt, log_min_opt, log_max_opt, vb_w_opt) with
           | (Some x_left, Some x_right, Some log_min, Some log_max, Some vb_width)
             when x_right > x_left && log_max > log_min ->
               let svg_lines = query_within svg_el ".baux-scrub-line" in
               let t_min = (10.0 ** log_min) *. 0.99 in
               let t_max = (10.0 ** log_max) *. 1.01 in
               let restore_after_scrub () =
                 hide_scrub_lines svg_lines;
                 match !pinned_seed with
                 | Some (s, sum_txt) ->
                     apply_seed_highlight s;
                     set_scrub_bar sum_txt
                 | None ->
                     set_scrub_bar default_summary
               in
               let scrub_at_client_x client_x =
                 if Float.is_finite client_x then begin
                   let rect = svg_el##getBoundingClientRect in
                   let rect_left = Js.float_of_number rect##.left in
                   let rect_width = Float.max 1.0 (Js.float_of_number rect##.width) in
                   let svg_x = (client_x -. rect_left) /. rect_width *. vb_width in
                   if (not (Float.is_finite svg_x))
                      || svg_x < x_left -. 6.0
                      || svg_x > x_right +. 6.0
                   then
                     restore_after_scrub ()
                   else begin
                     let u = clamp_float ((svg_x -. x_left) /. (x_right -. x_left)) 0.0 1.0 in
                     let log_t = log_min +. u *. (log_max -. log_min) in
                     let target_t = 10.0 ** log_t in
                     let log_target = Float.log10 target_t in
                     let best = ref None in
                     let best_dist = ref Float.infinity in
                     for i = 0 to Array.length knots - 1 do
                       let (r, pairs) = knots.(i) in
                       let r_flt = float_of_int r in
                       if r_flt >= t_min && r_flt <= t_max then begin
                         let d = Float.abs (Float.log10 r_flt -. log_target) in
                         if d < !best_dist then begin
                           best_dist := d;
                           best := Some (r, pairs)
                         end
                       end
                     done;
                     match !best with
                     | None -> restore_after_scrub ()
                     | Some (knot_round, knot_pairs) ->
                         let log_r = Float.log10 (float_of_int knot_round) in
                         let raw_snap_x =
                           x_left +. ((log_r -. log_min) /. (log_max -. log_min)) *. (x_right -. x_left)
                         in
                         let snap_x = clamp_float raw_snap_x x_left x_right in
                         let x_attr = Printf.sprintf "%.1f" snap_x in
                         List.iter (fun ln ->
                           ln##setAttribute (js "x1") (js x_attr);
                           ln##setAttribute (js "x2") (js x_attr);
                           ln##setAttribute (js "opacity") (js "0.85")
                         ) svg_lines;
                         set_scrub_bar (format_scrub_readout labels knot_round knot_pairs)
                   end
                 end
               in
               let on_mousemove = handler (fun (ev : mouseEvent Js.t) ->
                 scrub_at_client_x (Js.float_of_number ev##.clientX);
                 Js._true
               ) in
               let on_mouseleave = handler (fun (_ev : mouseEvent Js.t) ->
                 restore_after_scrub ();
                 Js._true
               ) in
               let on_touch = handler (fun (ev : touchEvent Js.t) ->
                 (match Js.Optdef.to_option (ev##.touches##item 0) with
                  | Some touch -> scrub_at_client_x (Js.float_of_number touch##.clientX)
                  | None -> ());
                 Js._true
               ) in
               let _ = addEventListener svg_el Event.mousemove on_mousemove Js._false in
               let _ =
                 addEventListener svg_el
                   (Event.make "mouseleave" : mouseEvent Js.t Event.typ)
                   on_mouseleave Js._false
               in
               let _ =
                 addEventListenerWithOptions svg_el Event.touchstart ~passive:Js._true on_touch
               in
               let _ =
                 addEventListenerWithOptions svg_el Event.touchmove ~passive:Js._true on_touch
               in
               ()
           | _ -> ()
         ) traj_svgs);

    (* 3. Paired Common-Random-Number Seed Tracing (.baux-seed-dot[data-seed]) *)
    if seed_dots <> [] then begin
      let summary_for_dot (d : bench_seed_dot) =
        if d.dot_summary <> "" then d.dot_summary
        else
          let matching = List.filter (fun m -> m.dot_seed = d.dot_seed) seed_dots in
          let arm_parts =
            List.filter_map (fun m ->
              if m.dot_arm <> "" && m.dot_val <> "" then
                Some (Printf.sprintf "%s: %s" m.dot_arm m.dot_val)
              else if m.dot_arm <> "" then
                Some m.dot_arm
              else if m.dot_summary <> "" then
                Some m.dot_summary
              else None
            ) matching
          in
          if arm_parts <> [] then
            Printf.sprintf "Seed %s  \u{00B7}  %s"
              d.dot_seed
              (String.concat "  \u{00B7}  " arm_parts)
          else default_summary
      in
      List.iter (fun d ->
        let on_enter = handler (fun (_ev : mouseEvent Js.t) ->
          hide_scrub_lines all_scrub_lines;
          apply_seed_highlight d.dot_seed;
          set_scrub_bar (summary_for_dot d);
          Js._true
        ) in
        let on_dot_click = handler (fun (_ev : mouseEvent Js.t) ->
          hide_scrub_lines all_scrub_lines;
          (match !pinned_seed with
           | Some (cur_s, _) when cur_s = d.dot_seed ->
               pinned_seed := None;
               clear_seed_classes ();
               set_scrub_bar default_summary
           | _ ->
               let sum_txt = summary_for_dot d in
               pinned_seed := Some (d.dot_seed, sum_txt);
               apply_seed_highlight d.dot_seed;
               set_scrub_bar sum_txt);
          Js._true
        ) in
        let _ =
          addEventListener d.dot_el
            (Event.make "mouseenter" : mouseEvent Js.t Event.typ)
            on_enter Js._false
        in
        let _ = addEventListener d.dot_el Event.click on_dot_click Js._false in
        ()
      ) seed_dots;

      let paired_svgs =
        query_within fig_el "svg[data-view=\"paired\"]"
      in
      let restore_seed_hover () =
        match !pinned_seed with
        | Some (s, sum_txt) ->
            apply_seed_highlight s;
            set_scrub_bar sum_txt
        | None ->
            clear_seed_classes ();
            set_scrub_bar default_summary
      in
      if paired_svgs <> [] then
        List.iter (fun p_svg ->
          let on_leave = handler (fun (_ev : mouseEvent Js.t) ->
            restore_seed_hover ();
            Js._true
          ) in
          let _ =
            addEventListener p_svg
              (Event.make "mouseleave" : mouseEvent Js.t Event.typ)
              on_leave Js._false
          in
          ()
        ) paired_svgs
      else
        List.iter (fun d ->
          let on_leave = handler (fun (_ev : mouseEvent Js.t) ->
            restore_seed_hover ();
            Js._true
          ) in
          let _ =
            addEventListener d.dot_el
              (Event.make "mouseleave" : mouseEvent Js.t Event.typ)
              on_leave Js._false
          in
          ()
        ) seed_dots;

      let on_doc_click = handler (fun (ev : mouseEvent Js.t) ->
        (match !pinned_seed with
         | None -> ()
         | Some _ ->
             let clicked_dot =
               match Js.Opt.to_option ev##.target with
               | None -> false
               | Some target_el ->
                   List.exists (fun d -> is_node_within_el target_el d.dot_el) seed_dots
             in
             if not clicked_dot then begin
               pinned_seed := None;
               clear_seed_classes ();
               set_scrub_bar default_summary
             end);
        Js._true
      ) in
      let _ = addEventListener document Event.click on_doc_click Js._false in
      ()
    end
  end

let init_bench_figures () =
  let figs = query_all "[data-bench-fig]" in
  List.iter mount_bench_figure figs

(* --- Main Entrypoint --- *)
let () =
  let booted = ref false in
  let boot () =
    if not !booted then begin
      booted := true;
      init_theme_toggle ();
      init_copy_buttons ();
      init_api_reference ();
      init_tokamak_cad ();
      init_bench_figures ()
    end
  in
  let on_load = handler (fun _ev ->
    boot ();
    Js._true
  ) in
  window##.onload := on_load;
  let state = Js.to_string (Js.Unsafe.get document (js "readyState")) in
  if state <> "loading" then boot ()
