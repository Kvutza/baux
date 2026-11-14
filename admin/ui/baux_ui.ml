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
      '      <strong>Figure 1.</strong> 3D cutaway geometries, 2D equilibrium and field cross-sections, and closed-loop rollout trajectories across tokamak models exported from <code>FUSE.jl</code> (TCV, SPARC, ARC, ITER, DIII-D; Meneghini et al., 2024).',
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
      yaw: -Math.PI * 0.5,
      pitch: 0.46,
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
      var st = evalState(uiState.policyKey, uiState.round, uiState.seedIdx);

      if (builderTag) builderTag.textContent = "FUSE.jl";
      if (lblCoils) lblCoils.textContent = "PF/CS Coils";
      if (lblField) lblField.textContent = "B-Field Lines";
      if (rowTF) rowTF.style.display = "inline-flex";

      if (badgeA) badgeA.innerHTML = "<strong>(a)</strong> 3D Cutaway CAD — " + escHtml(mach.shortName);
      if (badgeB) badgeB.innerHTML = "<strong>(b)</strong> Equilibrium & Flux — " + escHtml(mach.shortName);
      if (subtableTitle) subtableTitle.textContent = "FUSE.jl 1.2.1 Geometry";
      if (subtableDevice) subtableDevice.textContent = mach.shortName;
      if (objHdr) objHdr.textContent = "TORAX / FUSE SmoothMax (α = −3)";

      if (kvList) {
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

      if (wpoStrip) {
        var cLabels = ["LCFS / TEQUILA Shape", "Plasma Current <em>I<sub>p</sub></em>", "PF Coil Margin", "Actuators ∈ [−1, 1]", "Safety Factor <em>q</em><sub>95</sub>"];
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

    // Hydrate real FUSE.jl 3D CAD environments from assets/baux_env.json
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

    machineSel.addEventListener("change", function() {
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
      uiState.yaw = -Math.PI * 0.5;
      uiState.pitch = 0.46;
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
