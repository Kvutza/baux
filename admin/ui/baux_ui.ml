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
    | Some el ->
        let el_html : htmlElement Js.t = Js.Unsafe.coerce el in
        el_html##.innerText := js cmd
    | None -> ());
    (match Js.Opt.to_option install_details with
    | Some el ->
        let el_html : htmlElement Js.t = Js.Unsafe.coerce el in
        el_html##.innerHTML := js (Printf.sprintf "<span class=\"install-badge %s\">%s</span><span>%s</span>" badge badge desc)
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

(* --- Main Entrypoint --- *)
let () =
  let on_load = handler (fun _ev ->
    init_theme_toggle ();
    init_copy_buttons ();
    init_code_tabs ();
    init_install_switcher ();
    Js._true
  ) in
  window##.onload := on_load
