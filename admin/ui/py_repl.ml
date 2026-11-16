(* admin/ui/py_repl.ml
   Interactive in-browser Python REPL & inline runnable snippet controller.
   Talks to self-hosted stripped Pyodide via dedicated Web Worker (assets/repl_worker.js).
   Zero innerHTML / outerHTML / eval_string; pure DOM / createElementNS construction.
*)

[@@@warning "-32-27-33"]

open Js_of_ocaml
open Dom_html
open Py_presets

let js = Js.string
let to_s = Js.to_string
let svg_ns = "http://www.w3.org/2000/svg"




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

let is_defined (v : Js.Unsafe.any) : bool =
  Js.Optdef.test (Obj.magic v : _ Js.optdef) && Js.Opt.test (Obj.magic v : _ Js.opt)

let is_finite (v : float) : bool =
  match classify_float v with
  | FP_nan | FP_infinite -> false
  | _ -> true

let opt_str obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (to_s (Obj.magic v)) else None

let opt_int obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Some (int_of_float (Js.float_of_number (Obj.magic v))) else None

let opt_float obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then
    let fv = Js.float_of_number (Obj.magic v) in
    if is_finite fv then Some fv else None
  else None

let opt_bool obj field =
  let v = Js.Unsafe.get obj (js field) in
  if is_defined v then Js.to_bool (Obj.magic v : bool Js.t) else false

let json_stringify (v : Js.Unsafe.any) : string =
  let json_obj = Js.Unsafe.get Dom_html.window (js "JSON") in
  to_s (Js.Unsafe.meth_call json_obj "stringify" [| v |])

let json_parse (s : string) : Js.Unsafe.any option =
  try
    let json_obj = Js.Unsafe.get Dom_html.window (js "JSON") in
    Some (Js.Unsafe.meth_call json_obj "parse" [| Js.Unsafe.inject (js s) |])
  with _ -> None

(* ---------------------------------------------------------------------------
   Unified Runtime Controller (WASM + WebGPU Worker  <->  Local Native HTTP)
   --------------------------------------------------------------------------- *)

type runtime_mode = Wasm | LocalNative

type kernel_state =
  | Idle
  | Booting of string
  | Ready
  | Running
  | Interrupting of string
  | Interrupted of string

type cap_info = {
  mutable wasm_simd : bool;
  mutable wasm_simd_known : bool;
  mutable webgpu_available : bool;
  mutable webgpu_active : bool;
  mutable webgpu_adapter : string;
  mutable opfs_ready : bool;
  mutable opfs_mounted : bool;
  mutable coop_coep : bool;
  mutable native_checked : bool;
  mutable native_available : bool;
  mutable native_baux : bool;
  mutable native_cpu : bool;
  mutable native_metal : bool;
  mutable native_cuda : bool;
  mutable native_torch : bool;
  mutable native_jax : bool;
  mutable native_flame : bool;
  mutable native_tune : bool;
  mutable native_python : string;
  mutable native_reason : string;
}

let cap_ref : cap_info =
  {
    wasm_simd = false;
    wasm_simd_known = false;
    webgpu_available = false;
    webgpu_active = false;
    webgpu_adapter = "";
    opfs_ready = false;
    opfs_mounted = false;
    coop_coep = false;
    native_checked = false;
    native_available = false;
    native_baux = false;
    native_cpu = false;
    native_metal = false;
    native_cuda = false;
    native_torch = false;
    native_jax = false;
    native_flame = false;
    native_tune = false;
    native_python = "python3";
    native_reason = "Probing local kernel server...";
  }

let runtime_mode_ref : runtime_mode ref = ref Wasm
let worker_ref : Js.Unsafe.any option ref = ref None
let kernel_state_ref : kernel_state ref = ref Idle
let boot_step_ref = ref 0
let boot_total_ref = ref 4
let status_listeners : (kernel_state -> unit) list ref = ref []
let cap_listeners : (cap_info -> unit) list ref = ref []

let extract_playground_line (s : string) : int option =
  let needle = "File \"<playground>\", line " in
  let nlen = String.length needle in
  let slen = String.length s in
  let rec find_last idx best =
    if idx + nlen > slen then best
    else if String.sub s idx nlen = needle then
      let num_start = idx + nlen in
      let j = ref num_start in
      while !j < slen && s.[!j] >= '0' && s.[!j] <= '9' do
        incr j
      done;
      if !j > num_start then
        let num_str = String.sub s num_start (!j - num_start) in
        let parsed = try Some (int_of_string num_str) with _ -> None in
        find_last (!j) (if parsed <> None then parsed else best)
      else find_last (idx + 1) best
    else find_last (idx + 1) best
  in
  find_last 0 None
let pending_reqs : (int, Js.Unsafe.any -> unit) Hashtbl.t = Hashtbl.create 16
let next_req_id = ref 1
let sab_ref : Js.Unsafe.any option ref = ref None
let sab_int32_ref : Js.Unsafe.any option ref = ref None
let local_gen = ref 0
let pending_local_cbs : (int * (Js.Unsafe.any -> unit)) list ref = ref []

let notify_status st =
  kernel_state_ref := st;
  List.iter (fun fn -> fn st) !status_listeners

let add_status_listener fn =
  status_listeners := fn :: !status_listeners;
  fn !kernel_state_ref

let notify_caps () =
  List.iter (fun fn -> fn cap_ref) !cap_listeners

let add_cap_listener fn =
  cap_listeners := fn :: !cap_listeners;
  fn cap_ref

let is_cross_origin_isolated () : bool =
  let v = Js.Unsafe.get Dom_html.window (js "crossOriginIsolated") in
  is_defined v && Js.to_bool (Obj.magic v)

let ensure_interrupt_buffer () =
  let coi = is_cross_origin_isolated () in
  cap_ref.coop_coep <- coi;
  if not coi then None
  else
    match !sab_int32_ref with
    | Some _ as existing -> existing
    | None ->
        let sab_ctor = Js.Unsafe.get Dom_html.window (js "SharedArrayBuffer") in
        let i32_ctor = Js.Unsafe.get Dom_html.window (js "Int32Array") in
        let atomics = Js.Unsafe.get Dom_html.window (js "Atomics") in
        if is_defined sab_ctor && is_defined i32_ctor && is_defined atomics then
          try
            let sab = Js.Unsafe.new_obj sab_ctor [| Js.Unsafe.inject 4 |] in
            let view = Js.Unsafe.new_obj i32_ctor [| Js.Unsafe.inject sab |] in
            ignore
              (Js.Unsafe.meth_call atomics "store"
                 [| Js.Unsafe.inject view; Js.Unsafe.inject 0; Js.Unsafe.inject 0 |]);
            sab_ref := Some sab;
            sab_int32_ref := Some view;
            Some view
          with _ -> None
        else None

let set_sab_signal (sig_num : int) =
  match !sab_int32_ref with
  | Some view ->
      let atomics = Js.Unsafe.get Dom_html.window (js "Atomics") in
      if is_defined atomics then
        ignore
          (Js.Unsafe.meth_call atomics "store"
             [| Js.Unsafe.inject view; Js.Unsafe.inject 0; Js.Unsafe.inject sig_num |])
  | None -> ()

let fetch_json ~(meth : string) ~(url : string) ?body:(body : Js.Unsafe.any option)
    (cb : (int * string, string) result -> unit) =
  let fetch_fn = Js.Unsafe.get Dom_html.window (js "fetch") in
  if not (is_defined fetch_fn) then cb (Error "fetch API unavailable")
  else
    let base =
      [
        ("method", Js.Unsafe.inject (js meth));
        ("credentials", Js.Unsafe.inject (js "same-origin"));
      ]
    in
    let fields =
      match body with
      | Some b ->
          let hdrs = Js.Unsafe.obj [| ("Content-Type", Js.Unsafe.inject (js "application/json")) |] in
          base
          @ [
              ("headers", Js.Unsafe.inject hdrs);
              ("body", Js.Unsafe.inject (js (json_stringify b)));
            ]
      | None -> base
    in
    let init_obj = Js.Unsafe.obj (Array.of_list fields) in
    let promise =
      Js.Unsafe.meth_call Dom_html.window "fetch"
        [| Js.Unsafe.inject (js url); Js.Unsafe.inject init_obj |]
    in
    let on_resp =
      Js.wrap_callback (fun (resp : Js.Unsafe.any) ->
          let status =
            let raw_st = Js.Unsafe.get resp (js "status") in
            if is_defined raw_st then int_of_float (Js.float_of_number (Obj.magic raw_st)) else 0
          in
          let txt_p = Js.Unsafe.meth_call resp "text" [||] in
          let on_txt =
            Js.wrap_callback (fun (txt_js : Js.Unsafe.any) ->
                cb (Ok (status, to_s (Obj.magic txt_js))))
          in
          let on_txt_err =
            Js.wrap_callback (fun (err : Js.Unsafe.any) ->
                let msg = Option.value (opt_str err "message") ~default:"response read failed" in
                cb (Error msg))
          in
          ignore
            (Js.Unsafe.meth_call txt_p "then"
               [| Js.Unsafe.inject on_txt; Js.Unsafe.inject on_txt_err |]))
    in
    let on_err =
      Js.wrap_callback (fun (err : Js.Unsafe.any) ->
          let msg = Option.value (opt_str err "message") ~default:"network request failed" in
          cb (Error msg))
    in
    ignore (Js.Unsafe.meth_call promise "then" [| Js.Unsafe.inject on_resp; Js.Unsafe.inject on_err |])

let update_caps_from_worker (data : Js.Unsafe.any) =
  let changed = ref false in
  let set_b cur next =
    if next && not !cur then (
      cur := true;
      changed := true)
  in
  ignore set_b;
  let caps = Js.Unsafe.get data (js "caps") in
  if is_defined caps then (
    let simd = opt_bool caps "cpu_wasm_simd" || opt_bool caps "wasm_simd" in
    if (not cap_ref.wasm_simd_known) || simd <> cap_ref.wasm_simd then (
      cap_ref.wasm_simd <- simd;
      cap_ref.wasm_simd_known <- true;
      changed := true);
    let wg = opt_bool caps "webgpu" in
    if wg <> cap_ref.webgpu_active then (
      cap_ref.webgpu_active <- wg;
      if wg then cap_ref.webgpu_available <- true;
      changed := true);
    (match opt_str caps "webgpu_adapter" with
    | Some a when a <> "" && a <> cap_ref.webgpu_adapter ->
        cap_ref.webgpu_adapter <- a;
        changed := true
    | _ -> ());
    if opt_bool caps "opfs" || opt_bool caps "opfs_mounted" then (
      if not cap_ref.opfs_mounted then (
        cap_ref.opfs_mounted <- true;
        cap_ref.opfs_ready <- true;
        changed := true));
    let coi = opt_bool caps "coop_coep" || is_cross_origin_isolated () in
    if coi <> cap_ref.coop_coep then (
      cap_ref.coop_coep <- coi;
      changed := true));
  if opt_bool data "opfs_mounted" && not cap_ref.opfs_mounted then (
    cap_ref.opfs_mounted <- true;
    cap_ref.opfs_ready <- true;
    changed := true);
  if !changed then notify_caps ()

let probe_browser_caps () =
  cap_ref.coop_coep <- is_cross_origin_isolated ();
  let nav = Dom_html.window##.navigator in
  let gpu = Js.Unsafe.get nav (js "gpu") in
  cap_ref.webgpu_available <- is_defined gpu;
  let storage = Js.Unsafe.get nav (js "storage") in
  if is_defined storage && is_defined (Js.Unsafe.get storage (js "getDirectory")) then (
    try
      let p = Js.Unsafe.meth_call storage "getDirectory" [||] in
      let on_opfs_err =
        Js.wrap_callback (fun (_err : Js.Unsafe.any) ->
            cap_ref.opfs_ready <- false;
            notify_caps ())
      in
      let on_root =
        Js.wrap_callback (fun (root_dir : Js.Unsafe.any) ->
            if is_defined root_dir && is_defined (Js.Unsafe.get root_dir (js "getDirectoryHandle")) then
              let opts = Js.Unsafe.obj [| ("create", Js.Unsafe.inject Js._true) |] in
              let dp =
                Js.Unsafe.meth_call root_dir "getDirectoryHandle"
                  [| Js.Unsafe.inject (js "workspace"); Js.Unsafe.inject opts |]
              in
              let on_ws =
                Js.wrap_callback (fun (_ws : Js.Unsafe.any) ->
                    cap_ref.opfs_ready <- true;
                    notify_caps ())
              in
              ignore
                (Js.Unsafe.meth_call dp "then"
                   [| Js.Unsafe.inject on_ws; Js.Unsafe.inject on_opfs_err |]))
      in
      ignore
        (Js.Unsafe.meth_call p "then"
           [| Js.Unsafe.inject on_root; Js.Unsafe.inject on_opfs_err |])
    with _ -> ());
  notify_caps ()

let is_loopback_or_local_opt () : bool =
  try
    let loc = Dom_html.window##.location in
    let host = String.lowercase_ascii (to_s loc##.hostname) in
    let search = to_s loc##.search in
    host = "localhost" || host = "127.0.0.1" || host = "::1" || host = "[::1]"
    || contains_sub search "local=1"
  with _ -> false

let probe_local_native ?on_done:(on_done : (bool -> unit) option) () =
  if on_done = None && not (is_loopback_or_local_opt ()) then begin
    cap_ref.native_checked <- true;
    cap_ref.native_available <- false;
    cap_ref.native_reason <- "Static host — run './baux pages serve' locally";
    notify_caps ()
  end else
    let url = get_base_url () ^ "api/kernel/status" in
    fetch_json ~meth:"GET" ~url (fun res ->
      cap_ref.native_checked <- true;
      (match res with
      | Ok (status, txt) when status >= 200 && status < 300 -> (
          match json_parse txt with
          | Some obj when opt_bool obj "ok" && opt_bool obj "available" ->
              cap_ref.native_available <- true;
              cap_ref.native_python <-
                Option.value (opt_str obj "python") ~default:"python3";
              cap_ref.native_reason <- "Local Native Python kernel ready via baux pages serve";
              let caps = Js.Unsafe.get obj (js "capabilities") in
              if is_defined caps then (
                cap_ref.native_baux <- opt_bool caps "native_baux";
                cap_ref.native_cpu <- opt_bool caps "cpu";
                cap_ref.native_metal <- opt_bool caps "metal";
                cap_ref.native_cuda <- opt_bool caps "cuda";
                cap_ref.native_torch <- opt_bool caps "torch";
                cap_ref.native_jax <- opt_bool caps "jax";
                cap_ref.native_flame <- opt_bool caps "flame";
                cap_ref.native_tune <- opt_bool caps "baux_tune");
              notify_caps ();
              (match on_done with Some fn -> fn true | None -> ())
          | Some obj ->
              cap_ref.native_available <- false;
              cap_ref.native_reason <-
                Option.value (opt_str obj "error")
                  ~default:"Local kernel bridge unavailable";
              notify_caps ();
              (match on_done with Some fn -> fn false | None -> ())
          | None ->
              cap_ref.native_available <- false;
              cap_ref.native_reason <- "Static host (non-JSON /api/kernel/status)";
              notify_caps ();
              (match on_done with Some fn -> fn false | None -> ()))
      | Ok (status, _) ->
          cap_ref.native_available <- false;
          cap_ref.native_reason <-
            Printf.sprintf
              "Unavailable on static host (HTTP %d) — run './baux pages serve' locally" status;
          notify_caps ();
          (match on_done with Some fn -> fn false | None -> ())
      | Error msg ->
          cap_ref.native_available <- false;
          cap_ref.native_reason <- msg;
          notify_caps ();
          (match on_done with Some fn -> fn false | None -> ())))

let resolve_pending_error msg =
  let callbacks = Hashtbl.fold (fun _ v acc -> v :: acc) pending_reqs [] in
  Hashtbl.clear pending_reqs;
  let local_cbs = !pending_local_cbs in
  pending_local_cbs := [];
  let mk_err () =
    Js.Unsafe.obj
      [|
        ("ok", Js.Unsafe.inject Js._false);
        ("stdout", Js.Unsafe.inject (js ""));
        ("stderr", Js.Unsafe.inject (js msg));
        ("incomplete", Js.Unsafe.inject Js._false);
      |]
  in
  List.iter (fun cb -> cb (mk_err ())) callbacks;
  List.iter (fun (_, cb) -> cb (mk_err ())) local_cbs

let interrupt_kernel ?(force = false) () =
  match !runtime_mode_ref with
  | LocalNative ->
      if (not force) && !kernel_state_ref = Running then (
        notify_status (Interrupting "Interrupting...");
        let body = Js.Unsafe.obj [| ("force", Js.Unsafe.inject Js._false) |] in
        fetch_json ~meth:"POST"
          ~url:(get_base_url () ^ "api/kernel/interrupt")
          ~body (fun _ -> ()))
      else (
        local_gen := !local_gen + 1;
        let body = Js.Unsafe.obj [| ("force", Js.Unsafe.inject Js._true) |] in
        fetch_json ~meth:"POST"
          ~url:(get_base_url () ^ "api/kernel/interrupt")
          ~body (fun _ -> ());
        resolve_pending_error "Stopped.";
        notify_status (Interrupted "Stopped"))
  | Wasm ->
      let can_soft =
        (not force)
        && !kernel_state_ref = Running
        && is_cross_origin_isolated ()
        && !sab_int32_ref <> None
      in
      (match (!worker_ref, can_soft) with
      | Some w, true ->
          set_sab_signal 2;
          let msg = Js.Unsafe.obj [| ("type", Js.Unsafe.inject (js "interrupt")) |] in
          ignore (Js.Unsafe.meth_call w "postMessage" [| Js.Unsafe.inject msg |]);
          notify_status (Interrupting "Interrupting...")
      | Some w, false ->
          ignore (Js.Unsafe.meth_call w "terminate" [||]);
          worker_ref := None;
          set_sab_signal 0;
          resolve_pending_error "Stopped.";
          notify_status (Interrupted "Stopped")
      | None, _ ->
          resolve_pending_error "Stopped.";
          notify_status (Interrupted "Stopped"))

let ensure_worker () =
  match !worker_ref with
  | Some w -> w
  | None ->
      ignore (ensure_interrupt_buffer ());
      let worker_ctor = Js.Unsafe.get Dom_html.window (js "Worker") in
      let worker_rel =
        match Js.Opt.to_option (Dom_html.document##.documentElement##getAttribute (js "data-repl-worker")) with
        | Some v ->
            let s = to_s v in
            if s = "" then "assets/repl_worker.js" else s
        | None -> "assets/repl_worker.js"
      in
      let url = get_base_url () ^ worker_rel in
      let w = Js.Unsafe.new_obj worker_ctor [| Js.Unsafe.inject (js url) |] in
      notify_status (Booting "Loading...");
      let onmsg =
        Dom.handler (fun ev ->
            let data = Js.Unsafe.get ev (js "data") in
            update_caps_from_worker data;
            let typ =
              if Js.Optdef.test (Js.Unsafe.get data (js "type")) then
                to_s (Js.Unsafe.get data (js "type"))
              else ""
            in
            if typ = "status" then (
              if !runtime_mode_ref = Wasm then (
                let detail =
                  if Js.Optdef.test (Js.Unsafe.get data (js "detail")) then
                    to_s (Js.Unsafe.get data (js "detail"))
                  else ""
                in
                let stage =
                  if Js.Optdef.test (Js.Unsafe.get data (js "stage")) then
                    to_s (Js.Unsafe.get data (js "stage"))
                  else ""
                in
                (match opt_int data "step" with Some s -> boot_step_ref := s | None -> ());
                (match opt_int data "total_steps" with Some ts -> boot_total_ref := ts | None -> ());
                if stage = "ready" then (
                  boot_step_ref := !boot_total_ref;
                  notify_status Ready)
                else if stage = "interrupting" then notify_status (Interrupting detail)
                else if stage = "error" then notify_status (Interrupted detail)
                else notify_status (Booting detail)))
            else (
              let id_any = Js.Unsafe.get data (js "id") in
              if Js.Opt.test id_any then (
                let id_int = int_of_float (Js.float_of_number id_any) in
                match Hashtbl.find_opt pending_reqs id_int with
                | Some cb ->
                    Hashtbl.remove pending_reqs id_int;
                    (match !kernel_state_ref with
                    | Running | Interrupting _ when !runtime_mode_ref = Wasm -> notify_status Ready
                    | _ -> ());
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
            if !runtime_mode_ref = Wasm then
              notify_status (Interrupted ("Worker failed: " ^ msg));
            Js._true)
      in
      Js.Unsafe.set w (js "onmessage") onmsg;
      Js.Unsafe.set w (js "onerror") onerr;
      worker_ref := Some w;
      w

let send_local_native typ fields cb =
  let url_and_body =
    let obj = Js.Unsafe.obj (Array.of_list fields) in
    match typ with
    | "exec" -> Some (get_base_url () ^ "api/kernel/exec", Some obj, true)
    | "repl" -> Some (get_base_url () ^ "api/kernel/repl", Some obj, true)
    | "complete" -> Some (get_base_url () ^ "api/kernel/complete", Some obj, false)
    | "reset_ns" -> Some (get_base_url () ^ "api/kernel/reset", Some (Js.Unsafe.obj [||]), false)
    | _ -> None
  in
  match url_and_body with
  | None ->
      probe_local_native
        ~on_done:(fun ok ->
          if ok then notify_status Ready
          else notify_status (Interrupted cap_ref.native_reason);
          cb (Js.Unsafe.obj [| ("ok", Js.Unsafe.inject (Js.bool ok)) |]))
        ()
  | Some (url, body_opt, is_exec) ->
      let gen = !local_gen in
      let req_token = !next_req_id in
      next_req_id := req_token + 1;
      if is_exec then (
        pending_local_cbs := (req_token, cb) :: !pending_local_cbs;
        notify_status Running);
      fetch_json ~meth:"POST" ~url ?body:body_opt (fun res ->
          pending_local_cbs :=
            List.filter (fun (id, _) -> id <> req_token) !pending_local_cbs;
          if gen = !local_gen then (
            if is_exec && !runtime_mode_ref = LocalNative then (
              match !kernel_state_ref with
              | Running | Interrupting _ | Booting _ -> notify_status Ready
              | _ -> ());
            match res with
            | Ok (status, txt) when status >= 200 && status < 300 -> (
                match json_parse txt with
                | Some parsed ->
                    if typ = "complete" then (
                      let matches = Js.Unsafe.get parsed (js "matches") in
                      let compat =
                        Js.Unsafe.obj
                          [|
                            ("type", Js.Unsafe.inject (js "complete"));
                            ("items", matches);
                          |]
                      in
                      cb compat)
                    else cb parsed
                | None ->
                    cb
                      (Js.Unsafe.obj
                         [|
                           ("ok", Js.Unsafe.inject Js._false);
                           ("stdout", Js.Unsafe.inject (js ""));
                           ("stderr", Js.Unsafe.inject (js "Invalid JSON from /api/kernel"));
                           ("incomplete", Js.Unsafe.inject Js._false);
                         |]))
            | Ok (status, txt) ->
                cb
                  (Js.Unsafe.obj
                     [|
                       ("ok", Js.Unsafe.inject Js._false);
                       ("stdout", Js.Unsafe.inject (js ""));
                       ( "stderr",
                         Js.Unsafe.inject
                           (js (Printf.sprintf "Local kernel HTTP %d: %s" status txt)) );
                       ("incomplete", Js.Unsafe.inject Js._false);
                     |])
            | Error msg ->
                cb
                  (Js.Unsafe.obj
                     [|
                       ("ok", Js.Unsafe.inject Js._false);
                       ("stdout", Js.Unsafe.inject (js ""));
                       ("stderr", Js.Unsafe.inject (js msg));
                       ("incomplete", Js.Unsafe.inject Js._false);
                     |])))

let send_worker typ fields cb =
  match !runtime_mode_ref with
  | LocalNative -> send_local_native typ fields cb
  | Wasm ->
      let w = ensure_worker () in
      let req_id = !next_req_id in
      next_req_id := req_id + 1;
      Hashtbl.add pending_reqs req_id cb;
      if typ = "exec" || typ = "repl" then set_sab_signal 0;
      if !kernel_state_ref = Ready && (typ = "exec" || typ = "repl") then notify_status Running;
      let extra =
        if typ = "init" then
          let base_init = [ ("mountOpfs", Js.Unsafe.inject Js._true) ] in
          match !sab_ref with
          | Some sab -> ("interruptBuffer", sab) :: base_init
          | None -> base_init
        else []
      in
      let payload =
        Js.Unsafe.obj
          (Array.of_list
             (("type", Js.Unsafe.inject (js typ))
             :: ("id", Js.Unsafe.inject req_id)
             :: (fields @ extra)))
      in
      ignore (Js.Unsafe.meth_call w "postMessage" [| Js.Unsafe.inject payload |])

let warm_kernel () =
  match !runtime_mode_ref with
  | LocalNative -> probe_local_native ()
  | Wasm ->
      send_worker "init" [] (fun res ->
          if
            Js.Optdef.test (Js.Unsafe.get res (js "ok"))
            && not (Js.to_bool (Js.Unsafe.get res (js "ok")))
          then
            let msg =
              match opt_str res "stderr" with
              | Some s when s <> "" -> s
              | _ -> (
                  match opt_str res "error" with
                  | Some s when s <> "" -> s
                  | _ -> "Initialization failed")
            in
            notify_status (Interrupted msg))

let reset_kernel () =
  match !runtime_mode_ref with
  | LocalNative ->
      let restart = match !kernel_state_ref with Interrupted _ -> true | _ -> false in
      let body = Js.Unsafe.obj [| ("restart", Js.Unsafe.inject (Js.bool restart)) |] in
      notify_status (Booting "Resetting...");
      fetch_json ~meth:"POST"
        ~url:(get_base_url () ^ "api/kernel/reset")
        ~body (fun _ -> notify_status Ready)
  | Wasm -> (
      match (!worker_ref, !kernel_state_ref) with
      | Some _, Ready ->
          send_worker "reset_ns" [] (fun _ -> notify_status Ready)
      | _ ->
          interrupt_kernel ~force:true ();
          warm_kernel ())

let status_text = function
  | Idle -> "Idle"
  | Booting d ->
      if !boot_total_ref > 0 && !boot_step_ref > 0 then
        Printf.sprintf "%s (%d/%d)" d !boot_step_ref !boot_total_ref
      else d
  | Ready ->
      if !runtime_mode_ref = LocalNative then
        Printf.sprintf "Local (%s)" cap_ref.native_python
      else if cap_ref.webgpu_active && cap_ref.webgpu_adapter <> "" then
        "Ready · WebGPU (" ^ cap_ref.webgpu_adapter ^ ")"
      else "Ready"
  | Running -> "Running..."
  | Interrupting r -> r
  | Interrupted r -> r

(* ---------------------------------------------------------------------------
   Live OptimizerState Summary & Multi-View Scientific SVG Renderers
   --------------------------------------------------------------------------- *)

let json_num_list (raw : Js.Unsafe.any) =
  if not (is_defined raw) then []
  else
    let arr : Js.Unsafe.any Js.js_array Js.t = Obj.magic raw in
    let len = arr##.length in
    let out = ref [] in
    for i = 0 to len - 1 do
      let item = Js.array_get arr i in
      match Js.Optdef.to_option item with
      | Some v when is_defined v ->
          let fv = Js.float_of_number (Obj.magic v) in
          if is_finite fv then out := fv :: !out
      | _ -> ()
    done;
    List.rev !out

let json_obj_list (raw : Js.Unsafe.any) : Js.Unsafe.any list =
  if not (is_defined raw) then []
  else
    let arr : Js.Unsafe.any Js.js_array Js.t = Obj.magic raw in
    let len = arr##.length in
    let out = ref [] in
    for i = 0 to len - 1 do
      match Js.Optdef.to_option (Js.array_get arr i) with
      | Some v when is_defined v -> out := v :: !out
      | _ -> ()
    done;
    List.rev !out

let json_pair_list (raw : Js.Unsafe.any) : (float * float) list =
  List.filter_map
    (fun row ->
      let xs = json_num_list row in
      match xs with a :: b :: _ -> Some (a, b) | _ -> None)
    (json_obj_list raw)

let download_text_file ~filename ~mime ~text =
  try
    let blob_ctor = Js.Unsafe.get Dom_html.window (js "Blob") in
    let blob_parts = Js.array [| Js.Unsafe.inject (js text) |] in
    let blob_opts = Js.Unsafe.obj [| ("type", Js.Unsafe.inject (js mime)) |] in
    let blob = Js.Unsafe.new_obj blob_ctor [| Js.Unsafe.inject blob_parts; Js.Unsafe.inject blob_opts |] in
    let url_ctor = Js.Unsafe.get Dom_html.window (js "URL") in
    let blob_url : Js.js_string Js.t = Js.Unsafe.meth_call url_ctor "createObjectURL" [| Js.Unsafe.inject blob |] in
    let a = createA document in
    a##.href := blob_url;
    a##setAttribute (js "download") (js filename);
    ignore (Js.Unsafe.meth_call document##.body "appendChild" [| Js.Unsafe.inject a |]);
    a##click;
    ignore (Js.Unsafe.meth_call document##.body "removeChild" [| Js.Unsafe.inject a |]);
    ignore (Js.Unsafe.meth_call url_ctor "revokeObjectURL" [| Js.Unsafe.inject blob_url |])
  with _ -> ()

let save_svg_element (svg : element Js.t) (filename : string) =
  try
    let serializer_ctor = Js.Unsafe.get Dom_html.window (js "XMLSerializer") in
    let serializer = Js.Unsafe.new_obj serializer_ctor [||] in
    let svg_xml : Js.js_string Js.t =
      Js.Unsafe.meth_call serializer "serializeToString" [| Js.Unsafe.inject svg |]
    in
    download_text_file ~filename ~mime:"image/svg+xml;charset=utf-8" ~text:(to_s svg_xml)
  with _ -> ()

let make_fig_pane title subtitle =
  let card = mk ~cls:"py-fig-pane" "div" in
  let hdr = mk ~cls:"py-fig-hdr" "div" in
  let hdr_info = mk ~cls:"py-fig-hdr-info" "div" in
  Dom.appendChild hdr_info (mk ~cls:"py-fig-title" ~txt:title "div");
  if trim_s subtitle <> "" then
    Dom.appendChild hdr_info (mk ~cls:"py-fig-subtitle" ~txt:subtitle "div");
  let hdr_actions = mk ~cls:"py-fig-actions" "div" in
  let readout_el = mk ~cls:"py-fig-readout" "span" in
  let zoom_btn = mk ~cls:"py-ghost-btn py-fig-zoom" ~txt:"Zoom" "button" in
  zoom_btn##setAttribute (js "type") (js "button");
  zoom_btn##setAttribute (js "title") (js "Toggle full-height view for this figure");
  zoom_btn##.onclick :=
    Dom.handler (fun _ ->
        (match Js.Opt.to_option (card##.parentNode) with
        | Some parent_node ->
            let parent_el : element Js.t = Obj.magic parent_node in
            let card_cls = to_s card##.className in
            let focusing = not (contains_sub card_cls "is-focused") in
            let all_cards = parent_el##querySelectorAll (js ".py-fig-pane") in
            for i = 0 to all_cards##.length - 1 do
              match Js.Opt.to_option (all_cards##item i) with
              | Some c ->
                  c##.className := js "py-fig-pane";
                  (match Js.Opt.to_option (c##querySelector (js ".py-fig-zoom")) with
                  | Some zb -> set_text zb "Zoom"
                  | None -> ())
              | None -> ()
            done;
            let p_cls =
              List.filter
                (fun s -> s <> "has-focused-fig" && s <> "")
                (String.split_on_char ' ' (to_s parent_el##.className))
            in
            if focusing then (
              card##.className := js "py-fig-pane is-focused";
              parent_el##.className := js (String.concat " " (p_cls @ [ "has-focused-fig" ]));
              set_text zoom_btn "All")
            else (
              parent_el##.className := js (String.concat " " p_cls);
              set_text zoom_btn "Zoom")
        | None -> ());
        Js._true);
  let save_btn = mk ~cls:"py-ghost-btn py-fig-save" ~txt:"Save SVG" "button" in
  save_btn##setAttribute (js "type") (js "button");
  save_btn##setAttribute (js "title") (js "Download figure as SVG");
  save_btn##.onclick :=
    Dom.handler (fun _ ->
        (match Js.Opt.to_option (card##querySelector (js "svg")) with
        | Some svg -> save_svg_element svg "baux_figure.svg"
        | None -> ());
        Js._true);
  Dom.appendChild hdr_actions readout_el;
  Dom.appendChild hdr_actions zoom_btn;
  Dom.appendChild hdr_actions save_btn;
  Dom.appendChild hdr hdr_info;
  Dom.appendChild hdr hdr_actions;
  let stage = mk ~cls:"py-fig-stage" "div" in
  Dom.appendChild card hdr;
  Dom.appendChild card stage;
  (card, hdr_actions, readout_el, stage)

let append_legend_badge legend swatch_cls label =
  let item = mk ~cls:"py-fig-legend-item" "span" in
  Dom.appendChild item (mk ~cls:("py-fig-swatch " ^ swatch_cls) "span");
  Dom.appendChild item (mk ~cls:"py-fig-legend-label" ~txt:label "span");
  Dom.appendChild legend item

type curve_series = {
  c_label : string;
  c_xs : float list;
  c_ys : float list;
  c_dashed : bool;
}

let render_curves_card ~title ~subtitle ~x_label ~y_label ~log_y (series : curve_series list) =
  let card, _hdr_actions, readout_el, stage = make_fig_pane title subtitle in
  let non_empty = List.filter (fun s -> s.c_ys <> [] && s.c_xs <> []) series in
  if non_empty = [] then card
  else
    let all_y_raw = List.concat_map (fun s -> s.c_ys) non_empty in
    let use_log = log_y && List.for_all (fun v -> v > 0.0) all_y_raw in
    let tr_y v = if use_log then log10 (max 1e-12 v) else v in
    let all_x = List.concat_map (fun s -> s.c_xs) non_empty in
    let all_y = List.map tr_y all_y_raw in
    let x_lo_r = List.fold_left min (List.hd all_x) all_x in
    let x_hi_r = List.fold_left max (List.hd all_x) all_x in
    let x_sp = if x_hi_r -. x_lo_r < 1e-9 then 1.0 else x_hi_r -. x_lo_r in
    let y_lo_r = List.fold_left min (List.hd all_y) all_y in
    let y_hi_r = List.fold_left max (List.hd all_y) all_y in
    let y_sp_r = if y_hi_r -. y_lo_r < 1e-9 then 1.0 else y_hi_r -. y_lo_r in
    let y_lo = y_lo_r -. 0.07 *. y_sp_r in
    let y_hi = y_hi_r +. 0.07 *. y_sp_r in
    let y_sp = y_hi -. y_lo in
    let legend = mk ~cls:"py-fig-legend" "div" in
    let xl_lower = String.lowercase_ascii x_label in
    let show_end_val = contains_sub xl_lower "eval" || contains_sub xl_lower "step" in
    List.iteri
      (fun idx s ->
        let slot = idx mod 5 in
        let last_y = List.nth s.c_ys (List.length s.c_ys - 1) in
        let lbl_txt =
          if show_end_val then Printf.sprintf "%s (%.4g)" s.c_label last_y else s.c_label
        in
        let sw_cls =
          Printf.sprintf "py-swatch-%d%s" slot (if s.c_dashed then " is-dashed" else "")
        in
        append_legend_badge legend sw_cls lbl_txt)
      non_empty;
    Dom.appendChild stage legend;
    let w = 460.0 and h = 158.0 in
    let pl = 60.0 and pr = 16.0 and pt = 22.0 and pb = 28.0 in
    let pw = w -. pl -. pr in
    let ph = h -. pt -. pb in
    let x_px x = pl +. ((x -. x_lo_r) /. x_sp) *. pw in
    let y_px y = pt +. (1.0 -. ((tr_y y -. y_lo) /. y_sp)) *. ph in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let yy = pt +. (1.0 -. frac) *. ph in
      let v_t = y_lo +. frac *. y_sp in
      let v_disp = if use_log then 10.0 ** v_t else v_t in
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
      set_text lbl (Printf.sprintf "%.3g" v_disp);
      Dom.appendChild svg lbl
    done;
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let xx = pl +. frac *. pw in
      let xv = x_lo_r +. frac *. x_sp in
      let xtl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" xx);
            ("y", Printf.sprintf "%.1f" (h -. pb +. 12.0));
            ("text-anchor", "middle");
            ("class", "py-svg-tick");
          ]
      in
      set_text xtl (Printf.sprintf "%.3g" xv);
      Dom.appendChild svg xtl
    done;
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
    let ylbl =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" pl);
          ("y", "11.5");
          ("text-anchor", "start");
          ("class", "py-svg-axis-lbl");
        ]
    in
    set_text ylbl (if use_log then y_label ^ " (log)" else y_label);
    Dom.appendChild svg ylbl;
    List.iteri
      (fun idx s ->
        let slot = idx mod 5 in
        let rec zip_pts xs ys acc =
          match (xs, ys) with
          | x :: xtl, y :: ytl ->
              zip_pts xtl ytl (Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y) :: acc)
          | _ -> List.rev acc
        in
        let pts_list = zip_pts s.c_xs s.c_ys [] in
        let pts_str = String.concat " " pts_list in
        let line_cls =
          Printf.sprintf "py-svg-series py-svg-series-%d%s" slot
            (if s.c_dashed then " is-dashed" else "")
        in
        Dom.appendChild svg
          (svg_el "polyline" [ ("fill", "none"); ("points", pts_str); ("class", line_cls) ]);
        let npts = min (List.length s.c_xs) (List.length s.c_ys) in
        if npts > 0 then
          let lx = x_px (List.nth s.c_xs (npts - 1)) in
          let ly = y_px (List.nth s.c_ys (npts - 1)) in
          Dom.appendChild svg
            (svg_el "circle"
               [
                 ("cx", Printf.sprintf "%.2f" lx);
                 ("cy", Printf.sprintf "%.2f" ly);
                 ("r", "3.4");
                 ("class", Printf.sprintf "py-svg-series-dot py-svg-series-dot-%d" slot);
               ]))
      non_empty;
    let crosshair =
      svg_el "line"
        [
          ("y1", Printf.sprintf "%.1f" pt);
          ("y2", Printf.sprintf "%.1f" (pt +. ph));
          ("class", "py-svg-crosshair");
          ("style", "display: none; pointer-events: none;");
        ]
    in
    Dom.appendChild svg crosshair;
    let hover_dots =
      List.mapi
        (fun idx _s ->
          let slot = idx mod 5 in
          let dot =
            svg_el "circle"
              [
                ("r", "4.0");
                ("class", Printf.sprintf "py-svg-hover-dot py-svg-series-dot-%d" slot);
                ("style", "display: none; pointer-events: none;");
              ]
          in
          Dom.appendChild svg dot;
          dot)
        non_empty
    in
    let overlay =
      svg_el "rect"
        [
          ("x", Printf.sprintf "%.1f" pl);
          ("y", Printf.sprintf "%.1f" pt);
          ("width", Printf.sprintf "%.1f" pw);
          ("height", Printf.sprintf "%.1f" ph);
          ("fill", "transparent");
          ("class", "py-svg-overlay");
          ("pointer-events", "all");
          ("style", "cursor: crosshair;");
        ]
    in
    overlay##.onmousemove :=
      Dom.handler (fun ev ->
          let rect = overlay##getBoundingClientRect in
          let r_left = Js.float_of_number rect##.left in
          let ow = Js.float_of_number rect##.right -. r_left in
          if ow > 0.0 then (
            let rx = max 0.0 (min 1.0 ((Js.float_of_number ev##.clientX -. r_left) /. ow)) in
            let data_x = x_lo_r +. rx *. x_sp in
            let best_x = ref data_x in
            let best_dist = ref infinity in
            List.iter
              (fun s ->
                List.iter
                  (fun x ->
                    let d = abs_float (x -. data_x) in
                    if d < !best_dist then (
                      best_dist := d;
                      best_x := x))
                  s.c_xs)
              non_empty;
            let snap_x = !best_x in
            let sx = x_px snap_x in
            crosshair##setAttribute (js "x1") (js (Printf.sprintf "%.2f" sx));
            crosshair##setAttribute (js "x2") (js (Printf.sprintf "%.2f" sx));
            crosshair##.style##.display := js "";
            let readouts = ref [] in
            List.iter2
              (fun s dot ->
                let rec find_closest best_d best_pt xs ys =
                  match (xs, ys) with
                  | x :: xt, y :: yt ->
                      let d = abs_float (x -. snap_x) in
                      if d < best_d then find_closest d (Some (x, y)) xt yt
                      else find_closest best_d best_pt xt yt
                  | _ -> best_pt
                in
                match find_closest infinity None s.c_xs s.c_ys with
                | Some (px, py) when abs_float (px -. snap_x) < 1e-4 ->
                    let sy = y_px py in
                    dot##setAttribute (js "cx") (js (Printf.sprintf "%.2f" sx));
                    dot##setAttribute (js "cy") (js (Printf.sprintf "%.2f" sy));
                    dot##.style##.display := js "";
                    readouts := Printf.sprintf "%s %.4f" s.c_label py :: !readouts
                | _ -> dot##.style##.display := js "none")
              non_empty hover_dots;
            let x_lbl_clean =
              if contains_sub (String.lowercase_ascii x_label) "eval" then "Eval"
              else x_label
            in
            let head = Printf.sprintf "%s %.0f" x_lbl_clean snap_x in
            let full_txt = String.concat " · " (head :: List.rev !readouts) in
            set_text readout_el full_txt);
          Js._true);
    overlay##.onmouseout :=
      Dom.handler (fun _ ->
          crosshair##.style##.display := js "none";
          List.iter (fun d -> d##.style##.display := js "none") hover_dots;
          set_text readout_el "";
          Js._true);
    Dom.appendChild svg overlay;
    let xlbl =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (pl +. 0.5 *. pw));
          ("y", Printf.sprintf "%.1f" (h -. 3.0));
          ("text-anchor", "middle");
          ("class", "py-svg-axis-lbl");
        ]
    in
    set_text xlbl x_label;
    Dom.appendChild svg xlbl;
    Dom.appendChild stage svg;
    card

let curves_from_js (raw : Js.Unsafe.any) : element Js.t option =
  if not (is_defined raw) then None
  else
    let raw_series = json_obj_list (Js.Unsafe.get raw (js "series")) in
    let series =
      List.filter_map
        (fun item ->
          let ys = json_num_list (Js.Unsafe.get item (js "y")) in
          if ys = [] then None
          else
            let xs0 = json_num_list (Js.Unsafe.get item (js "x")) in
            let xs =
              if List.length xs0 = List.length ys then xs0
              else List.mapi (fun i _ -> float_of_int (i + 1)) ys
            in
            Some
              {
                c_label = Option.value (opt_str item "label") ~default:"series";
                c_xs = xs;
                c_ys = ys;
                c_dashed = opt_bool item "dashed";
              })
        raw_series
    in
    if series = [] then None
    else
      Some
        (render_curves_card
           ~title:(Option.value (opt_str raw "title") ~default:"Convergence Comparison")
           ~subtitle:(Option.value (opt_str raw "subtitle") ~default:"")
           ~x_label:(Option.value (opt_str raw "x_label") ~default:"Evaluations")
           ~y_label:(Option.value (opt_str raw "y_label") ~default:"Objective")
           ~log_y:(opt_bool raw "log_y")
           series)

let marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px (vals : float array) (level : float) :
    string =
  let len = Array.length vals in
  if nr < 2 || nz < 2 || len <> nr * nz then ""
  else
    let dr = (r1 -. r0) /. float_of_int (nr - 1) in
    let dz = (z1 -. z0) /. float_of_int (nz - 1) in
    let interp a b va vb =
      let dv = vb -. va in
      if abs_float dv < 1e-12 then 0.5 *. (a +. b)
      else
        let t = max 0.0 (min 1.0 ((level -. va) /. dv)) in
        a +. t *. (b -. a)
    in
    let buf = Buffer.create 512 in
    let emit (ax, ay) (bx, by) =
      Buffer.add_string buf
        (Printf.sprintf "M%.2f %.2fL%.2f %.2f" (x_px ax) (y_px ay) (x_px bx) (y_px by))
    in
    for iz = 0 to nz - 2 do
      let za = z0 +. float_of_int iz *. dz in
      let zb = za +. dz in
      let row0 = iz * nr in
      let row1 = (iz + 1) * nr in
      for ir = 0 to nr - 2 do
        let ra = r0 +. float_of_int ir *. dr in
        let rb = ra +. dr in
        let v00 = vals.(row0 + ir) in
        let v10 = vals.(row0 + ir + 1) in
        let v11 = vals.(row1 + ir + 1) in
        let v01 = vals.(row1 + ir) in
        let b0 = v00 >= level in
        let b1 = v10 >= level in
        let b2 = v11 >= level in
        let b3 = v01 >= level in
        let c_b = if b0 <> b1 then Some (interp ra rb v00 v10, za) else None in
        let c_r = if b1 <> b2 then Some (rb, interp za zb v10 v11) else None in
        let c_t = if b2 <> b3 then Some (interp ra rb v01 v11, zb) else None in
        let c_l = if b3 <> b0 then Some (ra, interp za zb v00 v01) else None in
        match (c_b, c_r, c_t, c_l) with
        | Some pb, Some pr, Some pt, Some pl ->
            let vc = 0.25 *. (v00 +. v10 +. v11 +. v01) in
            if vc >= level then (
              emit pb pr;
              emit pl pt)
            else (
              emit pb pl;
              emit pr pt)
        | _ ->
            let hits = List.filter_map (fun x -> x) [ c_b; c_r; c_t; c_l ] in
            (match hits with p :: q :: _ -> emit p q | _ -> ())
      done
    done;
    Buffer.contents buf

let render_tokamak_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"Tokamak Equilibrium" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, _hdr_actions, readout_el, stage = make_fig_pane title subtitle in
  let split = mk ~cls:"py-tokamak-split" "div" in
  let aside = mk ~cls:"py-tokamak-aside" "div" in
  let legend = mk ~cls:"py-fig-legend py-fig-legend-col" "div" in
  append_legend_badge legend "py-swatch-target" "Target LCFS";
  append_legend_badge legend "py-swatch-init" "Initial guess";
  append_legend_badge legend "py-swatch-enn" "ENN separatrix";
  append_legend_badge legend "py-swatch-coil-pos" "PF coil (+)";
  append_legend_badge legend "py-swatch-coil-neg" "PF coil (-)";
  Dom.appendChild aside legend;
  let r_rng = json_num_list (Js.Unsafe.get fig (js "r_range")) in
  let z_rng = json_num_list (Js.Unsafe.get fig (js "z_range")) in
  let r0, r1 = match r_rng with a :: b :: _ when b > a -> (a, b) | _ -> (0.4, 3.0) in
  let z0, z1 = match z_rng with a :: b :: _ when b > a -> (a, b) | _ -> (-1.6, 1.6) in
  let coils_n = List.length (json_obj_list (Js.Unsafe.get fig (js "coils"))) in
  let meta_list = mk ~cls:"py-tokamak-meta" "div" in
  Dom.appendChild meta_list (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "R [%.2f, %.2f] m" r0 r1) "div");
  Dom.appendChild meta_list (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "Z [%.2f, %.2f] m" z0 z1) "div");
  if coils_n > 0 then (
    Dom.appendChild meta_list
      (mk ~cls:"py-tokamak-meta-row" ~txt:(Printf.sprintf "%d PF/CS coils" coils_n) "div");
    Dom.appendChild meta_list
      (mk ~cls:"py-tokamak-meta-row" ~txt:"× X-point null" "div"));
  Dom.appendChild aside meta_list;
  let w = 290.0 and h = 230.0 in
  let pl = 46.0 and pr = 12.0 and pt = 10.0 and pb = 32.0 in
  let pw = w -. pl -. pr and ph = h -. pt -. pb in
  let scale = min (pw /. (r1 -. r0)) (ph /. (z1 -. z0)) in
  let rw = scale *. (r1 -. r0) and zh = scale *. (z1 -. z0) in
  let ox = pl +. 0.5 *. (pw -. rw) in
  let oy = pt +. 0.5 *. (ph -. zh) in
  let x_px r = ox +. (r -. r0) *. scale in
  let y_px z = oy +. (z1 -. z) *. scale in
  let svg =
    svg_el "svg"
      [
        ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
        ("class", "py-traj-svg py-tokamak-svg");
        ("role", "img");
        ("aria-label", title);
      ]
  in
  (* Frame and R/Z ticks *)
  Dom.appendChild svg
    (svg_el "rect"
       [
         ("x", Printf.sprintf "%.1f" ox);
         ("y", Printf.sprintf "%.1f" oy);
         ("width", Printf.sprintf "%.1f" rw);
         ("height", Printf.sprintf "%.1f" zh);
         ("class", "py-svg-frame");
       ]);
  for i = 0 to 3 do
    let f = float_of_int i /. 3.0 in
    let rv = r0 +. f *. (r1 -. r0) in
    let zv = z0 +. f *. (z1 -. z0) in
    let rx = x_px rv in
    let zy = y_px zv in
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" rx);
           ("x2", Printf.sprintf "%.1f" rx);
           ("y1", Printf.sprintf "%.1f" oy);
           ("y2", Printf.sprintf "%.1f" (oy +. zh));
           ("class", "py-svg-grid");
         ]);
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" ox);
           ("x2", Printf.sprintf "%.1f" (ox +. rw));
           ("y1", Printf.sprintf "%.1f" zy);
           ("y2", Printf.sprintf "%.1f" zy);
           ("class", "py-svg-grid");
         ]);
    let tx =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" rx);
          ("y", Printf.sprintf "%.1f" (oy +. zh +. 13.5));
          ("text-anchor", "middle");
          ("class", "py-svg-tick");
        ]
    in
    set_text tx (Printf.sprintf "%.2f" rv);
    Dom.appendChild svg tx;
    let ty =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (ox -. 6.0));
          ("y", Printf.sprintf "%.1f" (zy +. 2.5));
          ("text-anchor", "end");
          ("class", "py-svg-tick");
        ]
    in
    set_text ty (Printf.sprintf "%.2f" zv);
    Dom.appendChild svg ty
  done;
  (* Flux contours *)
  let fg = Js.Unsafe.get fig (js "flux_grid") in
  (if is_defined fg then
     let nr = Option.value (opt_int fg "nr") ~default:0 in
     let nz = Option.value (opt_int fg "nz") ~default:0 in
     let arr_of field = Array.of_list (json_num_list (Js.Unsafe.get fg (js field))) in
     let psi_init = arr_of "psi_init" in
     let psi_opt = arr_of "psi_opt" in
     (match (opt_float fg "psi_init_target", Array.length psi_init = nr * nz) with
     | Some lvl, true ->
         let d = marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_init lvl in
         if d <> "" then
           Dom.appendChild svg (svg_el "path" [ ("d", d); ("class", "py-svg-sep-init") ])
     | _ -> ());
     (match (opt_float fg "psi_opt_target", Array.length psi_opt = nr * nz) with
     | Some lvl, true ->
         let lo_v = Array.fold_left min psi_opt.(0) psi_opt in
         let hi_v = Array.fold_left max psi_opt.(0) psi_opt in
         let span = max 1e-6 (hi_v -. lo_v) in
         List.iter
           (fun df ->
             let d_sub =
               marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_opt
                 (lvl +. df *. span)
             in
             if d_sub <> "" then
               Dom.appendChild svg (svg_el "path" [ ("d", d_sub); ("class", "py-svg-flux-minor") ]))
           [ -0.14; -0.07; 0.07; 0.14 ];
         let d = marching_squares_path ~nr ~nz ~r0 ~r1 ~z0 ~z1 ~x_px ~y_px psi_opt lvl in
         if d <> "" then
           Dom.appendChild svg (svg_el "path" [ ("d", d); ("class", "py-svg-sep-opt") ])
     | _ -> ()));
  (* First wall & Target LCFS *)
  let wall = json_pair_list (Js.Unsafe.get fig (js "first_wall")) in
  if wall <> [] then (
    let pts = String.concat " " (List.map (fun (r, z) -> Printf.sprintf "%.2f,%.2f" (x_px r) (y_px z)) wall) in
    Dom.appendChild svg (svg_el "polygon" [ ("points", pts); ("class", "py-svg-first-wall") ]));
  let lcfs = json_pair_list (Js.Unsafe.get fig (js "lcfs")) in
  if lcfs <> [] then (
    let pts = String.concat " " (List.map (fun (r, z) -> Printf.sprintf "%.2f,%.2f" (x_px r) (y_px z)) lcfs) in
    Dom.appendChild svg (svg_el "polygon" [ ("points", pts); ("class", "py-svg-lcfs-target") ]));
  let overlay =
    svg_el "rect"
      [
        ("x", Printf.sprintf "%.1f" ox);
        ("y", Printf.sprintf "%.1f" oy);
        ("width", Printf.sprintf "%.1f" rw);
        ("height", Printf.sprintf "%.1f" zh);
        ("fill", "transparent");
        ("class", "py-svg-overlay");
        ("pointer-events", "all");
        ("style", "cursor: crosshair;");
      ]
  in
  overlay##.onmousemove :=
    Dom.handler (fun ev ->
        let rect = overlay##getBoundingClientRect in
        let r_left = Js.float_of_number rect##.left in
        let r_top = Js.float_of_number rect##.top in
        let w_box = Js.float_of_number rect##.right -. r_left in
        let h_box = Js.float_of_number rect##.bottom -. r_top in
        if w_box > 0.0 && h_box > 0.0 then (
          let rx = max 0.0 (min 1.0 ((Js.float_of_number ev##.clientX -. r_left) /. w_box)) in
          let ry = max 0.0 (min 1.0 ((Js.float_of_number ev##.clientY -. r_top) /. h_box)) in
          let r_val = r0 +. rx *. (r1 -. r0) in
          let z_val = z1 -. ry *. (z1 -. z0) in
          set_text readout_el (Printf.sprintf "R=%.2fm Z=%.2fm" r_val z_val));
        Js._true);
  overlay##.onmouseout :=
    Dom.handler (fun _ ->
        set_text readout_el "";
        Js._true);
  Dom.appendChild svg overlay;
  (* PF Coils *)
  List.iter
    (fun row ->
      match json_num_list row with
      | rc :: zc :: drc :: dzc :: curr :: rest ->
          let w_px = max 4.0 (drc *. scale) in
          let h_px = max 4.0 (dzc *. scale) in
          let cx0 = x_px rc -. 0.5 *. w_px in
          let cy0 = y_px zc -. 0.5 *. h_px in
          let cls =
            (if curr >= 0.0 then "py-svg-coil-pos" else "py-svg-coil-neg")
            ^ " py-svg-coil-hit"
          in
          let op = Printf.sprintf "%.2f" (0.28 +. 0.68 *. min 1.0 (abs_float curr)) in
          let r_coil =
            svg_el "rect"
              [
                ("x", Printf.sprintf "%.2f" cx0);
                ("y", Printf.sprintf "%.2f" cy0);
                ("width", Printf.sprintf "%.2f" w_px);
                ("height", Printf.sprintf "%.2f" h_px);
                ("rx", "1.5");
                ("fill-opacity", op);
                ("class", cls);
                ("pointer-events", "all");
              ]
          in
          let coil_txt =
            match rest with
            | ma :: _ -> Printf.sprintf "Coil R=%.2fm Z=%.2fm · %.2f MA" rc zc ma
            | [] -> Printf.sprintf "Coil R=%.2fm Z=%.2fm · %.2f" rc zc curr
          in
          r_coil##.onmouseover :=
            Dom.handler (fun _ ->
                set_text readout_el coil_txt;
                Js._true);
          r_coil##.onmouseout :=
            Dom.handler (fun _ ->
                set_text readout_el "";
                Js._true);
          Dom.appendChild svg r_coil
      | _ -> ())
    (json_obj_list (Js.Unsafe.get fig (js "coils")));
  (* X-point crosshair *)
  (match json_num_list (Js.Unsafe.get fig (js "x_point")) with
  | rx :: zx :: _ ->
      let xx = x_px rx and yy = y_px zx in
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.2f" (xx -. 5.0));
             ("x2", Printf.sprintf "%.2f" (xx +. 5.0));
             ("y1", Printf.sprintf "%.2f" (yy -. 5.0));
             ("y2", Printf.sprintf "%.2f" (yy +. 5.0));
             ("class", "py-svg-xpt");
           ]);
      Dom.appendChild svg
        (svg_el "line"
           [
             ("x1", Printf.sprintf "%.2f" (xx -. 5.0));
             ("x2", Printf.sprintf "%.2f" (xx +. 5.0));
             ("y1", Printf.sprintf "%.2f" (yy +. 5.0));
             ("y2", Printf.sprintf "%.2f" (yy -. 5.0));
             ("class", "py-svg-xpt");
           ])
  | _ -> ());
  let axlbl =
    svg_el "text"
      [
        ("x", Printf.sprintf "%.1f" (ox +. 0.5 *. rw));
        ("y", Printf.sprintf "%.1f" (h -. 3.0));
        ("text-anchor", "middle");
        ("class", "py-svg-axis-lbl");
      ]
  in
  set_text axlbl "R [m] vs Z [m]";
  Dom.appendChild svg axlbl;
  Dom.appendChild split svg;
  Dom.appendChild split aside;
  Dom.appendChild stage split;
  card

let render_field2d_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"2D Field" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, _hdr_actions, readout_el, stage = make_fig_pane title subtitle in
  let panels = json_obj_list (Js.Unsafe.get fig (js "panels")) in
  if panels = [] then card
  else
    let n_p = min 2 (List.length panels) in
    let w = 460.0 and h = 192.0 in
    let gap = 18.0 and pl = 10.0 and pr = 10.0 and pt = 20.0 and pb = 24.0 in
    let avail_w = w -. pl -. pr -. float_of_int (n_p - 1) *. gap in
    let box_s = min (avail_w /. float_of_int n_p) (h -. pt -. pb) in
    let total_w = float_of_int n_p *. box_s +. float_of_int (n_p - 1) *. gap in
    let start_x = 0.5 *. (w -. total_w) in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    List.iteri
      (fun p_idx p ->
        if p_idx < 2 then
          let nx = Option.value (opt_int p "nx") ~default:0 in
          let nz = Option.value (opt_int p "nz") ~default:0 in
          let vals = Array.of_list (json_num_list (Js.Unsafe.get p (js "vals"))) in
          if nx >= 2 && nz >= 2 && Array.length vals = nx * nz then (
            let vmin =
              match opt_float p "vmin" with
              | Some v -> v
              | None -> Array.fold_left min vals.(0) vals
            in
            let vmax =
              match opt_float p "vmax" with
              | Some v -> v
              | None -> Array.fold_left max vals.(0) vals
            in
            let span = if vmax -. vmin < 1e-12 then 1.0 else vmax -. vmin in
            let bx = start_x +. float_of_int p_idx *. (box_s +. gap) in
            let by = pt in
            let cw = box_s /. float_of_int nx in
            let ch = box_s /. float_of_int nz in
            let p_lbl =
              svg_el "text"
                [
                  ("x", Printf.sprintf "%.1f" (bx +. 0.5 *. box_s));
                  ("y", Printf.sprintf "%.1f" (pt -. 6.0));
                  ("text-anchor", "middle");
                  ("class", "py-svg-axis-lbl");
                ]
            in
            let label = Option.value (opt_str p "label") ~default:"Field" in
            set_text p_lbl label;
            Dom.appendChild svg p_lbl;
            for iz = 0 to nz - 1 do
              for ix = 0 to nx - 1 do
                let v = vals.(iz * nx + ix) in
                let norm = max 0.0 (min 1.0 ((v -. vmin) /. span)) in
                let bucket = int_of_float (floor (norm *. 15.999)) in
                Dom.appendChild svg
                  (svg_el "rect"
                     [
                       ("x", Printf.sprintf "%.2f" (bx +. float_of_int ix *. cw));
                       ("y", Printf.sprintf "%.2f" (by +. float_of_int (nz - 1 - iz) *. ch));
                       ("width", Printf.sprintf "%.2f" (cw +. 0.4));
                       ("height", Printf.sprintf "%.2f" (ch +. 0.4));
                       ("class", Printf.sprintf "py-heat-%d" bucket);
                     ])
              done
            done;
            Dom.appendChild svg
              (svg_el "rect"
                 [
                   ("x", Printf.sprintf "%.2f" bx);
                   ("y", Printf.sprintf "%.2f" by);
                   ("width", Printf.sprintf "%.2f" box_s);
                   ("height", Printf.sprintf "%.2f" box_s);
                   ("class", "py-svg-frame");
                 ]);
            let p_overlay =
              svg_el "rect"
                [
                  ("x", Printf.sprintf "%.2f" bx);
                  ("y", Printf.sprintf "%.2f" by);
                  ("width", Printf.sprintf "%.2f" box_s);
                  ("height", Printf.sprintf "%.2f" box_s);
                  ("fill", "transparent");
                  ("class", "py-svg-overlay");
                  ("pointer-events", "all");
                  ("style", "cursor: crosshair;");
                ]
            in
            p_overlay##.onmousemove :=
              Dom.handler (fun ev ->
                  let rect = p_overlay##getBoundingClientRect in
                  let r_left = Js.float_of_number rect##.left in
                  let r_top = Js.float_of_number rect##.top in
                  let bw = Js.float_of_number rect##.right -. r_left in
                  let bh = Js.float_of_number rect##.bottom -. r_top in
                  if bw > 0.0 && bh > 0.0 then (
                    let frac_x = max 0.0 (min 0.999999 ((Js.float_of_number ev##.clientX -. r_left) /. bw)) in
                    let frac_y = max 0.0 (min 0.999999 ((Js.float_of_number ev##.clientY -. r_top) /. bh)) in
                    let ix = max 0 (min (nx - 1) (int_of_float (frac_x *. float_of_int nx))) in
                    let iz_inv = int_of_float (frac_y *. float_of_int nz) in
                    let iz = max 0 (min (nz - 1) (nz - 1 - iz_inv)) in
                    let idx = iz * nx + ix in
                    if idx >= 0 && idx < Array.length vals then
                      set_text readout_el
                        (Printf.sprintf "%s (%d,%d) = %.3f" label ix iz vals.(idx)));
                  Js._true);
            p_overlay##.onmouseout :=
              Dom.handler (fun _ ->
                  set_text readout_el "";
                  Js._true);
            Dom.appendChild svg p_overlay))
      panels;
    (* Diverging colorbar strip at bottom *)
    let bar_w = 180.0 and bar_x = 0.5 *. (w -. 180.0) and bar_y = h -. 15.0 in
    for b = 0 to 15 do
      let bw = bar_w /. 16.0 in
      Dom.appendChild svg
        (svg_el "rect"
           [
             ("x", Printf.sprintf "%.2f" (bar_x +. float_of_int b *. bw));
             ("y", Printf.sprintf "%.2f" bar_y);
             ("width", Printf.sprintf "%.2f" (bw +. 0.4));
             ("height", "7.0");
             ("class", Printf.sprintf "py-heat-%d" b);
           ])
    done;
    let first_p = List.hd panels in
    let cmin = Option.value (opt_float first_p "vmin") ~default:(-1.0) in
    let cmax = Option.value (opt_float first_p "vmax") ~default:1.0 in
    let t_lo =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (bar_x -. 5.0));
          ("y", Printf.sprintf "%.1f" (bar_y +. 6.5));
          ("text-anchor", "end");
          ("class", "py-svg-tick");
        ]
    in
    set_text t_lo (Printf.sprintf "%.2g" cmin);
    Dom.appendChild svg t_lo;
    let t_hi =
      svg_el "text"
        [
          ("x", Printf.sprintf "%.1f" (bar_x +. bar_w +. 5.0));
          ("y", Printf.sprintf "%.1f" (bar_y +. 6.5));
          ("text-anchor", "start");
          ("class", "py-svg-tick");
        ]
    in
    set_text t_hi (Printf.sprintf "%.2g" cmax);
    Dom.appendChild svg t_hi;
    Dom.appendChild stage svg;
    card

let render_band1d_card (fig : Js.Unsafe.any) =
  let title = Option.value (opt_str fig "title") ~default:"1D Posterior & Uncertainty" in
  let subtitle = Option.value (opt_str fig "subtitle") ~default:"" in
  let card, _hdr_actions, readout_el, stage = make_fig_pane title subtitle in
  let xg = json_num_list (Js.Unsafe.get fig (js "x_grid")) in
  let mu = json_num_list (Js.Unsafe.get fig (js "mu")) in
  if xg = [] || List.length xg <> List.length mu then card
  else
    let y_true = json_num_list (Js.Unsafe.get fig (js "y_true")) in
    let lo_2se = json_num_list (Js.Unsafe.get fig (js "lo_2se")) in
    let hi_2se = json_num_list (Js.Unsafe.get fig (js "hi_2se")) in
    let mw_lo = json_num_list (Js.Unsafe.get fig (js "mw_lo")) in
    let mw_hi = json_num_list (Js.Unsafe.get fig (js "mw_hi")) in
    let tr_x = json_num_list (Js.Unsafe.get fig (js "train_x")) in
    let tr_y = json_num_list (Js.Unsafe.get fig (js "train_y")) in
    let legend = mk ~cls:"py-fig-legend" "div" in
    append_legend_badge legend "py-swatch-0" "ENN mean";
    if y_true <> [] then append_legend_badge legend "py-swatch-1 is-dashed" "True f(x)";
    if lo_2se <> [] && hi_2se <> [] then append_legend_badge legend "py-swatch-band" "mu +/- 2*se";
    if mw_lo <> [] && mw_hi <> [] then append_legend_badge legend "py-swatch-2 is-dashed" "McShane-Whitney";
    Dom.appendChild stage legend;
    let all_y = mu @ y_true @ lo_2se @ hi_2se @ mw_lo @ mw_hi @ tr_y in
    let x_lo = List.fold_left min (List.hd xg) xg in
    let x_hi = List.fold_left max (List.hd xg) xg in
    let x_sp = if x_hi -. x_lo < 1e-9 then 1.0 else x_hi -. x_lo in
    let y_lo_r = List.fold_left min (List.hd all_y) all_y in
    let y_hi_r = List.fold_left max (List.hd all_y) all_y in
    let y_sp_r = if y_hi_r -. y_lo_r < 1e-9 then 1.0 else y_hi_r -. y_lo_r in
    let y_lo = y_lo_r -. 0.06 *. y_sp_r in
    let y_hi = y_hi_r +. 0.06 *. y_sp_r in
    let y_sp = y_hi -. y_lo in
    let w = 460.0 and h = 180.0 in
    let pl = 56.0 and pr = 16.0 and pt = 16.0 and pb = 28.0 in
    let pw = w -. pl -. pr and ph = h -. pt -. pb in
    let x_px x = pl +. ((x -. x_lo) /. x_sp) *. pw in
    let y_px y = pt +. (1.0 -. ((y -. y_lo) /. y_sp)) *. ph in
    let svg =
      svg_el "svg"
        [
          ("viewBox", Printf.sprintf "0 0 %.0f %.0f" w h);
          ("class", "py-traj-svg");
          ("role", "img");
          ("aria-label", title);
        ]
    in
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let yy = pt +. (1.0 -. frac) *. ph in
      let v = y_lo +. frac *. y_sp in
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
      set_text lbl (Printf.sprintf "%.2g" v);
      Dom.appendChild svg lbl
    done;
    let pts_of xs ys =
      let rec go ax ay acc =
        match (ax, ay) with
        | x :: xt, y :: yt -> go xt yt (Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y) :: acc)
        | _ -> List.rev acc
      in
      String.concat " " (go xs ys [])
    in
    if List.length lo_2se = List.length xg && List.length hi_2se = List.length xg then (
      let top =
        List.map2 (fun x y -> Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y)) xg hi_2se
      in
      let bot =
        List.rev
          (List.map2 (fun x y -> Printf.sprintf "%.2f,%.2f" (x_px x) (y_px y)) xg lo_2se)
      in
      Dom.appendChild svg
        (svg_el "polygon"
           [ ("points", String.concat " " (top @ bot)); ("class", "py-svg-band-fill") ]));
    if List.length mw_hi = List.length xg && List.length mw_lo = List.length xg then (
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg mw_hi); ("class", "py-svg-mw-line") ]);
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg mw_lo); ("class", "py-svg-mw-line") ]));
    if List.length y_true = List.length xg then
      Dom.appendChild svg
        (svg_el "polyline"
           [ ("fill", "none"); ("points", pts_of xg y_true); ("class", "py-svg-band-true") ]);
    Dom.appendChild svg
      (svg_el "polyline"
         [ ("fill", "none"); ("points", pts_of xg mu); ("class", "py-svg-series py-svg-series-0") ]);
    let rec iter_pairs f xs ys =
      match (xs, ys) with
      | x :: xt, y :: yt ->
          f x y;
          iter_pairs f xt yt
      | _ -> ()
    in
    iter_pairs
      (fun x y ->
        Dom.appendChild svg
          (svg_el "circle"
             [
               ("cx", Printf.sprintf "%.2f" (x_px x));
               ("cy", Printf.sprintf "%.2f" (y_px y));
               ("r", "2.6");
               ("class", "py-svg-obs-dot-strong");
             ]))
      tr_x tr_y;
    Dom.appendChild svg
      (svg_el "line"
         [
           ("x1", Printf.sprintf "%.1f" pl);
           ("x2", Printf.sprintf "%.1f" (w -. pr));
           ("y1", Printf.sprintf "%.1f" (h -. pb));
           ("y2", Printf.sprintf "%.1f" (h -. pb));
           ("class", "py-svg-axis");
         ]);
    for g = 0 to 3 do
      let frac = float_of_int g /. 3.0 in
      let xx = pl +. frac *. pw in
      let xv = x_lo +. frac *. x_sp in
      let xtl =
        svg_el "text"
          [
            ("x", Printf.sprintf "%.1f" xx);
            ("y", Printf.sprintf "%.1f" (h -. pb +. 12.0));
            ("text-anchor", "middle");
            ("class", "py-svg-tick");
          ]
      in
      set_text xtl (Printf.sprintf "%.3g" xv);
      Dom.appendChild svg xtl
    done;
    let crosshair =
      svg_el "line"
        [
          ("y1", Printf.sprintf "%.1f" pt);
          ("y2", Printf.sprintf "%.1f" (pt +. ph));
          ("class", "py-svg-crosshair");
          ("style", "display: none; pointer-events: none;");
        ]
    in
    Dom.appendChild svg crosshair;
    let hover_dot =
      svg_el "circle"
        [
          ("r", "4.0");
          ("class", "py-svg-hover-dot py-svg-series-dot-0");
          ("style", "display: none; pointer-events: none;");
        ]
    in
    Dom.appendChild svg hover_dot;
    let xg_arr = Array.of_list xg in
    let mu_arr = Array.of_list mu in
    let lo_arr =
      if List.length lo_2se = Array.length xg_arr then Array.of_list lo_2se
      else if List.length mw_lo = Array.length xg_arr then Array.of_list mw_lo
      else mu_arr
    in
    let hi_arr =
      if List.length hi_2se = Array.length xg_arr then Array.of_list hi_2se
      else if List.length mw_hi = Array.length xg_arr then Array.of_list mw_hi
      else mu_arr
    in
    let overlay =
      svg_el "rect"
        [
          ("x", Printf.sprintf "%.1f" pl);
          ("y", Printf.sprintf "%.1f" pt);
          ("width", Printf.sprintf "%.1f" pw);
          ("height", Printf.sprintf "%.1f" ph);
          ("fill", "transparent");
          ("class", "py-svg-overlay");
          ("pointer-events", "all");
          ("style", "cursor: crosshair;");
        ]
    in
    overlay##.onmousemove :=
      Dom.handler (fun ev ->
          let rect = overlay##getBoundingClientRect in
          let r_left = Js.float_of_number rect##.left in
          let ow = Js.float_of_number rect##.right -. r_left in
          if ow > 0.0 then (
            let rx = max 0.0 (min 1.0 ((Js.float_of_number ev##.clientX -. r_left) /. ow)) in
            let data_x = x_lo +. rx *. x_sp in
            let best_idx = ref 0 in
            let best_dist = ref infinity in
            for i = 0 to Array.length xg_arr - 1 do
              let d = abs_float (xg_arr.(i) -. data_x) in
              if d < !best_dist then (
                best_dist := d;
                best_idx := i)
            done;
            let idx = !best_idx in
            let x = xg_arr.(idx) in
            let m = mu_arr.(idx) in
            let lo = lo_arr.(idx) in
            let hi = hi_arr.(idx) in
            let sx = x_px x in
            let sy = y_px m in
            crosshair##setAttribute (js "x1") (js (Printf.sprintf "%.2f" sx));
            crosshair##setAttribute (js "x2") (js (Printf.sprintf "%.2f" sx));
            crosshair##.style##.display := js "";
            hover_dot##setAttribute (js "cx") (js (Printf.sprintf "%.2f" sx));
            hover_dot##setAttribute (js "cy") (js (Printf.sprintf "%.2f" sy));
            hover_dot##.style##.display := js "";
            set_text readout_el (Printf.sprintf "x=%.3f · mu=%.4g · [%.3g, %.3g]" x m lo hi));
          Js._true);
    overlay##.onmouseout :=
      Dom.handler (fun _ ->
          crosshair##.style##.display := js "none";
          hover_dot##.style##.display := js "none";
          set_text readout_el "";
          Js._true);
    Dom.appendChild svg overlay;
    Dom.appendChild stage svg;
    card

let append_telemetry_band (panel : element Js.t) (state_raw : Js.Unsafe.any) =
  let band = mk ~cls:"py-telemetry-band" "div" in
  let stats = mk ~cls:"py-telemetry-stats" "div" in
  let vname = Option.value (opt_str state_raw "var") ~default:"state" in
  Dom.appendChild stats (mk ~cls:"py-telemetry-var" ~txt:vname "span");
  let add_stat label value =
    let item = mk ~cls:"py-telemetry-item" "span" in
    Dom.appendChild item (mk ~cls:"py-telemetry-lbl" ~txt:label "span");
    Dom.appendChild item (mk ~cls:"py-telemetry-val" ~txt:value "strong");
    Dom.appendChild stats item
  in
  add_stat "obs" (string_of_int (Option.value (opt_int state_raw "num_observations") ~default:0));
  add_stat "epoch" (string_of_int (Option.value (opt_int state_raw "epoch") ~default:0));
  add_stat "regions" (string_of_int (Option.value (opt_int state_raw "num_regions") ~default:1));
  add_stat "tr" (match opt_float state_raw "tr_length" with Some f -> Printf.sprintf "%.4g" f | None -> "-");
  add_stat "best" (match opt_float state_raw "best_y" with Some f -> Printf.sprintf "%.5f" f | None -> "-");
  let pend = Option.value (opt_int state_raw "pending_count") ~default:0 in
  if pend > 0 then add_stat "pending" (string_of_int pend);
  Dom.appendChild band stats;
  let best_vec_opt =
    match opt_str state_raw "best_params" with
    | Some bp -> Some ("best_params", bp)
    | None -> (
        let bx = json_num_list (Js.Unsafe.get state_raw (js "best_x")) in
        if bx <> [] then
          Some ("best_x", "[" ^ String.concat ", " (List.map (Printf.sprintf "%.4f") bx) ^ "]")
        else None)
  in
  (match best_vec_opt with
  | Some (lbl, txt) ->
      let row = mk ~cls:"py-best-row" "div" in
      Dom.appendChild row (mk ~cls:"py-telemetry-lbl" ~txt:lbl "span");
      let code_el = mk ~cls:"py-best-inline" ~txt "code" in
      let exp_btn = mk ~cls:"py-best-toggle" ~txt:"Expand" "button" in
      exp_btn##setAttribute (js "type") (js "button");
      exp_btn##.onclick :=
        Dom.handler (fun _ ->
            let cur = to_s code_el##.className in
            if contains_sub cur "is-expanded" then (
              code_el##.className := js "py-best-inline";
              set_text exp_btn "Expand")
            else (
              code_el##.className := js "py-best-inline is-expanded";
              set_text exp_btn "Collapse");
            Js._true);
      let cpy_btn = mk ~cls:"py-best-toggle" ~txt:"Copy" "button" in
      cpy_btn##setAttribute (js "type") (js "button");
      cpy_btn##.onclick :=
        Dom.handler (fun _ ->
            let clip = Js.Unsafe.get Dom_html.window##.navigator (js "clipboard") in
            if is_defined clip then (
              ignore (Js.Unsafe.meth_call clip "writeText" [| Js.Unsafe.inject (js txt) |]);
              set_text cpy_btn "Copied";
              ignore
                (Dom_html.window##setTimeout
                   (Js.wrap_callback (fun () -> set_text cpy_btn "Copy"))
                   (Js.float 1100.0)));
            Js._true);
      Dom.appendChild row code_el;
      Dom.appendChild row exp_btn;
      Dom.appendChild row cpy_btn;
      Dom.appendChild band row
  | None -> ());
  Dom.appendChild panel band

let render_fig_split (panel : element Js.t) (state_raw : Js.Unsafe.any) (fig : Js.Unsafe.any) =
  let has_opt = is_defined (Js.Unsafe.get state_raw (js "dim")) in
  let kind = Option.value (opt_str fig "kind") ~default:"" in
  let append_subcurves () =
    match curves_from_js (Js.Unsafe.get fig (js "curves")) with
    | Some c -> Dom.appendChild panel c
    | None -> ()
  in
  match kind with
  | "curves" ->
      (match curves_from_js fig with Some c -> Dom.appendChild panel c | None -> ());
      if has_opt then append_telemetry_band panel state_raw
  | "tokamak" ->
      Dom.appendChild panel (render_tokamak_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | "field2d" ->
      Dom.appendChild panel (render_field2d_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | "band1d" ->
      Dom.appendChild panel (render_band1d_card fig);
      if has_opt then append_telemetry_band panel state_raw;
      append_subcurves ()
  | _ -> if has_opt then append_telemetry_band panel state_raw

let render_state_panel ?empty_node (panel : element Js.t) (state_raw : Js.Unsafe.any) =
  clear_children (panel :> Dom.node Js.t);
  if not (is_defined state_raw) then (
    match empty_node with
    | Some node -> Dom.appendChild panel node
    | None ->
        let hint =
          mk ~cls:"py-inspector-empty" ~txt:"No figures or state yet." "p"
        in
        Dom.appendChild panel hint)
  else
    let fig_raw = Js.Unsafe.get state_raw (js "fig") in
    let has_opt = is_defined (Js.Unsafe.get state_raw (js "dim")) in
    (if is_defined fig_raw then render_fig_split panel state_raw fig_raw
     else if has_opt then (
       let y_hist = json_num_list (Js.Unsafe.get state_raw (js "y_history")) in
       let best_hist = json_num_list (Js.Unsafe.get state_raw (js "best_history")) in
       if best_hist <> [] then
         let xs = List.mapi (fun i _ -> float_of_int (i + 1)) best_hist in
         let series =
           [ { c_label = "Best y"; c_xs = xs; c_ys = best_hist; c_dashed = false } ]
           @
           if List.length y_hist = List.length xs then
             [ { c_label = "Raw eval y"; c_xs = xs; c_ys = y_hist; c_dashed = true } ]
           else []
         in
         Dom.appendChild panel
           (render_curves_card ~title:"Cumulative Best Objective"
              ~subtitle:"Best-so-far vs raw observations" ~x_label:"Evaluations"
              ~y_label:"Objective" ~log_y:false series);
       append_telemetry_band panel state_raw));

    let recent_raw = Js.Unsafe.get state_raw (js "recent") in
    if Js.Optdef.test recent_raw then (
      let rows : Js.Unsafe.any Js.js_array Js.t = Obj.magic recent_raw in
      if rows##.length > 0 then (
        let det = mk ~cls:"py-recent-details" "details" in
        let sum =
          mk ~cls:"py-recent-summary"
            ~txt:(Printf.sprintf "Recent evaluations (last %d)" rows##.length)
            "summary"
        in
        Dom.appendChild det sum;
        let tbl_wrap = mk ~cls:"py-recent-wrap" "div" in
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
        Dom.appendChild tbl_wrap tbl;
        Dom.appendChild det tbl_wrap;
        Dom.appendChild panel det))

let render_scope_panel ~on_inspect (panel : element Js.t) (scope_raw : Js.Unsafe.any) =
  clear_children (panel :> Dom.node Js.t);
  let items = json_obj_list scope_raw in
  if items = [] then (
    let wrap = mk ~cls:"py-scope-empty" "div" in
    Dom.appendChild wrap
      (mk ~cls:"py-inspector-empty"
         ~txt:"No user variables in scope yet. Run the script or evaluate an expression in >>>."
         "p");
    Dom.appendChild panel wrap)
  else (
    let wrap = mk ~cls:"py-scope-wrap" "div" in
    let hdr = mk ~cls:"py-scope-note" ~txt:"Click any variable to inspect in >>>" "div" in
    Dom.appendChild wrap hdr;
    let tbl = mk ~cls:"py-scope-table" "table" in
    let thead = mk "thead" in
    let hr = mk "tr" in
    List.iter (fun h -> Dom.appendChild hr (mk ~txt:h "th")) [ "Name"; "Type"; "Value / Shape" ];
    Dom.appendChild thead hr;
    Dom.appendChild tbl thead;
    let tbody = mk "tbody" in
    List.iter
      (fun item ->
        let name = Option.value (opt_str item "name") ~default:"" in
        let typ = Option.value (opt_str item "type") ~default:"" in
        let summary = Option.value (opt_str item "summary") ~default:"" in
        let kind = Option.value (opt_str item "kind") ~default:"obj" in
        if name <> "" then (
          let tr = mk ~cls:("py-scope-row is-" ^ kind) "tr" in
          let td_name = mk "td" in
          let btn = mk ~cls:"py-scope-name" ~txt:name "button" in
          btn##setAttribute (js "type") (js "button");
          btn##setAttribute (js "title") (js ("Inspect " ^ name ^ " in console"));
          btn##.onclick :=
            Dom.handler (fun _ ->
                on_inspect name;
                Js._true);
          Dom.appendChild td_name btn;
          let td_typ = mk ~cls:"py-scope-type" ~txt:typ "td" in
          let td_sum = mk ~cls:"py-scope-val" ~txt:summary "td" in
          Dom.appendChild tr td_name;
          Dom.appendChild tr td_typ;
          Dom.appendChild tr td_sum;
          Dom.appendChild tbody tr))
      items;
    Dom.appendChild tbl tbody;
    Dom.appendChild wrap tbl;
    Dom.appendChild panel wrap)

(* ---------------------------------------------------------------------------
   Pure-OCaml Python Syntax Tokenizer & Synchronized Line-Number Gutter
   --------------------------------------------------------------------------- *)

let is_ident_start c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'

let is_ident_char c =
  is_ident_start c || (c >= '0' && c <= '9')

let is_digit c = c >= '0' && c <= '9'

let is_py_keyword = function
  | "and" | "as" | "assert" | "async" | "await" | "break" | "class"
  | "continue" | "def" | "del" | "elif" | "else" | "except" | "False"
  | "finally" | "for" | "from" | "global" | "if" | "import" | "in"
  | "is" | "lambda" | "None" | "nonlocal" | "not" | "or" | "pass"
  | "raise" | "return" | "True" | "try" | "while" | "with" | "yield" ->
      true
  | _ -> false

let is_py_builtin = function
  | "abs" | "all" | "any" | "bool" | "dict" | "enumerate" | "float" | "int"
  | "isinstance" | "len" | "list" | "max" | "min" | "next" | "open" | "print"
  | "range" | "round" | "set" | "slice" | "sorted" | "str" | "sum" | "tuple"
  | "zip" ->
      true
  | _ -> false

let is_decorator_start src i =
  if i + 1 >= String.length src || not (is_ident_start src.[i + 1]) then false
  else
    let rec check k =
      if k < 0 then true
      else if src.[k] = '\n' then true
      else if src.[k] = ' ' || src.[k] = '\t' then check (k - 1)
      else false
    in
    check (i - 1)

let render_py_highlight (code_el : element Js.t) (src : string) =
  clear_children (code_el :> Dom.node Js.t);
  let len = String.length src in
  let plain_buf = Buffer.create 128 in
  let flush_plain () =
    if Buffer.length plain_buf > 0 then (
      let tn = document##createTextNode (js (Buffer.contents plain_buf)) in
      Dom.appendChild code_el tn;
      Buffer.clear plain_buf)
  in
  let emit_span cls txt =
    flush_plain ();
    Dom.appendChild code_el (mk ~cls ~txt "span")
  in
  let rec scan i expect_def =
    if i >= len then ()
    else
      let c = src.[i] in
      if c = '#' then (
        let j = ref (i + 1) in
        while !j < len && src.[!j] <> '\n' do
          incr j
        done;
        emit_span "code-comment" (String.sub src i (!j - i));
        scan !j false)
      else if c = '@' && is_decorator_start src i then (
        let j = ref (i + 1) in
        while !j < len && (is_ident_char src.[!j] || src.[!j] = '.') do
          incr j
        done;
        emit_span "code-decorator" (String.sub src i (!j - i));
        scan !j false)
      else if c = '\'' || c = '"' then (
        let q = c in
        let is_triple = i + 2 < len && src.[i + 1] = q && src.[i + 2] = q in
        let j = ref (i + if is_triple then 3 else 1) in
        let done_s = ref false in
        while !j < len && not !done_s do
          if src.[!j] = '\\' then j := min len (!j + 2)
          else if is_triple then
            if !j + 2 < len && src.[!j] = q && src.[!j + 1] = q && src.[!j + 2] = q then (
              j := !j + 3;
              done_s := true)
            else incr j
          else if src.[!j] = q || src.[!j] = '\n' then (
            if src.[!j] = q then incr j;
            done_s := true)
          else incr j
        done;
        emit_span "code-string" (String.sub src i (!j - i));
        scan !j false)
      else if
        (c = 'f' || c = 'r' || c = 'b' || c = 'u' || c = 'F' || c = 'R' || c = 'B' || c = 'U')
        && i + 1 < len
        && (src.[i + 1] = '\'' || src.[i + 1] = '"')
      then (
        Buffer.add_char plain_buf c;
        scan (i + 1) false)
      else if is_digit c || (c = '.' && i + 1 < len && is_digit src.[i + 1]) then (
        let j = ref i in
        if
          c = '0'
          && i + 1 < len
          && (src.[i + 1] = 'x' || src.[i + 1] = 'X' || src.[i + 1] = 'b' || src.[i + 1] = 'o')
        then (
          j := i + 2;
          while !j < len && is_ident_char src.[!j] do
            incr j
          done)
        else (
          while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
            incr j
          done;
          if !j < len && src.[!j] = '.' then (
            incr j;
            while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
              incr j
            done);
          if !j < len && (src.[!j] = 'e' || src.[!j] = 'E') then (
            let e_pos = !j in
            incr j;
            if !j < len && (src.[!j] = '+' || src.[!j] = '-') then incr j;
            if !j < len && is_digit src.[!j] then
              while !j < len && (is_digit src.[!j] || src.[!j] = '_') do
                incr j
              done
            else j := e_pos));
        emit_span "code-number" (String.sub src i (!j - i));
        scan !j false)
      else if is_ident_start c then (
        let j = ref (i + 1) in
        while !j < len && is_ident_char src.[!j] do
          incr j
        done;
        let word = String.sub src i (!j - i) in
        if is_py_keyword word then (
          emit_span "code-keyword" word;
          let next_def = word = "def" || word = "class" in
          scan !j next_def)
        else if expect_def then (
          emit_span "code-def" word;
          scan !j false)
        else if is_py_builtin word then (
          emit_span "code-builtin" word;
          scan !j false)
        else (
          Buffer.add_string plain_buf word;
          scan !j false))
      else if expect_def && (c = ' ' || c = '\t') then (
        Buffer.add_char plain_buf c;
        scan (i + 1) true)
      else (
        Buffer.add_char plain_buf c;
        scan (i + 1) false)
  in
  scan 0 false;
  flush_plain ();
  Dom.appendChild code_el (document##createTextNode (js "\n"))

let count_lines s =
  let n = ref 1 in
  String.iter (fun ch -> if ch = '\n' then incr n) s;
  !n

let sync_gutter_highlights (gutter : element Js.t) ~(active : int option) ~(err : int option) =
  let lines = gutter##.childNodes in
  let len = lines##.length in
  for i = 0 to len - 1 do
    match Js.Opt.to_option (lines##item i) with
    | Some node ->
        let el : element Js.t = Obj.magic node in
        let line_no = i + 1 in
        let is_act = match active with Some a -> a = line_no | None -> false in
        let is_err = match err with Some e -> e = line_no | None -> false in
        let cls =
          "py-ed-gutter-line"
          ^ (if is_act then " is-active-line" else "")
          ^ (if is_err then " is-err-line" else "")
        in
        el##.className := js cls
    | None -> ()
  done

let render_line_numbers (gutter : element Js.t) prev_ref ?(active = None) ?(err = None) n =
  if n <> !prev_ref then (
    prev_ref := n;
    clear_children (gutter :> Dom.node Js.t);
    for k = 1 to n do
      let line_el = mk ~cls:"py-ed-gutter-line" ~txt:(string_of_int k) "div" in
      line_el##setAttribute (js "data-line") (js (string_of_int k));
      Dom.appendChild gutter line_el
    done);
  sync_gutter_highlights gutter ~active ~err

let cursor_line_no (v : string) (cursor_pos : int) : int =
  let len = String.length v in
  let pos = min len (max 0 cursor_pos) in
  let line = ref 1 in
  for i = 0 to pos - 1 do
    if v.[i] = '\n' then incr line
  done;
  !line

let find_line_char_offset (v : string) (target_line : int) : int =
  let len = String.length v in
  let line = ref 1 in
  let idx = ref 0 in
  while !idx < len && !line < target_line do
    if v.[!idx] = '\n' then incr line;
    incr idx
  done;
  !idx

let longest_common_prefix (items : string list) : string =
  match items with
  | [] -> ""
  | [ s ] -> s
  | first :: rest ->
      let prefix_len = ref (String.length first) in
      List.iter
        (fun s ->
          let slen = String.length s in
          let m = min !prefix_len slen in
          let j = ref 0 in
          while !j < m && first.[!j] = s.[!j] do
            incr j
          done;
          prefix_len := !j)
        rest;
      String.sub first 0 !prefix_len

let parse_line_err_number (line : string) : int option =
  let needle = "File \"<playground>\", line " in
  let nlen = String.length needle in
  let slen = String.length line in
  let rec scan i =
    if i + nlen > slen then None
    else if String.sub line i nlen = needle then
      let num_start = i + nlen in
      let j = ref num_start in
      while !j < slen && line.[!j] >= '0' && line.[!j] <= '9' do
        incr j
      done;
      if !j > num_start then
        try Some (int_of_string (String.sub line num_start (!j - num_start)))
        with _ -> None
      else None
    else scan (i + 1)
  in
  scan 0

let append_console_stderr (out_pre : element Js.t) (text : string) (on_jump : int -> unit) =
  if String.length text > 0 then (
    let lines = String.split_on_char '\n' text in
    let last_err_line = ref None in
    let rec iter_lines = function
      | [] -> ()
      | [ "" ] -> ()
      | l :: rest ->
          let row = mk ~cls:"py-out-stderr-line" "div" in
          let sp = mk ~cls:"py-out-stderr" ~txt:(l ^ "\n") "span" in
          Dom.appendChild row sp;
          (match parse_line_err_number l with
          | Some n ->
              last_err_line := Some n;
              let jump_btn =
                mk ~cls:"py-err-link" ~txt:(Printf.sprintf "Jump to line %d" n) "button"
              in
              jump_btn##setAttribute (js "type") (js "button");
              jump_btn##setAttribute (js "title") (js (Printf.sprintf "Jump to line %d in editor" n));
              jump_btn##.onclick :=
                Dom.handler (fun _ ->
                    on_jump n;
                    Js._true);
              Dom.appendChild row jump_btn
          | None -> ());
          Dom.appendChild out_pre row;
          iter_lines rest
    in
    iter_lines lines;
    out_pre##.scrollTop := Js.float (float_of_int out_pre##.scrollHeight);
    !last_err_line)
  else None

(* ---------------------------------------------------------------------------
   Full Playground Workbench UI (#py-playground-root)
   --------------------------------------------------------------------------- *)

let append_console_chunk (out_pre : element Js.t) cls text =
  if String.length text > 0 then (
    let sp = mk ~cls ~txt:text "span" in
    Dom.appendChild out_pre sp;
    out_pre##.scrollTop := Js.float (float_of_int out_pre##.scrollHeight))

let resolve_url_preset_id () =
  try
    let url_ctor = Js.Unsafe.get Dom_html.window (js "URL") in
    let url_obj =
      Js.Unsafe.new_obj url_ctor [| Js.Unsafe.inject Dom_html.window##.location##.href |]
    in
    let params = Js.Unsafe.get url_obj (js "searchParams") in
    let q_ex = Js.Unsafe.meth_call params "get" [| Js.Unsafe.inject (js "example") |] in
    let q_p = Js.Unsafe.meth_call params "get" [| Js.Unsafe.inject (js "p") |] in
    let raw =
      if Js.Opt.test q_ex then Some (to_s (Obj.magic q_ex))
      else if Js.Opt.test q_p then Some (to_s (Obj.magic q_p))
      else
        let h = to_s Dom_html.window##.location##.hash in
        let h_clean =
          if String.length h > 0 && h.[0] = '#' then String.sub h 1 (String.length h - 1) else h
        in
        if h_clean <> "" then Some h_clean else None
    in
    match raw with
    | Some cand -> (match find_preset cand with Some p -> Some p.id | None -> None)
    | None -> None
  with _ -> None

let sync_url_preset (id_opt : string option) =
  try
    let url_ctor = Js.Unsafe.get Dom_html.window (js "URL") in
    let url_obj =
      Js.Unsafe.new_obj url_ctor [| Js.Unsafe.inject Dom_html.window##.location##.href |]
    in
    let params = Js.Unsafe.get url_obj (js "searchParams") in
    (match id_opt with
    | Some pid ->
        ignore
          (Js.Unsafe.meth_call params "set"
             [| Js.Unsafe.inject (js "example"); Js.Unsafe.inject (js pid) |])
    | None ->
        ignore
          (Js.Unsafe.meth_call params "delete" [| Js.Unsafe.inject (js "example") |]));
    let next_href : Js.js_string Js.t = Js.Unsafe.get url_obj (js "href") in
    ignore
      (Js.Unsafe.meth_call Dom_html.window##.history "replaceState"
         [| Js.Unsafe.inject Js.null; Js.Unsafe.inject (js ""); Js.Unsafe.inject next_href |])
  with _ -> ()

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
  | None -> (
      let picked =
        match resolve_url_preset_id () with
        | Some pid -> Option.value (find_preset pid) ~default:(List.hd presets)
        | None -> List.hd presets
      in
      (Some picked.id, picked.code))

let make_chevron_svg cls points =
  let svg =
    svg_el "svg"
      [
        ("class", cls);
        ("viewBox", "0 0 10 10");
        ("width", "10");
        ("height", "10");
        ("fill", "none");
        ("stroke", "currentColor");
        ("stroke-width", "1.5");
        ("stroke-linecap", "round");
        ("stroke-linejoin", "round");
        ("aria-hidden", "true");
      ]
  in
  Dom.appendChild svg (svg_el "polyline" [ ("points", points) ]);
  svg

let mount_playground (root : element Js.t) =
  clear_children (root :> Dom.node Js.t);
  let shell = mk ~cls:"py-pg-app" "div" in
  shell##setAttribute (js "data-mob-tab") (js "script");

  (* Band 1: Unified Top Bar (Editorial Example Switcher + Kernel Status + Controls) *)
  let topbar = mk ~cls:"py-pg-topbar" "div" in
  let ex_control = mk ~cls:"py-ex-control" "div" in
  let ex_trigger = mk ~cls:"py-ex-trigger" "button" in
  ex_trigger##setAttribute (js "type") (js "button");
  ex_trigger##setAttribute (js "aria-haspopup") (js "listbox");
  ex_trigger##setAttribute (js "aria-expanded") (js "false");
  ex_trigger##setAttribute (js "aria-controls") (js "py-ex-menu");
  let ex_count_el = mk ~cls:"py-ex-count" ~txt:"01/07" "span" in
  let ex_title_el = mk ~cls:"py-ex-title" ~txt:(List.hd presets).label "span" in
  let ex_caret_el = make_chevron_svg "py-ex-caret" "2,3.5 5,6.5 8,3.5" in
  Dom.appendChild ex_trigger ex_count_el;
  Dom.appendChild ex_trigger ex_title_el;
  Dom.appendChild ex_trigger ex_caret_el;

  let prev_btn = mk ~cls:"py-ex-step py-ex-prev" "button" in
  prev_btn##setAttribute (js "type") (js "button");
  prev_btn##setAttribute (js "aria-label") (js "Previous example");
  prev_btn##setAttribute (js "title") (js "Previous example");
  Dom.appendChild prev_btn (make_chevron_svg "py-ex-step-ico" "6.5,2 3.5,5 6.5,8");

  let next_btn = mk ~cls:"py-ex-step py-ex-next" "button" in
  next_btn##setAttribute (js "type") (js "button");
  next_btn##setAttribute (js "aria-label") (js "Next example");
  next_btn##setAttribute (js "title") (js "Next example");
  Dom.appendChild next_btn (make_chevron_svg "py-ex-step-ico" "3.5,2 6.5,5 3.5,8");

  Dom.appendChild ex_control ex_trigger;
  Dom.appendChild ex_control prev_btn;
  Dom.appendChild ex_control next_btn;

  let ex_menu = mk ~cls:"py-ex-menu" "div" in
  ex_menu##setAttribute (js "id") (js "py-ex-menu");
  ex_menu##setAttribute (js "role") (js "listbox");
  ex_menu##setAttribute (js "aria-label") (js "REPL examples");
  ex_menu##setAttribute (js "hidden") (js "hidden");
  let ex_menu_hdr = mk ~cls:"py-ex-menu-hdr" "div" in
  Dom.appendChild ex_menu_hdr (mk ~cls:"py-ex-menu-hdr-title" ~txt:"Examples" "span");
  Dom.appendChild ex_menu_hdr (mk ~cls:"py-ex-menu-hdr-hint" ~txt:"↑↓ · Enter · Esc" "span");
  let ex_menu_list = mk ~cls:"py-ex-menu-list" "div" in
  Dom.appendChild ex_menu ex_menu_hdr;
  Dom.appendChild ex_menu ex_menu_list;

  let controls = mk ~cls:"py-pg-controls" "div" in
  let mode_sel = createSelect document in
  mode_sel##.className := js "py-btn";
  mode_sel##setAttribute (js "aria-label") (js "Python runtime");
  mode_sel##setAttribute (js "hidden") (js "hidden");
  mode_sel##.style##.display := js "none";
  let wasm_opt = createOption document in
  wasm_opt##.value := js "wasm";
  set_text wasm_opt "Browser";
  let native_opt = createOption document in
  native_opt##.value := js "local_native";
  set_text native_opt "Local";
  Js.Unsafe.set native_opt (js "disabled") Js._true;
  native_opt##setAttribute (js "disabled") (js "disabled");
  Dom.appendChild mode_sel wasm_opt;
  Dom.appendChild mode_sel native_opt;

  let runtime_badge = mk ~cls:"py-runtime-badge" ~txt:"WASM" "span" in
  runtime_badge##setAttribute (js "title") (js "In-browser Pyodide WASM runtime");

  let status_pill = mk ~cls:"py-kernel-pill" "div" in
  let status_dot = mk ~cls:"py-status-dot" "span" in
  let status_lbl = mk ~cls:"py-status-text" ~txt:"Idle" "span" in
  let timing_lbl = mk ~cls:"py-timing-badge" ~txt:"" "span" in
  Dom.appendChild status_pill status_dot;
  Dom.appendChild status_pill status_lbl;
  Dom.appendChild status_pill timing_lbl;

  let run_btn = mk ~cls:"py-btn py-btn-primary" ~txt:"Run" "button" in
  run_btn##setAttribute (js "type") (js "button");
  run_btn##setAttribute (js "title") (js "Run script (Ctrl/Cmd + Enter)");
  let revert_btn = mk ~cls:"py-btn py-btn-revert" ~txt:"Revert" "button" in
  revert_btn##setAttribute (js "type") (js "button");
  revert_btn##setAttribute (js "title") (js "Revert changes to example");
  let reset_btn = mk ~cls:"py-btn" ~txt:"Reset" "button" in
  reset_btn##setAttribute (js "type") (js "button");
  reset_btn##setAttribute (js "title") (js "Reset namespace");
  let expand_btn = mk ~cls:"py-btn py-expand-btn" ~txt:"Expand" "button" in
  expand_btn##setAttribute (js "type") (js "button");
  expand_btn##setAttribute (js "title") (js "Toggle full-viewport workbench");
  Dom.appendChild controls mode_sel;
  Dom.appendChild controls runtime_badge;
  Dom.appendChild controls status_pill;
  Dom.appendChild controls run_btn;
  Dom.appendChild controls revert_btn;
  Dom.appendChild controls reset_btn;
  Dom.appendChild controls expand_btn;
  Dom.appendChild topbar ex_control;
  Dom.appendChild topbar controls;
  Dom.appendChild topbar ex_menu;

  (* Band 1b: Editorial Preset Details Bar *)
  let preset_bar = mk ~cls:"py-preset-bar" "div" in
  let preset_tag_el = mk ~cls:"py-preset-tag" ~txt:"" "span" in
  let preset_desc_el = mk ~cls:"py-preset-desc" ~txt:"" "span" in
  Dom.appendChild preset_bar preset_tag_el;
  Dom.appendChild preset_bar preset_desc_el;

  (* Band 2b: Mobile (<900px) Segmented Switcher *)
  let mob_tabs = mk ~cls:"py-mob-tabs" "div" in
  let mk_mob_tab tab_id label_txt =
    let b = mk ~cls:"py-mob-tab" ~txt:label_txt "button" in
    b##setAttribute (js "type") (js "button");
    b##setAttribute (js "data-tab") (js tab_id);
    (tab_id, b)
  in
  let mob_pairs =
    [
      mk_mob_tab "script" "Script";
      mk_mob_tab "plots" "Plots & State";
      mk_mob_tab "console" "Console";
    ]
  in
  List.iter (fun (_, b) -> Dom.appendChild mob_tabs b) mob_pairs;

  let set_mob_tab target =
    shell##setAttribute (js "data-mob-tab") (js target);
    List.iter
      (fun (tid, b) ->
        let base = "py-mob-tab" in
        b##.className := js (if tid = target then base ^ " active" else base))
      mob_pairs
  in
  set_mob_tab "script";
  List.iter
    (fun (tid, b) ->
      b##.onclick :=
        Dom.handler (fun _ ->
            set_mob_tab tid;
            Js._true))
    mob_pairs;

  (* Band 3: Asymmetric Editorial Grid *)
  let grid = mk ~cls:"py-pg-grid" "div" in
  let left_col = mk ~cls:"py-pg-left" "div" in
  let col_resizer = mk ~cls:"py-col-resizer" "div" in
  col_resizer##setAttribute (js "aria-label") (js "Resize columns");
  col_resizer##setAttribute (js "role") (js "separator");
  let right_col = mk ~cls:"py-pg-right" "div" in

  (* Editor Sub-pane *)
  let ed_pane = mk ~cls:"py-ed-pane" "div" in
  let ed_hdr = mk ~cls:"py-subbar" "div" in
  let ed_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild ed_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Script" "span");
  let ed_meta = mk ~cls:"py-subbar-meta" ~txt:"sparc_pf.py" "span" in
  Dom.appendChild ed_hdr_l ed_meta;
  let ed_hdr_r = mk ~cls:"py-subbar-right" "div" in
  let run_sel_btn = mk ~cls:"py-ghost-btn" ~txt:"Run sel" "button" in
  run_sel_btn##setAttribute (js "type") (js "button");
  run_sel_btn##setAttribute (js "title") (js "Run selection or current line in live namespace (Shift+Enter)");
  let wrap_btn = mk ~cls:"py-ghost-btn py-wrap-btn" ~txt:"Wrap" "button" in
  wrap_btn##setAttribute (js "type") (js "button");
  wrap_btn##setAttribute (js "title") (js "Toggle line wrapping");
  let save_py_btn = mk ~cls:"py-ghost-btn" ~txt:"Save" "button" in
  save_py_btn##setAttribute (js "type") (js "button");
  save_py_btn##setAttribute (js "title") (js "Download script as .py file");
  let copy_btn = mk ~cls:"py-ghost-btn" ~txt:"Copy" "button" in
  copy_btn##setAttribute (js "type") (js "button");
  copy_btn##setAttribute (js "title") (js "Copy script to clipboard");
  Dom.appendChild ed_hdr_r run_sel_btn;
  Dom.appendChild ed_hdr_r wrap_btn;
  Dom.appendChild ed_hdr_r save_py_btn;
  Dom.appendChild ed_hdr_r copy_btn;
  Dom.appendChild ed_hdr ed_hdr_l;
  Dom.appendChild ed_hdr ed_hdr_r;

  let ed_shell = mk ~cls:"py-ed-shell" "div" in
  let ed_gutter = mk ~cls:"py-ed-gutter" "div" in
  ed_gutter##setAttribute (js "aria-hidden") (js "true");
  let ed_stage = mk ~cls:"py-ed-stage" "div" in
  let active_line_band = mk ~cls:"py-ed-active-band" "div" in
  let highlight_pre = mk ~cls:"py-ed-highlight" "pre" in
  highlight_pre##setAttribute (js "aria-hidden") (js "true");
  let highlight_code = mk "code" in
  Dom.appendChild highlight_pre highlight_code;
  let editor = createTextarea document in
  editor##.className := js "py-ed-textarea";
  editor##setAttribute (js "wrap") (js "off");
  editor##setAttribute (js "spellcheck") (js "false");
  editor##setAttribute (js "autocomplete") (js "off");
  editor##setAttribute (js "autocorrect") (js "off");
  editor##setAttribute (js "autocapitalize") (js "off");
  editor##setAttribute (js "aria-label") (js "Python script editor");
  Dom.appendChild ed_stage active_line_band;
  Dom.appendChild ed_stage highlight_pre;
  Dom.appendChild ed_stage editor;
  Dom.appendChild ed_shell ed_gutter;
  Dom.appendChild ed_shell ed_stage;
  Dom.appendChild ed_pane ed_hdr;
  Dom.appendChild ed_pane ed_shell;

  let is_wrapped_ref = ref false in
  wrap_btn##.onclick :=
    Dom.handler (fun _ ->
        is_wrapped_ref := not !is_wrapped_ref;
        let w = !is_wrapped_ref in
        wrap_btn##.className :=
          js (if w then "py-ghost-btn py-wrap-btn active" else "py-ghost-btn py-wrap-btn");
        let toggle_cls (el : element Js.t) =
          let c = to_s el##.className in
          let has = contains_sub c "is-wrapped" in
          if w && not has then el##.className := js (c ^ " is-wrapped")
          else if (not w) && has then
            let parts =
              List.filter (fun s -> s <> "is-wrapped" && s <> "") (String.split_on_char ' ' c)
            in
            el##.className := js (String.concat " " parts)
        in
        toggle_cls ed_shell;
        toggle_cls ed_stage;
        editor##setAttribute (js "wrap") (js (if w then "soft" else "off"));
        let band_opt = Js.Opt.to_option (ed_stage##querySelector (js ".py-ed-active-band")) in
        (match band_opt with
        | Some b -> b##.style##.display := js (if w then "none" else "")
        | None -> ());
        Js._true);

  let row_resizer = mk ~cls:"py-row-resizer" "div" in
  row_resizer##setAttribute (js "aria-label") (js "Resize editor and console");
  row_resizer##setAttribute (js "role") (js "separator");

  (* Console + Unified REPL Pane *)
  let con_pane = mk ~cls:"py-con-pane" "div" in
  let con_hdr = mk ~cls:"py-subbar" "div" in
  let con_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild con_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Console" "span");
  let con_hdr_r = mk ~cls:"py-subbar-right" "div" in
  let clear_btn = mk ~cls:"py-ghost-btn" ~txt:"Clear" "button" in
  clear_btn##setAttribute (js "type") (js "button");
  Dom.appendChild con_hdr_r clear_btn;
  Dom.appendChild con_hdr con_hdr_l;
  Dom.appendChild con_hdr con_hdr_r;

  let console_pre = mk ~cls:"py-pg-console" "pre" in
  append_console_chunk console_pre "py-out-muted" "Loading Python...\n";

  let compl_bar = mk ~cls:"py-repl-compl" "div" in
  compl_bar##setAttribute (js "hidden") (js "hidden");
  compl_bar##.style##.display := js "none";

  let prompt_row = createForm document in
  prompt_row##.className := js "py-repl-prompt-row";
  let prompt_glyph = mk ~cls:"py-repl-glyph" ~txt:">>>" "span" in
  let repl_input = createTextarea document in
  repl_input##.className := js "py-repl-input";
  repl_input##.rows := 1;
  repl_input##setAttribute (js "placeholder") (js "Python expression...");
  repl_input##setAttribute (js "spellcheck") (js "false");
  repl_input##setAttribute (js "autocomplete") (js "off");
  repl_input##setAttribute (js "autocorrect") (js "off");
  repl_input##setAttribute (js "autocapitalize") (js "off");
  repl_input##setAttribute (js "aria-label") (js "Interactive Python prompt");
  let repl_hint = mk ~cls:"py-repl-hint" ~txt:"Enter eval · ⇧↵ newline · ^L clear" "span" in
  let repl_send = mk ~cls:"py-ghost-btn py-repl-send" ~txt:"Eval" "button" in
  repl_send##setAttribute (js "type") (js "submit");
  Dom.appendChild prompt_row prompt_glyph;
  Dom.appendChild prompt_row repl_input;
  Dom.appendChild prompt_row repl_hint;
  Dom.appendChild prompt_row repl_send;

  Dom.appendChild con_pane con_hdr;
  Dom.appendChild con_pane console_pre;
  Dom.appendChild con_pane compl_bar;
  Dom.appendChild con_pane prompt_row;

  Dom.appendChild left_col ed_pane;
  Dom.appendChild left_col row_resizer;
  Dom.appendChild left_col con_pane;

  (* Right Output & Figures Stage *)
  let insp_hdr = mk ~cls:"py-subbar" "div" in
  let insp_hdr_l = mk ~cls:"py-subbar-left" "div" in
  Dom.appendChild insp_hdr_l (mk ~cls:"py-subbar-title" ~txt:"Output" "span");
  let insp_tabs = mk ~cls:"py-insp-tabs" "div" in
  let tab_figs_btn = mk ~cls:"py-ghost-btn py-insp-tab active" ~txt:"Figures" "button" in
  tab_figs_btn##setAttribute (js "type") (js "button");
  let tab_scope_btn = mk ~cls:"py-ghost-btn py-insp-tab" ~txt:"Scope (0)" "button" in
  tab_scope_btn##setAttribute (js "type") (js "button");
  Dom.appendChild insp_tabs tab_figs_btn;
  Dom.appendChild insp_tabs tab_scope_btn;
  Dom.appendChild insp_hdr_l insp_tabs;
  let insp_chip = mk ~cls:"py-subbar-meta py-state-chip" ~txt:"" "span" in
  let insp_hdr_r = mk ~cls:"py-subbar-right" "div" in
  Dom.appendChild insp_hdr_r insp_chip;
  Dom.appendChild insp_hdr insp_hdr_l;
  Dom.appendChild insp_hdr insp_hdr_r;

  let insp_body = mk ~cls:"py-inspector-body" "div" in
  Dom.appendChild right_col insp_hdr;
  Dom.appendChild right_col insp_body;

  Dom.appendChild grid left_col;
  Dom.appendChild grid col_resizer;
  Dom.appendChild grid right_col;

  Dom.appendChild shell topbar;
  Dom.appendChild shell preset_bar;
  Dom.appendChild shell mob_tabs;
  Dom.appendChild shell grid;
  Dom.appendChild root shell;

  (* Resizer Drag Handlers *)
  col_resizer##.onmousedown :=
    Dom.handler (fun ev ->
        Dom.preventDefault ev;
        let move_handler = ref (Js.wrap_callback (fun _ -> ())) in
        let up_handler = ref (Js.wrap_callback (fun _ -> ())) in
        let on_move =
          Js.wrap_callback (fun (mev : #Dom_html.mouseEvent Js.t) ->
              let grid_rect = grid##getBoundingClientRect in
              let g_left = Js.float_of_number grid_rect##.left in
              let gw = Js.float_of_number grid_rect##.right -. g_left in
              if gw > 0.0 then (
                let client_x = Js.float_of_number mev##.clientX in
                let frac = (client_x -. g_left) /. gw in
                let pct = max 30.0 (min 70.0 (frac *. 100.0)) in
                ignore
                  (Js.Unsafe.meth_call grid##.style "setProperty"
                     [| Js.Unsafe.inject (js "--py-split-x"); Js.Unsafe.inject (js (Printf.sprintf "%.1f%%" pct)) |])))
        in
        let on_up =
          Js.wrap_callback (fun (_mev : #Dom_html.mouseEvent Js.t) ->
              ignore
                (Js.Unsafe.meth_call Dom_html.window "removeEventListener"
                   [| Js.Unsafe.inject (js "mousemove"); Js.Unsafe.inject !move_handler |]);
              ignore
                (Js.Unsafe.meth_call Dom_html.window "removeEventListener"
                   [| Js.Unsafe.inject (js "mouseup"); Js.Unsafe.inject !up_handler |]))
        in
        move_handler := on_move;
        up_handler := on_up;
        ignore
          (Js.Unsafe.meth_call Dom_html.window "addEventListener"
             [| Js.Unsafe.inject (js "mousemove"); Js.Unsafe.inject on_move |]);
        ignore
          (Js.Unsafe.meth_call Dom_html.window "addEventListener"
             [| Js.Unsafe.inject (js "mouseup"); Js.Unsafe.inject on_up |]);
        Js._false);

  row_resizer##.onmousedown :=
    Dom.handler (fun ev ->
        Dom.preventDefault ev;
        let move_handler = ref (Js.wrap_callback (fun _ -> ())) in
        let up_handler = ref (Js.wrap_callback (fun _ -> ())) in
        let on_move =
          Js.wrap_callback (fun (mev : #Dom_html.mouseEvent Js.t) ->
              let col_rect = left_col##getBoundingClientRect in
              let c_top = Js.float_of_number col_rect##.top in
              let ch = Js.float_of_number col_rect##.bottom -. c_top in
              if ch > 0.0 then (
                let client_y = Js.float_of_number mev##.clientY in
                let frac = (client_y -. c_top) /. ch in
                let pct = max 28.0 (min 78.0 (frac *. 100.0)) in
                ignore
                  (Js.Unsafe.meth_call left_col##.style "setProperty"
                     [| Js.Unsafe.inject (js "--py-ed-h"); Js.Unsafe.inject (js (Printf.sprintf "%.1f%%" pct)) |])))
        in
        let on_up =
          Js.wrap_callback (fun (_mev : #Dom_html.mouseEvent Js.t) ->
              ignore
                (Js.Unsafe.meth_call Dom_html.window "removeEventListener"
                   [| Js.Unsafe.inject (js "mousemove"); Js.Unsafe.inject !move_handler |]);
              ignore
                (Js.Unsafe.meth_call Dom_html.window "removeEventListener"
                   [| Js.Unsafe.inject (js "mouseup"); Js.Unsafe.inject !up_handler |]))
        in
        move_handler := on_move;
        up_handler := on_up;
        ignore
          (Js.Unsafe.meth_call Dom_html.window "addEventListener"
             [| Js.Unsafe.inject (js "mousemove"); Js.Unsafe.inject on_move |]);
        ignore
          (Js.Unsafe.meth_call Dom_html.window "addEventListener"
             [| Js.Unsafe.inject (js "mouseup"); Js.Unsafe.inject on_up |]);
        Js._false);

  let active_preset_ref : preset option ref = ref None in
  let prev_lines_ref = ref 0 in
  let user_dirty_ref = ref false in
  let err_line_ref : int option ref = ref None in
  let sync_switcher_ref : (unit -> unit) ref = ref (fun () -> ()) in

  let sync_revert_btn () =
    let can_revert = !user_dirty_ref && !active_preset_ref <> None in
    if can_revert then (
      revert_btn##removeAttribute (js "hidden");
      revert_btn##.style##.display := js "";
      Js.Unsafe.set revert_btn (js "disabled") Js._false;
      revert_btn##removeAttribute (js "disabled"))
    else (
      revert_btn##setAttribute (js "hidden") (js "hidden");
      revert_btn##.style##.display := js "none";
      Js.Unsafe.set revert_btn (js "disabled") Js._true;
      revert_btn##setAttribute (js "disabled") (js "disabled"))
  in

  let sync_active_line_band () =
    if !is_wrapped_ref then active_line_band##.style##.display := js "none"
    else
      let v = to_s editor##.value in
      let cur_line = cursor_line_no v editor##.selectionStart in
      let gutter_lines = ed_gutter##.childNodes in
      if cur_line >= 1 && cur_line <= gutter_lines##.length then
        match Js.Opt.to_option (gutter_lines##item (cur_line - 1)) with
        | Some node ->
            let el : element Js.t = Obj.magic node in
            let top_px =
              el##.offsetTop - int_of_float (Js.float_of_number (Obj.magic editor##.scrollTop))
            in
            let h_px = el##.offsetHeight in
            active_line_band##.style##.top := js (Printf.sprintf "%dpx" top_px);
            if h_px > 0 then active_line_band##.style##.height := js (Printf.sprintf "%dpx" h_px);
            active_line_band##.style##.display := js ""
        | None -> active_line_band##.style##.display := js "none"
      else active_line_band##.style##.display := js "none"
  in

  let sync_editor_view () =
    let src = to_s editor##.value in
    render_py_highlight highlight_code src;
    let n = count_lines src in
    let cur_line = cursor_line_no src editor##.selectionStart in
    render_line_numbers ed_gutter prev_lines_ref ~active:(Some cur_line) ~err:!err_line_ref n;
    let stem =
      match !active_preset_ref with
      | Some p -> p.id ^ ".py" ^ if !user_dirty_ref then " *" else ""
      | None -> "snippet.py"
    in
    set_text ed_meta (Printf.sprintf "%s · %d lines" stem n);
    highlight_pre##.scrollTop := editor##.scrollTop;
    highlight_pre##.scrollLeft := editor##.scrollLeft;
    ed_gutter##.scrollTop := editor##.scrollTop;
    sync_active_line_band ();
    sync_revert_btn ();
    !sync_switcher_ref ()
  in

  let focus_editor_line line_no =
    let src = to_s editor##.value in
    let offset = find_line_char_offset src line_no in
    err_line_ref := Some line_no;
    editor##focus;
    editor##.selectionStart := offset;
    editor##.selectionEnd := offset;
    let gutter_lines = ed_gutter##.childNodes in
    if line_no >= 1 && line_no <= gutter_lines##.length then
      match Js.Opt.to_option (gutter_lines##item (line_no - 1)) with
      | Some node ->
          let el : element Js.t = Obj.magic node in
          let target_top = max 0 (el##.offsetTop - 60) in
          editor##.scrollTop := Js.float (float_of_int target_top)
      | None -> ();
    sync_editor_view ()
  in

  let replace_editor_range ~st ~en ~text ~sel_start ~sel_end =
    let v = to_s editor##.value in
    editor##focus;
    editor##.selectionStart := st;
    editor##.selectionEnd := en;
    let applied =
      try
        Js.to_bool
          (Js.Unsafe.meth_call document "execCommand"
             [| Js.Unsafe.inject (js "insertText"); Js.Unsafe.inject Js._false; Js.Unsafe.inject (js text) |])
      with _ -> false
    in
    if not applied then (
      let left = String.sub v 0 st in
      let right = String.sub v en (String.length v - en) in
      editor##.value := js (left ^ text ^ right));
    editor##.selectionStart := sel_start;
    editor##.selectionEnd := sel_end;
    user_dirty_ref := true;
    err_line_ref := None;
    sync_editor_view ()
  in

  let handle_enter_indent () =
    let v = to_s editor##.value in
    let st = editor##.selectionStart in
    let en = editor##.selectionEnd in
    let line_start =
      let rec back i =
        if i <= 0 then 0
        else if v.[i - 1] = '\n' then i
        else back (i - 1)
      in
      back st
    in
    let line_prefix = String.sub v line_start (st - line_start) in
    let indent_len =
      let j = ref 0 in
      while !j < String.length line_prefix && (line_prefix.[!j] = ' ' || line_prefix.[!j] = '\t') do
        incr j
      done;
      !j
    in
    let base_indent = String.sub line_prefix 0 indent_len in
    let trimmed = String.trim line_prefix in
    let extra_indent =
      if String.length trimmed > 0 && trimmed.[String.length trimmed - 1] = ':' then "    "
      else ""
    in
    let insert = "\n" ^ base_indent ^ extra_indent in
    let new_pos = st + String.length insert in
    replace_editor_range ~st ~en ~text:insert ~sel_start:new_pos ~sel_end:new_pos
  in

  let handle_tab ~shift =
    let v = to_s editor##.value in
    let st = editor##.selectionStart in
    let en = editor##.selectionEnd in
    if (not shift) && st = en then
      replace_editor_range ~st ~en ~text:"    " ~sel_start:(st + 4) ~sel_end:(st + 4)
    else
      let line_start =
        let rec back i =
          if i <= 0 then 0
          else if v.[i - 1] = '\n' then i
          else back (i - 1)
        in
        back st
      in
      let line_end =
        let len = String.length v in
        let rec fwd i =
          if i >= len then len
          else if v.[i] = '\n' then i
          else fwd (i + 1)
        in
        fwd en
      in
      let block = String.sub v line_start (line_end - line_start) in
      let lines = String.split_on_char '\n' block in
      let new_lines =
        if shift then
          List.map
            (fun l ->
              let len = String.length l in
              let rec count_spaces i =
                if i < len && i < 4 && l.[i] = ' ' then count_spaces (i + 1)
                else i
              in
              let sp = count_spaces 0 in
              if sp > 0 then String.sub l sp (len - sp)
              else if len > 0 && l.[0] = '\t' then String.sub l 1 (len - 1)
              else l)
            lines
        else List.map (fun l -> "    " ^ l) lines
      in
      let new_block = String.concat "\n" new_lines in
      replace_editor_range ~st:line_start ~en:line_end ~text:new_block
        ~sel_start:line_start ~sel_end:(line_start + String.length new_block)
  in

  let handle_toggle_comment () =
    let v = to_s editor##.value in
    let st = editor##.selectionStart in
    let en = editor##.selectionEnd in
    let line_start =
      let rec back i =
        if i <= 0 then 0
        else if v.[i - 1] = '\n' then i
        else back (i - 1)
      in
      back st
    in
    let line_end =
      let len = String.length v in
      let rec fwd i =
        if i >= len then len
        else if v.[i] = '\n' then i
        else fwd (i + 1)
      in
      fwd en
    in
    let block = String.sub v line_start (line_end - line_start) in
    let lines = String.split_on_char '\n' block in
    let non_empty = List.filter (fun l -> String.trim l <> "") lines in
    let all_commented =
      non_empty <> []
      && List.for_all
           (fun l ->
             let t = String.trim l in
             String.length t > 0 && t.[0] = '#')
           non_empty
    in
    let new_lines =
      if all_commented then
        List.map
          (fun l ->
            let trimmed_start =
              let j = ref 0 in
              while !j < String.length l && (l.[!j] = ' ' || l.[!j] = '\t') do
                incr j
              done;
              !j
            in
            if trimmed_start < String.length l && l.[trimmed_start] = '#' then
              let remove_len =
                if trimmed_start + 1 < String.length l && l.[trimmed_start + 1] = ' ' then 2 else 1
              in
              String.sub l 0 trimmed_start
              ^ String.sub l (trimmed_start + remove_len)
                  (String.length l - trimmed_start - remove_len)
            else l)
          lines
      else List.map (fun l -> if String.trim l = "" then l else "# " ^ l) lines
    in
    let new_block = String.concat "\n" new_lines in
    replace_editor_range ~st:line_start ~en:line_end ~text:new_block
      ~sel_start:line_start ~sel_end:(line_start + String.length new_block)
  in

  revert_btn##.onclick :=
    Dom.handler (fun _ ->
        (match !active_preset_ref with
        | Some p ->
            user_dirty_ref := false;
            err_line_ref := None;
            editor##.value := js p.code;
            editor##.scrollTop := Js.float 0.0;
            editor##.scrollLeft := Js.float 0.0;
            sync_editor_view ()
        | None -> ());
        Js._true);

  (* Build Empty Stage helper *)
  let initial_ran_ref = ref false in
  let manual_abort_ref = ref false in
  let run_script_ref : (unit -> unit) ref = ref (fun () -> ()) in

  let make_empty_stage () =
    let box = mk ~cls:"py-empty-stage" "div" in
    Dom.appendChild box
      (mk ~cls:"py-empty-desc" ~txt:"Output and figures appear after running." "p");
    let boot_banner = mk ~cls:"py-boot-banner" "div" in
    let boot_bar = mk ~cls:"py-boot-bar" "div" in
    let boot_fill = mk ~cls:"py-boot-fill" "div" in
    let total = max 1 !boot_total_ref in
    let step = max 0 (min total !boot_step_ref) in
    let pct = float_of_int step /. float_of_int total *. 100.0 in
    boot_fill##.style##.width := js (Printf.sprintf "%.1f%%" pct);
    Dom.appendChild boot_bar boot_fill;
    let boot_txt = mk ~cls:"py-boot-txt" ~txt:(status_text !kernel_state_ref) "span" in
    Dom.appendChild boot_banner boot_bar;
    Dom.appendChild boot_banner boot_txt;
    Dom.appendChild box boot_banner;
    box
  in

  let sync_state_chip st_raw =
    if not (is_defined st_raw) then set_text insp_chip ""
    else
      let has_opt = is_defined (Js.Unsafe.get st_raw (js "dim")) in
      let has_fig = is_defined (Js.Unsafe.get st_raw (js "fig")) in
      if has_opt then
        let vn = Option.value (opt_str st_raw "var") ~default:"state" in
        let dm = Option.value (opt_int st_raw "dim") ~default:0 in
        set_text insp_chip (Printf.sprintf "%s · %dD" vn dm)
      else if has_fig then set_text insp_chip "figure"
      else set_text insp_chip ""
  in

  let (init_preset_id, init_code) = initial_playground_code () in
  active_preset_ref :=
    (match init_preset_id with Some pid -> find_preset pid | None -> None);
  editor##.value := js init_code;

  let total_presets = List.length presets in
  let preset_opts : (int * preset * element Js.t) list =
    List.mapi
      (fun idx p ->
        let btn = mk ~cls:"py-ex-opt" "button" in
        btn##setAttribute (js "type") (js "button");
        btn##setAttribute (js "role") (js "option");
        let num = mk ~cls:"py-ex-opt-num" ~txt:(Printf.sprintf "%02d" (idx + 1)) "span" in
        let body = mk ~cls:"py-ex-opt-body" "div" in
        let top = mk ~cls:"py-ex-opt-top" "div" in
        let name = mk ~cls:"py-ex-opt-name" ~txt:p.label "span" in
        let tag = mk ~cls:"py-ex-opt-tag" ~txt:p.tag "span" in
        let desc = mk ~cls:"py-ex-opt-desc" ~txt:p.description "span" in
        Dom.appendChild top name;
        Dom.appendChild top tag;
        Dom.appendChild body top;
        Dom.appendChild body desc;
        Dom.appendChild btn num;
        Dom.appendChild btn body;
        Dom.appendChild ex_menu_list btn;
        (idx, p, btn))
      presets
  in

  let last_state_ref : Js.Unsafe.any ref = ref (Js.Unsafe.inject Js.undefined) in
  let last_scope_ref : Js.Unsafe.any ref = ref (Js.Unsafe.inject Js.undefined) in
  let insp_tab_ref : string ref = ref "figures" in
  let inspect_var_ref : (string -> unit) ref = ref (fun _ -> ()) in

  let sync_insp_tabs () =
    let sc_cnt = List.length (json_obj_list !last_scope_ref) in
    set_text tab_scope_btn (Printf.sprintf "Scope (%d)" sc_cnt);
    let f_cls = "py-ghost-btn py-insp-tab" ^ if !insp_tab_ref = "figures" then " active" else "" in
    let s_cls = "py-ghost-btn py-insp-tab" ^ if !insp_tab_ref = "scope" then " active" else "" in
    tab_figs_btn##.className := js f_cls;
    tab_scope_btn##.className := js s_cls
  in

  let refresh_inspector () =
    sync_insp_tabs ();
    if !insp_tab_ref = "scope" then
      render_scope_panel ~on_inspect:(fun nm -> !inspect_var_ref nm) insp_body !last_scope_ref
    else
      render_state_panel ~empty_node:(make_empty_stage ()) insp_body !last_state_ref
  in

  tab_figs_btn##.onclick :=
    Dom.handler (fun _ ->
        insp_tab_ref := "figures";
        refresh_inspector ();
        Js._true);

  tab_scope_btn##.onclick :=
    Dom.handler (fun _ ->
        insp_tab_ref := "scope";
        refresh_inspector ();
        Js._true);

  let sync_preset_active () =
    let active_pair =
      match !active_preset_ref with
      | Some p ->
          List.find_opt (fun (_, cand, _) -> cand.id = p.id) preset_opts
      | None -> None
    in
    let dirty = !user_dirty_ref in
    (match active_pair with
    | Some (idx, p, _) ->
        set_text ex_count_el (Printf.sprintf "%02d/%02d" (idx + 1) total_presets);
        set_text ex_title_el (p.label ^ if dirty then " · edited" else "");
        ex_trigger##setAttribute (js "title") (js p.description);
        set_text preset_tag_el p.tag;
        set_text preset_desc_el p.description
    | None ->
        set_text ex_count_el "Custom";
        set_text ex_title_el "Custom snippet";
        ex_trigger##setAttribute (js "title") (js "Choose a built-in example");
        set_text preset_tag_el "CUSTOM";
        set_text preset_desc_el "Custom Python snippet");
    ex_trigger##.className :=
      js (if dirty then "py-ex-trigger is-dirty" else "py-ex-trigger");
    let active_id = match !active_preset_ref with Some p -> p.id | None -> "" in
    List.iter
      (fun (_, p, b) ->
        let sel_flag = p.id = active_id in
        b##.className := js (if sel_flag then "py-ex-opt active" else "py-ex-opt");
        b##setAttribute (js "aria-selected") (js (if sel_flag then "true" else "false")))
      preset_opts;
    sync_revert_btn ()
  in
  sync_switcher_ref := sync_preset_active;
  sync_editor_view ();
  refresh_inspector ();
  sync_preset_active ();

  let mark_mob_updates () =
    List.iter
      (fun (tid, b) ->
        if tid = "plots" then (
          let cur = to_s b##.className in
          if not (contains_sub cur "has-update") then b##.className := js (cur ^ " has-update")))
      mob_pairs
  in

  let apply_exec_result ~cmd_prefix res =
    let ok = Js.to_bool (Js.Unsafe.get res (js "ok")) in
    let out_s = to_s (Js.Unsafe.get res (js "stdout")) in
    let err_s = to_s (Js.Unsafe.get res (js "stderr")) in
    let dt_raw = Js.Unsafe.get res (js "elapsed_ms") in
    let dt_s =
      if Js.Optdef.test dt_raw then Printf.sprintf "%.1f ms" (Js.float_of_number dt_raw) else ""
    in
    set_text timing_lbl (if dt_s <> "" then "· " ^ dt_s else "");
    (match cmd_prefix with
    | Some pfx -> append_console_chunk console_pre "py-out-cmd" (pfx ^ "\n")
    | None -> clear_children (console_pre :> Dom.node Js.t));
    if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
    if err_s <> "" then (
      let err_line = append_console_stderr console_pre err_s focus_editor_line in
      (match err_line with
      | Some ln ->
          err_line_ref := Some ln;
          sync_editor_view ()
      | None -> ()));
    if out_s = "" && err_s = "" && ok && cmd_prefix = None then
      append_console_chunk console_pre "py-out-muted" "Done.\n";
    set_text prompt_glyph ">>>";
    let st_raw = Js.Unsafe.get res (js "state") in
    let sc_raw = Js.Unsafe.get res (js "scope") in
    last_state_ref := st_raw;
    if is_defined sc_raw then last_scope_ref := sc_raw;
    sync_state_chip st_raw;
    refresh_inspector ();
    mark_mob_updates ()
  in

  let run_script () =
    initial_ran_ref := true;
    manual_abort_ref := false;
    let src = to_s editor##.value in
    clear_children (console_pre :> Dom.node Js.t);
    append_console_chunk console_pre "py-out-muted" "Running...\n";
    set_text timing_lbl "";
    send_worker "exec"
      [
        ("code", Js.Unsafe.inject (js src));
        ("resetNamespace", Js.Unsafe.inject Js._true);
      ]
      (fun res -> apply_exec_result ~cmd_prefix:None res)
  in
  run_script_ref := run_script;

  let run_selection_or_line () =
    initial_ran_ref := true;
    manual_abort_ref := false;
    let v = to_s editor##.value in
    let st = editor##.selectionStart in
    let en = editor##.selectionEnd in
    let snippet =
      if st < en then String.sub v st (en - st)
      else
        let l0 =
          let rec back i = if i <= 0 then 0 else if v.[i - 1] = '\n' then i else back (i - 1) in
          back st
        in
        let l1 =
          let len = String.length v in
          let rec fwd i = if i >= len then len else if v.[i] = '\n' then i else fwd (i + 1) in
          fwd st
        in
        String.sub v l0 (l1 - l0)
    in
    let trimmed = String.trim snippet in
    if trimmed <> "" then (
      let first_line =
        match String.split_on_char '\n' trimmed with
        | h :: _ ->
            let s = String.trim h in
            if String.length s > 56 then String.sub s 0 53 ^ "..." else s
        | [] -> "..."
      in
      set_text timing_lbl "";
      send_worker "exec"
        [
          ("code", Js.Unsafe.inject (js snippet));
          ("resetNamespace", Js.Unsafe.inject Js._false);
        ]
        (fun res -> apply_exec_result ~cmd_prefix:(Some ("» " ^ first_line)) res))
  in

  let sync_runtime_badge caps =
    if !runtime_mode_ref = LocalNative then (
      let accel =
        if caps.native_metal then "Local · Metal"
        else if caps.native_cuda then "Local · CUDA"
        else "Local"
      in
      set_text runtime_badge accel;
      runtime_badge##setAttribute (js "title") (js ("Native Python: " ^ caps.native_python)))
    else if caps.webgpu_active then (
      set_text runtime_badge "WASM · WebGPU";
      runtime_badge##setAttribute (js "title")
        (js ("Pyodide WASM + WebGPU (" ^ caps.webgpu_adapter ^ ")")))
    else (
      set_text runtime_badge "WASM · SIMD";
      runtime_badge##setAttribute (js "title") (js "In-browser Pyodide WASM"))
  in

  add_cap_listener (fun caps ->
      sync_runtime_badge caps;
      if caps.native_checked then (
        if caps.native_available then (
          mode_sel##removeAttribute (js "hidden");
          mode_sel##.style##.display := js "";
          Js.Unsafe.set native_opt (js "disabled") Js._false;
          native_opt##removeAttribute (js "disabled");
          let accel =
            if caps.native_metal then " (Metal)"
            else if caps.native_cuda then " (CUDA)"
            else ""
          in
          set_text native_opt ("Local" ^ accel);
          mode_sel##setAttribute (js "title") (js "Switch between browser WASM and local Python"))
        else (
          mode_sel##setAttribute (js "hidden") (js "hidden");
          mode_sel##.style##.display := js "none";
          Js.Unsafe.set native_opt (js "disabled") Js._true;
          native_opt##setAttribute (js "disabled") (js "disabled");
          if !runtime_mode_ref = LocalNative then (
            runtime_mode_ref := Wasm;
            mode_sel##.value := js "wasm";
            notify_status Idle))));

  mode_sel##.onchange :=
    Dom.handler (fun _ ->
        let chosen = to_s mode_sel##.value in
        if chosen = "local_native" && cap_ref.native_available then (
          runtime_mode_ref := LocalNative;
          notify_caps ();
          notify_status Ready)
        else (
          runtime_mode_ref := Wasm;
          mode_sel##.value := js "wasm";
          notify_caps ();
          (match !worker_ref with
          | Some _ when Hashtbl.length pending_reqs = 0 -> notify_status Ready
          | Some _ -> notify_status Running
          | None ->
              notify_status Idle;
              warm_kernel ()));
        Js._true);

  add_status_listener (fun st ->
      let short_s =
        match st with
        | Idle -> "Idle"
        | Booting _ ->
            if !runtime_mode_ref = LocalNative then "Connecting..." else "Loading..."
        | Ready -> "Ready"
        | Running -> "Running..."
        | Interrupting _ -> "Stopping..."
        | Interrupted _ -> "Stopped"
      in
      set_text status_lbl short_s;
      status_pill##setAttribute (js "title") (js (status_text st));
      let cls =
        match st with
        | Ready -> "py-status-dot is-ready"
        | Booting _ | Running | Interrupting _ -> "py-status-dot is-busy"
        | Interrupted _ -> "py-status-dot is-err"
        | Idle -> "py-status-dot"
      in
      status_dot##.className := js cls;
      let reset_lbl =
        match st with
        | Running | Booting _ -> "Interrupt"
        | Interrupting _ -> "Force Stop"
        | Idle | Ready | Interrupted _ -> "Reset"
      in
      let reset_tip =
        match st with
        | Running -> "Interrupt execution"
        | Interrupting _ -> "Force stop"
        | Booting _ -> "Cancel loading"
        | Idle | Ready | Interrupted _ -> "Reset namespace"
      in
      set_text reset_btn reset_lbl;
      reset_btn##setAttribute (js "title") (js reset_tip);
      (match st with
      | Interrupted msg ->
          let cur_text =
            match Js.Opt.to_option console_pre##.textContent with
            | Some s -> String.trim (to_s s)
            | None -> ""
          in
          if cur_text = "" || cur_text = "Loading Python..." then begin
            clear_children (console_pre :> Dom.node Js.t);
            ignore (append_console_stderr console_pre (msg ^ "\n") focus_editor_line)
          end
      | _ -> ());
      if (not !initial_ran_ref) && !insp_tab_ref = "figures"
         && Js.Opt.test (insp_body##querySelector (js ".py-empty-stage"))
      then
        refresh_inspector ();
      if st = Ready && (not !initial_ran_ref) && (not !user_dirty_ref) && (not !manual_abort_ref) then
        run_script ());

  let menu_open_ref = ref false in
  let set_menu_open open_flag =
    menu_open_ref := open_flag;
    if open_flag then ex_menu##removeAttribute (js "hidden")
    else ex_menu##setAttribute (js "hidden") (js "hidden");
    ex_trigger##setAttribute (js "aria-expanded") (js (if open_flag then "true" else "false"));
    ex_control##.className := js (if open_flag then "py-ex-control is-open" else "py-ex-control")
  in

  let focus_opt_at idx =
    let wrapped = ((idx mod total_presets) + total_presets) mod total_presets in
    match List.find_opt (fun (i, _, _) -> i = wrapped) preset_opts with
    | Some (_, _, btn) -> btn##focus
    | None -> ()
  in

  let active_preset_idx () =
    match !active_preset_ref with
    | Some p -> (
        match List.find_opt (fun (_, cand, _) -> cand.id = p.id) preset_opts with
        | Some (i, _, _) -> Some i
        | None -> None)
    | None -> None
  in

  let select_preset p =
    set_menu_open false;
    active_preset_ref := Some p;
    user_dirty_ref := false;
    err_line_ref := None;
    sync_url_preset (Some p.id);
    editor##.value := js p.code;
    editor##.scrollTop := Js.float 0.0;
    editor##.scrollLeft := Js.float 0.0;
    sync_editor_view ();
    last_state_ref := Js.Unsafe.inject Js.undefined;
    last_scope_ref := Js.Unsafe.inject Js.undefined;
    insp_tab_ref := "figures";
    refresh_inspector ();
    sync_state_chip (Js.Unsafe.inject Js.undefined);
    if !kernel_state_ref = Ready then run_script ()
  in

  let step_preset delta =
    if total_presets > 0 then
      let next_i =
        match active_preset_idx () with
        | Some i -> ((i + delta) mod total_presets + total_presets) mod total_presets
        | None -> if delta >= 0 then 0 else total_presets - 1
      in
      let _, p, _ = List.nth preset_opts next_i in
      select_preset p
  in

  ex_trigger##.onclick :=
    Dom.handler (fun _ ->
        let next = not !menu_open_ref in
        set_menu_open next;
        Js._true);

  prev_btn##.onclick :=
    Dom.handler (fun _ ->
        step_preset (-1);
        Js._true);

  next_btn##.onclick :=
    Dom.handler (fun _ ->
        step_preset 1;
        Js._true);

  ex_trigger##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        if key = "ArrowDown" || key = "ArrowUp" then (
          Dom.preventDefault ev;
          set_menu_open true;
          let target_i =
            match active_preset_idx () with
            | Some i -> i
            | None -> if key = "ArrowDown" then 0 else total_presets - 1
          in
          focus_opt_at target_i;
          Js._false)
        else Js._true);

  List.iter
    (fun (idx, p, btn) ->
      btn##.onclick :=
        Dom.handler (fun _ ->
            select_preset p;
            Js._true);
      btn##.onkeydown :=
        Dom.handler (fun ev ->
            let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
            match key with
            | "ArrowDown" ->
                Dom.preventDefault ev;
                focus_opt_at (idx + 1);
                Js._false
            | "ArrowUp" ->
                Dom.preventDefault ev;
                focus_opt_at (idx - 1);
                Js._false
            | "Home" ->
                Dom.preventDefault ev;
                focus_opt_at 0;
                Js._false
            | "End" ->
                Dom.preventDefault ev;
                focus_opt_at (total_presets - 1);
                Js._false
            | "Escape" ->
                Dom.preventDefault ev;
                set_menu_open false;
                ex_trigger##focus;
                Js._false
            | _ -> Js._true))
    preset_opts;

  ignore
    (Dom_html.addEventListener document Dom_html.Event.click
       (Dom_html.handler (fun ev ->
            if !menu_open_ref then
              (match Js.Opt.to_option ev##.target with
              | Some tgt ->
                  let in_ctrl =
                    Js.to_bool
                      (Js.Unsafe.meth_call (ex_control :> Dom.node Js.t) "contains"
                         [| Js.Unsafe.inject tgt |])
                  in
                  let in_menu =
                    Js.to_bool
                      (Js.Unsafe.meth_call (ex_menu :> Dom.node Js.t) "contains"
                         [| Js.Unsafe.inject tgt |])
                  in
                  if not (in_ctrl || in_menu) then set_menu_open false
              | None -> set_menu_open false);
            Js._true))
       Js._false);

  let is_expanded_ref = ref false in
  let set_expanded exp =
    is_expanded_ref := exp;
    let parts =
      List.filter (fun s -> s <> "is-expanded" && s <> "") (String.split_on_char ' ' (to_s shell##.className))
    in
    shell##.className := js (String.concat " " (if exp then parts @ [ "is-expanded" ] else parts));
    set_text expand_btn (if exp then "Exit" else "Expand");
    sync_editor_view ()
  in
  expand_btn##.onclick :=
    Dom.handler (fun _ ->
        set_expanded (not !is_expanded_ref);
        Js._true);

  ignore
    (Dom_html.addEventListener document Dom_html.Event.keydown
       (Dom_html.handler (fun ev ->
            let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
            if !menu_open_ref && key = "Escape" then (
              set_menu_open false;
              ex_trigger##focus)
            else if !is_expanded_ref && key = "Escape" then
              set_expanded false;
            Js._true))
       Js._false);

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
          set_text timing_lbl (Printf.sprintf "· %.1f ms" (Js.float_of_number dt_raw));
        if out_s <> "" then append_console_chunk console_pre "py-out-stdout" out_s;
        if err_s <> "" then (
          let err_line = append_console_stderr console_pre err_s focus_editor_line in
          (match err_line with
          | Some ln ->
              err_line_ref := Some ln;
              sync_editor_view ()
          | None -> ()));
        set_text prompt_glyph (if inc then "..." else ">>>");
        let st_raw = Js.Unsafe.get res (js "state") in
        let sc_raw = Js.Unsafe.get res (js "scope") in
        last_state_ref := st_raw;
        if is_defined sc_raw then last_scope_ref := sc_raw;
        sync_state_chip st_raw;
        refresh_inspector ())
  in
  inspect_var_ref := (fun name ->
      repl_input##focus;
      push_repl_line name);

  let is_at_top (el : textAreaElement Js.t) =
    let pos = el##.selectionStart in
    let v = to_s el##.value in
    let rec check i =
      if i <= 0 then true
      else if v.[i - 1] = '\n' then false
      else check (i - 1)
    in
    check pos
  in

  let is_at_bottom (el : textAreaElement Js.t) =
    let pos = el##.selectionEnd in
    let v = to_s el##.value in
    let len = String.length v in
    let rec check i =
      if i >= len then true
      else if v.[i] = '\n' then false
      else check (i + 1)
    in
    check pos
  in

  let auto_size_repl_input () =
    let v = to_s repl_input##.value in
    let lines = count_lines v in
    repl_input##.rows := max 1 (min 6 lines)
  in

  let submit_repl () =
    let line = to_s repl_input##.value in
    repl_input##.value := js "";
    repl_input##.rows := 1;
    compl_bar##setAttribute (js "hidden") (js "hidden");
    compl_bar##.style##.display := js "none";
    clear_children (compl_bar :> Dom.node Js.t);
    push_repl_line line
  in

  run_btn##.onclick :=
    Dom.handler (fun _ ->
        run_script ();
        (if Dom_html.window##.innerWidth <= 900 then set_mob_tab "plots");
        Js._true);

  run_sel_btn##.onclick :=
    Dom.handler (fun _ ->
        run_selection_or_line ();
        (if Dom_html.window##.innerWidth <= 900 then set_mob_tab "console");
        Js._true);

  save_py_btn##.onclick :=
    Dom.handler (fun _ ->
        let fname =
          match !active_preset_ref with
          | Some p -> p.id ^ ".py"
          | None -> "baux_snippet.py"
        in
        download_text_file ~filename:fname ~mime:"text/x-python;charset=utf-8"
          ~text:(to_s editor##.value);
        Js._true);

  reset_btn##.onclick :=
    Dom.handler (fun _ ->
        manual_abort_ref := true;
        (match !kernel_state_ref with
        | Running | Booting _ -> interrupt_kernel ~force:false ()
        | Interrupting _ ->
            interrupt_kernel ~force:true ();
            set_text prompt_glyph ">>>";
            set_text timing_lbl ""
        | Idle | Ready | Interrupted _ ->
            reset_kernel ();
            last_state_ref := Js.Unsafe.inject Js.undefined;
            last_scope_ref := Js.Unsafe.inject Js.undefined;
            refresh_inspector ();
            sync_state_chip (Js.Unsafe.inject Js.undefined);
            set_text prompt_glyph ">>>";
            set_text timing_lbl "");
        Js._true);

  clear_btn##.onclick :=
    Dom.handler (fun _ ->
        clear_children (console_pre :> Dom.node Js.t);
        Js._true);

  copy_btn##.onclick :=
    Dom.handler (fun _ ->
        let clip = Js.Unsafe.get Dom_html.window##.navigator (js "clipboard") in
        if is_defined clip then (
          ignore (Js.Unsafe.meth_call clip "writeText" [| Js.Unsafe.inject editor##.value |]);
          set_text copy_btn "Copied";
          ignore
            (Dom_html.window##setTimeout
               (Js.wrap_callback (fun () -> set_text copy_btn "Copy"))
               (Js.float 1200.0)));
        Js._true);

  editor##.oninput :=
    Dom.handler (fun _ ->
        user_dirty_ref := true;
        sync_editor_view ();
        Js._true);

  editor##.onclick :=
    Dom.handler (fun _ ->
        sync_active_line_band ();
        let cur_l = cursor_line_no (to_s editor##.value) editor##.selectionStart in
        sync_gutter_highlights ed_gutter ~active:(Some cur_l) ~err:!err_line_ref;
        Js._true);

  editor##.onkeyup :=
    Dom.handler (fun _ ->
        sync_active_line_band ();
        let cur_l = cursor_line_no (to_s editor##.value) editor##.selectionStart in
        sync_gutter_highlights ed_gutter ~active:(Some cur_l) ~err:!err_line_ref;
        Js._true);

  editor##.onscroll :=
    Dom.handler (fun _ ->
        highlight_pre##.scrollTop := editor##.scrollTop;
        highlight_pre##.scrollLeft := editor##.scrollLeft;
        ed_gutter##.scrollTop := editor##.scrollTop;
        sync_active_line_band ();
        Js._true);

  editor##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let ctrl_or_meta = Js.to_bool ev##.ctrlKey || Js.to_bool ev##.metaKey in
        let shift = Js.to_bool ev##.shiftKey in
        if ctrl_or_meta && key = "Enter" then (
          Dom.preventDefault ev;
          run_script ();
          (if Dom_html.window##.innerWidth <= 900 then set_mob_tab "plots");
          Js._false)
        else if shift && key = "Enter" && not ctrl_or_meta then (
          Dom.preventDefault ev;
          run_selection_or_line ();
          Js._false)
        else if ctrl_or_meta && (key = "/" || key = "?") then (
          Dom.preventDefault ev;
          handle_toggle_comment ();
          Js._false)
        else if key = "Tab" then (
          Dom.preventDefault ev;
          handle_tab ~shift;
          Js._false)
        else if key = "Enter" && not ctrl_or_meta && not shift then (
          Dom.preventDefault ev;
          handle_enter_indent ();
          Js._false)
        else (
          err_line_ref := None;
          Js._true));

  prompt_row##.onsubmit :=
    Dom.handler (fun ev ->
        Dom.preventDefault ev;
        submit_repl ();
        Js._false);

  repl_input##.oninput :=
    Dom.handler (fun _ ->
        auto_size_repl_input ();
        Js._true);

  repl_input##.onkeydown :=
    Dom.handler (fun ev ->
        let key = to_s (Js.Optdef.get ev##.key (fun () -> js "")) in
        let shift = Js.to_bool ev##.shiftKey in
        let ctrl = Js.to_bool ev##.ctrlKey in
        let hlen = List.length !repl_history in
        if ctrl && (key = "l" || key = "L") then (
          Dom.preventDefault ev;
          clear_children (console_pre :> Dom.node Js.t);
          Js._false)
        else if key = "Enter" && not shift then (
          Dom.preventDefault ev;
          submit_repl ();
          Js._false)
        else if key = "Enter" && shift then (
          ignore
            (Dom_html.window##setTimeout
               (Js.wrap_callback auto_size_repl_input)
               (Js.float 0.0));
          Js._true)
        else if key = "Escape" then (
          compl_bar##setAttribute (js "hidden") (js "hidden");
          compl_bar##.style##.display := js "none";
          clear_children (compl_bar :> Dom.node Js.t);
          Js._false)
        else if key = "ArrowUp" && is_at_top repl_input && hlen > 0 then (
          Dom.preventDefault ev;
          let next_i = min (hlen - 1) (!hist_idx + 1) in
          hist_idx := next_i;
          repl_input##.value := js (List.nth !repl_history next_i);
          auto_size_repl_input ();
          repl_input##.selectionStart := 0;
          repl_input##.selectionEnd := 0;
          Js._false)
        else if key = "ArrowDown" && is_at_bottom repl_input then (
          Dom.preventDefault ev;
          if !hist_idx <= 0 then (
            hist_idx := -1;
            repl_input##.value := js "";
            auto_size_repl_input ())
          else (
            let next_i = !hist_idx - 1 in
            hist_idx := next_i;
            repl_input##.value := js (List.nth !repl_history next_i);
            auto_size_repl_input ());
          let len = String.length (to_s repl_input##.value) in
          repl_input##.selectionStart := len;
          repl_input##.selectionEnd := len;
          Js._false)
        else if key = "Tab" then (
          let cur = to_s repl_input##.value in
          let cursor = repl_input##.selectionStart in
          let rec scan i =
            if i <= 0 then 0
            else
              let c = cur.[i - 1] in
              if (c >= 'a' && c <= 'z')
                 || (c >= 'A' && c <= 'Z')
                 || (c >= '0' && c <= '9')
                 || c = '_' || c = '.'
              then scan (i - 1)
              else i
          in
          let start = scan cursor in
          let prefix = String.sub cur start (cursor - start) in
          if prefix <> "" then (
            Dom.preventDefault ev;
            send_worker "complete"
              [ ("prefix", Js.Unsafe.inject (js prefix)) ]
              (fun res ->
                let raw_items = Js.Unsafe.get res (js "items") in
                let items =
                  List.filter_map
                    (fun v -> if is_defined v then Some (to_s (Obj.magic v)) else None)
                    (json_obj_list raw_items)
                in
                match items with
                | [ single ] ->
                    let left = String.sub cur 0 start in
                    let right = String.sub cur cursor (String.length cur - cursor) in
                    repl_input##.value := js (left ^ single ^ right);
                    let new_c = start + String.length single in
                    repl_input##.selectionStart := new_c;
                    repl_input##.selectionEnd := new_c;
                    compl_bar##setAttribute (js "hidden") (js "hidden");
                    compl_bar##.style##.display := js "none";
                    clear_children (compl_bar :> Dom.node Js.t);
                    auto_size_repl_input ()
                | many when List.length many > 1 ->
                    let lcp = longest_common_prefix many in
                    if String.length lcp > String.length prefix then (
                      let left = String.sub cur 0 start in
                      let right = String.sub cur cursor (String.length cur - cursor) in
                      repl_input##.value := js (left ^ lcp ^ right);
                      let nc = start + String.length lcp in
                      repl_input##.selectionStart := nc;
                      repl_input##.selectionEnd := nc;
                      auto_size_repl_input ());
                    clear_children (compl_bar :> Dom.node Js.t);
                    compl_bar##removeAttribute (js "hidden");
                    compl_bar##.style##.display := js "";
                    List.iter
                      (fun item ->
                        let chip = mk ~cls:"py-repl-compl-chip" ~txt:item "button" in
                        chip##setAttribute (js "type") (js "button");
                        chip##.onclick :=
                          Dom.handler (fun _ ->
                              let now_val = to_s repl_input##.value in
                              let left = String.sub now_val 0 start in
                              let prefix_len =
                                if String.length lcp > String.length prefix then
                                  String.length lcp
                                else String.length prefix
                              in
                              let right =
                                if String.length now_val >= start + prefix_len then
                                  String.sub now_val (start + prefix_len)
                                    (String.length now_val - (start + prefix_len))
                                else ""
                              in
                              repl_input##.value := js (left ^ item ^ right);
                              let new_pos = start + String.length item in
                              repl_input##.selectionStart := new_pos;
                              repl_input##.selectionEnd := new_pos;
                              compl_bar##setAttribute (js "hidden") (js "hidden");
                              compl_bar##.style##.display := js "none";
                              clear_children (compl_bar :> Dom.node Js.t);
                              auto_size_repl_input ();
                              repl_input##focus;
                              Js._true);
                        Dom.appendChild compl_bar chip)
                      many
                | _ -> ());
            Js._false)
          else Js._true)
        else Js._true);

  (* Probe browser & local native capabilities, then warm WASM runtime on Playground *)
  probe_browser_caps ();
  probe_local_native ();
  let coi_pending = Js.Unsafe.get Dom_html.window (js "__bauxCoiPending") in
  let is_truthy v =
    Js.to_bool (Js.Unsafe.fun_call (Js.Unsafe.js_expr "Boolean") [| v |])
  in
  if is_truthy coi_pending then
    let warmed = ref false in
    let opts = Js.Unsafe.obj [| ("once", Js.Unsafe.inject Js._true) |] in
    let cb =
      Js.wrap_callback (fun (_ev : Js.Unsafe.any) ->
          if not !warmed then begin
            warmed := true;
            warm_kernel ()
          end)
    in
    ignore
      (Js.Unsafe.meth_call Dom_html.window "addEventListener"
         [| Js.Unsafe.inject (js "baux:coi-settled"); Js.Unsafe.inject cb; Js.Unsafe.inject opts |])
  else
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
                let pg_btn = mk ~cls:"code-pg" ~txt:"REPL" "button" in
                pg_btn##setAttribute (js "type") (js "button");
                pg_btn##setAttribute (js "title") (js "Open in REPL");
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
                                  append_console_chunk out "py-out-muted" "Done.\n";
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
                      append_console_chunk out "py-out-muted" "Running...\n";
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
                            append_console_chunk out "py-out-muted" "Done.\n";
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
                      Dom_html.window##.location##.href := js (get_base_url () ^ "repl.html");
                      Js._true))
  done

let init () : unit =
  (match Js.Opt.to_option (document##getElementById (js "py-playground-root")) with
  | Some root -> mount_playground root
  | None -> ());
  enhance_inline_blocks ()
