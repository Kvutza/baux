(* Structured native execution. IREE runtime semantics:
   https://iree.dev/reference/bindings/c-api/#runtime-api
   Bigarray storage and roots follow the OCaml C interface contract:
   https://ocaml.org/manual/5.2/intfc.html *)
open Bigarray

type dtype = Bool | I32 | I64 | U32 | U64 | F16 | BF16 | F32 | F64
type device = Cpu | Metal
type session : value mod contended portable
type allocation : value mod contended portable
type invocation : value mod contended portable
type model : value mod contended portable
type model_backend = Native | Tvm

type ('a, 'b) buffer = {
  allocation : allocation;
  kind : ('a, 'b) kind;
  dimensions : int list;
  dtype : int;
}

type packed = Data : ('a, 'b, c_layout) Genarray.t -> packed

type ('host, 'resident) layout = {
  specs : (int * int list) list;
  flatten : 'host -> packed list @@ portable;
  handles : 'resident -> allocation list @@ portable;
  pack : allocation list -> 'resident * allocation list @@ portable;
  read : 'resident -> 'host @@ portable;
}

external load_native :
  bytes ->
  string ->
  string ->
  (int * int array) array ->
  (int * int array) array ->
  session @@ portable = "bauxocaml_load"

external put_native :
  session -> int -> ('a, 'b, c_layout) Genarray.t -> allocation @@ portable
  = "bauxocaml_put"

external tvm_native :
  string ->
  string ->
  string ->
  (int * int array) array ->
  (int * int array) array ->
  session @@ portable = "bauxocaml_tvm"

external xla_native :
  string ->
  string ->
  string ->
  (int * int array) array ->
  (int * int array) array ->
  session @@ portable = "bauxocaml_xla"

external read_native : allocation -> ('a, 'b, c_layout) Genarray.t -> unit
  @@ portable = "bauxocaml_read"

external run_native : session -> allocation array -> allocation array
  @@ portable = "bauxocaml_run"

external submit_native : session -> allocation array -> invocation @@ portable
  = "bauxocaml_submit"

external ready_native : invocation -> bool @@ portable = "bauxocaml_ready"

external wait_native : invocation -> allocation array @@ portable
  = "bauxocaml_wait"

external create_model_native : int64 -> int -> model @@ portable
  = "bauxocaml_model_create"

external model_parameters : model -> int @@ portable
  = "bauxocaml_model_parameters"

external model_rows : model -> int @@ portable = "bauxocaml_model_rows"

external submit_model_native :
  model ->
  (int32, int32_elt, c_layout) Genarray.t ->
  (int32, int32_elt, c_layout) Genarray.t ->
  (int, int8_unsigned_elt, c_layout) Genarray.t ->
  invocation @@ portable = "bauxocaml_model_submit"

let code = function
  | Bool -> 0
  | I32 -> 1
  | I64 -> 2
  | U32 -> 3
  | U64 -> 4
  | F16 -> 5
  | BF16 -> 6
  | F32 -> 7
  | F64 -> 8

let default : type a b. (a, b) kind -> dtype = function
  | Float32 -> F32
  | Float64 -> F64
  | Int32 -> I32
  | Int64 -> I64
  | _ -> invalid_arg "choose an explicit dtype for raw tensor storage"

let compatible : type a b. (a, b) kind -> dtype -> bool =
 fun kind dtype ->
  match (kind, dtype) with
  | Int8_unsigned, Bool
  | Int16_unsigned, (F16 | BF16)
  | Int32, (I32 | U32)
  | Int64, (I64 | U64)
  | Float32, F32
  | Float64, F64 ->
      true
  | _ -> false

let tensor ?dtype kind dimensions =
  let dtype = match dtype with Some d -> d | None -> default kind in
  if not (compatible kind dtype) then
    invalid_arg "dtype and storage kind differ";
  let dimensions = Array.to_list dimensions in
  List.iter
    (fun n -> if n < 0 then invalid_arg "negative tensor dimension")
    dimensions;
  {
    specs = [ (code dtype, dimensions) ];
    flatten =
      (fun value ->
        if Array.to_list (Genarray.dims value) <> dimensions then
          invalid_arg "tensor shape differs";
        [ Data value ]);
    handles =
      (fun value ->
        if value.dimensions <> dimensions || value.dtype <> code dtype then
          invalid_arg "buffer shape differs";
        [ value.allocation ]);
    pack =
      (function
      | allocation :: rest ->
          ({ allocation; kind; dimensions; dtype = code dtype }, rest)
      | [] -> invalid_arg "missing resident tensor");
    read =
      (fun value ->
        if value.dimensions <> dimensions || value.dtype <> code dtype then
          invalid_arg "buffer shape differs";
        let result = Genarray.create kind c_layout (Array.of_list dimensions) in
        read_native value.allocation result;
        result);
  }

let unit =
  {
    specs = [];
    flatten = (fun () -> []);
    handles = (fun () -> []);
    pack = (fun xs -> ((), xs));
    read = (fun () -> ());
  }

let pair left right =
  {
    specs = left.specs @ right.specs;
    flatten = (fun (a, b) -> left.flatten a @ right.flatten b);
    handles = (fun (a, b) -> left.handles a @ right.handles b);
    pack =
      (fun xs ->
        let a, xs = left.pack xs in
        let b, xs = right.pack xs in
        ((a, b), xs));
    read = (fun (a, b) -> (left.read a, right.read b));
  }

let array count item =
  if count < 0 then invalid_arg "negative layout length";
  let check values =
    if Array.length values <> count then invalid_arg "tree length differs"
  in
  {
    specs = List.concat (List.init count (fun _ -> item.specs));
    flatten =
      (fun values ->
        check values;
        Array.to_list values |> List.concat_map item.flatten);
    handles =
      (fun values ->
        check values;
        Array.to_list values |> List.concat_map item.handles);
    pack =
      (fun xs ->
        let rest = ref xs in
        let values =
          Array.init count (fun _ ->
              let value, tail = item.pack !rest in
              rest := tail;
              value)
        in
        (values, !rest));
    read =
      (fun values ->
        check values;
        Array.map item.read values);
  }

let map ~host:(encode, decode) ~resident:(unpack, pack) layout =
  {
    specs = layout.specs;
    flatten = (fun value -> layout.flatten (encode value));
    handles = (fun value -> layout.handles (unpack value));
    pack =
      (fun xs ->
        let value, rest = layout.pack xs in
        (pack value, rest));
    read = (fun value -> decode (layout.read (unpack value)));
  }

let reconstruct layout handles =
  let result, rest = layout.pack (Array.to_list handles) in
  if rest <> [] then invalid_arg "extra resident tensors";
  result

type ('host, 'resident, 'output, 'result) t = {
  session : session;
  inputs : ('host, 'resident) layout;
  outputs : ('output, 'result) layout;
}

type 'result pending = {
  invocation : invocation;
  finish : allocation array -> 'result @@ portable;
  guard : Mutex.t;
  consumed : bool Atomic.t;
}

let specs layout =
  List.map (fun (dtype, shape) -> (dtype, Array.of_list shape)) layout.specs
  |> Array.of_list

let load ~bytecode ~entry ~device ~inputs ~outputs =
  let device = match device with Cpu -> "cpu" | Metal -> "metal" in
  let session =
    load_native bytecode entry device (specs inputs) (specs outputs)
  in
  { session; inputs; outputs }

let load_tvm ~library ~entry ~device ~inputs ~outputs =
  if String.contains library '\000' then invalid_arg "library path contains NUL";
  let device = match device with Cpu -> "cpu" | Metal -> "metal" in
  let session =
    tvm_native library entry device (specs inputs) (specs outputs)
  in
  { session; inputs; outputs }

let compile_xla ~source ~device ~inputs ~outputs =
  let device = match device with Cpu -> "cpu" | Metal -> "metal" in
  let session =
    xla_native source "main" device (specs inputs) (specs outputs)
  in
  { session; inputs; outputs }

let put program values =
  let buffers =
    List.map2
      (fun (dtype, _) (Data data) -> put_native program.session dtype data)
      program.inputs.specs
      (program.inputs.flatten values)
    |> Array.of_list
  in
  reconstruct program.inputs buffers

let read layout values = layout.read values

let run program values =
  let buffers = program.inputs.handles values |> Array.of_list in
  run_native program.session buffers |> reconstruct program.outputs

let submit program values =
  let buffers = program.inputs.handles values |> Array.of_list in
  let invocation = submit_native program.session buffers in
  {
    invocation;
    finish = reconstruct program.outputs;
    guard = Mutex.create ();
    consumed = Atomic.make false;
  }

let locked pending f =
  Mutex.lock pending.guard;
  Fun.protect ~finally:(fun () -> Mutex.unlock pending.guard) f

let ready pending =
  locked pending (fun () ->
      if Atomic.get pending.consumed then
        invalid_arg "invocation already consumed";
      ready_native pending.invocation)

let wait pending =
  locked pending (fun () ->
      if Atomic.get pending.consumed then
        invalid_arg "invocation already consumed";
      Atomic.set pending.consumed true;
      wait_native pending.invocation |> pending.finish)

type model_output = {
  token_losses : (float, float32_elt) buffer;
  sequence_nlls : (float, float32_elt) buffer;
}

type model_observation = {
  token_losses : (float, float32_elt, c_layout) Genarray.t;
  sequence_nlls : (float, float32_elt, c_layout) Genarray.t;
}

let create_on backend ~seed = create_model_native seed (match backend with Native -> 0 | Tvm -> 1)
let create_model ~seed = create_on Native ~seed

let model_layout model =
  map
    ~host:
      ( (fun (result : model_observation) ->
          (result.token_losses, result.sequence_nlls)),
        fun (token_losses, sequence_nlls) : model_observation ->
          { token_losses; sequence_nlls } )
    ~resident:
      ( (fun (result : model_output) ->
          (result.token_losses, result.sequence_nlls)),
        fun (token_losses, sequence_nlls) : model_output ->
          { token_losses; sequence_nlls } )
    (pair (tensor float32 [| model_rows model |]) (tensor float32 [| 2 |]))

let submit_model model ~tokens ~labels ~mask =
  let outputs = model_layout model in
  let invocation = submit_model_native model tokens labels mask in
  {
    invocation;
    finish = reconstruct outputs;
    guard = Mutex.create ();
    consumed = Atomic.make false;
  }

let read_model model output = (model_layout model).read output
