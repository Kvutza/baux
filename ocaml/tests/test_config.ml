open Baux_benchmarks

let valid =
  {|version=2
experiment="oxcaml-catalogue"
batch_sizes=[1,256]
workers=[1,4]
block_pairs=[32,128]
samples=2
dimension=16
quota_ms=20
seed=42
functions=["sphere", "ackley"]
|}

let reject text =
  match Experiment_config.of_string text with
  | exception Invalid_argument _ -> ()
  | _ -> failwith ("Accepted invalid TOML: " ^ text)

let () =
  let c = Experiment_config.of_string valid in
  assert (c.workers = [ 1; 4 ] && c.block_pairs = [ 32; 128 ]);
  assert (
    Experiment_config.to_string
      (Experiment_config.of_string (Experiment_config.to_string c))
    = Experiment_config.to_string c);
  assert ((Experiment_config.of_string ~samples_override:7 valid).samples = 7);
  List.iter reject
    [
      valid ^ "typo=1\n";
      valid ^ "samples=3\n";
      "version=2\n";
      "version=[\n";
      {|version=2
experiment="oxcaml-catalogue"
batch_sizes=[0]
workers=[1]
samples=2
dimension=16
|};
    ];
  let replace old replacement =
    String.concat replacement (Str.split_delim (Str.regexp_string old) valid)
  in
  List.iter
    (fun (old, value) -> reject (replace old value))
    [
      ("workers=[1,4]", "workers=[0]");
      ("workers=[1,4]", "workers=[1,1]");
      ("samples=2", "samples=0");
      ("dimension=16", "dimension=-1");
      ("quota_ms=20", "quota_ms=0");
      ("batch_sizes=[1,256]", "batch_sizes=[]");
      ("block_pairs=[32,128]", "block_pairs=[1.5]");
      ("functions=[\"sphere\", \"ackley\"]", "functions=[\"unknown\"]");
    ];
  print_endline "TOML syntax, schema, validation and round-trip tests passed."
