type t = private {
  batch_sizes : int list;
  workers : int list;
  block_pairs : int list;
  samples : int;
  dimension : int;
  quota_ms : int;
  seed : int;
  functions : string list;
}

val of_string : ?samples_override:int -> string -> t
val load : ?samples_override:int -> string -> t
val to_string : t -> string
