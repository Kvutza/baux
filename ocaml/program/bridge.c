// OCaml roots own Rust handles. Native work never retains an OCaml heap pointer.
#include "portable.h"
#include "model.h"
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include <stdlib.h>
#include <string.h>

#define HANDLE(v) (*(void**)Data_custom_val(v))
#define FINALIZER(name, drop) \
  static void name(value v) { if (HANDLE(v)) drop(HANDLE(v)); }
FINALIZER(sessionfinal, bauxprogram_drop)
FINALIZER(bufferfinal, bauxbuffer_drop)
FINALIZER(pendingfinal, bauxpending_drop)
FINALIZER(batchfinal, bauxbatch_drop)
FINALIZER(modelfinal, bauxmodel_drop)
#define OPS(name, final) \
  static struct custom_operations name = { \
    .identifier = "baux.program." #name, .finalize = final, \
    .compare = custom_compare_default, .hash = custom_hash_default, \
    .serialize = custom_serialize_default, .deserialize = custom_deserialize_default, \
    .compare_ext = custom_compare_ext_default, .fixed_length = custom_fixed_length_default }
OPS(sessionops, sessionfinal);
OPS(bufferops, bufferfinal);
OPS(pendingops, pendingfinal);
OPS(batchops, batchfinal);
OPS(modelops, modelfinal);

static value empty(struct custom_operations* ops) {
  value v = caml_alloc_custom(ops, sizeof(void*), 0, 1);
  HANDLE(v) = NULL;
  return v;
}
static void fail(void) { caml_failwith(bauxprogram_error()); }
static void modelfail(void) { caml_failwith(bauxmodel_error()); }

CAMLprim value bauxocaml_model_create(value seed) {
  CAMLparam1(seed);
  CAMLlocal1(result);
  result = empty(&modelops);
  uint64_t seed_value = Int64_val(seed);
  caml_enter_blocking_section();
  bauxmodel* model = bauxmodel_create(seed_value);
  caml_leave_blocking_section();
  if (!model) modelfail();
  HANDLE(result) = model;
  CAMLreturn(result);
}

CAMLprim value bauxocaml_model_parameters(value model) {
  CAMLparam1(model);
  CAMLreturn(Val_long(bauxmodel_parameters(HANDLE(model))));
}

CAMLprim value bauxocaml_model_rows(value model) {
  CAMLparam1(model);
  CAMLreturn(Val_long(bauxmodel_rows(HANDLE(model))));
}

CAMLprim value bauxocaml_model_submit(value model, value tokens, value labels, value mask) {
  CAMLparam4(model, tokens, labels, mask);
  CAMLlocal1(result);
  result = empty(&pendingops);
  struct caml_ba_array* t = Caml_ba_array_val(tokens);
  struct caml_ba_array* l = Caml_ba_array_val(labels);
  struct caml_ba_array* m = Caml_ba_array_val(mask);
  if (t->num_dims != 1 || l->num_dims != 1 || m->num_dims != 1 ||
      t->dim[0] != l->dim[0] || t->dim[0] != m->dim[0])
    caml_invalid_argument("model inputs require equal one-dimensional shapes");
  size_t count = t->dim[0];
  uint32_t* token_copy = malloc((count ? count : 1) * sizeof(*token_copy));
  uint32_t* label_copy = malloc((count ? count : 1) * sizeof(*label_copy));
  uint8_t* mask_copy = malloc(count ? count : 1);
  if (!token_copy || !label_copy || !mask_copy) {
    free(token_copy); free(label_copy); free(mask_copy);
    caml_raise_out_of_memory();
  }
  memcpy(token_copy, t->data, count * sizeof(*token_copy));
  memcpy(label_copy, l->data, count * sizeof(*label_copy));
  memcpy(mask_copy, m->data, count);
  bauxmodel* native_model = HANDLE(model);
  caml_enter_blocking_section();
  bauxpending* pending = bauxmodel_submit(native_model, token_copy, label_copy, mask_copy, count);
  free(token_copy); free(label_copy); free(mask_copy);
  caml_leave_blocking_section();
  if (!pending) modelfail();
  HANDLE(result) = pending;
  CAMLreturn(result);
}

// One allocation contains both descriptors and dimension arrays; no pointers
// into the moving OCaml heap cross the blocking section.
static bauxspec* specs(value values) {
  size_t count = Wosize_val(values), rank = 0;
  for (size_t i = 0; i < count; ++i) rank += Wosize_val(Field(Field(values, i), 1));
  bauxspec* result = calloc(1, count * sizeof(bauxspec) + rank * sizeof(size_t) + 1);
  if (!result) return NULL;
  size_t* dimensions = (size_t*)(result + count);
  for (size_t i = 0; i < count; ++i) {
    value pair = Field(values, i), shape = Field(pair, 1);
    result[i] = (bauxspec){dimensions, Wosize_val(shape), Long_val(Field(pair, 0))};
    for (size_t j = 0; j < result[i].rank; ++j) *dimensions++ = Long_val(Field(shape, j));
  }
  return result;
}

CAMLprim value bauxocaml_load(value bytes, value entry, value device, value ins, value outs) {
  CAMLparam5(bytes, entry, device, ins, outs);
  CAMLlocal1(result);
  result = empty(&sessionops);
  size_t length = caml_string_length(bytes), ni = Wosize_val(ins), no = Wosize_val(outs);
  // Copy descriptors before relinquishing the runtime lock.
  bauxspec* inputs = specs(ins);
  bauxspec* outputs = specs(outs);
  uint8_t* code = malloc(length ? length : 1);
  char* function = strdup(String_val(entry));
  char* driver = strdup(String_val(device));
  if (!inputs || !outputs || !code || !function || !driver) {
    free(inputs); free(outputs); free(code); free(function); free(driver);
    caml_raise_out_of_memory();
  }
  memcpy(code, Bytes_val(bytes), length);
  caml_enter_blocking_section();
  bauxprogram* program = bauxprogram_load(code, length, function, driver, inputs, ni, outputs, no);
  free(inputs); free(outputs); free(code); free(function); free(driver);
  caml_leave_blocking_section();
  if (!program) fail();
  HANDLE(result) = program;
  CAMLreturn(result);
}

CAMLprim value bauxocaml_put(value session, value dtype, value data) {
  CAMLparam3(session, dtype, data);
  CAMLlocal1(result);
  result = empty(&bufferops);
  struct caml_ba_array* a = Caml_ba_array_val(data);
  size_t shape[CAML_BA_MAX_NUM_DIMS];
  for (int i = 0; i < a->num_dims; ++i) shape[i] = a->dim[i];
  bauxspec spec = {shape, a->num_dims, Long_val(dtype)};
  size_t length = caml_ba_byte_size(a);
  void* source = a->data;
  bauxprogram* program = HANDLE(session);
  caml_enter_blocking_section();
  bauxbuffer* buffer = bauxprogram_put(program, &spec, source, length);
  caml_leave_blocking_section();
  if (!buffer) fail();
  HANDLE(result) = buffer;
  CAMLreturn(result);
}

CAMLprim value bauxocaml_read(value buffer, value data) {
  CAMLparam2(buffer, data);
  struct caml_ba_array* a = Caml_ba_array_val(data);
  void* destination = a->data;
  size_t length = caml_ba_byte_size(a);
  bauxbuffer* native = HANDLE(buffer);
  caml_enter_blocking_section();
  int ok = bauxbuffer_read(native, destination, length);
  caml_leave_blocking_section();
  if (!ok) fail();
  CAMLreturn(Val_unit);
}

static const bauxbuffer** handles(value values) {
  size_t count = Wosize_val(values);
  const bauxbuffer** result = malloc((count ? count : 1) * sizeof(*result));
  if (!result) caml_raise_out_of_memory();
  for (size_t i = 0; i < count; ++i) result[i] = HANDLE(Field(values, i));
  return result;
}
static value unpack(value batch) {
  CAMLparam1(batch);
  CAMLlocal2(result, buffer);
  size_t count = bauxbatch_count(HANDLE(batch));
  result = caml_alloc(count, 0);
  for (size_t i = 0; i < count; ++i) Store_field(result, i, Val_unit);
  for (size_t i = 0; i < count; ++i) {
    buffer = empty(&bufferops);
    bauxbuffer* native = bauxbatch_take(HANDLE(batch), i);
    if (!native) fail();
    HANDLE(buffer) = native;
    Store_field(result, i, buffer);
  }
  bauxbatch_drop(HANDLE(batch)); HANDLE(batch) = NULL;
  CAMLreturn(result);
}

CAMLprim value bauxocaml_run(value session, value values) {
  CAMLparam2(session, values);
  CAMLlocal1(batch);
  batch = empty(&batchops);
  const bauxbuffer** inputs = handles(values);
  size_t count = Wosize_val(values);
  bauxprogram* program = HANDLE(session);
  caml_enter_blocking_section();
  bauxbatch* native = bauxprogram_run(program, inputs, count);
  free(inputs);
  caml_leave_blocking_section();
  if (!native) fail();
  HANDLE(batch) = native;
  CAMLreturn(unpack(batch));
}

CAMLprim value bauxocaml_submit(value session, value values) {
  CAMLparam2(session, values);
  CAMLlocal1(result);
  result = empty(&pendingops);
  const bauxbuffer** inputs = handles(values);
  bauxpending* native = bauxprogram_submit(HANDLE(session), inputs, Wosize_val(values));
  free(inputs);
  if (!native) fail();
  HANDLE(result) = native;
  CAMLreturn(result);
}

// Baux_program's mutex serializes access to this consumable handle across domains.
CAMLprim value bauxocaml_ready(value pending) {
  CAMLparam1(pending);
  if (!HANDLE(pending)) caml_invalid_argument("invocation already consumed");
  CAMLreturn(Val_bool(bauxpending_ready(HANDLE(pending))));
}
CAMLprim value bauxocaml_wait(value pending) {
  CAMLparam1(pending);
  CAMLlocal1(batch);
  batch = empty(&batchops);
  bauxpending* call = HANDLE(pending);
  if (!call) caml_invalid_argument("invocation already consumed");
  HANDLE(pending) = NULL;
  caml_enter_blocking_section();
  bauxbatch* native = bauxpending_wait(call);
  caml_leave_blocking_section();
  if (!native) fail();
  HANDLE(batch) = native;
  CAMLreturn(unpack(batch));
}
