#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/printexc.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* 8-byte pointer + ownership flag inside an OCaml custom block.
   - Borrowed @ local arguments have owned = 0 (no Py_INCREF on entry, nulled
     before returning to Python).
   - Fresh / stolen @ unique results have owned = 1; C glue steals obj and sets
     obj = NULL, owned = 0 so the eventual OCaml GC finalizer is a no-op. */
struct baux_py_slot {
  PyObject *obj;
  int owned;
};

#define PY_SLOT(v) ((struct baux_py_slot *)Data_custom_val(v))

static void baux_py_finalize(value v) {
  struct baux_py_slot *s = PY_SLOT(v);
  if (s->owned && s->obj != NULL) {
    PyObject *o = s->obj;
    s->obj = NULL;
    s->owned = 0;
    if (Py_IsInitialized()) {
      PyGILState_STATE gstate = PyGILState_Ensure();
      Py_DECREF(o);
      PyGILState_Release(gstate);
    }
  }
}

static struct custom_operations baux_py_ops = {
    .identifier = "baux.oxcaml.py_obj",
    .finalize = baux_py_finalize,
    .compare = custom_compare_default,
    .hash = custom_hash_default,
    .serialize = custom_serialize_default,
    .deserialize = custom_deserialize_default,
    .compare_ext = custom_compare_ext_default,
    .fixed_length = custom_fixed_length_default,
};

static value alloc_py_slot(PyObject *obj, int owned) {
  value v = caml_alloc_custom(&baux_py_ops, sizeof(struct baux_py_slot), 0, 1);
  PY_SLOT(v)->obj = obj;
  PY_SLOT(v)->owned = owned;
  return v;
}

static PyObject *require_live(value v, const char *where) {
  PyObject *o = PY_SLOT(v)->obj;
  if (o == NULL) {
    caml_invalid_argument(where);
  }
  return o;
}

static void raise_py_error_in_ocaml(const char *fallback) {
  PyObject *ptype = NULL, *pvalue = NULL, *ptraceback = NULL;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", fallback);
  if (pvalue != NULL) {
    PyObject *s = PyObject_Str(pvalue);
    if (s != NULL) {
      const char *utf8 = PyUnicode_AsUTF8(s);
      if (utf8 != NULL) {
        snprintf(buf, sizeof(buf), "%s: %s", fallback, utf8);
      }
      Py_DECREF(s);
    }
  }
  Py_XDECREF(ptype);
  Py_XDECREF(pvalue);
  Py_XDECREF(ptraceback);
  caml_invalid_argument(buf);
}

/* --- Explicit Modal Reference Counting (newref & decref) --- */

CAMLprim value baux_py_newref(value v_borrowed) {
  CAMLparam1(v_borrowed);
  PyObject *o = require_live(v_borrowed, "Py_obj.newref: null handle");
  Py_INCREF(o);
  CAMLreturn(alloc_py_slot(o, 1));
}

CAMLprim value baux_py_decref(value v_unique) {
  CAMLparam1(v_unique);
  struct baux_py_slot *s = PY_SLOT(v_unique);
  if (!s->owned || s->obj == NULL) {
    caml_invalid_argument("Py_obj.decref: handle not owned");
  }
  PyObject *o = s->obj;
  s->obj = NULL;
  s->owned = 0;
  Py_DECREF(o);
  CAMLreturn(Val_unit);
}

CAMLprim value baux_py_refcount(value v_obj) {
  CAMLparam1(v_obj);
  PyObject *o = require_live(v_obj, "Py_obj.refcount: null handle");
  CAMLreturn(Val_long((long)Py_REFCNT(o)));
}

/* --- Borrowed @ local Readers (Zero Py_INCREF) --- */

CAMLprim value baux_py_to_string(value v_obj) {
  CAMLparam1(v_obj);
  CAMLlocal1(res);
  PyObject *o = require_live(v_obj, "Py_obj.to_string: null handle");
  Py_ssize_t len = 0;
  const char *utf8 = PyUnicode_AsUTF8AndSize(o, &len);
  if (utf8 == NULL) {
    raise_py_error_in_ocaml("Expected Python str");
  }
  res = caml_alloc_initialized_string((mlsize_t)len, utf8);
  CAMLreturn(res);
}

CAMLprim value baux_py_to_int(value v_obj) {
  CAMLparam1(v_obj);
  PyObject *o = require_live(v_obj, "Py_obj.to_int: null handle");
  long n = PyLong_AsLong(o);
  if (n == -1 && PyErr_Occurred()) {
    raise_py_error_in_ocaml("Expected Python int");
  }
  CAMLreturn(Val_long(n));
}

CAMLprim value baux_py_to_float(value v_obj) {
  CAMLparam1(v_obj);
  PyObject *o = require_live(v_obj, "Py_obj.to_float: null handle");
  double d = PyFloat_AsDouble(o);
  if (d == -1.0 && PyErr_Occurred()) {
    raise_py_error_in_ocaml("Expected Python float");
  }
  CAMLreturn(caml_copy_double(d));
}

CAMLprim value baux_py_to_f64_array(value v_obj) {
  CAMLparam1(v_obj);
  CAMLlocal1(arr);
  PyObject *o = require_live(v_obj, "Py_obj.to_f64_array: null handle");

  Py_buffer view;
  if (PyObject_GetBuffer(o, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0) {
    if (view.itemsize == (Py_ssize_t)sizeof(double) && view.ndim == 1 &&
        view.format != NULL &&
        (strcmp(view.format, "d") == 0 || strcmp(view.format, "=d") == 0 ||
         strcmp(view.format, "<d") == 0)) {
      Py_ssize_t n = view.shape[0];
      const double *src = (const double *)view.buf;
      arr = caml_alloc_float_array((mlsize_t)n);
      for (Py_ssize_t i = 0; i < n; i++) {
        Double_flat_field(arr, i) = src[i];
      }
      PyBuffer_Release(&view);
      CAMLreturn(arr);
    }
    PyBuffer_Release(&view);
  } else {
    PyErr_Clear();
  }

  PyObject *seq = PySequence_Fast(o, "Expected 1D float sequence");
  if (seq == NULL) {
    raise_py_error_in_ocaml("Expected 1D float sequence");
  }
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  PyObject **items = PySequence_Fast_ITEMS(seq);
  double *tmp = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
  if (tmp == NULL) {
    Py_DECREF(seq);
    caml_raise_out_of_memory();
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    double v = PyFloat_AsDouble(items[i]);
    if (v == -1.0 && PyErr_Occurred()) {
      free(tmp);
      Py_DECREF(seq);
      raise_py_error_in_ocaml("Non-numeric coordinate in sequence");
    }
    tmp[i] = v;
  }
  Py_DECREF(seq);
  arr = caml_alloc_float_array((mlsize_t)n);
  for (Py_ssize_t i = 0; i < n; i++) {
    Double_flat_field(arr, i) = tmp[i];
  }
  free(tmp);
  CAMLreturn(arr);
}

/* --- Fresh Single-Owner @ unique Constructors (ob_refcnt = 1) --- */

CAMLprim value baux_py_of_float(value v_f) {
  CAMLparam1(v_f);
  PyObject *o = PyFloat_FromDouble(Double_val(v_f));
  if (o == NULL) raise_py_error_in_ocaml("PyFloat_FromDouble failed");
  CAMLreturn(alloc_py_slot(o, 1));
}

CAMLprim value baux_py_of_int(value v_i) {
  CAMLparam1(v_i);
  PyObject *o = PyLong_FromLong(Long_val(v_i));
  if (o == NULL) raise_py_error_in_ocaml("PyLong_FromLong failed");
  CAMLreturn(alloc_py_slot(o, 1));
}

CAMLprim value baux_py_of_bool(value v_b) {
  CAMLparam1(v_b);
  PyObject *o = Bool_val(v_b) ? Py_True : Py_False;
  Py_INCREF(o);
  CAMLreturn(alloc_py_slot(o, 1));
}

CAMLprim value baux_py_of_string(value v_s) {
  CAMLparam1(v_s);
  PyObject *o =
      PyUnicode_FromStringAndSize(String_val(v_s), (Py_ssize_t)caml_string_length(v_s));
  if (o == NULL) raise_py_error_in_ocaml("PyUnicode_FromStringAndSize failed");
  CAMLreturn(alloc_py_slot(o, 1));
}

CAMLprim value baux_py_list_create(value v_len) {
  CAMLparam1(v_len);
  PyObject *lst = PyList_New((Py_ssize_t)Long_val(v_len));
  if (lst == NULL) raise_py_error_in_ocaml("PyList_New failed");
  CAMLreturn(alloc_py_slot(lst, 1));
}

/* PyList_SetItem borrows lst (@ local) and STEALS item (@ unique)! */
CAMLprim value baux_py_list_set_steal(value v_lst, value v_idx, value v_item) {
  CAMLparam3(v_lst, v_idx, v_item);
  PyObject *lst = require_live(v_lst, "Py_obj.List.set_steal: null list");
  struct baux_py_slot *is = PY_SLOT(v_item);
  if (!is->owned || is->obj == NULL) {
    caml_invalid_argument("Py_obj.List.set_steal: item not uniquely owned");
  }
  PyObject *item = is->obj;
  is->obj = NULL;
  is->owned = 0;
  if (PyList_SetItem(lst, (Py_ssize_t)Long_val(v_idx), item) < 0) {
    raise_py_error_in_ocaml("PyList_SetItem failed");
  }
  CAMLreturn(Val_unit);
}

CAMLprim value baux_py_dict_create(value v_unit) {
  CAMLparam1(v_unit);
  PyObject *d = PyDict_New();
  if (d == NULL) raise_py_error_in_ocaml("PyDict_New failed");
  CAMLreturn(alloc_py_slot(d, 1));
}

/* PyDict_SetItemString borrows dict (@ local) and increments item's refcount,
   so we consume item's unique reference via Py_DECREF and null its slot. */
CAMLprim value baux_py_dict_set_string(value v_dict, value v_key, value v_val) {
  CAMLparam3(v_dict, v_key, v_val);
  PyObject *d = require_live(v_dict, "Py_obj.Dict.set_string: null dict");
  struct baux_py_slot *vs = PY_SLOT(v_val);
  if (!vs->owned || vs->obj == NULL) {
    caml_invalid_argument("Py_obj.Dict.set_string: value not uniquely owned");
  }
  PyObject *item = vs->obj;
  vs->obj = NULL;
  vs->owned = 0;
  int rc = PyDict_SetItemString(d, String_val(v_key), item);
  Py_DECREF(item);
  if (rc < 0) {
    raise_py_error_in_ocaml("PyDict_SetItemString failed");
  }
  CAMLreturn(Val_unit);
}

/* --- Scoped Zero-INCREF 2D float64 Buffer Borrowing --- */

#define MAX_MATRIX_SLOTS 16

struct matrix_slot {
  int active;
  Py_buffer view;
  const double *data;
  Py_ssize_t rows;
  Py_ssize_t cols;
};

static struct matrix_slot g_matrices[MAX_MATRIX_SLOTS];

CAMLprim value baux_py_acquire_matrix(value v_obj) {
  CAMLparam1(v_obj);
  CAMLlocal1(tup);
  PyObject *o = require_live(v_obj, "F64_buffer.with_matrix: null handle");

  int slot_idx = -1;
  for (int i = 0; i < MAX_MATRIX_SLOTS; i++) {
    if (!g_matrices[i].active) {
      slot_idx = i;
      break;
    }
  }
  if (slot_idx < 0) {
    caml_failwith("F64_buffer: exhausted active matrix view slots");
  }

  struct matrix_slot *ms = &g_matrices[slot_idx];
  if (PyObject_GetBuffer(o, &ms->view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) {
    raise_py_error_in_ocaml("Expected C-contiguous float64 2D buffer");
  }
  if (ms->view.ndim != 2 || ms->view.itemsize != (Py_ssize_t)sizeof(double) ||
      ms->view.format == NULL ||
      (strcmp(ms->view.format, "d") != 0 && strcmp(ms->view.format, "=d") != 0 &&
       strcmp(ms->view.format, "<d") != 0)) {
    PyBuffer_Release(&ms->view);
    caml_invalid_argument("Expected 2D C-contiguous float64 buffer");
  }
  ms->data = (const double *)ms->view.buf;
  ms->rows = ms->view.shape[0];
  ms->cols = ms->view.shape[1];
  ms->active = 1;

  tup = caml_alloc_tuple(3);
  Store_field(tup, 0, Val_long(slot_idx));
  Store_field(tup, 1, Val_long((long)ms->rows));
  Store_field(tup, 2, Val_long((long)ms->cols));
  CAMLreturn(tup);
}

CAMLprim value baux_py_release_matrix(value v_slot) {
  CAMLparam1(v_slot);
  long idx = Long_val(v_slot);
  if (idx >= 0 && idx < MAX_MATRIX_SLOTS && g_matrices[idx].active) {
    g_matrices[idx].active = 0;
    g_matrices[idx].data = NULL;
    PyBuffer_Release(&g_matrices[idx].view);
  }
  CAMLreturn(Val_unit);
}

double baux_py_matrix_get(value v_slot, value v_r, value v_c) {
  long idx = Long_val(v_slot);
  long r = Long_val(v_r);
  long c = Long_val(v_c);
  if (idx < 0 || idx >= MAX_MATRIX_SLOTS || !g_matrices[idx].active) {
    return NAN;
  }
  const struct matrix_slot *ms = &g_matrices[idx];
  if (r < 0 || r >= ms->rows || c < 0 || c >= ms->cols) {
    return NAN;
  }
  return ms->data[(size_t)r * (size_t)ms->cols + (size_t)c];
}

CAMLprim value baux_py_matrix_get_byte(value v_slot, value v_r, value v_c) {
  return caml_copy_double(baux_py_matrix_get(v_slot, v_r, v_c));
}

CAMLprim value baux_py_output_create(value v_count) {
  CAMLparam1(v_count);
  Py_ssize_t n = (Py_ssize_t)Long_val(v_count);
  if (n < 0) caml_invalid_argument("Negative output count");
  PyObject *ba = PyByteArray_FromStringAndSize(NULL, n * (Py_ssize_t)sizeof(double));
  if (ba == NULL) raise_py_error_in_ocaml("PyByteArray_FromStringAndSize failed");
  CAMLreturn(alloc_py_slot(ba, 1));
}

CAMLprim value baux_py_output_set(value v_buf, value v_idx, double val) {
  PyObject *ba = PY_SLOT(v_buf)->obj;
  long idx = Long_val(v_idx);
  if (ba != NULL && idx >= 0 &&
      (idx + 1) * (Py_ssize_t)sizeof(double) <= PyByteArray_GET_SIZE(ba)) {
    double *dst = (double *)PyByteArray_AS_STRING(ba);
    dst[idx] = val;
  }
  return Val_unit;
}

CAMLprim value baux_py_output_set_byte(value v_buf, value v_idx, value v_val) {
  return baux_py_output_set(v_buf, v_idx, Double_val(v_val));
}

/* --- Callback Registration & Stealing Python C-Extension Entrypoints --- */

static value g_fn0[2] = {Val_unit, Val_unit};
static value g_fn1[2] = {Val_unit, Val_unit};
static value g_fn2[4] = {Val_unit, Val_unit, Val_unit, Val_unit};
static value g_fn5[2] = {Val_unit, Val_unit};
static int g_roots_registered = 0;

static void ensure_roots(void) {
  if (!g_roots_registered) {
    for (int i = 0; i < 2; i++) caml_register_generational_global_root(&g_fn0[i]);
    for (int i = 0; i < 2; i++) caml_register_generational_global_root(&g_fn1[i]);
    for (int i = 0; i < 4; i++) caml_register_generational_global_root(&g_fn2[i]);
    for (int i = 0; i < 2; i++) caml_register_generational_global_root(&g_fn5[i]);
    g_roots_registered = 1;
  }
}

CAMLprim value baux_py_set_slot0(value v_idx, value v_fn) {
  CAMLparam2(v_idx, v_fn);
  ensure_roots();
  long i = Long_val(v_idx);
  if (i >= 0 && i < 2) caml_modify_generational_global_root(&g_fn0[i], v_fn);
  CAMLreturn(Val_unit);
}

CAMLprim value baux_py_set_slot1(value v_idx, value v_fn) {
  CAMLparam2(v_idx, v_fn);
  ensure_roots();
  long i = Long_val(v_idx);
  if (i >= 0 && i < 2) caml_modify_generational_global_root(&g_fn1[i], v_fn);
  CAMLreturn(Val_unit);
}

CAMLprim value baux_py_set_slot2(value v_idx, value v_fn) {
  CAMLparam2(v_idx, v_fn);
  ensure_roots();
  long i = Long_val(v_idx);
  if (i >= 0 && i < 4) caml_modify_generational_global_root(&g_fn2[i], v_fn);
  CAMLreturn(Val_unit);
}

CAMLprim value baux_py_set_slot5(value v_idx, value v_fn) {
  CAMLparam2(v_idx, v_fn);
  ensure_roots();
  long i = Long_val(v_idx);
  if (i >= 0 && i < 2) caml_modify_generational_global_root(&g_fn5[i], v_fn);
  CAMLreturn(Val_unit);
}

static PyObject *steal_or_raise(value res) {
  if (Is_exception_result(res)) {
    value exn = Extract_exception(res);
    char *msg = caml_format_exception(exn);
    PyErr_SetString(PyExc_ValueError, msg);
    free(msg);
    return NULL;
  }
  struct baux_py_slot *s = PY_SLOT(res);
  PyObject *out = s->obj;
  int was_owned = s->owned;
  s->obj = NULL;
  s->owned = 0;
  if (out == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "OxCaml returned null Py_obj.t");
    return NULL;
  }
  if (!was_owned) {
    Py_INCREF(out);
  }
  return out;
}

static PyObject *py_catalogue(PyObject *self, PyObject *Py_UNUSED(ignored)) {
  (void)self;
  CAMLparam0();
  CAMLlocal1(res);
  res = caml_callback_exn(g_fn0[0], Val_unit);
  PyObject *out = steal_or_raise(res);
  CAMLreturnT(PyObject *, out);
}

static PyObject *py_describe(PyObject *self, PyObject *args) {
  (void)self;
  PyObject *id_obj = NULL;
  PyObject *dim_obj = NULL;
  if (!PyArg_UnpackTuple(args, "describe", 2, 2, &id_obj, &dim_obj)) {
    return NULL;
  }
  CAMLparam0();
  CAMLlocal3(v_a, v_b, res);
  v_a = alloc_py_slot(id_obj, 0);
  v_b = alloc_py_slot(dim_obj, 0);
  res = caml_callback2_exn(g_fn2[0], v_a, v_b);
  PY_SLOT(v_a)->obj = NULL;
  PY_SLOT(v_b)->obj = NULL;
  PyObject *out = steal_or_raise(res);
  CAMLreturnT(PyObject *, out);
}

static PyObject *py_eval_one(PyObject *self, PyObject *args) {
  (void)self;
  PyObject *id_obj = NULL;
  PyObject *pt_obj = NULL;
  if (!PyArg_UnpackTuple(args, "eval_one", 2, 2, &id_obj, &pt_obj)) {
    return NULL;
  }
  CAMLparam0();
  CAMLlocal3(v_a, v_b, res);
  v_a = alloc_py_slot(id_obj, 0);
  v_b = alloc_py_slot(pt_obj, 0);
  res = caml_callback2_exn(g_fn2[1], v_a, v_b);
  PY_SLOT(v_a)->obj = NULL;
  PY_SLOT(v_b)->obj = NULL;
  PyObject *out = steal_or_raise(res);
  CAMLreturnT(PyObject *, out);
}

static PyObject *py_eval_batch(PyObject *self, PyObject *args) {
  (void)self;
  PyObject *id_obj = NULL, *mat_obj = NULL, *backend_obj = NULL;
  PyObject *workers_obj = NULL, *bp_obj = NULL;
  if (!PyArg_UnpackTuple(args, "eval_batch", 5, 5, &id_obj, &mat_obj,
                         &backend_obj, &workers_obj, &bp_obj)) {
    return NULL;
  }
  CAMLparam0();
  CAMLlocalN(v_args, 5);
  CAMLlocal1(res);
  v_args[0] = alloc_py_slot(id_obj, 0);
  v_args[1] = alloc_py_slot(mat_obj, 0);
  v_args[2] = alloc_py_slot(backend_obj, 0);
  v_args[3] = alloc_py_slot(workers_obj, 0);
  v_args[4] = alloc_py_slot(bp_obj, 0);
  res = caml_callbackN_exn(g_fn5[0], 5, v_args);
  for (int i = 0; i < 5; i++) {
    PY_SLOT(v_args[i])->obj = NULL;
  }
  PyObject *out = steal_or_raise(res);
  CAMLreturnT(PyObject *, out);
}

static PyObject *py_refcount_probe(PyObject *self, PyObject *arg) {
  (void)self;
  CAMLparam0();
  CAMLlocal2(v_a, res);
  v_a = alloc_py_slot(arg, 0);
  res = caml_callback_exn(g_fn1[0], v_a);
  PY_SLOT(v_a)->obj = NULL;
  PyObject *out = steal_or_raise(res);
  CAMLreturnT(PyObject *, out);
}

static PyMethodDef baux_oxcaml_methods[] = {
    {"catalogue", py_catalogue, METH_NOARGS,
     "Return metadata for all 56 OxCaml benchmark problems."},
    {"describe", py_describe, METH_VARARGS,
     "Describe problem (id, dim) with bounds and dimensionality rule."},
    {"eval_one", py_eval_one, METH_VARARGS,
     "Evaluate one benchmark function at a 1D point using unboxed float#."},
    {"eval_batch", py_eval_batch, METH_VARARGS,
     "Evaluate (N, D) float64 batch using Scalar float# or NEON float64x2# SIMD."},
    {"refcount_probe", py_refcount_probe, METH_O,
     "Verify @ local zero-INCREF borrow, with_release, and @ unique steal."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef baux_oxcaml_module = {
    PyModuleDef_HEAD_INIT,
    "_baux_oxcaml",
    "OxCaml 5.2.0+ox 56-function benchmark suite with modal Python refcounting",
    -1,
    baux_oxcaml_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

static int g_ocaml_booted = 0;

PyMODINIT_FUNC PyInit__baux_oxcaml(void) {
  if (!g_ocaml_booted) {
    char *argv[2] = {"_baux_oxcaml", NULL};
    caml_startup(argv);
    g_ocaml_booted = 1;
  }
  return PyModule_Create(&baux_oxcaml_module);
}
