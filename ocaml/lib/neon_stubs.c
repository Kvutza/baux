#include <arm_neon.h>
#include <sleef.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>

/* Native-only library. All native routines operate on registers, do not
   allocate OCaml values, and retain no pointers into the OCaml heap. */
CAMLprim value baux_simd_bytecode(value unused) {
  (void)unused;
  caml_failwith("BAUX NEON execution requires native code");
}
float64x2_t baux_neon_abs(float64x2_t x) { return vabsq_f64(x); }
float64x2_t baux_neon_neg(float64x2_t x) { return vnegq_f64(x); }
float64x2_t baux_neon_sin(float64x2_t x) { return Sleef_sind2_u10advsimd(x); }
float64x2_t baux_neon_cos(float64x2_t x) { return Sleef_cosd2_u10advsimd(x); }
float64x2_t baux_neon_exp(float64x2_t x) { return Sleef_expd2_u10advsimd(x); }
float64x2_t baux_neon_log(float64x2_t x) { return Sleef_logd2_u10advsimd(x); }
float64x2_t baux_neon_pow(float64x2_t x, float64x2_t y) {
  return Sleef_powd2_u10advsimd(x, y);
}
/* The compiler also retains external addresses in its metadata. These native
   definitions match the builtins when a call is not intrinsified. */
float64x2_t caml_neon_float64x2_add(float64x2_t x, float64x2_t y) { return vaddq_f64(x,y); }
float64x2_t caml_neon_float64x2_sub(float64x2_t x, float64x2_t y) { return vsubq_f64(x,y); }
float64x2_t caml_neon_float64x2_mul(float64x2_t x, float64x2_t y) { return vmulq_f64(x,y); }
float64x2_t caml_neon_float64x2_div(float64x2_t x, float64x2_t y) { return vdivq_f64(x,y); }
float64x2_t caml_neon_float64x2_sqrt(float64x2_t x) { return vsqrtq_f64(x); }
float64x2_t caml_neon_float64x2_zip1(float64x2_t x, float64x2_t y) { return vzip1q_f64(x,y); }
float64x2_t caml_neon_float64x2_zip2(float64x2_t x, float64x2_t y) { return vzip2q_f64(x,y); }
float64x2_t caml_float64x2_low_of_float(double x) { return vsetq_lane_f64(x,vdupq_n_f64(0),0); }
double caml_float64x2_low_to_float(float64x2_t x) { return vgetq_lane_f64(x,0); }

#include <time.h>
double baux_monotonic_seconds(value unit) {
  (void)unit;
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}
