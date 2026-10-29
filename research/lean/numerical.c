/* Scalar checker bridge. Compile against the project's pinned Lean headers.
 * This is runtime plumbing, not an independently verified compiler/runtime.
 * All functions run on one dedicated host thread; no Lean objects cross FFI.
 */
#include <lean/lean.h>
#include <stdint.h>

extern void lean_initialize_runtime_module(void);
extern void lean_initialize_thread(void);
extern void lean_finalize_thread(void);
extern lean_object *initialize_baux__research_BauxResearch_Semantics_NumericalRuntime(uint8_t);

LEAN_EXPORT uint8_t baux_numerical_init(void) {
    lean_initialize_runtime_module();
    lean_initialize_thread();
    lean_object *result = initialize_baux__research_BauxResearch_Semantics_NumericalRuntime(1);
    if (lean_io_result_is_error(result)) {
        lean_io_result_show_error(result);
        lean_dec(result);
        lean_finalize_thread();
        return 0;
    }
    lean_dec(result);
    lean_io_mark_end_initialization();
    return 1;
}

LEAN_EXPORT void baux_numerical_fini(void) {
    lean_finalize_thread();
}
