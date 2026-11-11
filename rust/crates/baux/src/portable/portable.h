#pragma once
#include <stddef.h>
#include <stdint.h>

// Handles are owned and must be released once with the matching drop function.
// Input pointers/ranges must be valid, aligned and remain live during each call.
// No host data pointer is retained. Buffers and pending calls retain sessions.
// NULL result / zero read status means failure. error() is thread-local and
// valid until the next failing call on that thread. No Rust panic crosses C.
typedef struct bauxprogram bauxprogram;
typedef struct bauxbuffer bauxbuffer;
typedef struct bauxpending bauxpending;
typedef struct bauxbatch bauxbatch;
// dtype codes: bool, i32, i64, u32, u64, f16, bf16, f32, f64 (0..8).
typedef struct { const size_t* shape; size_t rank; size_t dtype; } bauxspec;
const char* bauxprogram_error(void);
// Native StableHLO -> TIR/Relax CPU compilation. importer is the bundled
// hlo_import executable path; its LLVM is isolated in a child process.
// Blocks; uploads no inputs. Specs must match the exported function. Returns
// an owning session, or NULL with details in bauxprogram_error(). Source and
// metadata ranges are borrowed only during this call. Buffers retain sessions.
bauxprogram* bauxprogram_compile(const uint8_t* source, size_t length,
                                 const char* entry, const char* importer,
                                 const bauxspec* inputs, size_t input_count,
                                 const bauxspec* outputs, size_t output_count);
// Compile StableHLO source with a main entry through the native XLA CPU plugin.
bauxprogram* bauxprogram_xla(const uint8_t*, size_t, const char*,
                             const bauxspec*, size_t, const bauxspec*, size_t);
bauxprogram* bauxprogram_load(const uint8_t*, size_t, const char*, const char*,
                              const bauxspec*, size_t, const bauxspec*, size_t);
// Load a trusted TVM shared library by path, retaining its native module.
bauxprogram* bauxprogram_tvm(const char*, const char*, const char*,
                             const bauxspec*, size_t, const bauxspec*, size_t);
void bauxprogram_drop(bauxprogram*);
bauxbuffer* bauxprogram_put(const bauxprogram*, const bauxspec*, const uint8_t*, size_t);
void bauxbuffer_drop(bauxbuffer*);
int bauxbuffer_read(const bauxbuffer*, uint8_t*, size_t);
bauxbatch* bauxprogram_run(const bauxprogram*, const bauxbuffer* const*, size_t);
bauxpending* bauxprogram_submit(const bauxprogram*, const bauxbuffer* const*, size_t);
void bauxpending_drop(bauxpending*);
int bauxpending_ready(const bauxpending*);
// Consumes pending, including on failure; caller must clear its handle first.
bauxbatch* bauxpending_wait(bauxpending*);
size_t bauxbatch_count(const bauxbatch*);
// Returns a new owning buffer reference; batch remains usable.
bauxbuffer* bauxbatch_take(const bauxbatch*, size_t);
void bauxbatch_drop(bauxbatch*);
