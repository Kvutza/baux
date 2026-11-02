#ifndef BAUX_NATIVE_FLAME_H
#define BAUX_NATIVE_FLAME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BauxFlameConfig {
    uint32_t layers, width, heads, vocab, dense_width, expert_width;
    uint32_t shared_width, experts, top_k, context;
    float epsilon, rope_base;
} BauxFlameConfig;

/* Device 0; FP32 compute from canonical, name-sorted, contiguous BF16 weights.
 * weights_len counts uint16_t elements, not bytes. Weights must be ready on
 * the CUDA legacy default stream. Handles are not safe for concurrent use.
 * max_tokens is the maximum full sequence length of a single problem.
 * All calls finish device work before returning, including failure paths.
 */
void* baux_flame_create(const BauxFlameConfig* config, uint32_t max_tokens,
                        char* error, size_t error_capacity);
void baux_flame_destroy(void* engine);
/* Sum of owned cudaMalloc allocation sizes; excludes cuBLAS internal storage. */
uint64_t baux_flame_workspace(const void* engine);
/* Canonical BF16 element count; zero for a null handle. */
uint64_t baux_flame_weights_len(const void* engine);

/* Flattened, unpadded problems processed sequentially. Nonzero masks select
 * targets; mask[0] never contributes. Each problem needs a scored target.
 * Returns 0 on success, 1 on failure; outputs are unspecified on failure.
 */
int baux_flame_losses(void* engine, const uint16_t* GPU_BF16_weights,
                      size_t weights_len, const int32_t* HOST_tokens_flat,
                      const uint8_t* HOST_masks_flat,
                      const uint32_t* HOST_lengths, size_t batch,
                      float* HOST_losses, char* error, size_t error_capacity);

/* HOST_logits receives length * vocab FP32 values in row-major order. */
int baux_flame_logits(void* engine, const uint16_t* GPU_weights,
                      size_t weights_len, const int32_t* HOST_tokens,
                      uint32_t length, float* HOST_logits,
                      char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
