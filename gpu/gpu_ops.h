#pragma once
/* GPU compute ops for LLM inference (RK3588 Mali-G610 via Vulkan).
 *
 * gpu_weight_t encapsulates a pre-dequantised weight matrix resident
 * in GPU-accessible memory, plus the Vulkan pipeline needed to run it.
 * Dequantisation happens once at model-load time; inference is pure GPU SGEMM. */

#include "vk_ctx.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_weight gpu_weight_t;

/* Dequantise GGUF weight W[N×K] to fp32 and upload to GPU memory.
 * wtype is a wl_type_t value (int to avoid header pull-in).
 * Returns NULL if GPU unavailable or allocation fails. */
gpu_weight_t *gpu_weight_load(gpu_ctx_t *ctx,
                               const void *W, int wtype,
                               int K, int N);
void          gpu_weight_free(gpu_ctx_t *ctx, gpu_weight_t *w);

/* y[N] += x[K] @ W[N×K]  — dispatches GPU SGEMM.
 * Caller must zero y[0..N-1] before the first accumulation.
 * Returns 0 on success, -1 on error (caller should fall back to CPU). */
int gpu_matmul(gpu_ctx_t *ctx, gpu_weight_t *w,
               float *y, const float *x, int K, int N);

#ifdef __cplusplus
}
#endif
