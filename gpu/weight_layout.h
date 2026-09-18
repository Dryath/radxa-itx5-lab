#pragma once
/* Weight reordering into NPU CNA fp16 layout.
 * Standalone — no ggml dependency.  Operates on raw GGUF tensor data. */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GGML type IDs (matches GGUF spec, kept here to avoid ggml.h dependency). */
typedef enum {
    WL_TYPE_F32  = 0,
    WL_TYPE_F16  = 1,
    WL_TYPE_Q4_0 = 2,
    WL_TYPE_Q4_1 = 3,
    WL_TYPE_Q5_0 = 6,
    WL_TYPE_Q5_1 = 7,
    WL_TYPE_Q8_0 = 8,
    WL_TYPE_Q2_K = 10,
    WL_TYPE_Q3_K = 11,
    WL_TYPE_Q4_K = 12,
    WL_TYPE_Q5_K = 13,
    WL_TYPE_Q6_K    = 14,
    WL_TYPE_Q8_K    = 15,
    WL_TYPE_IQ2_XXS = 16,
    WL_TYPE_IQ2_XS  = 17,
    WL_TYPE_IQ3_XXS = 18,
    WL_TYPE_IQ1_S   = 19,
    WL_TYPE_IQ4_NL  = 20,
    WL_TYPE_IQ3_S   = 21,
    WL_TYPE_IQ2_S   = 22,
    WL_TYPE_IQ4_XS  = 23,
    WL_TYPE_BF16    = 30,
    WL_TYPE_TQ1_0   = 34,
    WL_TYPE_TQ2_0   = 35,
} wl_type_t;

/* Bytes per row (K elements) for a given GGUF quantisation type.
 * Returns 0 for unknown types. */
size_t wl_row_bytes(wl_type_t type, int K);

/* Dequantise one row of raw GGUF tensor data to fp32.
 * src must be the start of the row (caller advances by wl_row_bytes per row).
 * K is the number of elements in the row.
 * Supported: F32, F16, BF16, Q4_0, Q8_0, Q4_K. */
void wl_dequant_row(const void *src, float *dst, wl_type_t type, int K);

/* Reorder a K×N weight matrix (row-major, row=output-kernel) into the NPU
 * CNA weight_fp16_idx layout, dequantising from any supported type.
 * dst must hold N*K uint16_t (fp16). Constraints: K%32==0, N%16==0. */
void wl_reorder(const void *src, uint16_t *dst, wl_type_t type, int K, int N);

/* Reorder an N-tile of rows starting at global row N_off.
 * dst holds N_tile*K uint16_t. */
void wl_reorder_n_tile(const void *src, uint16_t *dst,
                        wl_type_t type, int K, int N_total,
                        int N_off, int N_tile);

/* Reorder a K-tile (K_off..K_off+K_tile-1) across all N rows.
 * dst holds N*K_tile uint16_t. K_tile%32==0. */
void wl_reorder_k_tile(const void *src, uint16_t *dst,
                        wl_type_t type, int K_total, int N,
                        int K_off, int K_tile);

/* fp32 ↔ fp16 bit conversions (ARM NEON-accelerated when available). */
uint16_t wl_fp32_to_fp16(float f);
float    wl_fp16_to_fp32(uint16_t h);

#ifdef __cplusplus
}
#endif
