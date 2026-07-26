/* cpu_matmul_bench.c — fair CPU baseline for lfm2-rk3588 NPU-prefill comparison.
 *
 * Times ggml's optimized CPU mul_mat (the exact kernel llama.cpp uses: Q4_0
 * weights, dotprod) at the LFM2.5-8B-A1B expert-FFN shapes, 4 threads on A76.
 * Apples-to-apples vs the NPU warm dispatch in tools/prefill_expert_microbench.c.
 *
 * mul_mat(a[K,N] Q4_0, b[K,M] F32) -> c[N,M] F32 : N outputs for M tokens.
 * Values are irrelevant (timing only; ggml correctness is already trusted).
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_ll(const void *a, const void *b) {
    long long x = *(const long long*)a, y = *(const long long*)b;
    return (x>y)-(x<y);
}

typedef struct { const char *name; int K, N; } shape_t;

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 32;
    int nth = argc > 2 ? atoi(argv[2]) : 4;
    int iters = 50;
    shape_t shapes[] = {
        { "gate (K2048 N1792)", 2048, 1792 },
        { "up   (K2048 N1792)", 2048, 1792 },
        { "down (K1792 N2048)", 1792, 2048 },
        { "dense-ref (2048^2)", 2048, 2048 },
    };
    int ns = sizeof(shapes)/sizeof(*shapes);

    printf("ggml CPU mul_mat  M=%d  threads=%d  Q4_0 weights  %d iters\n", M, nth, iters);
    printf("\n  %-20s  %12s  %12s\n", "shape", "wall us/call", "us/token");
    printf("  --------------------  ------------  ------------\n");

    double sum_per_tok = 0;
    for (int si = 0; si < ns; si++) {
        int K = shapes[si].K, N = shapes[si].N;
        size_t mem = (size_t)64*1024*1024;
        struct ggml_init_params ip = { mem, NULL, false };
        struct ggml_context *ctx = ggml_init(ip);

        struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N); /* weights */
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M); /* activations */
        memset(w->data, 0, ggml_nbytes(w));
        float *xd = (float*)x->data;
        for (int i = 0; i < K*M; i++) xd[i] = (i%7)*0.1f - 0.3f;

        struct ggml_tensor *c = ggml_mul_mat(ctx, w, x);  /* [N, M] */
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);

        /* warm-up */
        ggml_graph_compute_with_ctx(ctx, gf, nth);

        long long s[64];
        for (int it = 0; it < iters; it++) {
            long long t0 = ggml_time_us();
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            s[it] = ggml_time_us() - t0;
        }
        qsort(s, iters, sizeof(long long), cmp_ll);
        double per_call = (double)s[iters/2];
        double per_tok = per_call / M;
        printf("  %-20s  %12.1f  %12.2f\n", shapes[si].name, per_call, per_tok);
        if (si < 3) sum_per_tok += per_tok;

        ggml_free(ctx);
    }
    printf("\n  Per-expert (gate+up+down) CPU per-token: %.2f us  (M=%d, %d threads)\n",
           sum_per_tok, M, nth);
    return 0;
}
