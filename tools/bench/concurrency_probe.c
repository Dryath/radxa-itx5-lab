/* concurrency_probe.c — CPU + NPU aggregate memory-bandwidth probe (RK3588).
 *
 * THE decisive experiment for this project: the LPDDR5 single-master ceiling is
 * ~25 GB/s (measured: one A76 core saturates it; the NPU hits the same number).
 * MoE decode is bandwidth-bound and already at that wall on CPU. The only way to
 * go faster (besides moving fewer bytes) is if the DDR controller's AGGREGATE
 * across separate masters exceeds 25 GB/s — i.e. CPU and NPU streaming at once
 * sum to more than either alone. Prior work only ever tested NPU-vs-NPU (same
 * device, 1.05x). This tests CPU(A76 NEON) + NPU(CNA) concurrently.
 *
 * Method: drive a fixed NPU fp16 matmul (weight streamed from DDR each dispatch)
 * in a tight loop = NPU memory traffic. Run STREAM-triad on N A76 threads =
 * CPU memory traffic. Measure each alone, then both concurrently.
 *
 * Build (from a checkout providing an NPU dispatch backend):
 *   gcc -O3 -mcpu=cortex-a76 -ffast-math -std=gnu11 -D__user= -D__kernel= \
 *       -D__force= -D__iomem= -I<backend>/npu -I/tmp/drm_compat \
 *       <backend>/npu/npu_drm.c <backend>/npu/npu_matmul_kernel.c \
 *       concurrency_probe.c -o /tmp/concurrency_probe -lm -lpthread
 * Run: taskset -c 0-7 /tmp/concurrency_probe [K] [N] [seconds] [cpu_threads]
 */
#include "npu_drm.h"
#include "npu_matmul_kernel.h"
#include "include/rknpu-ioctl.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

/* ---------- NPU side ---------- */
static int g_fd;
static npu_mem_t *g_tasks;
static size_t g_npu_bytes_per_dispatch;

static double npu_stream(double dur_s, long *iters_out){
    long iters = 0; double t0 = now_s(), t;
    do {
        int rc = npu_submit(g_fd, g_tasks->obj_addr, 1, 0x1,
                            RKNPU_JOB_PC | RKNPU_JOB_BLOCK, 2000, 0);
        if (rc < 0 && rc != -1){ fprintf(stderr,"npu_submit rc=%d\n",rc); break; }
        iters++; t = now_s();
    } while (t - t0 < dur_s);
    double el = now_s() - t0;
    *iters_out = iters;
    return (double)iters * g_npu_bytes_per_dispatch / el / 1e9;
}

/* ---------- CPU side ---------- */
#define CPU_N (16L*1024*1024)   /* 16M doubles per array = 128MB/thread, >> LLC */
typedef struct { int core; volatile int *run; double gbps; long passes; } cpu_arg_t;

static void *cpu_thread(void *p){
    cpu_arg_t *a = (cpu_arg_t*)p;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(a->core,&set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    double *x = malloc(CPU_N*8), *y = malloc(CPU_N*8), *z = malloc(CPU_N*8);
    for (long i=0;i<CPU_N;i++){ x[i]=1.0; y[i]=2.0; z[i]=3.0; }
    const double s = 3.0; long passes = 0; double t0 = now_s();
    while (*a->run){
        for (long i=0;i<CPU_N;i++) x[i] = y[i] + s*z[i];   /* triad: 3 arrays touched */
        passes++;
        if (x[passes & (CPU_N-1)] < 0) printf("x");        /* defeat DCE */
    }
    double el = now_s() - t0;
    a->passes = passes;
    a->gbps = (double)passes * 3.0 * CPU_N * 8 / el / 1e9;
    free(x); free(y); free(z);
    return NULL;
}

static double cpu_stream(int nthreads, int *cores, double dur_s){
    volatile int run = 1;
    pthread_t th[16]; cpu_arg_t args[16];
    for (int i=0;i<nthreads;i++){ args[i]=(cpu_arg_t){cores[i],&run,0,0}; pthread_create(&th[i],NULL,cpu_thread,&args[i]); }
    struct timespec ts={(time_t)dur_s,(long)((dur_s-(time_t)dur_s)*1e9)}; nanosleep(&ts,NULL);
    run = 0;
    double agg=0; for (int i=0;i<nthreads;i++){ pthread_join(th[i],NULL); agg+=args[i].gbps; }
    return agg;
}

/* CPU threads that run until an external flag clears (for concurrent phase) */
static volatile int g_cpu_run;
static int g_cpu_nthreads; static int g_cpu_cores[16];
static pthread_t g_cpu_th[16]; static cpu_arg_t g_cpu_args[16];
static void cpu_start(void){
    g_cpu_run=1;
    for (int i=0;i<g_cpu_nthreads;i++){ g_cpu_args[i]=(cpu_arg_t){g_cpu_cores[i],&g_cpu_run,0,0}; pthread_create(&g_cpu_th[i],NULL,cpu_thread,&g_cpu_args[i]); }
}
static double cpu_stop(void){
    g_cpu_run=0; double agg=0;
    for (int i=0;i<g_cpu_nthreads;i++){ pthread_join(g_cpu_th[i],NULL); agg+=g_cpu_args[i].gbps; }
    return agg;
}

int main(int argc, char **argv){
    int K = argc>1?atoi(argv[1]):4096;
    int N = argc>2?atoi(argv[2]):4096;
    double dur = argc>3?atof(argv[3]):3.0;
    int M = 4;
    g_cpu_nthreads = argc>4?atoi(argv[4]):4;
    int a76[4]={4,5,6,7}; for(int i=0;i<g_cpu_nthreads && i<4;i++) g_cpu_cores[i]=a76[i];
    /* if >4 threads requested, fill with A55 0-3 */
    for(int i=4;i<g_cpu_nthreads;i++) g_cpu_cores[i]=i-4;

    /* ----- NPU setup (mirrors matmul_probe) ----- */
    g_fd = npu_acquire_fd(); if (g_fd<0){ fprintf(stderr,"npu_acquire_fd failed\n"); return 1; }
    size_t input_sz=((size_t)M*K*2u)+4096, weight_sz=((size_t)K*N*2u)+4096, output_sz=((size_t)M*N*4u)+4096;
    npu_mem_t *tasks=npu_mem_alloc(g_fd,4096,RKNPU_MEM_KERNEL_MAPPING|RKNPU_MEM_NON_CACHEABLE,0);
    npu_mem_t *regcmd=npu_mem_alloc(g_fd,4096,RKNPU_MEM_NON_CACHEABLE,0);
    npu_mem_t *input=npu_mem_alloc(g_fd,input_sz,RKNPU_MEM_NON_CACHEABLE,0);
    npu_mem_t *weight=npu_mem_alloc(g_fd,weight_sz,RKNPU_MEM_NON_CACHEABLE,0);
    npu_mem_t *output=npu_mem_alloc(g_fd,output_sz,RKNPU_MEM_NON_CACHEABLE,0);
    if(!tasks||!regcmd||!input||!weight||!output){ fprintf(stderr,"GEM alloc failed\n"); return 1; }
    memset(weight->virt,0,weight_sz); memset(input->virt,0,input_sz); memset(output->virt,0,output_sz);
    uint64_t *ops=(uint64_t*)regcmd->virt;
    if (gen_matmul_fp16_ops(ops,M,K,N,(uint32_t)(input->dma_addr&0xFFFFFFFFu),
            (uint32_t)(weight->dma_addr&0xFFFFFFFFu),(uint32_t)(output->dma_addr&0xFFFFFFFFu),1) < 0){
        fprintf(stderr,"gen ops failed\n"); return 1; }
    struct rknpu_task *task=(struct rknpu_task*)tasks->virt; memset(task,0,sizeof(*task));
    task->enable_mask=0xd; task->int_mask=0x300; task->int_clear=0x1ffff;
    task->regcfg_amount=NPU_REGCFG_AMOUNT; task->regcmd_addr=regcmd->dma_addr;
    g_tasks=tasks;
    g_npu_bytes_per_dispatch = (size_t)K*N*2 + (size_t)M*K*2 + (size_t)M*N*4; /* weight rd + in rd + out wr */

    printf("# concurrency_probe K=%d N=%d M=%d dur=%.1fs cpu_threads=%d (cores", K,N,M,dur,g_cpu_nthreads);
    for(int i=0;i<g_cpu_nthreads;i++) printf(" %d",g_cpu_cores[i]); printf(")\n");
    printf("# NPU bytes/dispatch = %.1f MB (weight %.1f MB)\n",
           g_npu_bytes_per_dispatch/1e6, (double)K*N*2/1e6);

    /* warmup NPU */
    long it; npu_stream(0.3,&it);

    /* ----- Phase A: NPU solo ----- */
    double npu_solo = npu_stream(dur,&it);
    printf("A  NPU solo            : %6.2f GB/s   (%ld dispatches)\n", npu_solo, it);

    /* ----- Phase B: CPU solo ----- */
    double cpu_solo = cpu_stream(g_cpu_nthreads,g_cpu_cores,dur);
    printf("B  CPU solo (%d thr)    : %6.2f GB/s\n", g_cpu_nthreads, cpu_solo);

    /* ----- Phase C: concurrent ----- */
    cpu_start();
    double npu_c = npu_stream(dur,&it);
    double cpu_c = cpu_stop();
    printf("C  CONCURRENT          : CPU %6.2f + NPU %6.2f = %6.2f GB/s aggregate\n",
           cpu_c, npu_c, cpu_c+npu_c);
    /* Honest metrics:
     *  additivity = aggregate / (cpu_solo + npu_solo); ~1.0 = independent masters,
     *               <<1.0 = contention for a shared pool.
     *  headroom   = aggregate / best single-master solo; >1.1 = multi-master truly wins. */
    double additivity = (cpu_c+npu_c) / (cpu_solo + npu_solo);
    double best_solo  = cpu_solo > npu_solo ? cpu_solo : npu_solo;
    double headroom   = (cpu_c+npu_c) / best_solo;
    printf("\nadditivity = %.2f (1.0=independent, <1=contention)   headroom over best solo = %.2fx\n",
           additivity, headroom);
    printf("VERDICT: %s\n",
           headroom > 1.10 ? "MULTI-MASTER ADDS BANDWIDTH (heterogeneous decode viable)"
                           : "SHARED CEILING — CPU+NPU contend for one ~pool; NPU cannot add decode bandwidth");
    return 0;
}
