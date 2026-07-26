// STREAM-style memory bandwidth probe for RK3588. Build:
//   gcc -O3 -mcpu=cortex-a76 -fopenmp tools/membw.c -o /tmp/membw
// Run pinned: taskset -c 4-7 OMP_NUM_THREADS=4 /tmp/membw
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N (64L*1024*1024)   // 64M doubles = 512 MB per array, 1.5 GB total (>> caches)
static double a[N], b[N], c[N];

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(void){
  #pragma omp parallel for
  for(long i=0;i<N;i++){ a[i]=1.0; b[i]=2.0; c[i]=0.0; }
  const double scalar=3.0;
  double best_copy=0,best_triad=0,best_scale=0;
  for(int rep=0; rep<5; rep++){
    double t;
    t=now();
    #pragma omp parallel for
    for(long i=0;i<N;i++) c[i]=a[i];
    t=now()-t; double bw=2.0*N*sizeof(double)/t/1e9; if(bw>best_copy)best_copy=bw;

    t=now();
    #pragma omp parallel for
    for(long i=0;i<N;i++) b[i]=scalar*c[i];
    t=now()-t; bw=2.0*N*sizeof(double)/t/1e9; if(bw>best_scale)best_scale=bw;

    t=now();
    #pragma omp parallel for
    for(long i=0;i<N;i++) a[i]=b[i]+scalar*c[i];
    t=now()-t; bw=3.0*N*sizeof(double)/t/1e9; if(bw>best_triad)best_triad=bw;
  }
  printf("threads=%d  Copy=%.1f GB/s  Scale=%.1f GB/s  Triad=%.1f GB/s\n",
         omp_get_max_threads(), best_copy, best_scale, best_triad);
  return 0;
}
