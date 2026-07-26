/*
 * npu_gemm_test.c — minimal CNA GEMM via raw DRM ioctls
 *
 * Tests a 16×16 fp16 matrix multiply using the tracer.gguf weight tensor.
 * Builds the regcmd from the known-working 0.5B template, patching only
 * the IOVA and dimension registers.  Compares output to a CPU reference.
 *
 * Build:
 *   gcc -O2 -o npu_gemm_test \
 *       npu_gemm_test.c \
 *       -I. -lm
 *
 * Usage:
 *   ./npu_gemm_test [tracer.gguf path]
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "rknpu_ioctl.h"

/* ── constants ──────────────────────────────────────────────────────────── */

#define DRM_DEV        "/dev/dri/renderD129"
#define DIM            16            /* 16×16 weight matrix */
#define SEQ_LEN        1             /* single token */
#define ALIGN(x, a)    (((x) + (a) - 1) & ~((a) - 1))

/* GEM buffer flags — drop IOMMU_LIMIT_IOVA_ALIGNMENT so IOVA lands in low range. */
#define GEM_FLAGS_TASK   (RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE | \
                          RKNPU_MEM_KERNEL_MAPPING)
#define GEM_FLAGS_DATA   (RKNPU_MEM_NON_CONTIGUOUS | RKNPU_MEM_CACHEABLE)

/* ── fp16 helpers ───────────────────────────────────────────────────────── */

static float fp16_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1;
    uint32_t e = (h >> 10) & 0x1f;
    uint32_t m = h & 0x3ff;
    if (e == 0)   { float f = ldexpf((float)m, -24); return s ? -f : f; }
    if (e == 31)  { return m ? 0.0f/0.0f : (s ? -1.0f/0.0f : 1.0f/0.0f); }
    uint32_t bits = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

static uint16_t float_to_fp16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t s = bits >> 31;
    int      e = ((bits >> 23) & 0xff) - 127;
    uint32_t m = bits & 0x7fffff;
    if (e >= 16)  return (s << 15) | 0x7c00;
    if (e < -14)  return (s << 15);
    return (uint16_t)((s << 15) | ((e + 15) << 10) | (m >> 13));
}

/* ── GGUF reader — minimal, single F16 tensor ───────────────────────────── */

static int read_gguf_f16_tensor(const char *path, uint16_t *out, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    /* Check magic */
    uint32_t magic; fread(&magic, 4, 1, f);
    if (magic != 0x46554747u) {  /* 'GGUF' */
        fprintf(stderr, "Bad GGUF magic 0x%08x\n", magic);
        fclose(f); return -1;
    }
    uint32_t version; fread(&version, 4, 1, f);
    uint64_t n_tensors, n_kv;
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);

    /* Skip all key-value pairs — read type+key+value for each */
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; fread(&klen, 8, 1, f);
        fseek(f, klen, SEEK_CUR);   /* skip key string */
        uint32_t vtype; fread(&vtype, 4, 1, f);
        switch (vtype) {
        case 0: case 1: /* uint8/int8 */
            fseek(f, 1, SEEK_CUR); break;
        case 2: case 3: /* uint16/int16 */
            fseek(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6: /* uint32/int32/float32 */
            fseek(f, 4, SEEK_CUR); break;
        case 7: /* bool — 1 byte in gguf v3 */
            fseek(f, 1, SEEK_CUR); break;
        case 8: { /* string */
            uint64_t slen; fread(&slen, 8, 1, f); fseek(f, slen, SEEK_CUR); break; }
        case 9: { /* array */
            uint32_t atype; fread(&atype, 4, 1, f);
            uint64_t alen; fread(&alen, 8, 1, f);
            size_t elem = (atype <= 3) ? 2 : (atype <= 7) ? 4 : 8;
            fseek(f, alen * elem, SEEK_CUR); break; }
        case 10: case 11: /* uint64/int64 */
            fseek(f, 8, SEEK_CUR); break;
        case 12:           /* float64 */
            fseek(f, 8, SEEK_CUR); break;
        }
    }

    /* Read first tensor info */
    uint64_t nlen; fread(&nlen, 8, 1, f);
    fseek(f, nlen, SEEK_CUR);       /* skip name */
    uint32_t ndim; fread(&ndim, 4, 1, f);
    uint64_t dims[4] = {0}; fread(dims, 8, ndim, f);
    uint32_t dtype; fread(&dtype, 4, 1, f);
    uint64_t offset; fread(&offset, 8, 1, f);

    if (dtype != 1) { /* 1 = F16 */
        fprintf(stderr, "Expected F16 tensor (type 1), got %u\n", dtype);
        fclose(f); return -1;
    }

    /* Data section starts at alignment boundary after tensor info block */
    long align = 32;
    long pos = ftell(f);
    long data_start = ALIGN(pos, align);
    fseek(f, data_start + offset, SEEK_SET);
    fread(out, 2, n, f);
    fclose(f);
    return 0;
}

/* ── GEM helpers ────────────────────────────────────────────────────────── */

typedef struct { uint64_t obj; uint64_t dma; uint64_t mmap_off; void *uva; size_t size; } Gem;

static int gem_create(int fd, size_t size, uint32_t flags, uint32_t core_mask, Gem *g) {
    struct rknpu_mem_create m = {};
    m.size      = size;
    m.flags     = flags;
    m.core_mask = core_mask;  /* must match SUBMIT core_mask for IOMMU domain mapping */
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &m)) { perror("MEM_CREATE"); return -1; }
    g->obj  = m.obj_addr;
    g->dma  = m.dma_addr & 0xFFFFFFFFULL;
    g->size = size;
    struct rknpu_mem_map mm = { .handle = m.handle };
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mm)) { perror("MEM_MAP"); return -1; }
    g->mmap_off = mm.offset;
    g->uva = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, mm.offset);
    if (g->uva == MAP_FAILED) { perror("mmap"); return -1; }
    return 0;
}

static int gem_sync(int fd, Gem *g, uint32_t flags) {
    struct rknpu_mem_sync s = { .flags = flags, .obj_addr = g->obj, .size = g->size };
    if (ioctl(fd, DRM_IOCTL_RKNPU_MEM_SYNC, &s)) { perror("MEM_SYNC"); return -1; }
    return 0;
}

/* ── regcmd builder ─────────────────────────────────────────────────────── */

/*
 * Template from the 0.5B RKLLM trace (112 pairs, q_proj_L0 = 896×896 matmul).
 * We patch:
 *   - cna_feature_data_addr  (0x1070): input IOVA
 *   - cna_dcomp_addr0        (0x1110): weight IOVA
 *   - cna_dcomp_ctrl         (0x1100): set to 0 to try raw fp16 mode
 *   - cna_data_size1 0x1024  : datain_channel = DIM-1 (bits[15:0])
 *   - cna_weight_size2 0x1038: weight_kernels = DIM-1 (bits[13:0])
 *   - cna_data_size0  0x1020 : datain_height  = SEQ_LEN (bits[10:0])
 *   - dpu_dst_base_addr 0x4020: output IOVA
 *
 * All other registers kept identical to the working 0.5B dump.
 * Upper 16 bits of dimension register values are kept as-is (0x0201 tag).
 */

/* Verbatim fp16 CNA regcmd from tracer_matmul.rknn (16×16 F16 matmul).
 * Captured via npu_hook REGCMD_DUMP unit=CNA.
 * Only the 3 IO-address entries are patched at runtime (0x1070, 0x1110, 0x4020). */
#define RC_PAIRS 112
static const uint32_t rc_template[RC_PAIRS][2] = {
    {0x00b11040, 0x02010000}, {0x00001104, 0x02010000}, {0x00001100, 0x02010000},
    {0x0120100c, 0x02010000}, {0x000e4004, 0x10010000}, {0x0120100c, 0x02010000},
    {0x00201010, 0x02010000}, {0x00091014, 0x02010000},
    {0x00011020, 0x02010001}, {0x00101024, 0x0201000f},
    {0x00011028, 0x02010000}, {0x0001102c, 0x02010000},
    {0x02001030, 0x02010000}, {0x00201034, 0x02010000},
    {0x00101038, 0x02010101},
    {0x00b11040, 0x02010000}, {0x00011044, 0x02010000}, {0x000b104c, 0x02010000},
    {0x00001050, 0x02010001}, {0x00001054, 0x02010001},
    {0x00001058, 0x02010001}, {0x0000105c, 0x02010001},
    {0x00001060, 0x02010000}, {0x00001064, 0x02010000}, {0x00001068, 0x02010000},
    {0xc0801070, 0x0201ffff}, /* feat IOVA — patched */
    {0x00001074, 0x02010000}, {0x000f1078, 0x0201000f},
    {0x0004107c, 0x02010000}, {0xfffd1080, 0x02010fff},
    {0x00011084, 0x02010001}, {0x00101088, 0x02010000},
    {0x00001100, 0x02010000}, {0x00001104, 0x02010000},
    {0xd0001110, 0x0201ffff}, /* wt IOVA — patched */
    {0x00001140, 0x02010000}, {0x00001144, 0x02010000},
    {0x00001148, 0x02010000}, {0x0000114c, 0x02010000},
    {0x00001150, 0x02010000}, {0x00001154, 0x02010000},
    {0x00001158, 0x02010000}, {0x0000115c, 0x02010000},
    {0x00001160, 0x02010000}, {0x00001164, 0x02010000},
    {0x00001168, 0x02010000}, {0x0000116c, 0x02010000},
    {0x00001170, 0x02010000}, {0x00001174, 0x02010000},
    {0x00001178, 0x02010000}, {0x0000117c, 0x02010000},
    {0x00001180, 0x02010000}, {0x00001184, 0x02010000},
    {0x02003010, 0x08010000}, {0x00003014, 0x08010000},
    {0x000f3018, 0x08010000}, {0x0000301c, 0x08010000},
    {0x00003030, 0x08010000},
    {0x01e4400c, 0x10010000}, {0x00024010, 0x10014800},
    {0x00004014, 0x10010000},
    {0xc0004020, 0x1001ffff}, /* output IOVA — patched */
    {0x00104024, 0x10010000}, {0x00004030, 0x10010000},
    {0x00004034, 0x10010000}, {0x00004038, 0x10010000},
    {0x000f403c, 0x1001000f}, {0x00534040, 0x10010000},
    {0x00004044, 0x10010000}, {0x00004048, 0x10010000},
    {0x0000404c, 0x10010000}, {0x01264050, 0x10010000},
    {0x00004054, 0x10010000}, {0x000f4058, 0x10010000},
    {0x0000405c, 0x10010000}, {0x00534060, 0x10010000},
    {0x00004064, 0x10010000}, {0x00004068, 0x10010000},
    {0x0000406c, 0x10010000}, {0x03834070, 0x10010000},
    {0x00004074, 0x10010000}, {0x00014078, 0x10010000},
    {0x0000407c, 0x10010000}, {0x00004080, 0x10010000},
    {0x00014084, 0x10010001}, {0x00004088, 0x10010000},
    {0x00004090, 0x10010000}, {0x00004094, 0x10010000},
    {0x00004098, 0x10010000}, {0x0000409c, 0x10010000},
    {0x000040a0, 0x10010000}, {0x000040a4, 0x10010000},
    {0x000040a8, 0x10010000}, {0x000040ac, 0x10010000},
    {0x002040c0, 0x10010000}, {0x000040c4, 0x10010000},
    {0x00004100, 0x10010000}, {0x00004104, 0x10010000},
    {0x00004108, 0x10010000}, {0x0000410c, 0x10010000},
    {0x00004110, 0x10010000}, {0x00004114, 0x10010000},
    {0x00004118, 0x10010000}, {0x0000411c, 0x10010000},
    {0x00004120, 0x10010000}, {0x00004124, 0x10010000},
    {0x00004128, 0x10010000}, {0x0000412c, 0x10010000},
    /* PC data (RKNPU_PC_DATA_EXTRA_AMOUNT=4, not counted in regcfg_amount).
     * First entry: 0 = terminate (not a jump to next RKNN program page). */
    {0x00000000, 0x00000000}, {0x00240014, 0x01010000},
    {0x00000000, 0x00410000}, {0x000d0008, 0x00810000},
};

static void build_regcmd(uint32_t *rc, uint32_t feat_iova, uint32_t wt_iova,
                         uint32_t out_iova) {
    memcpy(rc, rc_template, sizeof(rc_template));
    uint32_t (*p)[2] = (uint32_t(*)[2])rc;

    for (int i = 0; i < RC_PAIRS; i++) {
        uint32_t reg = p[i][0] & 0xFFFF;
        switch (reg) {
        case 0x1070: /* cna_feature_data_addr = input IOVA */
            p[i][1] = feat_iova;
            break;
        case 0x1110: /* cna_dcomp_addr0 = weight IOVA */
            p[i][1] = wt_iova;
            break;
        case 0x4020: /* dpu_dst_base_addr = output IOVA */
            p[i][1] = out_iova;
            break;
        }
    }
}

/* ── task buffer builder ────────────────────────────────────────────────── */

static void build_task(struct rknpu_task *t, uint64_t regcmd_iova, uint32_t cfg) {
    memset(t, 0, sizeof(*t));
    t->flags        = 0;
    t->op_idx       = 2;   /* real RKNN CNA task uses op_idx=2 */
    t->enable_mask  = 0x0d;    /* CNA + PC */
    t->int_mask     = 0x300;
    t->int_clear    = 0x1ffff;
    /* regcfg_amount excludes the trailing RKNPU_PC_DATA_EXTRA_AMOUNT (4) control entries */
    t->regcfg_amount = cfg - RKNPU_PC_DATA_EXTRA_AMOUNT;
    t->regcmd_addr  = regcmd_iova;
}

/* ── CPU reference GEMM ─────────────────────────────────────────────────── */

/* out[i] = sum_k(in[k] * w[i*DIM + k])   (row-major weight) */
static void cpu_gemm_f16(uint16_t *in, uint16_t *w, float *out) {
    for (int i = 0; i < DIM; i++) {
        float acc = 0.0f;
        for (int k = 0; k < DIM; k++)
            acc += fp16_to_float(in[k]) * fp16_to_float(w[i * DIM + k]);
        out[i] = acc;
    }
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1]
                          : "weights.gguf";

    /* 1. Load weight tensor from GGUF */
    uint16_t w[DIM * DIM];
    if (read_gguf_f16_tensor(gguf_path, w, DIM * DIM) < 0) return 1;

    printf("Loaded %dx%d F16 weight from %s\n", DIM, DIM, gguf_path);
    printf("  w[0,0]=%.4f  w[0,15]=%.4f  w[15,0]=%.4f  w[15,15]=%.4f\n",
           fp16_to_float(w[0]), fp16_to_float(w[15]),
           fp16_to_float(w[240]), fp16_to_float(w[255]));

    /* 2. Build known input (identity: in[i] = 1.0 for all i) */
    uint16_t in_f16[DIM];
    for (int i = 0; i < DIM; i++) in_f16[i] = float_to_fp16(1.0f);

    /* 3. CPU reference */
    float ref[DIM];
    cpu_gemm_f16(in_f16, w, ref);
    printf("CPU reference: ");
    for (int i = 0; i < DIM; i++) printf("%.3f ", ref[i]);
    printf("\n");

    /* 4. Open NPU device */
    int fd = open(DRM_DEV, O_RDWR);
    if (fd < 0) { perror("open " DRM_DEV); return 1; }
    printf("Opened %s fd=%d\n", DRM_DEV, fd);

    /* DRM version check */
    struct drm_version ver = {};
    ioctl(fd, DRM_IOCTL_VERSION, &ver);
    printf("DRM driver: hw_ver=0x%x\n", ver.version_major);

    /* Replicate real driver's ACTION sequence */
    struct rknpu_action act = {};
    act.flags = RKNPU_GET_HW_VERSION;
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("HW_VERSION=0x%x\n", act.value);

    act.flags = RKNPU_GET_DRV_VERSION; act.value = 0;
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("DRV_VERSION=0x%x\n", act.value);

    act.flags = RKNPU_POWER_ON; act.value = 0;
    int pret = ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("POWER_ON ret=%d (0=success, already-on=ok)\n", pret);

    act.flags = RKNPU_GET_IOMMU_EN; act.value = 0;
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("IOMMU_EN=%u\n", act.value);

    act.flags = RKNPU_ACT_RESET; act.value = 0;
    ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
    printf("RESET done\n");

    /* 5. Allocate GEM buffers */
    size_t task_sz   = ALIGN(3 * sizeof(struct rknpu_task), 4096); /* 3 tasks (3 cores) */
    size_t regcmd_sz = ALIGN(RC_PAIRS * 8, 4096);
    /* Large data allocation to get a low IOVA from the IOMMU allocator.
     * CNA DMA appears limited to addresses < ~512MB; tiny allocations land near 4GB. */
    size_t io_sz     = 64 * 1024 * 1024;  /* 64MB → low IOVA */

    Gem task_gem, regcmd_gem, io_gem;
    if (gem_create(fd, task_sz,   GEM_FLAGS_TASK, 0x1, &task_gem))   return 1;
    if (gem_create(fd, regcmd_sz, GEM_FLAGS_DATA, 0x1, &regcmd_gem)) return 1;
    if (gem_create(fd, io_sz,     GEM_FLAGS_DATA, 0x1, &io_gem))     return 1;

    /* Use sections within io_gem for weight, input, output to share low IOVA */
    uint32_t wt_iova  = (uint32_t)io_gem.dma + 0;
    uint32_t in_iova  = (uint32_t)io_gem.dma + 0x1000;  /* 4KB after weight */
    uint32_t out_iova = (uint32_t)io_gem.dma + 0x2000;  /* 8KB after weight */
    char *wt_uva  = (char *)io_gem.uva + 0;
    char *in_uva  = (char *)io_gem.uva + 0x1000;
    char *out_uva = (char *)io_gem.uva + 0x2000;

    printf("GEM allocations:\n");
    printf("  task   dma=0x%08llx  uva=%p\n", (unsigned long long)task_gem.dma,   task_gem.uva);
    printf("  regcmd dma=0x%08llx  uva=%p\n", (unsigned long long)regcmd_gem.dma, regcmd_gem.uva);
    printf("  io(64M) dma=0x%08llx  uva=%p\n", (unsigned long long)io_gem.dma,    io_gem.uva);
    printf("  weight  dma=0x%08x\n", wt_iova);
    printf("  input   dma=0x%08x\n", in_iova);
    printf("  output  dma=0x%08x\n", out_iova);

    /* 6. Fill weight and input buffers */
    memcpy(wt_uva,  w,       DIM * DIM * 2);
    memcpy(in_uva,  in_f16,  DIM * 2);
    memset(out_uva, 0,       4096);

    /* 7. Build regcmds (core 0x1) */
    build_regcmd((uint32_t *)regcmd_gem.uva, in_iova, wt_iova, out_iova);

    printf("Built regcmd: feat=0x%08x  wt=0x%08x  out=0x%08x\n",
           in_iova, wt_iova, out_iova);

    /* 8. Build task structs for 3 cores (same regcmd, each core fires) */
    struct rknpu_task *tasks = (struct rknpu_task *)task_gem.uva;
    for (int c = 0; c < 3; c++)
        build_task(&tasks[c], regcmd_gem.dma, RC_PAIRS);

    printf("Task[0]: flags=%u op_idx=%u ena=0x%02x int_mask=0x%x int_clear=0x%x"
           " regcfg_amount=%u regcfg_offset=%u regcmd_addr=0x%llx\n",
           tasks[0].flags, tasks[0].op_idx, tasks[0].enable_mask,
           tasks[0].int_mask, tasks[0].int_clear,
           tasks[0].regcfg_amount, tasks[0].regcfg_offset,
           (unsigned long long)tasks[0].regcmd_addr);

    /* 9. MEM_SYNC all to device */
    gem_sync(fd, &task_gem,   RKNPU_MEM_SYNC_TO_DEVICE);
    gem_sync(fd, &regcmd_gem, RKNPU_MEM_SYNC_TO_DEVICE);
    gem_sync(fd, &io_gem,     RKNPU_MEM_SYNC_TO_DEVICE);

    /* 10. Submit for core 0x1 first */
    printf("Submitting to NPU core 0x1...\n");
    struct rknpu_submit sub = {};
    sub.flags           = 0x5;      /* same as real RKLLM driver */
    sub.timeout         = 6000;
    sub.task_obj_addr   = task_gem.obj;
    sub.task_base_addr  = 0;
    sub.core_mask       = 0x1;
    sub.task_number     = 3;        /* real driver: 3 tasks in buffer, all 3 subcores */
    sub.fence_fd        = -1;
    /* Real driver: all 3 subcores set (even for single physical core) */
    sub.subcore_task[0].task_start  = 0;
    sub.subcore_task[0].task_number = 1;
    sub.subcore_task[1].task_start  = 0;
    sub.subcore_task[1].task_number = 1;
    sub.subcore_task[2].task_start  = 0;
    sub.subcore_task[2].task_number = 1;

    int ret = ioctl(fd, DRM_IOCTL_RKNPU_SUBMIT, &sub);
    if (ret) {
        perror("SUBMIT");
        printf("  hw_elapse_time=%lld ns\n", (long long)sub.hw_elapse_time);
    } else {
        printf("  SUBMIT OK  hw_us=%lld\n", sub.hw_elapse_time / 1000);
    }

    /* 11. Sync output back and read */
    gem_sync(fd, &io_gem, RKNPU_MEM_SYNC_FROM_DEVICE);

    printf("NPU output (raw fp16 bytes → float):\n  ");
    uint16_t *out_f16 = (uint16_t *)out_uva;
    int match = 1;
    for (int i = 0; i < DIM; i++) {
        float npu_val = fp16_to_float(out_f16[i]);
        float diff    = fabsf(npu_val - ref[i]);
        if (diff > 0.01f * fabsf(ref[i]) + 0.01f) match = 0;
        printf("%.3f ", npu_val);
    }
    printf("\n");
    printf("%s (max_diff=%.4f vs CPU)\n",
           match ? "PASS — NPU output matches CPU reference!" :
                   "MISMATCH — output differs (check weight format / register config)",
           0.0f  /* simplified */);

    /* Print first 16 raw output bytes for diagnosis */
    printf("Raw output bytes: ");
    uint8_t *raw = (uint8_t *)out_uva;
    for (int i = 0; i < 32; i++) printf("%02x ", raw[i]);
    printf("\n");

    close(fd);
    return match ? 0 : 2;
}
