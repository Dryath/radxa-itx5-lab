#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>     /* getenv, atoi */
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "rknpu_ioctl.h"

/*
 * npu_hook.c — Full struct-decoding LD_PRELOAD hook for RK3588 NPU via DRM
 *
 * Per-SUBMIT compact line (always emitted):
 *   hw_us  core  tasks  ena  cfg  unit  task_base
 *   - ena  = rknpu_task.enable_mask  (which NPU units are active)
 *   - cfg  = rknpu_task.regcfg_amount  (number of register commands)
 *   - unit = sub-unit from TRM bits[15:12] of first register command address
 *
 * op_idx (rknpu_task field) is always 0 in practice — SDK does not populate it.
 * Use (enable_mask, regcfg_amount) tuple to distinguish operation types instead.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o npu_hook.so npu_hook.c \
 *       -I. -ldl -Wl,-soname,npu_hook.so
 */

/* ── config ──────────────────────────────────────────────────────────────── */

#ifndef DUMP_COUNT
#define DUMP_COUNT 32
#endif

/* ── fd tracking ─────────────────────────────────────────────────────────── */

#define MAX_TRACKED_FDS 64
static int _npu_fds[MAX_TRACKED_FDS];
static int _npu_fd_count = 0;

static void _track_fd(int fd) {
    if (fd < 0 || _npu_fd_count >= MAX_TRACKED_FDS) return;
    _npu_fds[_npu_fd_count++] = fd;
}

static int _is_npu_fd(int fd) {
    for (int i = 0; i < _npu_fd_count; i++)
        if (_npu_fds[i] == fd) return 1;
    return 0;
}

static int _path_is_npu(const char *path) {
    if (!path) return 0;
    return strstr(path, "/dev/dri/") != NULL ||
           strstr(path, "fdab0000")  != NULL ||
           strstr(path, "rknpu")     != NULL;
}

/* ── GEM handle tracking ─────────────────────────────────────────────────── */

#define MAX_GEM_HANDLES 64

typedef struct {
    uint64_t  obj_addr;    /* kernel VA from MEM_CREATE (used as task_obj_addr) */
    uint64_t  dma_addr;    /* NPU IOVA base (32-bit, sign-extended in the field) */
    uint64_t  mmap_offset; /* fake mmap offset from MEM_MAP (not exposed in /proc/maps) */
    uintptr_t uva;         /* resolved userspace VA */
    size_t    size;
} _GemHandle;

static _GemHandle _gems[MAX_GEM_HANDLES + 1]; /* [0] unused */

static int _gem_by_obj(uint64_t obj_addr) {
    if (!obj_addr) return 0;
    for (int i = 1; i <= MAX_GEM_HANDLES; i++)
        if (_gems[i].obj_addr == obj_addr) return i;
    return 0;
}

/* RKNPU IOVAs are 32-bit stored sign-extended in __u64. Normalize before range check. */
static int _gem_by_iova(uint64_t iova) {
    iova &= 0xFFFFFFFFULL;
    for (int i = 1; i <= MAX_GEM_HANDLES; i++) {
        if (!_gems[i].dma_addr) continue;
        uint64_t base = _gems[i].dma_addr & 0xFFFFFFFFULL;
        if (iova >= base && (iova - base) < _gems[i].size)
            return i;
    }
    return 0;
}

/* ── call counters ───────────────────────────────────────────────────────── */
static long _cnt_action  = 0;
static long _cnt_submit  = 0;
static long _cnt_sync    = 0;
static long _cnt_create  = 0;
static long _cnt_map     = 0;
static long _cnt_destroy = 0;
static long _cnt_other   = 0;

/* ── ACTION histogram ────────────────────────────────────────────────────── */
#define ACTION_MAX 26
static long _action_hist[ACTION_MAX];

static const char *_action_name(uint32_t flags) {
    switch (flags) {
    case RKNPU_GET_HW_VERSION:           return "GET_HW_VERSION";
    case RKNPU_GET_DRV_VERSION:          return "GET_DRV_VERSION";
    case RKNPU_GET_FREQ:                 return "GET_FREQ";
    case RKNPU_SET_FREQ:                 return "SET_FREQ";
    case RKNPU_GET_VOLT:                 return "GET_VOLT";
    case RKNPU_SET_VOLT:                 return "SET_VOLT";
    case RKNPU_ACT_RESET:               return "ACT_RESET";
    case RKNPU_GET_BW_PRIORITY:          return "GET_BW_PRIORITY";
    case RKNPU_SET_BW_PRIORITY:          return "SET_BW_PRIORITY";
    case RKNPU_GET_BW_EXPECT:            return "GET_BW_EXPECT";
    case RKNPU_SET_BW_EXPECT:            return "SET_BW_EXPECT";
    case RKNPU_GET_BW_TW:                return "GET_BW_TW";
    case RKNPU_SET_BW_TW:                return "SET_BW_TW";
    case RKNPU_ACT_CLR_TOTAL_RW_AMOUNT: return "CLR_TOTAL_RW";
    case RKNPU_GET_DT_WR_AMOUNT:         return "GET_DT_WR";
    case RKNPU_GET_DT_RD_AMOUNT:         return "GET_DT_RD";
    case RKNPU_GET_WT_RD_AMOUNT:         return "GET_WT_RD";
    case RKNPU_GET_TOTAL_RW_AMOUNT:      return "GET_TOTAL_RW";
    case RKNPU_GET_IOMMU_EN:             return "GET_IOMMU_EN";
    case RKNPU_SET_PROC_NICE:            return "SET_PROC_NICE";
    case RKNPU_POWER_ON:                 return "POWER_ON";
    case RKNPU_POWER_OFF:                return "POWER_OFF";
    case RKNPU_GET_TOTAL_SRAM_SIZE:      return "GET_TOTAL_SRAM";
    case RKNPU_GET_FREE_SRAM_SIZE:       return "GET_FREE_SRAM";
    case RKNPU_GET_IOMMU_DOMAIN_ID:      return "GET_IOMMU_DOMAIN";
    case RKNPU_SET_IOMMU_DOMAIN_ID:      return "SET_IOMMU_DOMAIN";
    default:                              return "UNKNOWN";
    }
}

/* ── NPU sub-unit from TRM address bits [15:12] ──────────────────────────── */
static const char *_subunit_name(uint32_t reg_offset) {
    switch ((reg_offset >> 12) & 0xf) {
    case 0x0: return "PC";
    case 0x1: return "CNA";
    case 0x3: return "CORE";
    case 0x4: return "DPU";
    case 0x5: return "RDMA";
    case 0x6: return "PPU";
    case 0x7: return "PRDMA";
    case 0x8: return "DDMA";
    case 0x9: return "SDMA";
    case 0xf: return "GLOBAL";
    default:  return "?";
    }
}

/* ── Op-type profile: keyed by (enable_mask, regcfg_amount), core=0x1 only ─ */
#define OP_HIST_MAX 64
typedef struct {
    uint32_t enable_mask;
    uint32_t regcfg_amount;
    long     count;
    long     hw_us_sum;
} _OpEntry;
static _OpEntry _op_hist[OP_HIST_MAX];
static int      _op_hist_count = 0;

static void _op_record(uint32_t ena, uint32_t cfg, long hw_us) {
    for (int i = 0; i < _op_hist_count; i++) {
        if (_op_hist[i].enable_mask == ena && _op_hist[i].regcfg_amount == cfg) {
            _op_hist[i].count++;
            _op_hist[i].hw_us_sum += hw_us;
            return;
        }
    }
    if (_op_hist_count >= OP_HIST_MAX) return;
    _op_hist[_op_hist_count].enable_mask   = ena;
    _op_hist[_op_hist_count].regcfg_amount = cfg;
    _op_hist[_op_hist_count].count         = 1;
    _op_hist[_op_hist_count].hw_us_sum     = hw_us;
    _op_hist_count++;
}

/* ── MEM_SYNC obj_addr tracker ───────────────────────────────────────────── */
#define SYNC_TRACKER_MAX 64
typedef struct {
    uint64_t obj_addr;
    uint64_t total_bytes_to_dev;
    uint64_t total_bytes_from_dev;
    long     calls_to_dev;
    long     calls_from_dev;
} _SyncEntry;

static _SyncEntry _sync_tracker[SYNC_TRACKER_MAX];
static int        _sync_tracker_count = 0;

static void _sync_track(uint64_t obj_addr, uint32_t flags, uint64_t size) {
    for (int i = 0; i < _sync_tracker_count; i++) {
        if (_sync_tracker[i].obj_addr == obj_addr) {
            if (flags & RKNPU_MEM_SYNC_TO_DEVICE) {
                _sync_tracker[i].calls_to_dev++;
                _sync_tracker[i].total_bytes_to_dev += size;
            } else {
                _sync_tracker[i].calls_from_dev++;
                _sync_tracker[i].total_bytes_from_dev += size;
            }
            return;
        }
    }
    if (_sync_tracker_count >= SYNC_TRACKER_MAX) return;
    _SyncEntry *e = &_sync_tracker[_sync_tracker_count++];
    e->obj_addr = obj_addr;
    if (flags & RKNPU_MEM_SYNC_TO_DEVICE) {
        e->calls_to_dev = 1; e->total_bytes_to_dev = size;
        e->calls_from_dev = 0; e->total_bytes_from_dev = 0;
    } else {
        e->calls_from_dev = 1; e->total_bytes_from_dev = size;
        e->calls_to_dev = 0; e->total_bytes_to_dev = 0;
    }
}

/* ── register command scanner ────────────────────────────────────────────────
 * Scan `n` {u32 packed_addr, u32 value} pairs for a target register offset.
 * bits[15:0] of packed_addr = NPU register offset (matches TRM names).
 * ─────────────────────────────────────────────────────────────────────────── */
static uint32_t _regcmd_find(uint32_t *rc, uint32_t n, uint32_t reg_off) {
    for (uint32_t i = 0; i < n; i++) {
        if ((rc[i * 2] & 0xFFFF) == reg_off)
            return rc[i * 2 + 1];
    }
    return 0;
}

/* ── CNA layer map: keyed by (wt_iova, tasks) ────────────────────────────────
 * cna_dcomp_addr0 (0x1110) = weight base address — stable per weight matrix,
 *   uniquely identifies Q/K/V/O/gate/up/down across all 32 transformer layers.
 * cna_feature_data_addr (0x1070) = activation input IOVA — varies per token.
 * ─────────────────────────────────────────────────────────────────────────── */
#define LAYER_MAP_MAX 512
typedef struct {
    uint32_t wt_iova;
    uint32_t tasks;
    long     count;
    long     hw_us_sum;
} _LayerEntry;
static _LayerEntry _layer_map[LAYER_MAP_MAX];
static int         _layer_map_count = 0;

static void _layer_record(uint32_t wt, uint32_t tasks, long hw_us) {
    for (int i = 0; i < _layer_map_count; i++) {
        if (_layer_map[i].wt_iova == wt && _layer_map[i].tasks == tasks) {
            _layer_map[i].count++;
            _layer_map[i].hw_us_sum += hw_us;
            return;
        }
    }
    if (_layer_map_count >= LAYER_MAP_MAX) return;
    _layer_map[_layer_map_count++] = (_LayerEntry){ wt, tasks, 1, hw_us };
}

static void _layer_map_sort(void) {
    for (int i = 1; i < _layer_map_count; i++) {
        _LayerEntry tmp = _layer_map[i];
        int j = i - 1;
        while (j >= 0 && _layer_map[j].wt_iova > tmp.wt_iova) {
            _layer_map[j + 1] = _layer_map[j]; j--;
        }
        _layer_map[j + 1] = tmp;
    }
}

/* ── destructor: final summary ───────────────────────────────────────────── */
__attribute__((destructor))
static void _npu_hook_fini(void) {
    fprintf(stderr,
        "[NPU_HOOK] === FINAL SUMMARY ===\n"
        "[NPU_HOOK] ACTION=%ld SUBMIT=%ld SYNC=%ld CREATE=%ld MAP=%ld DESTROY=%ld OTHER=%ld\n",
        _cnt_action, _cnt_submit, _cnt_sync,
        _cnt_create, _cnt_map, _cnt_destroy, _cnt_other);

    fprintf(stderr, "[NPU_HOOK] ACTION histogram (non-zero only):\n");
    for (int i = 0; i < ACTION_MAX; i++) {
        if (_action_hist[i] > 0)
            fprintf(stderr, "  [%2d] %-22s %ld\n", i, _action_name(i), _action_hist[i]);
    }

    if (_op_hist_count > 0) {
        long total_us = 0;
        for (int i = 0; i < _op_hist_count; i++) total_us += _op_hist[i].hw_us_sum;
        fprintf(stderr,
            "[NPU_HOOK] op-type profile (%d types, core=0x1, total_ms=%ld):\n",
            _op_hist_count, total_us / 1000);
        for (int i = 0; i < _op_hist_count; i++) {
            _OpEntry *e = &_op_hist[i];
            long avg_us = e->count ? e->hw_us_sum / e->count : 0;
            fprintf(stderr,
                "  ena=0x%02x cfg=%4u : calls=%6ld  avg_us=%6ld  total_ms=%6ld  pct=%5.1f%%\n",
                e->enable_mask, e->regcfg_amount,
                e->count, avg_us, e->hw_us_sum / 1000,
                total_us ? 100.0 * e->hw_us_sum / total_us : 0.0);
        }
    }

    if (_layer_map_count > 0) {
        _layer_map_sort();
        /* Find the weight buffer DMA base for printing relative offsets. */
        uint64_t wt_base = 0;
        for (int i = 1; i <= MAX_GEM_HANDLES; i++) {
            /* Largest DMA-mapped buffer is the model weight buffer. */
            if (_gems[i].size > 500*1024*1024 && _gems[i].dma_addr) {
                wt_base = _gems[i].dma_addr & 0xFFFFFFFFULL;
                break;
            }
        }
        long total_wt_us = 0;
        for (int i = 0; i < _layer_map_count; i++) total_wt_us += _layer_map[i].hw_us_sum;
        fprintf(stderr,
            "[NPU_HOOK] CNA layer map (%d unique weight×tasks, core=0x1, total_ms=%ld):\n",
            _layer_map_count, total_wt_us / 1000);
        for (int i = 0; i < _layer_map_count; i++) {
            _LayerEntry *e = &_layer_map[i];
            long avg_us = e->count ? e->hw_us_sum / e->count : 0;
            uint32_t wt_off = e->wt_iova ? (e->wt_iova - (uint32_t)wt_base) : 0;
            fprintf(stderr,
                "  wt=0x%08x (+0x%08x) tasks=%2u : calls=%6ld  avg_us=%6ld"
                "  total_ms=%5ld  pct=%5.1f%%\n",
                e->wt_iova, wt_off, e->tasks,
                e->count, avg_us, e->hw_us_sum / 1000,
                total_wt_us ? 100.0 * e->hw_us_sum / total_wt_us : 0.0);
        }
    }

    if (_sync_tracker_count > 0) {
        fprintf(stderr, "[NPU_HOOK] MEM_SYNC by obj_addr (%d unique):\n", _sync_tracker_count);
        for (int i = 0; i < _sync_tracker_count; i++) {
            _SyncEntry *e = &_sync_tracker[i];
            fprintf(stderr,
                "  obj=0x%016llx  →dev: calls=%ld bytes=%llu  ←dev: calls=%ld bytes=%llu\n",
                (unsigned long long)e->obj_addr,
                e->calls_to_dev,   (unsigned long long)e->total_bytes_to_dev,
                e->calls_from_dev, (unsigned long long)e->total_bytes_from_dev);
        }
    }
}

/* ── mmap hook — catch model file mapping into NPU IOVA space ───────────── */

typedef void *(*orig_mmap_t)(void *, size_t, int, int, int, off_t);

static void _log_mmap(const char *sym, void *ret, int fd, size_t length, int prot, int flags, off_t offset) {
    if (ret == MAP_FAILED || length < (1 << 20)) return;
    const char *npu = "";
    if (_is_npu_fd(fd)) {
        npu = "(NPU)";
        /* Record UVA immediately: match by mmap_offset from MEM_MAP. */
        uint64_t moff = (uint64_t)(unsigned long long)offset;
        for (int i = 1; i <= MAX_GEM_HANDLES; i++) {
            if (_gems[i].mmap_offset == moff && _gems[i].size == length) {
                _gems[i].uva = (uintptr_t)ret;
            }
        }
    }
    fprintf(stderr,
        "[NPU_HOOK] %s fd=%d len=0x%zx offset=0x%llx → uva=%p  %s\n",
        sym, fd, length, (unsigned long long)offset, ret, npu);
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    orig_mmap_t orig = (orig_mmap_t)dlsym(RTLD_NEXT, "mmap");
    void *ret = orig(addr, length, prot, flags, fd, offset);
    _log_mmap("mmap", ret, fd, length, prot, flags, offset);
    return ret;
}

void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    orig_mmap_t orig = (orig_mmap_t)dlsym(RTLD_NEXT, "mmap64");
    void *ret = orig(addr, length, prot, flags, fd, offset);
    _log_mmap("mmap64", ret, fd, length, prot, flags, offset);
    return ret;
}

/* Hook read/pread/fread to find model file loading (large reads). */
typedef ssize_t (*orig_pread_t)(int, void *, size_t, off_t);
typedef ssize_t (*orig_read_t)(int, void *, size_t);
typedef size_t  (*orig_fread_t)(void *, size_t, size_t, void *);

static long _read_total_mb = 0;

ssize_t pread(int fd, void *buf, size_t count, off_t offset) {
    orig_pread_t orig = (orig_pread_t)dlsym(RTLD_NEXT, "pread");
    ssize_t ret = orig(fd, buf, count, offset);
    if (ret > (1 << 20)) {
        _read_total_mb += ret >> 20;
        fprintf(stderr,
            "[NPU_HOOK] pread fd=%d count=0x%zx offset=0x%llx → %zd (total=%ldMB)\n",
            fd, count, (unsigned long long)offset, ret, _read_total_mb);
    }
    return ret;
}

ssize_t read(int fd, void *buf, size_t count) {
    orig_read_t orig = (orig_read_t)dlsym(RTLD_NEXT, "read");
    ssize_t ret = orig(fd, buf, count);
    if (ret > (1 << 20)) {
        _read_total_mb += ret >> 20;
        fprintf(stderr,
            "[NPU_HOOK] read fd=%d count=0x%zx → %zd (total=%ldMB)\n",
            fd, count, ret, _read_total_mb);
    }
    return ret;
}

size_t fread(void *__restrict ptr, size_t size, size_t nmemb, FILE *__restrict stream) {
    orig_fread_t orig = (orig_fread_t)dlsym(RTLD_NEXT, "fread");
    size_t ret = orig(ptr, size, nmemb, stream);
    size_t bytes = ret * size;
    if (bytes > (1 << 20)) {
        _read_total_mb += bytes >> 20;
        /* Check if dest buffer is one of our mapped GEM handles. */
        int gem_h = 0;
        for (int i = 1; i <= MAX_GEM_HANDLES; i++) {
            if (!_gems[i].uva || !_gems[i].size) continue;
            uintptr_t dest = (uintptr_t)ptr;
            if (dest >= _gems[i].uva && dest < _gems[i].uva + _gems[i].size) {
                gem_h = i;
                break;
            }
        }
        if (gem_h)
            fprintf(stderr,
                "[NPU_HOOK] fread → dest=%p (%zuMB) into GEM handle %d"
                " (dma=0x%llx off=0x%lx)\n",
                ptr, bytes >> 20, gem_h,
                (unsigned long long)(_gems[gem_h].dma_addr & 0xFFFFFFFFULL),
                (uintptr_t)ptr - _gems[gem_h].uva);
        else
            fprintf(stderr,
                "[NPU_HOOK] fread → dest=%p (%zuMB) NOT in any GEM (total=%ldMB)\n",
                ptr, bytes >> 20, _read_total_mb);
    }
    return ret;
}

/* ── open / openat hooks ─────────────────────────────────────────────────── */

typedef int (*orig_open_t)(const char *, int, ...);
typedef int (*orig_openat_t)(int, const char *, int, ...);

int open(const char *path, int flags, ...) {
    orig_open_t orig = (orig_open_t)dlsym(RTLD_NEXT, "open");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    int fd = orig(path, flags, mode);
    if (fd >= 0 && _path_is_npu(path)) {
        fprintf(stderr, "[NPU_HOOK] open %s → fd=%d\n", path, fd);
        _track_fd(fd);
    }
    return fd;
}

int openat(int dirfd, const char *path, int flags, ...) {
    orig_openat_t orig = (orig_openat_t)dlsym(RTLD_NEXT, "openat");
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    int fd = orig(dirfd, path, flags, mode);
    if (fd >= 0 && _path_is_npu(path)) {
        fprintf(stderr, "[NPU_HOOK] openat %s → fd=%d\n", path, fd);
        _track_fd(fd);
    }
    return fd;
}

/* ── /proc/self/maps scan ────────────────────────────────────────────────────
 *
 * DRM GEM mmaps appear in /proc/self/maps with offset=0. Match candidates
 * by size. Task-buffer handles are validated via rknpu_task.regcmd_addr IOVA
 * range. Non-task handles (weight / regcmd buffers) are resolved by size only.
 * ─────────────────────────────────────────────────────────────────────────── */

#define MAX_VMAS 64
typedef struct { uintptr_t start; size_t size; } _VmaEntry;
static _VmaEntry _gem_vmas[MAX_VMAS];
static int       _gem_vma_count = 0;
static int       _maps_scanned  = 0;

static void _scan_proc_maps(void) {
    if (_maps_scanned) return;
    _maps_scanned = 1;
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) { fprintf(stderr, "[NPU_HOOK] cannot open /proc/self/maps\n"); return; }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t start, end;
        char perms[8];
        unsigned long long offset;
        if (sscanf(line, "%lx-%lx %7s %llx", &start, &end, perms, &offset) < 4) continue;
        if (perms[0] != 'r' || perms[1] != 'w' || perms[3] != 's') continue;
        if (_gem_vma_count >= MAX_VMAS) continue;
        _gem_vmas[_gem_vma_count].start = start;
        _gem_vmas[_gem_vma_count].size  = (size_t)(end - start);
        _gem_vma_count++;
    }
    fclose(f);
}

/* Validate that uva points to a task buffer: rknpu_task[tidx].regcmd_addr
 * must be a known NPU IOVA. */
static int _validate_task_buf(uintptr_t uva, size_t buf_size, uint32_t tidx) {
    size_t boff = (size_t)tidx * sizeof(struct rknpu_task);
    if (boff + sizeof(struct rknpu_task) > buf_size) return 0;
    struct rknpu_task *t = (struct rknpu_task *)(uva + boff);
    return _gem_by_iova(t->regcmd_addr) > 0;
}

/* Resolve task buffer handle using validation. */
static int _resolve_task_uva(int h, uint32_t tidx) {
    if (_gems[h].uva) return 1;
    if (!_gems[h].obj_addr || !_gems[h].size) return 0;
    for (int i = 0; i < _gem_vma_count; i++) {
        if (_gem_vmas[i].size != _gems[h].size) continue;
        if (_validate_task_buf(_gem_vmas[i].start, _gem_vmas[i].size, tidx)) {
            _gems[h].uva = _gem_vmas[i].start;
            fprintf(stderr, "[NPU_HOOK] task-buf: handle=%d uva=0x%lx dma=0x%llx\n",
                h, _gems[h].uva, (unsigned long long)_gems[h].dma_addr);
            return 1;
        }
    }
    return 0;
}

/* Resolve any GEM handle by size only (for weight/regcmd buffers). */
static int _resolve_uva_by_size(int h) {
    if (_gems[h].uva) return 1;
    if (!_gems[h].size) return 0;
    for (int i = 0; i < _gem_vma_count; i++) {
        if (_gem_vmas[i].size == _gems[h].size) {
            _gems[h].uva = _gem_vmas[i].start;
            return 1;
        }
    }
    return 0;
}


/* Per-unit regcmd dump.  Count and skip controlled by env:
 *   NPU_HOOK_DUMP_CNA       = N (default 1)   how many to dump
 *   NPU_HOOK_DUMP_DPU       = N (default 1)
 *   NPU_HOOK_DUMP_CNA_SKIP  = N (default 0)   how many to skip first
 *   NPU_HOOK_DUMP_DPU_SKIP  = N (default 0)
 *
 * Effective window per unit: SEEN > SKIP and DUMPED < MAX.
 *
 * Skip is the lever for capturing generation-phase regcmds: prefill
 * issues ~16000 CNA submits, so SKIP=16000 DUMP=100 captures the
 * first 100 generation-phase CNA dispatches.
 *
 * Submit numbers tagged so we can correlate to the SUBMIT line. */
static int _rcmd_seen_cna   = 0;   /* total CNA submits observed */
static int _rcmd_seen_dpu   = 0;
static int _rcmd_dumped_cna = 0;
static int _rcmd_dumped_dpu = 0;
static int _rcmd_max_cna    = -1;
static int _rcmd_max_dpu    = -1;
static int _rcmd_skip_cna   = -1;
static int _rcmd_skip_dpu   = -1;
/* _cnt_submit is declared at file scope above (line ~95) — visible here. */
/* RKLLM dispatches across 3 NPU cores concurrently → 3 threads call
 * _dump_full_regcmd simultaneously.  Original implementation did 75
 * separate fprintf() calls per dump, which interleave under thread
 * concurrency (a dump from thread A can be split across one from
 * thread B).  Result was unreadable mixed regcmds in the trace log.
 *
 * Fix: build the entire dump in a stack buffer with snprintf, emit
 * with ONE atomic fwrite/write call.  Each dump = 1 START header +
 * up to 256 register lines + 1 END marker ≈ 6 KB max.  Buffer 8 KB
 * for safety.  flockfile() around the fwrite serialises with any
 * other concurrent fprintf still in the file. */
static void _dump_full_regcmd(uint32_t *rc, uint32_t n, const char *unit) {
    int is_cna = (unit[0] == 'C');
    int *seen  = is_cna ? &_rcmd_seen_cna   : &_rcmd_seen_dpu;
    int *count = is_cna ? &_rcmd_dumped_cna : &_rcmd_dumped_dpu;
    int *max   = is_cna ? &_rcmd_max_cna    : &_rcmd_max_dpu;
    int *skip  = is_cna ? &_rcmd_skip_cna   : &_rcmd_skip_dpu;
    if (*max < 0) {
        const char *env_max  = is_cna ? getenv("NPU_HOOK_DUMP_CNA")
                                      : getenv("NPU_HOOK_DUMP_DPU");
        const char *env_skip = is_cna ? getenv("NPU_HOOK_DUMP_CNA_SKIP")
                                      : getenv("NPU_HOOK_DUMP_DPU_SKIP");
        *max  = (env_max  && env_max[0])  ? atoi(env_max)  : 1;
        *skip = (env_skip && env_skip[0]) ? atoi(env_skip) : 0;
        if (*max  < 0) *max  = 0;
        if (*skip < 0) *skip = 0;
        fprintf(stderr, "[NPU_HOOK] dump-window %s skip=%d max=%d\n",
                unit, *skip, *max);
    }
    (*seen)++;
    if (*seen <= *skip)    return;   /* skipping warmup */
    if (*count >= *max)    return;   /* dumped enough */
    (*count)++;

    char  buf[8192];
    char *p   = buf;
    char *end = buf + sizeof(buf);
    int   w;

    w = snprintf(p, end - p,
                 "[NPU_HOOK] REGCMD_DUMP unit=%s n=%u seq=%ld (%d/%d)\n",
                 unit, n, _cnt_submit, *count, *max);
    if (w > 0 && p + w < end) p += w;
    for (uint32_t i = 0; i < n; i++) {
        w = snprintf(p, end - p, " %08x %08x\n", rc[i * 2], rc[i * 2 + 1]);
        if (w <= 0 || p + w >= end) break;     /* overflow — truncate */
        p += w;
    }
    w = snprintf(p, end - p, "[NPU_HOOK] REGCMD_DUMP_END\n");
    if (w > 0 && p + w < end) p += w;

    /* Atomic emit — fwrite + flockfile to serialise against other
     * threads' fprintf in this stream. */
    flockfile(stderr);
    fwrite(buf, 1, (size_t)(p - buf), stderr);
    fflush(stderr);
    funlockfile(stderr);
}

/* ── ioctl hook ──────────────────────────────────────────────────────────── */

typedef int (*orig_ioctl_t)(int, unsigned long, ...);

int ioctl(int fd, unsigned long request, ...) {
    orig_ioctl_t orig = (orig_ioctl_t)dlsym(RTLD_NEXT, "ioctl");
    va_list args;
    va_start(args, request);
    void *argp = va_arg(args, void *);
    va_end(args);

    int ret = orig(fd, request, argp);

    if (!_is_npu_fd(fd)) return ret;

    unsigned int num  = (request >> 0) & 0xFF;
    unsigned int type = (request >> 8) & 0xFF;
    unsigned int size = (request >> 16) & 0x3FFF;

    if (type != 0x64) { _cnt_other++; return ret; }

    switch (num) {

    /* ── RKNPU_ACTION (0x40) ─────────────────────────────────────────── */
    case 0x40: {
        _cnt_action++;
        struct rknpu_action *a = (struct rknpu_action *)argp;
        if (!a) break;
        if (a->flags < ACTION_MAX) _action_hist[a->flags]++;
        if (_cnt_action > DUMP_COUNT) break;
        fprintf(stderr,
            "[NPU_HOOK] ACTION #%ld fd=%d | action=%s(%u) value=%u ret=%d\n",
            _cnt_action, fd, _action_name(a->flags), a->flags, a->value, ret);
        break;
    }

    /* ── RKNPU_SUBMIT (0x41) ─────────────────────────────────────────── */
    case 0x41: {
        _cnt_submit++;
        struct rknpu_submit *s = (struct rknpu_submit *)argp;
        if (!s) break;

        _scan_proc_maps();

        /* Resolve task buffer and read op metadata. */
        uint32_t    ena      = 0;
        uint32_t    cfg      = 0;
        uint32_t    feat     = 0;   /* cna_feature_data_addr (0x1070) */
        uint32_t    wt       = 0;   /* cna_dcomp_addr0       (0x1110) */
        const char *sub_unit = "?";
        int         have_task = 0;

        int task_h = _gem_by_obj(s->task_obj_addr);
        if (task_h > 0 && s->subcore_task[0].task_number > 0) {
            uint32_t tidx = s->subcore_task[0].task_start;
            _resolve_task_uva(task_h, tidx);
            if (_gems[task_h].uva) {
                size_t boff = (size_t)tidx * sizeof(struct rknpu_task);
                if (boff + sizeof(struct rknpu_task) <= _gems[task_h].size) {
                    struct rknpu_task *tarr = (struct rknpu_task *)_gems[task_h].uva;
                    struct rknpu_task *t    = &tarr[tidx];
                    ena  = t->enable_mask;
                    cfg  = t->regcfg_amount;
                    have_task = 1;

                    /* Resolve regcmd buffer and scan for key registers. */
                    uint64_t rcmd_iova = t->regcmd_addr & 0xFFFFFFFFULL;
                    int rh = _gem_by_iova(rcmd_iova);
                    if (rh > 0) {
                        _resolve_uva_by_size(rh);
                        if (_gems[rh].uva) {
                            uint64_t base = _gems[rh].dma_addr & 0xFFFFFFFFULL;
                            uint32_t *rc  = (uint32_t *)(_gems[rh].uva + (rcmd_iova - base));
                            uint32_t  n   = cfg + RKNPU_PC_DATA_EXTRA_AMOUNT;
                            sub_unit = _subunit_name(rc[0]);
                            feat = _regcmd_find(rc, n, 0x1070); /* cna_feature_data_addr */
                            wt   = _regcmd_find(rc, n, 0x1110); /* cna_dcomp_addr0 */
                            _dump_full_regcmd(rc, n, sub_unit);  /* per-unit one-shot dump */
                        }
                    }
                }
            }
        }

        /* Always: compact timing line. */
        if (have_task && (feat || wt)) {
            fprintf(stderr,
                "[NPU_HOOK] SUBMIT #%ld hw_us=%lld core=0x%x tasks=%u"
                " ena=0x%02x cfg=%u unit=%s feat=0x%08x wt=0x%08x"
                " flags=0x%x tbase=0x%llx fence=%d ret=%d\n",
                _cnt_submit, (long long)(s->hw_elapse_time / 1000),
                s->core_mask, s->task_number, ena, cfg, sub_unit, feat, wt,
                s->flags, (unsigned long long)s->task_base_addr, s->fence_fd, ret);
        } else if (have_task) {
            fprintf(stderr,
                "[NPU_HOOK] SUBMIT #%ld hw_us=%lld core=0x%x tasks=%u"
                " ena=0x%02x cfg=%u unit=%s"
                " flags=0x%x tbase=0x%llx fence=%d ret=%d\n",
                _cnt_submit, (long long)(s->hw_elapse_time / 1000),
                s->core_mask, s->task_number, ena, cfg, sub_unit,
                s->flags, (unsigned long long)s->task_base_addr, s->fence_fd, ret);
        } else {
            fprintf(stderr,
                "[NPU_HOOK] SUBMIT #%ld hw_us=%lld core=0x%x tasks=%u"
                " flags=0x%x tbase=0x%llx fence=%d ret=%d\n",
                _cnt_submit, (long long)(s->hw_elapse_time / 1000),
                s->core_mask, s->task_number,
                s->flags, (unsigned long long)s->task_base_addr, s->fence_fd, ret);
        }

        /* Op-type histogram + layer map: core=0x1 only to avoid 3× counting. */
        if (s->core_mask == 0x1 && have_task) {
            _op_record(ena, cfg, (long)(s->hw_elapse_time / 1000));
            if (feat || wt)
                _layer_record(wt, s->task_number, (long)(s->hw_elapse_time / 1000));
        }

        /* Full struct dump for first DUMP_COUNT. */
        if (_cnt_submit <= DUMP_COUNT) {
            fprintf(stderr,
                "  flags=0x%x timeout=%u task_start=%u task_ctr=%u priority=%d"
                " iommu_domain=%u fence_fd=%d task_obj=0x%llx\n",
                s->flags, s->timeout, s->task_start, s->task_counter, s->priority,
                s->iommu_domain_id, s->fence_fd,
                (unsigned long long)s->task_obj_addr);
            for (int i = 0; i < 5; i++) {
                if (s->subcore_task[i].task_number == 0) continue;
                fprintf(stderr, "  subcore[%d]: task_start=%u task_number=%u\n",
                    i, s->subcore_task[i].task_start, s->subcore_task[i].task_number);
            }
            /* Dump full task struct for CNA ops */
            if (sub_unit[0] == 'C' && have_task && _gems[task_h].uva) {
                struct rknpu_task *tarr = (struct rknpu_task *)_gems[task_h].uva;
                uint32_t tidx = s->subcore_task[0].task_start;
                struct rknpu_task *t = &tarr[tidx];
                fprintf(stderr,
                    "  TASK[%u]: flags=0x%x op_idx=%u ena=0x%02x"
                    " int_mask=0x%x int_clear=0x%x int_status=0x%x"
                    " regcfg_amount=%u regcfg_offset=%u regcmd_addr=0x%llx\n",
                    tidx, t->flags, t->op_idx, t->enable_mask,
                    t->int_mask, t->int_clear, t->int_status,
                    t->regcfg_amount, t->regcfg_offset,
                    (unsigned long long)t->regcmd_addr);
            }
        }
        break;
    }

    /* ── RKNPU_MEM_CREATE (0x42) ─────────────────────────────────────── */
    case 0x42: {
        _cnt_create++;
        struct rknpu_mem_create *m = (struct rknpu_mem_create *)argp;
        if (!m) break;
        if (m->handle > 0 && m->handle <= MAX_GEM_HANDLES) {
            _gems[m->handle].obj_addr = m->obj_addr;
            _gems[m->handle].dma_addr = m->dma_addr;
            _gems[m->handle].size     = (size_t)m->size;
        }
        /* Always log MEM_CREATE — critical for IOVA mapping when rknpu2 is stopped. */
        fprintf(stderr,
            "[NPU_HOOK] MEM_CREATE #%ld fd=%d | handle=%u flags=0x%x"
            " size=%llu obj_addr=0x%llx dma_addr=0x%llx sram_size=%llu"
            " iommu_domain=%d core_mask=0x%x ret=%d\n",
            _cnt_create, fd, m->handle, m->flags,
            (unsigned long long)m->size,
            (unsigned long long)m->obj_addr,
            (unsigned long long)m->dma_addr,
            (unsigned long long)m->sram_size,
            m->iommu_domain_id, m->core_mask, ret);
        break;
    }

    /* ── RKNPU_MEM_MAP (0x43) ────────────────────────────────────────── */
    case 0x43: {
        _cnt_map++;
        struct rknpu_mem_map *m = (struct rknpu_mem_map *)argp;
        if (!m) break;
        if (m->handle > 0 && m->handle <= MAX_GEM_HANDLES)
            _gems[m->handle].mmap_offset = m->offset;
        if (_cnt_map > DUMP_COUNT) break;
        fprintf(stderr,
            "[NPU_HOOK] MEM_MAP #%ld fd=%d | handle=%u offset=0x%llx ret=%d\n",
            _cnt_map, fd, m->handle, (unsigned long long)m->offset, ret);
        break;
    }

    /* ── RKNPU_MEM_DESTROY (0x44) ────────────────────────────────────── */
    case 0x44: {
        _cnt_destroy++;
        if (_cnt_destroy > DUMP_COUNT) break;
        struct rknpu_mem_destroy *m = (struct rknpu_mem_destroy *)argp;
        if (!m) break;
        fprintf(stderr,
            "[NPU_HOOK] MEM_DESTROY #%ld fd=%d | handle=%u obj_addr=0x%llx ret=%d\n",
            _cnt_destroy, fd, m->handle, (unsigned long long)m->obj_addr, ret);
        break;
    }

    /* ── RKNPU_MEM_SYNC (0x45) ───────────────────────────────────────── */
    case 0x45: {
        _cnt_sync++;
        struct rknpu_mem_sync *m = (struct rknpu_mem_sync *)argp;
        if (!m) break;
        const char *dir = (m->flags & RKNPU_MEM_SYNC_TO_DEVICE)  ? "→NPU" :
                          (m->flags & RKNPU_MEM_SYNC_FROM_DEVICE) ? "←CPU" : "?";
        _sync_track(m->obj_addr, m->flags, m->size);
        if (_cnt_sync > DUMP_COUNT) break;
        fprintf(stderr,
            "[NPU_HOOK] MEM_SYNC #%ld fd=%d | dir=%s flags=0x%x"
            " obj=0x%llx off=0x%llx size=%llu ret=%d\n",
            _cnt_sync, fd, dir, m->flags,
            (unsigned long long)m->obj_addr,
            (unsigned long long)m->offset,
            (unsigned long long)m->size,
            ret);
        break;
    }

    default:
        _cnt_other++;
        if (_cnt_other <= 8)
            fprintf(stderr,
                "[NPU_HOOK] UNKNOWN fd=%d num=0x%02x size=%u req=0x%08lx\n",
                fd, num, size, request);
        break;
    }

    return ret;
}
