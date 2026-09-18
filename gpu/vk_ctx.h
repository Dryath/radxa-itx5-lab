#pragma once
/* Vulkan compute context — headless, compute-only.
 * One instance per process; shared by all GPU ops. */

#include <vulkan/vulkan.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpu_ctx gpu_ctx_t;

/* Returns NULL if Vulkan is unavailable or no compute device found. */
gpu_ctx_t *gpu_ctx_create(void);
void       gpu_ctx_destroy(gpu_ctx_t *ctx);

/* Internal: accessed by gpu_ops.c. */
VkDevice                              gpu_vkdevice(gpu_ctx_t *ctx);
VkPhysicalDevice                      gpu_vkphys(gpu_ctx_t *ctx);
VkQueue                               gpu_vkqueue(gpu_ctx_t *ctx);
VkCommandBuffer                       gpu_vkcmd(gpu_ctx_t *ctx);
VkFence                               gpu_vkfence(gpu_ctx_t *ctx);
VkPhysicalDeviceMemoryProperties     *gpu_mem_props(gpu_ctx_t *ctx);

/* Find a memory type index satisfying type_filter and property flags. */
int gpu_find_memtype(gpu_ctx_t *ctx, uint32_t type_filter,
                     VkMemoryPropertyFlags props);

/* One-shot command buffer helpers. */
void gpu_cmd_begin(gpu_ctx_t *ctx);
void gpu_cmd_submit_wait(gpu_ctx_t *ctx);

/* -------------------------------------------------------------------------
 * Shared utilities (used by gpu_ops.c and other GPU consumers)
 * ---------------------------------------------------------------------- */

typedef struct {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *mapped;
    VkDeviceSize   size;
} gpu_buf_t;

/* Allocate a GPU buffer.  On UMA prefers DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT.
 * If props includes VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, the buffer is persistently
 * mapped (b->mapped is non-NULL).  Returns 0 on success, -1 on failure. */
int  gpu_buf_alloc(gpu_ctx_t *ctx, gpu_buf_t *b, VkDeviceSize size,
                    VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
void gpu_buf_free (gpu_ctx_t *ctx, gpu_buf_t *b);

/* Load a compiled SPIR-V shader from GPU_SHADER_DIR.  Caller must free() the result.
 * dir_override (optional) overrides GPU_SHADER_DIR (e.g. for an alternate shader dir). */
uint32_t *gpu_load_spirv(const char *filename, size_t *out_bytes,
                         const char *dir_override);

#ifdef __cplusplus
}
#endif
