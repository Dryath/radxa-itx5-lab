/* Vulkan compute context for Mali-G610 (RK3588). */

#include "vk_ctx.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

struct gpu_ctx {
    VkInstance                       instance;
    VkPhysicalDevice                 phys;
    VkDevice                         device;
    VkQueue                          queue;
    uint32_t                         qfam;
    VkCommandPool                    cmd_pool;
    VkCommandBuffer                  cmd_buf;
    VkFence                          fence;
    VkPhysicalDeviceMemoryProperties mem_props;
};

static uint32_t find_compute_qfam(VkPhysicalDevice phys) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, NULL);
    VkQueueFamilyProperties *qp = malloc(n * sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qp);
    uint32_t found = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { found = i; break; }
    }
    free(qp);
    return found;
}

gpu_ctx_t *gpu_ctx_create(void) {
    gpu_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    /* Instance — headless compute, no extensions needed for step 1. */
    VkApplicationInfo ai = {
        .sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_0,
    };
    VkInstanceCreateInfo ic = {
        .sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &ai,
    };
    if (vkCreateInstance(&ic, NULL, &ctx->instance) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateInstance failed\n");
        goto fail;
    }

    /* Physical device — pick first one with a compute queue. */
    uint32_t nd = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &nd, NULL);
    if (nd == 0) { fprintf(stderr, "gpu: no Vulkan devices\n"); goto fail; }
    VkPhysicalDevice *devs = malloc(nd * sizeof(*devs));
    vkEnumeratePhysicalDevices(ctx->instance, &nd, devs);
    ctx->phys = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < nd; i++) {
        uint32_t qf = find_compute_qfam(devs[i]);
        if (qf != UINT32_MAX) { ctx->phys = devs[i]; ctx->qfam = qf; break; }
    }
    free(devs);
    if (ctx->phys == VK_NULL_HANDLE) {
        fprintf(stderr, "gpu: no device with compute queue\n");
        goto fail;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->phys, &props);
    vkGetPhysicalDeviceMemoryProperties(ctx->phys, &ctx->mem_props);
    fprintf(stderr, "gpu: %s  maxStorageBufRange=%.0fMB\n",
            props.deviceName,
            (double)props.limits.maxStorageBufferRange / (1 << 20));

    /* Logical device. */
    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->qfam,
        .queueCount       = 1,
        .pQueuePriorities = &pri,
    };
    VkDeviceCreateInfo dc = {
        .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos    = &qci,
    };
    if (vkCreateDevice(ctx->phys, &dc, NULL, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateDevice failed\n");
        goto fail;
    }
    vkGetDeviceQueue(ctx->device, ctx->qfam, 0, &ctx->queue);

    /* Command pool + buffer. */
    VkCommandPoolCreateInfo cpi = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->qfam,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(ctx->device, &cpi, NULL, &ctx->cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateCommandPool failed\n");
        goto fail;
    }
    VkCommandBufferAllocateInfo cai = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = ctx->cmd_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(ctx->device, &cai, &ctx->cmd_buf) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkAllocateCommandBuffers failed\n");
        goto fail;
    }

    /* Fence for submit sync. */
    VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(ctx->device, &fi, NULL, &ctx->fence) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateFence failed\n");
        goto fail;
    }

    return ctx;
fail:
    gpu_ctx_destroy(ctx);
    return NULL;
}

void gpu_ctx_destroy(gpu_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->device) {
        vkDeviceWaitIdle(ctx->device);
        if (ctx->fence)    vkDestroyFence(ctx->device, ctx->fence, NULL);
        if (ctx->cmd_pool) vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
        vkDestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

VkDevice                          gpu_vkdevice(gpu_ctx_t *c) { return c->device; }
VkPhysicalDevice                  gpu_vkphys(gpu_ctx_t *c)   { return c->phys; }
VkQueue                           gpu_vkqueue(gpu_ctx_t *c)   { return c->queue; }
VkCommandBuffer                   gpu_vkcmd(gpu_ctx_t *c)     { return c->cmd_buf; }
VkFence                           gpu_vkfence(gpu_ctx_t *c)   { return c->fence; }
VkPhysicalDeviceMemoryProperties *gpu_mem_props(gpu_ctx_t *c) { return &c->mem_props; }

int gpu_find_memtype(gpu_ctx_t *ctx, uint32_t filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx->mem_props.memoryTypeCount; i++) {
        if ((filter & (1u << i)) &&
            (ctx->mem_props.memoryTypes[i].propertyFlags & props) == props)
            return (int)i;
    }
    return -1;
}

void gpu_cmd_begin(gpu_ctx_t *ctx) {
    vkResetCommandBuffer(ctx->cmd_buf, 0);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(ctx->cmd_buf, &bi);
}

/* -------------------------------------------------------------------------
 * Shared buffer + SPIR-V helpers
 * ---------------------------------------------------------------------- */

#ifndef GPU_SHADER_DIR
#define GPU_SHADER_DIR "./shaders"
#endif

int gpu_buf_alloc(gpu_ctx_t *ctx, gpu_buf_t *b, VkDeviceSize size,
                   VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    memset(b, 0, sizeof(*b));
    b->size = size;

    VkBufferCreateInfo bci = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(ctx->device, &bci, NULL, &b->buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(ctx->device, b->buf, &req);

    int mi = gpu_find_memtype(ctx, req.memoryTypeBits,
                               props | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mi < 0)
        mi = gpu_find_memtype(ctx, req.memoryTypeBits, props);
    if (mi < 0) {
        fprintf(stderr, "gpu: no memtype for %.1fMB buf\n", (double)size / (1 << 20));
        vkDestroyBuffer(ctx->device, b->buf, NULL);
        b->buf = VK_NULL_HANDLE;
        return -1;
    }

    VkMemoryAllocateInfo mai = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = (uint32_t)mi,
    };
    if (vkAllocateMemory(ctx->device, &mai, NULL, &b->mem) != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, b->buf, NULL);
        b->buf = VK_NULL_HANDLE;
        return -1;
    }
    vkBindBufferMemory(ctx->device, b->buf, b->mem, 0);
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        vkMapMemory(ctx->device, b->mem, 0, size, 0, &b->mapped);
    return 0;
}

void gpu_buf_free(gpu_ctx_t *ctx, gpu_buf_t *b) {
    if (!b || !b->buf) return;
    if (b->mapped) vkUnmapMemory(ctx->device, b->mem);
    vkDestroyBuffer(ctx->device, b->buf, NULL);
    vkFreeMemory(ctx->device, b->mem, NULL);
    memset(b, 0, sizeof(*b));
}

uint32_t *gpu_load_spirv(const char *filename, size_t *out_bytes,
                         const char *dir_override) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s",
             dir_override ? dir_override : GPU_SHADER_DIR, filename);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gpu: open %s: %s\n", path, strerror(errno)); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    uint32_t *buf = (uint32_t *)malloc((size_t)len);
    if (buf) { fread(buf, 1, (size_t)len, f); *out_bytes = (size_t)len; }
    fclose(f);
    return buf;
}

void gpu_cmd_submit_wait(gpu_ctx_t *ctx) {
    /* Barrier: shader writes → host reads. */
    VkMemoryBarrier mb = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(ctx->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, NULL, 0, NULL);

    vkEndCommandBuffer(ctx->cmd_buf);
    vkResetFences(ctx->device, 1, &ctx->fence);
    VkSubmitInfo si = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &ctx->cmd_buf,
    };
    vkQueueSubmit(ctx->queue, 1, &si, ctx->fence);
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
}
