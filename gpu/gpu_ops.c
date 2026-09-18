/* GPU SGEMM via Vulkan compute — pre-loaded weight path.
 *
 * On RK3588 (UMA): CPU and GPU share LPDDR5.  All buffers use
 * HOST_VISIBLE | HOST_COHERENT memory, avoiding staging copies.
 * The GPU reads weight data directly from the same physical pages the
 * CPU dequantised into — no explicit transfer needed.
 *
 * SPIR-V shader: gpu/shaders/sgemm.spv  (compiled from sgemm.comp).
 * Path resolved via GPU_SHADER_DIR compile-time define (set by Makefile). */

#include "gpu_ops.h"
#include "vk_ctx.h"
#include "weight_layout.h"   /* wl_dequant_row, wl_row_bytes, wl_type_t */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef GPU_SHADER_DIR
#define GPU_SHADER_DIR "./shaders"
#endif

/* Buffer helpers (gpu_buf_alloc/free) and SPIR-V loader live in gpu/vk_ctx.c
 * so other GPU consumers can reuse them. */

/* -------------------------------------------------------------------------
 * Descriptor set layout (3 STORAGE_BUFFER bindings: x, W, y)
 * ---------------------------------------------------------------------- */

static VkDescriptorSetLayout make_ds_layout(gpu_ctx_t *ctx) {
    VkDescriptorSetLayoutBinding b[3];
    for (int i = 0; i < 3; i++) {
        b[i] = (VkDescriptorSetLayoutBinding){
            .binding         = (uint32_t)i,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo lci = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings    = b,
    };
    VkDescriptorSetLayout layout;
    vkCreateDescriptorSetLayout(gpu_vkdevice(ctx), &lci, NULL, &layout);
    return layout;
}

/* -------------------------------------------------------------------------
 * gpu_weight_t — self-contained Vulkan pipeline + buffers per weight
 * ---------------------------------------------------------------------- */

struct gpu_weight {
    gpu_buf_t  wbuf;   /* W[N×K] fp32, DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT */
    gpu_buf_t  xbuf;   /* x[K]   fp32, HOST_VISIBLE | HOST_COHERENT                */
    gpu_buf_t  ybuf;   /* y[N]   fp32, HOST_VISIBLE | HOST_COHERENT                */

    VkShaderModule        shader;
    VkDescriptorSetLayout ds_layout;
    VkPipelineLayout      pipe_layout;
    VkPipeline            pipeline;
    VkDescriptorPool      ds_pool;
    VkDescriptorSet       ds;

    int K, N;
};

gpu_weight_t *gpu_weight_load(gpu_ctx_t *ctx,
                               const void *W, int wtype,
                               int K, int N) {
    if (!ctx || !W) return NULL;

    gpu_weight_t *gw = calloc(1, sizeof(*gw));
    if (!gw) return NULL;
    gw->K = K; gw->N = N;

    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                              | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBufferUsageFlags sb = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    /* W buffer: large — dequantised fp32.  Written once at load; GPU reads every token. */
    VkDeviceSize wbytes = (VkDeviceSize)N * K * sizeof(float);
    if (gpu_buf_alloc(ctx, &gw->wbuf, wbytes, sb, hv) < 0) goto fail;

    /* Dequantise directly into the mapped GPU-visible buffer — no extra copy on UMA. */
    {
        size_t      row_bytes = wl_row_bytes((wl_type_t)wtype, K);
        const uint8_t *src   = (const uint8_t *)W;
        float         *dst   = (float *)gw->wbuf.mapped;
        for (int n = 0; n < N; n++)
            wl_dequant_row(src + (size_t)n * row_bytes, dst + (size_t)n * K,
                           (wl_type_t)wtype, K);
    }

    /* x / y: small hot buffers — rewritten every token. */
    if (gpu_buf_alloc(ctx, &gw->xbuf, (VkDeviceSize)K * sizeof(float), sb, hv) < 0) goto fail;
    if (gpu_buf_alloc(ctx, &gw->ybuf, (VkDeviceSize)N * sizeof(float), sb, hv) < 0) goto fail;

    /* Load SPIR-V and create shader module. */
    size_t spv_bytes = 0;
    uint32_t *spv = gpu_load_spirv("sgemm.spv", &spv_bytes, NULL);
    if (!spv) goto fail;

    VkShaderModuleCreateInfo smci = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_bytes,
        .pCode    = spv,
    };
    VkResult r = vkCreateShaderModule(gpu_vkdevice(ctx), &smci, NULL, &gw->shader);
    free(spv);
    if (r != VK_SUCCESS) { fprintf(stderr, "gpu: vkCreateShaderModule failed\n"); goto fail; }

    /* Descriptor set layout + pipeline layout. */
    gw->ds_layout = make_ds_layout(ctx);

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = sizeof(uint32_t) * 2,   /* K, N */
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &gw->ds_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcr,
    };
    if (vkCreatePipelineLayout(gpu_vkdevice(ctx), &plci, NULL, &gw->pipe_layout) != VK_SUCCESS)
        goto fail;

    /* Compute pipeline. */
    VkComputePipelineCreateInfo cpci = {
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
            .module              = gw->shader,
            .pName               = "main",
        },
        .layout = gw->pipe_layout,
    };
    if (vkCreateComputePipelines(gpu_vkdevice(ctx), VK_NULL_HANDLE, 1, &cpci, NULL,
                                  &gw->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "gpu: vkCreateComputePipelines failed\n");
        goto fail;
    }

    /* Descriptor pool + set. */
    VkDescriptorPoolSize dps = {
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &dps,
    };
    if (vkCreateDescriptorPool(gpu_vkdevice(ctx), &dpci, NULL, &gw->ds_pool) != VK_SUCCESS)
        goto fail;

    VkDescriptorSetAllocateInfo dsai = {
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = gw->ds_pool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &gw->ds_layout,
    };
    if (vkAllocateDescriptorSets(gpu_vkdevice(ctx), &dsai, &gw->ds) != VK_SUCCESS)
        goto fail;

    /* Bind x, W, y buffers to the descriptor set (binding 0=x, 1=W, 2=y). */
    VkDescriptorBufferInfo bi[3] = {
        { gw->xbuf.buf, 0, gw->xbuf.size },
        { gw->wbuf.buf, 0, gw->wbuf.size },
        { gw->ybuf.buf, 0, gw->ybuf.size },
    };
    VkWriteDescriptorSet wr[3];
    for (int i = 0; i < 3; i++) {
        wr[i] = (VkWriteDescriptorSet){
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = gw->ds,
            .dstBinding      = (uint32_t)i,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo     = &bi[i],
        };
    }
    vkUpdateDescriptorSets(gpu_vkdevice(ctx), 3, wr, 0, NULL);

    fprintf(stderr, "gpu: lm_head weight loaded — W=%dx%d  %.1fMB GPU-visible\n",
            N, K, (double)wbytes / (1 << 20));
    return gw;

fail:
    gpu_weight_free(ctx, gw);
    return NULL;
}

void gpu_weight_free(gpu_ctx_t *ctx, gpu_weight_t *gw) {
    if (!ctx || !gw) return;
    VkDevice dev = gpu_vkdevice(ctx);
    if (gw->ds_pool)     vkDestroyDescriptorPool(dev, gw->ds_pool, NULL);
    if (gw->pipeline)    vkDestroyPipeline(dev, gw->pipeline, NULL);
    if (gw->pipe_layout) vkDestroyPipelineLayout(dev, gw->pipe_layout, NULL);
    if (gw->ds_layout)   vkDestroyDescriptorSetLayout(dev, gw->ds_layout, NULL);
    if (gw->shader)      vkDestroyShaderModule(dev, gw->shader, NULL);
    gpu_buf_free(ctx, &gw->ybuf);
    gpu_buf_free(ctx, &gw->xbuf);
    gpu_buf_free(ctx, &gw->wbuf);
    free(gw);
}

/* -------------------------------------------------------------------------
 * gpu_matmul — dispatch the pre-built pipeline
 * ---------------------------------------------------------------------- */

int gpu_matmul(gpu_ctx_t *ctx, gpu_weight_t *gw,
               float *y, const float *x, int K, int N) {
    if (!ctx || !gw || K != gw->K || N != gw->N) return -1;

    /* Write x into the mapped input buffer (direct on UMA). */
    memcpy(gw->xbuf.mapped, x, (size_t)K * sizeof(float));

    /* Zero y buffer — caller already zeroed host y[], but GPU y buffer needs reset. */
    memset(gw->ybuf.mapped, 0, (size_t)N * sizeof(float));

    /* Record + dispatch. */
    gpu_cmd_begin(ctx);

    vkCmdBindPipeline(gpu_vkcmd(ctx), VK_PIPELINE_BIND_POINT_COMPUTE, gw->pipeline);
    vkCmdBindDescriptorSets(gpu_vkcmd(ctx), VK_PIPELINE_BIND_POINT_COMPUTE,
                             gw->pipe_layout, 0, 1, &gw->ds, 0, NULL);

    uint32_t pc[2] = { (uint32_t)K, (uint32_t)N };
    vkCmdPushConstants(gpu_vkcmd(ctx), gw->pipe_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

    uint32_t groups = ((uint32_t)N + 127u) / 128u;
    vkCmdDispatch(gpu_vkcmd(ctx), groups, 1, 1);

    /* gpu_cmd_submit_wait inserts host-read barrier before submitting. */
    gpu_cmd_submit_wait(ctx);

    /* Copy result from GPU y buffer to caller's output array. */
    memcpy(y, gw->ybuf.mapped, (size_t)N * sizeof(float));
    return 0;
}
