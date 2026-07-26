/* vk_membw.c — Mali-G610 (Vulkan compute) memory-bandwidth probe + CPU concurrency.
 *
 * Closes the GPU side of the P0.5 question: is the Mali GPU a useful 3rd memory
 * master, or does it contend for the same shared ~25-30 GB/s ceiling as CPU+NPU?
 *
 * A compute shader streams dst[i]=src[i]+k over a large SSBO pair (read+write =
 * 8 bytes/elem). Measures GPU solo, CPU solo (STREAM triad on A76), and both at once.
 *
 * Build:  gcc -O3 -mcpu=cortex-a76 -D_GNU_SOURCE vk_membw.c -o /tmp/vk_membw -lvulkan -lpthread
 * Run:    /tmp/vk_membw /tmp/vk_membw.spv [seconds] [cpu_threads]
 */
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VKCHECK(x) do{ VkResult _r=(x); if(_r){ fprintf(stderr,"vk error %d at %s:%d\n",_r,__FILE__,__LINE__); exit(1);} }while(0)
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

#define NELEM (64L*1024*1024)   /* 64M floats = 256 MB per buffer */

/* ---- CPU triad (same as concurrency_probe) ---- */
#define CPU_N (16L*1024*1024)
typedef struct { int core; volatile int *run; double gbps; } cpu_arg_t;
static void *cpu_thread(void *p){
    cpu_arg_t *a=p; cpu_set_t s; CPU_ZERO(&s); CPU_SET(a->core,&s);
    pthread_setaffinity_np(pthread_self(),sizeof(s),&s);
    double *x=malloc(CPU_N*8),*y=malloc(CPU_N*8),*z=malloc(CPU_N*8);
    for(long i=0;i<CPU_N;i++){x[i]=1;y[i]=2;z[i]=3;}
    const double k=3; long pass=0; double t0=now_s();
    while(*a->run){ for(long i=0;i<CPU_N;i++) x[i]=y[i]+k*z[i]; pass++; if(x[pass&(CPU_N-1)]<0)printf("x"); }
    a->gbps=(double)pass*3*CPU_N*8/(now_s()-t0)/1e9; free(x);free(y);free(z); return NULL;
}
static volatile int g_run; static int g_nt; static int g_cores[8]; static pthread_t g_th[8]; static cpu_arg_t g_a[8];
static void cpu_start(void){ g_run=1; for(int i=0;i<g_nt;i++){g_a[i]=(cpu_arg_t){g_cores[i],&g_run,0}; pthread_create(&g_th[i],NULL,cpu_thread,&g_a[i]);} }
static double cpu_stop(void){ g_run=0; double s=0; for(int i=0;i<g_nt;i++){pthread_join(g_th[i],NULL); s+=g_a[i].gbps;} return s; }
static double cpu_solo(double dur){ cpu_start(); struct timespec ts={(time_t)dur,(long)((dur-(time_t)dur)*1e9)}; nanosleep(&ts,NULL); return cpu_stop(); }

/* ---- Vulkan globals ---- */
static VkDevice dev; static VkQueue queue; static VkCommandBuffer cmd; static VkFence fence;
#define GPU_BYTES ((double)NELEM*4*2)  /* read src + write dst per dispatch */
static double gpu_stream(double d){
    long it=0; double t0=now_s();
    do{
        VKCHECK(vkResetFences(dev,1,&fence));
        VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
        VKCHECK(vkQueueSubmit(queue,1,&si,fence));
        VKCHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));
        it++;
    }while(now_s()-t0<d);
    return (double)it*GPU_BYTES/(now_s()-t0)/1e9;
}

static uint32_t find_mem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want){
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&want)==want) return i;
    return UINT32_MAX;
}
static VkBuffer mk_buf(VkPhysicalDevice pd, VkDeviceSize sz, VkDeviceMemory *mem, void **map){
    VkBuffer b; VkBufferCreateInfo bi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=sz,
        .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VKCHECK(vkCreateBuffer(dev,&bi,0,&b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,b,&mr);
    uint32_t mt=find_mem(pd,mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo ai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=mr.size,.memoryTypeIndex=mt};
    VKCHECK(vkAllocateMemory(dev,&ai,0,mem));
    VKCHECK(vkBindBufferMemory(dev,b,*mem,0));
    if(map) VKCHECK(vkMapMemory(dev,*mem,0,sz,0,map));
    return b;
}

int main(int argc,char**argv){
    const char *spv = argc>1?argv[1]:"/tmp/vk_membw.spv";
    double dur = argc>2?atof(argv[2]):3.0;
    g_nt = argc>3?atoi(argv[3]):4;
    int a76[4]={4,5,6,7}; for(int i=0;i<g_nt&&i<8;i++) g_cores[i]= i<4?a76[i]:i-4;

    /* instance */
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.apiVersion=VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance inst; VKCHECK(vkCreateInstance(&ici,0,&inst));
    uint32_t npd=0; vkEnumeratePhysicalDevices(inst,&npd,0);
    if(!npd){ fprintf(stderr,"no Vulkan device\n"); return 1; }
    VkPhysicalDevice pds[8]; if(npd>8)npd=8; vkEnumeratePhysicalDevices(inst,&npd,pds);
    VkPhysicalDevice pd=pds[0];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd,&pp);
    printf("# GPU: %s  Vulkan %u.%u.%u\n",pp.deviceName,
           VK_VERSION_MAJOR(pp.apiVersion),VK_VERSION_MINOR(pp.apiVersion),VK_VERSION_PATCH(pp.apiVersion));

    /* queue + device */
    uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,0);
    VkQueueFamilyProperties qf[16]; if(nq>16)nq=16; vkGetPhysicalDeviceQueueFamilyProperties(pd,&nq,qf);
    uint32_t qfi=UINT32_MAX; for(uint32_t i=0;i<nq;i++) if(qf[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qfi=i;break;}
    float pr=1; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=qfi,.queueCount=1,.pQueuePriorities=&pr};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    VKCHECK(vkCreateDevice(pd,&dci,0,&dev));
    vkGetDeviceQueue(dev,qfi,0,&queue);

    /* buffers */
    VkDeviceMemory m0,m1; void *map0; VkDeviceSize sz=NELEM*4;
    VkBuffer src=mk_buf(pd,sz,&m0,&map0); VkBuffer dst=mk_buf(pd,sz,&m1,0);
    for(long i=0;i<NELEM;i++) ((float*)map0)[i]=1.0f;

    /* shader module */
    FILE*f=fopen(spv,"rb"); if(!f){perror("spv");return 1;} fseek(f,0,SEEK_END); long sl=ftell(f); fseek(f,0,SEEK_SET);
    uint32_t*code=malloc(sl); if(fread(code,1,sl,f)!=(size_t)sl){return 1;} fclose(f);
    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,.codeSize=sl,.pCode=code};
    VkShaderModule sm; VKCHECK(vkCreateShaderModule(dev,&smci,0,&sm));

    /* descriptors */
    VkDescriptorSetLayoutBinding b[2]={
        {.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dlci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=2,.pBindings=b};
    VkDescriptorSetLayout dsl; VKCHECK(vkCreateDescriptorSetLayout(dev,&dlci,0,&dsl));
    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=8};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&dsl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    VkPipelineLayout pl; VKCHECK(vkCreatePipelineLayout(dev,&plci,0,&pl));
    VkComputePipelineCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"},.layout=pl};
    VkPipeline pipe; VKCHECK(vkCreateComputePipelines(dev,0,1,&cpci,0,&pipe));

    VkDescriptorPoolSize ps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=2};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=1,.poolSizeCount=1,.pPoolSizes=&ps};
    VkDescriptorPool dp; VKCHECK(vkCreateDescriptorPool(dev,&dpci,0,&dp));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=dp,.descriptorSetCount=1,.pSetLayouts=&dsl};
    VkDescriptorSet ds; VKCHECK(vkAllocateDescriptorSets(dev,&dsai,&ds));
    VkDescriptorBufferInfo bi0={.buffer=src,.range=sz},bi1={.buffer=dst,.range=sz};
    VkWriteDescriptorSet w[2]={
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi0},
        {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=ds,.dstBinding=1,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi1}};
    vkUpdateDescriptorSets(dev,2,w,0,0);

    /* command buffer: one dispatch */
    VkCommandPoolCreateInfo cpi={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=qfi};
    VkCommandPool cp; VKCHECK(vkCreateCommandPool(dev,&cpi,0,&cp));
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VKCHECK(vkAllocateCommandBuffers(dev,&cbai,&cmd));
    struct { uint32_t n; float k; } push={ (uint32_t)NELEM, 0.5f };
    uint32_t groups=65535;
    VkCommandBufferBeginInfo cbi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VKCHECK(vkBeginCommandBuffer(cmd,&cbi));
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,0);
    vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&push);
    vkCmdDispatch(cmd,groups,1,1);
    VKCHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHECK(vkCreateFence(dev,&fci,0,&fence));

    printf("# vk_membw NELEM=%ld (%.0f MB/buf) dur=%.1fs cpu_threads=%d\n",NELEM,(double)sz/1e6,dur,g_nt);
    gpu_stream(0.5); /* warmup */
    double g_solo=gpu_stream(dur);  printf("A  GPU solo            : %6.2f GB/s\n",g_solo);
    double c_solo=cpu_solo(dur);    printf("B  CPU solo (%d thr)    : %6.2f GB/s\n",g_nt,c_solo);
    cpu_start(); double g_c=gpu_stream(dur); double c_c=cpu_stop();
    printf("C  CONCURRENT          : CPU %6.2f + GPU %6.2f = %6.2f GB/s aggregate\n",c_c,g_c,c_c+g_c);
    double additivity=(c_c+g_c)/(c_solo+g_solo), best=c_solo>g_solo?c_solo:g_solo;
    printf("\nadditivity = %.2f   headroom over best solo = %.2fx\nVERDICT: %s\n",
           additivity,(c_c+g_c)/best,
           (c_c+g_c)>best*1.10?"GPU ADDS BANDWIDTH":"SHARED CEILING — GPU contends, no added bandwidth");
    return 0;
}
