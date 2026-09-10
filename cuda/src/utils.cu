#include "utils.h"
#include <cmath>
#include <time.h>
#include <sys/time.h>

void checkResult(float *ref, float *res, const int N) {
    double epsilon = 1.0E-8;
    for (int i = 0; i < N; i++) {
        if (abs(ref[i] - res[i]) > epsilon) {
            printf("Results don\'t match!\n");
            printf("%f(ref[%d] )!= %f(res[%d])\n", ref[i], i, res[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}


void checkResult(float ref, float res, const int N) {
    double epsilon = 1.0E-8;
    if (abs(ref - res) > epsilon) {
        printf("Results don\'t match!\n");
        printf("%f(ref )!= %f(res)\n", ref, res);
        return;
    }
    printf("Check result success!\n");
}


int cdiv(int x, int y) { 
    return (x + y - 1) / y; 
}


void print_prop(cudaDeviceProp prop) {
    printf("========================================\n");
    printf("CUDA 设备属性 (Device Properties)\n");
    printf("========================================\n\n");
    
    // ==================== 核心计算能力 ====================
    printf("【核心计算能力】Compute Capability\n");
    printf("  %-40s: %d.%d\n", 
           "major / 主计算能力版本", prop.major, prop.minor);
    printf("  %-40s: %d\n", 
           "multiProcessorCount / 流多处理器(SM)数量", prop.multiProcessorCount);
    printf("  %-40s: %d\n", 
           "maxThreadsPerMultiProcessor / 每SM最大常驻线程数", 
           prop.maxThreadsPerMultiProcessor);
    printf("  %-40s: %d\n", 
           "maxBlocksPerMultiProcessor / 每SM最大常驻Block数", 
           prop.maxBlocksPerMultiProcessor);
    printf("  %-40s: %s\n", 
           "integrated / 是否为集成GPU", 
           prop.integrated ? "是 (Yes)" : "否 (No)");
    printf("  %-40s: %s\n\n", 
           "ECCEnabled / ECC纠错是否启用", 
           prop.ECCEnabled ? "是 (Yes)" : "否 (No)");
    
    // ==================== 内存配置 ====================
    printf("【内存配置】Memory Configuration\n");
    printf("  %-40s: %.2f GB (%.2f MB, %zu bytes)\n", 
           "totalGlobalMem / 全局内存总大小", 
           prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0,
           prop.totalGlobalMem / 1024.0 / 1024.0,
           prop.totalGlobalMem);
    printf("  %-40s: %.2f KB (%zu bytes)\n", 
           "sharedMemPerBlock / 每Block共享内存", 
           prop.sharedMemPerBlock / 1024.0,
           prop.sharedMemPerBlock);
    printf("  %-40s: %.2f KB (%zu bytes)\n", 
           "sharedMemPerMultiprocessor / 每SM共享内存总量", 
           prop.sharedMemPerMultiprocessor / 1024.0,
           prop.sharedMemPerMultiprocessor);
    printf("  %-40s: %.2f MB (%d bytes)\n", 
           "l2CacheSize / L2缓存大小", 
           prop.l2CacheSize / 1024.0 / 1024.0,
           prop.l2CacheSize);
    printf("  %-40s: %d bits\n", 
           "memoryBusWidth / 显存位宽", prop.memoryBusWidth);
    printf("  %-40s: %.2f KB (%zu bytes)\n", 
           "totalConstMem / 常量内存大小", 
           prop.totalConstMem / 1024.0,
           prop.totalConstMem);
    printf("  %-40s: %zu bytes\n", 
           "memPitch / 最大内存复制Pitch", prop.memPitch);
    printf("  %-40s: %s\n\n", 
           "unifiedAddressing / 统一寻址支持", 
           prop.unifiedAddressing ? "支持 (Yes)" : "不支持 (No)");
    
    // ==================== 线程与执行配置 ====================
    printf("【线程与执行配置】Thread & Execution Configuration\n");
    printf("  %-40s: %d\n", 
           "maxThreadsPerBlock / 每Block最大线程数", 
           prop.maxThreadsPerBlock);
    printf("  %-40s: (%d, %d, %d)\n", 
           "maxThreadsDim / Block各维度最大尺寸", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  %-40s: (%d, %d, %d)\n", 
           "maxGridSize / Grid各维度最大尺寸", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  %-40s: %d (固定为32)\n", 
           "warpSize / Warp大小(线程数)", prop.warpSize);
    printf("  %-40s: %d (32-bit registers)\n", 
           "regsPerBlock / 每Block可用寄存器数", prop.regsPerBlock);
    printf("  %-40s: %d (32-bit registers)\n", 
           "regsPerMultiprocessor / 每SM可用寄存器数", 
           prop.regsPerMultiprocessor);
    printf("  %-40s: %s\n", 
           "concurrentKernels / 并发Kernel执行支持", 
           prop.concurrentKernels ? "支持 (Yes)" : "不支持 (No)");
    printf("  %-40s: %d\n", 
           "asyncEngineCount / 异步引擎数量", prop.asyncEngineCount);
    printf("  %-40s: %s\n", 
           "cooperativeLaunch / 协作式Kernel启动支持", 
           prop.cooperativeLaunch ? "支持 (Yes)" : "不支持 (No)");
    printf("  %-40s: %s\n", 
           "computePreemptionSupported / 计算抢占支持", 
           prop.computePreemptionSupported ? "支持 (Yes)" : "不支持 (No)");
    
    // 条件编译，只在CUDA 11+支持
    #ifdef CUDA_VERSION
    #if CUDA_VERSION >= 11000
    printf("  %-40s: %s\n", 
           "clusterLaunch / Cluster启动支持", 
           prop.clusterLaunch ? "支持 (Yes)" : "不支持 (No)");
    #endif
    #endif
    
    printf("  %-40s: %s\n\n", 
           "streamPrioritiesSupported / Stream优先级支持", 
           prop.streamPrioritiesSupported ? "支持 (Yes)" : "不支持 (No)");
    
    // ==================== 设备信息 ====================
    printf("【设备信息】Device Information\n");
    printf("  %-40s: %s\n", 
           "name / 设备名称", prop.name);
    
    printf("  %-40s: %02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n", 
           "uuid / 唯一标识符", 
           prop.uuid.bytes[0], prop.uuid.bytes[1], 
           prop.uuid.bytes[2], prop.uuid.bytes[3],
           prop.uuid.bytes[4], prop.uuid.bytes[5],
           prop.uuid.bytes[6], prop.uuid.bytes[7],
           prop.uuid.bytes[8], prop.uuid.bytes[9],
           prop.uuid.bytes[10], prop.uuid.bytes[11],
           prop.uuid.bytes[12], prop.uuid.bytes[13],
           prop.uuid.bytes[14], prop.uuid.bytes[15]);
    
    printf("  %-40s: Bus:%d, Device:%d, Domain:%d\n", 
           "pciBusID / PCI总线ID", 
           prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
    
    printf("  %-40s: %s\n", 
           "tccDriver / TCC驱动模式", 
           prop.tccDriver ? "是 (Yes)" : "否 (No)");
    printf("========================================\n");
}


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}


void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

void initialData_int(int *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = int(rand() & 0xff);
    }
}


void printMatrix(float *C, const int nx, const int ny) {
    float *ic = C;
    printf("Matrix<%d,%d>:\n", ny, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f ", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
}


void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}