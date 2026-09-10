#ifndef _UTILS_H_
#define _UTILS_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  // 需要 exit()

#define CHECK(call)                                                                                                    \
    {                                                                                                                  \
        const cudaError_t error = call;                                                                                \
        if (error != cudaSuccess) {                                                                                    \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                                                               \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));                                           \
            exit(1);                                                                                                   \
        }                                                                                                              \
    }

/**
 * 精度测试 - 标量
 */
void checkResult(float hostRef, float gpuRef, const int N);

/**
 * 精度测试 - 数组
 */
void checkResult(float *hostRef, float *gpuRef, const int N);

/**
 * 整数除法向上取整
 */
int cdiv(int x, int y);

/**
 * 打印设备信息
 */
void print_prop(cudaDeviceProp prop);

/**
 * 获取CPU时间（秒）
 */
double cpuSecond();

/**
 * 初始化浮点数组
 */
void initialData(float *ip, int size);

/**
 * 初始化整数数组
 */
void initialData_int(int *ip, int size);

/**
 * 打印矩阵
 */
void printMatrix(float *C, const int nx, const int ny);

/**
 * 初始化设备
 */
void initDevice(int devNum);

#endif // _UTILS_H_