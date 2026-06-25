#ifndef _MM_H_
#define _MM_H_

__global__ void mm(float *input, float *other, float *out, int M, int N);

void mm_r(float *input, float *other, float *out, const int size);

#endif