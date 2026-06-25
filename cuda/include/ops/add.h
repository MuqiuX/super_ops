#ifndef _ADD_H_
#define _ADD_H_

__global__ void add(float *input, float *other, float *out, int M, int N);

void add_r(float *input, float *other, float *out, const int size);

#endif