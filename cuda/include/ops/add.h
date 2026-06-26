#ifndef _ADD_H_
#define _ADD_H_

cudaError_t add(const float *a, const float *b, float *c, int M, int N);

void add_r(float *a, float *b, float *c, const int size);

#endif