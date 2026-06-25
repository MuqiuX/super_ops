#ifndef _UTILS_H_
#define _UTILS_H_
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

void checkResult(float * hostRef,float * gpuRef,const int N);

int cdiv(int x, int y);

#endif