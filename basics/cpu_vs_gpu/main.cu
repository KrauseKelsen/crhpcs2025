#include <cuda_runtime.h>
#include <stdio.h>

#include "matrix.h"

__global__ void transpose_matrix_naive_gpu(int n, int m, const float *origin, float *result)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < n && j < m)
  {
    result[j * m + i] = origin[i * n +j];
  }
}

void run_transpose_naive_gpu(const int n, const int m, const float *host_origin)
{
  float *dev_origin, *dev_result;
  cudaMalloc(&dev_origin, n * m * sizeof(float));
  cudaMalloc(&dev_result, n * m * sizeof(float));

  cudaMemcpy(dev_origin, host_origin, M * N * sizeof(float), cudaMemcpyHostToDevice);
}

int main(int argc, char **argv)
{
  printf("Hello, world!\n");
  int n = atol(argv[1]);
  int m = atol(argv[2]);
  float **matA = alloc_matrix(n, m);
  float **matB = alloc_matrix(m, n);
  init_matrix(n, m, 2, matA);
  //transpose_matrix_block(n, m, matA, matB);
  free_matrix(n, m, matA);
  free_matrix(m, n, matB);
  return 0;
}
