#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <map>
#include <string>
#include <curand_kernel.h>

void cpu_matmul_row_major(half *A, half *B, float *C, int M, int N, int K)
{
  // CPU matmul
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float sum = 0.0f;
      for (int k = 0; k < K; k++)
      {
        sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
      }
      C[i * N + j] = sum;
    }
  }
}

void compare_matrix(float *h_C, float *CPU_C, int length)
{
  const int verbose = 0;

  for (int i = 0; i < length; i++)
  {
    if (CPU_C[i] != h_C[i] && (verbose == 0))
      printf("incorrect: CPU_C : %f, C: %f\n", CPU_C[i], h_C[i]);

    if (verbose)
      printf("incorrect: CPU_C : %f, C: %f\n", CPU_C[i], h_C[i]);
  }
}

__global__ void fill_B(half *ptr, int offset)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  int base = offset * y + 4 * x;

  ptr[base] = 1.0f * (x % 2);
  ptr[base + 1] = 1.0f * (y % 3);
  ptr[base + 2] = 1.0f * (x % 4);
  ptr[base + 3] = 1.0f * (y % 5);
}

__global__ void fill_A(half *ptr, int offset)
{
  const char *pattern = "0110";

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  int base = offset * y + 4 * x;

  for (int i = 0; i < 4; i++)
  {
    if (pattern[i] == '1')
      ptr[base + i] = 1.0f;
    else
      ptr[base + i] = 0.0f;
  }
}

__device__ __forceinline__ unsigned int merge_half2_to_b32(half2 values)
{

  //===========merge two half into one .b32 register====================

  unsigned int merged_value;
  unsigned short *value_ptr = reinterpret_cast<unsigned short *>(&values);
  unsigned int upper_half = static_cast<unsigned int>(value_ptr[0]);
  unsigned int lower_half = static_cast<unsigned int>(value_ptr[1]);

  merged_value = (upper_half << 16) | lower_half;
  return merged_value;
}

__global__ void compress_A(half *ptr, int M, int K, unsigned int *compressed_A)
{
  int counter = 0;
  half2 tile = {0, 0};

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  int base = K * y + 4 * x;

  for (int i = 0; i < 4; i++)
  {
    if (ptr[base + i])
    {
      if (counter == 0)
      {
        tile.x = ptr[base + i];
        counter++;
      }
      else if (counter == 1)
      {
        tile.y = ptr[base + i];
        counter++;
      }
      else
      {
        printf("2 : 4 sparsity violated!!!\n");
        return;
      }
    }
  }

  compressed_A[K * y / 4 + x] = merge_half2_to_b32(tile);
}

__global__ void compress_B(half *ptr, int K, int N, unsigned *compressed_B)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  half2 values;
  values.x = ptr[2 * y * N + x];
  values.y = ptr[(2 * y + 1) * N + x];
  compressed_B[y * N + x] = merge_half2_to_b32(values);
}

__global__ void kernel(int M, int N, int K,
                       unsigned int *A, unsigned *B, float *C,
                       unsigned metadata)
{
  using namespace nvcuda;
  //===================thread information================
  int x = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  int laneid = threadIdx.x;
  int groupID = laneid >> 2;
  int threadID_in_group = laneid % 4;
  //====================do accumulation in shared memory=========================
  __shared__ float tile[16 * 8];

  tile[groupID * 8 + threadID_in_group * 2] = 0;
  tile[groupID * 8 + threadID_in_group * 2 + 1] = 0;
  tile[(groupID + 8) * 8 + threadID_in_group * 2] = 0;
  tile[(groupID + 8) * 8 + threadID_in_group * 2 + 1] = 0;
  //==========================variable declaration===========================
  int A_row = x * 16,
      B_row,
      A_col,
      B_col = y * 8,
      A_base, B_base, C_base; // the start address of each tile

  C_base = B_col + A_row * N;

  unsigned A1, A2, B1, B2;
  //====================================================================
  for (int k = 0; k < K; k += 16)
  { // iterate the reduction axis
    A_col = k;
    B_row = k;

    A_base = A_col / 4 + A_row * K / 4;
    B_base = B_col + B_row * N / 2;

    A1 = A[A_base + groupID * K / 4 + threadID_in_group];
    A2 = A[A_base + (groupID + 8) * K / 4 + threadID_in_group];
    B1 = B[B_base + threadID_in_group * N + groupID];
    B2 = B[B_base + (threadID_in_group + 4) * N + groupID];

    asm volatile(
        "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6,  %7},"
        "{%8, %9, %10, %11}, %12, 0x0;"
        : "=f"(tile[groupID * 8 + threadID_in_group * 2]),
          "=f"(tile[groupID * 8 + threadID_in_group * 2 + 1]),
          "=f"(tile[(groupID + 8) * 8 + threadID_in_group * 2]),
          "=f"(tile[(groupID + 8) * 8 + threadID_in_group * 2 + 1])
        : "r"(A1), "r"(A2),
          "r"(B1), "r"(B2),
          "f"(tile[groupID * 8 + threadID_in_group * 2]),
          "f"(tile[groupID * 8 + threadID_in_group * 2 + 1]),
          "f"(tile[(groupID + 8) * 8 + threadID_in_group * 2]),
          "f"(tile[(groupID + 8) * 8 + threadID_in_group * 2 + 1]),
          "r"(metadata));
  }

  C[C_base + groupID * N + threadID_in_group * 2] =
      tile[groupID * 8 + threadID_in_group * 2];
  C[C_base + groupID * N + threadID_in_group * 2 + 1] =
      tile[groupID * 8 + threadID_in_group * 2 + 1];
  C[C_base + (groupID + 8) * N + threadID_in_group * 2] =
      tile[(groupID + 8) * 8 + threadID_in_group * 2];
  C[C_base + (groupID + 8) * N + threadID_in_group * 2 + 1] =
      tile[(groupID + 8) * 8 + threadID_in_group * 2 + 1];
}

void matmul(int m, int n, int k,
            unsigned *A, unsigned *B, float *C,
            unsigned metadata)
{
  kernel<<<dim3(m / 16, n / 8), dim3(32, 1)>>>(m, n, k, A, B, C, metadata);
  cudaDeviceSynchronize();
}

int main(int argc, char *argv[])
{
  assert(argc == 5);
  //===========================choose pattern=============================
  // ! notice 2 : 4 sparsity is enforced for mma.sp
  std::string pattern_list[6] = {"0110", "1010", "1001", "0110", "0101", "0011"};
  std::string pattern = pattern_list[0];
  //=============================enumerate metadata================================
  // ! do not change this metadata map !
  std::map<std::string, unsigned> metaMap;
  metaMap["1100"] = 0x44444444;
  metaMap["1010"] = 0x77777777;
  metaMap["1001"] = 0x66666666;
  metaMap["0110"] = 0x33333333;
  metaMap["0101"] = 0x22222222;
  metaMap["0011"] = 0xEEEEEEEE;
  //=============================matrix sizes================================
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int K = atoi(argv[3]);

  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 64 == 0);
  //=============================allocate host================================
  half *h_A, *h_B;
  float *h_C, *CPU_C;

  h_A = (half *)malloc(M * K * sizeof(half));
  h_B = (half *)malloc(K * N * sizeof(half));
  h_C = (float *)malloc(M * N * sizeof(float));
  CPU_C = (float *)malloc(M * N * sizeof(float));
  //==============================allocate device========================================
  half *d_A, *d_B;
  float *d_C;
  unsigned *compressed_A, *compressed_B;

  cudaMalloc((void **)&d_A, M * K * sizeof(half));
  cudaMalloc((void **)&d_B, K * N * sizeof(half));
  cudaMalloc((void **)&d_C, M * N * sizeof(float));
  cudaMalloc((void **)&compressed_A, M * K / 4 * sizeof(unsigned int));
  cudaMalloc((void **)&compressed_B, K / 2 * N * sizeof(unsigned int));
  //==============================initialize matrices======================================
  fill_A<<<dim3(K / (16 * 4), M / 16), dim3(16, 16)>>>(d_A, K);
  fill_B<<<dim3(N / (16 * 4), K / 16), dim3(16, 16)>>>(d_B, N);
  //==============================compress A, B========================================
  int iter = atoi(argv[4]);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  float milliseconds = 0;

  cudaEventRecord(start);

  for (int i = 0; i < iter; i++)
  {
    compress_A<<<dim3(K / (16 * 4), M / 16), dim3(16, 16)>>>(d_A, M, K, compressed_A);
    compress_B<<<dim3(N / 16, K / (16 * 2)), dim3(16, 16)>>>(d_B, K, N, compressed_B);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&milliseconds, start, end);
  std::cout << "compression time: "
            << milliseconds / iter
            << " ms"
            << " averaged accross "
            << iter
            << " iterations "
            << std::endl;
  //===============================profile matmul===============================
  cudaEventRecord(start);

  for (int i = 0; i < iter; i++)
  {
    matmul(M, N, K, compressed_A, compressed_B, d_C, metaMap[pattern]);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, end);
  std::cout << "matmul time: "
            << milliseconds / iter
            << " ms"
            << " averaged accross "
            << iter
            << " iterations "
            << std::endl;
  //===============================checking results================================
  if (1)
  {
    printf("checking\n");
    cudaMemcpy(h_A, d_A, M * K * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, K * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cpu_matmul_row_major(h_A, h_B, CPU_C, M, N, K);
    compare_matrix(h_C, CPU_C, M * N);
  }
  //============================free allocations==================================
  free(h_A);
  free(h_B);
  free(h_C);
  free(CPU_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(compressed_A);
  cudaFree(compressed_B);
  //===================================return==========================
  printf("COMPLETED!\n");
  return 0;
}