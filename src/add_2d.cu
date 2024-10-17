#include <assert.h>
#include <stdio.h>
// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <string>

__global__ void MatrixMult(float *C, float *A, float *B, int wA, int hA, int wB,
                           int hB, bool verbose) {
  // // printf("blockIdx = %d, %d, dim = %d, %d thread x,y = %d, %d gridDim = %d
  // %d\n",
  // //        blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x,
  // threadIdx.y,
  // //        gridDim.x, gridDim.y);
  // int64_t index = blockIdx.x * blockDim.x + threadIdx.x; // 0..8388608 * 256
  // + 0..256
  // // int64_t stride = blockDim.x * gridDim.x;               // 256 * 8388608
  // y[index] = x[index] + y[index];
  // // for (int64_t i = index; i < n; i += stride)
  // //     y[i] = x[i] + y[i];

  // each block is 32x32
  // with 1024 threads that parallally take care of each element in the block
  int blockX = blockIdx.x * 32;
  int blockY = blockIdx.y * 32;
  int elemX = blockX + threadIdx.x;
  int elemY = blockY + threadIdx.y;
  if (elemX >= wB || elemY >= hA) return;
  if (verbose) {
    printf("block = %d, %d, thread x,y = %d, %d elem = %d  %d\n", blockIdx.x,
           blockIdx.y, threadIdx.x, threadIdx.y, elemX, elemY);
  }
  float s = 0;
  for (int x = 0; x < wA; ++x) {
    s += A[elemY * wA + x] * B[x * wB + elemX];
  }
  // c.shape = (width = wB, height =hA )
  C[elemY * wB + elemX] = s;
}

void printMetric(const char *msg, float *m, int wM, int hM) {
  printf("%s", msg);
  int idx = 0;
  for (int h = 0; h < hM; ++h) {
    for (int w = 0; w < wM; ++w) {
      printf("%4.1f ", m[idx]);
      idx++;
    }
    printf("\n");
  }
  printf("\n");
}
int main(void) {
  int block_size = 32;
  int widthMult = 1000;
  int heightMult = 1000;

  int wA = block_size * widthMult;
  int hA = block_size * heightMult;
  int wB = block_size * widthMult / 2;
  //  wA = 4;
  //  hA = 5;
  int hB = wA;
  int wC = wB;
  int hC = hA;

  dim3 matAdim(wA, hA, 1);
  dim3 matBdim(wB, hB, 1);
  dim3 matCdim(wC, hC, 1);
  uint64_t szMatA = matAdim.x * matAdim.y * sizeof(float);
  uint64_t szMatB = matBdim.x * matBdim.y * sizeof(float);
  uint64_t szMatC = matCdim.x * matCdim.y * sizeof(float);
  printf("Metrix A[%d,%d], B[%d %d]\n", wA, hA, wB, hB);
  printf("Metric Size a=[%.3fMB] b=[%.3fMB] c=[%.3fMB]\n", szMatA / 1e6,
         szMatB / 1e6, szMatC / 1e6);
  float *hostA, *hostB, *hostC;
  assert(cudaMallocHost(&hostA, size_t(szMatA)) == cudaSuccess);
  assert(cudaMallocHost(&hostB, szMatB) == cudaSuccess);
  assert(cudaMallocHost(&hostC, szMatC) == cudaSuccess);
  for (uint64_t i = 0; i < matAdim.x * matAdim.y;
       i++) {  // 524288 == 0x80000 // 819200 = 0xc8000
    hostA[i] = 0.1 * (std::rand() % 100);
  }
  for (uint64_t i = 0; i < matBdim.x * matBdim.y; i++) {
    hostB[i] = 0.1 * (std::rand() % 100);
  }
  if (false) {
    printMetric("Metrics A\n", hostA, wA, hA);
    printMetric("Metrics B\n", hostB, wB, hB);
  }
  float *devA, *devB, *devC;
  cudaMalloc(reinterpret_cast<void **>(&devA), szMatA);
  cudaMalloc(reinterpret_cast<void **>(&devB), szMatB);
  cudaMalloc(reinterpret_cast<void **>(&devC), szMatC);

  // copy metrics from host to dev
  {
    auto st = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(devA, hostA, szMatA, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(devB, hostB, szMatB, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto ed = std::chrono::high_resolution_clock::now();
    printf("Done with metrix copy to Device. Size: %.3fMB BW: %.3fGBps\n",
           1.0 * (szMatA + szMatB) / 1e6,
           1.0 * (szMatA + szMatB) /
               std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st)
                   .count());
  }
  dim3 thread(32, 32);
  dim3 gridD(matBdim.x / thread.x, matAdim.y / thread.y);
  if (gridD.x == 0) {
    gridD.x = 1;
  }
  if (gridD.y == 0) {
    gridD.y = 1;
  }
  auto stCU = std::chrono::high_resolution_clock::now();

  MatrixMult<<<gridD, thread>>>(devC, devA, devB, matAdim.x, matAdim.y,
                                matBdim.x, matBdim.y, false);
  cudaDeviceSynchronize();
  auto edCU = std::chrono::high_resolution_clock::now();
  float ops = 1.0 * wA * wB * hA;
  printf("Done with mul on gpu. OP: %.3fGOPS perf: %.3fGFLOPS\n", ops / 1e9,
         1.0 * ops /
             std::chrono::duration_cast<std::chrono::nanoseconds>(edCU - stCU)
                 .count());
  {
    auto st = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(hostC, devC, szMatC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto ed = std::chrono::high_resolution_clock::now();
    printf(
        "Done with metrix result copy from Device. Size: %.3fMB BW: %.3fGBps\n",
        1.0 * (szMatC) / 1e6,
        1.0 * (szMatC) /
            std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st)
                .count());
  }
  if (false) {
    printMetric("Metrics C\n", hostC, wC, hC);
  }

  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  // -------
  // -------
  // -------
  /*
  float *x, *y, *d_x, *d_y;
  dim3 dimsA(2, 2, 2);
  x = (float *)malloc(Nbytes);
  y = (float *)malloc(Nbytes);
  printf("Metric dimension : %ld  or in bytes: %.2fMB\n", N,
         1.0 * Nbytes / 1e6);
  cudaMalloc(&d_x, Nbytes);
  cudaMalloc(&d_y, Nbytes);

  for (int64_t i = 0; i < N; ++i) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  {
    auto st = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_x, x, Nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, Nbytes, cudaMemcpyHostToDevice);
    auto ed = std::chrono::high_resolution_clock::now();
    printf("Done with copy to Device. BW: %.3fGBps\n",
           1.0 * Nbytes /
               std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st)
                   .count());
  }
  int64_t blockSize = 256;
  int64_t numBlocks = (N + blockSize - 1) / blockSize;

  auto stCU = std::chrono::high_resolution_clock::now();
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);
  cudaDeviceSynchronize();
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);
  cudaDeviceSynchronize();
  auto edCU = std::chrono::high_resolution_clock::now();

  printf("Done with add on gpu. [NumBlock:%ld, BlockSize:%ld] OP: %.4fGOps\n",
         numBlocks, blockSize,
         1.0 * N /
             std::chrono::duration_cast<std::chrono::nanoseconds>(edCU - stCU)
                 .count());
  {
    auto st = std::chrono::high_resolution_clock::now();
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    auto ed = std::chrono::high_resolution_clock::now();
    printf("Done with copy to Host. BW: %.3fGBps\n",
           1.0 * Nbytes /
               std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st)
                   .count());
  }
  float maxError = 0.0f;
  for (int64_t i = 0; i < N; ++i) maxError = max(maxError, abs(y[i] - 4.0f));
  printf("Done with check. MaxError: %f\n", maxError);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  */
}
