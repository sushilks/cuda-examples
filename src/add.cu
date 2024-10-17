#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>

__global__ void add(int64_t n, float *x, float *y)
{

    // printf("blockIdx = %d, %d, dim = %d, %d thread x,y = %d, %d gridDim = %d %d\n",
    //        blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y,
    //        gridDim.x, gridDim.y);
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x; // 0..8388608 * 256 + 0..256
    // int64_t stride = blockDim.x * gridDim.x;               // 256 * 8388608
    y[index] = x[index] + y[index];
    // for (int64_t i = index; i < n; i += stride)
    //     y[i] = x[i] + y[i];
}

int main(void)
{
    int64_t N = (int64_t)1 << 31;
    int64_t Nbytes = N * sizeof(float);
    float *x, *y, *d_x, *d_y;
    x = (float *)malloc(Nbytes);
    y = (float *)malloc(Nbytes);
    printf("Metric dimension : %ld  or in bytes: %.2fMB\n", N, 1.0 * Nbytes / 1e6);
    cudaMalloc(&d_x, Nbytes);
    cudaMalloc(&d_y, Nbytes);

    for (int64_t i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    {
        auto st = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_x, x, Nbytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, Nbytes, cudaMemcpyHostToDevice);
        auto ed = std::chrono::high_resolution_clock::now();
        printf("Done with copy to Device. BW: %.3fGBps\n",
               1.0 * Nbytes / std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st).count());
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
           numBlocks, blockSize, 1.0 * N / std::chrono::duration_cast<std::chrono::nanoseconds>(edCU - stCU).count());
    {
        auto st = std::chrono::high_resolution_clock::now();
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        auto ed = std::chrono::high_resolution_clock::now();
        printf("Done with copy to Host. BW: %.3fGBps\n",
               1.0 * Nbytes / std::chrono::duration_cast<std::chrono::nanoseconds>(ed - st).count());
    }
    float maxError = 0.0f;
    for (int64_t i = 0; i < N; ++i)
        maxError = max(maxError, abs(y[i] - 4.0f));
    printf("Done with check. MaxError: %f\n", maxError);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
