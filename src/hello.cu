#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main(int argc, char **argv) {
    printf("Hello World from CPU!\n");
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}
