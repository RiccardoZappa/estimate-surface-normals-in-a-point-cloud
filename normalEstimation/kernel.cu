#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n");
}

void launchKernel()
{
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}