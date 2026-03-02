#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include "cstdio"

// CUDA Kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = dy * gridDim.x * blockDim.x + dx;
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

// CUDA Kernel for vector additon with warp incontinuity
__global__ void vectorAdd_with_warp_incontinuous(const float* A, const float* B, float* C, int numElements) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = dx * gridDim.y * blockDim.y + dy;
    if (idx < numElements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);
    int numElements = width * height;
    std::cout << "Vector addition of " << numElements << " elements" << std::endl;

    // 1. Host data allocation and initialization
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 2. Device data allocation (using helper macro)
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, size));
    checkCudaErrors(cudaMalloc((void**)&d_B, size));
    checkCudaErrors(cudaMalloc((void**)&d_C, size));

    // 3. Data Transfer H2D
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // 4. Kernel Launch with Timing
    GpuTimer timer;
    dim3 threadsPerBlock(32, 32);
    int blocksPerGrid = (numElements + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    std::cout << "Launching Kernel..." << std::endl;
    timer.Start();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    timer.Stop();
    float cost_time = timer.Elapsed();
    std::cout << "Kernel execution time: " << cost_time << " ms" << std::endl;

    GpuTimer timer2;
    timer2.Start();
    vectorAdd_with_warp_incontinuous<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    timer2.Stop();
    float cost_time2 = timer2.Elapsed();
    std::cout << "Kernel execution time (with warp incontinuity): " << cost_time2 << " ms" << std::endl;

    // 同一个warp下读取的数据是否连续，会造成3~4倍的耗时差异，非常显著。
    
    checkCudaErrors(cudaGetLastError());

    // 5. Data Transfer D2H
    checkCudaErrors(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // 6. Verification
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << "!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // 7. Cleanup
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    return 0;
}