#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>

// nvcc -O3 -lineinfo matrix_mul.cu -o matrix_mul -lcublas -lcudart && ./matrix_mul
// sudo /usr/local/NVIDIA-Nsight-Compute-2025.4/ncu --set full --kernel-name MatrixMulVectorized ./matrix_mul

#define checkCublasErrors(val) check_cublas( (val), #val, __FILE__, __LINE__ )
inline void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

bool verify_result(const float* ref, const float* res, int n, const char* name) {
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) max_error = std::max(max_error, std::abs(ref[i] - res[i]));
    std::cout << "[" << name << "] Max Error: " << max_error;
    if (max_error > 1e-3) { std::cout << " -> FAILED" << std::endl; return false; }
    else { std::cout << " -> PASSED" << std::endl; return true; }
}

// 1. Naive Kernel
__global__ void MatrixMulNaive(const float* A, const float* B, float* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= width) return;
    float sum = 0.0f;
    for (int k = 0; k < width; ++k) sum += A[row * width + k] * B[k * width + col];
    C[row * width + col] = sum;
}

// 2. Basic Tiled Kernel
#define TILE_WIDTH 32
__global__ void MatrixMulTiled(const float* A, const float* B, float* C, int width) {
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    float sum = 0.0f;
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (row < width && (m * TILE_WIDTH + tx) < width) s_A[ty][tx] = A[row * width + m * TILE_WIDTH + tx];
        else s_A[ty][tx] = 0.0f;
        if (col < width && (m * TILE_WIDTH + ty) < width) s_B[ty][tx] = B[(m * TILE_WIDTH + ty) * width + col];
        else s_B[ty][tx] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) sum += s_A[ty][k] * s_B[k][tx];
        __syncthreads();
    }
    if (row < width && col < width) C[row * width + col] = sum;
}

// 3. Tiled + float4 Vectorized Load (No Register Tiling)
__global__ void MatrixMulTiledVectorized(const float* A, const float* B, float* C, int width) {
    __shared__ float s_A[32][32];
    __shared__ float s_B[32][32];
    
    int tx = threadIdx.x; int ty = threadIdx.y;
    int tid = ty * 32 + tx; // 0..1023
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    float sum = 0.0f;

    for (int m = 0; m < width; m += 32) {
        // 使用 256 个线程通过 float4 协作加载 s_A (1024 floats)
        if (tid < 256) {
            int r = tid / 8;      // 0..31
            int c = (tid % 8) * 4; // 0, 4, 8...28
            float4 tmp = reinterpret_cast<const float4*>(&A[(blockIdx.y * 32 + r) * width + (m + c)])[0];
            s_A[r][c+0] = tmp.x; s_A[r][c+1] = tmp.y; s_A[r][c+2] = tmp.z; s_A[r][c+3] = tmp.w;
        }
        // 使用另外 256 个线程通过 float4 协作加载 s_B (1024 floats)
        if (tid >= 256 && tid < 512) {
            int t = tid - 256;
            int r = t / 8;
            int c = (t % 8) * 4;
            float4 tmp = reinterpret_cast<const float4*>(&B[(m + r) * width + (blockIdx.x * 32 + c)])[0];
            s_B[r][c+0] = tmp.x; s_B[r][c+1] = tmp.y; s_B[r][c+2] = tmp.z; s_B[r][c+3] = tmp.w;
        }
        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }
    if (row < width && col < width) C[row * width + col] = sum;
}

// 4. Optimized 2D Kernel (8x8 Tiling)
__global__ void MatrixMulOptimized2D(const float* A, const float* B, float* C, int width) {
    const int BM = 128; const int BN = 128; const int BK = 8;
    const int TM = 8;   const int TN = 8;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];
    int tx = threadIdx.x; int ty = threadIdx.y;
    float reg_C[TM][TN] = {0.0f}; float reg_A[TM]; float reg_B[TN];
    int tid = ty * 16 + tx;
    for (int m = 0; m < width; m += BK) {
        for (int i = 0; i < 4; ++i) {
            int load_row = (tid * 4 + i) / BK; int load_col = (tid * 4 + i) % BK;
            s_A[load_row][load_col] = A[(blockIdx.y * BM + load_row) * width + (m + load_col)];
        }
        for (int i = 0; i < 4; ++i) {
            int load_row = (tid * 4 + i) / BN; int load_col = (tid * 4 + i) % BN;
            s_B[load_row][load_col] = B[(m + load_row) * width + (blockIdx.x * BN + load_col)];
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) reg_A[i] = s_A[ty * TM + i][k];
            for (int j = 0; j < TN; ++j) reg_B[j] = s_B[k][tx * TN + j];
            for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) reg_C[i][j] += reg_A[i] * reg_B[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) C[(blockIdx.y * BM + ty * TM + i) * width + (blockIdx.x * BN + tx * TN + j)] = reg_C[i][j];
}

// 5. Full Optimized Kernel (float4 + 8x8 Tiling)
__global__ void MatrixMulVectorized(const float* A, const float* B, float* C, int width) {
    const int BM = 128; const int BN = 128; const int BK = 8;
    const int TM = 8;   const int TN = 8;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int tid = ty * 16 + tx;
    float reg_C[TM][TN] = {0.0f}; float reg_A[TM]; float reg_B[TN];
    for (int m = 0; m < width; m += BK) {
        int a_row = tid / 2; int a_col = (tid % 2) * 4;
        float4 tmp_a = reinterpret_cast<const float4*>(&A[(blockIdx.y * BM + a_row) * width + (m + a_col)])[0];
        s_A[a_row][a_col + 0] = tmp_a.x; s_A[a_row][a_col + 1] = tmp_a.y; s_A[a_row][a_col + 2] = tmp_a.z; s_A[a_row][a_col + 3] = tmp_a.w;
        int b_row = tid / 32; int b_col = (tid % 32) * 4;
        float4 tmp_b = reinterpret_cast<const float4*>(&B[(m + b_row) * width + (blockIdx.x * BN + b_col)])[0];
        s_B[b_row][b_col + 0] = tmp_b.x; s_B[b_row][b_col + 1] = tmp_b.y; s_B[b_row][b_col + 2] = tmp_b.z; s_B[b_row][b_col + 3] = tmp_b.w;
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            for (int i = 0; i < TM; ++i) reg_A[i] = s_A[ty * TM + i][k];
            for (int j = 0; j < TN; ++j) reg_B[j] = s_B[k][tx * TN + j];
            for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) reg_C[i][j] += reg_A[i] * reg_B[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; j += 4) {
            float4 tmp_res;
            tmp_res.x = reg_C[i][j + 0]; tmp_res.y = reg_C[i][j + 1]; tmp_res.z = reg_C[i][j + 2]; tmp_res.w = reg_C[i][j + 3];
            reinterpret_cast<float4*>(&C[(blockIdx.y * BM + ty * TM + i) * width + (blockIdx.x * BN + tx * TN + j)])[0] = tmp_res;
        }
    }
}

int main(void) {
    int width = 1024; int numElements = width * width;
    size_t size = numElements * sizeof(float);
    std::cout << "Matrix multiplication: " << width << "x" << width << std::endl;

    std::vector<float> h_A(numElements), h_B(numElements), h_Ref(numElements);
    std::vector<float> h_Naive(numElements), h_Tiled(numElements), h_TiledVec(numElements), h_Opt2D(numElements), h_Vec(numElements);
    for (int i = 0; i < numElements; ++i) { h_A[i] = rand() / (float)RAND_MAX; h_B[i] = rand() / (float)RAND_MAX; }

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, size)); checkCudaErrors(cudaMalloc((void**)&d_B, size)); checkCudaErrors(cudaMalloc((void**)&d_C, size));
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    GpuTimer timer;
    cublasHandle_t handle; cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    // Benchmarks
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, width, d_A, width, &beta, d_C, width);
    cudaDeviceSynchronize();
    timer.Start(); cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, width, d_A, width, &beta, d_C, width); timer.Stop();
    std::cout << "cuBLAS: " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_Ref.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaMemset(d_C, 0, size);
    timer.Start(); MatrixMulNaive<<<dim3(32, 32), dim3(32, 32)>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Naive: " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_Naive.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaMemset(d_C, 0, size);
    timer.Start(); MatrixMulTiled<<<dim3(32, 32), dim3(32, 32)>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Tiled: " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_Tiled.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaMemset(d_C, 0, size);
    timer.Start(); MatrixMulTiledVectorized<<<dim3(32, 32), dim3(32, 32)>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Tiled Vectorized (float4): " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_TiledVec.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaMemset(d_C, 0, size);
    timer.Start(); MatrixMulOptimized2D<<<dim3(8, 8), dim3(16, 16)>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Optimized 2D: " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_Opt2D.data(), d_C, size, cudaMemcpyDeviceToHost));

    cudaMemset(d_C, 0, size);
    timer.Start(); MatrixMulVectorized<<<dim3(8, 8), dim3(16, 16)>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Full Optimized (float4 + 8x8): " << timer.Elapsed() << " ms" << std::endl;
    checkCudaErrors(cudaMemcpy(h_Vec.data(), d_C, size, cudaMemcpyDeviceToHost));

    std::cout << "\n--- Numerical Verification ---" << std::endl;
    verify_result(h_Ref.data(), h_Naive.data(), numElements, "Naive");
    verify_result(h_Ref.data(), h_Tiled.data(), numElements, "Tiled");
    verify_result(h_Ref.data(), h_TiledVec.data(), numElements, "Tiled Vectorized");
    verify_result(h_Ref.data(), h_Opt2D.data(), numElements, "Optimized 2D");
    verify_result(h_Ref.data(), h_Vec.data(), numElements, "Full Optimized");

    cublasDestroy(handle); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}