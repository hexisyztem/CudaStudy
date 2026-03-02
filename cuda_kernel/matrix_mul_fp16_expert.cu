#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h> 

using namespace nvcuda;

// Block 级分块大小
const int BM = 64;
const int BN = 64;
const int BK = 16;

// Warp 级分块大小 (每个 Block 有 2x2 个 Warp)
const int WM = 32;
const int WN = 32;

#define checkCublasErrors(val) check_cublas( (val), #val, __FILE__, __LINE__ )
inline void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
        exit(EXIT_FAILURE);
    }
}

bool verify_result(const half* ref, const half* res, int n, const char* name) {
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::abs(__half2float(ref[i]) - __half2float(res[i])));
    }
    std::cout << "[" << name << "] Max Error: " << max_error;
    if (max_error > 0.5f) { std::cout << " -> FAILED" << std::endl; return false; }
    else { std::cout << " -> PASSED" << std::endl; return true; }
}

__global__ void MatrixMulExpert(const half* A, const half* B, half* C, int width) {
    // 1. 声明共享内存 (使用 Padding 消除 Bank Conflict)
    __shared__ half s_A[BM][BK + 8];
    __shared__ half s_B[BK][BN + 8];

    // 获取 Warp 索引
    int warp_id = threadIdx.y; 
    int warp_m = warp_id / 2; // 0 or 1
    int warp_n = warp_id % 2; // 0 or 1

    // 2. 声明 WMMA 片段 (Fragments)
    // 每个 Warp 负责 32x32，即 2x2 = 4 个 16x16 片段
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) wmma::fill_fragment(c_frag[i][j], 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    int tid = threadIdx.y * 32 + threadIdx.x; // 0..127

    // 3. K 维度大循环
    for (int k_step = 0; k_step < width; k_step += BK) {
        
        // --- 协作搬运 A: 64x16 从 Global -> Shared ---
        int a_row = tid / 2;     // 0..63
        int a_col = (tid % 2) * 8; // 0, 8
        reinterpret_cast<float4*>(&s_A[a_row][a_col])[0] = 
            reinterpret_cast<const float4*>(&A[(blockIdx.y * BM + a_row) * width + (k_step + a_col)])[0];

        // --- 协作搬运 B: 16x64 从 Global -> Shared ---
        int b_row = tid / 8;     // 0..15
        int b_col = (tid % 8) * 8; // 0..56
        reinterpret_cast<float4*>(&s_B[b_row][b_col])[0] = 
            reinterpret_cast<const float4*>(&B[(k_step + b_row) * width + (blockIdx.x * BN + b_col)])[0];

        __syncthreads();

        // --- 计算: 每个 Warp 内部进行子块迭代 ---
        for (int i = 0; i < 2; i++) {
            // 加载 Warp 负责的 A 片段
            wmma::load_matrix_sync(a_frag, &s_A[warp_m * WM + i * 16][0], BK + 8);
            for (int j = 0; j < 2; j++) {
                // 加载 Warp 负责的 B 片段
                wmma::load_matrix_sync(b_frag, &s_B[0][warp_n * WN + j * 16], BN + 8);
                // 累加
                wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
            }
        }
        
        __syncthreads();
    }

    // 4. 写回结果: 从 Fragment -> Shared -> Global
    __shared__ float s_C[BM][BN + 8];
    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            wmma::store_matrix_sync(&s_C[warp_m * WM + i * 16][warp_n * WN + j * 16], c_frag[i][j], BN + 8, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // 协作写回 Global (每个线程搬运 32 个元素)
    for (int i = 0; i < 32; ++i) {
        int c_tid = tid * 32 + i;
        int r = c_tid / BN;
        int c = c_tid % BN;
        C[(blockIdx.y * BM + r) * width + (blockIdx.x * BN + c)] = __float2half(s_C[r][c]);
    }
}

int main(void) {
    int width = 1024; int numElements = width * width;
    size_t size = numElements * sizeof(half);
    std::cout << "Expert Hierarchical Matrix multiplication" << std::endl;

    std::vector<half> h_A(numElements), h_B(numElements), h_Ref(numElements), h_Res(numElements);
    for (int i = 0; i < numElements; ++i) { 
        h_A[i] = __float2half(rand() / (float)RAND_MAX); 
        h_B[i] = __float2half(rand() / (float)RAND_MAX); 
    }

    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size); cudaMalloc((void**)&d_B, size); cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    cublasHandle_t handle; cublasCreate(&handle);
    float alpha = 1.0f; float beta = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, CUDA_R_16F, width, d_A, CUDA_R_16F, width, &beta, d_C, CUDA_R_16F, width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    timer.Start();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, CUDA_R_16F, width, d_A, CUDA_R_16F, width, &beta, d_C, CUDA_R_16F, width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    timer.Stop();
    std::cout << "cuBLAS GemmEx: " << timer.Elapsed() << " ms" << std::endl;
    cudaMemcpy(h_Ref.data(), d_C, size, cudaMemcpyDeviceToHost);

    cudaMemset(d_C, 0, size);
    dim3 block(32, 4); // 4 warps
    dim3 grid(width / BN, width / BM);
    
    timer.Start(); MatrixMulExpert<<<grid, block>>>(d_A, d_B, d_C, width); timer.Stop();
    std::cout << "Expert WMMA (Hierarchical): " << timer.Elapsed() << " ms" << std::endl;
    cudaMemcpy(h_Res.data(), d_C, size, cudaMemcpyDeviceToHost);

    verify_result(h_Ref.data(), h_Res.data(), numElements, "Expert");

    cublasDestroy(handle); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}