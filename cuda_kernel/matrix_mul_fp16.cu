#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h> // 引入 Tensor Cores WMMA 库

using namespace nvcuda;

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
    if (max_error > 0.51f) { std::cout << " -> FAILED" << std::endl; return false; }
    else { std::cout << " -> PASSED" << std::endl; return true; }
}

// 终极武器：Tensor Cores WMMA (Warp Matrix Multiply Accumulate)
__global__ void MatrixMulWMMA(const half* A, const half* B, half* C, int width) {
    // 声明 WMMA 的硬件片段 (Fragments)
    // 16x16x16 是 Tensor Cores 支持的标准块大小
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag; // 累加器使用 float

    // 初始化累加器为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 线程块的 Y 维度分配了 4 个 Warp (threadIdx.y = 0..3)
    int warp_id = threadIdx.y; 
    
    // 计算当前 Warp 负责的 C 矩阵的起始行和列
    int row = blockIdx.y * 64 + warp_id * 16;
    int col = blockIdx.x * 16;

    if (row >= width || col >= width) return;

    // K 维度循环
    for (int k = 0; k < width; k += 16) {
        // 同步加载 A 和 B 的 16x16 小块到 Tensor Cores 的专用寄存器中
        wmma::load_matrix_sync(a_frag, A + row * width + k, width);
        wmma::load_matrix_sync(b_frag, B + k * width + col, width);
        
        // 核心指令：调用硬件级 Tensor Cores 进行乘加！
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 将 float 类型的累加器存回共享内存 (因为无法直接存入 half* 指针)
    __shared__ float s_C[4][16 * 16]; 
    wmma::store_matrix_sync(s_C[warp_id], c_frag, 16, wmma::mem_row_major);

    // 每个 Warp 内的 32 个线程协作，将结果从 float 转换为 half 并写回全局内存
    int lane_id = threadIdx.x; // 0..31
    for (int i = 0; i < 8; ++i) { 
        int elem_idx = i * 32 + lane_id; // 每个线程处理 8 个元素，总计 256 个 (16x16)
        int r = elem_idx / 16;
        int c = elem_idx % 16;
        if (row + r < width && col + c < width) {
            C[(row + r) * width + (col + c)] = __float2half(s_C[warp_id][r * 16 + c]);
        }
    }
}

int main(void) {
    int width = 1024; int numElements = width * width;
    size_t size = numElements * sizeof(half);
    std::cout << "Matrix multiplication (Tensor Cores WMMA)" << std::endl;

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

    // --- cuBLAS 基准 ---
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, CUDA_R_16F, width, d_A, CUDA_R_16F, width, &beta, d_C, CUDA_R_16F, width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    timer.Start();
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, &alpha, d_B, CUDA_R_16F, width, d_A, CUDA_R_16F, width, &beta, d_C, CUDA_R_16F, width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    timer.Stop();
    std::cout << "cuBLAS GemmEx: " << timer.Elapsed() << " ms" << std::endl;
    cudaMemcpy(h_Ref.data(), d_C, size, cudaMemcpyDeviceToHost);

    // --- Tensor Cores WMMA ---
    cudaMemset(d_C, 0, size);
    
    // 配置执行网格
    // 线程块：x=32(一个Warp)，y=4(包含4个Warp)
    dim3 block(32, 4);
    // 网格：x 负责列(步长16)，y 负责行(步长64，因为4个Warp每个16)
    dim3 grid(width / 16, width / 64);
    
    timer.Start(); 
    MatrixMulWMMA<<<grid, block>>>(d_A, d_B, d_C, width); 
    timer.Stop();
    std::cout << "Tensor Cores WMMA: " << timer.Elapsed() << " ms" << std::endl;
    cudaMemcpy(h_Res.data(), d_C, size, cudaMemcpyDeviceToHost);

    verify_result(h_Ref.data(), h_Res.data(), numElements, "WMMA");

    cublasDestroy(handle); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
