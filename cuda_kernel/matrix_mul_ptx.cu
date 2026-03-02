#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdint.h>
#include <cuda_fp16.h>
#include <mma.h> 

using namespace nvcuda;

const int BM = 64; const int BN = 64; const int BK = 32;
const int WM = 32; const int WN = 32;

void verify_result_advanced(const half* ref, const half* res, int n, const char* name) {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float r_f = __half2float(ref[i]);
        float s_f = __half2float(res[i]);
        float abs_err = std::abs(r_f - s_f);
        max_abs_error = std::max(max_abs_error, abs_err);
        if (std::abs(r_f) > 1e-5) max_rel_error = std::max(max_rel_error, abs_err / std::abs(r_f));
    }
    printf("[%s]\n  -> Max Abs Error: %6f\n  -> Max Rel Error: %6f%%\n", name, max_abs_error, max_rel_error * 100.0f);
    if (max_rel_error < 0.001f || max_abs_error < 0.51f) printf("  -> VERIFICATION PASSED\n");
    else printf("  -> VERIFICATION FAILED\n");
}

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* global_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(global_ptr));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); }

__global__ void MatrixMulNaive(const half* A, const half* B, half* C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= width) return;
    float sum = 0.0f; 
    for (int k = 0; k < width; ++k) sum += __half2float(A[row * width + k]) * __half2float(B[k * width + col]);
    C[row * width + col] = __float2half(sum);
}

__global__ void MatrixMulPipelined(const half* A, const half* B, half* C, int width) {
    __shared__ half s_A[2][BM][BK + 8];
    __shared__ half s_B[2][BK][BN + 8];
    int tid = threadIdx.y * 32 + threadIdx.x;
    int warp_id = tid / 32; int warp_m = warp_id / 2; int warp_n = warp_id % 2;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) wmma::fill_fragment(c_frag[i][j], 0.0f);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    int a_row0 = tid / 4; int a_col0 = (tid % 4) * 8;
    int a_row1 = (tid + 128) / 4; int a_col1 = ((tid + 128) % 4) * 8;
    int b_row0 = tid / 8; int b_col0 = (tid % 8) * 8;
    int b_row1 = (tid + 128) / 8; int b_col1 = ((tid + 128) % 8) * 8;
    const half* src_a0 = &A[(blockIdx.y * BM + a_row0) * width + a_col0];
    const half* src_a1 = &A[(blockIdx.y * BM + a_row1) * width + a_col1];
    const half* src_b0 = &B[b_row0 * width + (blockIdx.x * BN + b_col0)];
    const half* src_b1 = &B[b_row1 * width + (blockIdx.x * BN + b_col1)];

    cp_async_16B(&s_A[0][a_row0][a_col0], src_a0);
    cp_async_16B(&s_A[0][a_row1][a_col1], src_a1);
    cp_async_16B(&s_B[0][b_row0][b_col0], src_b0);
    cp_async_16B(&s_B[0][b_row1][b_col1], src_b1);
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    int stage = 0;
    for (int m = 0; m < width; m += BK) {
        int next_stage = stage ^ 1; int next_m = m + BK;
        if (next_m < width) {
            cp_async_16B(&s_A[next_stage][a_row0][a_col0], src_a0 + next_m);
            cp_async_16B(&s_A[next_stage][a_row1][a_col1], src_a1 + next_m);
            cp_async_16B(&s_B[next_stage][b_row0][b_col0], src_b0 + next_m * width);
            cp_async_16B(&s_B[next_stage][b_row1][b_col1], src_b1 + next_m * width);
            cp_async_commit();
        }
        for (int k = 0; k < BK; k += 16) {
            wmma::load_matrix_sync(a_frag[0], &s_A[stage][warp_m * WM][k], BK + 8);
            wmma::load_matrix_sync(a_frag[1], &s_A[stage][warp_m * WM + 16][k], BK + 8);
            wmma::load_matrix_sync(b_frag[0], &s_B[stage][k][warp_n * WN], BN + 8);
            wmma::load_matrix_sync(b_frag[1], &s_B[stage][k][warp_n * WN + 16], BN + 8);
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
        if (next_m < width) { cp_async_wait_all(); __syncthreads(); }
        stage = next_stage;
    }
    __shared__ float s_C[BM][BN + 8];
    for(int i=0; i<2; i++) for(int j=0; j<2; j++) wmma::store_matrix_sync(&s_C[warp_m * WM + i * 16][warp_n * WN + j * 16], c_frag[i][j], BN + 8, wmma::mem_row_major);
    __syncthreads();
    for (int i = 0; i < 32; ++i) {
        int c_tid = tid * 32 + i; int r = c_tid / BN; int c = c_tid % BN;
        C[(blockIdx.y * BM + r) * width + (blockIdx.x * BN + c)] = __float2half(s_C[r][c]);
    }
}

void run_test(int width, float range) {
    int numElements = width * width;
    size_t size = numElements * sizeof(half);
    printf("\n>>> Testing Matrix %dx%d, Input Range [0, %.2f] <<<\n", width, width, range);

    std::vector<half> h_A(numElements), h_B(numElements), h_Ref(numElements), h_Res(numElements);
    srand(42); 
    for (int i = 0; i < numElements; ++i) { 
        h_A[i] = __float2half((rand() / (float)RAND_MAX) * range); 
        h_B[i] = __float2half((rand() / (float)RAND_MAX) * range); 
    }

    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size); cudaMalloc((void**)&d_B, size); cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    
    // Naive Reference
    timer.Start();
    MatrixMulNaive<<<dim3(width/32, width/32), dim3(32, 32)>>>(d_A, d_B, d_C, width);
    timer.Stop();
    float time_naive = timer.Elapsed();
    cudaMemcpy(h_Ref.data(), d_C, size, cudaMemcpyDeviceToHost);

    // Pipelined PTX
    cudaMemset(d_C, 0, size);
    // Warmup
    MatrixMulPipelined<<<dim3(width/64, width/64), dim3(32, 4)>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();
    
    timer.Start();
    MatrixMulPipelined<<<dim3(width/64, width/64), dim3(32, 4)>>>(d_A, d_B, d_C, width);
    timer.Stop();
    float time_ptx = timer.Elapsed();
    cudaMemcpy(h_Res.data(), d_C, size, cudaMemcpyDeviceToHost);

    printf("  -> Naive Time: %f ms\n", time_naive);
    printf("  -> PTX Pipelined Time: %f ms\n", time_ptx);
    printf("  -> Speedup vs Naive: %.2fx\n", time_naive / time_ptx);

    verify_result_advanced(h_Ref.data(), h_Res.data(), numElements, "Pipelined PTX");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(void) {
    run_test(1024, 1.0f);
    run_test(1024, 0.01f);
    return 0;
}