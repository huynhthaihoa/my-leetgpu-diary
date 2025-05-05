#include "solve.h"
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // Computes the global index of the element this thread will process
    // blockIdx.x: Index of the current block in the grid (x-dimension)
    // blockDim.x: Number of threads per block (x-dimension)
    // threadIdx.x: Index of the current thread within its block (x-dimension)
    // The formula blockIdx.x * blockDim.x + threadIdx.x maps the thread to a unique element in the vectors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Checks if the thread's index is within the bounds of the vectors
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
