#include "solve.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // Shared memory for tile
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Calculate input matrix indices
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load tile from input matrix A into shared memory
    if (y < rows && x < cols) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Calculate output matrix indices (transposed)
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y; 

    // Write tile to output matrix
    if (y < cols && x < rows) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
