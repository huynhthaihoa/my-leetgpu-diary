#include "solve.h"
#include <cuda_runtime.h>

// Defines the square tile size (16x16) used for shared memory optimization. 
// Tiles are submatrices that fit into shared memory to reduce global memory access and improve performance.
#define TILE_WIDTH 16

// CUDA kernel function that runs on the GPU
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    // Declares two shared memory arrays to store tiles of matrices A and B
    // Each is a 16x16 array of floats. Shared memory is fast, on-chip memory accessible by all threads in a block. These arrays hold submatrices (tiles) to minimize global memory accesses.
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate global row and column indices
    // blockIdx.y and blockIdx.x: Grid-level block indices in the y and x dimensions
    // threadIdx.y and threadIdx.x: Block-level thread indices within the block
    // Each block processes a TILE_WIDTH x TILE_WIDTH tile of C, and each thread within the block computes one element of that tile
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;  // in range [0, M - 1]
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;  // in range [0, K - 1]

    // Initializes a variable to accumulate the dot product for the C[row][col] element
    float sum = 0.0f;

    // Iterate over tiles along the shared dimension N (columns of A, rows of B)
    // The loop iterates [N / TILE_WIDTH] times to cover all elements needed for the dot product
    // Each iteration processes a TILE_WIDTH-wide strip of A and B
    for (int t = 0; t < (N - 1) / TILE_WIDTH + 1; t++) {
        // Load tileA (M x N) and tileB (N x K)

        // Check if the thread's assigned element is within A's bounds (row < M and column t * TILE_WIDTH + threadIdx.x < N)
        if (row < M && t * TILE_WIDTH + threadIdx.x < N)
            // Valid, loads the element from global memory (row-major indexing)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            // Out of bounds, sets the shared memory element to 0 to avoid undefined values in computations
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Check if the thread's assigned element is within B's bounds (col < K and row t * TILE_WIDTH + threadIdx.y < N)
        if (col < K && t * TILE_WIDTH + threadIdx.y < N)
            // Valid, loads the element from global memory (column-major indexing)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * K + col];
        else
            // Out of bounds, sets the shared memory element to 0 to avoid undefined values in computations
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronizes all threads in the block to ensure both tileA and tileB are fully loaded into shared memory before computation begins.
        __syncthreads();

        // Compute partial sum for this tile
        // Check if the thread's assigned element is within C's bounds (row < M and col < K)
        if (row < M && col < K) {
            // Iterates over the tile's width, multiplying corresponding elements from tileA and tileB and accumulating the result in sum
            // Uses shared memory for fast access, reducing global memory traffic
            for (int i = 0; i < TILE_WIDTH; i++) {
                sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
            }
        }

        // Synchronizes threads to ensure all computations are complete before loading the next tile. 
        // This prevents overwriting shared memory before all threads are done
        __syncthreads();
    }

    // Write result to output matrix C
    // Check if the thread's assigned element is within C's bound 
    // Store the accumulated sum in C[row * K + col] (row-major indexing)
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
