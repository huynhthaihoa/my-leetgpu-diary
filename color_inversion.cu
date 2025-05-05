#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    // Computes the 2D coordinates of the pixel this thread will process
    // blockIdx.x and blockIdx.y: Block indices in the grid (x and y dimensions)
    // blockDim.x and blockDim.y: Number of threads per block in the x and y dimensions
    // threadIdx.x and threadIdx.y: Thread indices within the block (x and y dimensions)
    // x is the pixel's column index, and y is the pixel's row index in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within image bounds
    if (x < width && y < height) {
        // Calculate the starting index of the pixel's RGBA values
        int idx = (y * width + x) * 4;

        // Invert R, G, B components (255 - value), leave A unchanged
        image[idx] = 255 - image[idx];         // R
        image[idx + 1] = 255 - image[idx + 1]; // G
        image[idx + 2] = 255 - image[idx + 2]; // B
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
void solve(unsigned char* image, int width, int height) {
    unsigned char *d_image;
    size_t size = width * height * 4 * sizeof(unsigned char);

    // Allocate device memory
    cudaMalloc(&d_image, size);

    // Copy input image to device
    cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);

    // Configure kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);

    // Copy result back to host
    cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
}
