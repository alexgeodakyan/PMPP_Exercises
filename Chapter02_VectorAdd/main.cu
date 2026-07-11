

#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h> // Include the cuRAND library to allow for random vector generation
#include "vecAdd.h"  // Include the header file for the vecAdd function

// this program performs the vector addition A + B = C for vectors of length 128.




int main() {
    int vectorSize = 128;
    float* A_h, * B_h, * C_h; // Host pointers for vectors A, B, and C

    vecAdd(A_h, B_h, C_h, vectorSize); // Call the vecAdd function to perform vector addition

    return 0;
}


void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float* A_d;     // Device pointer for vector A
    float* B_d;     // Device pointer for vector B
    float* C_d;     // Device pointer for vector C

    int size = n*sizeof(float); // Size of a vector in bytes

    // allocate device memory for vectors A, B, and C
    CUDA_ERROR_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_ERROR_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_ERROR_CHECK(cudaMalloc((void**)&C_d, size));

    // free device memory for vectors A, B, and C
    CUDA_ERROR_CHECK(cudaFree(A_d));
    CUDA_ERROR_CHECK(cudaFree(B_d));
    CUDA_ERROR_CHECK(cudaFree(C_d));
}