

#include <cuda_runtime.h>
#include <stdio.h>
#include "vecAdd.h"  // Include the header file for the vecAdd function
#include <math.h>

// this program performs the vector addition A + B = C for vectors of length 2^20.


int main() {
    int vectorSize = 1 << 20;   // 2^20

    float* A_h = (float*) malloc(vectorSize*sizeof(float));
    float* B_h = (float*) malloc(vectorSize*sizeof(float));
    float* C_h = (float*) malloc(vectorSize*sizeof(float));

    // initialize vectors with 1s for A and 2s for B.
    for (int i = 0; i < vectorSize; i++) {
        A_h[i] = 1.0f;
        B_h[i] = 2.0f;
    }


    vecAdd(A_h, B_h, C_h, vectorSize); // Call the vecAdd function to perform vector addition

    // check the result
    bool success = true;
    for (int i = 0; i < vectorSize; i++) {
        if (fabs(C_h[i] - (A_h[i] + B_h[i])) > 1e-5) {
            success = false;
            printf("Error at index %d: Expected %f, got %f\n", i, A_h[i] + B_h[i], C_h[i]);
            break;
        }
    }
    if (success) printf("Vector Addition Successful!\n");

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

    // copy vectors A and B to device memory
    CUDA_ERROR_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // launch kernel to perform vector addition
    float threadsPerBlock = 256.0;
    vecAddKernel<<<(n + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(A_d, B_d, C_d, n);

    CUDA_ERROR_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost)); // copy result vector C back to host memory

    // free device memory for vectors A, B, and C
    CUDA_ERROR_CHECK(cudaFree(A_d));
    CUDA_ERROR_CHECK(cudaFree(B_d));
    CUDA_ERROR_CHECK(cudaFree(C_d));
}

__global__ void vecAddKernel(float *A, float *B, float* C, int n) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        // Old code: slow because it accessed VRAM
        // C[i] = A[i] + B[i];

        // fix: assign sum to register first
        float tmp = A[i] + B[i];
        // then assign to output
        C[i] = tmp;
    }
}