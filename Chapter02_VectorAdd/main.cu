#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h> // Include the cuRAND library to allow for random vector generation

// this program performs the vector addition A + B = C for vectors of length 128.

// accepts host pointers to vectors A, B, and C, and the length of the vectors n
void vecAdd(float* A_h, float* B_h, float* C_h, int n);


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
    cudaError_t err = cudaMalloc((void**)&A_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&B_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&C_d, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // free device memory for vectors A, B, and C
    err = cudaFree(A_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaFree(B_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
    err = cudaFree(C_d);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}