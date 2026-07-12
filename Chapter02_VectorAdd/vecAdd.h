#ifndef VECADD_H
#define VECADD_H

// error checking macro
#define CUDA_ERROR_CHECK(err) \
    do { \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// accepts host pointers to vectors A, B, and C, and the length of the vectors n
void vecAdd(float* A_h, float* B_h, float* C_h, int n);

__global__ void vecAddKernel(float *A, float *B, float* C, int n);

#endif // VECADD_H