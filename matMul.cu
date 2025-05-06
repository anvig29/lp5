#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>

#define N 25

__global__ void matrixMultiplyKernel(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

class MatrixMultiplier {
private:
    float *hostA, *hostB, *hostC, *hostD;
    float *devA, *devB, *devC;
    int size;
    float cpuTime, gpuTime;

public:
    MatrixMultiplier() {
        size = N * N * sizeof(float);
        hostA = (float *)malloc(size);
        hostB = (float *)malloc(size);
        hostC = (float *)malloc(size);
        hostD = (float *)malloc(size);
        cudaMalloc((void **)&devA, size);
        cudaMalloc((void **)&devB, size);
        cudaMalloc((void **)&devC, size);
        cpuTime = gpuTime = 0.0f;
    }

    ~MatrixMultiplier() {
        free(hostA);
        free(hostB);
        free(hostC);
        free(hostD);
        cudaFree(devA);
        cudaFree(devB);
        cudaFree(devC);
    }

    void initializeMatrices() {
        for (int i = 0; i < N * N; i++) {
            hostA[i] = rand() % 100;
            hostB[i] = rand() % 100;
        }
    }

    void gpuMatrixMultiplication() {
        cudaMemcpy(devA, hostA, size, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, hostB, size, cudaMemcpyHostToDevice);

        dim3 dimBlock(16, 16);
        dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

        clock_t tic = clock();
        matrixMultiplyKernel<<<dimGrid, dimBlock>>>(devA, devB, devC);
        cudaDeviceSynchronize();
        clock_t toc = clock();

        gpuTime = ((float)(toc - tic)) / CLOCKS_PER_SEC;
        cudaMemcpy(hostC, devC, size, cudaMemcpyDeviceToHost);
    }

    void cpuMatrixMultiplication() {
        clock_t tic = clock();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += hostA[i * N + k] * hostB[k * N + j];
                }
                hostD[i * N + j] = sum;
            }
        }
        clock_t toc = clock();
        cpuTime = ((float)(toc - tic)) / CLOCKS_PER_SEC;
    }

    bool verifyEquality() {
        float tolerance = 1e-5;
        for (int i = 0; i < N * N; i++) {
            if (fabs(hostC[i] - hostD[i]) > tolerance) {
                printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, hostC[i], hostD[i]);
                return false;
            }
        }
        return true;
    }

    void printResults() {
        printf("CPU Time: %f seconds\n", cpuTime);
        printf("GPU Time: %f seconds\n", gpuTime);
        if (gpuTime > 0) {
            printf("Speed-Up Factor: %.2f x\n", cpuTime / gpuTime);
        } else {
            printf("Speed-Up Factor: N/A (GPU time too small)\n");
        }
    }
};

int main() {
    MatrixMultiplier matrixMultiplier;
    matrixMultiplier.initializeMatrices();
    matrixMultiplier.cpuMatrixMultiplication();
    matrixMultiplier.gpuMatrixMultiplication();

    bool success = matrixMultiplier.verifyEquality();
    matrixMultiplier.printResults();

    printf("Verification: %s\n", success ? "true" : "false");

    //nvcc -version
    //nvidia smi
    //nvcc matMul.cu
    // ./a.out
    return 0;
}
