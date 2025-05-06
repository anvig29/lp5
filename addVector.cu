#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
using namespace std;

#define N 10000

__global__ void vectorAdd(int *d_a, int *d_b, int *d_c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

class VectorAddition {
public:
    void performVectorAddition() {
        int *a, *b, *c, *d;
        int *d_a, *d_b, *d_c;
        size_t size = N * sizeof(int);

        a = (int *)malloc(size);
        b = (int *)malloc(size);
        c = (int *)malloc(size);
        d = (int *)malloc(size);

        srand(time(NULL));
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 100;
            b[i] = rand() % 100;
        }

        clock_t start_cpu = clock();
        for (int i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        clock_t end_cpu = clock();
        double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);
        cudaMalloc((void **)&d_c, size);
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);

        cudaMemcpy(d, d_c, size, cudaMemcpyDeviceToHost);
        cudaEventSynchronize(stop);
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

        bool match = true;
        for (int i = 0; i < N; i++) {
            if (c[i] != d[i]) {
                match = false;
                break;
            }
        }

        printf("CPU Time: %.6f s\n", cpu_time);
        printf("GPU Time: %.6f ms\n", gpu_time);
        printf("Speedup Factor: %.2f\n", (cpu_time) * 1000 / gpu_time);
        printf("Arrays Match(Verification): %s\n", match ? "Yes" : "No");

        free(a); free(b); free(c); free(d);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

int main() {
    VectorAddition vectorAdder;
    vectorAdder.performVectorAddition();

    //nvcc -version
    //nvidia smi
    //nvcc matMul.cu
    // ./a.out
    return 0;
}
