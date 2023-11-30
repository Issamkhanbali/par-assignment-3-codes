#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TILE_WIDTH 16

__global__ void TileMatrixMult(int* A, int* B, int* C, int A_rows, int A_cols, int B_cols) {
    int row_index = blockIdx.y * blockDim.y + threadIdx.y;
    int col_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_index < A_rows && col_index < B_cols) {
        int temp_sum = 0;
        for (int element = 0; element < A_cols; ++element) {
            temp_sum += A[row_index * A_cols + element] * B[element * B_cols + col_index];
        }
        C[row_index * B_cols + col_index] = temp_sum;
    }
}

int main() {
    int matrixARows, matrixACols, matrixBCols;
    printf("Enter the number of rows for matrix A: ");
    scanf("%d", &matrixARows);
    printf("Enter the number of columns for matrix A and rows for matrix B: ");
    scanf("%d", &matrixACols);
    printf("Enter the number of columns for matrix B: ");
    scanf("%d", &matrixBCols);

    int *hostA, *hostB, *hostC;
    int *deviceA, *deviceB, *deviceC;

    size_t sizeA = matrixARows * matrixACols * sizeof(int);
    size_t sizeB = matrixACols * matrixBCols * sizeof(int);
    size_t sizeC = matrixARows * matrixBCols * sizeof(int);

    hostA = (int*)malloc(sizeA);
    hostB = (int*)malloc(sizeB);
    hostC = (int*)malloc(sizeC);

    srand(time(NULL));
    for (int i = 0; i < matrixARows * matrixACols; ++i) {
        hostA[i] = rand() % 100;
    }
    for (int i = 0; i < matrixACols * matrixBCols; ++i) {
        hostB[i] = rand() % 100;
    }

    cudaMalloc((void**)&deviceA, sizeA);
    cudaMalloc((void**)&deviceB, sizeB);
    cudaMalloc((void**)&deviceC, sizeC);

    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim((matrixBCols + block_dim.x - 1) / block_dim.x, (matrixARows + block_dim.y - 1) / block_dim.y);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);

    TileMatrixMult<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC, matrixARows, matrixACols, matrixBCols);

    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    float elapsed_time = 0;
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f ms\n", elapsed_time);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostA);
    free(hostB);
    free(hostC);
    return 0;
}
