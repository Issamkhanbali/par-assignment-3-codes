#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_DIM 16
#define SHARED_DIM 16

__global__ void TiledMatrixProduct(int* A, int* B, int* C, int A_rows, int A_cols, int B_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sharedA[SHARED_DIM][SHARED_DIM];
    __shared__ int sharedB[SHARED_DIM][SHARED_DIM];

    int localRow, localCol;
    int tempSum = 0;

    for (int m = 0; m < (A_cols + SHARED_DIM - 1) / SHARED_DIM; ++m) {
        localRow = threadIdx.y;
        localCol = threadIdx.x;
        if (m * SHARED_DIM + localCol < A_cols && r < A_rows)
            sharedA[localRow][localCol] = A[r * A_cols + m * SHARED_DIM + localCol];
        else
            sharedA[localRow][localCol] = 0;

        if (m * SHARED_DIM + localRow < A_cols && c < B_cols)
            sharedB[localRow][localCol] = B[(m * SHARED_DIM + localRow) * B_cols + c];
        else
            sharedB[localRow][localCol] = 0;

        __syncthreads();

        for (int e = 0; e < SHARED_DIM; ++e) {
            tempSum += sharedA[localRow][e] * sharedB[e][localCol];
        }

        __syncthreads();
    }
    if (r < A_rows && c < B_cols) {
        C[r * B_cols + c] = tempSum;
    }
}

int main() {
    int matARows, matACols, matBCols;
    printf("Enter the dimensions of matrix A (Rows): ");
    scanf("%d", &matARows);
    printf("Enter the dimensions of matrix B (Columns A x Columns B): ");
    scanf("%d %d", &matACols, &matBCols);

    int *hostA, *hostB, *hostC;
    int *deviceA, *deviceB, *deviceC;

    size_t memSizeA = matARows * matACols * sizeof(int);
    size_t memSizeB = matACols * matBCols * sizeof(int);
    size_t memSizeC = matARows * matBCols * sizeof(int);

    hostA = (int*)malloc(memSizeA);
    hostB = (int*)malloc(memSizeB);
    hostC = (int*)malloc(memSizeC);

    srand(time(NULL));
    for (int i = 0; i < matARows * matACols; ++i) {
        hostA[i] = rand() % 10;
    }
    for (int i = 0; i < matACols * matBCols; ++i) {
        hostB[i] = rand() % 10;
    }

    cudaMalloc((void**)&deviceA, memSizeA);
    cudaMalloc((void**)&deviceB, memSizeB);
    cudaMalloc((void**)&deviceC, memSizeC);

    cudaMemcpy(deviceA, hostA, memSizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, memSizeB, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((matBCols + block.x - 1) / block.x, (matARows + block.y - 1) / block.y);

    cudaEvent_t beginEvent, endEvent;
    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(beginEvent);

    TiledMatrixProduct<<<grid, block>>>(deviceA, deviceB, deviceC, matARows, matACols, matBCols);

    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    float timeElapsed = 0;
    cudaEventElapsedTime(&timeElapsed, beginEvent, endEvent);

    cudaMemcpy(hostC, deviceC, memSizeC, cudaMemcpyDeviceToHost);
    printf("%f milliseconds\n", timeElapsed);

    cudaEventDestroy(beginEvent);
    cudaEventDestroy(endEvent);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostA);
    free(hostB);
    free(hostC);
    return 0;
}
