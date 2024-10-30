#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

#define PERFORMANCE_COMPARISON_MODE false

#if PERFORMANCE_COMPARISON_MODE
    #define BOARD_HEIGHT 512
    #define BOARD_WIDTH  512
#else
    #define BOARD_HEIGHT 10
    #define BOARD_WIDTH  10
#endif

#define COORD_TO_IDX(row, col) ((row) * BOARD_WIDTH + (col))

// The following only matter if PERFORMANCE_COMPARISON_MODE is true. 
#define NUM_STEPS 256
#define BOARD_SIZE BOARD_HEIGHT * BOARD_WIDTH


double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


void printBoard(int *board) {
    system("cls");
    printf("\n");
    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            int idx = i * BOARD_WIDTH + j;
            printf(board[idx] ? "X" : "O");
        }
        printf("\n");
    }
}


__host__ __device__ int numNeighbors(int* board, int row, int col) {
    int neighborCount = 0;
    if (row > 0) {
        neighborCount += board[COORD_TO_IDX(row - 1, col)];
    }
    if (row < BOARD_HEIGHT - 1) {
        neighborCount += board[COORD_TO_IDX(row + 1, col)];
    }
    if (col > 0) {
        neighborCount += board[COORD_TO_IDX(row, col - 1)];
    }
    if (col < BOARD_WIDTH - 1) {
        neighborCount += board[COORD_TO_IDX(row, col + 1)];
    }

    if (row > 0 && col > 0) {
        neighborCount += board[COORD_TO_IDX(row - 1, col - 1)];
    }
    if (row > 0 && col < BOARD_WIDTH - 1) {
        neighborCount += board[COORD_TO_IDX(row - 1, col + 1)];
    }
    if (row < BOARD_HEIGHT - 1 && col > 0) {
        neighborCount += board[COORD_TO_IDX(row + 1, col - 1)];
    }
    if (row < BOARD_HEIGHT - 1 && col < BOARD_WIDTH - 1) {
        neighborCount += board[COORD_TO_IDX(row + 1, col + 1)];
    }

    return neighborCount;
}


__global__ void stepBoardCuda(int* board, int* newBoard) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    int neighborCount = numNeighbors(board, row, col);

    int idx = row * BOARD_WIDTH + col;
    if (board[idx]) {
        newBoard[idx] = neighborCount == 2 || neighborCount == 3;
    } else {
        newBoard[idx] = neighborCount == 3;
    }
}


int* stepBoard(int* board) {
    int* newBoard = (int*) malloc(BOARD_HEIGHT * BOARD_WIDTH * sizeof(int));
    for (int i = 0; i < BOARD_HEIGHT * BOARD_WIDTH; i++) {
        newBoard[i] = board[i];
    }

    for (int i = 0; i < BOARD_HEIGHT; i++) {
        for (int j = 0; j < BOARD_WIDTH; j++) {
            int neighborCount = numNeighbors(board, i, j);
            int idx = i * BOARD_WIDTH + j;
            if (board[idx]) {
                newBoard[idx] = neighborCount == 2 || neighborCount == 3;
            } else {
                newBoard[idx] = neighborCount == 3;
            }
        }
    }
    return newBoard;
}


int* calculateCPUPerf(int* board) {
    int *cpuBoard = (int*) malloc(BOARD_SIZE * sizeof(int));
    memcpy(cpuBoard, board, BOARD_SIZE * sizeof(int));
    
    // Warm-up.
    for (int i = 0; i < 5; i++) {
        stepBoard(cpuBoard);
    }

    double startTimeCPU = getTime();
    for (int i = 0; i < NUM_STEPS; i++) {
        cpuBoard = stepBoard(cpuBoard);
    }
    double endTimeCPU = getTime();
    printf("CPU Time for %d Steps = %f ms\n", NUM_STEPS, 1000 * (endTimeCPU - startTimeCPU));

    return cpuBoard;
}


int* calculateGPUPerf(int* board) {
    int* h_newBoard = (int*) malloc(BOARD_SIZE * sizeof(int));

    int* d_board;
    int* d_newBoard;
    cudaMalloc(&d_board, BOARD_SIZE * sizeof(int));
    cudaMalloc(&d_newBoard, BOARD_SIZE * sizeof(int));

    cudaMemcpy(d_board, board, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_newBoard, board, BOARD_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Warm-up.
    for (int i = 0; i < 5; i++) {
        stepBoardCuda<<<BOARD_HEIGHT, BOARD_WIDTH>>>(d_board, d_newBoard);
    }

    double startTimeGPU = getTime();
    for (int i = 0; i < NUM_STEPS; i++) {
        stepBoardCuda<<<BOARD_HEIGHT, BOARD_WIDTH>>>(d_board, d_newBoard);
        if (i != NUM_STEPS - 1) {
            int* temp = d_board;
            d_board = d_newBoard;
            d_newBoard = temp;
        }
    }
    double endTimeGPU = getTime();
    printf("GPU Time for %d Steps = %f ms\n", NUM_STEPS, 1000 * (endTimeGPU - startTimeGPU));

    cudaMemcpy(h_newBoard, d_newBoard, BOARD_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    return h_newBoard;
}


int main(int argc, char** argv) {
    int* board = (int*) malloc(BOARD_SIZE * sizeof(int));
    for (int i = 0; i < BOARD_SIZE; i++) {
        board[i] = 0;
    }

    if (PERFORMANCE_COMPARISON_MODE) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            board[i] = (int) ((float) rand() / RAND_MAX > 0.5);
        }

        int* cpuBoard = calculateCPUPerf(board);
        int* gpuBoard = calculateGPUPerf(board);

        bool correct = true;
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (gpuBoard[i] != cpuBoard[i]) {
                correct = false;
                break;
            }
        }
        printf("CPU and GPU Versions match = %d\n", correct);
    } else {
        // Glider.
        board[0] = 1;
        board[BOARD_WIDTH + 1] = 1;
        board[BOARD_WIDTH + 2] = 1;
        board[2 * BOARD_WIDTH] = 1;
        board[2 * BOARD_WIDTH + 1] = 1;

        while (true) {
            printBoard(board);
            sleep(1);
            board = stepBoard(board);
        }   
    }
}
