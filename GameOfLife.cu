#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

#define PERFORMANCE_COMPARISON_MODE true

#if PERFORMANCE_COMPARISON_MODE
    #define BOARD_HEIGHT 256
    #define BOARD_WIDTH  256
    #define NUM_STEPS 256
#else
    #define BOARD_HEIGHT 10
    #define BOARD_WIDTH  10
#endif

#define COORD_TO_IDX(row, col) ((row) * BOARD_WIDTH + (col))

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void printBoard(int *board) {
    // system("cls");
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
            // printf("%d ", neighborCount);

            int idx = i * BOARD_WIDTH + j;
            if (board[idx]) {
                // live
                if (neighborCount < 2 || neighborCount > 3) {
                    newBoard[idx] = 0;
                }
            } else {
                // dead
                if (neighborCount == 3) {
                    newBoard[idx] = 1;
                }
            }
        }
        // printf("\n");
    }
    return newBoard;
}


int main(int argc, char** argv) {
    int board_size = BOARD_HEIGHT * BOARD_WIDTH;
    int* board = (int*) malloc(board_size * sizeof(int));
    for (int i = 0; i < board_size; i++) {
        board[i] = 0;
    }

    if (PERFORMANCE_COMPARISON_MODE) {
        for (int i = 0; i < board_size; i++) {
            board[i] = (int) ((float) rand() / RAND_MAX > 0.5);
        }

        // CPU
        int *cpuBoard = (int*) malloc(board_size * sizeof(int));
        memcpy(cpuBoard, board, board_size * sizeof(int));
        double start_time_cpu = get_time();
        for (int i = 0; i < NUM_STEPS; i++) {
            cpuBoard = stepBoard(cpuBoard);
        }
        double end_time_cpu = get_time();
        printf("CPU Time for %d Steps = %f ms\n", NUM_STEPS, 1000 * (end_time_cpu - start_time_cpu));

        // GPU

        int* h_newBoard = (int*) malloc(board_size * sizeof(int));

        int* d_board;
        int* d_newBoard;
        cudaMalloc(&d_board, board_size * sizeof(int));
        cudaMalloc(&d_newBoard, board_size * sizeof(int));

        cudaMemcpy(d_board, board, board_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_newBoard, board, board_size * sizeof(int), cudaMemcpyHostToDevice);

        // Warm-up
        for (int i = 0; i < 5; i++) {
            stepBoardCuda<<<BOARD_HEIGHT, BOARD_WIDTH>>>(d_board, d_newBoard);
        }

        double start_time_gpu = get_time();
        for (int i = 0; i < NUM_STEPS; i++) {
            stepBoardCuda<<<BOARD_HEIGHT, BOARD_WIDTH>>>(d_board, d_newBoard);
            if (i != NUM_STEPS - 1) {
                int* temp = d_board;
                d_board = d_newBoard;
                d_newBoard = temp;
            }
        }
        double end_time_gpu = get_time();
        printf("GPU Time for %d Steps = %f ms\n", NUM_STEPS, 1000 * (end_time_gpu - start_time_gpu));

        cudaMemcpy(h_newBoard, d_newBoard, board_size * sizeof(int), cudaMemcpyDeviceToHost);

        bool correct = true;
        for (int i = 0; i < board_size; i++) {
            if (h_newBoard[i] != cpuBoard[i]) {
                correct = false;
                break;
            }
        }
        printf("CPU and GPU Versions match = %d\n", correct);
    } else {
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
