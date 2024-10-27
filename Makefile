.PHONY = all run clean

NVCC = nvcc

all:
	@$(NVCC) -o GameOfLife GameOfLife.cu

run:
	@$(NVCC) -o GameOfLife GameOfLife.cu
	@./GameOfLife

clean:
	rm GameOfLife