CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcudart -lcusparse

all: bspmm_latency_evaluator

bspmm_latency_evaluator: bspmm_latency_evaluator.cpp bspmm.h
	nvcc $(INC) bspmm_latency_evaluator.cpp -o bspmm_latency_evaluator $(LIBS)

clean:
	rm -f bspmm_latency_evaluator

test:
	@echo "\n==== SpMM BLOCKED ELL Test ====\n"
	./bspmm_latency_evaluator

.PHONY: clean all test%