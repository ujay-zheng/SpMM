# cuSPARSE(Block Sparse)

### Support

You can directly run the code if

* **Architecture** ≥ NVIDIA AMPERE
* **Compute capability** ≥ 8.0

You should modify lines 13 and 14 of file [bspmm_latency_evaluator.cpp](https://github.com/ujay-zheng/SpMM/blob/main/cuSPARSE/bspmm_latency_evaluator.cpp#L13) if

* **Architecture** ≥ Volta
* **Compute capability** ≥ 7.0

```
define cuda_date_type CUDA_R_32F --> define cuda_date_type CUDA_R_16F
define data_type float --> define data_type __half
```

### Document Links

- [cuSPARSE(block sparse)documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm)
- [cuSPARSE(block sparse)blog](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)

### Quick Start

##### make a new directory

```shell
cd /SpMM/cuSPARSE && mkdir out
```

##### Building

* Linux

```shell
make
```

* Windows/Linux

I'll write a CMakeList.txt someday, but not now. So before I write a CMakelist you can just run on Linux

##### run

```shell
./bspmm_latency_evaluator /SpMM/guide.txt out/cuSPARSE_block.csv
```

