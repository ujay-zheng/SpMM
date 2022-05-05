# cuSPARSE(Block Sparse)

### Support

You can directly run the code if

* **Architecture** ≥ NVIDIA AMPERE
* **Compute capability** ≥ 8.0

You should change [bspmm_latency_evaluator.cpp](https://github.com/ujay-zheng/SpMM/blob/main/cuSPARSE/bspmm_latency_evaluator.cpp#L13)  line 13 and line 14 if

* **Architecture** ≥ Volta
* **Compute capability** ≥ 7.0

```
define cuda_date_type CUDA_R_32F --> define cuda_date_type CUDA_R_16F
define data_type float --> define data_type __half
```

### Document Links

- [cuSPARSE(block sparse)documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm)
- [cuSPARSE(block sparse)blog](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)

### Docker

We highly recommend you use Docker(Nvidia Docker) to build and run the project,after cloning this repo and enter the directory,run

```
docker build . -t cusparse-block
docker run --runtime=nvidia -v {You SpMM repo path}:/SpMM -itd cusparse-block
```

You have to make sure that the source data of matrix must be included in you repo path!

### Quick Start

##### Building

* Linux

```
make
```

* Windows/Linux

I'll write a CMakeList.txt someday, but not now. So before I write a CMakelist you can just run on Linux

#### Matrix Data

* data format

​	I save the source data for sparse and dense matrices in a multiplication process in a txt file. The first row stores the row number of the sparse matrix, the column number of the sparse matrix and the column number of the dense matrix, then the next row stores br(block row), bc(block column) and sparsity. After that is the source data of the row-major sparse matrix, and finally is the source data of the column-major dense matrix. You have to make sure that a txt file has only these elements!By the way, there is no requirement for the name of the txt file.

```
4 4 4
2 2 0.5

0 0 1 1
0 0 1 1
1 1 0 0 
1 1 0 0

1 1 1 1
1 1 1 1
1 1 1 1
1 1 1 1
```

* get a guide file

​	Because the source data of matrix can be in any path on your computer, [bspmm_latency_evaluator](https://github.com/ujay-zheng/SpMM/blob/main/cuSPARSE/bspmm_latency_evaluator) needs to know the path of each file, which is provided by guide.txt. You can get a guide file manually, but by running matrix.py, you just need to provide the path to the folder that contains all the source data, and the path where the generated guide file should go. The program will search all the files in the folder you give file, and write each file path found to the guide file in the specified path line by line.

​	Before run matrix.py, make sure you've install numpy.

```
python matrix.py {the folder contains the data} {guide file path}
```

* begin test

```
./bspmm_latency_evaluator {guide file} {CSV output file}
```

