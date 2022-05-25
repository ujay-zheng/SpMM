# cuSPARSE(CSR)

### Document Links

- [cuSPARSE(csr)documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm)

### Quick Start

##### make a new directory

```shell
cd /SpMM/cuSPARSE/csr && mkdir build
```

##### Building

* Linux

```shell
cd build
cmake ..
make
```

* Windows/Linux

I'm not sure whether the CMakeList.txt can work correctly on Windows, I have never tried it.

##### run

The SpMM function of cuSPARSE provides three different algorithms for processing matrices in csr format. As the [documentation](https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-function-spmm) says, some algorithms can achieve different performance with different layouts of dense matrices. The user just needs to select the algorithm they want to use by providing a number representing the algorithm and the program will find the layout that best suits the algorithm

```shell
./cusparse_csr /SpMM/guide.txt out/cuSPARSE_block.csv {1 or 2 or 3}
```