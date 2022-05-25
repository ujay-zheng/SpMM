# SpMM Bench

This repo is a collection of the latency tests for SpMM libraries. Each library has its own folder with a README inside to tell you how to use them.

### Docker

We highly recommend you use Docker(Nvidia Docker) to run any subdirectory of this repo, after cloning this repo then you can go into any subdirectory to build docker image via Dockerfile. 

* for cuSPARSE and Sputnik

```
docker build . -t image-name
docker run --runtime=nvidia -v {You SpMM repo path}:/SpMM -itd image-name
```



You have to make sure that the source data of matrix must be included in you repo path which will be mounted to docker container!  If not, you must generate matrix in the docker container. We recommend that you put the matrix data in the top level directory of this repo, that is,under SpMM/. 

### Matrix Data

* data format

​	I save the source data for sparse and dense matrices in a multiplication process in a txt file. The first row stores the row number of the sparse matrix, the column number of the sparse matrix and the column number of the dense matrix, then the next row stores br(block row), bc(block column) and sparsity. We assume that the mask pattern is block pattern, if you want to use unstructured pattern, just set br=bc=1. After that is the source data of the row-major sparse matrix, and finally is the source data of the column-major dense matrix. You have to make sure that a txt file has only these elements!By the way, there is no requirement for the name of the txt file.

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

​	Using C++ to traverse the files in a folder is not a convenient thing, especially when the situation is complicated, instead, Python is more suitable. So we use SpMM/matrix.py to generate a guide.txt which records the path of all source data files in the user-given path. Reading guide.txt is easier for C++.

```shell
python3 matrix.py /SpMM/matrix /SpMM/guide.txt
```

