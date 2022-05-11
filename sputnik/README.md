# Sputnik

### Document Links

* [google-research/sputnik](https://github.com/google-research/sputnik)
* [Sparse GPU Kernels For Deep Learning](https://arxiv.org/abs/2006.10901)

### Quickly Start

##### make a new directory

```shell
cd /SpMM/spuntik && mkdir build
```

##### building

* Linux

```shell
cd build
cmake .. 
make
```

* Windows

I'm not sure whether the CMakeList.txt can work correctly on Windows, I have never tried it. 

##### run

```shell
./Sputnik /SpMM/guide.txt /SpMM/sputnik/build/out.csv
```

