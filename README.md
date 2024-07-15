## Project Name: `mini_rnn`
> Original Repository: https://github.com/mti-lab/rnn-descent 

`Relative NN-Descent` is an indexing project introduced by `Relative NN-Descent: A Fast Index Construction for Graph-Based Approximate Nearest Neighbor Search` in ACM MM 2023.

It was implemented with a Faiss-based backend. It is hard to run smoothly, I reorganized its code to make it easy to read, use and understand.

Its indexing speed is really fast, almost 2X to kgraph and 4X to hnsw. I use it for fast indexing and won the Runner-up Award in [2024 SIGMOD Programming Contest](https://2024.sigmod.org/sigmod_awards.shtml)

## Quick Start

> Tested works on Ubuntu.

```shell
./run.sh 
```

OR 

```shell
rm -rf build 
mkdir build 
cd build 
cmake ..
make -j 

cd ..


./build/rnn \
/home/jeff/data/sift1m/sift1m_base.fvecs \
out_index.ivecs
```

#### Parameters to Use
```cpp
rnndescent::rnn_para para;
para.S = 36;
para.T1 = 3;
para.T2 = 8;

rnndescent::Matrix<float> base_data;
base_data.load(base_path, 128, 0, 4); /// path, dim, skip, #bytes_per_elem
```