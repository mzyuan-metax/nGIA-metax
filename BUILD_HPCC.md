# Build on HPCC

This is a guide to build nGIA on HPCC.

- HPCC: 2.33.0.x
- OS  : ubuntu22.04

## 1. Init Envs

```
export HPCC_PATH="/opt/hpcc"
export CUDA_PATH=/usr/local/cuda
export CUCC_PATH=${HPCC_PATH}/tools/cu-bridge
export PATH=${HPCC_PATH}/ompi/include:${CUDA_PATH}/bin:${HPCC_PATH}/htgpu_llvm/bin:${HPCC_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${HPCC_PATH}/lib:${HPCC_PATH}/ompi/lib:${HPCC_PATH}/htgpu_llvm/lib:${LD_LIBRARY_PATH}
```

## 2. Init DB

**DATA**

```
$ mkdir temp
$ gzip -c data/current_NCBI_gg16S_unaligned.fasta.gz > temp/gene.fasta
$ gzip -c data/Homo_sapiens.GRCh37.70.pep.all.fa.gz > temp/Homo_sapiens.GRCh37.70.pep.all.fa
```

**Build** 

```
$ g++ -fopenmp makeDB/src/makeDB.cpp -o temp/makeDB
```

**Init**

```
$ temp/makeDB -i temp/gene.fasta -o temp -t 0
SSD users are recommended to use CPUs with 8 cores or more.
The bottleneck is small file I/O performance.
Supports up to 4,294,967,295 sequences, length of 65536.

Make data base begin:   Fri Sep 19 13:24:45 2025
Make index: 100.00%
Read 417 and 278 valid.
Sorting...finish
Write info...finish
Write index...finish
Make data base: 100.00%
Write data.bin...finish
Make data base finish:  Fri Sep 19 13:24:46 2025
Make data base total:   0.63 seconds.
```

## 3.signal node with CUDA

```
// build
$ /opt/hpcc/tools/cu-bridge/bin/cucc  CUDA/signalNode/src/cluster.cpp CUDA/signalNode/src/func.cu -o temp/cluster

// run
$ temp/cluster -i temp -o temp -w 5 -s 0.85
Data type:      Gene
Similarity:     0.85
Word length:    5
Precise mode:   0
Threads:        1
Reads count:    278
Longest:        65360
Shortest:       58
Reads/thread:   278

Compute offset: 0.00123306 seconds.
Read data:      0.002964 seconds.

278/278

Cluster:        278
Write data: 278/278
Save file...finish
Save result:    0.0205307 seconds.
0.785529 seconds.
```

## 4.MultiNode node with CUDA

```
// build
$ /opt/hpcc/tools/cu-bridge/bin/cucc -I/opt/hpcc/ompi/include/ -L/opt/hpcc/ompi/lib -lmpi CUDA/MultiNode/src/cluster.cpp CUDA/MultiNode/src/func.cu -o temp/cluster

// run
$ /opt/hpcc/ompi/bin/mpirun -n 2  --allow-run-as-root temp/cluster -i temp -o temp -w 5 -s 0.85
.85
Data type:      Gene
Similarity:     0.85
Word length:    5
Precise mode:   0
Threads:        2
Reads count:    278
Longest:        65360
Shortest:       58
Reads/thread:   139

Compute offset: 0.00196843 seconds.
Read data:      0.00141387 seconds.

278/278

Cluster:        278
Write data: 278/278
Save file...finish
Save result:    0.0279737 seconds.
1.60833 seconds.
```