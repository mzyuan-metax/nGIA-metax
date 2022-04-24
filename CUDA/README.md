# nGIA序列聚类(去冗余)工具套件CUDA版本  
## MultiNode  
多节点版本，支持MPI，
### 多节点版本编译  
nvcc -I"MPI include" -L"MPI lib" -lmpi src/cluster.cpp src/func.cpp -o cluster  
可以参考对应的Makefile  
### 多节点版本运行  
mpirun -n <threadCount> ./cluster -i <inputDir> -o <outputDir> -w <wordLength> -s <similarity>  
可以参考对应的Makefile  
## SignalNode  
单节点版本，由多节点版本去掉MPI通讯后简化而来。  
单节点性能与多节点跑1个MPI进程性能相同，但优化掉MPI数据结构也不会更快。  
### 单节点版本编译  
nvcc src/cluster.cpp src/func.cpp -o cluster  
可以参考对应的Makefile  
### 单节点版本运行  
./cluster -i <inputDir> -o <outputDir> -w <wordLength> -s <similarity>  
可以参考对应的Makefile  
## 参数说明  
-i 输入目录，就是makeDB的输出目录  
-o 输出目录，保存结果的目录  
-w 短词长度，基因序列是4-8，蛋白序列是2-3，根据相似度不同，短词长度范围也不同，代码会自动调整  
-s 相似度，范围是0.1-1.0  
## 输出说明  
represent.fasta 数据集去冗余之后的结果  
cluster.txt 序列剧烈的结果，无缩进的行是代表序列名，有缩进的是类内序列名  
