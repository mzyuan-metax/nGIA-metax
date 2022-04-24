# nGIA序列聚类(去冗余)工具套件  
## 介绍  
采用贪婪增量算法的序列聚类(去冗余)工具套件，当前唯一可以进行序列精确聚类的工具，也是速度最快的(用多节点)。  
支持单节点,多节点,CUDA平台,OneAPI平台。  
### 应用范围  
-> 序列个数 1-2,147,483,647  
-> 序列长度 1-65536  
-> 相似度 0.1-1.0  
以上范围都可以通过更改数据结构int为long，轻松扩展，但不推荐。因为数据集太大一次要跑几天，没人会等，没意义。  
### 使用介绍  
首先用makeDB生成数据库。  
选择一个版本(多/单节点，CUDA/OneAPI)的聚类工具进行聚类。  
具体可以参考每个文件夹中的readme
#### 用单节点，cuda版本进行基因序列聚类  
创建工作目录  
mkdir temp  
解压数据  
gzip -c data/current_NCBI_gg16S_unaligned.fasta.gz > temp/gene.fasta  
编译makeDB工具  
g++ -fopenmp makeDB/src/makeDB.cpp -o temp/makeDB  
生成数据库  
temp/makeDB -i temp/gene.fasta -o temp -t 0  
编译单节点，CUDA版本聚类工具  
nvcc CUDA/signalNode/src/cluster.cpp CUDA/signalNode/src/func.cu -o temp/cluster  
聚类  
temp/cluster -i temp -o temp -w 5 -s 0.85  
### 运行需求  
需要Nvidia或者Intel的显卡。  
多节点版本需要MPI支持。  
### 瓶颈分析  
序列长度越长，计算量越大，也就越慢。  
生成数据库阶段瓶颈是硬盘小文件IO。  
多节点版本，数据小，节点多，会使得通信占比过高，平均一张卡至少要给1G数据。  
