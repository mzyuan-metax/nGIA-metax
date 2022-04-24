# 数据库生成工具  
## 介绍  
从原始的fasta文件生成聚类需要的数据库。  
支持多线程，如果用ssd，最好能有8线程。  
### 编译  
g++ -fopenmp src/makeDB.cpp -o makeDB  
可以参考Makefile  
### 运行  
./makeDB -i ../data/Homo_sapiens.GRCh37.70.pep.all.fa data -o ../ -t 1  
可以参考Makefile  
### 参数说明  
-i 输入的fasta文件  
-o 生成数据库保存目录  
-t 输入数据的类型，0是基因序列，1是蛋白序列  
### 输出说明  
data.bin 在输出目录生成的数据库文件，包含聚类需要的数据结构。  
data.fasta 输入文件在输出目录的软链接，用于聚类后生成聚类结果。  
