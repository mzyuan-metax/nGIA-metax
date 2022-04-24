#ifndef __FUNCH__
#define __FUNCH__

#include <iostream>  // std::string
#include <vector>  // vector

// 1080上没问题 A100上速度和1080一样 这两个参数得调
#define BLOCK 128  // 线程块数
#define WARP 128  // 每个线程块的线程数
#define HTD cudaMemcpyHostToDevice  // 内存拷显存
#define DTH cudaMemcpyDeviceToHost  // 显存拷内存

//--------数据结构--------//
struct Option {  // 各种参数
  // 输入参数
  std::string inputDir;  // 输入文件路径
  std::string outputDir;  // 输出文件路径
  int wordLength;  // 短词长度
  float similarity;  // 相似度阈值
  int precise;  // 精确模式
  int type;  // 0:基因序列 1:蛋白序列
  // MPI参数
  int rank;  // 进程编号
  int size;  // 总进程数
  // 数据信息
  int readsCount;  // 每个进程需要处理的序列数量
  int readsCountSum;  // 总序列数量
  int longest;  // 序列最长
  int shortest;  // 序列最短
};
struct Index {
  std::vector<int> nameLengths;  // 序列名长度
  std::vector<int> readLengths;  // 序列长度
  std::vector<long> offsets;  // 源文件索引
  std::vector<long> packOffsets;  // 压缩数据索引
};
// Data.buf_*数据结构:
// length    长度     int * 1
// netLength 净长度   int * 1
// bases     氨基酸   int * 28
// packed    打包数据 int * (length+31)/32*5
struct Data {
  // 主机数据
  long *offsets_h;  // 数据偏移h
  unsigned int *buf_h;  // 数据h
  // 设备数据
  long *offsets_d;  // 数据偏移d
  unsigned int *buf_d;  // 数据d
};
// Bench.represent_*数据结构:
// length    长度     int
// netLength 净长度   int
// bases     碱基     int * 4
// NULL      保留空位 int * 2
// packed    打包数据 int * (length+31)/32*32/32*2
struct Bench {  // 工作台
  // 主机数据
  int *tops;  // MPI缓冲 全局编号+长度
  unsigned int *represent_h;  // 代表序列h
  int *remains;  // 剩余序列
  int remainCount;  // 剩余序列数
  int *jobs_h;  // 任务h 序列编号+结果(-1:失败 0:成功)
  int jobCount;  // 任务数
  int *cluster;  // 聚类结果
  // 设备数据
  unsigned int *represent_d;  // 代表序列d
  int *jobs_d;  // 任务d
  // 统一数据
  unsigned int *list;  // 短词列表
};
//--------声明函数--------//
void init(int argc, char **argv, Option &option);  // 初始化
void readIndex(Option &option, Index &index);  // 读数据索引
void readData(Option &option, Index &index, Data &data);  // 读数据
void clustering(Option &option, Data &data, Bench &bench);  // 聚类
void saveResult(Option &option, Index &index, Bench &bench);  // 保存结果
void finish(Data &data, Bench &bench);  // 收尾

#endif  // __FUNCH__
