// func.cu
// 单元测试通过后再标记ok
// 修改或优化函数后一定要与ok的函数比对结果

#include <iostream>  // cout
#include <fstream>  // fstream
#include <vector>  // vector
#include "timer.h"  // 计时器
#include "cmdline.h"  // 解析器
#include "func.h"  // 数据结构与函数

// init 初始化 ok
void init(int argc, char **argv, Option &option) { 
  {  // 解析参数
    cmdline::parser parser;
    parser.add<std::string>("input", 'i', "input directory", true, "");
    parser.add<std::string>("output", 'o', "output directory", true, "");
    parser.add<int>("word", 'w', "word length gene:4-8, protein:2-3", true, 0);
    parser.add<int>("precise", 'p', "precise mode 0 or 1", false, 0);
    parser.add<float>("similarity", 's', "similarity 0.1-1.0", true, 0.95);
    parser.parse_check(argc, argv);
    option.inputDir = parser.get<std::string>("input");  // 输入文件路径
    option.outputDir = parser.get<std::string>("output");  // 输出文件路径
    option.wordLength = parser.get<int>("word");  // 短词长度
    option.precise = parser.get<int>("precise");  // 精确模式
    option.similarity = parser.get<float>("similarity");  // 相似度阈值
  }
  {  // MPI初始化
    option.rank = 0;
    option.size = 1;
  }
  {  // 选择设备
    int deviceCount;  // 节点上的设备数
    cudaGetDeviceCount(&deviceCount);
    int device = option.rank%deviceCount;
    cudaSetDevice(device);  // 根据rank选择设备
  }
  cudaDeviceSynchronize();  // 同步
}

// readIndex 读数据索引 ok
void readIndex(Option &option, Index &index) {
  // 常用数据
  int readsCountSum=0;  // 总序列数
  int rank = option.rank;  // MPI进程号
  int size = option.size;  // MPI进程数
  std::ifstream dataFile(option.inputDir+"/data.bin");  // 读文件
  {  // 读数据信息
    dataFile.read((char*)&option.readsCountSum, sizeof(int));  // 总序列数
    readsCountSum = option.readsCountSum;
    dataFile.read((char*)&option.longest, sizeof(int));  // 最长序列
    dataFile.read((char*)&option.shortest, sizeof(int));  // 最短序列
    dataFile.read((char*)&option.type, sizeof(int));  // 序列种类
    option.readsCount = (readsCountSum+size-1)/size;  // 每线程序列数
  }
  {  // 规范参数
    option.similarity = max(0.1f, option.similarity);
    option.similarity = min(1.0f, option.similarity);
    option.precise = max(0, option.precise);
    option.precise = min(1, option.precise);
    float similarity = option.similarity;
    if (option.type == 0) {  // 基因序列
      if (similarity < 0.75f) option.precise = 1;  // 相似度低 精确模式
      option.wordLength = max(4, option.wordLength);
      option.wordLength = min(8, option.wordLength);
      if (similarity < 0.88f) option.wordLength = min(7, option.wordLength);
      if (similarity < 0.86f) option.wordLength = min(6, option.wordLength);
      if (similarity < 0.84f) option.wordLength = min(5, option.wordLength);
      if (similarity < 0.81f) option.wordLength = min(4, option.wordLength);
    } else {  // 蛋白序列
      if (similarity < 0.5f) option.precise = 1;  // 相似度低 精确模式
      option.wordLength = max(2, option.wordLength);
      option.wordLength = min(3, option.wordLength);
      if (similarity < 0.7f) option.wordLength = min(2, option.wordLength);
    }
  }
  if (rank == 0) {  // 线程0输出信息
    std::string dataType[2] = {"Gene", "Protein"};
    std::cout << "Data type:\t" << dataType[option.type] << "\n";
    std::cout << "Similarity:\t" << option.similarity << "\n";
    std::cout << "Word length:\t" << option.wordLength << "\n";
    std::cout << "Precise mode:\t" << option.precise << "\n";
    std::cout << "Threads:\t" << option.size << "\n";
    std::cout << "Reads count:\t" << option.readsCountSum << std::endl;
    std::cout << "Longest:\t" << option.longest << std::endl;
    std::cout << "Shortest:\t" << option.shortest << std::endl;
    std::cout << "Reads/thread:\t" << option.readsCount << std::endl;
    std::cout << std::endl;  // 隔开下个信息块
  }
  {  // 读取索引信息
    index.nameLengths.resize(readsCountSum);  // 序列名长度
    index.readLengths.resize(readsCountSum);  // 序列长度
    index.offsets.resize(readsCountSum);  // 序列名索引
    index.packOffsets.resize(readsCountSum);  // 序列数据索引
    dataFile.read((char*)index.nameLengths.data(), sizeof(int)*readsCountSum);
    dataFile.read((char*)index.readLengths.data(), sizeof(int)*readsCountSum);
    dataFile.read((char*)index.offsets.data(), sizeof(long)*readsCountSum);
    dataFile.read((char*)index.packOffsets.data(), sizeof(long)*readsCountSum);
  }
  dataFile.close();
}

// readData 读数据 ok
void readData(Option &option, Index &index, Data &data) {
  // 常用数据
  int rank = option.rank;  // MPI进程号
  int size = option.size;  // MPI进程数
  int readsCountSum = option.readsCountSum;  // 总序列数
  int readsCount = option.readsCount;  // 每线程处理序列数
  int type = option.type;  // 输入数据类型
  long bufLength = 0; // 数据总长度
  Timer timer;  // 计时器
  {  // 序列长度与偏移 ok
    timer.start();
    cudaMallocHost(&data.offsets_h, sizeof(long)*readsCount);  // 数据偏移h
    cudaMalloc(&data.offsets_d, sizeof(long)*readsCount);  // ~d
    for (int i=rank, j=0; i<readsCountSum; i+=size, j++) {  // 生成偏移
      data.offsets_h[j] = bufLength;
      if (type ==0) {  // 基因序列
        bufLength += 6+(index.readLengths[i]+31)/32*2;  // 数据总长度
      } else {  // 蛋白序列
        bufLength += 30+(index.readLengths[i]+31)/32*5;  // 数据总长度
      }
    }
    if (data.offsets_h[readsCount-1] == 0) {  // 序列数不整除线程
      data.offsets_h[readsCount-1] = bufLength;
      bufLength += 30;  // 防访存越界
    }
    cudaMemcpy(data.offsets_d, data.offsets_h, sizeof(long)*readsCount, HTD);
    cudaDeviceSynchronize();  // 同步
    if (rank == 0) {std::cout << "Compute offset:\t"; timer.getDuration();}
  }
  {  // 读数据 ok
    timer.start();
    cudaMallocHost(&data.buf_h, sizeof(unsigned int)*bufLength);  // 数据h
    cudaMalloc(&data.buf_d, sizeof(unsigned int)*bufLength);  // ~d
    std::ifstream dataFile(option.inputDir+"/data.bin");  // 序列文件
    for (int i=rank, j=0; i<readsCountSum; i+=size, j++) {  // 读序列数据
      dataFile.seekg(index.packOffsets[i], std::ios::beg);  // 跳转到数据头
      int length;  // 要拷贝的数据长度
      if (type == 0) {  // 基因序列
        length = sizeof(unsigned int)*(6+(index.readLengths[i]+31)/32*2);
      } else {  // 蛋白序列
        length = sizeof(unsigned int)*(30+(index.readLengths[i]+31)/32*5);
      }
      dataFile.read((char*)(data.buf_h+data.offsets_h[j]), length);  // 读数据
    }
    dataFile.close();
    cudaMemcpy(data.buf_d, data.buf_h, sizeof(unsigned int)*bufLength, DTH);
    cudaDeviceSynchronize();  // 同步
    if (rank == 0) {std::cout << "Read data:\t"; timer.getDuration();}
  }
  if (rank == 0) std::cout << std::endl;  // 隔开下个信息块
}

// initBench 初始化bench ok
void initBench(Option &option, Bench &bench) {
  int readsCount = option.readsCount;  // 每线程处理序列数
  int type = option.type;  // 输入数据类型
  cudaMallocHost(&bench.tops, sizeof(int)*(option.size+1)*2);  // MPI头序列序号
  int length;  // 头序列数据长度
  if (type == 0) {  // 基因序列
    length = 6+(option.longest+31)/32*2;
  } else {  // 蛋白序列
    length = 30+(option.longest+31)/32*5;
  }
  cudaMallocHost(&bench.represent_h, sizeof(unsigned int)*length);  // 头序列h
  cudaMalloc(&bench.represent_d, sizeof(unsigned int)*length);  // ~d
  cudaMallocHost(&bench.remains, sizeof(int)*readsCount);  // 剩余序列h
  for (int i=0; i<readsCount; i++) bench.remains[i] = i;
  bench.remainCount = readsCount;  // 剩余序列数
  cudaMallocHost(&bench.jobs_h, sizeof(int)*readsCount*2);  // 任务h
  cudaMalloc(&bench.jobs_d, sizeof(int)*readsCount*2);  // ~d
  for (int i=0; i<readsCount; i++) {  // -1:失败 0:成功
    bench.jobs_h[i*2] = i;
    bench.jobs_h[i*2+1] = -1;
  }
  bench.jobCount = readsCount;  // 任务数
  cudaMallocHost(&bench.cluster, sizeof(int)*readsCount);  // 聚类结果
  for (int i=0; i<readsCount; i++) bench.cluster[i] = -1;  // -1:未聚类
  cudaMallocManaged(&bench.list, sizeof(unsigned int)*2048);  // 短词列表
  cudaDeviceSynchronize();  // 同步
}

// updateRepresent 更新代表序列 ok
void updateRepresent(Option &option, Data &data, Bench &bench) {
  // 常用数据
  int rank = option.rank;  // MPI进程号
  int size = option.size;  // MPI进程数
  {  // 更新 聚类结果
    int top = *bench.tops;  // 代表序列全局编号
    for (int i=0; i<bench.jobCount; i++) {  // 遍历任务列表
      int index = bench.jobs_h[i*2];
      int result = bench.jobs_h[i*2+1];  // -1:聚类失败 0:聚类成功
      if (result == 0) bench.cluster[index] = top;  // 聚类成功
    }
  }
  {  // 更新 未聚类列表 任务列表
    int count = 0;  // 聚类未成功数
    for (int i=0; i<bench.remainCount; i++) {
      int index = bench.remains[i];
      if (bench.cluster[index] == -1) {  // 如果这个序列还没聚类
        bench.remains[count] = index;  // 未聚类列表
        bench.jobs_h[count*2] = index;  // 任务列表
        bench.jobs_h[count*2+1] = 0;  // 任务需要执行
        count += 1;
      }
    }
    bench.remainCount = count;
    bench.jobCount = count;
    cudaMemcpy(bench.jobs_d, bench.jobs_h, sizeof(int)*count*2, HTD);
  }
  {  // 更新 代表序列
    if (bench.remainCount > 0) {  // 如果聚类还没完成
      int index = bench.remains[0];  // 代表序列的本地编号
      unsigned int *bufBegin = data.buf_h+data.offsets_h[index];  // 数据位置
      int length;  // 序列占用长度
      if (option.type == 0) {  // 基因序列
        length = 6+(*bufBegin+31)/32*2;
      } else {  // 蛋白序列
        length = 30+(*bufBegin+31)/32*5;
      }
      memcpy(bench.represent_h, bufBegin, sizeof(unsigned int)*length);
      *bench.tops = index*size+rank;  // 全局编号
      *(bench.tops+1) = length;  // 序列占用长度
    } else {
      *bench.tops = option.readsCountSum;  // 聚类完成
    }
  }
  {  // 广播 代表序列
    if (*bench.tops == option.readsCountSum) return;  // 聚类完成
    int order = *bench.tops%size;  //代表序列所在进程
    int length = *(bench.tops+1);  // 数据编码长度
    cudaMemcpy(bench.represent_d, bench.represent_h, sizeof(int)*length, HTD);
    if (rank == order) bench.cluster[bench.remains[0]] = *bench.tops;  // 聚类
  }
  cudaDeviceSynchronize();
}

// generateListGen 生成短词列表 基因 ok
void generateListGen(int wordLength, Bench &bench) {
  unsigned int *list = bench.list;
  memset(list, 0, sizeof(unsigned int)*2048);  // 清零 1024=32768/32
  int netLength = *(bench.represent_h+1);  // 净长度
  unsigned int *packed = bench.represent_h+6;
  unsigned int mask = (1<<wordLength*2)-1;  // 短词的mask
  //　计算
  if (netLength < 16) return;  // 序列太短就跳过 后面短词过滤函数也一样
  unsigned int word = 0;  // 短词
  for (int i=0; i<wordLength-1; i++) {  // 短词初始化
    word <<= 2;
    word += (*(packed+0)>>i&1)<<0;
    word += (*(packed+1)>>i&1)<<1;
  }
  for (int i=wordLength-1, k=wordLength-1; i<netLength; i++, k++) {
    word <<= 2;
    word += (*(packed+0)>>k&1)<<0;
    word += (*(packed+1)>>k&1)<<1;
    word &= mask;
    unsigned int order = word/32;  // 第几个数
    unsigned int key = 1<<word%32;  // 第几个位
    list[order] &= ~key;
    list[order] += key;
    if (k == 31) {k = -1; packed += 2;}
  }
  cudaDeviceSynchronize();
}

// generateListPro 生成短词列表 蛋白 ok
void generateListPro(int wordLength, Bench &bench) {
  unsigned int *list = bench.list;
  memset(list, 0, sizeof(unsigned int)*2048);  // 清零 1024=32768/32
  int netLength = *(bench.represent_h+1);  // 净长度
  unsigned int *packed = bench.represent_h+30;
  unsigned int mask = (1<<wordLength*5)-1;  // 短词的mask
  //　计算
  if (netLength < 16) return;  // 序列太短就跳过 后面短词过滤函数也一样
  unsigned int word = 0;  // 短词
  for (int i=0; i<wordLength-1; i++) {  // 短词初始化
    word <<= 5;
    word += (*(packed+0)>>i&1)<<0;
    word += (*(packed+1)>>i&1)<<1;
    word += (*(packed+2)>>i&1)<<2;
    word += (*(packed+3)>>i&1)<<3;
    word += (*(packed+4)>>i&1)<<4;
  }
  for (int i=wordLength-1, k=wordLength-1; i<netLength; i++, k++) {
    word <<= 5;
    word += (*(packed+0)>>k&1)<<0;
    word += (*(packed+1)>>k&1)<<1;
    word += (*(packed+2)>>k&1)<<2;
    word += (*(packed+3)>>k&1)<<3;
    word += (*(packed+4)>>k&1)<<4;
    word &= mask;
    unsigned int order = word/32;  // 第几个数
    unsigned int key = 1<<word%32;  // 第几个位
    list[order] &= ~key;
    list[order] += key;
    if (k == 31) {k = -1; packed += 5;}
  }
  cudaDeviceSynchronize();
}

// updateJobs 更新任务列表 ok
void updateJobs(Bench &bench) {
  int count = 0;  // 剩余序列数
  int *result = bench.jobs_h;  // 上次的结果
  int *job = bench.jobs_h;  // 新的任务
  for (int i=0; i<bench.jobCount; i++, result+=2) {
    if (*(result+1) == 0) {  // -1:任务不通过 0:任务通过
      *job = *result;
      *(job+1) = 0;
      job += 2;
      count += 1;
    }
  }
  bench.jobCount = count;
  cudaMemcpy(bench.jobs_d, bench.jobs_h, sizeof(int)*count*2, HTD);
  cudaDeviceSynchronize();
}

// kernel_baseFilterGen 前置过滤基因 ok
__global__ void kernel_baseFilterGen(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, float similarity) {
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 查询序列长度
    int sum = 0;  // 相同氨基酸数
    for (int i=0; i<4; i++) {
      sum += min(*(bufBegin+2+i), *(represent+2+i));
    }
    int length = ceil((float)length2*similarity);
    if (sum < length) jobs[idx*2+1] = -1;  // 过滤失败
  }
}

// kernel_baseFilterPro 前置过滤蛋白 ok
__global__ void kernel_baseFilterPro(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, float similarity) {
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 查询序列长度
    int sum = 0;  // 相同氨基酸数
    for (int i=0; i<28; i++) {
      sum += min(*(bufBegin+2+i), *(represent+2+i));
    }
    int length = ceil((float)length2*similarity);
    if (sum < length) jobs[idx*2+1] = -1;  // 过滤失败
  }
}

// kernel_wordFilterGen 短词过滤基因 ok
__global__ void kernel_wordFilterGen(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, int wordLength,
float similarity, unsigned int *listTemp) {
  int netLength1 = *(represent+1);
  if (netLength1 < 16) return;  // 净长度太短 跳过短词过滤
  // 准备数据
  __shared__ unsigned int list[2048];
  for (int i=threadIdx.x; i<2048; i+=blockDim.x) {
    list[i] = listTemp[i];
  }
  unsigned short mask = (1<<wordLength*2)-1;  // 短词的mask
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    // 准备数据
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 长度
    int netLength2 = *(bufBegin+1);  // 净长度
    if (netLength2 < 16) return;  // 净长度太短 跳过短词过滤
    unsigned int *packed2 = bufBegin+6;  // 打包数据
    //　计算
    int sum = 0;
    unsigned short word = 0;  // 短词
    for (int i=0; i<netLength2; i+=32, packed2+=2) {
      for (int j=0; j<32; j++) {
        if (i+j == netLength2) break; 
        word <<= 2;
        word += (*(packed2+0)>>j&1)<<0;
        word += (*(packed2+1)>>j&1)<<1;
        word &= mask;
        int order = word/32;
        unsigned int key = 1<<word%32;
        if (i+j > wordLength-2) sum += ((list[order]&key) > 0);
      }
    }
    // 处理结果
    int len = length2-wordLength+1;  // 理论短词数
    int length = ceil((float)length2*(1.0f-similarity));
    int minLen = len-length*wordLength;
    if (sum < minLen) jobs[idx*2+1] = -1;  // 过滤失败
    __syncthreads();  // 同步一下 否则对性能有影响
  }
}

// kernel_wordFilterPro 短词过滤蛋白 ok
__global__ void kernel_wordFilterPro(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, int wordLength,
float similarity, unsigned int *listTemp) {
  int netLength1 = *(represent+1);
  if (netLength1 < 16) return;  // 净长度太短 跳过短词过滤
  // 准备数据
  __shared__ unsigned int list[1024];
  for (int i=threadIdx.x; i<1024; i+=blockDim.x) {
    list[i] = listTemp[i];
  }
  unsigned short mask = (1<<wordLength*5)-1;  // 短词的mask
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    // 准备数据
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 长度
    int netLength2 = *(bufBegin+1);  // 净长度
    if (netLength2 < 16) return;  // 净长度太短 跳过短词过滤
    unsigned int *packed2 = bufBegin+30;  // 打包数据
    //　计算
    int sum = 0;
    unsigned short word = 0;  // 短词
    for (int i=0; i<netLength2; i+=32, packed2+=5) {
      for (int j=0; j<32; j++) {
        if (i+j == netLength2) break; 
        word <<= 5;
        word += (*(packed2+0)>>j&1)<<0;
        word += (*(packed2+1)>>j&1)<<1;
        word += (*(packed2+2)>>j&1)<<2;
        word += (*(packed2+3)>>j&1)<<3;
        word += (*(packed2+4)>>j&1)<<4;
        word &= mask;
        int order = word/32;
        unsigned int key = 1<<word%32;
        if (i+j > wordLength-2) sum += ((list[order]&key) > 0);
      }
    }
    // 处理结果
    int len = length2-wordLength+1;  // 理论短词数
    int length = ceil((float)length2*(1.0f-similarity));
    int minLen = len-length*wordLength;
    if (sum < minLen) jobs[idx*2+1] = -1;  // 过滤失败
    __syncthreads();  // 同步一下 否则对性能有影响
  }
}

// kernel_dynamicGen 动态规划基因 ok
__global__ void kernel_dynamicGen(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, float similarity) {
  // 准备数据1
  int length1 = *represent;  // 长度
  int netLength1 = *(represent+1);  // 净长度
  int netLen321 = (netLength1+31)/32;  // 净长度32补齐个数
  unsigned int *packed1 = represent+6;  // 会自动缓存 别用shared memory
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    // 准备数据2 指针偏移和下标的速度一样
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 长度
    int netLength2 = *(bufBegin+1);  // 净长度
    int netLen322 = (netLength2+31)/32;  // 净32补齐
    unsigned int *packed2 = bufBegin+6;  // 打包数据
    // 准备中间数据 1是行2是列
    unsigned int line[2048];  // 保存结果 二进制 2048=65536/32
    memset(line, 0xFF, 2048*sizeof(unsigned int));  // 0:匹配 1:不匹配
    int shift = ceil((float)length1-(float)length2*similarity);  // 偏移量
    shift = (shift+31)/32;  // 偏移块数
    int cols = 32;  // 竖向计算行数
    // 计算
    for (int i=0; i<netLen322; i++) {  // 遍历列
      if ((i+1)*32 > netLength2) cols = netLength2%32;
      unsigned int column[32] = {0};  // 进位
      unsigned int Col0 = packed2[i*2+0];  // 低位
      unsigned int Col1 = packed2[i*2+1];  // 高位
      int jstart = max(i-shift, 0);
      int jend = min(i+shift, netLen321-1);
      for (int j=jstart; j<=jend; j++) {  // 遍历行
        unsigned int Row0 = packed1[j*2+0];  // 低位
        unsigned int Row1 = packed1[j*2+1];  // 高位
        unsigned int row = line[j];  // 上一行结果
        for (int k=0; k<cols; k++) {  // 32*32的核心
          unsigned int col0 = 0x00000000;  // 1位扩展成32位
          if (Col0>>k&1) col0 = 0xFFFFFFFF;
          unsigned int col1 = 0x00000000;
          if (Col1>>k&1) col1 = 0xFFFFFFFF;
          unsigned int temp0 = Row0 ^ col0;
          unsigned int temp1 = Row1 ^ col1;
          unsigned int match = (~temp0)&(~temp1);
          unsigned int unmatch = ~match;
          unsigned int temp3 = row & match;
          unsigned int temp4 = row & unmatch;
          unsigned int carry = column[k];
          unsigned int temp5 = row + carry;
          unsigned int carry1 = temp5 < row;
          temp5 += temp3;
          unsigned int carry2 = temp5 < temp3;
          carry = carry1 | carry2;
          row = temp5 | temp4;
          column[k] = carry;
        }
        line[j] = row;
      }
    }
    // 统计结果
    int sum = 0;  // 比对得分
    int rows = 32;  // 横向计算列数
    for (int i=0; i<netLen321; i++) {
      if ((i+1)*32 > netLength1) rows = netLength1%32;
      unsigned int row = line[i];
      for (int j=0; j<rows; j++) sum += row>>j&1^1;
    }
    int cutoff = ceil((float)length2*similarity);
    if (sum < cutoff) jobs[idx*2+1] = -1;
  }
}

// kernel_dynamicPro 动态规划 蛋白 ok
__global__ void kernel_dynamicPro(unsigned int *buf, long *offsets,
unsigned int *represent, int *jobs, int jobCount, float similarity) {
  // 准备数据1
  int length1 = *represent;  // 长度
  int netLength1 = *(represent+1);  // 净长度
  int netLen321 = (netLength1+31)/32;  // 净长度32补齐个数
  unsigned int *packed1 = represent+30;  // 会自动缓存 别用shared memory
  int index = blockDim.x*blockIdx.x+threadIdx.x;  // 线程编号
  int loop = gridDim.x*blockDim.x;
  for (int idx=index; idx<jobCount; idx+=loop) {
    // 准备数据2 指针偏移和下标的速度一样
    unsigned int *bufBegin = buf+offsets[jobs[idx*2]];  // 起始位置
    int length2 = *bufBegin;  // 长度
    int netLength2 = *(bufBegin+1);  // 净长度
    int netLen322 = (netLength2+31)/32;  // 净32补齐
    unsigned int *packed2 = bufBegin+30;  // 打包数据
    // 准备中间数据 1是行2是列
    unsigned int line[2048];  // 保存结果 二进制 2048=65536/32
    memset(line, 0xFF, 2048*sizeof(unsigned int));  // 0:匹配 1:不匹配
    int shift = ceil((float)length1-(float)length2*similarity);  // 偏移量
    shift = (shift+31)/32;  // 偏移块数
    int cols = 32;  // 竖向计算行数
    // 计算
    for (int i=0; i<netLen322; i++) {  // 遍历列
      if ((i+1)*32 > netLength2) cols = netLength2%32;
      unsigned int column[32] = {0};  // 进位
      unsigned int Col0 = packed2[i*5+0];  // 低位
      unsigned int Col1 = packed2[i*5+1];
      unsigned int Col2 = packed2[i*5+2];
      unsigned int Col3 = packed2[i*5+3];
      unsigned int Col4 = packed2[i*5+4];  // 高位
      int jstart = max(i-shift, 0);
      int jend = min(i+shift, netLen321-1);
      for (int j=jstart; j<=jend; j++) {  // 遍历行
        unsigned int Row0 = packed1[j*5+0];  // 低位
        unsigned int Row1 = packed1[j*5+1];
        unsigned int Row2 = packed1[j*5+2];
        unsigned int Row3 = packed1[j*5+3];
        unsigned int Row4 = packed1[j*5+4];  // 高位
        unsigned int row = line[j];  // 上一行结果
        for (int k=0; k<cols; k++) {  // 32*32的核心
          unsigned int col0 = 0x00000000;  // 1位扩展成32位
          if (Col0>>k&1) col0 = 0xFFFFFFFF;
          unsigned int col1 = 0x00000000;
          if (Col1>>k&1) col1 = 0xFFFFFFFF;
          unsigned int col2 = 0x00000000;
          if (Col2>>k&1) col2 = 0xFFFFFFFF;
          unsigned int col3 = 0x00000000;
          if (Col3>>k&1) col3 = 0xFFFFFFFF;
          unsigned int col4 = 0x00000000;
          if (Col4>>k&1) col4 = 0xFFFFFFFF;
          unsigned int temp0 = Row0 ^ col0;
          unsigned int temp1 = Row1 ^ col1;
          unsigned int temp2 = Row2 ^ col2;
          unsigned int temp3 = Row3 ^ col3;
          unsigned int temp4 = Row4 ^ col4;
          unsigned int match = (~temp0)&(~temp1)&(~temp2)&(~temp3)&(~temp4);
          unsigned int unmatch = ~match;
          temp3 = row & match;
          temp4 = row & unmatch;
          unsigned int carry = column[k];
          unsigned int temp5 = row + carry;
          unsigned int carry1 = temp5 < row;
          temp5 += temp3;
          unsigned int carry2 = temp5 < temp3;
          carry = carry1 | carry2;
          row = temp5 | temp4;
          column[k] = carry;
        }
        line[j] = row;
      }
    }
    // 统计结果
    int sum = 0;  // 比对得分
    int rows = 32;  // 横向计算列数
    for (int i=0; i<netLen321; i++) {
      if ((i+1)*32 > netLength1) rows = netLength1%32;
      unsigned int row = line[i];
      for (int j=0; j<rows; j++) sum += row>>j&1^1;
    }
    int cutoff = ceil((float)length2*similarity);
    if (sum < cutoff) jobs[idx*2+1] = -1;
  }
}

// clustering 聚类
void clustering(Option &option, Data &data, Bench &bench) {
  // 常用数据
  int rank = option.rank;  // MPI进程号
  int wordLength = option.wordLength;  // 短词长度
  int precise = option.precise;  // 是否启用精确模式
  int type = option.type;  // 输入数据类型
  float similarity = option.similarity;  // 相似度阈值
  int readsCountSum = option.readsCountSum;  // 总序列数
  initBench(option, bench);  // 初始化bench
  while (true) {  // 聚类
    {  // 准备工作
      updateRepresent(option, data, bench);  // 更新代表序列
      if (rank == 0) {  // 打印进度
        std::cout << "\r" << *bench.tops << "/" << readsCountSum << std::flush;
        if (*bench.tops == readsCountSum) std::cout << "\n";
      }
      if (*bench.tops == readsCountSum) break;  // 聚类完成
      if (type == 0) {  // 基因序列
        generateListGen(wordLength, bench);  // 生成短词列表
      } else {  // 蛋白序列
        generateListPro(wordLength, bench);  // 生成短词列表
      }
    }
    {  // 前置过滤 ok
      updateJobs(bench);
      if (option.type == 0) {  // 基因序列
        kernel_baseFilterGen<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d,
        bench.jobs_d, bench.jobCount, similarity);  // 前置过滤
      } else {  // 蛋白序列
        kernel_baseFilterPro<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d,
        bench.jobs_d, bench.jobCount, similarity);  // 前置过滤
      }
      cudaDeviceSynchronize();
      int byteCount = sizeof(int)*bench.jobCount*2;  // 要拷贝字节数
      cudaMemcpy(bench.jobs_h, bench.jobs_d, byteCount, DTH);
    }
    if (!precise) {  // 短词过滤 ok
      updateJobs(bench);
      if (option.type == 0) {  // 基因
        kernel_wordFilterGen<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d, bench.jobs_d, 
        bench.jobCount, wordLength, similarity, bench.list);  // 短词过滤
      } else {  // 蛋白
        kernel_wordFilterPro<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d, bench.jobs_d, 
        bench.jobCount, wordLength, similarity, bench.list);  // 短词过滤
      }
      cudaDeviceSynchronize();
      int byteCount = sizeof(int)*bench.jobCount*2;  // 要拷贝字节数
      cudaMemcpy(bench.jobs_h, bench.jobs_d, byteCount, DTH);
    }
    {  // 序列比对 ok
      updateJobs(bench);
      if (option.type == 0) {  // 基因
        kernel_dynamicGen<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d,
        bench.jobs_d, bench.jobCount, similarity);  // 动态规划
      } else {  // 蛋白
        kernel_dynamicPro<<<BLOCK, WARP>>>
        (data.buf_d, data.offsets_d, bench.represent_d,
        bench.jobs_d, bench.jobCount, similarity);  // 动态规划
      }
      cudaDeviceSynchronize();
      int byteCount = sizeof(int)*bench.jobCount*2;  // 要拷贝字节数
      cudaMemcpy(bench.jobs_h, bench.jobs_d, byteCount, DTH);
    }
  }
  if (rank == 0) std::cout << std::endl;  // 隔开下个信息块
}

// saveResult 保存结果 ok
void saveResult(Option &option, Index &index, Bench &bench) {
  // 常用数据
  int rank = option.rank;  // MPI进程号
  int size = option.size;  // MPI进程数
  int readsCount = option.readsCount;  // 每线程处理序列数
  int readsCountSum = option.readsCountSum;  // 总序列数
  std::vector<std::vector<int>> represents;  // 代表序列
  std::vector<long> reprOffsets;  // represets.txt的偏移
  std::vector<long> clusOffsets;  // cluster.fasta的偏移
  std::string line;  // 一行数据
  int count = 0;  // 代表序列个数
  Timer timer; timer.start();  // 计时器
  { // 汇总结果
    represents.resize(readsCountSum);  // 聚类的结果
    int *cluster;  // 汇总聚类结果
    cudaMallocHost(&cluster, sizeof(int)*readsCount*size);
    memcpy(cluster, bench.cluster, sizeof(int)*readsCount);
    // MPI_Gather(bench.cluster, readsCount, MPI_INT, cluster,
    //   readsCount, MPI_INT, 0, MPI_COMM_WORLD);  // 汇总
    // MPI_Bcast(cluster, readsCount*size, MPI_INT, 0, MPI_COMM_WORLD);  // 广播
    for (int i=0; i<readsCountSum; i++) {  // 遍历结果 找代表序列
      int index = i%size*readsCount+i/size;
      represents[cluster[index]].push_back(i);  // 归到类中
    }
    cudaFreeHost(cluster);  // 释放
    for (int i=0; i<readsCountSum; i++) count += (represents[i].size()>0);
    if (rank == 0) std::cout << "Cluster:\t" << count << std::endl;
  }
  {  // 计算偏移
    clusOffsets.resize(readsCountSum);  // cluster.fasta的偏移
    reprOffsets.resize(readsCountSum);  // represents.txt的偏移
    long clusOffset=0, reprOffset=0;  // 上个序列的偏移
    for (int i=0; i<readsCountSum; i++) {  // 遍历所有类
      for (int j=0; j<represents[i].size(); j++) {  // 遍历类中序列
        int idx = represents[i][j];  // 序列的编号
        clusOffsets[idx] = clusOffset;
        clusOffset += index.nameLengths[idx]+3;
        if (j==0) {  // 这个是代表序列
          reprOffsets[idx] = reprOffset;
          reprOffset += index.nameLengths[idx]+index.readLengths[idx]+2;
          clusOffset -= 2;  // 代表序列不缩进
        }
      }
    }
  }
  {  // 写入结果
    std::ifstream file(option.inputDir+"/data.fasta");  // 输入文件
    std::ofstream clusFile(option.outputDir+"/cluster.txt");  // 聚类文件
    std::ofstream reprFile(option.outputDir+"/represent.fasta");  // 代表序列
    std::string line, name, read;  // 一行数据 序列名 序列数据
    for (int i=rank; i<readsCountSum; i+=size) {  // 遍历序列
      if (rank == 0 && i%(1000*size) == 0) std::cout << "\rWrite data: "
        << i << "/" << readsCountSum << std::flush;  // 打印进度
      file.seekg(index.offsets[i], std::ios::beg);  // 跳到序列开头
      getline(file, line);  // 读序列名
      if (line[line.size()-1] == '\r') line.pop_back();  // 去除\r
      name = line;  // 序列名
      read.clear();  // 序列数据
      while (file.peek() != EOF && file.peek() != '>') {  // 读序列
        getline(file, line);
        if (line[line.size()-1] == '\r') line.pop_back();  // 去除\r
        read += line;
      }
      if (represents[i].size()>0) {  // 是代表序列
        // 先写cluster.txt
        clusFile.seekp(clusOffsets[i], std::ios::beg);
        clusFile << name+"\n";
        reprFile.seekp(reprOffsets[i], std::ios::beg);
        reprFile << name+"\n"+read+"\n";
      } else {  // 类内序列
        clusFile.seekp(clusOffsets[i], std::ios::beg);
        clusFile << "  "+name+"\n";
      }
    }
    if (rank == 0) {
      std::cout <<"\rWrite data: "<<readsCountSum <<"/"<<readsCountSum<<"\n";
      std::cout << "Save file..." << std::flush;
    }
    file.close();
    clusFile.close();
    reprFile.close();
    if (rank == 0) std::cout << "finish\n";
  }
  if (rank == 0) {std::cout << "Save result:\t"; timer.getDuration();}
}

// finish 收尾 ok
void finish(Data &data, Bench &bench) {
  cudaFreeHost(data.buf_h);  // 释放空间
  cudaFree(data.buf_d);
  cudaFreeHost(data.offsets_h);
  cudaFree(data.offsets_d);
  cudaFreeHost(bench.tops);
  cudaFreeHost(bench.represent_h);
  cudaFree(bench.represent_d);
  cudaFreeHost(bench.remains);
  cudaFreeHost(bench.jobs_h);
  cudaFree(bench.jobs_d);
  cudaFreeHost(bench.cluster);
  cudaFree(bench.list);
  cudaDeviceSynchronize();  // 同步
}
