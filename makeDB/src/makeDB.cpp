/*
data.bin:
  int 序列数
  int 序列最长
  int 序列最短
  int 序列类型 0:gene 1:protein
  vector<int> 序列名长度
  vector<int> 序列长度
  vector<long> 原始文件中序列偏移
  vector<long> 压缩数据偏移
  压缩数据
    unsigned int 长度
    unsigned int 净长度
    unsigned int 碱基统计*4 或 氨基酸统计*28
    unsigned int 压缩数据
data.fast: 输入文件的软连接
用法:
makeDB -i 输入文件 -o 输出目录 -t 0基因序列 1蛋白序列
2022-04-07 by 鞠震
*/
// 这不是重点 不用管效率 代码越简单越好
// 为支持大数据集 不能把数据都读入内存 要先生成索引

#include <iostream>  // cout
#include <fstream>  // fstream
#include <iomanip>  // setprecision
#include <string>  // string
#include <vector>  // vector
#include <algorithm>  // sort
#include <omp.h>  // openmp
#include "timer.h"  // 计时器
#include "cmdline.h"  // 解析器
//--------数据--------//
struct Option {  // 输入选项
  std::string inputFile;  // 输入文件
  std::string outputDir;  // 输出路径
  int type;  // 数据类型 0基因 1蛋白
};
struct Read {  // 记录一条序列的位置 为了排序
  long offset;  // 序列的起始位置
  long nameLength;  // 序列名长度
  int readLength;  // 序列数据长度 最长65536
};
//--------函数--------//
// parse 解析输入选项
void parse(int argc, char **argv, Option &option) {
  cmdline::parser parser;  // 解析器
  parser.add<std::string>("input", 'i', "input file", true, "");
  parser.add<std::string>("output", 'o', "output directory", true, "");
  parser.add<int>("type", 't', "input type 0:gene 1:protein", true, 0);
  parser.parse_check(argc, argv);
  option.inputFile = parser.get<std::string>("input");
  option.outputDir = parser.get<std::string>("output");
  option.type = parser.get<int>("type");
  if (option.type<0 || option.type>1) {std::cout << "t: 0-1\n"; exit(0);}
  std::cout << "SSD users are recommended to use CPUs with 8 cores or more.\n";
  std::cout << "The bottleneck is small file I/O performance.\n";
  std::cout << "Supports up to 4,294,967,295 sequences, length of 65536.\n";
  std::cout << "\n";  // 隔开下个模块
}
// makeIndex 生成文件索引
void makeIndex(Option &option, std::vector<Read> &reads) {
  std::string cmd = "";  // shell命令
  cmd += "inputFile="+option.inputFile+" && ";  // 输入文件
  cmd += "inputDir=$(cd $(dirname $inputFile); pwd) && ";  // 输入目录绝对路径
  cmd += "realInput=$inputDir/$(basename $inputFile) && ";  // 输入文件绝对路径
  cmd += "outputDir=$(cd "+option.outputDir+"; pwd) && ";  // 输出目录绝对路径
  cmd += "realOutput=$outputDir/data.fasta && ";  // 输出文件绝对路径
  cmd += "ln -sf $realInput $realOutput";  // 建立软链接
  system(cmd.c_str());
  std::ifstream file(option.outputDir+"/data.fasta");  // 打开输入
  file.seekg(0, std::ios::end);
  long endPoint = file.tellg();  // 文件总长度
  file.seekg(0, std::ios::beg);
  std::string line;  // 读入一行
  long logCount = 0;  // 打印进度
  while(file.peek() != EOF) {  // 读到文件结束
    Read read;
    // 读序列名
    read.offset = file.tellg();  // 序列起始位置
    getline(file, line);  // 读序列名
    if (line[line.size()-1] == '\r') line.pop_back();  // 去除\r
    read.nameLength = line.size();  // 序列名长度
    // 读序列数据
    read.readLength = 0;  // 序列数据长度清零
    while (file.peek() != EOF && file.peek() != '>') {
      getline(file, line);
      if (line[line.size()-1] == '\r') line.pop_back();  // 去除\r
      read.readLength += line.size();
    }
    // 写入节点 打印进度
    if (read.readLength <= 65536) reads.push_back(read);  // 序列最长65536
    if (logCount%(1000*100) == 0) std::cout << "\rMake index: "
      << std::setiosflags(std::ios::fixed) << std::setprecision(2)
      << (float)file.tellg()/(float)endPoint*100.0f << "%" << std::flush;
    logCount += 1;
  }
  std::cout << "\rMake index: 100.00%\n";
  std::cout << "Read " << logCount << " and " << reads.size() << " valid.\n";
  file.close();
  // 排序
  std::cout << "Sorting..." << std::flush;
  std::stable_sort(reads.begin(), reads.end(),
  [](const Read &a, const Read &b) {return a.readLength > b.readLength;});
  std::cout << "finish\n";
}
// makeDataGen 生成基因数据
void makeDataGen(std::string &read, std::vector<unsigned int> &buffer) {
  int length = read.size();  // 长度
  buffer.clear(); buffer.resize(6+(length+31)/32*2);  // 分配空间
  unsigned int pack0=0, pack1=0;  // 打包后数据
  unsigned int *packed = buffer.data()+6;  // 打包后数据存储
  int bases[4] = {0};  // 碱基数
  unsigned int base = 0;  // 碱基
  int count = 0;  // 打包了多少碱基
  for (int i=0; i<length; i++) {
    switch (read[i]) {  // 氨基酸转数字
      case 'a': base=0; bases[0]+=1; break;
      case 'c': base=1; bases[1]+=1; break;
      case 'g': base=2; bases[2]+=1; break;
      case 't': base=3; bases[3]+=1; break;
      case 'u': base=3; bases[3]+=1; break;
      case 'A': base=0; bases[0]+=1; break;
      case 'C': base=1; bases[1]+=1; break;
      case 'G': base=2; bases[2]+=1; break;
      case 'T': base=3; bases[3]+=1; break;
      case 'U': base=3; bases[3]+=1; break;
      default : base=4; break;
    }
    if (base < 4) {  // 打包
      pack0 >>= 1; pack1 >>= 1;
      pack0 += ((base>>0)&1)<<31;  // 低位
      pack1 += ((base>>1)&1)<<31;  // 高位
      count += 1;
      if (count%32 == 0) {  // 每32个氨基酸存储一次
        *(packed+0) = pack0;
        *(packed+1) = pack1;
        packed += 2;
      }
    }
  }
  if (count%32 > 0) {  // 需要补齐
    pack0 >>= (32-count%32);
    pack1 >>= (32-count%32);
    *(packed+0) = pack0;
    *(packed+1) = pack1;
  }
  buffer[0] = length;  // 长度
  buffer[1] = count;  // 净长度
  for (int i=0; i<4; i++) buffer[2+i] = bases[i];
}
// makeDataPro 生成蛋白数据
void makeDataPro(std::string &read, std::vector<unsigned int> &buffer) {
  int length = read.size();  // 长度
  buffer.clear(); buffer.resize(30+(length+31)/32*5);  // 分配空间
  unsigned int pack0=0, pack1=0, pack2=0, pack3=0, pack4=0;  // 打包后数据
  unsigned int *packed = buffer.data()+30;  // 打包后数据存储
  int bases[28] = {0};  // 氨基酸数
  unsigned int base = 0;  // 氨基酸
  int count = 0;  // 打包了多少碱基
  for (int i=0; i<length; i++) {
    switch (read[i]) {  // 氨基酸转数字
      case 'a': base= 0; bases[ 0]+=1; break;
      case 'b': base= 1; bases[ 1]+=1; break;
      case 'c': base= 2; bases[ 2]+=1; break;
      case 'd': base= 3; bases[ 3]+=1; break;
      case 'e': base= 4; bases[ 4]+=1; break;
      case 'f': base= 5; bases[ 5]+=1; break;
      case 'g': base= 6; bases[ 6]+=1; break;
      case 'h': base= 7; bases[ 7]+=1; break;
      case 'i': base= 8; bases[ 8]+=1; break;
      case 'j': base= 9; bases[ 9]+=1; break;
      case 'k': base=10; bases[10]+=1; break;
      case 'l': base=11; bases[11]+=1; break;
      case 'm': base=12; bases[12]+=1; break;
      case 'n': base=13; bases[13]+=1; break;
      case 'o': base=14; bases[14]+=1; break;
      case 'p': base=15; bases[15]+=1; break;
      case 'q': base=16; bases[16]+=1; break;
      case 'r': base=17; bases[17]+=1; break;
      case 's': base=18; bases[18]+=1; break;
      case 't': base=19; bases[19]+=1; break;
      case 'u': base=20; bases[20]+=1; break;
      case 'v': base=21; bases[21]+=1; break;
      case 'w': base=22; bases[22]+=1; break;
      case 'x': base=23; bases[23]+=1; break;
      case 'y': base=24; bases[24]+=1; break;
      case 'z': base=25; bases[25]+=1; break;
      case '*': base=26; bases[26]+=1; break;
      case '-': base=27; bases[27]+=1; break;
      case 'A': base= 0; bases[ 0]+=1; break;
      case 'B': base= 1; bases[ 1]+=1; break;
      case 'C': base= 2; bases[ 2]+=1; break;
      case 'D': base= 3; bases[ 3]+=1; break;
      case 'E': base= 4; bases[ 4]+=1; break;
      case 'F': base= 5; bases[ 5]+=1; break;
      case 'G': base= 6; bases[ 6]+=1; break;
      case 'H': base= 7; bases[ 7]+=1; break;
      case 'I': base= 8; bases[ 8]+=1; break;
      case 'J': base= 9; bases[ 9]+=1; break;
      case 'K': base=10; bases[10]+=1; break;
      case 'L': base=11; bases[11]+=1; break;
      case 'M': base=12; bases[12]+=1; break;
      case 'N': base=13; bases[13]+=1; break;
      case 'O': base=14; bases[14]+=1; break;
      case 'P': base=15; bases[15]+=1; break;
      case 'Q': base=16; bases[16]+=1; break;
      case 'R': base=17; bases[17]+=1; break;
      case 'S': base=18; bases[18]+=1; break;
      case 'T': base=19; bases[19]+=1; break;
      case 'U': base=20; bases[20]+=1; break;
      case 'V': base=21; bases[21]+=1; break;
      case 'W': base=22; bases[22]+=1; break;
      case 'X': base=23; bases[23]+=1; break;
      case 'Y': base=24; bases[24]+=1; break;
      case 'Z': base=25; bases[25]+=1; break;
      default : base=28; break;
    }
    if (base < 28) {  // 打包
      pack0 >>= 1; pack1 >>= 1; pack2 >>= 1; pack3 >>= 1; pack4 >>= 1;
      pack0 += ((base>>0)&1)<<31;  // 低位
      pack1 += ((base>>1)&1)<<31;
      pack2 += ((base>>2)&1)<<31;
      pack3 += ((base>>3)&1)<<31;
      pack4 += ((base>>4)&1)<<31;  // 高位
      count += 1;
      if (count%32 == 0) {  // 每32个氨基酸存储一次
        *(packed+0) = pack0;
        *(packed+1) = pack1;
        *(packed+2) = pack2;
        *(packed+3) = pack3;
        *(packed+4) = pack4;
        packed += 5;
      }
    }
  }
  if (count%32 > 0) {  // 需要补齐
    pack0 >>= (32-count%32);
    pack1 >>= (32-count%32);
    pack2 >>= (32-count%32);
    pack3 >>= (32-count%32);
    pack4 >>= (32-count%32);
    *(packed+0) = pack0;
    *(packed+1) = pack1;
    *(packed+2) = pack2;
    *(packed+3) = pack3;
    *(packed+4) = pack4;
  }
  buffer[0] = length;  // 长度
  buffer[1] = count;  // 净长度
  for (int i=0; i<28; i++) buffer[2+i] = bases[i];
}
// makeDB 生成数据库
void makeDB(Option &option, std::vector<Read> &reads) {
  // 常用数据
  int readsCount = reads.size();  // 序列数
  std::vector<int> nameLengths, readLengths;  // 长度
  std::vector<long> offsets, packOffsets;  // 偏移
  std::ofstream dataFile(option.outputDir+"/data.bin");  //存储数据
  {  // 写入 序列种类 序列数 最长 最短
    std::cout << "Write info..." << std::flush;
    dataFile.write((char*)&readsCount, sizeof(int));  // 序列数字
    dataFile.write((char*)&reads[0].readLength, sizeof(int));  // 最长
    dataFile.write((char*)&reads[readsCount-1].readLength, sizeof(int));  // 最短
    dataFile.write((char*)&option.type, sizeof(int));  // 序列类型
    std::cout << "finish\n";
  }
  {  // 计算偏移
    std::cout << "Write index..." << std::flush;
    nameLengths.resize(readsCount);  // 序列名长度
    readLengths.resize(readsCount);  // 原数据长度
    offsets.resize(readsCount);  // 原文件偏移
    for (int i=0; i<readsCount; i++) {  // 长度
      nameLengths[i] =reads[i].nameLength;
      readLengths[i] =reads[i].readLength;
      offsets[i] = reads[i].offset;
    }
    packOffsets.resize(readsCount);  // 压缩后偏移
    packOffsets[0] = sizeof(int)*4+(sizeof(int)+sizeof(long))*readsCount*2;
    for (int i=1; i<readsCount; i++) {  // 计算偏移
      if (option.type == 0) {  // 基因序列的偏移
        int temp = sizeof(unsigned int)*(6+(readLengths[i-1]+31)/32*2);
        packOffsets[i] += packOffsets[i-1]+temp;
      } else {  // 蛋白序列的偏移
        int temp = sizeof(unsigned int)*(30+(readLengths[i-1]+31)/32*5);
        packOffsets[i] += packOffsets[i-1]+temp;
      }
    }
    dataFile.write((char*)nameLengths.data(), sizeof(int)*readsCount);
    dataFile.write((char*)readLengths.data(), sizeof(int)*readsCount);
    dataFile.write((char*)offsets.data(), sizeof(long)*readsCount);
    dataFile.write((char*)packOffsets.data(), sizeof(long)*readsCount);
    dataFile.close();
    std::cout << "finish\n";
  }
  #pragma omp parallel
  {  // 读数据并压缩 90%以上耗时在这里 需要多核处理器
    int thread = omp_get_num_procs();  // 线程数
    int number = omp_get_thread_num();  // 线程编号
    std::ifstream file(option.outputDir+"/data.fasta");  // 读文件
    std::fstream dataFile(option.outputDir+"/data.bin",
      std::ios::in|std::ios::out);  // 写文件 必须io模式 append不能seek
    std::string line, readLine;  // 缓冲 一条序列
    std::vector<unsigned int> buffer;  // 生成的数据
    #pragma omp barrier  // 同步一下
    for (int i=number; i<readsCount; i+=thread) {  // 遍历序列
      file.seekg(offsets[i], std::ios::beg);  // 移到序列起始
      getline(file, line);  // 读序列名
      readLine.clear();  // 用前先清空
      while (file.peek() != EOF && file.peek() != '>') {  // 把多行数据读成一行
        getline(file, line);
        if (line[line.size()-1] == '\r') line.pop_back();  // 去除\r
        readLine += line;
      }
      if (option.type == 0) {  // 处理基因序列
        makeDataGen(readLine, buffer);
      } else {  // 处理蛋白序列
        makeDataPro(readLine, buffer);
      }
      dataFile.seekp(packOffsets[i], std::ios::beg);
      dataFile.write((char*)buffer.data(), sizeof(unsigned int)*buffer.size());
      if (i%(1000*thread) == 0) std::cout << "\rMake data base: "
        << std::setiosflags(std::ios::fixed) << std::setprecision(2)
        << (float)i/(float)readsCount*100.0f << "%" << std::flush;  // 打印进度
    }
    file.close();
    #pragma omp master
    {std::cout << "\rMake data base: 100.00%\nWrite data.bin..." << std::flush;}
    dataFile.close();
    #pragma omp barrier  // 同步一下
    #pragma omp master
    {std::cout << "finish\n";}
  }
}
//--------主函数--------//
int main(int argc, char **argv) {
  Timer timer; timer.start();  // 开始计时
  Option option;  // 输入选项
  parse(argc, argv, option);  // 解析输入选项
  std::cout << "Make data base begin:\t"; timer.getTimeNow();  // 开始时间戳
  std::vector<Read> reads;  // 序列长度与偏移
  makeIndex(option, reads);  // 生成文件索引
  makeDB(option, reads);  // 生成数据库
  std::cout << "Make data base finish:\t"; timer.getTimeNow();  // 结束时间戳
  std::cout << "Make data base total:\t"; timer.getDuration();  // 耗时
}
