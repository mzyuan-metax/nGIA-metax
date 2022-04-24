#include "func.h"  // 数据结构与函数
#include "timer.h"  // 计时器

int main(int argc, char **argv) {
  Timer timer; timer.start();
  Option option;  // 各种参数
  init(argc, argv, option);  // 初始化
  Index index;  // 数据索引
  readIndex(option, index);  // 读数据索引
  Data data;  // 数据
  readData(option, index, data);  // 读数据
  Bench bench;  // 工作台
  clustering(option, data, bench);  // 聚类
  saveResult(option, index, bench);  // 保存结果
  finish(data, bench);  // 收尾
  if (option.rank == 0) timer.getDuration();
}
