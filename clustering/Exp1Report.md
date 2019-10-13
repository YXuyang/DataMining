# Homework1 实验报告
姓名：袁旭阳      学号：201914693
  实验名称：Clustering with sklearn

## 一、实验要求
数据集：<br>
![Image text](https://raw.githubusercontent.com/YXuyang/DataMining/master/clustering/img-folder/datas.png)

实验要求：测试sklearn中以下聚类算法在以上两个数据集上的聚类效果
![Image text](https://raw.githubusercontent.com/YXuyang/DataMining/master/clustering/img-folder/clustermethod.png)

评估指标：<br>
![Image text](https://raw.githubusercontent.com/YXuyang/DataMining/master/clustering/img-folder/evaluation.png)
<br>
<br>
## 二、实验过程

###（一）数据处理
两个数据集均从sklearn集成的数据集中获取，分别是digit数据集和20newsgroups数据集

#### digit数据集
加载方式：<br>
`sklearn.datasets.load_digits`
<br>
数据集介绍：<br>
手写体数字图像数据集，训练数据样本3823条，测试数据1797条，图像数据通过8X8的像素矩阵表示，共有64个像素维度。
1个目标维度用来标记每个图像样本代表的数字类别。
该数据没有缺失的特征值，并且不论是训练还是测试样本.在数字类别方面都采样得非常平均，是一份非常规整的数据集。


#### 20newsgroups数据集
加载方式：<br>
`sklearn.datasets.fetch_20newsgroups`
<br>
数据集介绍：<br>
20newsgroups数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。
数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合

###（二）聚类算法调用与评估



###（三）比较 8 种聚类

## 三、实验结果