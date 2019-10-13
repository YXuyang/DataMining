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

### （一）数据处理
两个数据集均从sklearn集成的数据集中获取，分别是digit数据集和20newsgroups数据集

#### digit数据集:
加载方式：<br>
`sklearn.datasets.load_digits`
<br>
数据集介绍：<br>
手写体数字图像数据集，训练数据样本3823条，测试数据1797条，图像数据通过8X8的像素矩阵表示，共有64个像素维度。
1个目标维度用来标记每个图像样本代表的数字类别。
该数据没有缺失的特征值，并且不论是训练还是测试样本.在数字类别方面都采样得非常平均，是一份非常规整的数据集。


#### 20newsgroups数据集:
加载方式：<br>
`sklearn.datasets.fetch_20newsgroups`
<br>
数据集介绍：<br>
20newsgroups数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。
数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合.在这里聚类cluster数设为4种

### （二）聚类算法调用与评估
定义函数，分别调用 sklearn 中的八种聚类算法对 Tweets 数据进
行聚类并返回聚类标签。利用 sklearn 自带的 NMI 评估函数对每个聚
类算法聚类效果进行单独评价，得到评估分数


### （三）比较 8 种聚类
#### K-:means：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>


#### Affinity Propagation：
算法简述：
通过在样本对之间发送消息直到收敛的方式来创建聚类。然后使用少量模范样本作为聚类中心来描述数据集，而这些模范样本可以被认为是最能代表数据集中其它数据的样本。在样本对之间发送的消息表示一个样本作为另一个样本的模范样本的 适合程度，适合程度值在根据通信的反馈不断更新。更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。
<br>
参数：<br>
damping : float, optional, default: 0.5       防止更新过程中数值震荡<br>
max_iter : int, optional, default: 200<br>
convergence_iter : int, optional, default: 15
　　如果类簇数目在达到这么多次迭代以后仍然不变的话，就停止迭代。<br>
copy : boolean, optional, default: True
　　Make a copy of input data.<br>
preference : array-like, shape (n_samples,) or float, optional
每个points的preference。具有更大preference的点更可能被选为exemplar。类簇的数目受此值的影响，如果没有传递此参数，它们会被设置成input similarities的中值。<br>
affinity : string, optional, default=``euclidean``
度量距离的方式，推荐precomputed and euclidean这两种，euclidean uses the negative squared euclidean distance between points.<br>
verbose : boolean, optional, default: False
代码：<br>
```
#参数damping: float, optional, default: 0.5
#Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping).
#This in order to avoid numerical oscillations when updating these values (messages).
#参数preference: array-like, shape (n_samples,) or float, optional
#The number of exemplars, ie of clusters, is influenced by the input preferences value.
bench(AffinityPropagation(damping=0.5, preference=None),name="affinity", data=data)
```
<br>

#### Mean-Shift ：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
#### Spectral Clustering：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
#### Ward Hierarchical Clustering：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
#### Agglomerative Clustering：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
#### DBSCAN：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
#### Gaussian Mixture：
算法简述：
1. 将每个对象看作一类，计算两两之间的最小距离；
2. 将距离最小的两个类合并成一个新类；
3. 重新计算新类与所有类之间的距离；
4. 重复2、3，直到所有类最后合并成一类
代码：<br>
```
bench(KMeans(init='random', n_clusters=n_digits, n_init=10),name="k-means", data=data)
```
<br>
## 三、实验结果