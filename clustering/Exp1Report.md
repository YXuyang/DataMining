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
<br>
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
1.在未被标记的数据点中随机选择一个点作为起始中心点center；
2.找出以center为中心半径为radius的区域中出现的所有数据点，认为这些点同属于一个聚类C。同时在该聚类中记录数据点出现的次数加1。
3.以center为中心点，计算从center开始到集合M中每个元素的向量，将这些向量相加，得到向量shift。
4.center = center + shift。即center沿着shift的方向移动，移动距离是||shift||。
5.重复步骤2、3、4，直到shift的很小（就是迭代到收敛），记住此时的center。注意，这个迭代过程中遇到的点都应该归类到簇C。
6.如果收敛时当前簇C的center与其它已经存在的簇C2中心的距离小于阈值，那么把C2和C合并，数据点出现次数也对应合并。否则，把C作为新的聚类。
7.重复1、2、3、4、5直到所有的点都被标记为已访问。
8.分类：根据每个类，对每个点的访问频率，取访问频率最大的那个类，作为当前点集的所属类。
<br>
代码：<br>
```
#参数bandwidth: float, optional
#Bandwidth used in the RBF kernel.
#If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth;
bench(MeanShift(bandwidth=0.8),name="MeanShift", data=data)
```
<br>

#### Spectral Clustering：

算法简述：<br>
输入：样本集D=(x1,x2,...,xn)，相似矩阵的生成方式, 降维后的维度k1, 聚类方法，聚类后的维度k2 <br>
输出： 簇划分C(c1,c2,...ck2).　<br>
　　　　1) 根据输入的相似矩阵的生成方式构建样本的相似矩阵S<br>
　　　　2）根据相似矩阵S构建邻接矩阵W，构建度矩阵D<br>
　　　　3）计算出拉普拉斯矩阵L<br>
　　　　4）构建标准化后的拉普拉斯矩阵D−1/2LD−1/2<br>
　　　　5）计算D−1/2LD−1/2最小的k1个特征值所各自对应的特征向量f<br>
　　　　6) 将各自对应的特征向量f组成的矩阵按行标准化，最终组成n×k1维的特征矩阵F<br>
　　　　7）对F中的每一行作为一个k1维的样本，共n个样本，用输入的聚类方法进行聚类，聚类维数为k2。<br>
　　　　8）得到簇划分C(c1,c2,...ck2).<br>
<br>
代码：<br>
```
#参数n_clusters: integer, optional
#The dimension of the projection subspace.
#affinity:nearest_neighbors' ,precomputed
bench(SpectralClustering(n_clusters=n_digits,affinity="nearest_neighbors"),name="Spectral", data=data)
```
<br>

#### Ward Hierarchical Clustering：

算法简述：<br>
整个聚类过程其实是建立了一棵树，在建立过程中，可以通过第二步上设置一个阈值，当最近的两个类的距离大于这个阈值，则认为迭代终止。
Ward 最小化所有聚类内的平方差总和。这是一种方差最小化(variance-minimizing )的优化方向， 这是与k-means 的目标函数相似的优化方法，但是用 agglomerative hierarchical的方法处理。
<br>
算法简述：<br>
1 将每个对象看作一类，计算两两之间的最小距离；<br>
2 将距离最小的两个类合并成一个新类；<br>
3 重新计算新类与所有类之间的距离；<br>
4 重复2、3，直到所有类最后合并成一类。<br>
<br>
代码：<br>
```
#参数linkage: optional (default=”ward”)
#ward minimizes the variance of the clusters being merged.
#参数n_clusters: int, default
#The number of clusters to find.
bench(AgglomerativeClustering(linkage='ward',n_clusters=n_digits),name="WardHierarchical", data=data)
```
<br>

#### Agglomerative Clustering：
The AgglomerativeClustering 使用自下而上的方法进行层次聚类:开始是每一个对象是一个聚类， 并且聚类别相继合并在一起。 连接标准决定用于合并策略的度量。
Maximum 或 complete linkage 最小化成对聚类间最远样本距离。<br>
Average linkage 最小化成对聚类间平均样本距离值。<br>
Single linkage 最小化成对聚类间最近样本距离值<br>
算法简述：<br>
1.将每一个元素单独定为一类<br>
2.重复：每一轮都合并指定距离(对指定距离的理解很重要)最小的类<br>
3.直到所有的元素都归为同一类<br>
<br>
代码：<br>
```
# 参数linkage:  {“ward”, “complete”, “average”, “single”}, optional
# average uses the average of the distances of each observation of the two sets.
# 参数n_clusters: int, default
# The number of clusters to find
bench(AgglomerativeClustering(linkage='average',n_clusters=n_digits),name="Agglomerative", data=data)
```
<br>

#### DBSCAN：
算法简述：<br>
输入: 包含n个对象的数据库，半径e，最少数目MinPts;<br>
输出:所有生成的簇，达到密度要求。<br>
1. Repeat<br>
2. 从数据库中抽出一个未处理的点；<br>
3. IF抽出的点是核心点 THEN 找出所有从该点密度可达的对象，形成一个簇；<br>
4. ELSE 抽出的点是边缘点(非核心对象)，跳出本次循环，寻找下一个点；<br>
5. UNTIL 所有的点都被处理。<br>
<br>

代码：<br>
```
#参数eps: float, optional  The maximum distance between two samples for them to be considered as in the same neighborhood
#参数min_samples: int, optional  The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
#This includes the point itself.
bench(DBSCAN(eps=5,min_samples=18),name="DBSCAN", data=data)
```
<br>

#### Gaussian Mixture：
聚类算法大多数采用相似度来判断，而相似度又大多数采用欧式距离长短来衡量，而GMM采用了新的判断依据—–概率，即通过属于某一类的概率大小来判断最终的归属类别<br>
GMM的基本思想就是：任意形状的概率分布都可以用多个高斯分布函数去近似，也就是GMM就是有多个单高斯密度分布组成的，每一个Gaussian叫”Component”，线性的加成在一起就组成了GMM概率密度
<br>
参数：<br>
1. n_components: 混合高斯模型个数，默认为 1 <br>
2. covariance_type: 协方差类型，包括 {‘full’,‘tied’, ‘diag’, ‘spherical’} 四种，full 指每个分量有各自不同的标准协方差矩阵，完全协方差矩阵（元素都不为零）， tied 指所有分量有相同的标准协方差矩阵（HMM 会用到），diag 指每个分量有各自不同对角协方差矩阵（非对角为零，对角不为零）， spherical 指每个分量有各自不同的简单协方差矩阵，球面协方差矩阵（非对角为零，对角完全相同，球面特性），默认‘full’ 完全协方差矩阵 <br>

3. tol：EM 迭代停止阈值，默认为 1e-3. <br>
4. reg_covar: 协方差对角非负正则化，保证协方差矩阵均为正，默认为 0 <br>
5. max_iter: 最大迭代次数，默认 100 <br>
6. n_init: 初始化次数，用于产生最佳初始参数，默认为 1 <br>
7. init_params: {‘kmeans’, ‘random’}, defaults to ‘kmeans’. 初始化参数实现方式，默认用 kmeans 实现，也可以选择随机产生 <br>
8. weights_init: 各组成模型的先验权重，可以自己设，默认按照 7 产生 <br>
9. means_init: 初始化均值，同 8 <br>
10. precisions_init: 初始化精确度（模型个数，特征个数），默认按照 7 实现 <br>
<br>
代码：<br>
```
#n_components ：高斯模型的个数，即聚类的目标个数
#covariance_type : 通过EM算法估算参数时使用的协方差类型，默认是”full”
#full：每个模型使用自己的一般协方差矩阵
#tied：所用模型共享一个一般协方差矩阵
#diag：每个模型使用自己的对角线协方差矩阵
#spherical：每个模型使用自己的单一方差
bench(mixture.GaussianMixture(n_components=n_digits,covariance_type='full'),name="Gaussian", data=data)
```
<br>
## 三、实验结果