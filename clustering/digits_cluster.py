from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

np.random.seed(42)
print()

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
n_digits = len(np.unique(digits.target))
labels = digits.target

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tNMI\t\thomo\tcompl\t')


def bench(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t'
          % (name, (time() - t0),
             metrics.normalized_mutual_info_score(labels, estimator.labels_,average_method='arithmetic'),
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             ))

bench(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="k-means", data=data)

bench(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

#参数damping: float, optional, default: 0.5
#Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping).
#This in order to avoid numerical oscillations when updating these values (messages).
#参数preference: array-like, shape (n_samples,) or float, optional
#The number of exemplars, ie of clusters, is influenced by the input preferences value.
bench(AffinityPropagation(damping=0.5, preference=None),name="affinity", data=data)

#参数bandwidth: float, optional
#Bandwidth used in the RBF kernel.
#If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth;
bench(MeanShift(bandwidth=0.8),name="MeanShift", data=data)

#参数n_clusters: integer, optional
#The dimension of the projection subspace.
#affinity:nearest_neighbors' ,precomputed
bench(SpectralClustering(n_clusters=n_digits,affinity="nearest_neighbors"),name="SpectralClustering", data=data)


#参数linkage: optional (default=”ward”)
#ward minimizes the variance of the clusters being merged.
#参数n_clusters: int, default
#The number of clusters to find.
bench(AgglomerativeClustering(linkage='ward',n_clusters=n_digits),name="WardHiera", data=data)

# 参数linkage:  {“ward”, “complete”, “average”, “single”}, optional
# average uses the average of the distances of each observation of the two sets.
# 参数n_clusters: int, default
# The number of clusters to find
bench(AgglomerativeClustering(linkage='average',n_clusters=n_digits),name="Agglomera", data=data)

#参数eps: float, optional  The maximum distance between two samples for them to be considered as in the same neighborhood
#参数min_samples: int, optional  The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
#This includes the point itself.
bench(DBSCAN(eps=0.5,min_samples=1,metric='cosine'),name="DBSCAN", data=data)

#n_components ：高斯模型的个数，即聚类的目标个数
#covariance_type : 通过EM算法估算参数时使用的协方差类型，默认是”full”
#full：每个模型使用自己的一般协方差矩阵
#tied：所用模型共享一个一般协方差矩阵
#diag：每个模型使用自己的对角线协方差矩阵
#spherical：每个模型使用自己的单一方差
#bench(GaussianMixture(n_components=n_digits,covariance_type='full'),name="Gaussian", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1

print(82 * '_')