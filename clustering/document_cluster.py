from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2,
                             stop_words='english')
data=vectorizer.fit_transform(dataset.data)

svd = TruncatedSVD(2)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
data= lsa.fit_transform(data)



# print("n_digits: %d, \t n_samples %d, \t n_features %d"
#       % (n_digits, n_samples, n_features))

print('%-18s\t%-5s\t%-5s\t%-5s\t%-5s'%("init","time","NMI","homo","compl"))
# print(digits)
def bench(estimator,name,data):
    t0 = time()
    y_pred=estimator.fit_predict(data)
    print('%-18s\t%.2fs\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0),
             metrics.normalized_mutual_info_score(labels,y_pred,average_method='arithmetic'),
             metrics.homogeneity_score(labels, y_pred),
             metrics.completeness_score(labels, y_pred)
             ))

bench(KMeans(init='random', n_clusters=4, n_init=4),
              name="k-means", data=data)

bench(KMeans(init='k-means++', n_clusters=4, n_init=10),
              name="k-means++", data=data)


bench(AffinityPropagation(damping=0.75, preference=None),name="AffinityPropagation",data=data)

bench(MeanShift(bandwidth=0.0751),name="MeanShift",data=data)

bench(SpectralClustering(n_clusters=4,assign_labels="discretize",
                             random_state=0,affinity="nearest_neighbors"),name="SpectralClustering",data=data)

bench(AgglomerativeClustering(linkage="ward", n_clusters=4),name="WardHierarchical",data=data)

bench(AgglomerativeClustering(n_clusters=4),name="Agglomerative",data=data)

bench(DBSCAN(eps=0.02, min_samples=40),name="DBSCAN",data=data)

bench(mixture.GaussianMixture(n_components=4, covariance_type='full'),name="GaussianMixture",data=data)