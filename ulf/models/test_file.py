from sklearn.datasets import load_wine
import numpy as np
from graph_encoder import GraphClustering
import sklearn.metrics.pairwise as pairwise
import scipy.io
mat = scipy.io.loadmat('COIL20.mat')
features = np.array(mat['fea'])
ground_truth = mat['gnd']

ground_truths = []

for m in ground_truth:
    ground_truths.append(m[0])

Y= np.array(ground_truths)

from spectral_clustering import SpectralClustering_

modelu = SpectralClustering_(features,7).spectral_model()
print(modelu)
# data_result = GraphClustering(features).training()

# labels = np.array(Y).transpose()

# from sklearn.cluster import KMeans
# kmeans_sae = KMeans(n_clusters=3, init='random', random_state=None,max_iter=500).fit(data_result)

# def randindex(labels1,labels2):
#     tp,tn,fp,fn = 0.0,0.0,0.0,0.0
#     for point1 in range(len(labels1)):
#         for point2 in range(len(labels2)):
#             tp += 1 if labels1[point1] == labels1[point2] and labels2[point1] == labels2[point2] else 0
#             tn += 1 if labels1[point1] != labels1[point2] and labels2[point1] != labels2[point2] else 0
#             fp += 1 if labels1[point1] != labels1[point2] and labels2[point1] == labels2[point2] else 0
#             fn += 1 if labels1[point1] == labels1[point2] and labels2[point1] != labels2[point2] else 0
#     return (tp+tn) /(tp+tn+fp+fn)

# print(kmeans_sae.labels_)
# print('kmeans_sae is :', randindex(kmeans_sae.labels_, labels))