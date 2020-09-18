#Clustering using iris dataset

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns     #enhances graphing features
df = data('iris')
df.head()
df.Species.value_counts()
df1 = df.select_dtypes(exclude=['object'])
df1.head()

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()  #scaling data
df1_scaled = scalar.fit_transform(df1)
type(df1_scaled)
df1_scaled.describe() #it converts to different format
pd.DataFrame(df1_scaled).describe()
pd.DataFrame(df1_scaled).head()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  #hyper parameters

kmeans.fit(df1_scaled)
kmeans.inertia_  #sum of sq distances of samples to their centeroid  #sholud be as low as possible, ideally = 0   #0 means each point is its own cluster
kmeans.cluster_centers_  #-ve values, clustering not possible
kmeans.labels_
df1_scaled.shape
kmeans.n_iter_  #iterations to stabilise the clusters
kmeans.predict(df1)
df1.head()

clusterNos = kmeans.labels_
clusterNos
type(clusterNos)

df1.groupby([clusterNos]).mean()
pd.options.display.max_columns =None
df1.groupby([clusterNos]).mean()
df.groupby(['Species']).mean() #0 = versicolor, 1 = verginica, 2 = setosa

#%%
#hierarchical clustering
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))

plt.figure(figsize = (10,7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(df1_scaled, method='ward'))
plt.axhline(y=15, color='r', linestyle='--')
plt.show();

#another method for Hcluster from sklearn
from sklearn.cluster import AgglomerativeClustering
aggCluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
aggCluster.fit_predict(df1_scaled)
aggCluster
aggCluster.labels_
