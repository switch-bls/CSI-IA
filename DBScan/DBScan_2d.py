import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import animation
import numpy as np
from sklearn.cluster import DBSCAN

data_frame = pd.read_csv('../all_e.csv')
 
fig = plt.figure()
clustering = DBSCAN(eps=1700, min_samples=35).fit(data_frame)
labels = clustering.labels_

print(labels)
pca = PCA(n_components=2)
pca.fit(data_frame)
data_frame_reduced = pca.fit_transform(data_frame)

pca_data_frame = plt.scatter(x=data_frame_reduced[:,0],y=data_frame_reduced[:,1], s = 2, c=labels)
plt.show()
