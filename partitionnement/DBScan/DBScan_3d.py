import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import animation
import numpy as np
from sklearn.cluster import DBSCAN

data_frame = pd.read_csv('../data/1000_final_data.csv')
 
fig = plt.figure()
clustering = DBSCAN(eps=1500, min_samples=20).fit(data_frame)
labels = clustering.labels_

nb_in_cluster = []
values = np.unique(labels,return_counts=True)
{k:v for k,v in zip(*values)}
print(len(values[1]))
print(np.std(values[1]))
print(values[1])
pca = PCA(n_components=3)
pca.fit(data_frame)
data_frame_reduced = pca.fit_transform(data_frame)

pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 4, c=labels)


def rotate(angle):
    pca_data_frame.view_init(azim=angle)
angle = 2
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save('DBScan_esp1700_MinPts35.gif', writer=animation.PillowWriter(fps=24))
