import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import animation
import numpy as np
from sklearn.cluster import DBSCAN

data_frame = pd.read_csv('../data/1000_final_data.csv')
 


fig = plt.figure()

eps = []
min_S = []
nb_in_cluster = []
variance_in_cluster = []
for i in range(12):
    for j in range(12):
        e = pow(i+1,2)*4
        m = int(pow(j+1,2))*2
        clustering = DBSCAN(eps=e, min_samples=m).fit(data_frame)
        labels = clustering.labels_

        values = np.unique(labels,return_counts=True)
        {k:v for k,v in zip(*values)}
        eps.append(e)
        min_S.append(m)
        nb_in_cluster.append(len(values[0]))
        variance_in_cluster.append(np.std(values[1]))
        print("epsilon : " + str(e))
        print("min_s : " + str(m))
        print("equart-type : " + str(np.std(values[1])))
        print("nb : " + str(len(values[0])))
        print("")

numpy_array = np.array([eps,min_S,nb_in_cluster,variance_in_cluster])
df = pd.DataFrame(numpy_array)
df.to_csv("test", index = False)

pca = PCA(n_components=3)
pca.fit(data_frame)
data_frame_reduced = pca.fit_transform(data_frame)

pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 4, c=labels)
plt.show()

#def rotate(angle):
#    pca_data_frame.view_init(azim=angle)
#angle = 2
#ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
#ani.save('DBScan_esp1700_MinPts35.gif', writer=animation.PillowWriter(fps=24))
