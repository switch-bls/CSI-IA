import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import animation
import numpy as np
data_frame = pd.read_csv('../all_e.csv')
 
LABEL_COLOR_MAP = {0 : 'b', 1 : 'r', 2 : 'g', 3 : 'c',4 : 'm', 5 : 'y', 6 : 'lime'}

model = KMeans(n_clusters = 7).fit(data_frame)

label_color = [LABEL_COLOR_MAP[l] for l in model.predict(data_frame)]
fig = plt.figure()


model = PCA(n_components = 3)
model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)



pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 2, c=label_color)
plt.show()

#def rotate(angle):
#    pca_data_frame.view_init(azim=angle)
#angle = 2
#ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
#ani.save('inhadr_tsne1.gif', writer=animation.PillowWriter(fps=30))
