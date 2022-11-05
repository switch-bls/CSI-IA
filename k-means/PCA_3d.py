import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_frame = pd.read_csv('../all_e.csv')

model = PCA(n_components = 3)
model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)
pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 2, c=data_frame_reduced[:,2], cmap='viridis', linewidth=0.5)
plt.show()