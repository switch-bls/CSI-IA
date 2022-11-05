import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data_frame = pd.read_csv('../all_e.csv')

model = PCA(n_components = 2)
model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)

pca_data_frame = plt.scatter(data_frame_reduced[:,0],data_frame_reduced[:,1])
plt.colorbar()
plt.show()