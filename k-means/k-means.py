import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data_frame = pd.read_csv('../all_e_partition.csv')
 
model = KMeans(n_clusters = 7).fit(data_frame)

label_color = model.predict(data_frame)

data_frame['k-means'] = label_color
data_frame.to_csv("final_data_partition.csv", index = False)
model = PCA(n_components = 2)
model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)

pca_data_frame = plt.scatter(data_frame_reduced[:,0],data_frame_reduced[:,1], c=label_color)

print(model.components_.shape)

plt.show()