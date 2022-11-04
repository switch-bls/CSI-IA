import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data_frame = pd.read_csv('all_e.csv')
 
LABEL_COLOR_MAP = {0 : 'b', 1 : 'r', 2 : 'g', 3 : 'c',4 : 'm', 5 : 'y', 6 : 'lime'}

model = KMeans(n_clusters = 7).fit(data_frame)

label_color = [LABEL_COLOR_MAP[l] for l in model.predict(data_frame)]


model = PCA(n_components = 2)
model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)

pca_data_frame = plt.scatter(data_frame_reduced[:,0],data_frame_reduced[:,1], c=label_color)

print(model.components_.shape)

plt.show()