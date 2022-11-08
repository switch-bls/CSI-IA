from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# load dataset as pandas dataframe
df = pd.read_csv('../data/final_data.csv')
pca_out = PCA().fit(df)

loadings = pca_out.components_
num_pc = pca_out.n_features_
model = PCA(n_components = 3)
model.fit_transform(df)

pca_data_frame = plt.axes(projection='3d')
for a in range(len(df.column)):
    label = df.column.values[a]
    ax.text(model[x], y, z, label, zdir)

plt.show()