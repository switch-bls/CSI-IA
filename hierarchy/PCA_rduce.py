import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

data_frame = pd.read_csv('../data/1000_final_data.csv')

model = PCA(n_components = 189)
model.fit_transform(data_frame)

print(model.explained_variance_ratio_)

print(np.cumsum(model.explained_variance_ratio_))