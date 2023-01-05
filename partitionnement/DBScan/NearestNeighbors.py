from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



dataset = pd.read_csv('../data/1000_final_data.csv')

neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(dataset)
distances, indices = neighbors_fit.kneighbors(dataset)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

plt.show()