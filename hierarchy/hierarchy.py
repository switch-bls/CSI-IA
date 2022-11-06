import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

data_frame = pd.read_csv('../all_e_10k.csv')
linked = linkage(data_frame, method="ward")
labelList = data_frame['n']

dendrogram(linked,
 orientation= 'top',
 distance_sort= 'ascending',
 show_leaf_counts= True)

plt.show()