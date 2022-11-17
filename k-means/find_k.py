import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data_frame = pd.read_csv('../data/1000_final_no_model_e.csv')

inertia = []
K_range = range(1,20)
for k in K_range:
    model = KMeans(n_clusters = k).fit(data_frame)
    inertia.append(model.inertia_)
plt.plot(K_range, inertia)
plt.xlabel('Nombre de Clusters')
plt.ylabel('Cout du Modele (Inertia)')

plt.show()
