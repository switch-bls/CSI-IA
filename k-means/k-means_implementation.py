import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import numpy as np

def euclidianDist(row1, row2):
    sum = 0     
    for d in range(0, row1.size): sum += pow(row1[d] - row2[d],2)
    return np.sqrt(sum)

def kmeansImp(nb_centroids, data):
    #creation des centroids
    data_size = len(data.index)

    centroids = []
    for i in range(0, nb_centroids):
        rand = random.randint(0, data_size)
        centroids.append(data.loc[rand])
    
    

    
    nb_points_par_centroid = [0 for i in range(nb_centroids)]

    for index_row, row in data.iterrows():
        dist_min = 1000000000
        index_min = 0
        for index_cent,centroid in centroids:
            distance = euclidianDist(row,centroid)
            if distance < dist_min :
                dist_min = data
                index_min = index_cent
        
        nb_points_par_centroid += 1
            
    for centroid in centroids: print(centroid[0])


data_frame = pd.read_csv('../all_e_10k.csv')

data_dict = data_frame.to_dict('index')

kmeansImp(3, data_frame)

#LABEL_COLOR_MAP = {0 : 'b', 1 : 'r', 2 : 'g', 3 : 'c',4 : 'm', 5 : 'y', 6 : 'lime'}

#label_color = [LABEL_COLOR_MAP[l] for l in model.predict(data_frame)]


#model = PCA(n_components = 2)
#model.fit_transform(data_frame)
#data_frame_reduced = model.fit_transform(data_frame)

#pca_data_frame = plt.scatter(data_frame_reduced[:,0],data_frame_reduced[:,1], c=label_color)

#print(model.components_.shape)

#plt.show()


