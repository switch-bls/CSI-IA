import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import numpy as np

def faire_converger_centroids(centroids, data):
    nb_centroids = len(centroids)
    data_size = len(data.index)
    done = False
    for l in range(30):
        nb_points_par_centroid = [0 for i in range(nb_centroids)]
        somme_des_points = [[0 for j in range(centroids[0].size)]  for i in range(nb_centroids)]

        for index_row, row in data.iterrows():
            dist_min = 1000000000
            index_min = 0

            for index_cent, centroid in enumerate(centroids):
                distance = euclidianDist(row,centroid)
                if distance < dist_min :
                    dist_min = distance
                    index_min = index_cent

            nb_points_par_centroid[index_min] += 1
            
            somme_des_points[index_min] = [somme_des_points[index_min][i] + row[i] for i in range(row.size)]

        nouveaux_centroids =  [[somme_des_points[i][j]/nb_points_par_centroid[i] for j in range(len(somme_des_points[0]))] for i in range(len(somme_des_points))]
        done = True
        centroids.ndarray.tolist
        for i in range(len(centroids)):
            print(type(centroids[i]))
            print(type(nouveaux_centroids[i]))
            if centroids[i] != nouveaux_centroids[i]:
                done = False
                break
        print(nouveaux_centroids)
    return nouveaux_centroids
 



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

    faire_converger_centroids(centroids, data)

    clusters = [[] for i in range(nb_centroids)]

    for index_row, row in data.iterrows():
        dist_min = 1000000000
        index_min = 0

        for index_cent, centroid in enumerate(centroids):
            distance = euclidianDist(row,centroid)
            if distance < dist_min :
                dist_min = distance
                index_min = index_cent

        clusters[index_min].append(index_row)


    colors = [0 for i in range(len(data_frame.index))]
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            colors[clusters[i][j]] = i

    return colors

data_frame = pd.read_csv('../all_e_10k.csv')

c = kmeansImp(3, data_frame)

model = PCA(n_components = 3)

model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)
pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 2,  c = c)

plt.show()
