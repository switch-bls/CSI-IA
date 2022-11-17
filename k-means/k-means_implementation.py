import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import numpy as np

def faire_converger_centroids(centroids, data):
    nb_centroids = len(centroids)
    data_size = len(data.index)
    dim = centroids[0].size
    done = False
    iter = 0
    while not done:
        nb_points_par_centroid = [0 for i in range(nb_centroids)]
        somme_des_points = [[0 for j in range(dim)]  for i in range(nb_centroids)]

        #Pour chaque ligne dans la bdd
        for index_row, row in data.iterrows():
            dist_min = 1000000000
            index_min = 0
            # On attribue la ligne au centroide le plus proche
            for index_cent, centroid in enumerate(centroids):
                distance = euclidianDist(dim,row,centroid)
                if distance < dist_min :
                    dist_min = distance
                    index_min = index_cent

            nb_points_par_centroid[index_min] += 1
            
            somme_des_points[index_min] = [somme_des_points[index_min][i] + row[i] for i in range(dim)]

        #On calcul les nouveau centroids
        nouveaux_centroids =  [[somme_des_points[i][j]/nb_points_par_centroid[i] for j in range(len(somme_des_points[0]))] for i in range(len(somme_des_points))]
        done = True
        
        #Si les centroids ne change plus de positions ils ne peuvent plus converger alors on a terminé
        for i in range(len(centroids)):
            for j in range(len(centroids)):
                if centroids[i][j] != nouveaux_centroids[i][j]:
                    done = False
                    break
        centroids = nouveaux_centroids
        print("Iteration : " + str(iter))
        iter += 1
    return nouveaux_centroids
 



def euclidianDist(dim,row1, row2):
    sum = 0     
    for d in range(0, dim): sum += pow(row1[d] - row2[d],2)
    return np.sqrt(sum)

def kmeansImp(nb_centroids, data):
    #creation des centroids
    data_size = len(data.index)
    dim = len(data.columns)
    centroids = []

    #Les centroids sont placé sur des données choisi aléatoirement dans la base
    for i in range(0, nb_centroids):
        rand = random.randint(0, data_size)
        centroids.append(data.loc[rand])

    faire_converger_centroids(centroids, data)

    clusters = [[] for i in range(nb_centroids)]
    #On definit les clusther finaux
    for index_row, row in data.iterrows():
        dist_min = 1000000000
        index_min = 0

        for index_cent, centroid in enumerate(centroids):
            distance = euclidianDist(dim,row,centroid)
            if distance < dist_min :
                dist_min = distance
                index_min = index_cent

        clusters[index_min].append(index_row)

    #creer un tableau de valeur qui associe une donnée a un numero de clusther
    colors = [0 for i in range(len(data_frame.index))]
    for i in range(len(clusters)):
        for j in range(len(clusters[i])):
            colors[clusters[i][j]] = i

    return colors

data_frame = pd.read_csv('../data/final_data.csv')

c = kmeansImp(7, data_frame)

model = PCA(n_components = 3)

model.fit_transform(data_frame)
data_frame_reduced = model.fit_transform(data_frame)
pca_data_frame = plt.axes(projection='3d')
pca_data_frame.scatter3D(xs=data_frame_reduced[:,0],ys=data_frame_reduced[:,1],zs=data_frame_reduced[:,2], s = 2,  c = c)

plt.show()
