from mlxtend.data import iris_data
from mlxtend.plotting import plot_pca_correlation_graph
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import math
data_frame = pd.read_csv('final_data_partition.csv')

feature_names = [data_frame.columns.values]

df = pd.DataFrame()
for i in range (7):
    col_name = []
    tab = []
    for index, values in data_frame.iteritems():
        corr, _ = pearsonr(data_frame[index],data_frame['k-means'+str(i)])
        if corr >= 0.15:
            col_name.append(data_frame[index].name)
            tab.append(round(corr,2))
        print(str(data_frame[index].name) + " " + str(round(corr,2)))
    df = pd.DataFrame()    
    df['col'+str(i)] = col_name
    df['k-means'+str(i)] = tab
    df.to_csv("cor"+str(i)+".csv", index = False)
