# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

data_frame = pd.read_csv('../../data/training_data_set.csv')
target = pd.read_csv('../../data/target.csv')

# Fit regression model
regr = DecisionTreeRegressor(max_depth= 30)

n = 2
for i in range(n):
    print("i:",i)
    size = len(data_frame.index)
    nb_elements_by_cluster = int(size//n)/size
    couple = []
    for j in range(n):
        #couple.append([data_frame.iloc[nb_elements_by_cluster*(j):nb_elements_by_cluster*(j+1),:],target.iloc[nb_elements_by_cluster*(j):nb_elements_by_cluster*(j+1),:] ])
        couple.append([data_frame.sample(frac = nb_elements_by_cluster),target.sample(frac = nb_elements_by_cluster)])
    test_data = couple.pop(i)
    for c in couple:
        regr.fit(c[0],c[1])

    
    # Predict
    y_2 = regr.predict(test_data[0])

    pdf = pd.DataFrame({"real": np.array(test_data[1]['price']), "pred": np.array(y_2)})

    accuracy = len(pdf[pdf.real==pdf.pred])/len(pdf)
    print(f"Accuracy = {round(100*accuracy, 2)}%")

    print(f"MAE: {mean_absolute_error(np.array(test_data[1]['price']), regr.predict(test_data[0]))} ")
    print(f"MSE: {mean_squared_error(np.array(test_data[1]['price']), regr.predict(test_data[0]))} ")
    print(f"R2: {round(100*r2_score(np.array(test_data[1]['price']), regr.predict(test_data[0])), 2)}%")

print(f"MAE: {mean_absolute_error(np.array(target['price']), regr.predict(data_frame))} ")
print(f"MSE: {mean_squared_error(np.array(target['price']), regr.predict(data_frame))} ")
print(f"R2: {round(100*r2_score(np.array(target['price']), regr.predict(data_frame)), 2)}%")

