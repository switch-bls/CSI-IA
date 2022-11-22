# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

data_frame = pd.read_csv('../../data/training_data_set.csv')

# Fit regression model
regr = DecisionTreeRegressor(max_depth= 20)
target = pd.read_csv('../../data/target.csv')
regr.fit( data_frame,target)

# Predict
y_2 = regr.predict(data_frame)

print(np.array(target['price']))
print(np.array(y_2))
pdf = pd.DataFrame({"real": np.array(target['price']), "pred": np.array(y_2)})
accuracy = len(pdf[pdf.real==pdf.pred])/len(pdf)
print(f"Accuracy = {round(100*accuracy, 2)}%")

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

print(f"(skl) accuracy = {round(100*regr.score( data_frame,target['price']), 2)}%")

print(f"MAE: {mean_absolute_error(np.array(target['price']), regr.predict(data_frame))} ")
print(f"MSE: {mean_squared_error(np.array(target['price']), regr.predict(data_frame))} ")
print(f"R2: {round(100*r2_score(np.array(target['price']), regr.predict(data_frame)), 2)}%")
