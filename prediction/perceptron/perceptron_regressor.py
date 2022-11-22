from sklearn.linear_model import Perceptron
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score


data_frame = pd.read_csv('../../data/training_data_set.csv')
target = pd.read_csv('../../data/target.csv')


clf = Perceptron(n_jobs=-1)

clf.fit(data_frame, target)

print(f"MAE: {mean_absolute_error(np.array(target['price']), clf.predict(data_frame))} ")
print(f"MSE: {mean_squared_error(np.array(target['price']), clf.predict(data_frame))} ")
print(f"R2: {round(100*r2_score(np.array(target['price']), clf.predict(data_frame)), 2)}%")

print(clf.score(data_frame, target['price']))

