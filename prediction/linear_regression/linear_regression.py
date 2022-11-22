from sklearn import linear_model as LM
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

data_frame = pd.read_csv('../../data/training_data_set.csv')
target = pd.read_csv('../../data/target.csv')



reg = LM.LinearRegression()

reg.fit(data_frame, np.array(target['price']))

print(f"Coefficients are {reg.coef_}")
print(f"Itercepts is {reg.intercept_} ")
print(f"So the model is T = {reg.coef_[0]}W + {reg.coef_[1]}H +{reg.intercept_} ")

print(f"MAE: {mean_absolute_error(np.array(target['price']), reg.predict(data_frame))} ")
print(f"MSE: {mean_squared_error(np.array(target['price']), reg.predict(data_frame))} ")
print(f"R2: {round(100*r2_score(np.array(target['price']), reg.predict(data_frame)), 2)}%")