# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

data_frame = pd.read_csv('../../all_e.csv')

# Fit regression model
regr = DecisionTreeRegressor(max_depth= 10)

#Nombre d'echantillon
n = 3
size = len(data_frame.index)
#Taille de cahque echantillon
nb = int(size//n)


remaining_data = data_frame

couple = []
#Creer de groupe de donnée aleatoir de meme taille
for j in range(n):
    sample = remaining_data.sample(nb)
    remaining_data = remaining_data.drop(labels=sample.index)

    price_sample = sample['price']
    trainning = sample.drop("price", axis = 1)

    couple.append([trainning, price_sample])

for i in range(n):
    print("i:",i)
    splited_data = couple.copy()

    #Retirer un groupe qui servira a tester la prediction
    test_data = splited_data.pop(i)  

    #Dataframe vide
    price = pd.DataFrame()
    trainning_data = pd.DataFrame(columns=data_frame.drop('price', axis = 1).columns)

    #Rassemebler le rester des groupes 
    for c in couple:
        trainning_data = pd.concat([trainning_data,c[0]])
        price = pd.concat([price,c[1]])

    #Entrainer l'algoritme    
    regr.fit(trainning_data,price)
    # Predict
    print(f"MAE: {mean_absolute_error(np.array(test_data[1]), regr.predict(test_data[0]))} ")
    print(f"MSE: {mean_squared_error(np.array(test_data[1]), regr.predict(test_data[0]))} ")
    print(f"R2: {round(100*r2_score(np.array(test_data[1]), regr.predict(test_data[0])), 2)}%")

