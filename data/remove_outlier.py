import pandas
import numpy as np
df = pandas.read_csv("no_model_no_index_e.csv")

numeric_col = ['year','price','mileage','mpg','engineSize']

for col in [numeric_col]:
    print(col)
    for x in col:
        q75,q25 = np.percentile(df.loc[:,x],[75,25])
        intr_qr = q75-q25
    
        max = q75+(1.7*intr_qr)
        min = q25-(1.7*intr_qr)
    
        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan
    df = df.dropna(axis = 0)
df.to_csv("no_model_no_index_no_outlier.csv", index = False)
print(len(df.index))