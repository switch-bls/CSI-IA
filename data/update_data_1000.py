
import sys
import pandas as pa

def standardize(df): 
    df = (df-df.mean())/df.std()
    return df
def normalize_with_min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'no_model_no_index_no_outlier.csv'

df = pa.read_csv(csv_file, low_memory=False)

#delete empty datas
df = df.dropna(thresh=10)

#standardisation
df = standardize(df)

print ("standardization : ok")
#normalization
for col in df.columns:
    df[col] = (normalize_with_min_max_scaling(df[col])*1000).astype(int)

print ("normalization : ok")

df.to_csv("1000_final_"+csv_file, index = False)

print ("csv : ok")

print (df)

    