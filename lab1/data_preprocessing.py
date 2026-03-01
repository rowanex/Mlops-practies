import pandas as pd
import numpy as np
import pickle
import glob
import os
from sklearn.preprocessing import StandardScaler

def load_and_concat(folder):
    files = glob.glob(f'{folder}/*.csv')
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

train_df = load_and_concat('train')
test_df = load_and_concat('test')

features = ['hour', 'temperature']

scaler = StandardScaler()
train_df[features] = scaler.fit_transform(train_df[features])
test_df[features] = scaler.transform(test_df[features])

train_df.to_csv('train/train_scaled.csv', index=False)
test_df.to_csv('test/test_scaled.csv', index=False)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Preprocessing done. Scaler saved.")
