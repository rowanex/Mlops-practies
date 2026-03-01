import numpy as np
import pandas as pd
import os


np.random.seed(42)


def generate_temperature_data(n=200, noise=0.5, anomalies=False):
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    temp = 15 + 10 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, noise, n)
    season = np.repeat(np.arange(n // 24 + 1), 24)[:n]
    temp += season * 0.05
    if anomalies:
        idx = np.random.choice(n, size=int(n * 0.05), replace=False)
        temp[idx] += np.random.choice([-15, 15], size=len(idx))
    label = (temp > 15).astype(int)
    return pd.DataFrame({'hour': hours, 'temperature': temp, 'label': label})


os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

generate_temperature_data(300, noise=0.5).to_csv('train/dataset_clean.csv', index=False)
generate_temperature_data(300, noise=6.0).to_csv('train/dataset_noisy.csv', index=False)
generate_temperature_data(300, noise=1.0, anomalies=True).to_csv('train/dataset_anomalies.csv', index=False)

generate_temperature_data(100, noise=0.5).to_csv('test/dataset_clean.csv', index=False)
generate_temperature_data(100, noise=2.0, anomalies=True).to_csv('test/dataset_anomalies.csv', index=False)

print("Data created: 3 train datasets, 2 test datasets")
