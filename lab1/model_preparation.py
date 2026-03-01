import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train/train_scaled.csv')

X = train_df[['hour', 'temperature']]
y = train_df['label']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained on {len(train_df)} samples. Saved to model.pkl")
