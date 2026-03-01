import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

test_df = pd.read_csv('test/test_scaled.csv')

X_test = test_df[['hour', 'temperature']]
y_test = test_df['label']

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model test accuracy is: {accuracy:.3f}")
