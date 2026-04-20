import pandas as pd
from catboost.datasets import titanic

# Загружаем данные 
train_df, _ = titanic()

# Создаем папку data
import os
if not os.path.exists('data'):
    os.makedirs('data')

# Сохраняем датасет в csv файл
train_df.to_csv('data/titanic.csv', index=False)

print("Датасет успешно сохранен в lab4/data/titanic.csv")
print(f"Количество строк: {len(train_df)}")