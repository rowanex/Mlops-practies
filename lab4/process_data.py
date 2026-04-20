import pandas as pd

# Читаем исходный файл
df = pd.read_csv('data/titanic.csv')

# Выбираем только нужные колонки
selected_columns = ['Pclass', 'Sex', 'Age']
df_modified = df[selected_columns]

# Перезаписываем файл
df_modified.to_csv('data/titanic.csv', index=False)

print("Датасет модифицирован. Оставлены колонки:", selected_columns)
print(df_modified.head())