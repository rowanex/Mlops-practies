import pandas as pd

# Читаем текущую версию датасета (уже без пропусков)
df = pd.read_csv('data/titanic.csv')

# Применяем One-Hot Encoding к колонке 'Sex'
# drop_first=True удаляет одну колонку, чтобы избежать мультиколлинеарности 
# (например, останется только Sex_male: 1 - мужчина, 0 - женщина)
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Сохраняем финальную версию
df_encoded.to_csv('data/titanic.csv', index=False)

print("One-Hot Encoding выполнен успешно.")
print("Новые колонки:", df_encoded.columns.tolist())
print(df_encoded.head())