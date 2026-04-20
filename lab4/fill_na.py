import pandas as pd

# Читаем модифицированный ранее файл
df = pd.read_csv('data/titanic.csv')

# Проверяем, есть ли пустые значения
print(f"Пропусков в Age до: {df['Age'].isna().sum()}")

# Вычисляем среднее и заполняем пропуски
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

print(f"Средний возраст для заполнения: {mean_age:.2f}")
print(f"Пропусков в Age после: {df['Age'].isna().sum()}")

# Сохраняем новую версию датасета
df.to_csv('data/titanic.csv', index=False)
print("Файл data/titanic.csv обновлен.")