# lab3

Простый микросервис с моделью машинного обучения, упакованный в Docker.

## Что есть в работе

- `app/main.py` - API на `FastAPI`;
- `app/model.py` - модель классификации Iris;
- `Dockerfile` - сборка контейнера;
- `docker-compose.yml` - запуск через Docker Compose.

## Сборка

```powershell
docker build -t iris-api:latest .
```

## Запуск

```powershell
docker run --name iris-api-container -p 8000:8000 iris-api:latest
```

или

```powershell
docker compose up --build
```

## Проверка

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
