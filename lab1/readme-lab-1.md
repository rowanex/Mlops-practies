## Запуск
```bash
cd lab1
chmod +x pipeline.sh
bash pipeline.sh
```

## Проверено на Ubuntu:22.04 (Docker)
```bash
docker run -it -v /<path_to_project>/lab1:/lab1 ubuntu:22.04 bash
cd /lab1
bash pipeline.sh
```