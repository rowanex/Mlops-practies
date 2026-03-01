#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# === 1. Проверяем и устанавливаем Python ===
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found, installing..."
    apt-get update -q
    apt-get install -y python3 python3-pip
fi

# === 2. Проверяем и устанавливаем pip ===
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found, installing..."
    apt-get install -y python3-pip
fi

# === 3. Устанавливаем зависимости (пробуем все варианты) ===
echo "=== Installing dependencies ==="
pip3 install scikit-learn pandas numpy --quiet --break-system-packages 2>/dev/null || \
pip3 install scikit-learn pandas numpy --quiet 2>/dev/null || \
pip install scikit-learn pandas numpy --quiet --break-system-packages 2>/dev/null || \
pip install scikit-learn pandas numpy --quiet 2>/dev/null || true

echo "=== Step 1: Creating data ==="
python3 data_creation.py

echo "=== Step 2: Preprocessing data ==="
python3 data_preprocessing.py

echo "=== Step 3: Training model ==="
python3 model_preparation.py

echo "=== Step 4: Testing model ==="
python3 model_testing.py