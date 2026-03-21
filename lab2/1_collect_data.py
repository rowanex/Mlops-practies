"""
Stage 1: Data Collection
Downloads the Titanic dataset from GitHub (datasciencedojo/datasets).
Method: HTTP GET (equivalent to wget).
"""
import os
import requests
import pandas as pd

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "titanic_raw.csv")
URL = (
    "https://raw.githubusercontent.com/"
    "datasciencedojo/datasets/master/titanic.csv"
)

def collect_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[Stage 1] Downloading Titanic dataset from:\n  {URL}")
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    with open(RAW_FILE, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(RAW_FILE)
    print(f"[Stage 1] Saved to '{RAW_FILE}'")
    print(f"[Stage 1] Shape: {df.shape}")
    print(f"[Stage 1] Columns: {list(df.columns)}")
    print(df.head(3).to_string())

if __name__ == "__main__":
    collect_data()
