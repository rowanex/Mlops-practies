"""
Stage 2: Data Preparation
- Missing value imputation
- Feature engineering (FamilySize, IsAlone)
- Encoding (Sex, Embarked)
- StandardScaler
- Train/Test split (80/20, stratified)
"""
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "titanic_raw.csv")

def prepare_dataset():
    print("[Stage 2] Loading raw data...")
    df = pd.read_csv(RAW_FILE)
    print(f"[Stage 2] Raw shape: {df.shape}")

    # Imputation (pandas 3.0 compatible — без inplace=True)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())

    # Drop low-value columns
    df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"], inplace=True)

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # Encoding
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    print(f"[Stage 2] Features: {list(df.columns)}")
    print(f"[Stage 2] Missing values:\n{df.isnull().sum()}")

    # Split
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[Stage 2] Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")

    # Save
    with open(os.path.join(DATA_DIR, "train.pkl"), "wb") as f:
        pickle.dump((X_train_sc, y_train.values), f)
    with open(os.path.join(DATA_DIR, "test.pkl"), "wb") as f:
        pickle.dump((X_test_sc, y_test.values), f)
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(DATA_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    print("[Stage 2] All datasets and preprocessors saved.")

if __name__ == "__main__":
    prepare_dataset()
