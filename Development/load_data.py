import pandas as pd
from pathlib import Path


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


file_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df = load_data(file_path)

print(df.head())
print(df.shape)