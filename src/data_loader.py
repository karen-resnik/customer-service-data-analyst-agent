from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "bitext_customer_support.csv"


def download_dataset() -> pd.DataFrame:
    """Download the Bitext customer support dataset and return it as a DataFrame."""
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

    return df


def load_dataset_from_csv() -> pd.DataFrame:
    """Load the dataset from a local CSV file if it exists, otherwise download it."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    return download_dataset()


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""
    print("Dataset loaded successfully!")
    print()
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    dataframe = load_dataset_from_csv()
    print_dataset_summary(dataframe)