import pandas as pd

from data_loader import load_dataset_from_csv


def print_section(title: str) -> None:
    """Print a formatted section title."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print useful information about the Bitext dataset."""
    print_section("Basic Info")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print_section("Missing Values")
    print(df.isna().sum())

    print_section("Categories")
    categories = sorted(df["category"].unique())
    print(f"Number of categories: {len(categories)}")
    print(categories)

    print_section("Rows per Category")
    print(df["category"].value_counts())

    print_section("Intents")
    intents = sorted(df["intent"].unique())
    print(f"Number of intents: {len(intents)}")
    print(intents)

    print_section("Rows per Intent")
    print(df["intent"].value_counts())

    print_section("Sample Rows")
    sample_columns = ["instruction", "category", "intent", "response"]
    print(df[sample_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    dataframe = load_dataset_from_csv()
    inspect_dataset(dataframe)