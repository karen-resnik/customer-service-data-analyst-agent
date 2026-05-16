from typing import Any
import pandas as pd
from src.data_loader import load_dataset_from_csv


def get_dataset() -> pd.DataFrame:
    """Load and return the Bitext customer support dataset."""
    return load_dataset_from_csv()


def normalize_text(value: str) -> str:
    """Normalize user-provided text for case-insensitive matching."""
    return value.strip().upper()


def list_categories() -> list[str]:
    """Return all unique dataset categories sorted alphabetically."""
    df = get_dataset()
    return sorted(df["category"].unique().tolist())


def list_intents() -> list[str]:
    """Return all unique dataset intents sorted alphabetically."""
    df = get_dataset()
    return sorted(df["intent"].unique().tolist())


def get_intents_by_category(category: str) -> list[str]:
    """Return all intents that belong to a specific category."""
    df = get_dataset()
    normalized_category = normalize_text(category)

    filtered_df = df[df["category"] == normalized_category]
    return sorted(filtered_df["intent"].unique().tolist())


def filter_dataset(
    category: str | None = None,
    intent: str | None = None,
) -> pd.DataFrame:
    """Filter the dataset by optional category and/or intent."""
    df = get_dataset()

    if category is not None:
        df = df[df["category"] == normalize_text(category)]

    if intent is not None:
        df = df[df["intent"] == intent.strip().lower()]

    return df


def count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> int:
    """Count rows after applying optional category and/or intent filters."""
    filtered_df = filter_dataset(category=category, intent=intent)
    return len(filtered_df)


def show_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
) -> list[dict[str, Any]]:
    """Return example customer instructions and responses from the dataset."""
    filtered_df = filter_dataset(category=category, intent=intent)

    sample_columns = ["instruction", "category", "intent", "response"]
    examples = filtered_df[sample_columns].head(n)

    return examples.to_dict(orient="records")


def get_intent_distribution(category: str) -> dict[str, int]:
    """Return the intent distribution for a specific category."""
    filtered_df = filter_dataset(category=category)

    distribution = filtered_df["intent"].value_counts().to_dict()
    return {intent: int(count) for intent, count in distribution.items()}