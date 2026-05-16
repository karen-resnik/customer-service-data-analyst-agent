from typing import Any
import pandas as pd
from src.data_loader import load_dataset_from_csv


def get_dataset() -> pd.DataFrame:
    """Load and return the Bitext customer support dataset."""
    return load_dataset_from_csv()


def normalize_category(value: str) -> str:
    """Normalize a user-provided category for case-insensitive matching."""
    return value.strip().upper()


def normalize_intent(value: str) -> str:
    """Normalize a user-provided intent for case-insensitive matching."""
    return value.strip().lower()


def list_categories() -> list[str]:
    """Return all unique dataset categories sorted alphabetically."""
    df = get_dataset()
    return sorted(df["category"].unique().tolist())


def list_intents() -> list[str]:
    """Return all unique dataset intents sorted alphabetically."""
    df = get_dataset()
    return sorted(df["intent"].unique().tolist())


def validate_category(category: str) -> str:
    """Return a normalized category, or raise ValueError if it does not exist."""
    normalized_category = normalize_category(category)
    valid_categories = list_categories()

    if normalized_category not in valid_categories:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories are: {', '.join(valid_categories)}."
        )

    return normalized_category


def validate_intent(intent: str) -> str:
    """Return a normalized intent, or raise ValueError if it does not exist."""
    normalized_intent = normalize_intent(intent)
    valid_intents = list_intents()

    if normalized_intent not in valid_intents:
        raise ValueError(
            f"Unknown intent '{intent}'. "
            f"Valid intents are: {', '.join(valid_intents)}."
        )

    return normalized_intent


def get_intents_by_category(category: str) -> list[str]:
    """Return all intents that belong to a specific category."""
    df = get_dataset()
    valid_category = validate_category(category)

    filtered_df = df[df["category"] == valid_category]
    return sorted(filtered_df["intent"].unique().tolist())


def filter_dataset(
    category: str | None = None,
    intent: str | None = None,
) -> pd.DataFrame:
    """Filter the dataset by optional category and/or intent."""
    df = get_dataset()

    if category is not None:
        valid_category = validate_category(category)
        df = df[df["category"] == valid_category]

    if intent is not None:
        valid_intent = validate_intent(intent)
        df = df[df["intent"] == valid_intent]

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


def search_examples(query: str, n: int = 5) -> list[dict[str, Any]]:
    """Search customer instructions and responses for a keyword or phrase."""
    df = get_dataset()
    normalized_query = query.strip().lower()

    if not normalized_query:
        raise ValueError("Search query cannot be empty.")

    mask = (
        df["instruction"].str.lower().str.contains(normalized_query, regex=False)
        | df["response"].str.lower().str.contains(normalized_query, regex=False)
        | df["category"].str.lower().str.contains(normalized_query, regex=False)
        | df["intent"].str.lower().str.contains(normalized_query, regex=False)
    )

    sample_columns = ["instruction", "category", "intent", "response"]
    examples = df.loc[mask, sample_columns].head(n)

    return examples.to_dict(orient="records")