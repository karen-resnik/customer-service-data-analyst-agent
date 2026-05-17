from typing import Any
import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from src.data_loader import load_dataset_from_csv

class EmptyInput(BaseModel):
    """Input schema for tools that do not require parameters."""


class CategoryInput(BaseModel):
    """Input schema for tools that require a dataset category."""

    category: str = Field(
        ...,
        description="Dataset category, for example REFUND, ACCOUNT, SHIPPING, or FEEDBACK.",
    )


class IntentInput(BaseModel):
    """Input schema for tools that require a dataset intent."""

    intent: str = Field(
        ...,
        description="Dataset intent, for example get_refund, cancel_order, or complaint.",
    )


class CountRowsInput(BaseModel):
    """Input schema for counting rows with optional filters."""

    category: str | None = Field(
        default=None,
        description="Optional dataset category filter, for example REFUND or ACCOUNT.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional dataset intent filter, for example get_refund or complaint.",
    )


class ShowExamplesInput(BaseModel):
    """Input schema for showing examples from the dataset."""

    category: str | None = Field(
        default=None,
        description="Optional dataset category filter, for example SHIPPING or REFUND.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional dataset intent filter, for example get_refund or change_shipping_address.",
    )
    n: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of examples to return. Must be between 1 and 10.",
    )


class SearchExamplesInput(BaseModel):
    """Input schema for searching examples by keyword or phrase."""

    query: str = Field(
        ...,
        min_length=1,
        description="Keyword or phrase to search for in instructions, responses, categories, or intents.",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of matching examples to return. Must be between 1 and 10.",
    )

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

def get_sample_records(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """Return compact sample records from a DataFrame."""
    sample_columns = ["instruction", "category", "intent", "response"]
    sample_df = df[sample_columns].head(n)
    return sample_df.to_dict(orient="records")


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
    return get_sample_records(filtered_df, n=n)


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


def get_category_overview(category: str) -> dict[str, Any]:
    """Return a compact overview of a category for summarization or analysis."""
    valid_category = validate_category(category)
    filtered_df = filter_dataset(category=valid_category)

    intent_distribution = filtered_df["intent"].value_counts().to_dict()

    return {
        "category": valid_category,
        "row_count": len(filtered_df),
        "intents": sorted(filtered_df["intent"].unique().tolist()),
        "intent_distribution": {
            intent: int(count) for intent, count in intent_distribution.items()
        },
        "sample_records": get_sample_records(filtered_df, n=5),
    }


def get_intent_overview(intent: str) -> dict[str, Any]:
    """Return a compact overview of an intent for summarization or analysis."""
    valid_intent = validate_intent(intent)
    filtered_df = filter_dataset(intent=valid_intent)

    category_values = sorted(filtered_df["category"].unique().tolist())

    return {
        "intent": valid_intent,
        "categories": category_values,
        "row_count": len(filtered_df),
        "sample_records": get_sample_records(filtered_df, n=5),
    }


DATASET_TOOLS = [
    StructuredTool.from_function(
        func=list_categories,
        name="list_categories",
        description=(
            "List all high-level customer service categories in the dataset. "
            "Use this when the user asks what categories exist."
        ),
        args_schema=EmptyInput,
    ),
    StructuredTool.from_function(
        func=list_intents,
        name="list_intents",
        description=(
            "List all specific customer service intents in the dataset. "
            "Use this when the user asks what intents exist."
        ),
        args_schema=EmptyInput,
    ),
    StructuredTool.from_function(
        func=get_intents_by_category,
        name="get_intents_by_category",
        description=(
            "Return the intents that belong to a specific category. "
            "Use this when the user asks what intents are included in a category."
        ),
        args_schema=CategoryInput,
    ),
    StructuredTool.from_function(
        func=count_rows,
        name="count_rows",
        description=(
            "Count dataset rows, optionally filtered by category and/or intent. "
            "Use this for questions like 'How many refund requests are there?' "
            "or 'How many complaint examples are in the dataset?'"
        ),
        args_schema=CountRowsInput,
    ),
    StructuredTool.from_function(
        func=show_examples,
        name="show_examples",
        description=(
            "Show example customer instructions and agent responses, optionally filtered "
            "by category and/or intent. Use this when the user asks for examples from "
            "a known category or intent."
        ),
        args_schema=ShowExamplesInput,
    ),
    StructuredTool.from_function(
        func=get_intent_distribution,
        name="get_intent_distribution",
        description=(
            "Return the distribution of intents inside a specific category. "
            "Use this when the user asks for intent distribution within a category."
        ),
        args_schema=CategoryInput,
    ),
    StructuredTool.from_function(
        func=search_examples,
        name="search_examples",
        description=(
            "Search for examples using a keyword or phrase across customer instructions, "
            "responses, categories, and intents. Use this when the user describes an idea "
            "in natural language instead of naming an exact category or intent."
        ),
        args_schema=SearchExamplesInput,
    ),
    StructuredTool.from_function(
        func=get_category_overview,
        name="get_category_overview",
        description=(
            "Return a compact overview of a category, including row count, intents, "
            "intent distribution, and sample records. Use this for summarizing or "
            "analyzing a whole category."
        ),
        args_schema=CategoryInput,
    ),
    StructuredTool.from_function(
        func=get_intent_overview,
        name="get_intent_overview",
        description=(
            "Return a compact overview of an intent, including row count, category, "
            "and sample records. Use this for summarizing or analyzing a specific intent."
        ),
        args_schema=IntentInput,
    ),
]