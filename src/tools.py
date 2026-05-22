from typing import Any
import pandas as pd

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.data_loader import load_dataset_from_csv
from src.query_understanding import understand_dataset_query

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

class SearchPlan(BaseModel):
    """Internal search plan for natural-language example search."""

    text_terms: list[str] = Field(
        default_factory=list,
        description="Expanded keyword and phrase search terms.",
    )
    preferred_category: str | None = Field(
        default=None,
        description="Optional category to prioritize in search results.",
    )
    preferred_intents: list[str] = Field(
        default_factory=list,
        description="Optional intents to prioritize in search results.",
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

def resolve_category(category: str) -> str:
    """Resolve a user-provided category using exact validation, then LLM mapping."""
    try:
        return validate_category(category)
    except ValueError:
        understanding = understand_dataset_query(category)

        if understanding.category is None:
            raise

        return validate_category(understanding.category)


def resolve_intent(intent: str) -> str:
    """Resolve a user-provided intent using exact validation, then LLM mapping."""
    try:
        return validate_intent(intent)
    except ValueError:
        understanding = understand_dataset_query(intent)

        if understanding.intent is None:
            raise

        return validate_intent(understanding.intent)


def get_intents_by_category(category: str) -> list[str]:
    """Return all intents that belong to a specific category."""
    df = get_dataset()
    valid_category = resolve_category(category)

    filtered_df = df[df["category"] == valid_category]
    return sorted(filtered_df["intent"].unique().tolist())


def get_category_intent_map() -> dict[str, list[str]]:
    """Return a mapping of each category to its intents."""
    df = get_dataset()

    category_intent_map = (
        df.groupby("category")["intent"]
        .unique()
        .apply(lambda values: sorted(values.tolist()))
        .to_dict()
    )

    return {
        category: intents
        for category, intents in sorted(category_intent_map.items())
    }


def filter_dataset(
    category: str | None = None,
    intent: str | None = None,
) -> pd.DataFrame:
    """Filter the dataset by optional category and/or intent."""
    df = get_dataset()

    if category is not None:
        valid_category = resolve_category(category)
        df = df[df["category"] == valid_category]

    if intent is not None:
        valid_intent = resolve_intent(intent)
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

def build_search_plan(query: str) -> SearchPlan:
    """Build a search plan from a natural-language query.

    Text terms are natural-language phrases only.
    Dataset labels are represented separately as preferred category and intents.
    """
    normalized_query = query.strip().lower()

    text_terms = {normalized_query}
    preferred_category: str | None = None
    preferred_intents: list[str] = []

    # Refund-related broad search
    if any(
        phrase in normalized_query
        for phrase in {
            "refund",
            "refunded",
            "money back",
            "compensation",
            "reimbursement",
        }
    ):
        preferred_category = "REFUND"
        text_terms.update(
            {
                "refund",
                "money back",
                "compensation",
                "reimbursement",
            }
        )

    # Refund-related specific intents
    if any(
        phrase in normalized_query
        for phrase in {
            "refund policy",
            "money back policy",
            "refund rules",
            "refund conditions",
            "am i eligible for a refund",
            "eligible for refund",
        }
    ):
        preferred_category = "REFUND"
        preferred_intents.append("check_refund_policy")
        text_terms.update(
            {
                "refund policy",
                "money back policy",
                "eligible for refund",
            }
        )

    if any(
        phrase in normalized_query
        for phrase in {
            "track refund",
            "refund status",
            "where is my refund",
            "status of my refund",
            "refund progress",
        }
    ):
        preferred_category = "REFUND"
        preferred_intents.append("track_refund")
        text_terms.update(
            {
                "track refund",
                "refund status",
                "where is my refund",
            }
        )

    if any(
        phrase in normalized_query
        for phrase in {
            "get refund",
            "request refund",
            "want refund",
            "want a refund",
            "want my money back",
            "want money back",
            "get my money back",
            "get money back",
            "need my money back",
            "ask for my money back",
            "compensation",
            "reimbursement",
            "wanting their money back",
            "money back",
        }
    ):
        preferred_category = "REFUND"
        preferred_intents.append("get_refund")
        text_terms.update(
            {
                "get refund",
                "request refund",
                "money back",
                "compensation",
                "reimbursement",
            }
        )

    # Complaint / feedback
    if any(
        phrase in normalized_query
        for phrase in {
            "complaint",
            "complain",
            "claim",
            "dissatisfied",
            "unhappy",
            "not happy",
        }
    ):
        preferred_category = "FEEDBACK"
        text_terms.update(
            {
                "complaint",
                "claim",
                "dissatisfied",
                "unhappy",
                "not happy",
            }
        )

    if any(
        phrase in normalized_query
        for phrase in {
            "complaint",
            "complain",
            "consumer claim",
            "file a claim",
            "make a claim",
        }
    ):
        preferred_category = "FEEDBACK"
        preferred_intents.append("complaint")
        text_terms.update(
            {
                "complaint",
                "claim",
                "file a claim",
            }
        )

    # Cancellation
    if any(
        phrase in normalized_query
        for phrase in {
            "cancel",
            "cancellation",
            "cancelling",
        }
    ):
        text_terms.update(
            {
                "cancel",
                "cancellation",
                "cancelling",
            }
        )

    if "cancellation fee" in normalized_query:
        preferred_category = "CANCEL"
        preferred_intents.append("check_cancellation_fee")
        text_terms.update(
            {
                "cancellation fee",
                "fee",
            }
        )

    if any(
        phrase in normalized_query
        for phrase in {
            "cancel order",
            "cancel my order",
            "cancelling order",
            "cancel purchase",
            "cancel my purchase",
        }
    ):
        preferred_category = "ORDER"
        preferred_intents.append("cancel_order")
        text_terms.update(
            {
                "cancel order",
                "cancel purchase",
            }
        )

    # Shipping address
    if any(
        phrase in normalized_query
        for phrase in {
            "shipping address",
            "delivery address",
            "change address",
            "update address",
        }
    ):
        preferred_category = "SHIPPING"
        text_terms.update(
            {
                "shipping address",
                "delivery address",
                "change address",
                "update address",
            }
        )

    return SearchPlan(
        text_terms=sorted(text_terms),
        preferred_category=preferred_category,
        preferred_intents=sorted(set(preferred_intents)),
    )


def search_examples(query: str, n: int = 5) -> list[dict[str, Any]]:
    """Search examples using LLM query understanding plus text-search fallback."""
    df = get_dataset()
    normalized_query = query.strip().lower()

    if not normalized_query:
        raise ValueError("Search query cannot be empty.")

    query_understanding = understand_dataset_query(normalized_query)
    search_plan = build_search_plan(normalized_query)

    sample_columns = ["instruction", "category", "intent", "response"]
    result_parts: list[pd.DataFrame] = []

    if query_understanding.intent:
        intent_matches = df[df["intent"] == query_understanding.intent]
        result_parts.append(intent_matches)

    if query_understanding.category:
        category_matches = df[df["category"] == query_understanding.category]
        result_parts.append(category_matches)

    if search_plan.preferred_intents:
        intent_matches = df[df["intent"].isin(search_plan.preferred_intents)]
        result_parts.append(intent_matches)

    if search_plan.preferred_category:
        category_matches = df[df["category"] == search_plan.preferred_category]
        result_parts.append(category_matches)

    keyword_mask = pd.Series(False, index=df.index)

    for term in search_plan.text_terms:
        term_mask = (
            df["instruction"].str.lower().str.contains(term, regex=False)
            | df["response"].str.lower().str.contains(term, regex=False)
        )
        keyword_mask = keyword_mask | term_mask

    result_parts.append(df.loc[keyword_mask])

    matched_df = pd.concat(result_parts)
    matched_df = matched_df.drop_duplicates().head(n)

    return matched_df[sample_columns].to_dict(orient="records")


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

def get_dataset_metadata() -> dict[str, Any]:
    """Return metadata and important notes about the Bitext dataset."""
    df = get_dataset()

    return {
        "name": "Bitext Customer Service Tagged Training Dataset",
        "source": "Bitext",
        "hugging_face_dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "description": (
            "A customer service training dataset containing customer instructions, "
            "high-level categories, specific intents, and example assistant responses."
        ),
        "data_type_note": (
            "The dataset is intended for training and evaluating customer-service chatbots. "
            "It uses templated placeholders such as {{Order Number}}, {{Website URL}}, "
            "{{Customer Support Phone Number}}, and {{Person Name}} instead of direct customer identifiers. "
            "Based on these placeholders and the training purpose, it appears to be synthetic or de-identified "
            "rather than raw production customer conversations."
        ),
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "column_descriptions": {
            "flags": "Text variation or generation tags.",
            "instruction": "Customer message or request.",
            "category": "High-level customer service topic.",
            "intent": "Specific customer service intent.",
            "response": "Example assistant response.",
        },
        "category_count": int(df["category"].nunique()),
        "intent_count": int(df["intent"].nunique()),
        "categories": list_categories(),
        "intents": list_intents(),
        "limitations": [
            "The dataset may not reflect real production customer conversations.",
            "Some examples contain typos, informal language, or synthetic placeholders.",
            "The responses are examples and should not be treated as official policy.",
            "The dataset should be inspected before drawing business conclusions.",
        ],
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
        func=get_category_intent_map,
        name="get_category_intent_map",
        description=(
            "Return a mapping of every dataset category to the intents inside it. "
            "Use this when the user asks to describe, list, compare, or explain "
            "the intents in each category."
        ),
        args_schema=EmptyInput,
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
            "Search for example customer instructions and agent responses using a natural-language "
            "keyword or phrase. This tool expands common phrases such as money back, complaint, "
            "cancellation, or shipping address into relevant dataset categories or intents internally. "
            "Use this when the user describes an idea in natural language instead of naming an exact "
            "category or intent."
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
    StructuredTool.from_function(
        func=get_dataset_metadata,
        name="get_dataset_metadata",
        description=(
            "Return metadata about the Bitext customer service dataset, including source, "
            "row count, columns, category and intent counts, data type notes, and limitations. "
            "Use this for questions about where the dataset comes from, whether it is real "
            "customer data, what columns it has, or dataset limitations."
        ),
        args_schema=EmptyInput,
    ),
]