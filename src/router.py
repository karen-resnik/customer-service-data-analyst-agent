import json
import os
from enum import StrEnum
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
ROUTER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class QueryType(StrEnum):
    """Supported query routing labels."""

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"
    OUT_OF_SCOPE = "out_of_scope"


class RouteResult(BaseModel):
    """Result returned by the query router."""

    query_type: QueryType = Field(
        ...,
        description="The type of query: structured, unstructured, or out_of_scope.",
    )
    reason: str = Field(
        ...,
        description="Short explanation of why this route was chosen.",
    )


ROUTER_SYSTEM_PROMPT = """
You are a query router for a customer service dataset analysis agent.

Your job is to classify the user's query into exactly one of these labels:

1. structured
Use this for concrete, data-driven questions about the Bitext customer service dataset.
Examples:
- What categories exist in the dataset?
- How many refund requests are there?
- Show me 3 examples from the SHIPPING category.
- What is the distribution of intents in the ACCOUNT category?
- Show me examples of people wanting their money back.

2. unstructured
Use this for open-ended analysis or summarization questions about the Bitext customer service dataset.
Examples:
- Summarize the FEEDBACK category.
- How do agents typically respond to cancellation requests?
- What are common complaint patterns?
- Describe how refund-related responses are written.

3. out_of_scope
Use this for anything that is not asking about the Bitext customer service dataset.
Examples:
- Who won the 2024 Champions League?
- Who is the president of France?
- Write me a poem about customer service.
- What is the best CRM software?

Important rules:
- The agent can only answer questions about the dataset.
- Do not answer the user's question.
- Do not use general knowledge.
- Only classify the query.
- Return valid JSON only.
- Do not wrap the JSON in markdown.

Return this exact JSON shape:
{
  "query_type": "structured" | "unstructured" | "out_of_scope",
  "reason": "short explanation"
}
""".strip()


def get_nebius_client() -> OpenAI:
    """Create and return an OpenAI-compatible Nebius client."""
    load_dotenv()

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to your .env file before running the router."
        )

    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
    )


def parse_router_response(content: str) -> RouteResult:
    """Parse the router model's JSON response into a RouteResult."""
    try:
        data: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Router returned invalid JSON: {content}") from error

    return RouteResult.model_validate(data)


def route_query(query: str) -> RouteResult:
    """Classify a user query as structured, unstructured, or out-of-scope."""
    if not query.strip():
        return RouteResult(
            query_type=QueryType.OUT_OF_SCOPE,
            reason="The query is empty.",
        )

    client = get_nebius_client()

    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query,
                    }
                ],
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Router returned an empty response.")

    return parse_router_response(content)


if __name__ == "__main__":
    test_queries = [
        "What categories exist in the dataset?",
        "Summarize the FEEDBACK category.",
        "Who is the president of France?",
    ]

    for test_query in test_queries:
        print("=" * 80)
        print(test_query)
        print(route_query(test_query))