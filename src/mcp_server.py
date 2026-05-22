from typing import Any

from fastmcp import FastMCP

from src.tools import (
    count_rows,
    get_category_intent_map,
    get_dataset_metadata,
    list_categories,
    search_examples,
    show_examples,
)


mcp = FastMCP(
    name="customer-service-data-analyst-agent",
)


@mcp.tool
def mcp_list_categories() -> list[str]:
    """List all available customer service dataset categories."""
    return list_categories()


@mcp.tool
def mcp_get_category_intent_map() -> dict[str, list[str]]:
    """Return the mapping of dataset categories to their intents."""
    return get_category_intent_map()


@mcp.tool
def mcp_count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> int:
    """Count dataset rows, optionally filtered by category and/or intent."""
    return count_rows(category=category, intent=intent)


@mcp.tool
def mcp_show_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 5,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Show dataset examples filtered by optional category or intent."""
    return show_examples(
        category=category,
        intent=intent,
        n=n,
        offset=offset,
    )


@mcp.tool
def mcp_search_examples(
    query: str,
    n: int = 5,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search dataset examples using a natural-language query."""
    return search_examples(
        query=query,
        n=n,
        offset=offset,
    )


@mcp.tool
def mcp_get_dataset_metadata() -> dict[str, Any]:
    """Return dataset metadata, source, columns, and limitations."""
    return get_dataset_metadata()


if __name__ == "__main__":
    mcp.run()