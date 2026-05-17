from pprint import pprint

from src.tools import (
    DATASET_TOOLS,
    count_rows,
    get_category_overview,
    get_intent_distribution,
    get_intent_overview,
    get_intents_by_category,
    list_categories,
    list_intents,
    search_examples,
    show_examples,
)


def debug_plain_python_tools() -> None:
    """Run examples that call the plain Python tool functions directly."""
    print("\n" + "=" * 80)
    print("Plain Python tools")
    print("=" * 80)

    print("\n1. Categories:")
    pprint(list_categories())

    print("\n2. Intents:")
    pprint(list_intents())

    print("\n3. Intents in REFUND category:")
    pprint(get_intents_by_category("refund"))

    print("\n4. Count rows in REFUND category:")
    pprint(count_rows(category="REFUND"))

    print("\n5. Count rows with get_refund intent:")
    pprint(count_rows(intent="get_refund"))

    print("\n6. Intent distribution in ACCOUNT category:")
    pprint(get_intent_distribution("ACCOUNT"))

    print("\n7. Show 2 examples from SHIPPING category:")
    pprint(show_examples(category="SHIPPING", n=2))

    print("\n8. Search examples for 'money back':")
    pprint(search_examples(query="money back", n=3))

    print("\n9. Category overview for FEEDBACK:")
    feedback_overview = get_category_overview("FEEDBACK")
    pprint(
        {
            "category": feedback_overview["category"],
            "row_count": feedback_overview["row_count"],
            "intents": feedback_overview["intents"],
            "intent_distribution": feedback_overview["intent_distribution"],
            "sample_record_count": len(feedback_overview["sample_records"]),
        }
    )

    print("\n10. Intent overview for cancel_order:")
    cancel_order_overview = get_intent_overview("cancel_order")
    pprint(
        {
            "intent": cancel_order_overview["intent"],
            "categories": cancel_order_overview["categories"],
            "row_count": cancel_order_overview["row_count"],
            "sample_record_count": len(cancel_order_overview["sample_records"]),
        }
    )


def debug_langchain_wrapped_tools() -> None:
    """Run examples that call the LangChain StructuredTool wrappers."""
    print("\n" + "=" * 80)
    print("LangChain StructuredTool wrappers")
    print("=" * 80)

    tools_by_name = {tool.name: tool for tool in DATASET_TOOLS}

    print("\n1. Tool names:")
    pprint(list(tools_by_name.keys()))

    print("\n2. Invoke count_rows tool:")
    pprint(tools_by_name["count_rows"].invoke({"category": "REFUND"}))

    print("\n3. Invoke show_examples tool:")
    pprint(
        tools_by_name["show_examples"].invoke(
            {
                "category": "SHIPPING",
                "n": 1,
            }
        )
    )

    print("\n4. Invoke search_examples tool:")
    pprint(
        tools_by_name["search_examples"].invoke(
            {
                "query": "money back",
                "n": 2,
            }
        )
    )

    print("\n5. Invoke get_category_overview tool:")
    overview = tools_by_name["get_category_overview"].invoke(
        {
            "category": "FEEDBACK",
        }
    )
    pprint(
        {
            "category": overview["category"],
            "row_count": overview["row_count"],
            "intents": overview["intents"],
            "sample_record_count": len(overview["sample_records"]),
        }
    )


def main() -> None:
    """Run temporary debugging examples for dataset tools."""
    debug_plain_python_tools()
    debug_langchain_wrapped_tools()


if __name__ == "__main__":
    main()