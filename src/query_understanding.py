import json
import os
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
QUERY_UNDERSTANDING_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class QueryUnderstanding(BaseModel):
    """Structured interpretation of a user query against the dataset schema."""

    category: str | None = Field(
        default=None,
        description="Best matching dataset category, or null if no category is clear.",
    )
    intent: str | None = Field(
        default=None,
        description="Best matching dataset intent, or null if no single intent is appropriate.",
    )
    reason: str = Field(
        ...,
        description="Short explanation for the mapping.",
    )


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """
You map user questions about a customer service dataset to the closest dataset category and intent.

The dataset has these categories and intents:

1. ACCOUNT:
    - create_account: user wants to create/register a new account.
    - delete_account: user wants to delete/remove/close an account.
    - edit_account: user wants to update account information or profile details.
    - recover_password: user forgot, lost, needs to reset, retrieve, or recover a password.
    - registration_problems: user has trouble signing up, registering, or creating an account.
    - switch_account: user wants to change/switch between accounts.

2. CANCEL:
    - check_cancellation_fee: user asks about cancellation fees, penalties, costs, or whether they can cancel without paying.

3. CONTACT:
    - contact_customer_service: user wants to contact customer service/support generally.
    - contact_human_agent: user wants a real person, human agent, live agent, representative, or not a bot.

4. DELIVERY:
    - delivery_options: user asks about available delivery options or delivery methods.
    - delivery_period: user asks how long delivery takes or delivery timeframes.

5. FEEDBACK:
    - complaint: user wants to file a complaint, claim, report dissatisfaction, or say they are unhappy.
    - review: user wants to leave, write, edit, or submit a review/rating/feedback.

6. INVOICE:
    - check_invoice: user wants to check, view, verify, or understand an invoice/receipt/bill.
    - get_invoice: user wants to receive, download, resend, or obtain an invoice/receipt/bill.

7. ORDER:
    - cancel_order: user wants to cancel an order or purchase.
    - change_order: user wants to modify/change an existing order.
    - place_order: user wants to place or make an order.
    - track_order: user asks where an order/package/shipment is or wants tracking/status.

8. PAYMENT:
    - check_payment_methods: user asks what payment methods/options are accepted.
    - payment_issue: user reports payment problems, failed payment, double charge, incorrect charge, unauthorized charge, billing problem.

9. REFUND:
    - check_refund_policy: user asks about refund policy, eligibility, conditions, money-back guarantee, or when refunds are allowed.
    - get_refund: user wants to request/get money back, compensation, reimbursement, or a refund.
    - track_refund: user asks where their refund is or wants refund status/progress.

10. SHIPPING:
    - change_shipping_address: user wants to change/update/modify an existing shipping or delivery address.
    - set_up_shipping_address: user wants to add/create/set up a shipping or delivery address.

11. SUBSCRIPTION:
    - newsletter_subscription: user wants to subscribe/unsubscribe/manage newsletters, mailing lists, promotional emails, or email updates.

Instructions:
- Return valid JSON only. Do not use markdown.
- Your goal is to map the user query to the most specific dataset label that is appropriate.
- Prefer a specific intent over a broad category whenever the user's wording implies a concrete user need.
- When you return an intent, also return the category that contains that intent.
- Return intent as null only when the user is clearly asking about a whole category, a category-level summary, a category-level count, or the list/distribution of intents in a category.
- If the user asks for examples using natural language, choose the single best intent whenever possible.
- If two intents are plausible, choose the closest one based on the user's wording and explain the ambiguity briefly in the reason.
- If the query is dataset-related but only category-level, return the best category and intent null.
- If the query is not about the dataset at all, return category null and intent null.
- Never invent category or intent names outside the list above.
- The reason must refer only to the returned category and intent. Do not mention candidate intents or fields that are not in the JSON schema.

Return this exact JSON shape:
{
  "category": "CATEGORY_NAME_OR_NULL",
  "intent": "intent_name_or_null",
  "candidate_intents": ["intent_name"],
  "confidence": "high" | "medium" | "low",
  "reason": "short explanation"
}
""".strip()


def get_nebius_client() -> OpenAI:
    """Create and return an OpenAI-compatible Nebius client."""
    load_dotenv()

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to your .env file before running query understanding."
        )

    return OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=api_key,
    )


def parse_query_understanding_response(content: str) -> QueryUnderstanding:
    """Parse the model JSON response into QueryUnderstanding."""
    try:
        data: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Query understanding returned invalid JSON: {content}") from error

    return QueryUnderstanding.model_validate(data)


def understand_dataset_query(query: str) -> QueryUnderstanding:
    """Map a natural-language query to a dataset category and/or intent."""
    if not query.strip():
        return QueryUnderstanding(
            category=None,
            intent=None,
            reason="The query is empty.",
        )

    client = get_nebius_client()

    response = client.chat.completions.create(
        model=QUERY_UNDERSTANDING_MODEL,
        messages=[
            {
                "role": "system",
                "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT,
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
        raise ValueError("Query understanding returned an empty response.")

    return parse_query_understanding_response(content)