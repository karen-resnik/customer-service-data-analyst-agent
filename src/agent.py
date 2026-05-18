import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.tools import DATASET_TOOLS


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
AGENT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


AGENT_SYSTEM_PROMPT = """
You are a customer service data analyst agent.

You answer questions only about the Bitext customer service dataset.

You have access to dataset tools for:
- listing categories and intents
- counting rows
- showing examples
- searching examples
- getting category and intent overviews
- checking intent distributions

Rules:
- Use tools to answer dataset questions.
- Do not answer from general knowledge.
- If the question is unrelated to the dataset, say that you can only answer questions about the Bitext customer service dataset.
- For counts, distributions, categories, intents, and examples, always use tools.
- For summaries, use tools first to gather evidence, then summarize based on the tool results.
- Keep answers clear and concise.
""".strip()


def create_chat_model() -> ChatOpenAI:
    """Create the Nebius-backed chat model used by the agent."""
    load_dotenv()

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY is not set. Add it to your .env file before running the agent."
        )

    return ChatOpenAI(
        model=AGENT_MODEL,
        api_key=api_key,
        base_url=NEBIUS_BASE_URL,
        temperature=0,
    )


def create_agent() -> Any:
    """Create and return the LangGraph ReAct agent."""
    model = create_chat_model()

    return create_react_agent(
        model=model,
        tools=DATASET_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
    )


def run_agent_once(question: str) -> str:
    """Run the agent once and return the final response text."""
    agent = create_agent()

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content=question),
            ]
        },
        config={
            "recursion_limit": 10,
        },
    )

    final_message = result["messages"][-1]
    return str(final_message.content)
