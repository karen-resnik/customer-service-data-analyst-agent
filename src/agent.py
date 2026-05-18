import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.router import QueryType, route_query
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

def run_agent_with_trace(question: str) -> tuple[str, list[str]]:
    """Run the agent and return the final answer plus tool-call trace steps."""
    agent = create_agent()
    trace_steps: list[str] = []
    final_answer = ""

    for step in agent.stream(
        {
            "messages": [
                HumanMessage(content=question),
            ]
        },
        config={
            "recursion_limit": 10,
        },
        stream_mode="values",
    ):
        latest_message = step["messages"][-1]

        if isinstance(latest_message, AIMessage):
            if latest_message.tool_calls:
                for tool_call in latest_message.tool_calls:
                    trace_steps.append(
                        f"Tool call: {tool_call['name']}\n"
                        f"Tool args: {tool_call['args']}"
                    )
            elif latest_message.content:
                final_answer = str(latest_message.content)

        elif isinstance(latest_message, ToolMessage):
            trace_steps.append(
                f"Tool result from {latest_message.name}:\n"
                f"{latest_message.content}"
            )

    return final_answer, trace_steps


def answer_question(question: str) -> str:
    """Route a question and answer it with the agent when it is in scope."""
    route_result = route_query(question)

    if route_result.query_type == QueryType.OUT_OF_SCOPE:
        return (
            "I can only answer questions about the Bitext customer service dataset. "
            f"Router reason: {route_result.reason}"
        )

    return run_agent_once(question)


def answer_question_with_trace(question: str) -> tuple[str, list[str]]:
    """Route a question and answer it, returning the final answer and trace steps."""
    route_result = route_query(question)

    trace_steps = [
        f"Router decision: {route_result.query_type.value}",
        f"Router reason: {route_result.reason}",
    ]

    if route_result.query_type == QueryType.OUT_OF_SCOPE:
        answer = (
            "I can only answer questions about the Bitext customer service dataset."
        )
        return answer, trace_steps

    answer, agent_trace_steps = run_agent_with_trace(question)
    trace_steps.extend(agent_trace_steps)

    return answer, trace_steps