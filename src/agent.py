import os
import json
import sqlite3

from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.router import QueryType, route_query
from src.tools import DATASET_TOOLS


NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
AGENT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHECKPOINT_DB_PATH = Path("memory/checkpoints.sqlite")

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
- If a tool returns an empty list [], do not invent examples, counts, categories, intents, or responses.
- If a tool returns no matching examples, try one more relevant tool call with a broader query or a more likely category/intent.
- If no matching dataset evidence is found after retrying, clearly say that no matching examples were found in the dataset.
- Only show examples that appear in tool results. Never create synthetic examples.
- Keep answers clear and concise.
- When the user asks for "more", "next", "another", or "additional" examples, use the previous category, intent, or search query from the conversation history.
- For follow-up requests asking for more examples, call the same example tool again with an offset equal to the number of examples already shown for that same category, intent, or search query.
- If the previous request showed 2 examples and the user asks for 2 more, use offset=2.
- If the previous request showed 3 examples and the user asks for 3 more, use offset=3.
- Do not claim that there are no more examples unless you call the relevant example tool with an increased offset and it returns an empty list.
""".strip()

def question_asks_for_examples(question: str) -> bool:
    """Return True if the user is asking for dataset examples."""
    normalized_question = question.lower()
    example_terms = {
        "example",
        "examples",
        "show me",
        "give me",
        "find examples",
    }
    return any(term in normalized_question for term in example_terms)


def tool_result_is_empty_list(content: str) -> bool:
    """Return True if a tool message content represents an empty list."""
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        return content.strip() == "[]"

    return parsed_content == []

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
    """Create and return the LangGraph ReAct agent with persistent checkpoints."""
    model = create_chat_model()

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    return create_react_agent(
        model=model,
        tools=DATASET_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def run_agent_once(question: str, session_id: str = "default") -> str:
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
            "configurable": {
                "thread_id": session_id,
            },
        },
    )

    final_message = result["messages"][-1]
    return str(final_message.content)

def run_agent_with_trace(question: str, session_id: str = "default") -> tuple[str, list[str]]:
    """Run the agent and return the final answer plus tool-call trace steps."""
    agent = create_agent()
    trace_steps: list[str] = []
    final_answer = ""
    example_tool_was_called = False
    example_tool_found_results = False

    for step in agent.stream(
        {
            "messages": [
                HumanMessage(content=question),
            ]
        },
        config={
            "recursion_limit": 10,
            "configurable": {
                "thread_id": session_id,
            },
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
            if latest_message.name in {"search_examples", "show_examples"}:
                example_tool_was_called = True

                if not tool_result_is_empty_list(str(latest_message.content)):
                    example_tool_found_results = True
        
        if (
            question_asks_for_examples(question)
            and example_tool_was_called
            and not example_tool_found_results
        ):
            final_answer = (
                "I couldn't find matching examples in the dataset for that request. "
                "Try using a broader phrase, or ask for examples from a known category or intent."
            )

    return final_answer, trace_steps


def answer_question(question: str, session_id: str = "default") -> str:
    """Route a question and answer it with the agent when it is in scope."""
    route_result = route_query(question)

    if route_result.query_type == QueryType.OUT_OF_SCOPE:
        return (
            "I can only answer questions about the Bitext customer service dataset. "
            f"Router reason: {route_result.reason}"
        )

    return run_agent_once(question, session_id=session_id)


def answer_question_with_trace(question: str, session_id: str = "default") -> tuple[str, list[str]]:
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

    answer, agent_trace_steps = run_agent_with_trace(question, session_id=session_id,)
    trace_steps.extend(agent_trace_steps)

    return answer, trace_steps