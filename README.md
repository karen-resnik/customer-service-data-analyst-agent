# Customer Service Data Analyst Agent

A CLI-based data analyst agent for the Bitext customer service dataset.

The agent answers questions about dataset categories, intents, counts, examples, summaries, metadata, and follow-up questions. It also exposes selected dataset tools through an MCP server.

## What this project does

The agent can answer questions like:

    What categories exist in the dataset?
    How many refund requests did we get?
    Show me examples of people wanting their money back.
    Show me 2 more.
    Can you summarize this category?
    Where did this dataset come from?
    What do you remember about me?

It supports:

- LLM-based query routing
- LangGraph ReAct agent with dataset tools
- Natural-language search over dataset examples
- Persistent conversation memory with SQLite checkpoints
- Persistent user profile memory
- FastMCP server exposing dataset tools

## Model choice

All LLM calls use Nebius Token Factory through the OpenAI-compatible API.

Model used:

    Qwen/Qwen3-30B-A3B-Instruct-2507

This model is used for:

- query routing
- the main ReAct agent
- natural-language category/intent understanding
- user profile memory extraction

I chose this model because it offers a good balance between reasoning quality and cost. The project needs reliable structured outputs, tool-routing decisions, and natural-language interpretation, but it does not require a very large or expensive frontier model. Using the same model for all LLM roles also keeps the implementation simpler and easier to debug.

## Architecture overview

The project is organized around a few main components:

    main.py
        CLI entry point. Handles user input, prints tool traces, supports --session,
        and integrates user profile memory.

    src/router.py
        LLM-based router. Classifies each user question as:
        - structured
        - unstructured
        - out_of_scope

    src/agent.py
        LangGraph ReAct agent. It calls dataset tools, uses persistent SQLite
        checkpoints, and supports follow-up questions through session memory.

    src/tools.py
        Dataset analysis tools. These tools provide category lists, counts,
        examples, summaries, metadata, and natural-language example search.

    src/query_understanding.py
        LLM-based mapper that converts natural-language phrases into the best
        matching dataset category and intent.

    src/memory.py
        Persistent user profile memory. Stores stable profile facts and observed
        asked-about topics under memory/profiles/.

    src/mcp_server.py
        FastMCP server exposing selected dataset tools.

    examples/mcp_client.py
        Example MCP client that starts the MCP server, lists available tools,
        and calls one tool.

## Dataset

The project uses the Bitext customer service dataset:

    bitext/Bitext-customer-support-llm-chatbot-training-dataset

The dataset includes customer instructions, high-level categories, specific intents,
and example assistant responses.

The expected columns are:

    flags
    instruction
    category
    intent
    response

Runtime dataset files are stored under data/ and are ignored by git.

## Setup

Clone the repository:

    git clone https://github.com/karen-resnik/customer-service-data-analyst-agent.git
    cd customer-service-data-analyst-agent

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

Create a .env file:

    cp .env.example .env

Then edit .env and add your Nebius API key:

    NEBIUS_API_KEY=your_api_key_here

## Run the CLI

Run without an explicit session:

    python main.py

Run with persistent memory:

    python main.py --session my_session

Using the same session ID later restores the conversation state and user profile memory.

## Memory

This project implements two types of memory.

### Conversation memory

Conversation memory uses LangGraph checkpoints with SQLite.

Checkpoint files are stored under:

    memory/checkpoints.sqlite

This lets the agent resolve follow-ups like:

    Show me 3 examples from REFUND.
    Show me 2 more.
    Can you summarize this category?

The same --session value is used as the LangGraph thread ID.

### User profile memory

User profile memory stores distilled profile facts under:

    memory/profiles/<session_id>.json

It can remember things like:

- user's name
- explicit interests
- answer style preferences
- topics the user repeatedly asks about

## Dataset tools

The main LangGraph agent can use these dataset tools:

    list_categories
    list_intents
    get_intents_by_category
    get_category_intent_map
    count_rows
    show_examples
    search_examples
    get_intent_distribution
    get_category_overview
    get_intent_overview
    get_dataset_metadata

Important examples:

    show_examples(category="REFUND", n=3)

Returns examples from a known category or intent.

    search_examples(query="people wanting their money back", n=3)

Uses LLM query understanding to map natural language to the most relevant category
or intent before returning dataset examples.

    get_dataset_metadata()

Returns dataset source, row count, columns, categories, intents, and limitations.

## MCP Server

This project exposes selected dataset tools through a FastMCP server.

Start the server:

    python -m src.mcp_server

The server runs over stdio and is intended to be launched by an MCP client.

### Exposed MCP tools

    mcp_list_categories
    mcp_get_category_intent_map
    mcp_count_rows
    mcp_show_examples
    mcp_search_examples
    mcp_get_dataset_metadata

This satisfies the requirement to expose at least three tools through MCP.

## Example MCP client

Run the included MCP client:

    python examples/mcp_client.py

The client starts the MCP server as a subprocess, lists available tools, and calls:

    mcp_list_categories

Expected output includes:

    ACCOUNT
    CANCEL
    CONTACT
    DELIVERY
    FEEDBACK
    INVOICE
    ORDER
    PAYMENT
    REFUND
    SHIPPING
    SUBSCRIPTION

The client uses this FastMCP config:

    config = {
        "mcpServers": {
            "customer-service-data-analyst-agent": {
                "command": "python",
                "args": ["-m", "src.mcp_server"],
            }
        }
    }

## Development checks

Run a quick CLI smoke test:

    python main.py --session smoke_test

Suggested questions:

    What categories exist in the dataset?
    Show me examples from cancellation.
    Show 4 examples where users want to contact a human.
    Give me examples of customers asking where their package is.
    Where did this dataset come from?
    Show me 3 examples from REFUND.
    Show me 2 more.
    Can you summarize this category?
    What do you remember about me?
    Who is the president of France?

The last question should be rejected as out of scope because the agent only answers
questions about the Bitext customer service dataset.

## Git ignored runtime files

The following runtime/local files are ignored by git:

    .env
    .venv/
    data/
    memory/
    __pycache__/

Do not commit API keys, local memory databases, or downloaded dataset files.
