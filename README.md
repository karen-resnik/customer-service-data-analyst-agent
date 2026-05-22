# Customer Service Data Analyst Agent

A CLI-based customer service data analyst agent for the Bitext customer service dataset.

The agent can answer questions about dataset categories, intents, counts, examples, summaries, and metadata. It uses:

- LangGraph ReAct agent
- LLM-based router
- LLM-based query understanding for natural-language example search
- Persistent conversation memory with SQLite checkpoints
- Persistent user profile memory
- FastMCP server exposing dataset tools

## Setup

Create and activate a virtual environment, then install dependencies:

    python -m pip install -r requirements.txt

Create a `.env` file with your Nebius API key:

    NEBIUS_API_KEY=your_api_key_here

## Run the CLI

    python main.py

To use persistent memory across restarts, provide a session ID:

    python main.py --session my_session

Example conversation:

    You: Show me 3 examples from REFUND.
    You: Show me 2 more.
    You: Can you summarize this category?
    You: What do you remember about me?

## MCP Server

This project also exposes dataset tools through a FastMCP server.

Start the MCP server:

    python -m src.mcp_server

The server runs over stdio and is intended to be launched by an MCP client.

### Exposed MCP tools

- `mcp_list_categories`
- `mcp_get_category_intent_map`
- `mcp_count_rows`
- `mcp_show_examples`
- `mcp_search_examples`
- `mcp_get_dataset_metadata`

### Example MCP client

Run the included client:

    python examples/mcp_client.py

The client starts the MCP server as a subprocess, lists the available tools, and calls:

    mcp_list_categories

Expected result:

    ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION

## Notes

Runtime memory files are stored under `memory/` and are ignored by git.

Dataset files are stored under `data/` and are ignored by git.
