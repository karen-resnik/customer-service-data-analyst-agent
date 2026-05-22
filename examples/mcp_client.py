import asyncio

from fastmcp import Client


async def main() -> None:
    """Call one tool from the local MCP server."""
    config = {
        "mcpServers": {
            "customer-service-data-analyst-agent": {
                "command": "python",
                "args": ["-m", "src.mcp_server"],
            }
        }
    }

    async with Client(config) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}")

        result = await client.call_tool("mcp_list_categories", {})
        print()
        print("mcp_list_categories result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())