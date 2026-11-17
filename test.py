
import asyncio

from mcp_client_test import (
    get_all_tools,
    tavily_mcp_search
)


async def main():

    # First, display all available MCP tools
    await get_all_tools()

    # Test Tavily MCP search
    query = "Best tourist attractions in Las Vegas"

    result = await tavily_mcp_search(query)

    print("\nFINAL RESULT:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
