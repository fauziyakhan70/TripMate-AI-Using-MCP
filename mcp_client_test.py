
import os
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# SSL configuration
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

# Initialize MCP client
client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        }
    }
)

# Store Tavily search tool
tavily_search_tool = None


# Get all available MCP tools
async def get_all_tools():

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    return tools


# Get Tavily search tool
async def get_tavily_search_tool():

    global tavily_search_tool

    if tavily_search_tool is not None:
        return tavily_search_tool

    tools = await client.get_tools()

    print("\nAvailable MCP Tools:")

    for tool in tools:
        print(tool.name)

    tavily_search_tool = next(
        (
            tool for tool in tools
            if tool.name == "tavily_search"
        ),
        None
    )

    if tavily_search_tool is None:
        raise ValueError(
            "tavily_search tool not found. Check available tool names."
        )

    return tavily_search_tool


# Execute Tavily MCP search
async def tavily_mcp_search(query: str):

    tool = await get_tavily_search_tool()

    print(f"\nSearching Tavily for: {query}")

    result = await tool.ainvoke({
        "query": query
    })

    print("\nTavily Search Results:\n")
    print(result)

    return result
