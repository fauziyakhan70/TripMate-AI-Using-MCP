
import os
import sys
import shutil
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq


# =====================================================
# 1. PROJECT AND ENVIRONMENT CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

WEATHER_SERVER_PATH = (
    PROJECT_DIR / "custom_weather_mcp_server.py"
)

load_dotenv(dotenv_path=ENV_FILE)

# SSL certificates
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =====================================================
# 2. API KEYS
# =====================================================

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY is missing.")

if not AVIATION_STACK_API_KEY:
    print("WARNING: AVIATIONSTACK_API_KEY is missing.")

if not OPENWEATHER_API_KEY:
    print("WARNING: OPENWEATHER_API_KEY is missing.")


# =====================================================
# 3. LOCATE UVX
# =====================================================

def find_uvx():
    """
    Locate uvx from:
    1. Current system PATH
    2. Active Python virtual environment
    3. uv executable as a fallback
    """

    uvx_path = shutil.which("uvx")

    if uvx_path:
        return uvx_path, ["aviationstack-mcp"]

    # Check the active virtual environment
    venv_uvx = Path(sys.prefix) / "bin" / "uvx"

    if venv_uvx.exists():
        return str(venv_uvx), ["aviationstack-mcp"]

    # Fallback: use uv tool run
    uv_path = (
        shutil.which("uv")
        or str(Path(sys.prefix) / "bin" / "uv")
    )

    if Path(uv_path).exists():
        return uv_path, [
            "tool",
            "run",
            "aviationstack-mcp"
        ]

    raise RuntimeError(
        "Neither uvx nor uv was found.\n"
        "Install using: python -m pip install uv\n"
        "Or: brew install uv"
    )


AVIATION_COMMAND, AVIATION_ARGS = find_uvx()

print("Project directory:", PROJECT_DIR)
print("Weather MCP server:", WEATHER_SERVER_PATH)
print("Aviation executable:", AVIATION_COMMAND)


# =====================================================
# 4. MCP SERVER ENVIRONMENTS
# =====================================================

# Preserve the current environment.
AVIATION_ENV = os.environ.copy()

# Keep the variable name used in your original
# AviationStack MCP configuration.
AVIATION_ENV["AVIATION_STACK_API_KEY"] = (
    AVIATION_STACK_API_KEY or ""
)

WEATHER_ENV = os.environ.copy()

WEATHER_ENV["OPENWEATHER_API_KEY"] = (
    OPENWEATHER_API_KEY or ""
)


# =====================================================
# 5. GROQ LLM
# =====================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=300
)


# =====================================================
# 6. MCP CLIENT CONFIGURATION
# =====================================================

if not WEATHER_SERVER_PATH.exists():
    raise FileNotFoundError(
        f"Weather MCP server not found: {WEATHER_SERVER_PATH}"
    )


client = MultiServerMCPClient(

    {
        # ---------------------------------------------
        # TAVILY MCP
        # ---------------------------------------------

        "tavily": {

            "transport": "streamable_http",

            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            )
        },


        # ---------------------------------------------
        # AVIATIONSTACK MCP
        # ---------------------------------------------

        "aviationstack": {

            "transport": "stdio",

            "command": AVIATION_COMMAND,

            "args": AVIATION_ARGS,

            "env": AVIATION_ENV
        },


        # ---------------------------------------------
        # CUSTOM WEATHER MCP
        # ---------------------------------------------

        "weather": {

            "transport": "stdio",

            # Use the same Python as the FastAPI app.
            "command": sys.executable,

            "args": [
                str(WEATHER_SERVER_PATH)
            ],

            "env": WEATHER_ENV
        }
    }
)


# =====================================================
# 7. DIAGNOSTIC FUNCTION
# =====================================================

async def get_all_tools():
    """
    Connect to each MCP server independently.

    A failure in one server will not prevent
    diagnostics for the other servers.
    """

    all_tools = []

    for server_name in (
        "tavily",
        "aviationstack",
        "weather"
    ):

        try:

            tools = await client.get_tools(
                server_name=server_name
            )

            all_tools.extend(tools)

            print(
                f"\nAvailable tools from {server_name} MCP:"
            )

            for tool in tools:
                print(" -", tool.name)

        except Exception as error:

            print(
                f"\nCould not connect to {server_name} MCP:"
            )

            print(str(error))

    return all_tools


# =====================================================
# 8. TAVILY MCP
# =====================================================

search_tool = None


async def initialize_mcp():
    """
    Initialize only Tavily MCP.
    """

    global search_tool

    if search_tool is not None:
        return

    if not TAVILY_API_KEY:
        raise ValueError(
            "TAVILY_API_KEY is missing in .env"
        )

    tools = await client.get_tools(
        server_name="tavily"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    search_tool = tools_by_name.get(
        "tavily_search"
    )

    if search_tool is None:

        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            "Tavily search tool not found. "
            f"Available tools: {available_tools}"
        )


async def tavily_mcp_search(query: str):

    await initialize_mcp()

    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )

    return result


# =====================================================
# 9. AVIATIONSTACK MCP
# =====================================================

aviation_tools = {}


async def initialize_aviation_tools():
    """
    Initialize only AviationStack MCP.
    """

    global aviation_tools

    if aviation_tools:
        return

    if not AVIATION_STACK_API_KEY:
        raise ValueError(
            "AVIATIONSTACK_API_KEY is missing in .env"
        )

    tools = await client.get_tools(
        server_name="aviationstack"
    )

    aviation_tools = {
        tool.name: tool
        for tool in tools
    }

    if not aviation_tools:

        raise RuntimeError(
            "AviationStack MCP returned no tools."
        )

    print("\nAviationStack MCP connected.")

    for tool_name in aviation_tools:
        print(" -", tool_name)


async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict = None
):
    """
    Execute an AviationStack MCP tool.
    """

    await initialize_aviation_tools()

    tool = aviation_tools.get(tool_name)

    if tool is None:

        available_tools = ", ".join(
            sorted(aviation_tools.keys())
        )

        raise ValueError(
            f"Tool '{tool_name}' not found. "
            f"Available tools: {available_tools}"
        )

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result


# =====================================================
# 10. WEATHER MCP
# =====================================================

weather_tool = None
forecast_tool = None


async def initialize_weather_tools():
    """
    Initialize only the custom Weather MCP server.
    """

    global weather_tool
    global forecast_tool

    if (
        weather_tool is not None
        and forecast_tool is not None
    ):
        return

    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is missing in .env"
        )

    tools = await client.get_tools(
        server_name="weather"
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    weather_tool = tools_by_name.get(
        "get_current_weather"
    )

    forecast_tool = tools_by_name.get(
        "get_forecast"
    )

    missing_tools = []

    if weather_tool is None:
        missing_tools.append(
            "get_current_weather"
        )

    if forecast_tool is None:
        missing_tools.append(
            "get_forecast"
        )

    if missing_tools:

        available_tools = ", ".join(
            tools_by_name.keys()
        )

        raise RuntimeError(
            f"Missing Weather MCP tools: {missing_tools}. "
            f"Available tools: {available_tools}"
        )

    print("\nWeather MCP connected.")

    for tool_name in tools_by_name:
        print(" -", tool_name)


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    result = await weather_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    result = await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

    return result


# =====================================================
# 11. DESTINATION EXTRACTOR
# =====================================================

def extract_destination(query: str):
    """
    Extract destination from user travel request.
    """

    prompt = f"""
Extract only the destination city or country
from the following travel request.

Travel request:
{query}

Return only the destination name.
Do not explain.
"""

    response = llm.invoke(prompt)

    destination = response.content.strip()

    # Remove surrounding quotation marks if present.
    destination = destination.strip("\"'")

    if not destination:
        raise ValueError(
            "Could not extract a destination from the query."
        )

    return destination


# =====================================================
# 12. TEST / DIAGNOSTIC ENTRY POINT
# =====================================================

if __name__ == "__main__":

    import asyncio

    async def main():

        print("\n================================")
        print("TRIPMATE MCP SERVER DIAGNOSTICS")
        print("================================")

        await get_all_tools()

        print("\n================================")
        print("MCP DIAGNOSTICS COMPLETED")
        print("================================")

    asyncio.run(main())
