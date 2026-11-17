
import os
import json
import uuid
import asyncio
import certifi
import operator
import psycopg

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search
)


# =====================================================
# 1. ENVIRONMENT CONFIGURATION
# =====================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please configure it in your .env file."
    )


def get_database_url():

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL "
            "External Database URL to .env"
        )

    if "sslmode=" not in database_url:

        separator = (
            "&" if "?" in database_url else "?"
        )

        database_url = (
            f"{database_url}{separator}sslmode=require"
        )

    return database_url


# =====================================================
# 2. GROQ LLM CONFIGURATION
# =====================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

# One shared LLM configuration.
# Used only by the Itinerary Agent in this backend.

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1200
)


# =====================================================
# 3. TOKEN / CONTEXT OPTIMIZATION
# =====================================================

def compact_text(data, max_chars=1500):
    """
    Convert MCP results into compact text.

    This prevents unnecessarily large MCP
    responses from entering Groq prompts.
    """

    if data is None:
        return "Information unavailable."

    try:

        if isinstance(data, (dict, list)):

            text = json.dumps(
                data,
                ensure_ascii=False,
                default=str
            )

        else:

            text = str(data)

    except Exception:

        text = str(data)

    if len(text) > max_chars:

        text = (
            text[:max_chars]
            + "\n[Additional content omitted]"
        )

    return text


def count_records(data):
    """
    Count directory records without printing
    or sending the complete directory to Groq.
    """

    if isinstance(data, list):
        return len(data)

    if isinstance(data, dict):

        for key in (
            "data",
            "results",
            "airports",
            "airlines"
        ):

            value = data.get(key)

            if isinstance(value, list):
                return len(value)

        return len(data)

    return 0


# =====================================================
# 4. ASYNC MCP EXECUTION
# =====================================================

def run_async(coro):
    """
    Execute asynchronous MCP calls from
    synchronous LangGraph nodes.
    """

    return asyncio.run(coro)


# =====================================================
# 5. LANGGRAPH STATE
# =====================================================

class TravelState(TypedDict):

    messages: Annotated[
        list[AnyMessage],
        operator.add
    ]

    user_query: str

    flight_results: str

    hotel_results: str

    weather_results: str

    itinerary: str

    llm_calls: int


# =====================================================
# 6. FLIGHT AGENT
# =====================================================

def flight_agent(state: TravelState):

    print("\nINSIDE FLIGHT AGENT\n")

    try:

        # Fetch airport directory using AviationStack MCP.

        airports = run_async(
            aviation_mcp_call(
                "list_airports"
            )
        )

        # Fetch airline directory using AviationStack MCP.

        airlines = run_async(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        airport_count = count_records(airports)

        airline_count = count_records(airlines)

        print(
            f"Airport directory records: {airport_count}"
        )

        print(
            f"Airline directory records: {airline_count}"
        )

        # These are directory tools, not route-specific
        # live flight search or booking tools.

        flight_data = f"""
FLIGHT INFORMATION

User Request:
{state["user_query"]}

AviationStack MCP Status:
- Airport directory successfully queried.
- Airport directory records: {airport_count}
- Airline directory successfully queried.
- Airline directory records: {airline_count}

Important:
The available directory information does not
confirm live flights, route availability,
flight schedules, or ticket prices.

Route-specific flight information requires
an appropriate flight search API.
"""

    except Exception as e:

        print("Flight MCP Error:", str(e))

        flight_data = f"""
FLIGHT INFORMATION

Flight directory information is currently
unavailable.

Reason: {str(e)}

Do not assume flight availability or prices.
"""

    return {

        "flight_results": compact_text(
            flight_data,
            max_chars=1200
        ),

        "messages": [
            AIMessage(
                content="Flight information processed."
            )
        ],

        # No Groq call in this agent.
        "llm_calls": state.get("llm_calls", 0)

    }


# =====================================================
# 7. HOTEL AGENT
# =====================================================

def hotel_agent(state: TravelState):

    print("\nINSIDE HOTEL AGENT\n")

    query = f"""
Find hotel recommendations for this travel request:

{state["user_query"]}

Return the most relevant results.

Include when available:

1. Hotel name
2. Location
3. Price
4. Rating
5. Booking URL

Do not invent hotel prices or ratings.
"""

    try:

        hotel_results = run_async(
            tavily_mcp_search(query)
        )

        # Reduce raw Tavily output.

        hotel_results = compact_text(
            hotel_results,
            max_chars=2200
        )

    except Exception as e:

        print("Hotel MCP Error:", str(e))

        hotel_results = (
            "Hotel information unavailable: "
            + str(e)
        )

    return {

        "hotel_results": hotel_results,

        "messages": [
            AIMessage(
                content="Hotel information fetched."
            )
        ],

        # Tavily search does not use our Groq LLM.
        "llm_calls": state.get("llm_calls", 0)

    }


# =====================================================
# 8. WEATHER AGENT
# =====================================================

def weather_agent(state: TravelState):

    print("\nINSIDE WEATHER AGENT\n")

    try:

        # Extract destination using mcp_client.py.

        city = extract_destination(
            state["user_query"]
        )

        print("Extracted Destination:", city)

        # Current weather.

        weather_data = run_async(
            weather_mcp_search(city)
        )

        # Weather forecast.

        forecast_data = run_async(
            forecast_mcp_search(city)
        )

        weather_results = f"""
DESTINATION:
{city}

CURRENT WEATHER:
{compact_text(weather_data, 600)}

WEATHER FORECAST:
{compact_text(forecast_data, 900)}
"""

        weather_results = compact_text(
            weather_results,
            max_chars=1700
        )

        # extract_destination() currently makes
        # one Groq call in mcp_client.py.

        current_llm_calls = (
            state.get("llm_calls", 0) + 1
        )

    except Exception as e:

        print("Weather MCP Error:", str(e))

        weather_results = (
            "Weather information unavailable: "
            + str(e)
        )

        current_llm_calls = (
            state.get("llm_calls", 0)
        )

    return {

        "weather_results": weather_results,

        "messages": [
            AIMessage(
                content="Weather information fetched."
            )
        ],

        "llm_calls": current_llm_calls

    }


# =====================================================
# 9. ITINERARY AGENT
# =====================================================

def itinerary_agent(state: TravelState):

    print("\nINSIDE ITINERARY AGENT\n")

    # Keep the input context compact.

    user_query = compact_text(
        state.get("user_query"),
        max_chars=600
    )

    flight_results = compact_text(
        state.get("flight_results"),
        max_chars=1000
    )

    hotel_results = compact_text(
        state.get("hotel_results"),
        max_chars=1800
    )

    weather_results = compact_text(
        state.get("weather_results"),
        max_chars=1200
    )

    prompt = f"""
You are TripMate AI, a professional travel planner.

USER TRAVEL REQUEST:
{user_query}


FLIGHT INFORMATION:
{flight_results}


HOTEL INFORMATION:
{hotel_results}


WEATHER INFORMATION:
{weather_results}


YOUR TASK:

Create a complete, practical and personalized
travel itinerary using the available information.

OUTPUT FORMAT:

# 1. Trip Summary

- Destination
- Duration
- Travel overview
- Budget overview


# 2. Flight Information

- Departure and arrival information if available.
- Airline information if available.
- Estimated airfare only when supported.
- Clearly mention when live flight data is unavailable.


# 3. Hotel Suggestions

- Hotel names.
- Locations.
- Prices and ratings when available.
- Booking links when available.


# 4. Weather Information

- Current weather.
- Forecast.
- Recommended clothing.
- Weather-related travel precautions.


# 5. Day-by-Day Itinerary

For every travel day include:

- Morning activities.
- Afternoon activities.
- Evening activities.
- Suggested sightseeing.
- Food recommendations.


# 6. Estimated Budget

Include:

- Transportation.
- Accommodation.
- Food.
- Activities.
- Miscellaneous expenses.
- Total estimated cost.

Do not invent prices.
Clearly label estimates and unavailable costs.


# 7. Final Recommendations

Provide useful travel tips and suggestions.


IMPORTANT RULES:

1. Never invent flight availability.
2. Never invent hotel booking details.
3. Never invent weather information.
4. Do not repeat the same information.
5. Keep the response concise but useful.
6. Respect the user's stated budget.
7. Mention unavailable information honestly.
8. Use clear Markdown formatting.
"""

    try:

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are TripMate AI, "
                        "an expert multi-agent "
                        "travel planning assistant."
                    )
                ),

                HumanMessage(
                    content=prompt
                )
            ]
        )

        itinerary = response.content

    except Exception as e:

        print("Itinerary Generation Error:", str(e))

        itinerary = f"""
# TripMate AI - Travel Plan

Your travel information was collected,
but the final itinerary could not be
generated by the AI model.

Error: {str(e)}

Collected Information:

## Flights
{flight_results}

## Hotels
{hotel_results}

## Weather
{weather_results}

Please try again after the Groq request
limit has recovered.
"""

    return {

        "itinerary": itinerary,

        "messages": [
            AIMessage(
                content="Travel itinerary generated."
            )
        ],

        "llm_calls": (
            state.get("llm_calls", 0) + 1
        )

    }


# =====================================================
# 10. FINAL RESPONSE AGENT
# =====================================================

def final_agent(state: TravelState):

    print("\nINSIDE FINAL AGENT\n")

    # Do not call Groq again.
    # The Itinerary Agent already generated
    # the complete final travel response.

    final_answer = state.get(
        "itinerary",
        "Unable to generate travel itinerary."
    )

    return {

        "messages": [
            AIMessage(
                content=final_answer
            )
        ]

    }


# =====================================================
# 11. BUILD LANGGRAPH WORKFLOW
# =====================================================

graph = StateGraph(TravelState)

# Register agents.

graph.add_node(
    "flight_agent",
    flight_agent
)

graph.add_node(
    "hotel_agent",
    hotel_agent
)

graph.add_node(
    "weather_agent",
    weather_agent
)

graph.add_node(
    "itinerary_agent",
    itinerary_agent
)

graph.add_node(
    "final_agent",
    final_agent
)


# Sequential workflow.

graph.add_edge(
    START,
    "flight_agent"
)

graph.add_edge(
    "flight_agent",
    "hotel_agent"
)

graph.add_edge(
    "hotel_agent",
    "weather_agent"
)

graph.add_edge(
    "weather_agent",
    "itinerary_agent"
)

graph.add_edge(
    "itinerary_agent",
    "final_agent"
)

graph.add_edge(
    "final_agent",
    END
)


# =====================================================
# 12. POSTGRESQL CHECKPOINTER
# =====================================================

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)

# Initialize PostgreSQL checkpoint tables.

checkpointer.setup()

# Compile graph with persistent memory.

travel_graph = graph.compile(
    checkpointer=checkpointer
)


# =====================================================
# 13. FASTAPI INTEGRATION
# =====================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None
):

    if not user_input or not user_input.strip():

        raise ValueError(
            "Please provide a travel request."
        )

    if not thread_id:

        thread_id = (
            f"user_{uuid.uuid4().hex}"
        )

    config = {

        "configurable": {
            "thread_id": thread_id
        }

    }

    initial_state = {

        "messages": [
            HumanMessage(
                content=user_input
            )
        ],

        "user_query": user_input,

        "flight_results": "",

        "hotel_results": "",

        "weather_results": "",

        "itinerary": "",

        "llm_calls": 0

    }

    print("\n================================")
    print("STARTING TRIPMATE AI WORKFLOW")
    print("================================\n")

    result = travel_graph.invoke(
        initial_state,
        config=config
    )

    # Final Agent appends the final answer
    # as the last AIMessage.

    final_answer = result["messages"][-1].content

    print("\n================================")
    print("TRIPMATE AI WORKFLOW COMPLETED")
    print("================================\n")

    return {

        "thread_id": thread_id,

        "answer": final_answer,

        "flight_results": result.get(
            "flight_results",
            ""
        ),

        "hotel_results": result.get(
            "hotel_results",
            ""
        ),

        "weather_results": result.get(
            "weather_results",
            ""
        ),

        "itinerary": result.get(
            "itinerary",
            ""
        ),

        "llm_calls": result.get(
            "llm_calls",
            0
        )

    }
