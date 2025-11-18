from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from ai.research_agent.model_config import MODEL_CONFIG
from ai.research_agent.schemas.graph_state import ResearchAgentState
from ai.research_agent.schemas.query import QueryAndFilters


class QueryAndFiltersList(BaseModel):
    """
    Schema for a list of QueryAndFilters objects.
    Used when multiple queries with filters are needed.
    """

    queries: list[QueryAndFilters]


def reason_about_queries(messages: list):
    """Generate reasoning about how to break down a user's question into multiple targeted search queries."""
    # Get model
    model = MODEL_CONFIG["write_queries"]

    # --- Build prompts ---
    system_prompt = (
        "You are a reasoning agent for a philosophical research system. Your job is to reason about how to "
        "break down the user's question into targeted search queries.\n\n"
        "Important context: The search retrieves large chunks (1000 tokens) containing full philosophical "
        "excerpts, not just snippets. This means:\n"
        "- Each query should target broader concepts and arguments, not narrow facts\n"
        "- Fewer queries are needed since each result provides substantial context\n\n"
        "Reason about:\n"
        "- What is the core philosophical question or concept?\n"
        "- Which specific authors or works would be most valuable?\n"
        "- Should you approach from different philosophical angles or traditions?\n"
        "- For simple questions: 1-3 queries may suffice\n"
        "- For complex multi-faceted questions: 3-5 queries\n\n"
        "Output ONLY YOUR REASONING, no additional text or AND NO queries!"
    )

    user_prompt = f"Here is the chat history:\n{messages}"

    # --- Invoke LLM ---
    chat_history = [system_prompt, user_prompt]
    result = model.invoke(chat_history)

    return result.content


def write_queries(state: ResearchAgentState):
    """
    Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """
    # Get model
    model = MODEL_CONFIG["write_queries"]
    structured_model = model.with_structured_output(QueryAndFiltersList)

    # --- Extract state variables ---
    resources = state.get("resources", [])
    queries = state.get("queries_made", [])
    messages = state.get("messages", [])

    # --- Reason about queries ---
    reasoning = reason_about_queries(messages)

    # --- Build prompts ---
    system_prompt = (
        "You are a semantic search assistant for philosophical research. Generate targeted search queries "
        "based on your previous reasoning.\n\n"
        "Your reasoning:\n"
        f'"""\n{reasoning}\n"""\n\n'
        "Important guidelines:\n"
        "- The database contains LARGE chunks (1000 tokens) with full philosophical arguments\n"
        "- Use broader, conceptual queries that capture philosophical topics and debates\n"
        "- Examples: 'Kant categorical imperative moral philosophy' not 'define categorical imperative'\n"
        "- CRITICAL: Put author/work names in 'filters', NOT in the search query string\n"
        "- The 'query' field should describe concepts, arguments, and topics only\n"
        "- Filters use fuzzy matching, so approximate names work fine\n\n"
        "Generate 1-3 queries for simple questions, 3-5 for complex multi-faceted questions.\n\n"
        "Each query object has:\n"
        "- 'query': conceptual search string for semantic search\n"
        "- 'filters': object with optional 'author' and/or 'source_title'"
    )

    user_prompt = f"Here is the chat history:\n{messages}"

    # Include previous research results if available
    if len(queries) != 0:
        user_prompt += (
            f"\n\nPrevious research results: {resources}"
            f"\nPrevious queries made: {queries}"
            "\n\nConsider what information gaps remain and avoid redundant queries."
        )

    # --- Invoke LLM ---
    chat_history = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    result = structured_model.invoke(chat_history)

    # --- Update state with new query ---
    return {"queries": result.queries, "queries_made": [*state["queries_made"], *result.queries]}
