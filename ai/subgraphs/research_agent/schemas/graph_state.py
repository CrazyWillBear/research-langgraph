from typing import TypedDict

from ai.subgraphs.research_agent.schemas.conversation import Conversation


class ResearchAgentState(TypedDict):
    """State schema for the Research Agent subgraph."""

    messages: list              # Conversation messages
    conversation: Conversation  # Contains final user message + summarized context

    response: str               # Final response generated

    queries: list               # Queries for vector db
    queries_feedback: str       # Feedback for research queries
    query_satisfied: bool       # If the query results were satisfactory

    resource_summaries: list    # Recap summaries of the resources
