from typing import Callable

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai.subgraphs.research_agent.nodes.assess_resources import assess_resources
from ai.subgraphs.research_agent.nodes.create_conversation import create_conversation
from ai.subgraphs.research_agent.nodes.query_vector_db import query_vector_db
from ai.subgraphs.research_agent.nodes.write_response import write_response
from ai.subgraphs.research_agent.nodes.write_queries import write_queries
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState
from dbs.postgres_filters import PostgresFilters
from dbs.qdrant import Qdrant


class ResearchAgent:
    """Research Agent subgraph for querying vector DBs and summarizing results."""

    def __init__(self, qdrant = None, postgres_filters = None):
        """Initialize the Research Agent subgraph."""

        self.graph = None
        self.qdrant = qdrant if qdrant is not None else Qdrant()
        self.postgres_filters = postgres_filters if postgres_filters is not None else PostgresFilters()

    @staticmethod
    def wrap(func: Callable, *args, **kwargs) -> Callable:
        """Wrap a node so it receives `state` plus any extra args/kwargs."""

        def wrapped(state):
            return func(state, *args, **kwargs)

        return wrapped

    def run(self, conversation: dict) -> str:
        """Invoke the Research Agent subgraph with a conversation."""

        res = self.graph.invoke(conversation)
        return res.get('response', 'No response available')

    def build(self) -> None:
        """
        Build the Research Agent subgraph.
        Constructs a state graph that queries vector databases, evaluates results,
        and generates summaries until satisfaction criteria are met.
        """

        # --- Initialize graph ---
        g = StateGraph(ResearchAgentState)

        # --- Add nodes ---
        g.add_node("create_conversation", create_conversation)
        g.add_node("write_queries", write_queries)
        g.add_node("query_vector_db", self.wrap(query_vector_db, self.qdrant))
        g.add_node("assess_resources", assess_resources)
        g.add_node("write_response", write_response)

        # --- Add edges ---
        g.add_edge(START, "create_conversation")
        g.add_edge("create_conversation", "write_queries")
        g.add_edge("write_queries", "query_vector_db")
        g.add_edge("query_vector_db", "assess_resources")
        g.add_edge("write_response", END)

        # --- Add conditional edges ---
        g.add_conditional_edges(
            "assess_resources",
            lambda state: "write_response" if state["query_satisfied"] else "write_queries"
        )

        self.graph = g.compile()
