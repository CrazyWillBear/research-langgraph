import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, MatchValue, FieldCondition
from rapidfuzz import process

from ai.research_agent.schemas.graph_state import ResearchAgentState
from embed.embed import embed

# --- Constants ---
VEC_DB_URL = "http://localhost:6333"
VEC_COLLECTION = "philosophy"


def query_vector_db(state: ResearchAgentState):
    """
    Query the vector database with the given query and filters.
    Uses fuzzy matching to find best-matching authors and sources from PostgreSQL metadata.
    Returns accumulated resources and increments query count.
    """

    # --- Extract state variables ---
    queries = state.get("queries")
    old_resources = state.get("resources", [])
    new_resources = old_resources.copy()

    # --- Keep track of ids for de-duplication efforts ---
    seen_ids = set()

    # --- Initialize database clients ---
    qdrant_client = QdrantClient(VEC_DB_URL)

    try:
        with psycopg2.connect(
                host="localhost",
                port=5432,
                dbname="filters",
                user="munir",
                password="123"
        ) as conn, \
                conn.cursor() as cur:

            # --- Query DB once for authors and sources ---
            cur.execute("SELECT DISTINCT authors FROM filters;")
            all_authors = [row[0] for row in cur.fetchall()]
            cur.execute("SELECT DISTINCT sources FROM filters;")
            all_sources = [row[0] for row in cur.fetchall()]

            # --- Conduct queries ---
            for query in queries:
                vector = embed(query.query)
                filters = query.filters

                # Build filter conditions
                _filter_conditions = []
                if filters is not None:
                    author = filters.author
                    source_title = filters.source_title

                    # Fuzzy match author
                    if author is not None:
                        best_author = process.extractOne(author, all_authors)
                        if best_author:
                            _filter_conditions.append(
                                FieldCondition(key="author", match=MatchValue(value=best_author[0]))
                            )

                    # Fuzzy match source
                    if source_title is not None:
                        best_source = process.extractOne(source_title, all_sources)
                        if best_source:
                            _filter_conditions.append(
                                FieldCondition(key="source", match=MatchValue(value=best_source[0]))
                            )

                # Build filter only if we have conditions
                _filter = Filter(must=_filter_conditions) if _filter_conditions else None

                # Query vector database
                results = qdrant_client.query_points(
                    collection_name=VEC_COLLECTION,
                    query=vector,
                    limit=2,
                    query_filter=_filter
                )

                # Deduplicate points and add to resources
                for point in results.points:
                    if point.id not in seen_ids:
                        seen_ids.add(point.id)
                        new_resources.append(point)

    finally:
        # --- Close Qdrant client ---
        qdrant_client.close()

    # --- Update state ---
    return {"resources": new_resources}
