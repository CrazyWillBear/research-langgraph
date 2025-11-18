"""
Query vector database node for the research agent.

This module provides functionality to query a Qdrant vector database with
multiple queries concurrently, using global connection pools for optimal
performance. Queries are batched and executed in parallel using ThreadPoolExecutor.

Key optimizations:
- Uses global QdrantClient singleton (with gRPC support)
- Uses global Postgres connection pool
- Batches queries for concurrent execution
- Fuzzy matching for author and source filters
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client.http.models import Filter, MatchValue, FieldCondition
from rapidfuzz import process

from ai.db_connections import get_qdrant_client, get_postgres_pool
from ai.research_agent.schemas.graph_state import ResearchAgentState
from embed.embed import embed

# --- Constants ---
VEC_COLLECTION = os.getenv("VEC_COLLECTION", "philosophy")


def _execute_single_query(qdrant_client, query, all_authors, all_sources):
    """
    Execute a single query against the vector database.
    
    Args:
        qdrant_client: QdrantClient instance
        query: Query object with query string and filters
        all_authors: List of all available authors for fuzzy matching
        all_sources: List of all available sources for fuzzy matching
    
    Returns:
        List of points from the query results
    """
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

    return results.points


def query_vector_db(state: ResearchAgentState):
    """
    Query the vector database with the given queries and filters.
    Uses global QdrantClient and Postgres connection pool for optimal performance.
    Batches queries to execute them concurrently.
    Uses fuzzy matching to find best-matching authors and sources from PostgreSQL metadata.
    Returns accumulated resources and increments query count.
    """

    # --- Extract state variables ---
    queries = state.get("queries")
    old_resources = state.get("resources", [])
    new_resources = old_resources.copy()

    # If no queries, return current resources unchanged
    if not queries:
        return {"resources": new_resources}

    # --- Keep track of ids for de-duplication efforts ---
    seen_ids = set()

    # --- Get global database connections ---
    qdrant_client = get_qdrant_client()
    postgres_pool = get_postgres_pool()

    # Get connection from pool
    conn = postgres_pool.getconn()
    try:
        with conn.cursor() as cur:
            # --- Query DB once for authors and sources ---
            cur.execute("SELECT DISTINCT authors FROM filters;")
            all_authors = [row[0] for row in cur.fetchall()]
            cur.execute("SELECT DISTINCT sources FROM filters;")
            all_sources = [row[0] for row in cur.fetchall()]

        # --- Batch queries: execute all queries concurrently ---
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(_execute_single_query, qdrant_client, query, all_authors, all_sources): query
                for query in queries
            }

            # Collect results as they complete
            for future in as_completed(future_to_query):
                try:
                    points = future.result()
                    # Deduplicate points and add to resources
                    for point in points:
                        if point.id not in seen_ids:
                            seen_ids.add(point.id)
                            new_resources.append(point)
                except Exception as e:
                    # Log error but continue processing other queries
                    print(f"Error processing query: {e}")

    finally:
        # Return connection to pool (don't close it)
        postgres_pool.putconn(conn)

    # --- Update state ---
    return {"resources": new_resources}
