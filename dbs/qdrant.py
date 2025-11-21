from qdrant_client import QdrantClient, models
from qdrant_client.http.models import MatchValue, FieldCondition, Filter
from rapidfuzz import process

from dbs.postgres_filters import PostgresFilters
from dbs.query import QueryAndFilters
from embed.embed import Embeder


class Qdrant:
    """Qdrant vector database client with fuzzy-matched filtering."""

    # --- Constants ---
    URL = "localhost"
    PORT = 6334
    COLLECTION = "philosophy"

    # --- Methods ---
    def __init__(self):
        """Initialize Qdrant database client."""
        # --- Initialize database clients ---
        self.client = QdrantClient(url=self.URL, grpc_port=self.PORT, prefer_grpc=True)
        self.postgres_client = PostgresFilters()
        self.embedder = Embeder()

    def __enter__(self):
        """Enter context manager for Qdrant client."""
        return self

    def __exit__(self):
        """Exit context manager for Qdrant client."""
        # --- Close Qdrant client ---
        self.client.close()

    def query(self, query: QueryAndFilters) -> list[dict]:
        """Query the Qdrant vector database with fuzzy-matched filters."""
        res = []
        seen_ids = set()

        all_authors = self.postgres_client.all_authors
        all_sources = self.postgres_client.all_sources

        vector = self.embedder.embed(query.query)
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
        results = self.client.query_points(
            collection_name=self.COLLECTION,
            query=vector,
            limit=2,
            query_filter=_filter
        )

        # Deduplicate points and add to resources
        for point in results.points:
            if point.id not in seen_ids:
                seen_ids.add(point.id)
                res.append(point.payload)

        return res

    def batch_query(self, queries: list[QueryAndFilters]):
        """Batch query Qdrant with per-query fuzzy filters."""

        results_out = []
        all_authors = self.postgres_client.all_authors
        all_sources = self.postgres_client.all_sources

        # --- Batch embed all query texts ---
        query_texts = [q.query for q in queries]
        vectors = self.embedder.embed_batch(query_texts)

        # --- Build SearchRequest list ---
        search_requests = []

        for q, vector in zip(queries, vectors):
            filter_obj = None

            if q.filters:
                conditions = []
                f = q.filters

                # fuzzy match author
                if f.author:
                    best_author = process.extractOne(f.author, all_authors)
                    if best_author:
                        conditions.append(
                            FieldCondition(
                                key="author",
                                match=MatchValue(value=best_author[0])
                            )
                        )

                # fuzzy match source
                if f.source_title:
                    best_source = process.extractOne(f.source_title, all_sources)
                    if best_source:
                        conditions.append(
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=best_source[0])
                            )
                        )

                if conditions:
                    filter_obj = Filter(must=conditions)

            # Add search request
            search_requests.append(
                models.QueryRequest(
                    query=vector,
                    limit=2,
                    filter=filter_obj,
                    with_payload=True,
                    with_vector=False
                )
            )

        # --- Execute all queries in a single batch ---
        batch_results = self.client.query_batch_points(
            collection_name=self.COLLECTION,
            requests=search_requests
        )

        # --- Convert Qdrant results into your desired payload lists ---
        for response in batch_results:
            seen_ids = set()
            for point in response.points:
                if point.id not in seen_ids:
                    seen_ids.add(point.id)
                    results_out.append(point.payload)

        return results_out
