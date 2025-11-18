# Cogito AI Service (will rename repo later)

## Overview
This is the LangGraph/LangChain component of what will become a larger academic research tool. The goal of this project
is to assist scholars in their research with AI tools, enabling them to gather and analyze information from various
sources.

## Todo
- [ ] Optimize *everything*
  - [ ] Keep one global QdrantClient (optionally gRPC) + Postgres connection pool
  - [ ] Remove extra LLM reasoning pass (single query generation call, cap to 1–3 queries)
  - [ ] Trim chat history with a rolling text summary (replace full past messages)
  - [ ] Cache authors/sources list in memory with TTL; skip DB hit if unchanged
  - [ ] Batch embeddings in one API call; add a simple in-process LRU embedding cache
  - [ ] Limit resources passed to LLM: top K (e.g. 5–8) by score, truncate long texts
  - [ ] Batch Qdrant queries
  - 

## More info
Please read [my blog post about this project](https://blog.williamchastain.com/Cogito-(Ergo-Sum)) for information about
the project's current state, goals, and functionality.

## Project Structure
```aiignore
embed/*                             # Text embedding model implementation
chat/*                              # Chat history related code (not used ATM)
nodes/*                             # LangGraph nodes for what will be the overarching graph (not used ATM)
ai                                  # AI-related code (agent implementations, LangGraph graphs, etc.)
├───models/*                        # LLM model implementations (GPT & LLaMA)
├───nodes/*                         # Nodes for the overarching graph (not used ATM)
├───research_agent_subgraph         # Subgraph for research agent
│   │   graph.py                    # Defines the subgraph's nodes and edges
│   │   graph_state.py              # Defines the subgraph's state
│   │   query_filter_schemas.py     # Pydantic schemas for query filtering
│   ├───nodes                       # Defines the subgraph's nodes
│   │   │   assess_summary.py       # Node to assess summary quality
│   │   │   check_statisfaction.py  # Node to check satisfaction of query results
│   │   │   entry.py                # Entry node
│   │   │   query_vector_db.py      # Node to query vector DB
│   │   │   summarize.py            # Node to summarize/respond to user
│   │   │   write_query.py          # Node to write queries for vector DB
└───util/*                          # Util functions
```

## License
This project is licensed under the PolyForm Noncommercial License 1.0.0. You may use and modify the code for personal,
educational, or internal purposes, but **_not for commercial purposes_**.