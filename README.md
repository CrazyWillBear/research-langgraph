# Cogito AI

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Configuration](#configuration-default-values-are-for-local-testing-will-update-once-containerized)
- [How it works](#how-it-works-high-level-flow)
- [Architecture Overview](#architecture-overview)
- [Licensing](#licensing)

## Overview

Cogito AI is a Q&A style agentic research assistant for philosophy research. It uses LangGraph/LangChain components, vector search, and LLMs to gather, assess, and summarize primary philosophical sources to build detailed answers with citations.

## Features

- Conversation normalization and summarization.
- Structured query generation for semantic search (with optional author/source filters).
- Batched vector DB queries ([Qdrant](https://qdrant.tech/)) with fuzzy matching to stored metadata ([PostgreSQL](https://www.postgresql.org/)).
- Parallel summarization of retrieved resources.
- Resource sufficiency classifier (Yes/No) with optional feedback generation.
- Final response generation with citations and quoted evidence.

## Prerequisites

- Python 3.10+ and `pip install -r requirements.txt`
- GPU strongly recommended for the default embedding model (`BAAI/bge-large-en-v1.5`) used by `sentence-transformers`. CPU runs are possible but much slower and may require changing the embedder device.
- Running services (I plan to containerize and publish databases later):
  - Qdrant vector DB (local or remote).
  - PostgreSQL containing filter metadata (authors/sources) used by `PostgresFilters` (the code expects a DB named `filters` by default).
- LLM access (change model usage in `ai/subgraphs/research_agent/model_config.py` as needed):
  - OpenAI-compatible API keys if using `langchain_openai.ChatOpenAI` models (used in `ai/models/gpt.py`) — set `OPENAI_API_KEY`.
  - Ollama or local LLM for `langchain_ollama.ChatOllama`.

## Usage

1. Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source venv/bin/activate
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure database connections in `dbs/qdrant.py` and `dbs/postgres_filters.py`.
4. Set `OPENAI_API_KEY` environment variable or configure local LLM access as needed in `ai/subgraphs/research_agent/model_config.py`.
5. Run the interactive CLI:

    ```bash
    python main.py
    ```
   
## Configuration (default values are for local testing; will update once containerized)

- Qdrant client (`dbs/qdrant.py`)
  - URL (default: `"localhost"`)
  - PORT (default: `6334`)
  - COLLECTION (default: `"philosophy"`)
- PostgreSQL client(`dbs/postgres_filters.py`)
  - HOST (default: `"localhost"`)
  - PORT (default: `5432`)
  - DBNAME (default: `"filters"`)
  - USER (default: `"munir"`)
  - PASSWORD (default: `"123"`)
- Embeddings (`embed/embed.py`)
  - DEVICE (default: `"cuda"` if GPU available, else `"cpu"`)
- LLM Models (`ai/subgraphs/research_agent/model_config.py`)
  - Change model classes and parameters as needed for your LLM access.

## How it works (high-level flow)

1. The user interacts through `main.py`; the last human message + conversation history are normalized in `create_conversation`.
2. `write_queries` produces structured queries (JSON schema `QueryAndFilters`) for semantic search.
3. `query_vector_db` batch-queries Qdrant using embedded queries, then summarizes retrieved documents (parallelized).
4. `assess_resources` decides if sufficient research exists; if not, the loop writes new queries and fetches more resources.
5. Once satisfied, `summarize` synthesizes a final answer that cites the gathered sources.

## Architecture Overview

- The agent is implemented as a small state graph in `ResearchAgent` (see `ai/subgraphs/research_agent/research_agent.py`).
- Individual nodes live in `ai/subgraphs/research_agent/nodes/`:
  - `create_conversation.py` — normalize and summarize incoming conversation/history.
  - `write_queries.py` — produce structured vector search queries (Pydantic models).
  - `query_vector_db.py` — call Qdrant, summarize retrieved resources in parallel.
  - `assess_resources.py` — decide whether more search is needed, optionally produce feedback.
  - `summarize.py` — synthesize final response using gathered research.
- Model configuration per-node is in `ai/subgraphs/research_agent/model_config.py`.
- Vector DB client wraps Qdrant and fuzzily maps filters to author/source names in Postgres (`dbs/qdrant.py`).
- Embeddings: `embed/embed.py` wraps [SentenceTransformers](https://huggingface.co/sentence-transformers) ([BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) by default).
- `main.py` provides a simple interactive CLI loop for conversation and invoking the research agent.

## Licensing + Copyright
Copyright (c) William Chastain. All rights reserved.
This software is licensed under the PolyForm Noncommercial License 1.0.0. This project, while open-source, *may not* be used for commercial purposes. See the [LICENSE](LICENSE.md) file for details.
