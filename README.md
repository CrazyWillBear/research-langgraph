# Cogito AI Service (will rename repo later)

## Overview
This is the LangGraph/LangChain component of what will become a larger academic research tool. The goal of this project
is to assist scholars in their research with AI tools, enabling them to gather and analyze information from various
sources.

## Todo
- [ ] Optimize *everything*
  - [ ] Keep one global QdrantClient (maybe switch to gRPC?) + Postgres connection pool
  - [ ] Remove extra LLM reasoning pass in write_queries
  - [ ] Trim chat history by summarizing old chat
  - [ ] Cache authors/sources list in memory with TTL; skip DB hit if unchanged
  - [ ] Batch embeddings in one API call by switching to OpenAI model
  - [ ] Batch Qdrant queries

## More info
Please read [my blog post about this project](https://blog.williamchastain.com/Cogito-(Ergo-Sum)) for information about
the project's current state, goals, and functionality.

## License
This project is licensed under the PolyForm Noncommercial License 1.0.0. You may use and modify the code for personal,
educational, or internal purposes, but **_not for commercial purposes_**.