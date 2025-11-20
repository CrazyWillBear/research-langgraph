import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState
from dbs.qdrant import Qdrant


def summarize_resource(model, resource_text):
    system_msg = SystemMessage(content=(
        "You are a summarizing agent. Summarize the following resource with these guidelines:\n"
        "- Keep it concise (should be around half the size of original)\n"
        "- Focus on key arguments, concepts, and ideas presented\n"
        "- Retain any important quotes\n"
        "- Return the summary in full sentences and paragraphs\n\n"
    ))

    user_msg = HumanMessage(content=f"Here is a resource to summarize:\n---\n{resource_text}\n---\n")
    res = model.invoke([system_msg, user_msg], reasoning={"effort": "minimal"})
    return gpt_extract_content(res)

def query_vector_db(state: ResearchAgentState, qdrant: Qdrant):
    """
    Query the vector database with the given query and filters.
    Uses fuzzy matching to find best-matching authors and sources from PostgreSQL metadata.
    Returns accumulated resources and increments query count.
    """
    print("::Querying vector database and summarizing sources...", end="", flush=True)
    start = time.perf_counter()

    # --- Get model ---
    model = MODEL_CONFIG["query_vector_db"]

    # --- Extract state variables ---
    queries = state.get("queries")
    new_resources = []
    old_summaries = state.get("resource_summaries", [])
    new_summaries = old_summaries.copy()

    # Query the db
    responses = qdrant.batch_query(queries)
    for payload in responses:
        content = payload.get("text", "")
        author = payload.get("author", "Unknown Author")
        source_title = payload.get("source", "Unknown Source")
        resource_text = f'"""\n{content}\n"""\n- {author}, {source_title}\n'
        new_resources.append(resource_text)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_resource = {executor.submit(summarize_resource, model, r): r for r in new_resources}
        for future in as_completed(future_to_resource):
            summary = future.result()
            new_summaries.append(gpt_extract_content(summary))

    end = time.perf_counter()
    print(f"\r\033[K::Vector database queried and sources summarized in {end - start:.2f}s")

    return {"resource_summaries": new_summaries}
