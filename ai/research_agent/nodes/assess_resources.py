from langchain_core.messages import SystemMessage, HumanMessage

from ai.research_agent.model_config import MODEL_CONFIG
from ai.research_agent.schemas.graph_state import ResearchAgentState

# Max queries allowed
MAX_QUERIES = 10

def assess_resources(state: ResearchAgentState):
    """
    Check if the research results are sufficient to answer the user's query.
    Returns True if results are adequate or max query limit is reached; False otherwise.
    """

    # --- Get model ---
    model = MODEL_CONFIG["assess_resources"]

    # --- Build prompts ---
    system_msg = SystemMessage(content=(
        "You are a reasoning assistant that evaluates whether the provided research is sufficient to answer the user's "
        "query adequately.\n\n"
        "Think step-by-step using chain-of-thought reasoning in <thinking>...</thinking> tags:\n"
        "- Does the research cover all aspects of the user's question?\n"
        "- Are there significant gaps in the information?\n"
        "- Is the information detailed and specific enough?\n"
        "- Would the user be satisfied with an answer based on this research?\n\n"
        "After your reasoning, output ONLY 'Yes' or 'No' outside the thinking tags."
    ))

    user_msg = HumanMessage(content=(
        f"Conversation messages:\n{state['messages']}\n\n"
        f"Research results:\n{state.get('resources', [])}"
    ))

    result = model.invoke([system_msg, user_msg])

    # --- Extract decision after thinking tags ---
    text = getattr(result, "content", str(result)) or ""
    if "</thinking>" in text:
        text = text.split("</thinking>", 1)[-1].strip()

    # Parse decision
    answer = text.lower().strip()
    satisfied = answer.startswith("yes") or (len(state["queries_made"]) >= MAX_QUERIES)

    return {"query_satisfied": satisfied}
