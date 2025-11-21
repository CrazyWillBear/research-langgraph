import time

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState

# Max queries allowed
MAX_SOURCES = 3

def get_feedback(last_message, resource_summaries):
    """Get feedback on why the current research resources are insufficient to answer the user's query."""

    # Get configured feedback model
    feedback_model = MODEL_CONFIG["assess_resources_feedback"]

    # Build feedback prompt (system and user message)
    feedback_system_msg = SystemMessage(content=(
        "You are an assistant that provides feedback on why the current research resources are insufficient to answer "
        "the user's query. Provide specific reasons and suggestions for what additional research is needed.\n"
    ))

    feedback_user_msg = HumanMessage(content=(
        f"Here is the user's last message:\n{last_message}\n\n"
        f"Here are summaries of the research results obtained so far:\n{resource_summaries}\n"
        "Explain why this research is insufficient and what additional research is needed."
    ))

    # Invoke feedback model and extract output
    feedback_result = feedback_model.invoke([feedback_system_msg, feedback_user_msg], reasoning={"effort": "minimal"})
    feedback = gpt_extract_content(feedback_result)

    return feedback

def assess_resources(state: ResearchAgentState):
    """Assess whether the collected research resources are sufficient to answer the user's query."""

    # Start timing and log
    print("::Assessing resources...", end="", flush=True)
    start = time.perf_counter()

    # Extract graph state variables
    conversation = state.get("conversation", {})
    resource_summaries = state.get("resource_summaries") or "No research resources collected yet."

    # Get configured model
    classifier_model = MODEL_CONFIG["assess_resources_classifier"]

    # Build prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a reasoning assistant that evaluates whether the provided research is sufficient to answer the user's query.\n"
        "Decide if the current research can support a satisfactory answer now. Just make sure it at least covers all"
        "aspects of the question.\n\n"
        "Return NOTHING but 'Yes' if the research is sufficient, or 'No' if more research is needed.\n"
    ))

    last_message = conversation.get("last_user_message", "No last user message found")
    user_msg = HumanMessage(content=(
        f"Here is the user's last message:\n{last_message}\n\n"
        f"Here are summaries of the research results obtained so far:\n{resource_summaries}\n"
    ))

    # Invoke model and extract output
    result = classifier_model.invoke([system_msg, user_msg], reasoning={"effort": "minimal"})
    query_satisfied = gpt_extract_content(result).strip().lower() == "yes"  # yes = True, otherwise False

    # If not satisfied, get feedback on what additional research is needed
    if not query_satisfied:
        feedback = get_feedback(last_message, resource_summaries)
    else:
        feedback = ""

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Resources assessed in {end - start:.2f}s")

    return {"query_satisfied": query_satisfied or len(resource_summaries) >= MAX_SOURCES, "queries_feedback": feedback}

