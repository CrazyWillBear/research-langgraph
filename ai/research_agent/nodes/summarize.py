from langchain_core.messages import SystemMessage, HumanMessage

from ai.research_agent.model_config import MODEL_CONFIG
from ai.research_agent.schemas.graph_state import ResearchAgentState


def summarize(state: ResearchAgentState):
    """
    Summarize the most relevant sources from the research results to inform a response.
    Combines conversation history with research resources to generate a cited answer.
    """

    # --- Get model ---
    model = MODEL_CONFIG["summarize"]

    # --- Extract state variables ---
    messages = state["messages"]
    resources = state["resources"]

    # --- Define system prompt ---
    system_prompt = ("You must now respond to the user's last message given the following resources that you've "
                     "'researched'. Use specific quotes, respond in a chat-like manner, and cite all sources at the "
                     "end using MLA8 format.\n\n"
                     "Reason step by step in <thinking>...</thinking> tags. Consider:\n"
                     "- What resources are most relevant to the question?\n"
                     "- Based on those resources, what's the answer?\n"
                     "- How do the resources support that answer?\n\n"
                     "After reasoning, output your final response.")

    # --- Build chat history ---
    chat_history = messages + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=str(resources))
    ]

    # --- Invoke LLM ---
    result = model.invoke(chat_history)

    # --- Extract and clean output ---
    text = getattr(result, "content", str(result)) or ""
    if "<thinking>" in text:
        text = text.split("</thinking>")[-1].strip()

    # --- Update state ---
    return {"response": text}
