import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.conversation import Conversation
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState


def create_conversation(state: ResearchAgentState):
    """
    Normalize incoming invocation dict into the `conversation` key required by ResearchAgentState.
    Accepts flexible caller input: `messages`, `history`, or a prebuilt `conversation`.
    """
    print("::Starting conversation and summarization...", end="", flush=True)
    start = time.perf_counter()

    # Get model
    model = MODEL_CONFIG["create_conversation"]

    # Extract incoming raw messages
    if 'messages' in state and isinstance(state['messages'], list):
        incoming_messages = state['messages']
    else:
        incoming_messages = []

    # Find last user message
    last_user = ''
    for m in reversed(incoming_messages):
        if isinstance(m, HumanMessage):
            last_user = getattr(m, "content", "")
            break

    # Build prior context
    context_parts: List[str] = [
        getattr(m, "content", "") for m in incoming_messages[1:-1]
        if getattr(m, "content", None)
    ]
    context = '\n'.join(context_parts) if context_parts else ''

    # Build prompts
    system_msg = SystemMessage(content=(
        "You are a conversation summarizer. Your job is to summarize the conversation between the user and the AI "
        "assistant, focusing on the key points discussed, questions asked, and any relevant context that would help.\n"
        "Your summary should at most half the length of the original conversation. If there is no conversation, just "
        "say 'Conversation is empty.'"
    ))

    user_msg = HumanMessage(content=(
        "Conversation history:\n"
        f"{context}\n\n"
    ))

    # Invoke model
    result = model.invoke([system_msg, user_msg], reasoning={"effort": "minimal"})
    summarized = gpt_extract_content(result)

    # Set conversation object
    conversation: Conversation = {
        'last_user_message': last_user,
        'summarized_context': summarized,
    }

    # Initialize remaining required keys if absent
    state.setdefault('response', '')
    state.setdefault('queries', [])
    state.setdefault('queries_feedback', '')
    state.setdefault('query_satisfied', False)
    state.setdefault('resource_summaries', list())

    end = time.perf_counter()
    print(f"\r\033[K::Conversation initialized in {end - start:.2f}s")

    return {"conversation": conversation, "messages": []}
