from langchain_openai import ChatOpenAI


# GPT 5 low temperature model
gpt5 = ChatOpenAI(
    model="gpt-5",
    temperature=0.0
)

# GPT 5 mini low temperature model
gpt5_mini = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.0
)

# GPT 5 nano low temperature model
gpt5_nano = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0
)


def gpt_extract_content(result):
    """Extract the main text content from a model.invoke() result, ignoring any 'reasoning' or auxiliary objects."""

    content_list = getattr(result, "content", result)

    # If it's already a string, return it
    if isinstance(content_list, str):
        return content_list.strip()

    # If it's a list of messages, find the first text message
    if isinstance(content_list, list):
        for msg in content_list:
            if isinstance(msg, dict) and msg.get("type") == "text":
                return msg.get("text", "").strip()

    # Fallback: convert to string
    return str(result).strip()
