from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ai.research_agent.research_agent import ResearchAgent


if __name__ == "__main__":
    # Conversation setup
    conversation = {
        "messages": [
            SystemMessage(content="You are a helpful philosophical research assistant.")
        ]
    }

    # Build agent
    agent = ResearchAgent()

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        conversation["messages"].append(HumanMessage(content=user_input))

        # Run agent
        output = agent.run(conversation)
        print("Thinking...", end="")
        print("\rAI:", output)

        conversation["messages"].append(AIMessage(content=output))
