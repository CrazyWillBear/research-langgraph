import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from ai.subgraphs.research_agent.research_agent import ResearchAgent

START_TEXT = \
r"""                                                               
      # ###                                                          
    /  /###  /                            #                          
   /  /  ###/                            ###        #                
  /  ##   ##                              #        ##                
 /  ###                                            ##                
##   ##           /###        /###      ###      ########    /###    
##   ##          / ###  /    /  ###  /   ###    ########    / ###  / 
##   ##         /   ###/    /    ###/     ##       ##      /   ###/  
##   ##        ##    ##    ##     ##      ##       ##     ##    ##   
##   ##        ##    ##    ##     ##      ##       ##     ##    ##   
 ##  ##        ##    ##    ##     ##      ##       ##     ##    ##   
  ## #      /  ##    ##    ##     ##      ##       ##     ##    ##   
   ###     /   ##    ##    ##     ##      ##       ##     ##    ##   
    ######/     ######      ########      ### /    ##      ######    
      ###        ####         ### ###      ##/      ##      ####     
                                   ###                               
                             ####   ###                              
                           /######  /#                               
                          /     ###/                                 

Copyright (c) 2025 William Chastain (williamchastain.com). All rights reserved.
This software is licensed under the PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0)

Cogito AI: An agentic Q&A research assistant for philosophy.
Type 'exit' or 'quit' to end the conversation.

-=-=-=-
"""

if __name__ == "__main__":
    """Main entry point for running the Cogito AI research assistant in a console loop."""

    print(START_TEXT)

    # Conversation setup
    conversation = {
        "messages": [
            SystemMessage(content="You are a helpful philosophical research assistant.")
        ]
    }

    # Build agent
    agent = ResearchAgent()
    agent.build()

    # Main loop
    while True:
        # Get user input
        user_input = input("[User]: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        conversation["messages"].append(HumanMessage(content=user_input))
        print()

        # Run agent with timing
        start = time.perf_counter()         # start timing
        output = agent.run(conversation)    # invoke/run agent
        end = time.perf_counter()           # end timing

        # Print output and time taken
        print(f"\n[AI]:\n---\n{output}\n---\nTime was {end - start:.2f}s\n")

        # Append AI message to conversation
        conversation["messages"].append(AIMessage(content=output))
