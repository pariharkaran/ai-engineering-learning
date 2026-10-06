
import asyncio
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

FILESYSTEM_PATH = ("/Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-6")

# /Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical
async def main():

    # 1. Configure the MCP server
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    FILESYSTEM_PATH,
                ],
            }
        }
    )

    # 2. Discover the MCP tools
    all_tools = await client.get_tools()

    # 3. Allow only read-only tools for this exercise
    allowed_tools = {
        "read_text_file",
        "read_multiple_files",
        "list_directory",
        "list_directory_with_sizes",
        "directory_tree",
        "search_files",
        "get_file_info",
        "list_allowed_directories",
    }

    tools = [
        tool for tool in all_tools
        if tool.name in allowed_tools
    ]

    print("Read-only MCP tools:")
    for tool in tools:
        print("-", tool.name)

    # 4. Initialize Groq
    # Set GROQ_API_KEY in your environment first.
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL_OPEN"),
        temperature=0,
    )

    # 5. Create the LangChain agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are a filesystem assistant. "
            "Use the provided filesystem tools when needed. "
            "Only access files within the configured directory. "
            "Do not claim to have read a file unless a tool "
            "actually returned its contents. "
            "Answer using the tool results. "
            "If the requested information is unavailable, "
            "say so clearly."
        ),
    )

    # 6. Ask the agent a question
    user_query = input("\nAsk a filesystem question: ")

    response = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": user_query}
            ]
        }
    )

    # 7. Print the final answer
    print("\n--- Agent Answer ---")

    for message in response["messages"]:
        if message.type == "ai" and message.content:
            print(message.content)


if __name__ == "__main__":
    asyncio.run(main())