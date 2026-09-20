
import asyncio
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

FILESYSTEM_PATH = (
    "/Users/karanparihar/Documents/Devs/AI-Learning/"
    "ai-engineering-learning/week-6/filesystem_demo"
)


async def main():
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
            },
            "fetch": {
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-fetch"],
            },
        }
    )

    tools = await client.get_tools()

    print("\nAvailable tools:")
    for tool in tools:
        print(f"- {tool.name}")

    llm = ChatGroq(
            model=os.environ.get("GROQ_MODEL_OPEN"),
            temperature=0,
        )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are an assistant with filesystem and web-fetch tools. "
            "Use the appropriate tools to answer the user's question. "
            "Do not claim to have read a file or webpage unless a tool "
            "successfully returned its content."
        ),
    )

    # user_query = (
    #     "Read notes.txt from the filesystem and summarize its main points."
    # )

    # user_query = (
    #     "Fetch https://reactnative.dev and tell me the page title "
    #     "and a short summary."
    # )

    user_query = input("Ask me: ")

    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": user_query}
            ]
        }
    )

    print("\nAgent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())