
import asyncio
import os
from pathlib import Path


from langchain_groq import ChatGroq
from langchain.agents import create_agent

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from this project's .env file
load_dotenv()

github_token = os.getenv("GITHUB_PA_TOKEN")
if not github_token:
    raise RuntimeError(
        "GITHUB_PERSONAL_ACCESS_TOKEN is missing. "
        "Check your .env file."
    )

# Path to the Go binary we built
github_server = Path.home() / "go" / "bin" / "github-mcp-server"

if not github_server.exists():
    raise FileNotFoundError(
        f"GitHub MCP server not found at {github_server}"
    )

client = MultiServerMCPClient(
    {
        "github": {
            "transport": "stdio",
            "command": str(github_server),
            "args": ["--read-only", "stdio"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
            },
        }
    }
)




async def main():
    # Load tools from GitHub MCP
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} GitHub MCP tools")

    # Read-only tools are enforced by the MCP server configuration
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL"),
        temperature=0,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
            You are a GitHub assistant.

            Use the available GitHub tools to answer questions about repositories,
            commits, issues, pull requests, and users.

            Instructions:
            - Use tools for GitHub information; do not invent results.
            - Ask the user for missing required details, such as repository owner
              or repository name.
            - For repository searches, use search_repositories.
            - For commit lists, use list_commits.
            - For exact commit details, use get_commit.
            - For repository contents, use get_file_contents.
            - When asked how many repositories match a search, use the search
              results' reported total count if available. Do not confuse the
              number of returned results with the total count.
            - Summarize results clearly and include repository names and URLs
              when available.
            - Do not perform write operations.
            """,
    )

    print("\nGitHub MCP Agent is ready!")
    print("Examples:")
    print("- Search repositories about LangChain")
    print("- List commits from facebook/react-native")
    print("- Find repositories owned by vercel")
    print("- Type 'exit' to quit")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        response = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        final_message = response["messages"][-1]
        print("\nAgent:", final_message.content)


if __name__ == "__main__":
    asyncio.run(main())