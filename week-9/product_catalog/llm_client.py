
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

SERVER_PATH = Path(__file__).parent / "server.py"


async def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your environment or .env file."
        )

    # Configure the LLM
    llm = ChatGroq(
        model=os.environ.get("GROQ_MODEL_OPEN"),
        temperature=0,
    )

    # Connect to the MCP server using the same Python environment
    client = MultiServerMCPClient(
        {
            "product_catalog": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
            }
        }
    )

    # Discover tools from the MCP server
    tools = await client.get_tools()

    print("\nDiscovered MCP tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Create an agent that can choose and call those tools
    
    agent = create_agent(
        llm,
        tools,
        system_prompt="""
            You are a Product Catalog Assistant.
            
            Your responsibility is to answer questions about the products
            available in the connected product catalog.
    
            Rules:
            1. Use the available MCP tools to retrieve product information.
            2. Never invent product names, prices, stock levels, or categories.
            3. All product prices are in Indian Rupees (INR).
            4. If the user asks about product availability, check the stock.
            5. If a product is not found, clearly say it was not found.
            6. If a question is unrelated to the product catalog, politely
               explain that you can only help with product catalog questions.
            7. Keep answers concise and base product claims on tool results.
        """
    )


    print("\nProduct Catalog AI is ready!")
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            response = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                }
            )

            print("\nAI:", response["messages"][-1].content, "\n")

        except Exception as exc:
            print(f"\nRequest failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())

# Find all products under ₹2,000.
# Are any accessories out of stock?
# Is the USB-C Hub available?
# Find products in the Computers category.