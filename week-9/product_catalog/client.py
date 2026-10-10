
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PATH = Path(__file__).parent / "server.py"

server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_PATH)],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP connection
            await session.initialize()
            print("\nConnected to Product Catalog Server")

            # 1. Discover tools dynamically
            tools_result = await session.list_tools()

            print("\n--- Available Tools ---")
            for tool in tools_result.tools:
                print(f"- {tool.name}: {tool.description}")

            # 2. Search products
            print("\n--- Search: accessories ---")
            result = await session.call_tool(
                "search_products",
                {"query": "accessories"},
            )
            print(result.content)

            # 3. Filter products by price
            print("\n--- Products under ₹2,000 ---")
            result = await session.call_tool(
                "filter_products_by_price",
                {"max_price": 2000},
            )
            print(result.content)

            # 4. Check stock
            print("\n--- Stock: USB-C Hub ---")
            result = await session.call_tool(
                "check_product_stock",
                {"product_name": "USB-C Hub"},
            )
            print(result.content)

            # 5. Read the catalog resource
            print("\n--- Complete Product Catalog ---")
            result = await session.read_resource("catalog://products")
            for item in result.contents:
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
