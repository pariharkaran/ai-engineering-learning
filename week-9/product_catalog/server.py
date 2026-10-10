
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Product Catalog Server")

# Load product data from JSON
DATA_FILE = Path(__file__).parent / "products.json"


def load_products() -> list[dict]:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@mcp.resource("catalog://products")
def get_product_catalog() -> str:
    """Return the complete product catalog as JSON."""
    return json.dumps(load_products(), indent=2)


@mcp.tool()
def search_products(query: str) -> list[dict]:
    """Search products by name or category, ignoring letter case."""
    products = load_products()
    query = query.strip().casefold()

    if not query:
        return []

    return [
        product
        for product in products
        if query in product["name"].casefold()
        or query in product["category"].casefold()
    ]


@mcp.tool()
def filter_products_by_price(max_price: float) -> list[dict]:
    """Return products priced at or below the specified maximum price."""
    if max_price < 0:
        raise ValueError("Maximum price cannot be negative.")

    return [
        product
        for product in load_products()
        if product["price"] <= max_price
    ]


@mcp.tool()
def check_product_stock(product_name: str) -> dict:
    """Find a product by exact name and return its stock status."""
    for product in load_products():
        if product["name"].casefold() == product_name.strip().casefold():
            return {
                "id": product["id"],
                "name": product["name"],
                "stock": product["stock"],
                "available": product["stock"] > 0,
            }

    return {
        "error": f"Product '{product_name}' was not found."
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
