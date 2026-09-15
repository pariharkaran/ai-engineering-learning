import os
from dotenv import load_dotenv
import requests

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL_OPEN"),
    temperature=0
)

@tool
def get_country_info(country: str) -> str:
    """Get a country's capital, currency, and timezone."""

    url = f"https://countries.dev/name/{country}"

    response = requests.get(url)
    response.raise_for_status()

    countries = response.json()

    # Find exact country-name match
    country_data = next(
        (
            item
            for item in countries
            if item["name"].lower() == country.lower()
        ),
        None
    )

    if country_data is None:
        raise ValueError(
            f"Could not find exact country: {country}"
        )

    capital = country_data["capital"]

    currency = country_data["currencies"][0]

    currency_code = currency["code"]
    currency_name = currency["name"]

    timezone = country_data["timezones"][0]

    return (
        f"Country: {country_data['name']}\n"
        f"Capital: {capital}\n"
        f"Currency: {currency_name} ({currency_code})\n"
        f"Timezone: {timezone}"
    )

@tool
def get_currency_rate(currency_code: str) -> str:
    """Get the latest exchange rate of a currency against USD.
    Input must be a three-letter ISO currency code such as INR, EUR, or JPY."""

    url = (
        f"https://api.frankfurter.dev/v2/rate/"
        f"{currency_code.upper()}/USD"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    return (
        f"1 {data['base']} = "
        f"{data['rate']} {data['quote']} "
        f"(rate date: {data['date']})"
    )

tools = [
    get_country_info,
    get_currency_rate
]

agent = create_agent(
    model=llm,
    tools=tools
)

input = input("Enter country name: ")

user_query = f"""
    For {input}, tell me:
    1. What is its capital?
    2. What is its currency?
    3. What is the current value of its currency compared with USD?
    """

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": user_query
        }
    ]
})

print(result["messages"][-1].content)