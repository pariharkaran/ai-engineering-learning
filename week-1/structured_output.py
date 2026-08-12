import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

text = input("Enter your expense: ")

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[
        {
            "role": "system",
            "content": """
            You are an expense extraction system.

            Extract expense information from the user's text.

            Return only the requested structured data.
            Do not invent information that is not present.
            """
        },
        {
            "role": "user",
            "content": text
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "expense",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number"
                    },
                    "currency": {
                        "type": "string"
                    },
                    "merchant": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "date": {
                        "type": "string"
                    }
                },
                "required": [
                    "amount",
                    "currency",
                    "merchant",
                    "category",
                    "description",
                    "date"
                ],
                "additionalProperties": False
            }
        }
    }
)

expense = json.loads(
    response.choices[0].message.content
)

print("\nExtracted Expense:")
print(json.dumps(expense, indent=2))