import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[
        {
            "role": "user",
            "content": """
            Explain AI Engineering to a React Native developer.
            Cover LLMs, APIs, prompts, context windows,
            structured output, and AI agents.
            """
        }
    ]
)

print("Response:")
print(response.choices[0].message.content)

print("\nUsage:")
print(response.usage)