import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(
    api_key = os.getenv("GEMINI_API_KEY")
)

prompt = input("You: ")



response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents={
        "role": "user",
        "parts": [
            {"text": prompt}
        ]
    },
   config=types.GenerateContentConfig(
        system_instruction="""
       You are a senior React Native mentor teaching
a developer who is transitioning into AI Engineering.

Explain AI concepts using React Native analogies
whenever possible.

Keep explanations practical.

When appropriate, provide a small code example.

Do not assume knowledge of machine learning mathematics.
        """
    ),
)

print (response.text)