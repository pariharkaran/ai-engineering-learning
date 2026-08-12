import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=api_key)

prompt = input("You: ")

response = client.models.generate_content(
     model="gemini-3.5-flash",
contents=prompt
)

print(response.text)