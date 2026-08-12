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
    config = types.GenerateContentConfig(
        system_instruction = """
        You are a strict technical interviewer.

        Give concise answers.
        Do not provide the answer immediately.
        Instead, ask the candidate questions that test their understanding.
        """
    )
)

print (response.text)