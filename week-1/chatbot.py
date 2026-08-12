import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# if not client.api_key:
#     raise ValueError("GEMINI_API_KEY not found in environment variables.")

conversation = []

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "clear":
        conversation = []
        print("Conversation cleared.")
        continue

    if user_input.lower() in ["exit", "quit"]:
        conversation = []
        print("Exiting the chat. Goodbye!")
        break

    conversation.append({
        "role":"user",
        "parts":[
            {"text": user_input}
        ]
    })

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=conversation
    )

    print(f"\nAI: {response.text}")    

    conversation.append({
        "role":"model",
        "parts":[
            {"text": response.text}
        ]
    })