import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in environment variables.")
    sys.exit(1)

client = Groq(api_key=api_key)
conversation = []

print("--- Groq (qwen3.5 ) Chat Initialized ---")
print("Type 'clear' to reset chat, or 'exit'/'quit' to end session.")

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting the chat. Goodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() == "clear":
        conversation = []
        print("Conversation cleared.")
        continue

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    conversation.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=conversation
        )
        ai_message = response.choices[0].message.content
        print(f"\nAI: {ai_message}")

        conversation.append({"role": "assistant", "content": ai_message})
    except Exception as e:
        print(f"\nError generating response: {e}")