import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# 1. Ask the user for their prompt
user_prompt = input("Enter your prompt for the AI: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)

print(f"\nThe answer for your question '{user_prompt}' is: ")
print("\nAI Answer:")
print(response.choices[0].message.content)