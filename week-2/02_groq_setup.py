import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")

print(f"Groq API Key initialized... {bool(groq_api_key)}")

# ChatGroq can read GROQ_API_KEY from the environment.
model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1
)

print("Groq model initialized...")