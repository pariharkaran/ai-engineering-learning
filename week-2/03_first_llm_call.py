import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1
)

response = model.invoke("Explain Meaning of Karan name in one sentence")

if "</think>" in response.content:
    actual_response = response.content.split("</think>",1)[1].strip()
else :
    actual_response = response.content.strip()

print(f"\nResponse from Groq: \n{response.content}\n")