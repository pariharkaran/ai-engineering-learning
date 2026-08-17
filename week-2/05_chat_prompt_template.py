import os 

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains the meaning of terms in one sentence."),
    ("human", "Explain Meaning of {name} in one sentence")
])

input = input("Enter a term to understand via ai: ")

formatted_prompt = prompt.invoke({"name": input})

print(f"formatted_prompt : \n{formatted_prompt}\n")

response = model.invoke(formatted_prompt)

print(f"\nresponse : \n{response.content}\n")