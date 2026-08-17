import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv();

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1,
    model_kwargs={"top_p":1},
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains the meaning of terms in one sentence."),
    ("human", "Explain Meaning of {name} in one sentence")
])

input = input("Please anything with ai: ")

chain = prompt | model | StrOutputParser()

for chunk in chain.stream({"name":input}):
    print(chunk,end="",flush=True)