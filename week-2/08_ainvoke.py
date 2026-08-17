import os
import asyncio
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv();

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1,
)

prompt = ChatPromptTemplate([
    (
        "system",
        "You are a helpful assistant that explains the meaning "
        "of terms in one sentence."
    ),
    (
        "human",
        "Explain the meaning of {name} in one sentence."
    ),
])

chain = prompt | model | StrOutputParser();

async def ask_ai(prompt):
    response = await chain.ainvoke({"name":prompt})
    return response


async def main():
    # prompt = input("What do you want to explore using ai: ")
    # response = await chain.ainvoke({"name":prompt})    
    start = time.perf_counter()

    results  = await asyncio.gather(
        ask_ai("Karan"),
        ask_ai("React"),
        ask_ai("Nature")
    )

    ellapsed_time = time.perf_counter() - start
    for result in results:
        print(f"\nresponse : \n{result}\n")


if __name__ == "__main__":
    asyncio.run(main())