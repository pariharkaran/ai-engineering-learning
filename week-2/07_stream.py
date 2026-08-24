import os
import time

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
    ("system",  """
        You are an Senior React Native mentor.
    
        Explain {topic} in simple terms.
    
        Requirements:
        - Use beginner-friendly language.
        - Give one real-world analogy.
        - Include one practical example.
        """),
    ("human", "Explain Meaning of {topic} in simple terms")
])

input = input("Please anything with ai: ")

chain = prompt | model | StrOutputParser()

print("\n--- STREAM() ---\n")

start_time = time.perf_counter()
first_chunk_time = None

for chunk in chain.stream({"topic":input}):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter()
    print(chunk,end="",flush=True)

end_time = time.perf_counter()

time_to_first_chunk = first_chunk_time - start_time
total_time = end_time - start_time

print("\n")
print(f"Time to first chunk: {time_to_first_chunk:.2f} seconds")
print(f"Total response time: {total_time:.2f} seconds")