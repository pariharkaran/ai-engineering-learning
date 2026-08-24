import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# 1. Create the local model interface
model = ChatOllama(
    model="qwen3:4b",
)


# 2. Create a reusable prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are an Senio React Native mentor.

    Explain {topic} in simple terms.

    Requirements:
    - Use beginner-friendly language.
    - Give one real-world analogy.
    - Include one practical example.
    """
)


# 3. Convert AIMessage into a simple string
parser = StrOutputParser()


# 4. Create LCEL pipeline
chain = prompt | model | parser

input_prompt = input("Ask topics with AI: ")

print("\n--- STREAM() ---\n")

start_time = time.perf_counter()
first_chunk_time = None

for chunk in chain.stream(
    {
        "topic": input_prompt,
    }
):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter()

    print(chunk, end="", flush=True)

end_time = time.perf_counter()

time_to_first_chunk = first_chunk_time - start_time
total_time = end_time - start_time

print("\n")
print(f"Time to first chunk: {time_to_first_chunk:.2f} seconds")
print(f"Total response time: {total_time:.2f} seconds")