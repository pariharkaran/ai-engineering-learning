# Week 2: Deep Dive into LangChain & Groq

Welcome to **Week 2**! This guide breaks down the core concepts of building applications with LangChain and Groq. Each topic is explained in simple terms with an illustrative code example.

---

## Table of Contents

1. [LangChain Setup](#1-langchain-setup)
2. [Groq API Setup](#2-groq-api-setup)
3. [First LLM Call](#3-first-llm-call)
4. [Prompt Template](#4-prompt-template)
5. [Chat Prompt Template](#5-chat-prompt-template)
6. [LCEL Chain (LangChain Expression Language)](#6-lcel-chain-langchain-expression-language)
7. [Streaming Responses](#7-streaming-responses)
8. [Asynchronous Invoke (Concurrency)](#8-asynchronous-invoke-concurrency)
9. [Asynchronous Streaming](#9-asynchronous-streaming)
10. [Structured Output with Pydantic](#10-structured-output-with-pydantic)

---

### 1. LangChain Setup

- **File reference:** [01_langchain_setup.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/01_langchain_setup.py)

#### 💡 Explanation

Imagine LangChain is a giant toolbox full of specialized tools for building AI assistants. Before you start building, you need to make sure the toolbox has arrived and is open on your desk. This script simply checks if the toolbox is ready to use and prints its version number.

#### 💻 Code Example

```python
import langchain

print("LangChain is working!")
print("LangChain version:", langchain.__version__)
```

---

### 2. Groq API Setup

- **File reference:** [02_groq_setup.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/02_groq_setup.py)

#### 💡 Explanation

To talk to a smart AI brain (like the models hosted on Groq), you need a secret password (an **API Key**) to authenticate yourself. You also need to choose which specific AI brain (model) you want to talk to. Think of it like calling a company's hotline: the API Key is your account ID, and the model name is the specific agent you asked to speak with.

#### 💻 Code Example

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from a .env file
load_dotenv()

# Initialize the Groq model
model = ChatGroq(
    model=os.environ.get("GROQ_MODEL"),
    temperature=1
)
print("Groq model initialized successfully!")
```

---

### 3. First LLM Call

- **File reference:** [03_first_llm_call.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/03_first_llm_call.py)

#### 💡 Explanation

This is your first simple conversation with the AI. You send a direct message or question using the `invoke` method, and the AI replies. It is like sending a text message to a smart friend and waiting for their answer.

#### 💻 Code Example

```python
response = model.invoke("Explain Meaning of Karan name in one sentence")
print("AI Response:", response.content)
```

---

### 4. Prompt Template

- **File reference:** [04_prompt_template.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/04_prompt_template.py)

#### 💡 Explanation

Instead of rewriting a long instruction from scratch every time, you can create a fill-in-the-blank form (a template). For instance, `"Explain the meaning of {name} in one sentence."` You just provide the name, and the template formats the request for you automatically.

#### 💻 Code Example

```python
from langchain_core.prompts import ChatPromptTemplate

# Create the blank form
prompt = ChatPromptTemplate.from_template("Explain Meaning of {name} in one sentence")

# Fill in the blank
formatted_prompt = prompt.invoke({"name": "Karan"})

# Send the filled form to the model
response = model.invoke(formatted_prompt)
print(response.content)
```

---

### 5. Chat Prompt Template

- **File reference:** [05_chat_prompt_template.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/05_chat_prompt_template.py)

#### 💡 Explanation

AI models understand conversations better when messages are separated by roles. We specify:

1. **System**: The persona or ground rules for the AI (e.g., _"You are a helpful dictionary assistant"_).
2. **Human**: The actual query or task from the user.

It is like instructing an actor about their character before they deliver their performance.

#### 💻 Code Example

```python
from langchain_core.prompts import ChatPromptTemplate

# Define system rules and human input structure
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains the meaning of terms in one sentence."),
    ("human", "Explain Meaning of {name} in one sentence")
])

formatted_prompt = prompt.invoke({"name": "React"})
response = model.invoke(formatted_prompt)
print(response.content)
```

---

### 6. LCEL Chain (LangChain Expression Language)

- **File reference:** [06_lcel_chain.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/06_lcel_chain.py)

#### 💡 Explanation

Instead of manually calling the template first, passing the result to the model, and then printing the raw result, we can build an **assembly line** (or chain) using the `|` (pipe) operator.

1. The **input** goes into the **Prompt Template**.
2. The template outputs a prompt into the **Model**.
3. The model outputs raw data into the **Output Parser**, which extracts only the clean text response.

#### 💻 Code Example

```python
from langchain_core.output_parsers import StrOutputParser

# Build the assembly line
chain = prompt | model | StrOutputParser()

# Invoke the entire chain at once
response = chain.invoke({"name": "React"})
print(response)
```

---

### 7. Streaming Responses

- **File reference:** [07_stream.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/07_stream.py)

#### 💡 Explanation

Usually, the AI waits to generate the entire answer before sending it back. With **Streaming**, the AI sends the words to you one by one as they are being thought of, similar to how ChatGPT types out its answers live or how captions display letter-by-letter on a screen.

#### 💻 Code Example

```python
chain = prompt | model | StrOutputParser()

# Stream response chunk-by-chunk
for chunk in chain.stream({"name": "Nature"}):
    print(chunk, end="", flush=True)
```

---

### 8. Asynchronous Invoke (Concurrency)

- **File reference:** [08_ainvoke.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/08_ainvoke.py)

#### 💡 Explanation

If you have to ask the AI three different questions, waiting for each one to finish before starting the next takes a lot of time. Asynchronous invocation (`ainvoke`) allows you to send all three questions at the same time and collect their answers when they are done, just like sending three separate emails simultaneously instead of waiting to receive a response to the first before writing the second.

#### 💻 Code Example

```python
import asyncio
import time

async def ask_ai(name):
    # Asynchronously invoke the chain
    return await chain.ainvoke({"name": name})

async def main():
    start = time.perf_counter()
    # Query all three terms at the same time
    results = await asyncio.gather(
        ask_ai("Karan"),
        ask_ai("React"),
        ask_ai("Nature")
    )
    print(f"Done in {time.perf_counter() - start:.2f} seconds!")
    for result in results:
        print("-", result)

# Run the async loop
asyncio.run(main())
```

---

### 9. Asynchronous Streaming

- **File reference:** [09_astream.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/09_astream.py)

#### 💡 Explanation

This combines streaming (getting words one by one as they generate) with asynchronous programming (allowing other tasks to run in the background without freezing the application while waiting for the next word). It ensures your application stays extremely responsive.

#### 💻 Code Example

```python
async def main():
    # Asynchronously stream the output
    async for chunk in chain.astream({"name": "Python"}):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

### 10. Structured Output with Pydantic

- **File reference:** [10_structured_output.py](file:///Users/varshid-innvonix/Documents/Karan/AI-Engineer/ai-engineer-learning/practical/week-2/10_structured_output.py)

#### 💡 Explanation

Normally, AI outputs free-form text, which can be hard for a computer program to read consistently. Structured output forces the AI to fill out a strict digital questionnaire (a data schema, using `Pydantic`). For example, instead of a random paragraph, the AI must return a structured object containing specific fields: `success` (True/False), `name`, `category`, and `difficulty`.

#### 💻 Code Example

```python
from pydantic import BaseModel

# Define the structure of the data you want
class Technology(BaseModel):
    success: bool
    name: str | None = None
    category: str | None = None
    difficulty: str | None = None
    message: str

# Instruct the model to conform to this structure
structured_model = model.with_structured_output(Technology)

chain = prompt_template | structured_model
response = chain.invoke({"user_prompt": "FastAPI"})

print("Structured Response:")
print(f"Success: {response.success}")
print(f"Name: {response.name}")
print(f"Category: {response.category}")
print(f"Difficulty: {response.difficulty}")
print(f"Message: {response.message}")
```
