import os

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------------------
# 1. Create Groq model
# ----------------------------------------

load_dotenv()

model = ChatGroq(
    model=os.environ.get("GROQ_MODEL")
)


# ----------------------------------------
# 2. Create three prompt versions
# ----------------------------------------

normal_prompt = """
Explain FlatList in React Native.
"""


good_prompt = """
Explain FlatList in React Native.

Explain:
1. What FlatList is
2. Why it is used
3. When we should use it
4. Give a simple example
"""


better_prompt = """
You are explaining React Native concepts to
a developer who already knows the basics of React Native.

Explain FlatList in simple, developer-friendly language.

Cover these topics:

1. What FlatList is
2. Why FlatList is useful
3. FlatList vs ScrollView
4. Important commonly-used props
5. A simple React Native example

Keep the explanation concise and practical.

Avoid advanced concepts that are not necessary
for understanding FlatList.
"""


# ----------------------------------------
# 3. Create ChatPromptTemplates
# ----------------------------------------

prompts = {
    "NORMAL PROMPT": ChatPromptTemplate.from_messages([
        ("human", normal_prompt)
    ]),

    "GOOD PROMPT": ChatPromptTemplate.from_messages([
        ("human", good_prompt)
    ]),

    "BETTER PROMPT": ChatPromptTemplate.from_messages([
        ("human", better_prompt)
    ]),
}


# ----------------------------------------
# 4. Run one prompt
# ----------------------------------------

def run_prompt(name, prompt):

    messages = prompt.format_messages()

    response = model.invoke(messages)

    return name, response


# ----------------------------------------
# 5. Run all prompts in parallel
# ----------------------------------------

with ThreadPoolExecutor(max_workers=3) as executor:

    futures = [
        executor.submit(
            run_prompt,
            name,
            prompt
        )
        for name, prompt in prompts.items()
    ]

    results = [
        future.result()
        for future in futures
    ]


# ----------------------------------------
# 6. Display responses
# ----------------------------------------

for name, response in results:

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(response.content)