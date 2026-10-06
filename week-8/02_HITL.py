import os
from dotenv import load_dotenv

from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


load_dotenv()

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL_OPEN"),
    temperature=0,
)

class TranslationState(TypedDict):
    word: str
    meaning: str
    approved: bool
    hindi: str
    gujarati: str
    message: str

def human_approval(state: TranslationState):
    approved = interrupt({
        "question": "Is this meaning correct?",
        "word": state["word"],
        "meaning": state["meaning"]
    })

    return {
        "approved": approved,
        "message": ""
    }

def route_after_approval(state: TranslationState):
    if state["approved"]:
        return "translate_to_hindi"

    return "aborted"

def user_aborted(state: TranslationState):
    return {
        "message": "User aborted the translation."
    }

def analyze_word(state: TranslationState):
    word = state["word"]

    prompt = f"""
        You are a language expert.
        Analyze the English word:
        "{word}"
        Identify its most common meaning in everyday usage.
        Return only a short explanation of the meaning,
        in no more than 30 words.
        """

    response = llm.invoke(prompt)
    return {
        "meaning": response.content
    }

def translate_to_hindi(state: TranslationState):
    word = state["word"]
    meaning = state["meaning"]

    prompt = f"""
        Translate the English word "{word}" into Hindi.
        Context/meaning:
        {meaning}
        Return only the Hindi translation.
        Do not provide explanations.
        """

    response = llm.invoke(prompt)
    return {
        "hindi": response.content.strip()
    }

def translate_to_gujarati(state: TranslationState):
    word = state["word"]
    meaning = state["meaning"]

    prompt = f"""
        Translate the English word "{word}" into Gujarati.
        Context/meaning:
        {meaning}
        Return only the Gujarati translation.
        Do not provide explanations.
        """

    response = llm.invoke(prompt)
    return {
        "gujarati": response.content.strip()
    }

builder = StateGraph(TranslationState)

builder.add_node("analyze_word", analyze_word)
builder.add_node("human_approval", human_approval)
builder.add_node("translate_to_hindi", translate_to_hindi)
builder.add_node("translate_to_gujarati", translate_to_gujarati)
builder.add_node("user_aborted", user_aborted)

builder.add_edge(START, "analyze_word")
builder.add_edge("analyze_word", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "translate_to_hindi": "translate_to_hindi",
        "aborted": "user_aborted"
    }
)
builder.add_edge("translate_to_hindi","translate_to_gujarati")
builder.add_edge("translate_to_gujarati",END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)

config = {
    "configurable": {
        "thread_id": "translation-hitl-002"
    }
}

user_input = input("Enter an English word to translate into Hindi and Gujarati: ")

result = graph.invoke(
    {
        "word": user_input,
        "meaning": "",
        "approved": False,
        "hindi": "",
        "gujarati": ""
    },
    config=config
)

# Check whether the graph is waiting for human input
if "__interrupt__" in result:

    interrupt_data = result["__interrupt__"][0]

    print("\n--- Human Approval Required ---")
    print(f"Word: {interrupt_data.value['word']}")
    print(f"Meaning: {interrupt_data.value['meaning']}")

    approval_input = input("\nApprove meaning? (y/n): ").strip().lower()

    if approval_input == "y":

        # Resume graph with human approval
        result = graph.invoke(
            Command(resume=True),
            config=config
        )

    else:

        # Resume graph with rejection
        result = graph.invoke(
            Command(resume=False),
            config=config
        )

print("\n--- Final Result ---")
print("\nEnglish:", result["word"])
print("Meaning:", result["meaning"])
print("Hindi:", result["hindi"])
print("Gujarati:", result["gujarati"])
print("Message:", result["message"])