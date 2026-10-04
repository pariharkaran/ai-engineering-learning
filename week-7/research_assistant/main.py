import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

import requests

load_dotenv()

llm = ChatGroq(
    model=os.environ.get("GROQ_MODEL_OPEN"),
    temperature=0
)

# --------------------------------------------------
# 1. State
# --------------------------------------------------

class ResearchState(TypedDict):
    question: str
    plan: str
    web_research: str
    mcp_research: str
    research: str
    summary: str
    final_answer: str


# --------------------------------------------------
# 2. Planner Agent
# --------------------------------------------------

def planner_node(state: ResearchState):
    print("\n🧠 Planner Agent")

    question = state["question"]

    prompt = f"""
        You are a research planning agent.

        Create a short research plan for the user's question.

        User question:
        {question}

        Return a numbered list of 3 to 5 steps.

        Do not answer the question.
        Only create the research plan.
        """

    response = llm.invoke(prompt)

    plan = response.content

    print("\nGenerated Plan:")
    print(plan)

    return {
        "plan": plan
    }


def search_wikipedia(query: str) -> str:
    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AI-Engineering-ResearchAssistant/1.0"
    }

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 3,
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("query", {}).get("search", [])

    if not results:
        return "No Wikipedia results found."

    research = []

    for item in results:
        title = item["title"]
        snippet = item["snippet"]

        research.append(
            f"Title: {title}\n"
            f"Information: {snippet}"
        )

    return "\n\n".join(research)

# --------------------------------------------------
# 3. Retrieval Agent
# --------------------------------------------------

def web_retrieval_node(state: ResearchState):
    print("\n🔎 Retrieval Agent")

    question = state["question"]
    plan = state["plan"]

    print("\nResearch plan:")
    print(plan)

    try:
        research = search_wikipedia(question)

    except Exception as e:
        print(f"Retrieval error: {e}")

        research = "Unable to retrieve information."

    return {
        "web_research": research
    }


def mcp_retrieval_node(state: ResearchState):
    print("\n🔌 MCP Retrieval Agent")

    question = state["question"]

    # Temporary placeholder.
    # We will connect the real MCP server in the next step.

    research = (
        f"MCP research placeholder for: {question}\n"
        "This branch will later retrieve information "
        "using an MCP server."
    )

    return {
        "mcp_research": research
    }

def combine_research_node(state: ResearchState):
    print("\n🔀 Combining Research")

    web_research = state["web_research"]
    mcp_research = state["mcp_research"]

    combined_research = (
        "=== Web Research ===\n"
        f"{web_research}\n\n"
        "=== MCP Research ===\n"
        f"{mcp_research}"
    )

    return {
        "research": combined_research
    }

# --------------------------------------------------
# 4. Summarizer Agent
# --------------------------------------------------

def summarizer_node(state: ResearchState):
    print("\n📝 Summarizer Agent")

    question = state["question"]
    research = state["research"]

    prompt = f"""
        You are a research summarization agent.

        Summarize the retrieved research information for the
        user's question.

        User question:
        {question}

        Retrieved research:
        {research}

        Instructions:
        - Keep only information relevant to the question.
        - Do not invent facts.
        - Clearly identify important points.
        - Keep the summary concise.
        """

    response = llm.invoke(prompt)

    summary = response.content

    print("\nGenerated Summary:")
    print(summary)

    return {
        "summary": summary
    }


# --------------------------------------------------
# 5. Final Answer Agent
# --------------------------------------------------

def final_answer_node(state: ResearchState):
    print("\n💬 Final Answer Agent")

    question = state["question"]
    plan = state["plan"]
    summary = state["summary"]

    prompt = f"""
        You are the final answer agent in a research assistant.

        Answer the user's question using the research summary.

        User question:
        {question}

        Research plan:
        {plan}

        Research summary:
        {summary}

        Instructions:
        - Directly answer the user's question.
        - Be clear and concise.
        - Do not invent information.
        - If the research is insufficient, say so.
        """

    response = llm.invoke(prompt)

    final_answer = response.content

    return {
        "final_answer": final_answer
    }

# --------------------------------------------------
# 6. Build Graph
# --------------------------------------------------

graph = StateGraph(ResearchState)

graph.add_node("planner", planner_node)
graph.add_node("web_retrieval",web_retrieval_node)
graph.add_node("mcp_retrieval",mcp_retrieval_node)
graph.add_node("combine_research",combine_research_node)
graph.add_node("summarizer",summarizer_node)
graph.add_node("final_answer",final_answer_node)

# --------------------------------------------------
# 7. Define Workflow
# --------------------------------------------------

graph.add_edge(START, "planner")

# Fan-out
graph.add_edge("planner", "web_retrieval")
graph.add_edge("planner", "mcp_retrieval")

# Fan-in
graph.add_edge("web_retrieval", "combine_research")
graph.add_edge("mcp_retrieval", "combine_research")

# Continue workflow
graph.add_edge("combine_research", "summarizer")
graph.add_edge("summarizer", "final_answer")
graph.add_edge("final_answer", END)


# --------------------------------------------------
# 8. Compile
# --------------------------------------------------

app = graph.compile()

# --------------------------------------------------
# 9. Run Application
# --------------------------------------------------

if __name__ == "__main__":

    question = input("\nEnter your research question: ")

    result = app.invoke({
        "question": question,
        "plan": "",
        "web_research": "",
        "mcp_research": "",
        "summary": "",
        "final_answer": ""
    })

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(result["final_answer"])