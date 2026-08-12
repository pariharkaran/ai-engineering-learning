from src.llm import create_llm
from src.utils.display import (
    print_header, print_concept, print_divider, print_subheader, print_error, print_info
)

def run_experiment():
    """
    Runs Experiment 5: Grounding and Hallucination.
    Demonstrates model response behavior when asked questions outside the provided context.
    """
    print_header("Experiment 5: Grounding & Hallucination")

    # --- EDUCATIONAL CONTENT ---
    print_concept(
        "What is Hallucination?",
        "Hallucination is when an LLM generates text that is grammatically correct and sounds highly plausible, "
        "but is factually incorrect, nonsensical, or unsupported by any input data. "
        "LLMs do not have an internal mechanism for 'truth' or a database to look up facts; they simply predict "
        "the most likely sequence of tokens based on patterns learned during training."
    )

    print_concept(
        "What is Grounding?",
        "Grounding is the process of anchoring the model's responses to verified, external source data (context). "
        "By providing a document in the prompt and instructing the model to *only* answer using that document, "
        "we dramatically reduce the occurrence of hallucinations."
    )

    print_concept(
        "Grounding Limitations",
        "  * Prompting REDUCES hallucinations, but it does NOT eliminate them. Under high temperature or complex queries, "
        "models can still misinterpret context or invent facts.\n"
        "  * Retrieval-Augmented Generation (RAG) is a pattern that automates this by retrieving relevant "
        "documents from database storage (like Vector databases) and pasting them into the context window dynamically. "
        "(RAG will be covered in future lessons)."
    )

    # Fictional context
    fictional_context = (
        "Company: TechNova Corp\n"
        "Product: NovaStream Media Hub\n"
        "Refund Policy:\n"
        "Customers can request a full refund for any NovaStream device within 7 days of delivery. "
        "Requests made after 7 days are strictly non-refundable. All refund devices must be returned "
        "in their original packaging."
    )

    print_divider()
    print(f"Provided Context:\n{fictional_context}")

    # Grounding instructions
    grounding_instructions = (
        "Instructions: You are a customer support agent. Answer the question using ONLY the provided context. "
        "If the answer cannot be found in the context, state: 'I am sorry, but I do not have enough information to answer that question.' "
        "Do not make up any facts or infer policies not explicitly listed."
    )

    # Question 1: Answerable from context
    q1 = "Can I get a refund if I bought the NovaStream 5 days ago?"
    # Question 2: Unanswerable from context
    q2 = "Who pays for the shipping fees when returning the NovaStream device for a refund?"

    llm = create_llm(temperature=0.0)

    # ==========================================
    # TEST 1: Question 1 (Grounded & Answerable)
    # ==========================================
    print_subheader("Test 1: Grounded Question (Answerable from Context)")
    print(f"Question: '{q1}'")
    
    prompt_q1 = f"{grounding_instructions}\n\nContext:\n{fictional_context}\n\nQuestion: {q1}"
    print_info("Querying model...")
    try:
        res_q1 = llm.invoke(prompt_q1)
        clean_content = res_q1.content
        if "<think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        print(f"\nResponse:\n{clean_content}\n")
    except Exception as e:
        print_error(f"Failed Test 1: {e}")

    # ==========================================
    # TEST 2: Question 2 (Ungrounded, With Grounding Instructions)
    # ==========================================
    print_divider()
    print_subheader("Test 2: Ungrounded Question (With Grounding Instructions)")
    print(f"Question: '{q2}'")
    
    prompt_q2_grounded = f"{grounding_instructions}\n\nContext:\n{fictional_context}\n\nQuestion: {q2}"
    print_info("Querying model (instructed to remain grounded)...")
    try:
        res_q2_grounded = llm.invoke(prompt_q2_grounded)
        clean_content = res_q2_grounded.content
        if "<think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        print(f"\nResponse:\n{clean_content}\n")
        print_info("Observation: The model correctly abstains from inventing shipping rules since they are not in the text.")
    except Exception as e:
        print_error(f"Failed Test 2: {e}")

    # ==========================================
    # TEST 3: Question 2 (Ungrounded, WITHOUT Grounding Instructions - Testing Hallucination)
    # ==========================================
    print_divider()
    print_subheader("Test 3: Ungrounded Question (WITHOUT Grounding Instructions)")
    print(f"Question: '{q2}'")
    
    prompt_q2_ungrounded = f"Context:\n{fictional_context}\n\nQuestion: {q2}"
    print_info("Querying model (no safety guidelines provided - high risk of hallucination)...")
    try:
        res_q2_ungrounded = llm.invoke(prompt_q2_ungrounded)
        clean_content = res_q2_ungrounded.content
        if "<think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        print(f"\nResponse:\n{clean_content}\n")
        print_info(
            "Observation: Without negative constraints, models often default to generating a plausible answer "
            "(e.g. inventing a shipping policy or claiming the customer pays), demonstrating a classic hallucination."
        )
    except Exception as e:
        print_error(f"Failed Test 3: {e}")
