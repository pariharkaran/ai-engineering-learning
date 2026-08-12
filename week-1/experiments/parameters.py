from src.llm import create_llm
from src.utils.display import (
    print_header, print_concept, print_metric, print_divider, print_subheader, print_error, print_info,
    YELLOW, RESET, BOLD
)

def run_experiment():
    """
    Runs Experiment 2: Model Parameters.
    Demonstrates the effects of Temperature, Top-P, and Max Output Tokens.
    """
    print_header("Experiment 2: Model Parameters (Sampling & Limits)")

    # ==========================================
    # 1. TEMPERATURE EXPERIMENT
    # ==========================================
    print_subheader("1. Temperature Experiment")
    
    print_concept(
        "What is Temperature?",
        "Temperature controls the 'randomness' or 'creativity' of the model's outputs. "
        "Technically, it scales the logits (the raw scores calculated for each possible next token) before "
        "applying a softmax function to turn them into probabilities:\n"
        "  - Scale = Logits / Temperature\n"
        "  * A temperature of 0.0 makes the model fully deterministic (it always selects the token with the absolute highest probability, known as greedy search).\n"
        "  * A low temperature (e.g., 0.2) compresses probabilities, making the model stay conservative and predictable.\n"
        "  * A high temperature (e.g., 1.0+) flattens probabilities, making less common tokens much more likely to be selected."
    )
    
    print_concept(
        "Important Caveats",
        "  * Temperature is NOT 'intelligence'. Higher temperature does not make the model smarter; it just makes it more random.\n"
        "  * Temperature does NOT guarantee correctness. It actually increases the likelihood of factually incorrect outputs."
    )

    creative_prompt = "Write a short creative metaphor (under 15 words) about what debugging code feels like."
    print(f"Prompt: '{creative_prompt}'\n")

    temperatures = [0.0, 0.5, 1.0]
    for temp in temperatures:
        print_info(f"Running query with Temperature = {temp}...")
        try:
            llm = create_llm(temperature=temp)
            res = llm.invoke(creative_prompt)
            # Remove Qwen reasoning tags for cleaner reading if present
            clean_content = res.content
            if "<think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            print(f"  {BOLD}Temperature {temp}:{RESET} {clean_content}")
        except Exception as e:
            print_error(f"Failed temp={temp}: {e}")
            
    print_divider()

    # ==========================================
    # 2. TOP-P EXPERIMENT
    # ==========================================
    print_subheader("2. Top-P (Nucleus) Sampling Experiment")
    
    print_concept(
        "What is Top-P?",
        "Top-P (or Nucleus Sampling) limits token selection to a subset of vocabulary whose *cumulative probability* "
        "reaches the threshold P. For example, if top_p = 0.1, the model only considers the top candidate tokens "
        "that make up the first 10% of the probability distribution. The remaining 90% are completely discarded.\n"
        "  * Difference from Temperature: Temperature scales *all* probabilities. Top-P *chops off* the long tail of low-probability words.\n"
        "  * Why change one at a time? Changing both simultaneously makes it impossible to trace which parameter caused a change in behavior."
    )

    # Use a fixed temperature (e.g. 1.0) and vary top-p
    fixed_temp = 1.0
    top_p_values = [0.1, 0.5, 0.9]
    
    print(f"Temperature: Fixed at {fixed_temp}")
    print(f"Prompt: '{creative_prompt}'\n")

    for tp in top_p_values:
        print_info(f"Running query with Temperature = {fixed_temp}, Top-P = {tp}...")
        try:
            llm = create_llm(temperature=fixed_temp, top_p=tp)
            res = llm.invoke(creative_prompt)
            clean_content = res.content
            if "<think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            print(f"  {BOLD}Top-P {tp}:{RESET} {clean_content}")
        except Exception as e:
            print_error(f"Failed top_p={tp}: {e}")

    print_divider()

    # ==========================================
    # 3. MAXIMUM OUTPUT TOKENS EXPERIMENT
    # ==========================================
    print_subheader("3. Maximum Output Tokens Experiment")
    
    print_concept(
        "What is Maximum Output Tokens?",
        "This parameter tells the model/API host to halt generation once it reaches the specified number of output tokens. "
        "  * Output Limit ≠ Context Window. The output limit is a cap on the *response size* only, whereas the context window restricts the *total tokens* (input + output).\n"
        "  * Stop Conditions: The model will stop either because it hit the max output limit (abrupt truncation) OR because it generated a natural End-Of-Sequence (EOS) token (clean completion before reaching the limit)."
    )

    long_prompt = "Write a detailed explanation of why the sky is blue, covering Rayleigh scattering. Write at least two paragraphs."
    print(f"Prompt: '{long_prompt}'\n")

    token_limits = [15, 60, 200]
    for limit in token_limits:
        print_info(f"Running query with Max Output Tokens limit = {limit}...")
        try:
            llm = create_llm(temperature=0.0, max_tokens=limit)
            res = llm.invoke(long_prompt)
            
            # Extract actual output tokens
            usage = getattr(res, "usage_metadata", {})
            actual_out = usage.get("output_tokens", "N/A") if usage else "N/A"
            
            clean_content = res.content.strip()
            # If it contains think tags, we might want to check length of text without thoughts or show truncation
            # We will show the raw snippet so the user sees where it stops, thoughts included or excluded.
            has_thoughts = "<think>" in clean_content
            
            print(f"\n  {BOLD}[Configured Limit: {limit} tokens | Actual Output: {actual_out} tokens]{RESET}")
            # Show snippet
            snippet = clean_content[:400] + ("..." if len(clean_content) > 400 else "")
            print(f"  Response Preview:\n  ---\n  {snippet}\n  ---")
            
            # Print stop reason if present in response_metadata
            resp_meta = getattr(res, "response_metadata", {})
            finish_reason = resp_meta.get("finish_reason", "unknown")
            print(f"  Finish Reason: {YELLOW}{finish_reason}{RESET}")
            
        except Exception as e:
            print_error(f"Failed limit={limit}: {e}")

