from src.llm import create_llm
from src.config import DEFAULT_MODEL, get_model_metrics
from src.utils.display import (
    print_header, print_concept, print_metric, print_divider, print_subheader, print_error, print_info
)

def run_experiment():
    """
    Runs Experiment 4: Context & Token Limits.
    Demonstrates context usage scaling and explains context windows.
    """
    print_header("Experiment 4: Context Windows & Usage Tracking")

    # Fetch model details from config
    metrics = get_model_metrics(DEFAULT_MODEL)
    context_window = metrics.get("context_window", 131072)

    # --- EDUCATIONAL CONTENT ---
    print_concept(
        "What is a Context Window?",
        f"The Context Window is the maximum boundary of tokens that the model can process in a single request/response cycle. "
        f"For your active model ({DEFAULT_MODEL}), the context window limit is {context_window:,} tokens.\n\n"
        f"  * Context Window: The maximum container capacity.\n"
        f"  * Actual Context Usage: The actual amount of liquid (tokens) you put inside the container for a specific request."
    )

    print_concept(
        "Key Equation of Context Usage",
        "Your total token usage is calculated as:\n"
        "  Total Usage = System Instructions + User Prompt + Conversation History + Provided Context + AI Output Response\n\n"
        "If the sum of these elements exceeds the model's Context Window, the request will fail with a context limit error, "
        "or the history will be truncated, leading to lost memory or chopped answers."
    )

    print_concept(
        "Context Window ≠ Context Usage",
        "It is a common misconception that all models have the same context window, or that querying a model always "
        "consumes its entire context window. In reality, you only pay for/consume the tokens actually processed."
    )

    # Controlled context blocks of varying sizes
    small_context = (
        "Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy. "
        "It is by far the weakest of the four fundamental interactions, approximately 10^38 times weaker than the strong interaction. "
        "Yet it is the most significant interaction at the macroscopic scale, determining the motion of planets, stars, galaxies, and even light."
    )

    medium_context = (
        small_context + "\n\n"
        "On Earth, gravity gives weight to physical objects, and the Moon's gravity causes the ocean tides. "
        "The gravitational attraction of the original gaseous matter in the universe caused it to begin coalescing and forming stars "
        "and caused the stars to group together into galaxies, so gravity is responsible for many of the large-scale structures in the Universe. "
        "Gravity has an infinite range, although its effects become weaker as objects get further away.\n\n"
        "Gravity is most accurately described by the general theory of relativity (proposed by Albert Einstein in 1915), which describes "
        "gravity not as a force, but as a consequence of the curvature of spacetime caused by the uneven distribution of mass."
    )

    large_context = (
        medium_context + "\n\n"
        "For most applications, gravity is well approximated by Newton's law of universal gravitation, which describes gravity as a force "
        "causing any two bodies to be attracted to each other, with force proportional to the product of their masses and inversely proportional "
        "to the square of the distance between them. The acceleration due to gravity on Earth's surface, denoted as g, is approximately "
        "9.8 meters per second squared (32 ft/s^2). This means that, ignoring air resistance, the speed of an object falling freely near the "
        "Earth's surface will increase by about 9.8 meters per second every second.\n\n"
        "Historically, the investigation of gravity began in ancient times. Philosophers like Aristotle believed that objects fell because "
        "they traveled toward their natural place. Later, Galileo Galilei determined that all objects accelerate at the same rate when falling. "
        "Sir Isaac Newton then unified these observations with planetary motions in his Principia, proposing that gravity is a universal force."
    )

    # Fixed question to ask the model using the contexts
    question = "Summarize the key point of the text in one sentence."

    contexts = [
        ("Small Context (~50 words)", small_context),
        ("Medium Context (~150 words)", medium_context),
        ("Larger Context (~300 words)", large_context),
    ]

    llm = create_llm(temperature=0.0)

    for label, ctx in contexts:
        print_divider()
        print_subheader(f"Testing with: {label}")
        
        # Build prompt using context
        prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
        print_info(f"Length of text sent (chars): {len(prompt)}")
        print_info("Sending query...")

        try:
            res = llm.invoke(prompt)
            clean_content = res.content
            if "<think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()

            print(f"\nResponse:\n{clean_content}\n")

            # Extract token details
            usage = getattr(res, "usage_metadata", {})
            if usage:
                print_metric("Input Tokens (Prompt + Context)", usage.get("input_tokens", "N/A"))
                print_metric("Output Tokens (Response)", usage.get("output_tokens", "N/A"))
                print_metric("Total Tokens", usage.get("total_tokens", "N/A"))
                
                # Check percent of context window used
                input_t = usage.get("input_tokens", 0)
                pct_used = (input_t / context_window) * 100
                print_metric("Percentage of Context Window Used", f"{pct_used:.5f}", "%")
            else:
                print_info("Token usage metadata not returned by API.")
        except Exception as e:
            print_error(f"Failed to query model with {label}: {e}")
