import os
import sys

# Resolve parent directory path to allow 'src' imports from anywhere
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.config import validate_config, DEFAULT_MODEL
from src.utils.display import print_header, print_error, print_info
from src.experiments import basic_call, parameters, chat_history, context, hallucination, cost

def main():
    # Defensive verification of settings
    if not validate_config():
        print_error("Configuration Error: GROQ_API_KEY is missing, empty, or placeholder.")
        print_info("Please open your '.env' file in 'week-1/' and update GROQ_API_KEY with your valid key.")
        print_info("You can copy '.env.example' to '.env' if it does not exist.")
        sys.exit(1)

    while True:
        print("\n" + "=" * 40)
        print("          LLM BEHAVIOR LAB")
        print("=" * 40)
        print("1. Basic LLM Call")
        print("2. Model Parameters")
        print("3. Chat History")
        print("4. Context")
        print("5. Hallucination")
        print("6. Token Usage & Cost")
        print("\n0. Exit")
        print("=" * 40)
        
        try:
            choice = input("\nSelect an option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting. Goodbye!")
            break
            
        if choice == "0":
            print("\nExiting. Goodbye!")
            break
        elif choice == "1":
            try:
                basic_call.run_experiment()
            except Exception as e:
                print_error(f"Experiment 1 failed: {e}")
        elif choice == "2":
            try:
                parameters.run_experiment()
            except Exception as e:
                print_error(f"Experiment 2 failed: {e}")
        elif choice == "3":
            try:
                chat_history.run_experiment()
            except Exception as e:
                print_error(f"Experiment 3 failed: {e}")
        elif choice == "4":
            try:
                context.run_experiment()
            except Exception as e:
                print_error(f"Experiment 4 failed: {e}")
        elif choice == "5":
            try:
                hallucination.run_experiment()
            except Exception as e:
                print_error(f"Experiment 5 failed: {e}")
        elif choice == "6":
            try:
                cost.run_experiment()
            except Exception as e:
                print_error(f"Experiment 6 failed: {e}")
        else:
            print_error("Invalid selection. Please choose an option between 0 and 6.")
            
if __name__ == "__main__":
    main()