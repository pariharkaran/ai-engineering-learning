import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Extract API key and configured model name
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Accept DEFAULT_MODEL, fallback to GROQ_MODEL, or use a default model
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or os.getenv("GROQ_MODEL") or "qwen/qwen3.6-27b"

# Model specs: pricing per 1,000,000 tokens (in USD) and context window
MODEL_METRICS = {
    "llama-3.1-8b-instant": {
        "context_window": 131072,
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.08,
    },
    "qwen/qwen3.6-27b": {
        "context_window": 131072,
        "input_cost_per_million": 0.60,
        "output_cost_per_million": 3.00,
    },
    "llama-3.3-70b-versatile": {
        "context_window": 131072,
        "input_cost_per_million": 0.59,
        "output_cost_per_million": 0.79,
    }
}

# Fallback metrics for unknown models
DEFAULT_METRICS = {
    "context_window": 131072,
    "input_cost_per_million": 0.10,
    "output_cost_per_million": 0.10,
}

def get_model_metrics(model_name: str) -> dict:
    """
    Returns a dictionary of metrics for a given model name.
    """
    return MODEL_METRICS.get(model_name, DEFAULT_METRICS)

def validate_config() -> bool:
    """
    Validates that the required API key exists.
    """
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "" or GROQ_API_KEY.startswith("your_"):
        return False
    return True
