from langchain_groq import ChatGroq
from src.config import GROQ_API_KEY, DEFAULT_MODEL

def create_llm(
    model: str = None,
    temperature: float = None,
    top_p: float = None,
    max_tokens: int = None
) -> ChatGroq:
    """
    Factory function to construct a ChatGroq instance.
    
    Parameters:
        model (str): Name of the Groq model. Defaults to the value in config.
        temperature (float): Controls randomness (0.0 to 1.0/2.0).
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0).
        max_tokens (int): Maximum number of tokens to generate in the output.
    """
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "groq_api_key": GROQ_API_KEY  # Pass api key to the constructor
    }
    
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
        
    return ChatGroq(**kwargs)
