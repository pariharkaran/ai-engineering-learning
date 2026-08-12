from src.config import get_model_metrics

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Calculates the estimated cost of an LLM request based on the model's pricing.
    
    Formula:
        Cost = (Tokens / 1,000,000) * Price_Per_Million
        
    Returns a dictionary with details.
    """
    metrics = get_model_metrics(model_name)
    input_price_per_million = metrics.get("input_cost_per_million", 0.10)
    output_price_per_million = metrics.get("output_cost_per_million", 0.10)
    
    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost
    
    return {
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }
