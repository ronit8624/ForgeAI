import math
from typing import Any, Dict

def calculator(expression: str) -> str:
    """Very simple safe calculator – evaluates arithmetic expressions.
    Uses Python's eval with limited globals.
    """
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names["abs"] = abs
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
