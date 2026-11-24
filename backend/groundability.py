import re
from typing import List, Dict
from .config import settings
from .config import get_gemini_llm

# Initialize Gemini LLM
llm = get_gemini_llm()

def check_groundability(requirements: Dict, docs: List[str]) -> List[str]:
    """Inspect the uploaded document contents to see whether each requirement can be supported.
    Returns a list of requirement keys that appear *unsupported*.
    Simple heuristic: if any word from the requirement value is not found in the combined document text,
    we flag it. For a more robust check you could ask the LLM, but this keeps it deterministic.
    """
    unsupported = []
    # Combine all document texts into one blob for quick searching
    combined = " ".join(docs).lower()
    for key, value in requirements.items():
        if isinstance(value, str):
            # tokenise on whitespace and ignore very common stop‑words
            words = [w for w in re.findall(r"\w+", value.lower()) if len(w) > 2]
            if not words:
                continue
            # If less than half of the words appear in the documents, consider unsupported
            match_count = sum(1 for w in words if w in combined)
            if match_count < len(words) / 2:
                unsupported.append(key)
    return unsupported
