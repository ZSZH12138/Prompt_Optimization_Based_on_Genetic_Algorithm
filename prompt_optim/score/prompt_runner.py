from __future__ import annotations
from typing import Dict

def fill_prompt(raw_prompt: str, case: Dict) -> str:
    filled = raw_prompt
    filled+=f"\ntext:{case.get('text','')}"
    return filled

def build_messages(filled_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
        {"role": "user", "content": filled_prompt}
    ]