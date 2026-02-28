from __future__ import annotations
import time
from typing import Dict, List, Optional
from openai import OpenAI

class LLMClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 120):
        if not api_key:
            raise ValueError("LLM_API_KEY is empty. Please set it in your .env file.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        retries: int = 3,
        retry_sleep: float = 2.0
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(retry_sleep * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")
