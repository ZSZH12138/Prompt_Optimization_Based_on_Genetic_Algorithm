from __future__ import annotations
from typing import Dict, Any, Tuple
from .llm_client import LLMClient
from .utils import safe_json_loads

FACT_PROMPT = """You are a strict factuality checker for summarization.

You will be given:
- SOURCE: the original text
- SUMMARY: a model-generated summary

Task:
1) Extract up to {max_claims} ATOMIC factual claims from SUMMARY that are verifiable from SOURCE.
   - A claim should be a single checkable statement.
   - Do NOT include opinions, generic advice, or tautologies.
2) For each claim, decide whether it is DIRECTLY SUPPORTED by SOURCE.
   - supported=true only if SOURCE explicitly supports it.
   - If unsupported or not mentioned, supported=false.
3) Provide a short evidence snippet (<=20 words) from SOURCE if supported, else empty.

Return ONLY valid JSON with this schema:
{{
  "claims": [
    {{"claim": "...", "supported": true/false, "evidence": "..."}}
  ],
  "notes": "..."
}}
"""

def factuality_score(
    client: LLMClient,
    source: str,
    summary: str,
    judge_model: str,
    max_claims: int,
    max_tokens: int
) -> Tuple[int, Dict[str, Any]]:
    messages = [
        {"role": "system", "content": FACT_PROMPT.format(max_claims=max_claims)},
        {"role": "user", "content": f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}"}
    ]
    raw = client.chat(messages=messages, model=judge_model, temperature=0.0, max_tokens=max_tokens)
    obj, err = safe_json_loads(raw)
    details: Dict[str, Any] = {"raw": raw, "error": err, "parsed": obj}

    if not obj or not isinstance(obj, dict) or "claims" not in obj:
        return 40, details

    claims = obj.get("claims", [])
    if not claims:
        return 55, details

    supported = 0
    total = 0
    for c in claims:
        if isinstance(c, dict) and "supported" in c:
            total += 1
            if bool(c["supported"]):
                supported += 1

    if total == 0:
        return 55, details

    ratio = supported / total
    score = int(round(ratio * 100))
    return score, details
