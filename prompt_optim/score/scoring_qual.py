from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
from .llm_client import LLMClient
from .utils import safe_json_loads

PAIRWISE_PROMPT = """You are an expert evaluator for long-form text summarization.

You will be given:
- SOURCE: the original text
- ANSWER_A: a summary
- ANSWER_B: another summary

Evaluate which answer is better for a busy reader, using these criteria:
1) Faithfulness: no added facts, aligns with SOURCE
2) Coverage: captures the important points
3) Structure/clarity: well organized, easy to scan
4) Usefulness: actionable takeaways if applicable
5) Concision: not overly verbose

Return ONLY valid JSON:
{
  "score_A": 0-100,
  "score_B": 0-100,
  "winner": "A" or "B" or "tie",
  "rationale": "short"
}

Important:
- Do not reward verbosity.
- Base judgment primarily on faithfulness and coverage.
"""

def quality_pairwise_score(
    client: LLMClient,
    source: str,
    cand: str,
    baseline: str,
    judge_model: str,
    max_tokens: int
) -> Tuple[int, Dict[str, Any]]:
    """Return candidate quality score (0-100) vs baseline, debiased by swapping order twice."""

    def one_pass(a: str, b: str) -> Tuple[Optional[tuple], Dict[str, Any]]:
        messages = [
            {"role": "system", "content": PAIRWISE_PROMPT},
            {"role": "user", "content": f"SOURCE:\n{source}\n\nANSWER_A:\n{a}\n\nANSWER_B:\n{b}"}
        ]
        raw = client.chat(messages=messages, model=judge_model, temperature=0.0, max_tokens=max_tokens)
        obj, err = safe_json_loads(raw)
        pair = None
        if obj and isinstance(obj, dict) and "score_A" in obj and "score_B" in obj:
            try:
                sa = int(obj["score_A"])
                sb = int(obj["score_B"])
                pair = (sa, sb)
            except Exception:
                pair = None
        return pair, {"raw": raw, "parsed": obj, "error": err}

    # pass1: candidate is A
    s1, d1 = one_pass(cand, baseline)
    # pass2: swapped
    s2, d2 = one_pass(baseline, cand)

    details: Dict[str, Any] = {"pass1": d1, "pass2": d2}
    scores = []

    if s1:
        scores.append(s1[0])
    if s2:
        scores.append(s2[1])

    if not scores:
        return 55, details

    return int(round(sum(scores) / len(scores))), details
