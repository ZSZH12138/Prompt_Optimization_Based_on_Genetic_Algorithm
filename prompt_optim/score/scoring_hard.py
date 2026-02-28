from __future__ import annotations
import re
from typing import Dict, Tuple, Any
from .utils import safe_json_loads

def infer_constraints(prompt: str) -> Dict[str, Any]:
    p = prompt.lower()
    constraints = {
        "require_json": False,
        "require_mermaid": False,
        "require_markdown": False,
        "require_numbered_1_to_7": False,
    }

    # JSON signals
    if ("json" in p and ("only" in p or "format" in p)) or "following json" in p:
        constraints["require_json"] = True

    # Mermaid signals
    if "mermaid" in p or "```mermaid" in prompt:
        constraints["require_mermaid"] = True

    # Markdown signals
    if "markdown" in p:
        constraints["require_markdown"] = True

    # Detect explicit numbered 1..7 instructions
    nums = re.findall(r'^\s*(\d+)\.\s', prompt, flags=re.MULTILINE)
    if nums:
        uniq = sorted(set(int(x) for x in nums))
        if len(uniq) >= 7 and uniq[:7] == [1,2,3,4,5,6,7]:
            constraints["require_numbered_1_to_7"] = True

    return constraints

def hard_score(prompt: str, answer: str) -> Tuple[int, bool, Dict[str, Any]]:
    """Returns: (score 0-100, critical_fail, details)."""
    constraints = infer_constraints(prompt)
    details: Dict[str, Any] = {"constraints": constraints, "checks": {}}
    score = 100
    critical_fail = False

    if constraints["require_json"]:
        obj, err = safe_json_loads(answer.strip())
        ok = obj is not None and isinstance(obj, dict)
        details["checks"]["json_parse_ok"] = ok
        if not ok:
            details["checks"]["json_error"] = err
            critical_fail = True
            return 0, True, details

    if constraints["require_mermaid"]:
        ok = "```mermaid" in answer
        details["checks"]["mermaid_block_ok"] = ok
        if not ok:
            score -= 35

    if constraints["require_markdown"]:
        lines = answer.splitlines()
        md_ok = any(re.match(r'^\s*#{1,6}\s+\S', ln) for ln in lines) or \
                any(re.match(r'^\s*[-*]\s+\S', ln) for ln in lines) or \
                ("**" in answer)
        details["checks"]["markdown_ok"] = md_ok
        if not md_ok:
            score -= 25

    if constraints["require_numbered_1_to_7"]:
        count = sum(1 for ln in answer.splitlines() if re.match(r'^\s*[1-7]\.\s+\S', ln))
        ok = count >= 7
        details["checks"]["numbered_1_to_7_ok"] = ok
        details["checks"]["numbered_count"] = count
        if not ok:
            score -= 30

    score = max(0, min(100, score))
    return score, critical_fail, details
