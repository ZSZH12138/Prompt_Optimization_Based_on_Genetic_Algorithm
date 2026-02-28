from __future__ import annotations
from typing import Dict, Any
import statistics

def aggregate_case(hard: int, fact: int, qual: int, weights: Dict[str, float], critical_fail: bool) -> Dict[str, Any]:
    if critical_fail:
        final = min(20, int(round(0.5 * hard + 0.5 * fact)))
        return {"final": final, "capped": True}

    final = (
        weights["hard"] * hard +
        weights["fact"] * fact +
        weights["qual"] * qual
    )
    final = int(round(max(0, min(100, final))))
    return {"final": final, "capped": False}

def aggregate_prompt(case_scores: list[int], stability_lambda: float) -> Dict[str, float]:
    avg = float(sum(case_scores)) / max(1, len(case_scores))
    std = float(statistics.pstdev(case_scores)) if len(case_scores) > 1 else 0.0
    fitness = avg - stability_lambda * std
    return {"avg": round(avg, 2), "std": round(std, 2), "fitness": round(fitness, 2)}
