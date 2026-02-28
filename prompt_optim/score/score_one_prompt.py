from __future__ import annotations
import json
import argparse

from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, JUDGE_MODEL, LLM_TIMEOUT,
    WEIGHTS, USE_PAIRWISE_QUALITY, BASELINE_PROMPT,
    MAX_OUTPUT_TOKENS, JUDGE_MAX_TOKENS, MAX_CLAIMS
)
from llm_client import LLMClient
from prompt_runner import fill_prompt, build_messages
from scoring_hard import hard_score
from scoring_fact import factuality_score
from scoring_qual import quality_pairwise_score
from aggregate import aggregate_case

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="prompt index in structured_features.json")
    ap.add_argument("--case", type=str, default="", help="case_id to run; empty=all")
    args = ap.parse_args()

    client = LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
    structured = load_json("structured_features.json")
    cases = load_json("eval_cases.json")

    raw_prompt = structured[args.index]["raw_prompt"]

    baseline_outputs = {}
    if USE_PAIRWISE_QUALITY:
        for case in cases:
            if args.case and case["case_id"] != args.case:
                continue
            filled = fill_prompt(BASELINE_PROMPT, case)
            baseline_outputs[case["case_id"]] = client.chat(build_messages(filled), LLM_MODEL, 0.0, MAX_OUTPUT_TOKENS)

    results = {}
    for case in cases:
        if args.case and case["case_id"] != args.case:
            continue

        filled_prompt = fill_prompt(raw_prompt, case)
        answer = client.chat(build_messages(filled_prompt), LLM_MODEL, 0.0, MAX_OUTPUT_TOKENS)

        h, critical, _ = hard_score(raw_prompt, answer)
        f, _ = factuality_score(client, case["text"], answer, JUDGE_MODEL, MAX_CLAIMS, JUDGE_MAX_TOKENS)

        if USE_PAIRWISE_QUALITY:
            q, _ = quality_pairwise_score(client, case["text"], answer, baseline_outputs[case["case_id"]], JUDGE_MODEL, JUDGE_MAX_TOKENS)
        else:
            q = 60

        agg = aggregate_case(h, f, q, WEIGHTS, critical)
        results[case["case_id"]] = {"hard": h, "fact": f, "qual": q, "final": agg["final"], "capped": agg["capped"]}

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
