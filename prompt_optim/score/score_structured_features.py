from __future__ import annotations
import json
import os
from tqdm import tqdm
from datetime import datetime
import shutil
from .config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, JUDGE_MODEL, LLM_TIMEOUT,
    WEIGHTS, STABILITY_LAMBDA, USE_PAIRWISE_QUALITY, BASELINE_PROMPT,
    MAX_OUTPUT_TOKENS, JUDGE_MAX_TOKENS, MAX_CLAIMS, OUTPUT_DIR,CURRENT_DIR
)
from .llm_client import LLMClient
from .prompt_runner import fill_prompt, build_messages
from .utils import ensure_dir, write_text
from .scoring_hard import hard_score
from .scoring_fact import factuality_score
from .scoring_qual import quality_pairwise_score
from .aggregate import aggregate_case, aggregate_prompt

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_structured_features(structured):
    print("----------------正在进行评分------------------")
    orig_outputs_dir = os.path.join(CURRENT_DIR, "..", "data", "outputs")
    if os.path.exists(orig_outputs_dir):
        shutil.rmtree(orig_outputs_dir)

    ensure_dir(OUTPUT_DIR)
    client = LLMClient(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)
    eval_path=os.path.join(CURRENT_DIR, "..", "data", "eval_cases.json")
    cases = load_json(eval_path)

    baseline_outputs = {}
    if USE_PAIRWISE_QUALITY:
        for case in tqdm(cases, desc="Generating baseline outputs"):
            filled = fill_prompt(BASELINE_PROMPT, case)
            messages = build_messages(filled)
            out = client.chat(messages=messages, model=LLM_MODEL, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
            baseline_outputs[case["case_id"]] = out
            write_text(os.path.join(OUTPUT_DIR, f"baseline_{case['case_id']}.md"), out)

    for idx, item in enumerate(tqdm(structured, desc="Scoring prompts")):
        raw_prompt = item.get("raw_prompt", "")
        prompt_dir = os.path.join(OUTPUT_DIR, f"prompt_{idx:02d}")
        ensure_dir(prompt_dir)

        by_case = {}
        case_final_scores = []

        for case in cases:
            filled_prompt = fill_prompt(raw_prompt, case)
            messages = build_messages(filled_prompt)
            answer = client.chat(messages=messages, model=LLM_MODEL, temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

            ans_path = os.path.join(prompt_dir, f"{case['case_id']}_answer.md")
            write_text(ans_path, answer)

            h, critical, hard_details = hard_score(raw_prompt, answer)
            write_text(os.path.join(prompt_dir, f"{case['case_id']}_hard.json"),
                       json.dumps(hard_details, ensure_ascii=False, indent=2))

            f, fact_details = factuality_score(
                client=client,
                source=case["text"],
                summary=answer,
                judge_model=JUDGE_MODEL,
                max_claims=MAX_CLAIMS,
                max_tokens=JUDGE_MAX_TOKENS
            )
            write_text(os.path.join(prompt_dir, f"{case['case_id']}_fact.json"),
                       json.dumps(fact_details, ensure_ascii=False, indent=2))

            if USE_PAIRWISE_QUALITY:
                baseline = baseline_outputs[case["case_id"]]
                q, qual_details = quality_pairwise_score(
                    client=client,
                    source=case["text"],
                    cand=answer,
                    baseline=baseline,
                    judge_model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS
                )
                write_text(os.path.join(prompt_dir, f"{case['case_id']}_qual.json"),
                           json.dumps(qual_details, ensure_ascii=False, indent=2))
            else:
                q = 60

            agg = aggregate_case(hard=h, fact=f, qual=q, weights=WEIGHTS, critical_fail=critical)
            final_case = agg["final"]
            case_final_scores.append(final_case)

            by_case[case["case_id"]] = {
                "hard": h,
                "fact": f,
                "qual": q,
                "final": final_case,
                "capped": agg["capped"],
                "answer_file": os.path.relpath(ans_path, OUTPUT_DIR)
            }

        totals = aggregate_prompt(case_final_scores, STABILITY_LAMBDA)

        scores_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "weights": WEIGHTS,
            "stability_lambda": STABILITY_LAMBDA,
            "avg": totals["avg"],
            "std": totals["std"],
            "fitness": totals["fitness"],
            "by_case": by_case
        }

        item["scores"] = scores_obj
        item["score"] = totals["fitness"]

    return structured