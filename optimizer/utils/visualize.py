import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path("prompt_optim/data")
IMG_DIR = Path("static/optimizer")
IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_population(gen_idx: int):
    file = DATA_DIR / f"selected_prompts{gen_idx}.json"
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_stats():
    max_scores = []
    avg_scores = []

    for i in range(1, 12):
        pop = load_population(i)
        scores = [p["score"] for p in pop]
        max_scores.append(max(scores))
        avg_scores.append(sum(scores) / len(scores))

    return max_scores, avg_scores


def plot_fitness_curve():
    max_scores, avg_scores = compute_stats()

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 12), max_scores, marker="o", label="Max Score")
    plt.plot(range(1, 12), avg_scores, marker="s", label="Average Score")

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Population Fitness Evolution")
    plt.legend()
    plt.grid(True)

    out_path = IMG_DIR / "fitness_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path.name


def plot_boxplot():
    gens = [1, 6, 11]
    data = []

    for g in gens:
        pop = load_population(g)
        data.append([p["score"] for p in pop])

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[f"G{g}" for g in gens])
    plt.ylabel("Score")
    plt.title("Score Distribution Across Generations")
    plt.grid(True, axis="y")

    out_path = IMG_DIR / "score_boxplot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path.name


def get_best_prompt_gen11():
    pop = load_population(11)
    max_score = max(p["score"] for p in pop)
    candidates = [p for p in pop if p["score"] == max_score]
    chosen = random.choice(candidates)

    return {
        "score": max_score,
        "raw_prompt": chosen["raw_prompt"]
    }