"""
Part II Analysis: Measure self-play outcomes from judged results.

Usage:
    python3 part2_analysis.py outputs/judged_rows.csv

Reads the judged_rows.csv output from judge.py and reports:
- Overall validity rate, collision rate, avg score per player
- Notable questions (hardest, most collisions)
- Saves a summary CSV for comparing across experiments
- Generates plots for the report
"""

import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ── Record what settings you used ─────────────────────────────
# Update these each time you run a new experiment!
EXPERIMENT_NOTES = {
    "temperature": 1.5,
    "prompt": (
        "You are playing Scattergories."
        "Category: {category}\n"
        "Letter: {letter}\n"
    f   "Name one {category} that starts with the letter {letter}.\n"
        "Reply with ONLY the answer. One or two words max."
    ),
    "model": "llama3.2:3b",
    "rounds": 10,
}

OUTPUT_DIR = Path("outputs")


def load_judged_rows(path):
    """Load judged_rows.csv from judge.py output."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def analyze(rows):
    """Compute overall and notable-question metrics."""

    # ── Overall metrics per player ────────────────────────────
    by_player = defaultdict(list)
    for row in rows:
        by_player[row["player_id"]].append(row)

    print("=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)

    player_summaries = []
    for player_id, player_rows in sorted(by_player.items()):
        total = len(player_rows)
        valid = sum(1 for r in player_rows if r["valid"] == "1")
        points = sum(1 for r in player_rows if r["score"] == "1")
        collisions = sum(1 for r in player_rows if r["collision"] == "1")

        print(f"\n  {player_id}:")
        print(f"    Total answers:   {total}")
        print(f"    Valid:           {valid}/{total} ({100*valid/total:.1f}%)")
        print(f"    Collisions:      {collisions}/{total} ({100*collisions/total:.1f}%)")
        print(f"    Points:          {points}/{total} ({100*points/total:.1f}%)")
        print(f"    Avg score:       {points/total:.3f}")

        player_summaries.append({
            "player_id": player_id,
            "total": total,
            "valid": valid,
            "valid_pct": 100 * valid / total,
            "collisions": collisions,
            "collision_pct": 100 * collisions / total,
            "points": points,
            "avg_score": points / total,
        })

    # ── Per-question stats (for notable questions + plots) ────
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)

    question_stats = []
    for qid in sorted(by_question.keys()):
        q_rows = by_question[qid]
        letter = q_rows[0]["letter"]
        category = q_rows[0]["category"]
        total = len(q_rows)
        valid = sum(1 for r in q_rows if r["valid"] == "1")
        collisions = sum(1 for r in q_rows if r["collision"] == "1")
        points = sum(1 for r in q_rows if r["score"] == "1")

        question_stats.append({
            "question_id": qid,
            "letter": letter,
            "category": category,
            "valid_pct": 100 * valid / total,
            "collision_pct": 100 * collisions / total,
            "avg_score": points / total,
        })

    # ── Notable questions ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("NOTABLE QUESTIONS")
    print("=" * 60)

    by_validity = sorted(question_stats, key=lambda x: x["valid_pct"])
    print("\n  Lowest validity (hardest):")
    for q in by_validity[:5]:
        print(f"    {q['question_id']} ({q['letter']}, {q['category']}): {q['valid_pct']:.1f}% valid")

    by_collision = sorted(question_stats, key=lambda x: x["collision_pct"], reverse=True)
    print("\n  Highest collision rate:")
    for q in by_collision[:5]:
        print(f"    {q['question_id']} ({q['letter']}, {q['category']}): {q['collision_pct']:.1f}% collisions")

    by_score = sorted(question_stats, key=lambda x: x["avg_score"], reverse=True)
    print("\n  Highest scoring:")
    for q in by_score[:5]:
        print(f"    {q['question_id']} ({q['letter']}, {q['category']}): avg score {q['avg_score']:.3f}")

    return player_summaries, question_stats


def save_experiment_summary(player_summaries):
    """Append this experiment's results to a running summary CSV."""
    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    file_exists = summary_path.exists()

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "temperature", "rounds", "prompt",
            "player_id", "total", "valid", "valid_pct",
            "collisions", "collision_pct", "points", "avg_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for p in player_summaries:
            writer.writerow({
                "model": EXPERIMENT_NOTES["model"],
                "temperature": EXPERIMENT_NOTES["temperature"],
                "rounds": EXPERIMENT_NOTES["rounds"],
                "prompt": EXPERIMENT_NOTES["prompt"],
                **p,
            })

    print(f"\nAppended to {summary_path}")


def make_plots(question_stats):
    """Generate plots for the report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp = EXPERIMENT_NOTES["temperature"]

    # ── Plot 1: Overall bar chart (validity, collision, score) ─
    labels = [f"{q['question_id']}" for q in question_stats]
    valid_pcts = [q["valid_pct"] for q in question_stats]
    collision_pcts = [q["collision_pct"] for q in question_stats]
    avg_scores = [q["avg_score"] * 100 for q in question_stats]  # scale to %

    fig, ax = plt.subplots(figsize=(14, 5))
    x = range(len(labels))
    width = 0.28
    ax.bar([i - width for i in x], valid_pcts, width, label="Valid %", color="steelblue")
    ax.bar(x, collision_pcts, width, label="Collision %", color="salmon")
    ax.bar([i + width for i in x], avg_scores, width, label="Avg Score (×100)", color="seagreen")
    ax.set_xlabel("Question ID")
    ax.set_ylabel("Percentage")
    ax.set_title(f"Self-Play Per-Question Metrics (temp={temp})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"selfplay_per_question_temp{temp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # ── Plot 2: Validity vs Collision scatter ─────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(valid_pcts, collision_pcts, alpha=0.6, edgecolors="black", linewidth=0.5)
    for q in question_stats:
        ax.annotate(q["question_id"], (q["valid_pct"], q["collision_pct"]),
                     fontsize=5, alpha=0.7)
    ax.set_xlabel("Validity %")
    ax.set_ylabel("Collision %")
    ax.set_title(f"Validity vs Collision Rate (temp={temp})")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"selfplay_validity_vs_collision_temp{temp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 part2_analysis.py outputs/judged_rows.csv")
        sys.exit(1)

    judged_path = sys.argv[1]
    print(f"Analyzing: {judged_path}\n")

    # Print experiment settings
    print("=" * 60)
    print("EXPERIMENT SETTINGS")
    print("=" * 60)
    for key, val in EXPERIMENT_NOTES.items():
        print(f"  {key}: {val}")
    print()

    rows = load_judged_rows(judged_path)
    player_summaries, question_stats = analyze(rows)

    # Save summary CSV (appends, so you build up results across experiments)
    save_experiment_summary(player_summaries)

    # Generate plots
    make_plots(question_stats)
