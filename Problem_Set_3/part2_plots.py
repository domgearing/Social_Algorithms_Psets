"""
Part II Plots: Comparison histograms for the report.

Generates:
1. Original prompt across temperatures (1.0, 1.5, 2.0): validity, collision, avg score
2. Two prompts compared at temp 1.5: validity, collision, avg score
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")


def load_summary(path):
    """Load experiment_summary.csv and return list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def avg_players(rows):
    """Average metrics across the two players for a single experiment."""
    valid_pct = np.mean([float(r["valid_pct"]) for r in rows])
    collision_pct = np.mean([float(r["collision_pct"]) for r in rows])
    avg_score = np.mean([float(r["avg_score"]) for r in rows])
    return valid_pct, collision_pct, avg_score


def plot_across_temperatures(all_rows):
    """Plot 1: Original prompt across temperatures."""
    # Filter to original prompt only
    original_runs = [r for r in all_rows if "Scattergories" in r["prompt"]]

    # Group by temperature
    temps = sorted(set(r["temperature"] for r in original_runs))
    valid_pcts, collision_pcts, avg_scores = [], [], []

    for t in temps:
        t_rows = [r for r in original_runs if r["temperature"] == t]
        v, c, s = avg_players(t_rows)
        valid_pcts.append(v)
        collision_pcts.append(c)
        avg_scores.append(s * 100)  # scale to %

    x = np.arange(len(temps))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, valid_pcts, width, label="Validity %", color="steelblue")
    bars2 = ax.bar(x, collision_pcts, width, label="Collision %", color="salmon")
    bars3 = ax.bar(x + width, avg_scores, width, label="Avg Score (×100)", color="seagreen")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Percentage")
    ax.set_title("Self-Play: Original Prompt Across Temperatures")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}" for t in temps])
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()

    path = OUTPUT_DIR / "plot_temp_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_prompt_comparison(all_rows):
    """Plot 2: Two prompts compared at temp 1.5."""
    # Filter to temp 1.5 only
    t15_rows = [r for r in all_rows if r["temperature"] == "1.5"]

    original = [r for r in t15_rows if "Scattergories" in r["prompt"]]
    creative = [r for r in t15_rows if "unusual" in r["prompt"]]

    if not original or not creative:
        print("Missing data for prompt comparison at temp 1.5")
        return

    orig_v, orig_c, orig_s = avg_players(original)
    crea_v, crea_c, crea_s = avg_players(creative)

    labels = ["Validity %", "Collision %", "Avg Score (×100)"]
    orig_vals = [orig_v, orig_c, orig_s * 100]
    crea_vals = [crea_v, crea_c, crea_s * 100]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width/2, orig_vals, width, label="Original (Scattergories)", color="steelblue")
    bars2 = ax.bar(x + width/2, crea_vals, width, label="Creative (unusual/uncommon)", color="coral")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Percentage")
    ax.set_title("Self-Play: Prompt Comparison at Temperature 1.5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()

    path = OUTPUT_DIR / "plot_prompt_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    all_rows = load_summary(summary_path)
    print(f"Loaded {len(all_rows)} rows from {summary_path}\n")

    plot_across_temperatures(all_rows)
    plot_prompt_comparison(all_rows)

    print("\nDone! Check outputs/ for the PNGs.")
