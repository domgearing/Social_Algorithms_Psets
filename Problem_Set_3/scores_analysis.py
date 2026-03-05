#!/usr/bin/env python3
"""Analyze `scores_*.csv` pairwise results and create visualizations + summary CSV.

This script reads all `scores_*.csv` files under a directory (default: `pairwise_outputs`),
normalizes player/opponent identifiers to model-level names, and produces:

- `pairwise_outputs/analysis/scores_summary.csv` — per-player/per-opponent summary rows
- `pairwise_outputs/analysis/avg_score_matrix_by_model_scores.csv` — model x model matrix
- PNGs in `pairwise_outputs/analysis/`:
  - `heatmap_scores_by_model.png` (model vs model heatmap)
  - `self_vs_cross_bar.png` (per-model self vs cross average)
  - `grouped_bar_by_opponent.png` (how each model scored vs each opponent)

The script also writes a short explanation text file describing the visualization choices.

Why these visuals?
- Heatmap: compact overview of how each model performs against every opponent (including itself).
- Self-vs-cross bar: isolates self-play behavior vs cross-play to highlight self-consistency or overfitting.
- Grouped bar: shows per-opponent performance for each model, useful for comparing pairwise strengths.

These complement a tabular CSV summary so you can programmatically inspect exact numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_score_files(scores_dir: Path):
    return sorted(Path(scores_dir).glob("scores_*.csv"))


def normalize_model_id(s: str) -> str:
    """Normalize various player/opponent ids to a canonical model base name.

    Examples: 'answers_gemma2_2b_selfA.csv' -> 'gemma', 'llama3.2_3b' -> 'llama'
    """
    import re

    if pd.isna(s):
        return s
    s = str(s)
    # strip common prefixes and file suffixes
    for p in ("answers_", "answer_", "answers-", "pairwise_outputs/answers_"):
        if s.startswith(p):
            s = s[len(p):]
            break
    for suf in ("_selfA", "_selfB", "_self", "-selfA", "-selfB", ".csv"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    # take alphabetic prefix as canonical model name
    m = re.match(r"^([A-Za-z]+)", s)
    if m:
        return m.group(1).lower()
    return Path(s).stem


def load_scores(scores_dir: Path) -> pd.DataFrame:
    files = find_score_files(scores_dir)
    if not files:
        raise FileNotFoundError(f"No scores_*.csv files in {scores_dir}")

    rows = []
    score_cols = ["avg_points_per_answer", "avg_points", "avg_score", "points", "score"]
    for f in files:
        df = pd.read_csv(f)
        # prefer rows that already have player_id/opponent_id
        if "player_id" in df.columns and "opponent_id" in df.columns:
            sc = next((c for c in score_cols if c in df.columns), None)
            for _, r in df.iterrows():
                s = float(r[sc]) if sc is not None and pd.notna(r[sc]) else np.nan
                rows.append({"player_id": r["player_id"], "opponent_id": r["opponent_id"], "score": s})
        else:
            # infer left/right from filename: scores_left_vs_right.csv
            stem = f.stem
            parts = stem.split("scores_")[-1].split("_vs_")
            left = parts[0] if len(parts) >= 1 else None
            right = parts[1] if len(parts) == 2 else None
            sc = next((c for c in score_cols if c in df.columns), None)
            # if df has a player/opponent-like column try to use them
            for _, r in df.iterrows():
                pid = r.get("player_id") or r.get("player") or left or "unknown"
                # Infer opponent based on filename (left vs right) and pid content
                raw_pid = str(pid)
                pid_norm = normalize_model_id(raw_pid)
                left_norm = normalize_model_id(left) if left else None
                right_norm = normalize_model_id(right) if right else None
                if left_norm and pid_norm == left_norm:
                    opp = right if right is not None else right_norm
                elif right_norm and pid_norm == right_norm:
                    opp = left if left is not None else left_norm
                else:
                    # fallback: try explicit columns, then right
                    opp = r.get("opponent_id") or r.get("opponent") or right or "unknown"

                s = float(r[sc]) if sc is not None and sc in df.columns and pd.notna(r[sc]) else np.nan
                rows.append({"player_id": pid, "opponent_id": opp, "score": s})

    pairs = pd.DataFrame(rows)
    # normalize
    pairs["player_model"] = pairs["player_id"].astype(str).map(normalize_model_id)
    pairs["opponent_model"] = pairs["opponent_id"].astype(str).map(normalize_model_id)
    return pairs


def make_summary(pairs: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = (
        pairs.groupby(["player_model", "opponent_model"])["score"]
        .agg(avg_score="mean", rounds="count")
        .reset_index()
    )
    summary["match_type"] = np.where(summary["player_model"] == summary["opponent_model"], "self", "cross")
    summary.to_csv(outdir / "scores_summary.csv", index=False)
    return summary


def matrix_by_model(summary: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    mat = summary.pivot(index="player_model", columns="opponent_model", values="avg_score")
    mat.to_csv(outdir / "avg_score_matrix_by_model_scores.csv")
    return mat


def plots(mat: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    sns.set(style="whitegrid")
    outdir.mkdir(parents=True, exist_ok=True)

    # Heatmap: model vs model
    plt.figure(figsize=(max(6, mat.shape[1] * 1.2), max(4, mat.shape[0] * 0.8)))
    ax = sns.heatmap(mat, annot=True, fmt=".3f", cmap="vlag", center=mat.stack().median())
    ax.set_title("Average score: player model (rows) vs opponent model (cols)")
    plt.tight_layout()
    plt.savefig(outdir / "heatmap_scores_by_model.png", dpi=200)
    plt.close()

    # Self vs cross: per-model average of self matches vs cross matches
    self_mean = summary[summary["match_type"] == "self"].set_index("player_model")["avg_score"]
    cross_mean = (
        summary[summary["match_type"] == "cross"].groupby("player_model")["avg_score"].mean()
    )
    comp = pd.DataFrame({"self": self_mean, "cross": cross_mean}).fillna(np.nan)
    plt.figure(figsize=(8, max(3, len(comp) * 0.4)))
    comp.plot(kind="bar", rot=45)
    plt.ylabel("Average score")
    plt.title("Self-play vs Cross-play average per model")
    plt.tight_layout()
    plt.savefig(outdir / "self_vs_cross_bar.png", dpi=200)
    plt.close()

    # Grouped bar: how each model did against each opponent (cross and self included)
    pivot = summary.pivot(index="player_model", columns="opponent_model", values="avg_score")
    pivot.plot(kind="bar", figsize=(max(8, pivot.shape[1] * 0.8), max(4, pivot.shape[0] * 0.6)))
    plt.ylabel("Average score")
    plt.title("Per-model performance vs each opponent")
    plt.legend(title="opponent_model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "grouped_bar_by_opponent.png", dpi=200)
    plt.close()


def write_explanation(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    txt = (
        "Visualizations created:\n"
        "- heatmap_scores_by_model.png: compact matrix of how each model scores vs each opponent.\n"
        "- self_vs_cross_bar.png: compares each model's self-play average vs its average against other models; useful to spot whether a model is more successful against itself (consistency) or performs better vs others.\n"
        "- grouped_bar_by_opponent.png: shows per-opponent bars so you can inspect which opponents each model beats or loses to.\n\n"
        "Reasoning:\n"
        "Heatmaps provide immediate at-a-glance pairwise strengths. Self-vs-cross highlights internal consistency or self-bias. Grouped bars make it easy to compare per-opponent differences across models. Together they cover both detailed numeric analysis (CSV) and visual pattern recognition.\n"
    )
    (outdir / "scores_analysis_explanation.txt").write_text(txt)


def main():
    p = argparse.ArgumentParser(description="Analyze scores_*.csv pairwise outputs")
    p.add_argument("--scores-dir", default="pairwise_outputs", help="directory containing scores_*.csv files")
    p.add_argument("--outdir", default="pairwise_outputs/analysis", help="output directory for CSVs and plots")
    args = p.parse_args()

    scores_dir = Path(args.scores_dir)
    outdir = Path(args.outdir)

    pairs = load_scores(scores_dir)
    summary = make_summary(pairs, outdir)
    mat = matrix_by_model(summary, outdir)
    plots(mat, summary, outdir)
    write_explanation(outdir)

    print("Wrote:")
    print(" -", outdir / "scores_summary.csv")
    print(" -", outdir / "avg_score_matrix_by_model_scores.csv")
    print(" - plots to", outdir)


if __name__ == "__main__":
    main()
