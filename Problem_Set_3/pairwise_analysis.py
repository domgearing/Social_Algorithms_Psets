#!/usr/bin/env python3
"""Summarize pairwise judged results and produce visualizations.

Writes CSV summaries and PNG plots to an analysis directory.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_judged(judged_dir: Path) -> pd.DataFrame:
    judged_dir = Path(judged_dir)
    files = sorted(judged_dir.glob("judged_*.csv"))
    if not files:
        raise FileNotFoundError(f"No judged_*.csv files in {judged_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_scores(scores_dir: Path) -> pd.DataFrame:
    """Load scores_*.csv files and return a pairs-like DataFrame with
    columns: player_id, opponent_id, score
    """
    scores_dir = Path(scores_dir)
    files = sorted(scores_dir.glob("scores_*.csv"))
    if not files:
        raise FileNotFoundError(f"No scores_*.csv files in {scores_dir}")

    rows = []
    # candidate score columns in order of preference
    score_cols = ["avg_points_per_answer", "avg_points", "avg_score", "points", "score"]
    for f in files:
        df = pd.read_csv(f)
        # try to determine opponent from columns
        if "opponent_id" in df.columns and "player_id" in df.columns:
            for _, r in df.iterrows():
                # pick first available score column
                s = None
                for c in score_cols:
                    if c in df.columns:
                        s = r[c]
                        break
                if s is None:
                    # try any numeric column
                    for c in df.columns:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            s = r[c]
                            break
                rows.append({"player_id": r["player_id"], "opponent_id": r["opponent_id"], "score": float(s)})
        else:
            # infer from filename: scores_left_vs_right.csv
            stem = f.stem
            parts = stem.split("scores_")[-1].split("_vs_")
            left = right = None
            if len(parts) == 2:
                left, right = parts
            # pick best score column
            score_col = next((c for c in score_cols if c in df.columns), None)
            if score_col is None:
                # fallback to first numeric column
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                score_col = num_cols[0] if num_cols else None
            for _, r in df.iterrows():
                pid = r.get("player_id") if "player_id" in df.columns else None
                if pid is None and left and left in r.to_string():
                    pid = left
                # if still unknown, try to build pid from row index
                if pid is None:
                    pid = r.get("player", left)
                # opponent: infer from filename if not present
                opp = r.get("opponent_id") if "opponent_id" in df.columns else None
                if opp is None:
                    opp = right if left and left in str(pid) else left
                s = float(r[score_col]) if score_col and score_col in df.columns else float(np.nan)
                rows.append({"player_id": pid, "opponent_id": opp, "score": s})

    pairs_df = pd.DataFrame(rows)
    return pairs_df


def build_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    # Expect each round_key to contain the two players' rows for that round.
    rows = []
    for rk, g in df.groupby("round_key"):
        players = list(g["player_id"])
        scores = list(g["score"])
        valids = list(g["valid"])
        # For each player row, create one row per opponent (handles >2 players robustly)
        for i, pid in enumerate(players):
            for j, opp in enumerate(players):
                if i == j:
                    continue
                rows.append(
                    {
                        "round_key": rk,
                        "player_id": pid,
                        "opponent_id": opp,
                        "score": int(scores[i]),
                        "valid": int(valids[i]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_scores(pairs_df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    # Build average score matrix: rows=player, cols=opponent
    mat = pairs_df.groupby(["player_id", "opponent_id"])["score"].mean().unstack(fill_value=np.nan)
    outdir.mkdir(parents=True, exist_ok=True)
    mat.to_csv(outdir / "avg_score_matrix.csv")
    # Also produce a model-level aggregated matrix (normalize IDs)
    def normalize_model(x: str) -> str:
        if pd.isna(x):
            return x
        s = str(x)
        # strip common prefixes
        for p in ("answers_", "answer_", "answers-"):
            if s.startswith(p):
                s = s[len(p):]
                break
        # remove common suffixes
        for suf in ("_selfA", "_selfB", "_self", "-selfA", "-selfB"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        # if file-like path, take basename
        s = Path(s).name
        return s

    pairs_df = pairs_df.copy()
    pairs_df["player_model"] = pairs_df["player_id"].astype(str).map(normalize_model)
    pairs_df["opponent_model"] = pairs_df["opponent_id"].astype(str).map(normalize_model)
    model_mat = pairs_df.groupby(["player_model", "opponent_model"])["score"].mean().unstack(fill_value=np.nan)
    model_mat.to_csv(outdir / "avg_score_matrix_by_model.csv")
    return mat


def validity_rates(df: pd.DataFrame, outdir: Path) -> pd.Series:
    vr = df.groupby("player_id")["valid"].mean().sort_index()
    vr.to_csv(outdir / "validity_rates.csv", header=["validity_rate"] )
    return vr


def plots(mat: pd.DataFrame, vr: pd.Series, pairs_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # Heatmap: average score per player vs opponent
    plt.figure(figsize=(8, max(4, len(mat) * 0.6)))
    ax = sns.heatmap(mat, annot=True, fmt=".3f", cmap="vlag", center=0.5, cbar_kws={"label": "avg score"})
    ax.set_title("Average score per player vs opponent")
    plt.tight_layout()
    plt.savefig(outdir / "avg_score_heatmap.png", dpi=200)
    plt.close()

    # Validity bar chart (skip if not provided)
    if not vr.empty:
        plt.figure(figsize=(6, max(3, len(vr) * 0.4)))
        vr.plot(kind="bar", color="C1")
        plt.ylabel("Validity rate")
        plt.ylim(0, 1.02)
        plt.title("Validity rate per model")
        plt.tight_layout()
        plt.savefig(outdir / "validity_rates.png", dpi=200)
        plt.close()

    # Self-play vs cross-play
    self_mask = pairs_df["player_id"] == pairs_df["opponent_id"]
    cross_mean = pairs_df.groupby("player_id")["score"].mean()
    # Create a simple comparison plot showing cross-play averages (best-effort self-play requires judged_all including self matches)
    df_sc = pd.DataFrame({"cross_play": cross_mean}).fillna(0)
    df_sc.to_csv(outdir / "cross_play_avg.csv")

    plt.figure(figsize=(8, max(3, max(1, len(df_sc)) * 0.4)))
    df_sc.plot(kind="bar")
    plt.ylabel("Average score")
    plt.title("Cross-play average scores per model")
    plt.tight_layout()
    plt.savefig(outdir / "cross_play_avg.png", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Analyze pairwise judged results and create visualizations")
    p.add_argument("--judged-dir", default="pairwise_outputs", help="directory containing judged_*.csv files")
    p.add_argument("--scores-dir", default="pairwise_outputs", help="directory containing scores_*.csv files")
    p.add_argument("--use-scores", action="store_true", help="use scores_*.csv files instead of judged_*.csv")
    p.add_argument("--outdir", default="pairwise_outputs/analysis", help="output directory for CSVs and plots")
    args = p.parse_args()

    judged_dir = Path(args.judged_dir)
    outdir = Path(args.outdir)

    if args.use_scores:
        pairs_df = load_scores(args.scores_dir)
        # when using scores, validity rates are not available; create placeholder
        mat = summarize_scores(pairs_df, outdir)
        vr = pd.Series(dtype=float)
        plots(mat, vr, pairs_df, outdir)
    else:
        df = load_judged(judged_dir)
        pairs_df = build_pairwise(df)
        mat = summarize_scores(pairs_df, outdir)
        vr = validity_rates(df, outdir)
        plots(mat, vr, pairs_df, outdir)

    print("Wrote:")
    print(" -", outdir / "avg_score_matrix.csv")
    print(" -", outdir / "validity_rates.csv")
    print(" - plots to", outdir)


if __name__ == "__main__":
    main()

