#!/usr/bin/env python3
"""Generate two distinct answer files per model and judge them against each other.

This creates two runs for each model (suffixes _selfA and _selfB) using different
seed ranges so the answers are distinct, then calls `judge.py` to score the match.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ASSIGNMENT = Path(__file__).resolve().parent / "assignment3_starter.py"
JUDGE = Path(__file__).resolve().parent / "judge.py"


def safe_player_id(model: str, suffix: str) -> str:
    return model.replace("/", "_").replace(":", "_") + f"_{suffix}"


def gen_answers(model: str, temperature: float, rounds: int, seed_start: int, player_id: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"answers_{player_id}.csv"
    cmd = [
        sys.executable,
        str(ASSIGNMENT),
        "generate-answers",
        "--model",
        model,
        "--rounds",
        str(rounds),
        "--temperature",
        str(temperature),
        "--player-id",
        player_id,
        "--outdir",
        str(outdir),
        "--out",
        str(out_csv),
        "--seed-start",
        str(seed_start),
    ]
    print("Generating answers:", model, "->", out_csv.name, "seed_start=", seed_start)
    subprocess.check_call(cmd, cwd=ASSIGNMENT.parent)
    return out_csv


def judge_pair(file_a: Path, file_b: Path, outdir: Path, openai_model: str, cache: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / f"scores_{file_a.stem}_vs_{file_b.stem}.csv"
    details = outdir / f"judged_{file_a.stem}_vs_{file_b.stem}.csv"
    cmd = [
        sys.executable,
        str(JUDGE),
        str(file_a),
        str(file_b),
        "--out",
        str(summary),
        "--details",
        str(details),
        "--cache",
        str(cache),
        "--sleep",
        "0.1",
    ]
    print("Judging:", file_a.name, "vs", file_b.name)
    subprocess.check_call(cmd, cwd=JUDGE.parent)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="Comma-separated list of model tags")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-offset", type=int, default=100000)
    p.add_argument("--outdir", default="pairwise_outputs")
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--cache", default="pairwise_outputs/.judge_cache.json")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    outdir = Path(args.outdir)
    cache = Path(args.cache)

    for i, model in enumerate(models):
        base_seed = args.seed_start + i * (args.seed_offset * 2)
        a_id = safe_player_id(model, "selfA")
        b_id = safe_player_id(model, "selfB")
        file_a = gen_answers(model, args.temperature, args.rounds, base_seed, a_id, outdir)
        file_b = gen_answers(model, args.temperature, args.rounds, base_seed + args.seed_offset, b_id, outdir)

        # Judge the two runs against each other
        judge_pair(file_a, file_b, outdir, args.openai_model, cache)

    print("All self-play matches generated and judged. Results in:", outdir)


if __name__ == "__main__":
    main()
