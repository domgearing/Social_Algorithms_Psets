#!/usr/bin/env python3
"""Run pairwise 2-player competitions across local models.

Usage examples:
  python3 pairwise_runner.py --models "llama3.2:3b,llama3.2:7b,llama3.2:13b" \
    --temperature 1.0 --rounds 2

If `--models` is omitted the script will attempt to query a local inference
server at http://localhost:11434/api/models to discover available models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
ASSIGNMENT_SCRIPT = ROOT.parent / "Problem_Set_3" / "assignment3_starter.py"
JUDGE_SCRIPT = ROOT.parent / "Problem_Set_3" / "judge.py"
OUTDIR = ROOT / "pairwise_outputs"


def discover_models() -> List[str]:
    # Try common local API endpoint for model listing (ollama/other local servers)
    url = "http://localhost:11434/api/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Expect a list of model descriptors or strings
            if isinstance(data, list):
                models = []
                for item in data:
                    if isinstance(item, str):
                        models.append(item)
                    elif isinstance(item, dict) and "name" in item:
                        models.append(item["name"])
                return models
    except Exception:
        pass
    return []


def gen_answers_for_model(model: str, temperature: float, rounds: int, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    player_id = model.replace("/", "_").replace(":", "_")
    out_csv = outdir / f"answers_{player_id}.csv"
    cmd = [
        sys.executable,
        str(ASSIGNMENT_SCRIPT),
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
    ]
    print("Generating answers for", model)
    subprocess.check_call(cmd)
    return out_csv


def judge_pair(files: List[Path], outdir: Path, openai_model: str = "gpt-4o-mini") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    names = [p.stem for p in files]
    out_summary = outdir / f"scores_{names[0]}_vs_{names[1]}.csv"
    out_details = outdir / f"judged_{names[0]}_vs_{names[1]}.csv"
    cmd = [
        sys.executable,
        str(JUDGE_SCRIPT),
        str(files[0]),
        str(files[1]),
        "--model",
        openai_model,
        "--out",
        str(out_summary),
        "--details",
        str(out_details),
    ]
    print("Judging pair:", files[0].name, "vs", files[1].name)
    subprocess.check_call(cmd)
    return out_summary


def aggregate_pair_summaries(summaries: List[Path], dest: Path) -> None:
    rows = []
    for p in summaries:
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["match_file"] = p.name
                rows.append(r)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No summary rows to write")
        return
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None, help="Comma-separated list of local model tags.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--openai-model", default="gpt-4o-mini")
    p.add_argument("--outdir", default=str(OUTDIR))
    args = p.parse_args()

    outdir = Path(args.outdir)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = discover_models()
        if not models:
            print("No models discovered on local server and --models not provided. Exiting.")
            sys.exit(1)

    if len(models) < 3:
        print("Please provide at least 3 models (found {}).").format(len(models))
        # still continue with what we have

    # Limit to first 5 models to keep judge costs reasonable
    models = models[:5]

    # 1) generate one answer file per model
    generated_files = []
    for model in models:
        out_csv = gen_answers_for_model(model, args.temperature, args.rounds, outdir)
        generated_files.append(out_csv)

    # 2) run pairwise matches
    summaries = []
    for a, b in combinations(generated_files, 2):
        summary = judge_pair([a, b], outdir, openai_model=args.openai_model)
        summaries.append(summary)

    # 3) aggregate results
    agg = outdir / "aggregated_pairwise_summaries.csv"
    aggregate_pair_summaries(summaries, agg)
    print(f"Wrote aggregated summary: {agg}")


if __name__ == "__main__":
    main()
