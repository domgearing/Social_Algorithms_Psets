"""
Part II: Self-Play with One Model

Two-phase workflow (as required by the assignment):
  Phase A (this script): Generate answers and write to CSV files.
  Phase B (separate):    Run judge.py on the CSV files to validate and score.
"""

from pathlib import Path
from tqdm import tqdm
from assignment3_starter import (
    ollama_generate,
    normalize_answer,
    load_questions,
    write_csv,
    build_player_prompt,
)

# ── Configuration ──────────────────────────────────────────────
MODEL = "llama3.2:3b"
TEMPERATURE = 1.0
ROUNDS = 10          # number of rounds per question (more = more stable estimates)
MAX_TOKENS = 16
TOP_K = 40
QUESTIONS_CSV = "scattergories_questions.csv"
OUTPUT_DIR = Path("outputs")


# ── Generate answers for one player ───────────────────────────
def generate_answers(player_id, questions, rounds):
    rows = []
    total = len(questions) * rounds

    with tqdm(total=total, desc=f"Generating {player_id}") as pbar:
        for q in questions:
            prompt = build_player_prompt(q.letter, q.category)

            for round_idx in range(rounds):
                raw = ollama_generate(
                    model=MODEL,
                    prompt=prompt,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    max_tokens=MAX_TOKENS,
                )
                answer = normalize_answer(raw)

                # Post-processing to clean up the answer:
                answer = answer.split("\n")[0]        # take first line only
                answer = " ".join(answer.split()[:4]) # cap at 4 words

                rows.append({
                    "question_id": q.question_id,
                    "letter": q.letter,
                    "category": q.category,
                    "round_idx": round_idx,
                    "answer": answer,
                    "model": MODEL,
                    "player_id": player_id,
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                })
                pbar.update(1)

    return rows


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = load_questions(QUESTIONS_CSV)
    print(f"Loaded {len(questions)} questions")
    print(f"Model: {MODEL}, Temp: {TEMPERATURE}, Rounds: {ROUNDS}")
    print(f"Total generations per player: {len(questions) * ROUNDS}")
    print()

    # Generate for player 1
    p1_rows = generate_answers("player1", questions, ROUNDS)
    p1_path = OUTPUT_DIR / "answers_player1.csv"
    write_csv(p1_path, p1_rows)
    print(f"Saved {len(p1_rows)} rows to {p1_path}\n")

    # Generate for player 2 (same model, same prompt, same temp)
    p2_rows = generate_answers("player2", questions, ROUNDS)
    p2_path = OUTPUT_DIR / "answers_player2.csv"
    write_csv(p2_path, p2_rows)
    print(f"Saved {len(p2_rows)} rows to {p2_path}\n")

    # Quick peek at collisions before judging
    print("── Quick collision check (before judging) ──")
    collisions = 0
    total_compared = 0
    for i in range(len(p1_rows)):
        if p1_rows[i]["round_idx"] == p2_rows[i]["round_idx"]:
            total_compared += 1
            if p1_rows[i]["answer"] == p2_rows[i]["answer"]:
                collisions += 1

    print(f"Identical answers: {collisions}/{total_compared} ({100*collisions/total_compared:.1f}%)")
    print()
    print("Next step: run the judge in your terminal:")
    print(f"  python3 judge.py {p1_path} {p2_path} --out outputs/scores.csv --details outputs/judged_rows.csv")
