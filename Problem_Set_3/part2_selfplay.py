"""
Part II: Self-Play with One Model

Two-phase workflow (as required by the assignment):
  Phase A (this script): Generate answers and write to CSV files.
  Phase B (separate):    Run judge.py on the CSV files to validate and score.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
TEMPERATURE = 1.5
ROUNDS = 10         # number of rounds per question (more = more stable estimates)
MAX_TOKENS = 16
TOP_K = 40
NUM_WORKERS = 8     # number of parallel Ollama calls - for faster generation
QUESTIONS_CSV = "scattergories_questions.csv"
OUTPUT_DIR = Path("outputs")


# ── Single call (used by each thread) ─────────────────────────
def generate_one(question, round_idx, player_id):
    """Generate one answer for one (question, round) pair."""
    prompt = build_player_prompt(question.letter, question.category)
    raw = ollama_generate(
        model=MODEL,
        prompt=prompt,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
    )
    answer = normalize_answer(raw)
    answer = answer.split("\n")[0]        # take first line only
    answer = " ".join(answer.split()[:4]) # cap at 4 words

    return {
        "question_id": question.question_id,
        "letter": question.letter,
        "category": question.category,
        "round_idx": round_idx,
        "answer": answer,
        "model": MODEL,
        "player_id": player_id,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
    }


# ── Generate answers for one player (parallel) ───────────────
def generate_answers(player_id, questions, rounds):
    """Generate all answers using parallel Ollama calls."""
    tasks = [(q, r) for q in questions for r in range(rounds)]
    rows = [None] * len(tasks)  # pre-allocate to preserve order

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_idx = {}
        for idx, (q, r) in enumerate(tasks):
            future = executor.submit(generate_one, q, r, player_id)
            future_to_idx[future] = idx

        with tqdm(total=len(tasks), desc=f"Generating {player_id}") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                rows[idx] = future.result()
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
