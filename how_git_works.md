Working directory  →  Staging area  →  Local repository  →  Remote repository
 (your files)         (git add)        (git commit)        (git push/pull)

 git add = choose what goes into the next snapshot

git commit = save that snapshot locally

git push = share those snapshots with others

git pull = bring others’ snapshots to you


HOW TO RUN CONTAINER LOCALLY:

1. Clone github branch/repo


2. Install uv:

pip install uv


3. Install deps from lock:

uv sync --frozen


4. Run:

uv run pytest