## Dev setup (Codespaces)
1. Click "Code" -> "Codespaces" -> "Create codespace on main"
2. Wait for container build
3. Done (deps installed automatically through `uv sync --frozen`)

## Run tests
uv run pytest

## Run notebooks
Open in VS Code Jupyter (kernel: repo environment)


## Pull Requests
submit pull request before merging code into main branch

Pull Requests run CI that checks code to ensure it works on main branch and runs before accepting merge

requires approval of at least one other person before merge
