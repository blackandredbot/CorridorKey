```bash
git clone https://github.com/nikopueringer/CorridorKey.git
cd CorridorKey
uv sync --group dev    # installs all dependencies + dev tools (pytest, ruff)
```

No manual virtualenv creation, no `pip install` — uv handles everything.
