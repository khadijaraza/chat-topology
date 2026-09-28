# chat-topology

Analyze chat exports and explore conversation topology.

## Folder structure

```
chat-topology/
├── config/          # Tunable config files (YAML or JSON)
├── data/
│   ├── raw/         # Drop exported JSON chat files here (not tracked by git)
│   └── processed/   # Intermediate pipeline outputs (not tracked by git)
├── frontend/        # Three.js visualization app (future)
├── scripts/         # CLI entrypoints that call src/ modules
├── src/             # Python modules (core logic)
├── requirements.txt
└── README.md
```

| Path | Purpose |
|------|---------|
| `data/raw/` | Place your exported chat JSON files here manually. |
| `data/processed/` | Pipeline writes intermediate artifacts here. |
| `src/` | Reusable Python modules. |
| `scripts/` | Thin CLI wrappers; import and call `src/`. |
| `config/` | Project settings and tunable parameters. |
| `frontend/` | Browser-based Three.js app (not started yet). |

## Setup

### Create and activate the virtual environment (macOS)

```bash
cd chat-topology
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

`en_core_web_sm` is the small English spaCy model (medium accuracy, fast on a laptop for thousands of documents).

### How to activate venv on macOS

From the project root:

```bash
source venv/bin/activate
```

Your shell prompt should show `(venv)`. To deactivate:

```bash
deactivate
```

## Pipeline

Add exported JSON files to `data/raw/` before ingesting, then run in order:

```bash
source venv/bin/activate

python scripts/run_ingest.py              # raw exports -> data/processed/conversations.json
python scripts/run_phrase_detection.py    # -> data/processed/phrases.json (review before continuing)
python scripts/run_extraction.py          # -> data/processed/chat_keywords.json
python scripts/run_graph_build.py         # -> data/processed/graph.gpickle
python scripts/run_layout.py              # -> data/processed/layout.json + layout_preview.png
python scripts/run_layout_interactive.py  # optional: data/processed/layout_interactive.html (pan/zoom/hover)
```

Each step prints a summary meant to be reviewed before moving to the next — especially `run_phrase_detection.py`'s output, since bad phrase merges propagate into every downstream node and edge.

See `CLAUDE.md` for project context, design decisions, and current milestone status.
