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

```bash
source venv/bin/activate
python scripts/run_ingest.py
python scripts/run_extraction.py
```

Add exported JSON files to `data/raw/` before ingesting.
