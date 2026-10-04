# chat-topology

Analyze chat exports and explore conversation topology.

## Folder structure

```
chat-topology/
├── config/          # Tunable config files (current pipeline)
│   └── legacy/      # Tunable config files (superseded word-level pipeline)
├── data/
│   ├── raw/         # Drop exported JSON chat files here (not tracked by git)
│   └── processed/   # Intermediate pipeline outputs (not tracked by git)
├── frontend/        # Three.js / React Three Fiber visualization app
├── scripts/         # CLI entrypoints for the current pipeline; call src/ modules
│   └── legacy/      # CLI entrypoints for the superseded word-level pipeline
├── src/             # Python modules for the current pipeline (core logic)
│   └── legacy/      # Python modules for the superseded word-level pipeline
├── requirements.txt
└── README.md
```

| Path | Purpose |
|------|---------|
| `data/raw/` | Place your exported chat JSON files here manually. |
| `data/processed/` | Current pipeline's intermediate artifacts (plus `conversations.json`, shared with the legacy pipeline). |
| `data/processed/legacy/` | Superseded word-level pipeline's intermediate artifacts. |
| `src/` | Reusable Python modules for the current (proximity-based) pipeline. |
| `src/legacy/` | Modules for the superseded word-frequency/graph pipeline -- kept working, not deleted; see `CLAUDE.md`'s "Pipeline reconstruction" for why it was replaced. |
| `scripts/` | Thin CLI wrappers for the current pipeline; import and call `src/`. |
| `scripts/legacy/` | Thin CLI wrappers for the superseded pipeline; import and call `src/legacy/`. |
| `config/` | Project settings and tunable parameters for the current pipeline. |
| `config/legacy/` | Tunable parameters for the superseded pipeline. |
| `frontend/` | Browser-based Three.js / React Three Fiber app. |

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

python scripts/run_ingest.py               # raw exports -> data/processed/conversations.json
python scripts/run_segmentation.py         # -> data/processed/segments.json (review before continuing)
python scripts/run_grounded_extraction.py  # -> segment_keywords.json + segment_embeddings.npz
python scripts/run_conversation_concepts.py  # -> conversation_concepts.json (review)
python scripts/run_macro_domains.py        # -> macro_domains.json (review)
python scripts/run_proximity_layout.py     # -> proximity_layout.json + preview PNG (review)
python scripts/run_build_output_v3.py      # -> data/processed/data.json (schema v3)
```

Each step prints a summary meant to be reviewed before moving to the next. Then copy
`data/processed/data.json` to `frontend/public/data.json` by hand to see it in the app.

An earlier word-frequency/graph pipeline (`scripts/legacy/`) is kept working but superseded --
see `CLAUDE.md`'s "Pipeline reconstruction" section for the full story of why, and that
pipeline's own command order.

See `CLAUDE.md` for project context, design decisions, and current milestone status.
