# chat-topology — project context for Claude Code

Personal project: a Python pipeline that turns personal AI chat exports (Claude, ChatGPT,
Gemini) into an interactive 3D point cloud (Three.js/React Three Fiber). Topics are nodes
in 3D space: spatial clustering = relatedness, height = engagement depth, color =
conversational friction. Built on macOS, medium-sized dataset.

## How I like to work — read this before making changes

- **Incremental milestones, not big jumps.** Build and verify one milestone at a time. If
  asked to fix or extend one part of the pipeline, don't silently start the next milestone
  too.
- **Every tunable value goes in `config/*.yaml`.** Never hardcode a threshold, weight, or
  list that I might want to adjust later — see the existing config files for the pattern
  (comments explaining *why* the default is what it is, not just what it does).
- **Every pipeline script needs a manual "before you move on" checkpoint.** Print a
  reviewable summary (counts, top-N examples, thresholds used) — don't just write the
  output file silently. `scripts/run_phrase_detection.py` is the reference example.
- **Explain the mechanism before patching a bug**, especially anything touching PMI,
  statistics, or NLP extraction. I want to understand *why* something went wrong, not just
  get a diff. If a fix has a real limitation or edge case, say so plainly rather than
  presenting it as complete.
- Test any pipeline change end-to-end with synthetic data before calling it done —
  reproduce the bug/scenario, then verify the fix actually resolves it.

## Architecture

```
config/           tunable YAML configs
data/raw/         chat exports go here manually (gitignored)
data/processed/   pipeline intermediate outputs (gitignored)
frontend/         Vite + React + TypeScript + React Three Fiber app (milestones 8-9 done)
scripts/          thin CLI entrypoints; import from src/, do the printing/summarizing
src/              core pipeline logic; scripts import from here, not the reverse
```

`frontend/src/components/`: `Scene.tsx` (Canvas, camera, lighting, OrbitControls, data
fetch, overview/detail drill-down state), `PointCloud.tsx` (instanced spheres — renders
either concept nodes or supertopic nodes, same component, see `RenderableNode` in
`types.ts`), `EdgeLines.tsx` (drei `Line`, top-N by weight, hover-highlighted), `Legend.tsx`
(counts + back-to-overview button). `frontend/src/lib/selectTopEdges.ts` caps rendered
edges (`EDGE_RENDER_LIMIT`). `frontend/src/types.ts` mirrors `build_final_json()`'s schema
— keep both in sync by hand.

## Pipeline (run in this order)

```bash
source venv/bin/activate
python scripts/run_ingest.py              # data/raw/*  -> conversations.json
python scripts/run_phrase_detection.py    # -> phrases.json  (REVIEW before continuing)
python scripts/run_extraction.py          # -> chat_keywords.json
python scripts/run_concept_clustering.py  # -> chat_concepts.json + concepts.json  (REVIEW before continuing)
python scripts/run_graph_build.py         # -> graph.gpickle
python scripts/run_supertopic_clustering.py  # -> supertopics.json  (REVIEW before continuing)
python scripts/run_layout.py              # -> layout.json + layout_preview.png
python scripts/run_layout_interactive.py  # optional -> layout_interactive.html (pan/zoom/hover)
python scripts/run_height_scores.py       # -> height_scores.json
python scripts/run_build_output.py        # runs ingest/extraction/graph/layout/height/merge,
                                           # reapplying the reviewed concepts.json + reading
                                           # supertopics.json as-is (see below)
```

`scripts/run_build_output.py` is the one command to run once phrase detection and concept
clustering have each been reviewed at least once — it re-ingests, re-extracts, and reapplies
the *already-reviewed* `concepts.json` mapping to the fresh extraction (cheap; it does NOT
re-embed/re-cluster, which takes several minutes), then runs graph build through merge in one
shot. It falls back gracefully to ungrouped/unmerged data if either review step's output file
doesn't exist yet, matching the pattern already established for `phrases.json`.
`run_supertopic_clustering.py` is NOT re-run by `run_build_output.py` at all (unlike concept
clustering, there's no "reapply cheaply" step for it) — `build_final_json()` just reads
whatever `supertopics.json` is already on disk, so rerun it manually after a pipeline change
that meaningfully shifts the concept set (its embeddings would otherwise reference concepts
that no longer exist, or miss new ones -- harmless since unmapped concepts just get no
supertopic, see `_load_concept_to_supertopic()`, but stale).

`scripts/inspect_schema.py <path>` introspects a raw export's JSON structure without
assuming a platform schema — run it first when wiring up a new export format, before
writing a parser.

No automated test suite exists. "Test end-to-end with synthetic data" (above) means
running the pipeline scripts against small hand-built fixtures in a scratch dir, not `pytest`.

Copying `data/processed/data.json` to `frontend/public/data.json` is a deliberate **manual**
step, not automated by any script — this is so it's always obvious whether the frontend is
showing stale or fresh output. If the running dev server "isn't showing changes" after a
pipeline rerun, check `frontend/public/data.json`'s mtime before assuming the frontend broke.

## Key decisions and gotchas — don't relitigate or silently undo these

- **`graph.gpickle` uses stdlib `pickle`**, not `nx.write_gpickle` (removed in NetworkX
  3.x). Preserves float PMI precision exactly; fine since the pipeline is Python-only.
  Switch to GraphML/JSON only when something outside Python needs to read the graph
  directly.
- **Layout coordinates normalize to `[-100, 100]`** on whichever axis has the larger
  extent; both axes scale by the *same* factor to preserve cluster shape (never
  independently stretch x and y). The range is `coordinate_range` in `config/layout.yaml`
  — treat it as the contract with the eventual Three.js scene bounds.
- **`nx.spring_layout` requires `scipy`** — not an automatic networkx dependency, must
  stay pinned in `requirements.txt`.
- **Keyword extraction must never double-count a phrase and its component words from the
  same span.** A chunk like "machine learning" incrementing `machine`, `learning`, AND
  `machine learning` together guarantees near-perfect (fake) PMI between the components —
  this was a real bug. `segment_with_phrases()` in `src/extract_keywords.py` fixes it by
  ensuring each span contributes to exactly one output term. Preserve that invariant in
  any future extraction changes.
- **Two-tier alias resolution** (`config/keyword_aliases.yaml`): single-word aliases
  (`git`/`github`) resolve at the lemma level, *before* phrase detection ever runs, so
  compounds like "git repo"/"github repo" unify from the start. Multi-word/acronym aliases
  (`ucsd` → `uc san diego`) resolve once at the end on fully-formed terms, since there's no
  1:1 token mapping when word counts differ. Don't collapse this into a single pass —
  tried it, it breaks compound merging (see chat history if this needs revisiting).
- **Phrase detection prunes redundant sub-phrases**: a shorter phrase is dropped when a
  longer one accounts for almost all its occurrences (`redundancy_ratio` in
  `config/phrase_detection.yaml`). This is why `uc san` disappears once `uc san diego` is
  found, while `san diego` correctly stays separate (it occurs independently too often to
  be redundant).
- **Known gap**: phrase detection only merges NOUN–NOUN adjacency (matches
  `extract_keywords.py`'s restriction to NOUN/PROPN). ADJ–NOUN compounds like "neural
  network" aren't corpus-level-merged yet.
- **HTML/LaTeX/code markup leaking through as junk keywords was a real bug, now fixed.**
  `SPECIFICITY_PATTERNS` in `src/extract_keywords.py` originally only caught a
  backslash/digit at the very *start* of a term; spaCy noun chunks routinely start
  mid-expression (`n/3 \rceil`, `print(f"{time`, `</script`), so markup slipped through
  unless matched anywhere in the string. Extended with patterns for backslash-anywhere,
  braces, angle brackets, code quotes, and brackets — verified against real junk samples
  with zero false positives on real topic phrases.
- `pip install` needs `--break-system-packages` on this machine's Python; `spacy download`
  may need the model wheel installed directly instead (PEP 668 restrictions) — see README.
- **Ingestion currently only implements Gemini.** `src/ingest.py` declares
  `Platform = Literal["claude", "chatgpt", "gemini"]`, but `load_all_conversations()` only
  reads `data/raw/gemini/` — there's no `claude`/`chatgpt` parser yet, despite the project
  description covering all three. Adding one means writing a `parse_<platform>_export()`
  path parallel to the existing Gemini one, not extending it.
- Config directory also includes `config/keyword_stopwords.yaml` (junk terms excluded from
  extraction, hand-curated) alongside `keyword_aliases.yaml`, `layout.yaml`, and
  `phrase_detection.yaml`.
- **Known gap**: the top nodes by `chat_count` are dominated by generic filler words
  (`reason`, `limit`, `choice`, `check`, `power`...) that add little topical signal.
  `keyword_stopwords.yaml` is designed for exactly this ("edit this list as you review
  extraction output") but hasn't been extended to cover them yet — flagged, not fixed.
- **Graph nodes are semantic concepts, not raw keywords** (see "Concept clustering"
  below) — `build_cooccurrence_graph()` itself is unaware of this; it just operates on
  whatever "terms" its input chat-level lists contain, which is `chat_concepts.json`
  (concept-grouped) rather than `chat_keywords.json` (raw) as of the concept-clustering
  redesign. Don't repoint `run_graph_build.py`/`run_height_scores.py` back at
  `chat_keywords.json` without deliberately deciding to drop concept clustering.
- **Edge-count explosion was a real bug, now bounded via `config/graph.yaml`.**
  `min_chat_count` alone doesn't bound total edges: one keyword-rich chat contributes
  `C(k, 2)` edges for its `k` surviving terms, so a handful of very long conversations
  blew the raw-keyword graph up to 7.9M edges (768MB `data.json`) even after
  `min_chat_count` filtering. Fixed with two changes: `min_chat_count` raised from 2 to 3,
  and a new `max_edges_per_node` (default 40) that prunes edges to each node's strongest
  ties via `prune_edges_to_top_k_per_node()` — keeps an edge if it's in *either* endpoint's
  top-K by weight, bounding total edges to at most `N * max_edges_per_node`. **Real
  limitation**: pruning happens *after* the full dense graph is built, so it shrinks the
  final output but not peak memory/time during construction.
- **Concept clustering** (`src/concept_clustering.py`, `config/concept_clustering.yaml`,
  `scripts/run_concept_clustering.py`) groups keywords into semantic concepts *before*
  graph build, on purpose — this is a genuinely different signal from the co-occurrence
  edges: two keywords can end up in the same concept because they mean similar things
  (sentence-transformers embeddings), independent of whether they ever co-occur in a chat.
  HDBSCAN (density-based) clusters the embeddings rather than a fixed keyword count
  (k-means) because the keyword distribution is heavily power-law skewed. Like phrase
  detection, this is a **manual review gate** (`scripts/run_concept_clustering.py`'s
  checkpoint output) — `run_build_output.py` does NOT recompute it, it reapplies the
  reviewed `concepts.json` mapping to fresh extraction instead (re-embedding+re-clustering
  ~35k terms took 9+ minutes with the stronger model).
  - Started with `all-MiniLM-L6-v2` (~80MB): only merged 30% of keywords into a real
    cluster, and produced a real polysemy collision (`fingertip` merged with `pro tip` —
    the model has no sentence context to disambiguate senses of "tip" on short
    decontextualized phrases). Tuning `min_cluster_size`/`min_samples` did NOT fix this —
    confirmed the collision was identical before and after tuning, proving it was an
    embedding-space limitation, not a density-threshold one.
  - Switched to `all-mpnet-base-v2` (~420MB, slower CPU inference): fixed the polysemy
    collision confirmed directly (`finger`/`fingertip`/`thumb` cluster is now clean), but
    merge rate stayed flat at ~29-32% regardless of model or tuning — this now looks like
    a real property of the data (most keywords in a personal chat corpus genuinely are
    one-off/unique), not something further tuning will move.
  - Both `sentence-transformers` (pulls in `torch`) and `hdbscan` (pulls in `scikit-learn`)
    are real, heavy dependencies (`torch` alone is hundreds of MB) — fully local/offline
    inference, no API calls, but a meaningfully bigger install than the rest of the stack.
- **React 19 is pinned to exactly `19.2.8`** in `frontend/package.json` (not `^19.2.8`) —
  `@react-three/fiber@9.7.0`'s peer range is `>=19 <19.3`, and Vite's react-ts template
  installs whatever the caret range resolves to, which was `19.3.0` and broke `npm install`
  with an ERESOLVE conflict. Don't loosen this pin without checking R3F's peer range again.
- **`frontend/src/main.tsx` deliberately has no `<StrictMode>`.** With React 19.2.8 +
  `@react-three/fiber` 9.7.0, StrictMode's double-invoked mount/cleanup/mount left the R3F
  `<canvas>` element stuck at the browser's default un-sized state (300x150, `position:
  static`) instead of the resize-observed 100%/100% it should get — confirmed by comparing
  computed canvas size with StrictMode present vs. removed. Revisit if a future
  `@react-three/fiber` release documents this as fixed.
- **`PointCloud.tsx` uses spheres over boxes** via drei's `Instances`/`Instance` (single
  GPU-instanced draw call) — chosen for how it reads as an abstract "point cloud" rather
  than a grid of blocks; cost-wise the two are equivalent at this node count since geometry
  is uploaded once and reused per instance.
- **Supertopics are a second, coarser clustering layer on top of concepts**
  (`src/supertopic_clustering.py`, `config/supertopic_clustering.yaml`,
  `scripts/run_supertopic_clustering.py`) — built because 5,521 concept nodes was still too
  many to read as a graph, and the goal was a hard cap (comfortably under 100), which
  concept clustering's HDBSCAN can't directly target (it finds however many clusters
  naturally exist). Uses **agglomerative clustering with a fixed `n_clusters`** instead,
  over concept-label embeddings (same model as concept clustering, reused). Every concept
  is force-assigned to some supertopic — no "noise" bucket, unlike concept clustering — a
  hard count cap and "some things left ungrouped" are in tension.
  - **Must cluster `graph.gpickle`'s node list, not `concepts.json` directly.**
    `concepts.json` is concept clustering's full, unfiltered output (tens of thousands of
    entries, mostly singletons that never survive `config/graph.yaml`'s `min_chat_count`
    filter); clustering that instead of the graph's actual ~5-6k nodes pulls in massive
    amounts of noise the graph never shows anyway (confirmed: first attempt read
    `concepts.json`'s 27,525 entries and produced garbage groupings dominated by leftover
    junk fragments). `scripts/run_supertopic_clustering.py` reads `graph.gpickle`.
  - **`linkage: average` produced one dominant catch-all cluster** (1,472 of 5,521 concepts,
    27% of the whole graph, labeled "reason" but containing everything from "abbreviation"
    to "academy" to "accent") plus several other oversized, incoherent clusters — a known
    failure mode of average-linkage with a hard cluster-count cap on heterogeneous data.
    Switched to **`linkage: ward`** (minimizes within-cluster variance): max cluster dropped
    to 170 (3.1%), and the large majority of the 80 clusters read as genuinely coherent
    themes on inspection.
  - A super-topic's `(x, y, z)` in `data.json` is the **centroid** of its member concept
    nodes' positions, not an independently laid-out position; `chat_count` is a true
    distinct-chat count (a chat touching several concepts in the same super-topic only
    counts once, computed in `_supertopic_chat_counts()`), not a naive sum, which would
    double-count. Edges between concepts in the *same* super-topic are dropped when rolling
    up into `superedges`; only cross-super-topic weight survives (summed).
- **`data.json` schema is now version 2**: nodes gained a `supertopic: string | null` field,
  and the payload gained top-level `supertopics` and `superedges` arrays (empty/all-null if
  `supertopics.json` doesn't exist yet — same graceful-degradation pattern as concepts).
  `validate_schema()` checks referential integrity both ways (every node's `supertopic`
  points at a real supertopic id when supertopics exist; every superedge's endpoints exist).
- **Frontend defaults to an "Overview" of supertopics, not the full concept point cloud.**
  `Scene.tsx` holds `viewMode: 'overview' | 'detail'`; clicking a supertopic node drills into
  its member concepts (filtering `data.nodes`/`data.edges` by `node.supertopic`), and drei's
  `Bounds` (`fit clip observe`, remounted via a `key` on view state) reframes the camera to
  fit whatever's currently rendered. Falls back straight to the flat concept view (old
  behavior) if `data.supertopics` is empty, e.g. an older `data.json` built before this
  layer existed.

## Milestone status (11-milestone build plan)

**Done:** graph construction (PMI-weighted co-occurrence, configurable pruning),
force-directed layout with static + interactive previews, corpus-level phrase detection
(NPMI-based, with redundancy pruning), keyword alias/canonicalization (two-tier), height
score computation (log1p + min-max normalized, YAML-configurable weights), semantic
concept clustering (embeddings + HDBSCAN, grouping keywords into graph nodes), a second
supertopic clustering layer (agglomerative, fixed count, <100 groups) for graph legibility,
unified `data.json` Python→frontend contract (schema v2, with supertopics/superedges) with
schema validation, React Three Fiber point cloud scaffold (Vite + React 19.2.8 + drei),
edge rendering + hover/click interaction + legend, overview/detail drill-down between
supertopics and their member concepts with camera auto-fit.

**Next up:** VADER + custom-phrase friction scoring and color mapping (the "conversational
friction = color" dimension from the project description hasn't been built yet).

**Deferred to v2** (don't build unless explicitly asked): terrain mesh interpolation,
Ollama-based LLM friction judging, community-detection coloring, 2D D3 network view.

## Stack

Python, NetworkX, spaCy, PyYAML, scipy, matplotlib, plotly, sentence-transformers (+ torch),
hdbscan (+ scikit-learn) (pipeline) → Vite, React 19.2.8, TypeScript, React Three Fiber /
Three.js, drei (frontend). Config-driven parameterization throughout — see `config/` for
the existing pattern before adding a new tunable.
