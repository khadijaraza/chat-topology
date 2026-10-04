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
config/           tunable YAML configs (current pipeline)
config/legacy/    tunable YAML configs for the superseded word-level pipeline
data/raw/         chat exports go here manually (gitignored)
data/processed/   pipeline intermediate outputs, current pipeline (gitignored)
data/processed/legacy/  pipeline intermediate outputs, superseded word-level pipeline
                  (gitignored) -- conversations.json (shared input) and segment_keywords.json
                  (shared, current pipeline needs it too) stay in data/processed/ root
frontend/         Vite + React + TypeScript + React Three Fiber app (milestones 8-9 done)
scripts/          thin CLI entrypoints for the current pipeline; import from src/
scripts/legacy/   thin CLI entrypoints for the superseded word-level pipeline
src/              current pipeline's core logic; scripts import from here, not the reverse
src/legacy/       superseded word-level pipeline's core logic (chat_keywords -> phrase
                  detection -> word-concept clustering -> PMI graph -> spring layout ->
                  supertopics), kept working, not deleted -- see "Pipeline reconstruction"
                  below for the full history of what replaced it and why
```

**The `legacy/` split (src, scripts, config) is a strict superset/subset relationship, not a
duplicate pipeline living side by side with unclear boundaries.** Every file in
`src/legacy/`, `scripts/legacy/`, `config/legacy/` belongs to the ORIGINAL word-level
pipeline (chat-level keyword frequency -> corpus-level phrase detection -> word-concept
HDBSCAN clustering -> PMI co-occurrence graph -> spring layout -> agglomerative supertopics)
and produces `data.json` schema v2. Nothing outside `legacy/` depends on anything inside it,
by construction -- confirmed via a full import-graph check when the split was made (see
"Pipeline reconstruction" -> "Directory reorganization" below). A few files are genuinely
**mixed** (shared utility + legacy-only function in the same file) and deliberately stay in
`src/` root rather than being split further, each with a docstring note explaining which part
is which:
- **`src/extract_keywords.py`**: NLP helpers (`get_nlp`, `get_stopwords`, `is_valid_term`,
  `normalize_phrase`, `normalize_token_lemma`, `content_nouns_in_span`) are shared --
  `src/grounded_extraction.py` (current) imports them directly. `extract_keywords()` itself
  (the whole-chat frequency entrypoint) is legacy-only.
- **`src/layout.py`**: `get_coordinate_range()`/`normalize_positions()` are shared --
  `src/proximity_layout.py` (current) reuses them to rescale UMAP output to the same
  world-coordinate contract. `compute_layout()` (spring layout on a graph) is legacy-only.
- **`src/height_score.py`**: `get_height_config()` is shared (current pipeline reuses its
  `z_range`). `compute_height_scores()` (the 3-metric graph-based formula) is legacy-only.
- **`src/build_output.py`**: contains both `build_final_json()`/`validate_schema()` (schema
  v2, legacy-only) and `build_final_json_v3()`/`validate_schema_v3()` (schema v3, current).
- **`src/text_embedding.py`**: `get_embedding_model()` was moved HERE from
  `src/legacy/concept_clustering.py` specifically so `src/segmentation.py` and
  `src/grounded_extraction.py` (both current) could keep using the same cached
  sentence-transformers loader without importing from `legacy/` -- this is what let
  `concept_clustering.py` move to `legacy/` as a whole instead of staying mixed.

`frontend/src/components/`: `Scene.tsx` (Canvas, camera, lighting, OrbitControls, data
fetch, 3-tier macro/concept/segment drill-down state — see "Pipeline reconstruction" ->
Step 6 below), `PointCloud.tsx` (instanced spheres — renders whichever tier is active, same
component throughout, see `RenderableNode` in `types.ts`), `DensityTerrain.tsx` (optional
DPGMM terrain mesh, toggled via `Legend.tsx`, rendered outside `Bounds` so it doesn't affect
per-tier camera auto-fit — see "Continuous density field" below), `Legend.tsx` (counts +
back-to-overview button + terrain toggle). `EdgeLines.tsx` and `frontend/src/lib/selectTopEdges.ts` are
**unused, kept not deleted** — schema v3 has no edges, proximity replaced them entirely;
they're only reachable from schema v2's superseded code path, which no longer exists in
`Scene.tsx`. `frontend/src/types.ts` mirrors `build_final_json_v3()`'s schema — keep both in
sync by hand. `frontend/src/types.ts` also still declares the v2 interfaces
(`TopicNode`/`Supertopic`/`TopicEdge`/`GraphData`) for reference; they're not live code.

## Pipeline (run in this order)

```bash
source venv/bin/activate
python scripts/run_ingest.py              # data/raw/*  -> conversations.json
python scripts/run_segmentation.py        # -> segments.json  (REVIEW before continuing;
                                           #    Phase 1)
python scripts/run_grounded_extraction.py # -> segment_keywords.json + segment_embeddings.npz
                                           #    (REVIEW before continuing; Phase 2)
python scripts/run_conversation_concepts.py  # -> conversation_concepts.json  (REVIEW;
                                           #    Step 2 -- HDBSCAN on segment embeddings)
python scripts/run_macro_domains.py       # -> macro_domains.json  (REVIEW; Step 3 --
                                           #    mutual-kNN + Leiden/CPM community detection
                                           #    directly on segments, TF-ICF single-term
                                           #    labels, explicit Background/Minor Orbit bucket)
python scripts/run_proximity_layout.py    # -> proximity_layout.json + preview PNG  (REVIEW;
                                           #    Step 4 -- UMAP projection, replaces graph layout)
python scripts/run_density_field.py       # optional -> density_field.json + preview PNG
                                           #    (REVIEW -- DPGMM continuous terrain, an
                                           #    alternative/additional reading to Macro
                                           #    Domains' discrete boundaries, not required)
python scripts/run_build_output_v3.py     # -> data.json (schema v3: segments/
                                           #    conversation_concepts/macro_domains, no edges;
                                           #    + density_field if the above was run)
```

Copy `data/processed/data.json` to `frontend/public/data.json` by hand when you want the
frontend to pick up a fresh run (see below).

**Superseded pipeline** (`scripts/legacy/`, produces schema v2 -- kept working, not deleted,
not run by default; see "Pipeline reconstruction" for why it was replaced):

```bash
python scripts/run_ingest.py                     # shared with the current pipeline
python scripts/legacy/run_phrase_detection.py    # -> phrases.json  (REVIEW before continuing)
python scripts/legacy/run_extraction.py          # -> chat_keywords.json
python scripts/legacy/run_concept_clustering.py  # -> chat_concepts.json + concepts.json  (REVIEW)
python scripts/legacy/run_graph_build.py         # -> graph.gpickle
python scripts/legacy/run_supertopic_clustering.py  # -> supertopics.json  (REVIEW)
python scripts/legacy/run_layout.py              # -> layout.json + layout_preview.png
python scripts/legacy/run_layout_interactive.py  # optional -> layout_interactive.html
python scripts/legacy/run_height_scores.py       # -> height_scores.json
python scripts/legacy/run_build_output.py        # runs ingest/extraction/graph/layout/height/
                                                  # merge, reapplying reviewed concepts.json +
                                                  # reading supertopics.json as-is (see below)
python scripts/legacy/run_segment_concept_clustering.py  # -> segment_concepts.json +
                                                  # segment_concept_summary.json -- Phase 3 of
                                                  # the reconstruction, itself ALSO superseded
                                                  # by run_conversation_concepts.py above; kept
                                                  # for the same reason, not part of either
                                                  # pipeline's normal run order
```

`scripts/legacy/run_build_output.py` is the one command to run once phrase detection and
concept clustering have each been reviewed at least once — it re-ingests, re-extracts, and
reapplies the *already-reviewed* `concepts.json` mapping to the fresh extraction (cheap; it
does NOT re-embed/re-cluster, which takes several minutes), then runs graph build through
merge in one shot. It falls back gracefully to ungrouped/unmerged data if either review
step's output file doesn't exist yet, matching the pattern already established for
`phrases.json`. `run_supertopic_clustering.py` is NOT re-run by `run_build_output.py` at all
(unlike concept clustering, there's no "reapply cheaply" step for it) — `build_final_json()`
just reads whatever `supertopics.json` is already on disk, so rerun it manually after a
pipeline change that meaningfully shifts the concept set (its embeddings would otherwise
reference concepts that no longer exist, or miss new ones -- harmless since unmapped
concepts just get no supertopic, see `_load_concept_to_supertopic()`, but stale).

`scripts/inspect_schema.py <path>` introspects a raw export's JSON structure without
assuming a platform schema — run it first when wiring up a new export format, before
writing a parser. Shared by both pipelines; stays in `scripts/` root.

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
- **Pipeline reconstruction is underway (4 phases), Phase 1 done.** The existing
  chat-level, frequency-based extraction still produced a graph judged "insignificant" even
  after concept + supertopic clustering. Root cause: whole-chat extraction blends unrelated
  sub-topics within one chat into false co-occurrence, and frequency-ranking has no way to
  prefer a summarizing keyword over a merely-repeated one. Fix is architectural: segment
  each chat into discourse-cohesive units first (**Phase 1, done**), then extract by
  semantic similarity-to-segment instead of frequency (**Phase 2**), refine concept
  induction's embedding strategy (**Phase 3**), then restructure the graph around segments
  with tangents linked back to their parent chat (**Phase 4**). Phases 2-4 are intentionally
  not yet speced in detail — their parameters depend on what Phase 1's real segment output
  looks like. Full plan: `~/.claude/plans/wobbly-cuddling-lynx.md` (if still present).
  - **`src/segmentation.py` / `config/segmentation.yaml` / `scripts/run_segmentation.py`**:
    sliding 2-turn windows (stride 1), embedded with the same `all-mpnet-base-v2` model
    (chunk + mean-pool windows over the 384-token limit rather than truncating -- real turns
    are heavy-tailed, p90 522 words, max 3,588 words, so truncation would silently drop
    content even at small window sizes). Two independent detectors: TextTiling-style valley
    detection for abrupt shifts, rolling similarity-to-initial-centroid for gradual drift.
    Segments get labeled `"core"` (largest by word count) or `"tangent"` (everything else).
  - **Hard-shift detection needed BOTH a relative and an absolute criterion, not just
    z-score.** A pure `depth_score > mean + z*std` adaptive threshold badly over-segmented
    on real data: the longest real chat (486 turns) flagged 90 of 484 candidate boundaries
    (18.6%) as hard shifts at `z=1.0`. Reading the actual chat content at those boundaries
    (a long, coherent astronomy/data-analysis session, not several unrelated topics)
    confirmed these were natural sub-topic variation within one coherent task, not real
    tangents -- a z-score threshold scales badly with sequence length since longer chats
    sample more of the trajectory's own noise tail. Fixed by requiring the boundary's raw
    similarity to also fall below an absolute floor (`hard_shift_max_similarity: 0.4`) in
    addition to being a relative outlier -- cut that chat to 28 flagged boundaries (5.8%)
    while a clean synthetic abrupt-topic-switch test case (raw similarity 0.24) still passed
    easily. Don't revert to z-score alone without re-checking real long-chat behavior.
  - **Known interpretive nuance**: many 2-segment chats show a short `"tangent"` segment
    *first*, followed by a large `"core"` segment -- this is usually the drift detector
    correctly noticing the conversation hadn't "settled" into its real topic during the
    first couple of exchanges yet (the drift anchor is the mean of the first
    `drift_min_consecutive` windows), not a genuine off-topic digression. `"tangent"` here
    means "not part of the dominant segment," not necessarily "irrelevant."
  - **Phase 2 (`src/grounded_extraction.py` / `config/grounded_extraction.yaml` /
    `scripts/run_grounded_extraction.py`, output `segment_keywords.json`)**: per segment,
    generate candidates the same way `extract_keywords.py` does (noun chunks/entities/
    PROPN, same stopword + `SPECIFICITY_PATTERNS` filtering), but rank by cosine similarity
    to the segment's own embedding instead of frequency, then MMR-diversify the top-K. A
    keyword that *summarizes* a segment now outranks one that just repeats. Real quality
    jump confirmed by reading 20 real segments' actual extracted keywords next to their
    text (not just counts) -- e.g. `"malloc allocation bug"`, `"complex conjugate
    eigenvalue"`, `"quintessential xinjiang dish"`, `"familial hypercholesterolemia"`.
    - **Fenced code-block stripping (part of the original plan) doesn't work on this
      corpus.** `html_to_text()` (`src/ingest.py`) already discards `<pre>`/`<code>` tag
      boundaries when converting HTML to plain text, so there's no fence marker left to
      strip at the text level -- code ends up structurally indistinguishable from prose.
      Only LaTeX (`$...$`/`$$...$$`, survives as literal characters), CLI flags, and
      assignment-looking fragments (`x = 5`) are stripped via `strip_non_discursive()`;
      residual code-syntax junk in candidates still relies on `SPECIFICITY_PATTERNS`.
      Properly fixing fenced-block stripping means changing ingestion to preserve
      `<pre>`/`<code>` boundaries as explicit markers -- a separate, earlier-pipeline change.
    - **MMR needed a token-overlap penalty, not cosine similarity alone.** Real data:
      a "greedy algorithm" segment let 3-4 near-identical surface variants
      (`"greedy algorithm"`, `"greedy algorithms"`, `"greedy algorithms explain"`) stack in
      the top 8 -- embedding similarity under-penalizes phrases that share most of their
      words but embed slightly differently. Fixed with three layered mechanisms (all in
      `src/grounded_extraction.py`): `collapse_surface_variants()` groups candidates by a
      canonical lemma root tuple (trailing conversational verbs/prepositions stripped, see
      `TRAILING_STRIP_LEMMAS`) *before* embedding, so near-duplicates never reach MMR as
      separate candidates; `mmr_select()`'s diversity penalty is
      `max(cosine_similarity, token_jaccard_overlap)` with overlap ≥
      `jaccard_hard_threshold` forcing an automatic near-veto; `prune_substrings()` is a
      post-MMR safety net dropping any selected phrase that's an exact substring of another.
    - **Foreign-language fragments need a language-agnostic guard, not a language-specific
      one.** Real data: a Norwegian sign-translation segment surfaced `"kjør i"` (a cut-off
      verb+preposition fragment) as its top keyword -- spaCy's English pipeline tags
      unfamiliar foreign tokens unreliably (often PROPN). Fixed narrowly:
      `MIN_FINAL_TOKEN_LENGTH` rejects any candidate whose last word is under 2 characters
      (real content-bearing final nouns rarely are, regardless of language) -- doesn't
      require knowing what language the fragment is in. **Known residual gap**: a
      single mis-tagged foreign token that *isn't* a short trailing fragment (e.g. bare
      `"kjør"`, the verb itself) can still surface, since nothing currently requires
      single-word candidates to be genuine entities/proper nouns rather than any
      PROPN-mistagged OOV word. Narrower than the original bug; not yet fixed.
    - `normalize_phrase()` (`src/extract_keywords.py`) only strips leading `a/an/the`;
      other leading determiners (`that malloc`, `my c program`, `what tool`) leaked into
      candidates. Fixed locally in `grounded_extraction.py`
      (`_strip_leading_determiner()`) rather than in the shared function, which the
      already-validated legacy `extract_keywords()`/`concept_clustering()` pathway still
      depends on -- deliberately not touched without reason.
    - Chunk+pool embedding logic (originally written for Phase 1's window embedding) was
      extracted into `src/text_embedding.py` so Phase 2 could reuse it without duplication;
      `src/segmentation.py` now imports from there too.
  - **Phase 3 (`cluster_segment_keywords_into_concepts()` in `src/concept_clustering.py`,
    `scripts/run_segment_concept_clustering.py`, output `segment_concepts.json` +
    `segment_concept_summary.json`)**: same HDBSCAN clustering core as the original concept
    layer, but embeds each term WITH lightweight co-occurrence context (its top
    co-occurring keywords from whichever segment it scored highest in --
    `build_term_contexts()`) instead of the bare decontextualized term. Verified against a
    genuinely polysemous bare word ("bank"): context correctly split financial-sense
    segments from river-sense segments into two separate concepts, no cross-contamination.
    Both this and the original `cluster_keywords_into_concepts()` now share a common
    `_build_concepts_from_embeddings()` tail (refactored, no behavior change to the
    original pathway). **Real, large result**: merge rate jumped from ~30% (original
    chat-keyed clustering) to 97% (8,701 of 8,928 keywords merged into a real cluster), and
    concept count dropped to 1,488 (from 5,521 graph nodes under the old pathway) --
    grounded, segment-scoped candidates give HDBSCAN far more separable signal than
    frequency-ranked whole-chat keywords did. Writes new files rather than overwriting
    `concepts.json`/`chat_concepts.json` -- nothing downstream (graph build, supertopic
    clustering) has been rewired to consume this yet; that's Phase 4.
    - **Found via real-data single-word spot-checking, not the top-N-by-chat_count
      preview**: literal parentheses (`pish(stdin`, `realloc(word`) were never in
      `SPECIFICITY_PATTERNS` -- only braces/brackets/backslash/quotes/angle-brackets were
      added during the earlier markup-leak fix. Fixed by adding `re.compile(r"[()]")` to
      the shared list (verified against the full junk/legit regression battery, zero false
      positives) -- affects both this pathway and the original one, since the list lives in
      `extract_keywords.py`.
      **Known residual gap**: dotted attribute/method-style fragments (`copytree.size`,
      `tree.root.successor`) aren't filtered either -- periods were never added to
      `SPECIFICITY_PATTERNS`. Rarer than the parens case; not yet fixed.
  - **Second round of real-data-driven fixes, on top of Phase 2/3** (all in
    `src/grounded_extraction.py` / `src/concept_clustering.py`):
    - **Morphological grounding** (`dictionary_snap`, `canonicalize_by_corpus_frequency`,
      `build_canonicalization_map`): typos/shorthand ("thrifte") get snapped to a real
      dictionary word within edit-distance 1 (`pyspellchecker`, symmetric-delete style), and
      separately, a rare single-word variant gets snapped to a much more frequent,
      string-similar dominant form elsewhere in the corpus (Jaro-Winkler >= 0.88, hand-
      implemented in pure Python rather than adding another dependency for one function --
      verified against reference values, e.g. MARTHA/MARHTA ~0.9611). Two-pass pipeline:
      `extract_segment_keywords()` now generates raw candidates for every segment once
      (accumulating global term frequency), builds the canonicalization map, *then* does the
      embed/rank/MMR/prune tail -- avoids running spaCy twice per segment despite
      canonicalization needing corpus-wide context unavailable until every segment's raw
      candidates have been seen.
      **Real bug found via testing, not assumed**: without a frequency guard, dictionary
      snapping "corrected" `malloc` (a real, frequently-used technical term not in a general-
      English dictionary, 20+ occurrences) to `mallow` (an unrelated real word one edit
      away). A typo is inherently rare; real technical jargon repeats. Fixed with
      `dictionary_snap_max_frequency: 2` -- only terms occurring at most twice in the whole
      corpus are eligible for dictionary snapping at all.
    - **Entity-weighted, specificity-based concept labeling**
      (`label_concepts_by_specificity`, `ENTITY_LABELS_BOOSTED`,
      `entity_label_multiplier`): concept labels are now chosen by
      `cosine_similarity(term, cluster_centroid) * inverse_segment_frequency *
      entity_multiplier` instead of raw chat/segment frequency, which tended to pick generic
      abstract terms ("approach", "discussion") over specific domain anchors purely because
      generic terms sit closer to a cluster's geometric center. Verified synthetically first
      (a fabricated "plato's symposium" vs. generic-terms cluster picks the specific entity,
      where the old frequency-based approach provably picks the most generic term) --
      confirms the mechanism is correctly implemented, independent of whether spaCy's NER
      tags real text well.
      **Deliberate reinterpretation, stated plainly**: the original proposal specified
      "inverse *cluster* frequency" (`log(N_clusters / N_clusters_containing_similar_terms)`)
      -- not implemented literally, since HDBSCAN produces a hard partition where a term
      belongs to exactly one cluster, so that denominator has no well-defined meaning
      without an expensive separate cross-cluster similarity pass. Implemented as inverse
      *segment* frequency instead (classic IDF, segments as "documents") -- same goal
      (penalize ubiquitous terms, reward rarities), well-defined signal already available.
      **Real bug found on real data, not assumed**: rewarding rarity rewards ALL rarity, not
      just genuine domain specificity -- this picked `"sam jose"` (a literal typo of "san
      jose", appearing in exactly 1 segment) as a concept's label over the correctly-spelled,
      far more common "san jose". Fixed with `min_label_segment_frequency: 2` -- a term must
      appear in at least 2 segments to be *eligible* as a label (still a valid cluster
      member either way, just not the representative). Confirmed fixed on the exact real
      cluster that surfaced the bug.
      **Known real limitation, not fully fixable within scope**: `en_core_web_sm` (this
      pipeline's spaCy model throughout) has real NER precision limits on informal/technical
      chat text -- real examples found: `"big plate chicken"[ORG]`, `"lamb soup dish"
      [PERSON]`, `"instagram story"[PERSON]` are all mistagged. The scoring mechanism itself
      is correct; a meaningful fraction of its entity-multiplier boosts fire on mistagged
      terms. A larger spaCy model (`en_core_web_md`/`lg`) would likely improve NER precision
      but is a separate, heavier change (bigger download, slower *whole-pipeline* processing,
      not just this narrow concern) -- not made without being asked.
    - **Dangling-boundary + technical-compound protection**
      (`TECHNICAL_HYPHEN_COMPOUNDS`, `_ensure_tokenizer_protections`,
      `_repair_dangling_boundary`): spaCy's default tokenizer split `"little-o"` into
      `["little", "-", "o"]`, and the lone `"o"` then failed length checks and got dropped,
      leaking the dangling fragment `"little-"` into candidates. Fixed at the root cause via
      `nlp.tokenizer.add_special_case()` (so known compounds like `little-o`/`big-O`/
      `n-gram`/`p-value` never get split in the first place), plus a second-line-of-defense
      regex repair (`^[-_~+=/]+|[-_~+=/]+$`) for anything not in the whitelist.

- **Phase 4 of the original 4-phase reconstruction (segment-level graph + tangent edges) is
  superseded, not built.** After seeing Phases 1-3's real output, decided on a different
  visualization philosophy instead: a **3-tier proximity hierarchy** (segment nodes ->
  Conversation Concepts -> Macro Domains) with **UMAP 2D projection replacing the PMI
  co-occurrence graph and edges entirely** (confirmed via AskUserQuestion). Tangents are
  conveyed by proximity + same-`chat_id` highlighting on hover/click, not edges. The
  original word-level `concept_clustering.py` (chat-keyed pathway) and
  `supertopic_clustering.py` are **kept in the repo, superseded** -- validated working code,
  just no longer what `build_output.py`/the frontend read once the new pipeline is wired in.
  Full plan: `~/.claude/plans/wobbly-cuddling-lynx.md`.
  - **Step 1, segment embeddings, done.** `extract_segment_keywords()`
    (`src/grounded_extraction.py`) now returns `(results, segment_embeddings)` -- captures
    the `segment_vector` it was already computing for MMR ranking instead of discarding it.
    `scripts/run_grounded_extraction.py` persists `data/processed/segment_embeddings.npz`
    (`segment_ids: str[]`, `embeddings: float32[N, 768]` via `np.savez`) alongside the
    existing JSON output. Real run: 1,287 embeddings persisted, no change to keyword output.
  - **Step 2, Conversation Concepts, done.** `src/conversation_concepts.py` +
    `config/conversation_concepts.yaml` + `scripts/run_conversation_concepts.py`: HDBSCAN
    directly on segment embeddings (not keyword embeddings) groups semantically similar
    segments into local neighborhoods. Own tuned config
    (`min_cluster_size: 3, min_samples: 4`) since 1,287 segments is a much smaller N than
    concept clustering's 8,928 keywords. Each cluster labeled via its **medoid segment**
    (closest to centroid) using that segment's already-computed top keyword -- simpler than
    a fresh specificity-weighted pass, and real output didn't demand one. Noise points
    become singleton concepts. Output: `data/processed/conversation_concepts.json`
    (`{label: {segment_ids: [...]}}`). **Real result, reviewed, no further tuning needed**:
    1,287 segments -> 908 concepts (31 real multi-segment clusters + 877 singletons), 32%
    merge rate. All sampled real clusters genuinely coherent on inspection (e.g. "frequency
    domain analysis", "funeral prayer", "c code", "tess light curves", "korean skincare
    rec", "am2320 sensor").
  - **Step 3, Macro Domains -- rebuilt from scratch, twice, both times because a fixed-count
    global partition has a real resolution limit.** First implementation clustered
    conversation-concept centroids via agglomerative/ward into a fixed `n_clusters: 12`
    (matching `supertopic_clustering.py`'s proven approach). Found and fixed one real bug
    that way (96.6% of concepts were HDBSCAN singletons, force-merging them alongside 31 real
    clusters produced a 27%-of-corpus catch-all; fixed with a two-stage
    cluster-then-nearest-centroid-assign approach) and it looked clean at 12 domains, 23%
    max share. **Still wrong, caught by closer inspection**: forcing 100% coverage into
    exactly 12 buckets was itself the problem, not just the earlier singleton-input bug --
    domains like "korean skincare rec" still silently contained unrelated one-offs
    ("webpack plugin", "malai boti chicken") that had simply been nearest-centroid-assigned
    to whichever domain was least-bad, distorting both the domain's content and its label
    (a single medoid segment's top keyword, picked from a cluster that wasn't actually pure).
    **Rebuilt entirely around a fundamentally different algorithm** (a full external redesign
    proposal, accepted and implemented as specified): mutual k-NN graph + Leiden community
    detection under the Constant Potts Model (CPM), with an explicit noise bucket instead of
    forced 100% coverage, and TF-ICF labeling instead of medoid-keyword labeling.
    `src/macro_domains.py` (full rewrite) + `config/macro_domains.yaml` (full rewrite) +
    `scripts/run_macro_domains.py` (full rewrite).
    - **Why CPM, not Ward, fixes the resolution limit -- not just patches around it.**
      Ward (and modularity-based methods generally) evaluate a candidate merge against the
      density of the WHOLE graph/dataset, so two small-but-genuinely-distinct communities
      can score better merged than separate once the graph is large enough -- that's the
      "resolution limit," and it's structural, not a tuning mistake (confirmed: this exact
      failure mode survived multiple linkage/tuning attempts at both the supertopic tier and
      this one). CPM instead evaluates each community against a FIXED resolution parameter
      gamma, independent of the rest of the graph, so a tight 5-segment community and a tight
      60-segment community can both be accepted on their own merits without one swallowing
      the other.
    - **`build_mutual_knn_graph()`**: edge between two segments only when each is in the
      OTHER's top-`knn_k` nearest neighbors (cosine similarity, `knn_k: 15`) -- mutual, not
      one-directional, so a single popular/central segment can't become a false hub bridging
      unrelated communities together (which would just recreate the resolution-limit problem
      inside the graph itself).
    - **Explicit noise bucket (`BACKGROUND_LABEL = "Background / Minor Orbit"`)**: a segment
      is routed there, not force-assigned to a real domain, if (a) its Leiden community is
      too small (`min_community_size: 5`) or (b) its own average cosine similarity to the
      OTHER members of its community is too low (`min_avg_similarity: 0.25`) even if the
      community itself is big enough. This is the direct fix for the "webpack plugin in
      korean skincare rec" bug -- a real domain's centroid/label is no longer computed over
      segments that don't actually belong.
      **Real bug found in this fix itself, on real data**: `min_community_size` was checked
      against Leiden's ORIGINAL community size, before the `min_avg_similarity` filter ran --
      a community of 6 could lose 5 members to that per-segment filter and still pass the
      size check (checked against the stale count of 6), leaving a "real" domain with 1
      surviving segment. Symptom: `sizes = sorted(len(members) for ...)` showed `min=1` in
      the checkpoint output despite `min_community_size: 5`. Fixed by computing
      `surviving_size` in a first pass (community size AFTER removing individually-dissimilar
      members), then checking min_community_size against THAT count, not Leiden's raw output
      -- verified fixed (rerun: `min=5`, matching the config exactly, no exceptions).
    - **`label_communities_by_tf_icf()`**: term frequency (count of distinct member segments
      containing the term) times inverse CLUSTER frequency (`log((n_communities+1) /
      (communities containing term + 1)) + 1` -- distinct from `conversation_concepts.py`'s
      inverse SEGMENT frequency), top 3 terms joined with `&`. Fixes the single-medoid
      fragility directly: a label is no longer one segment's top keyword, it's whatever terms
      are actually concentrated across the WHOLE community. Real labels read as genuine
      themes ("circuit analysis & ohm law & kirchhoff current law", "roc & rightmost pole &
      laplace transform", "brighten ingredient & hyaluronic acid & acrylate allergy") --
      not hand-composed category names like "Circuit Theory & Phasors", since nothing
      synthesizes new wording; stated plainly as a real, expected limitation of an automated
      label, not hidden.
    - **Real result, reviewed, tuned once**: `cpm_resolution: 0.05` (an initial guess, no
      literature default for this graph size/density) produced 67 domains -- correctly
      coherent, but too fragmented relative to the ~25-45 target. Lowered to `0.025`: 38 real
      domains (target hit), `Background / Minor Orbit` 481 of 1,287 segments (37%). Read
      every sampled domain's real text: all 38 are genuinely tight, single-territory clusters
      (permutation/probability, UCSD logistics, circuit analysis, Urdu/funeral prayer, C
      pointer bugs, thermodynamics, Laplace transforms, Nordic travel, Korean skincare, a
      single specific gift-shopping conversation) -- no incoherent catch-all survived. Sampled
      the Background bucket too: genuinely disparate one-offs (a robotic-mount question, a
      CPU/GPU/TPU comparison, a movie discussion, circuit breakers), not discarded real
      signal -- consistent with this corpus's already-established finding that most personal-
      assistant chat topics are genuinely one-off (see concept clustering's merge-rate
      history above). 37% background is a real, stated tradeoff of not forcing coverage, not
      hidden as if it were 0%.
    - **`assign_concepts_to_domains()`**: since domains are now clustered directly from
      segments (not from conversation-concept centroids), a conversation concept's
      `macro_domain_id` is a separate, derived step -- majority vote among its member
      segments' own domain assignments. A multi-segment concept's segments CAN legitimately
      split across domains (HDBSCAN's segment-embedding neighborhoods and Leiden's mutual-kNN
      communities are different partitions of the same 1,287 points) -- majority vote is a
      reasonable single answer, not a claim of 100% internal agreement.
    - **Downstream wiring fixed to prefer direct segment membership over concept-derived
      membership** wherever both were available (`scripts/run_proximity_layout.py`,
      `src/build_output.py`'s `build_final_json_v3()`): `macro_domains.json`'s `segment_ids`
      field (authoritative, from Leiden) is read directly for domain centroids and each
      segment's own `macro_domain_id`, instead of re-deriving domain membership by walking
      through concepts (which could disagree with a segment's actual Leiden assignment, per
      the point directly above).
    - **New dependencies**: `leidenalg`, `python-igraph` (the C-backed graph library
      leidenalg operates on).
    - **Follow-up fix: labels truncated to a single tight anchor term, chained `&` list
      dropped entirely.** The first version of `label_communities_by_tf_icf()` joined the
      top 3 TF-ICF-scored terms per domain with `" & "` (e.g. "fundamental period &
      rectangular pulse & convolution") -- functionally fine, but real output read as noisy
      and long rather than a clean map label. Changed to a single term: the highest-scoring
      term that's already `<= max_label_words` (3) words long, falling back to truncating
      the single highest-scoring term's first `max_label_words` words only when no real
      extracted term in that community is already short enough on its own.
      `top_terms_per_label` config key renamed to `max_label_words` (same rename in
      `config/macro_domains.yaml` and `get_macro_domains_config()`). Real output:
      "fundamental period", "circuit analysis", "ucsd", "essay", "block size" -- clean single
      terms, no chains. Reran Steps 4-5 after the change; domain count/coherence unaffected
      (labeling doesn't touch clustering).
  - **Step 4, UMAP projection, done.** `src/proximity_layout.py` + `config/
    proximity_layout.yaml` + `scripts/run_proximity_layout.py`: `project_to_2d()` runs
    `umap.UMAP(metric="cosine")` on the 1,287 segment embeddings directly (replaces the PMI
    co-occurrence graph + spring layout for this pathway entirely), rescaled to the existing
    `[-100, 100]` contract by reusing `normalize_positions()` from `src/layout.py` unchanged
    (already fully generic over any `{key: (x, y)}` dict). Conversation-concept and
    macro-domain positions are the **centroid** of their member segments' positions
    (`compute_group_centroids()`, shared code, same pattern as supertopic centroids in the
    word-level pipeline). Segment height (Z) is `log1p(word_count)` min-max scaled to
    `[0, z_range]` (reuses `z_range` from `config/height_weights.yaml`) -- a deliberate
    simplification vs. `height_score.py`'s 3-metric formula, since a segment belongs to
    exactly one chat and has no cross-chat frequency/sustained-focus signal to combine.
    Tangent-pull (`apply_tangent_pull()`) blends each `"tangent"` segment's position toward
    its own chat's `"core"` segment centroid (`P_final = (1-alpha)*P_tangent +
    alpha*P_core`, `tangent_pull_alpha: 0.15`, disabled at `0`). New dependency
    `umap-learn==0.5.12` (pulls in numba/llvmlite).
    - **Real finding (partially resolved by the later Step 3 rebuild, not by UMAP tuning
      itself): the corpus has two genuinely different embedding "registers."** With the
      original 12-domain Ward/two-stage Macro Domains, the scatter preview showed only 2 of
      12 domains (the hard-STEM/CS ones) as fully separate islands, with the other 9
      "everyday life" domains sharing one large, continuous region. Tried tightening UMAP
      params once (`umap_n_neighbors: 15 -> 10`, `umap_min_dist: 0.1 -> 0.05`, current
      values) to check whether this was a layout-parameter artifact: local separability
      improved, but the 2-islands-vs-1-shared-region topology didn't change -- read as a real
      property of `all-mpnet-base-v2`'s embedding space (technical/jargon-heavy language sits
      far from everyday conversational language, but different everyday-life *topics* don't
      have that same separation from each other), not something UMAP tuning alone would fix.
      **After Step 3's rebuild (mutual-kNN + Leiden/CPM, 38 domains instead of 12)**, this
      same preview shows real, visible spatial fanning: STEM domains cluster on one side,
      programming/CS on another, personal/travel/humanities on a third -- far more
      differentiated than the old 2-islands-vs-1-blob shape, though still not claimed as
      "fully separated for every one of the 38" (the preview PNG's 39 always-on labels
      visually overlap at this density -- a rendering-density limitation of the static
      matplotlib checkpoint, not the underlying data; the frontend's per-node hover/click
      doesn't have that problem). The underlying cause (STEM jargon vs. everyday-register
      embedding distance) is still real and unchanged -- more, tighter macro domains gave
      UMAP more genuinely distinct clusters to spread apart, which is what improved the
      picture, not a fix to that root cause itself.
  - **Real bug found while building Step 5, in Step 2's code
    (`cluster_segments_into_concepts()`, `src/conversation_concepts.py`): concepts were
    keyed by LABEL TEXT, not a stable id -- silently dropped 99 (later found to be 99 out
    of 1,287, 7.7%) segments.** `concept_to_segments[label] = member_ids` used the medoid's
    top keyword as the dict key. Two different clusters can legitimately land on the same
    top keyword for their medoid (e.g. two unrelated singleton segments both surfacing
    "ucsd") -- when that happened, the second cluster's write silently overwrote the
    first's entire member list in the dict, even though `segment_to_concept` (keyed by
    segment id, never collides) still pointed those orphaned segments at a label that no
    longer listed them as members. **Why Step 2's checkpoint didn't catch this**: the
    printed concept *count* (908) looked stable and plausible either way -- checking that
    count, and reading sampled real text (which this project's review discipline already
    does), doesn't surface a dict that's silently missing entries; only reconciling total
    membership against the segment count (1,287) would have. Only surfaced now because
    Step 5's `validate_schema_v3()` caught a segment with no `conversation_concept_id`.
    Fixed with numeric `" (2)"`, `" (3)"`, ... suffixing on label collision (same pattern as
    `_dedupe_slugs()` in `build_output.py`) so every cluster gets its own dict entry
    regardless of label collisions -- cluster assignments themselves are unchanged, only
    how they're written out. **Verified fixed**: re-running Steps 2-5 end to end now
    reconciles exactly (1,287 segments -> 949 concepts, up from 908 -- the 41 real clusters
    that had been silently merging into others now correctly stand alone -- -> 12 domains,
    still all thematically coherent on inspection). This is a good example of why the "test
    end-to-end with synthetic data" rule alone wasn't enough here -- a small synthetic
    fixture is unlikely to produce a real label collision; this needed the full real corpus
    to surface.
  - **Step 5, `build_output.py` schema v3, done.** `build_final_json_v3()` +
    `validate_schema_v3()` (`src/build_output.py`) + `scripts/run_build_output_v3.py`: pure
    merge of the five upstream checkpoint scripts' already-reviewed JSON artifacts (no
    re-ingestion/re-extraction/re-clustering, unlike `run_build_output.py`'s v2 pathway).
    Schema: `segments`/`conversation_concepts`/`macro_domains` top-level arrays, **no
    `edges`/`superedges` at all** -- proximity replaces them entirely, confirmed via
    AskUserQuestion. A segment's `id` is its own `segment_id` string (already unique and
    JS-safe, unlike concept/domain labels which need `_dedupe_slugs()`); its `label` is the
    highest-scoring grounded keyword from `segment_keywords.json` (explicitly picked by max
    score, not list position -- MMR diversifies, it doesn't guarantee the final list stays
    sorted). `build_final_json()` (v2) and `SCHEMA_VERSION` (2) are untouched -- v3 is
    additive, same "superseded, not deleted" treatment as the word-level pipeline itself.
    **Known edge case, not yet handled by the frontend (Step 6's job)**: 2 of 1,287
    segments have `label: ""` -- they're the two segments Phase 2 already documented as
    producing zero keyword candidates. Schema validation intentionally allows an empty
    string here (`isinstance(..., str)`, not `and segment["label"]`) rather than crashing
    the whole build over 2 segments; the frontend needs its own fallback (e.g. show the
    segment's chat context instead of a blank label) when rendering these.
  - **Step 6, frontend 3-tier drill-down, done.** `Scene.tsx` rewritten to consume schema v3
    directly (the v2 overview/detail pathway is gone from live code, not just superseded
    data -- see the "Architecture" section above). `viewMode: "macro" | "concept" |
    "segment"`, each level filtering the next tier down by its parent id
    (`macro_domain_id`/`conversation_concept_id`) and mapping into the shared
    `RenderableNode` shape; `handleBack` steps up exactly one tier. No `EdgeLines` --
    `Bounds fit clip observe` is the only thing reframing the camera between tiers now.
    `PointCloud.tsx`'s per-node scale switched from `chat_count` to a generic `size` field
    (word_count/segment_count/concept_count depending on tier, since each tier's "how big is
    this" metric differs) and gained a `highlightedIds` prop for the same-`chat_id` pulse.
    Per-instance color, not opacity, conveys both the hover-highlight and the tangent/
    same-chat states -- drei's instanced `Instance` doesn't reliably support per-instance
    opacity, only color, confirmed by checking before relying on it.
    - **Same-chat_id highlighting**: hovering/clicking a segment highlights every other
      *currently-rendered* segment sharing its `chat_id` (`Scene.tsx`'s `highlightedIds`
      useMemo). **Known, stated simplification**: this only reaches siblings within the
      active concept's rendered set, not ones that drifted into a different concept/macro
      domain (a tangent segment can easily belong to a different one than its chat's core
      segment) -- doing that would mean rendering across tiers simultaneously, out of scope
      for this pass.
    - **Real bug found via manual browser testing, not caught by any automated check**: one
      concept's legend title rendered as `43e85f9ad6284630::seg3` -- a raw internal
      segment id leaking into a user-facing label. Root cause: `find_medoid()`
      (`src/conversation_concepts.py`) picks a cluster's medoid by pure geometric similarity
      to centroid, with no awareness of which segments have a real keyword; when the medoid
      it picked was one of the 2 corpus-wide zero-candidate segments (already known from
      Phase 2), `cluster_segments_into_concepts()`'s old fallback
      (`top_keyword_by_segment.get(medoid_id, medoid_id)`) used the segment id itself as the
      label. Neither the Step 2 checkpoint (reads sampled real text, not every label
      string) nor `validate_schema_v3()` (checks label is a non-empty string, doesn't know
      what a "real" label looks like) could have caught this -- only surfaced by actually
      clicking through the rendered UI. Fixed by giving `find_medoid()` an optional
      `has_keyword` set: it now prefers the highest-centroid-similarity member that HAS a
      keyword, only falling back to the pure geometric medoid if truly no cluster member has
      one -- and even then, the label falls back to the string `"(untitled)"`, never a raw
      id. **Verified fixed**: re-ran Steps 2-5, confirmed zero concept labels contain
      `"::seg"` anymore; the affected concept now reads `"(untitled)"` cleanly. This is a
      concrete instance of a broader pattern worth remembering: text-sample checkpoints and
      schema validation both missed this because neither one asks "does this look like a
      real keyword" -- only clicking through the actual rendered artifact did. Manual UI
      testing found a real bug that automated checks structurally couldn't.
    - **Known cosmetic quirks, not fixed (low priority, visual only)**: drei's `Bounds`
      framing is inconsistent for single-node scenes (singleton concepts) -- sometimes zooms
      extremely close (camera appears inside the sphere), sometimes frames normally;
      Legend's node-count line doesn't pluralize ("1 segments"). Neither affects data
      correctness or the drill-down mechanics themselves.
  - **Step 7, full pipeline run + real-data review, done.** Reran Steps 2-5
    (`run_conversation_concepts.py` -> `run_macro_domains.py` -> `run_proximity_layout.py` ->
    `run_build_output_v3.py`) after the medoid-labeling fix, copied the result to
    `frontend/public/data.json`, and exercised the actual rendered app in a browser: macro
    domain overview loads (12 domains, correct counts) -> click drills into a domain's
    conversation concepts (verified: "family sharing", 39 concepts) -> click drills into a
    concept's segments (verified: reached segment tier, hover tooltip renders real segment
    text) -> back button steps up exactly one tier each time (verified both concept->macro
    and segment->concept). `npx tsc --noEmit` and `npm run build` both pass clean.
  - **Directory reorganization, done (after Step 7).** With the reconstruction fully working,
    the superseded word-level pipeline's files were sorted into `src/legacy/`,
    `scripts/legacy/`, `config/legacy/` (see "Architecture" above for exactly which files and
    why a few stay mixed in `src/` root instead). This was a real dependency-graph exercise,
    not a blind move -- `grep`ing every `from src.` import across `src/*.py` and
    `scripts/*.py` first surfaced a genuine entanglement: `src/segmentation.py` and
    `src/grounded_extraction.py` (both current pipeline) imported `get_embedding_model` from
    `src/concept_clustering.py` (otherwise entirely legacy). Fixed by moving just that one
    small, self-contained function to `src/text_embedding.py` (already the shared embedding
    utility module) before moving `concept_clustering.py` to `legacy/` wholesale -- confirmed
    this was the ONLY such entanglement by re-running the same import grep afterward and
    finding zero remaining `src.legacy.*` imports from outside `legacy/`.
    - Every moved script's `PROJECT_ROOT = Path(__file__).resolve().parent.parent` needed an
      extra `.parent` (files one directory deeper now) -- missing this would have silently
      pointed `PROJECT_ROOT` at `scripts/` instead of the repo root, breaking every
      `data/processed/...` path in that script. Verified by actually re-running
      `scripts/legacy/run_extraction.py` from its new location against the real corpus
      end-to-end, not just checking the diff.
    - Every moved config's path constant (e.g. `DEFAULT_CONCEPT_CONFIG_PATH`) was updated to
      `config/legacy/...yaml`; verified each one resolves to a real file
      (`path.is_file()`) after the move, not just that the string looks right.
    - All cross-references in docstrings/comments/printed messages (e.g. "Run
      scripts/run_concept_clustering.py first") were updated to their new
      `scripts/legacy/...` paths -- but only for scripts that actually moved; messages
      pointing at `run_ingest.py` and `run_grounded_extraction.py` (both current, unmoved)
      were deliberately left unprefixed. Got this wrong once in a first pass (a blanket
      find-replace would have mis-prefixed those two) -- did per-script targeted replacements
      instead.
    - `python -c "import src.X"` was run for every current AND every legacy module after the
      move to confirm the import graph resolves cleanly in both directions before considering
      this done.
    - **Extended to `data/processed/` in a follow-up pass**: legacy-only intermediate
      artifacts (`chat_keywords.json`, `phrases.json`, `concepts.json`, `chat_concepts.json`,
      `graph.gpickle`, `layout.json`, `height_scores.json`, `supertopics.json`, and Phase 3's
      now-also-superseded `segment_concepts.json`/`segment_concept_summary.json`) moved to
      `data/processed/legacy/`; `conversations.json` (shared input, from `run_ingest.py`) and
      `segment_keywords.json` (current pipeline's own output, which the now-legacy
      `run_segment_concept_clustering.py` also happens to read) stay in `data/processed/`
      root since both pipelines depend on them.
      **Real latent bug found and fixed along the way, not just a file move**:
      `src/build_output.py`'s `build_final_json()` (v2) and `build_final_json_v3()` (v3) had
      previously shared the exact same `OUTPUT_PATH` constant (`data/processed/data.json`) --
      running `scripts/legacy/run_build_output.py` after the current pipeline would have
      silently overwritten the canonical schema-v3 file the frontend actually reads, with no
      warning at all. Split into `OUTPUT_PATH` (current, stays at
      `data/processed/data.json`) and a new `LEGACY_OUTPUT_PATH`
      (`data/processed/legacy/data.json`) -- the two pipelines' outputs can no longer collide
      on the same filename. **Verified concretely, not just by inspection**: ran
      `scripts/legacy/run_build_output.py` against the real corpus end-to-end (full
      re-ingest/re-extraction, ~9 minutes), then confirmed `data/processed/data.json` still
      read `"version": 3` (schema v3, untouched by the legacy run) while the new
      `data/processed/legacy/data.json` correctly held `"version": 2` (schema v2) --
      the two files coexist independently instead of one silently clobbering the other.
      **One path reference was still missed by that verification run**: `src/
      extract_keywords.py`'s `DEFAULT_PHRASES_PATH` (used internally by `extract_keywords()`
      itself, not passed in by any caller) still pointed at the old root
      `data/processed/phrases.json` -- missed because the initial sweep grepped only
      `scripts/*.py` for stale legacy data paths, and this constant lives in `src/`. Its
      failure mode was silent, not a crash: `get_phrase_lookup()` already degrades
      gracefully to single-word keywords when the phrases file isn't found at the given
      path (by design, for the case where phrase detection genuinely hasn't been run yet)
      -- so the `scripts/legacy/run_build_output.py` verification run above completed
      successfully and proved the `data.json` split, but silently ran without real phrase
      merging. Fixed by re-pointing the constant at `data/processed/legacy/phrases.json`
      and confirming with `.is_file()` that it resolves; not re-run against the full corpus
      again since the fix is a one-line path correction to long-established, unchanged
      lookup logic, not new code needing a fresh end-to-end validation.

- **Continuous density field (DPGMM terrain), built.** A second, alternative reading of the
  same corpus alongside the discrete Macro Domains tier -- explicitly requested after Macro
  Domains landed, previously listed as a deferred/optional "terrain mesh interpolation"
  stretch goal. Fits a Dirichlet Process Gaussian Mixture Model over segment (x, y)
  positions and renders the result as a topographic terrain mesh (recurring themes ->
  "mountains," one-off tangents -> "lowland plains") layered underneath the existing point
  cloud, toggleable, not a replacement. `src/density_field.py` + `config/density_field.yaml`
  + `scripts/run_density_field.py` (backend) + `frontend/src/components/DensityTerrain.tsx`
  (frontend). New dependency: none (`sklearn.mixture.BayesianGaussianMixture` already
  available via the existing `scikit-learn` transitive dependency from `hdbscan`).
  - **Fit in 2D UMAP layout space, not the original 768-dim embedding space** -- a
    deliberate, stated scope decision, not the literal "infinite mixture over segment
    embeddings" framing originally proposed. Projecting a high-dimensional Gaussian's
    covariance structure into the already-lossy 2D UMAP view for contouring is a separate,
    much messier problem; fitting directly on the (x, y) positions already used for the
    point cloud keeps "what's rendered" and "what's modeled" the same space, at the cost of
    the mixture only seeing proximity, not the original semantic distances UMAP compressed
    away.
  - **Real, hard-won tuning finding: `weight_concentration_prior` (the DP concentration
    parameter, alpha) could NOT push effective component count above ~7**, across a 67x
    range (0.3 to 20) and three different initializations (k-means++, random,
    random_from_data) -- confirmed NOT a local-optimum artifact by testing all three
    independently and getting the same ~5-7 regardless. Root cause: with unconstrained
    (full) covariance, a handful of large elliptical components already cover the data's
    broad "continents" (the same macro-scale shape UMAP itself showed) at higher likelihood
    than many small ones; the prior only matters once it stops being the bottleneck, and it
    never became the bottleneck here. What actually controls effective count is constraining
    component SIZE: `covariance_prior_scale` (shrinks the prior covariance a component's
    estimate gets pulled toward) and `degrees_of_freedom_prior` (how strongly it's pulled).
    Confirmed by direct sweep on real data: `covariance_prior_scale` 1.0 -> 16 effective
    components, 0.1 -> 21, 0.02 -> 24; adding `degrees_of_freedom_prior: 10` at
    `covariance_prior_scale: 0.3` -> 32, landing close to Macro Domains' 38 for a comparable
    "how many real territories" reading between the discrete and continuous layers. Final
    config: `weight_concentration_prior: 5.0`, `covariance_prior_scale: 0.3`,
    `degrees_of_freedom_prior: 10.0`, `max_components: 60` (the upper truncation bound the
    stick-breaking prior prunes down from, not a claim of exactly-60 components).
  - **Rendering needed a SECOND real fix on top of the tuning above: raw log-density, even
    percentile-clipped, still wasn't legible.** The tight components the tuning above
    requires (to get enough effective count) produce a few genuinely narrow, high-precision
    Gaussians -- a tight component's peak height scales with `1/sqrt(det(Sigma))`, so those
    few spikes end up hundreds of nats taller than the ridges between components (confirmed:
    log-density range `[-306, -7]` on real data). A raw min-max OR percentile-clipped
    (1st-99.5th) color/height scale over that still saturates almost the entire populated
    area to "maximum" and hides all real texture -- tried percentile clipping first, visually
    confirmed it didn't fix it before reaching for a different transform. Fixed with
    **percentile-RANK normalization** (histogram equalization): `normalized_density` is each
    grid cell's rank in the empirical distribution of all grid cells' log-density, not the
    log-density itself -- guarantees a legible visual spread regardless of how skewed the
    raw values are. `evaluate_density_grid()` returns both: `log_density` (the literal,
    undistorted math) and `normalized_density` (rendering-only, what the frontend actually
    uses for terrain height + color). Verified visually: the real preview PNG now shows
    clear mountain/valley texture matching where the real segment clusters sit, not a flat
    saturated plain.
  - **Known, accepted visual artifact**: bright "starburst" rays radiate from each
    mountain's peak in the real render. This is a real, faithful property of summed
    elongated full-covariance Gaussian tails (components aren't forced to be circular), not
    a rendering bug -- confirmed by checking that the ray directions correspond to each
    component's actual covariance orientation. Left as-is; didn't chase further given the
    overall shape is otherwise legible and correct.
  - **Schema v3 gained an optional `density_field` field** (`{x, y, normalized_density,
    terrain_z_range} | null`) -- same graceful-degradation pattern as v2's `supertopics`:
    `run_density_field.py` is its own manual review gate, not required for the rest of the
    schema to work. `build_final_json_v3()` reads only the fields the frontend needs from
    `density_field.json` (drops the larger `log_density` array) rather than passing the
    whole file through.
  - **Frontend (`DensityTerrain.tsx`)**: a `THREE.PlaneGeometry` in the XY plane (matching
    the scene's existing x/y-horizontal, z-up convention, no rotation needed), vertex Z
    displaced by `normalized_density * terrain_z_range`, vertex-colored via a hand-rolled
    dark-to-gold ramp (no colormap dependency for one use site). **Real indexing bug caught
    before shipping, not after**: `PlaneGeometry`'s vertex buffer is row-major starting at
    the TOP (`y = +height/2`) going down, while the Python-side density grid's y-axis
    ascends (`np.linspace(-coordinate_range, +coordinate_range, ...)`) -- naively indexing
    `density[row][col]` by buffer row would have rendered the terrain upside-down on the y
    axis. Fixed by flipping the row index (`gridRow = resolution - 1 - row`); x/columns
    needed no flip (both ascend the same direction). Rendered OUTSIDE the `<Bounds>` wrapper
    in `Scene.tsx` deliberately -- `Bounds` auto-fits the camera to its children, and the
    terrain spans the full coordinate range regardless of which tier/domain is selected,
    which would wreck the existing per-tier auto-fit framing if it were inside. Off by
    default (`showTerrain` state), a Legend button toggles it -- an opt-in overlay under the
    existing point cloud, not a replacement, confirmed working in a live browser session
    (toggled on: real terrain renders matching the Python preview's shape exactly, drag-
    rotate confirms genuine 3D height displacement; toggled off: point cloud renders
    normally, unaffected).

## Milestone status (11-milestone build plan)

**Done:** graph construction (PMI-weighted co-occurrence, configurable pruning),
force-directed layout with static + interactive previews, corpus-level phrase detection
(NPMI-based, with redundancy pruning), keyword alias/canonicalization (two-tier), height
score computation (log1p + min-max normalized, YAML-configurable weights), semantic
concept clustering (embeddings + HDBSCAN, grouping keywords into graph nodes), a second
supertopic clustering layer (agglomerative, fixed count, <100 groups) for graph legibility,
unified `data.json` Python→frontend contract (schema v2, with supertopics/superedges) with
schema validation, React Three Fiber point cloud scaffold (Vite + React 19.2.8 + drei),
edge rendering + hover/click interaction + legend (schema v2, now superseded), topic
segmentation (Phase 1), grounded/MMR-diversified per-segment extraction (Phase 2),
context-aware concept induction (Phase 3), the full proximity-based 3-tier reconstruction
that followed (Conversation Concepts, Macro Domains, UMAP projection, schema v3, 3-tier
frontend drill-down with same-chat highlighting, no edges), Macro Domains' rebuild onto
mutual-kNN + Leiden/CPM community detection with TF-ICF single-term labeling, and the
continuous DPGMM density-field terrain (toggleable overlay under the point cloud) -- see
"Key decisions" above for the whole story.

**Next up:** the proximity-based 3-tier reconstruction and both of its later Macro Domains
revisions (Ward/two-stage -> mutual-kNN/Leiden/CPM, chained-term labels -> single tight
anchor terms) are done, plus the DPGMM continuous density field as an additional toggleable
visualization layer -- see "Key decisions" above for the full mechanism/tuning story on
each. One open thread from the reconstruction, not yet acted on: the same-`chat_id`
highlight's known simplification (only reaches siblings in the currently-rendered tier).
VADER + custom-phrase friction scoring and color mapping (the "conversational friction =
color" dimension from the project description) is also still unbuilt, independent of all of
the above.

**Deferred to v2** (don't build unless explicitly asked): Ollama-based LLM friction judging,
community-detection coloring, 2D D3 network view. (Terrain mesh interpolation, formerly
listed here, is done -- see "Continuous density field (DPGMM terrain)" above.)

## Stack

Python, NetworkX, spaCy, PyYAML, scipy, matplotlib, plotly, sentence-transformers (+ torch),
hdbscan (+ scikit-learn), umap-learn, leidenalg (+ python-igraph) (pipeline) → Vite,
React 19.2.8, TypeScript, React Three Fiber / Three.js, drei (frontend). Config-driven
parameterization throughout — see `config/` for the existing pattern before adding a new
tunable.
