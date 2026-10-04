#!/usr/bin/env python3
"""Merge the proximity-based 3-tier pipeline's outputs (segments ->
conversation concepts -> macro domains) into data/processed/data.json,
schema v3.

Step 5 of the pipeline reconstruction (see CLAUDE.md). Run after
run_proximity_layout.py. Supersedes scripts/legacy/run_build_output.py's
schema-v2 output as what the frontend reads -- that script and the
word-level pipeline it depends on (src/legacy/) stay in the repo, just
unused going forward.

Unlike scripts/legacy/run_build_output.py, this does NOT re-run ingestion/
extraction/clustering itself -- it purely merges already-reviewed JSON artifacts from
the five upstream checkpoint scripts (run_ingest, run_segmentation,
run_grounded_extraction, run_conversation_concepts, run_macro_domains,
run_proximity_layout), each a manual review gate in its own right.
Also merges data/processed/density_field.json (run_density_field.py) if it
exists -- optional, same graceful-degradation pattern as v2's
supertopics.json: the continuous DPGMM terrain layer isn't required for the
discrete segments/concepts/domains tiers to work.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_output import (  # noqa: E402
    CONVERSATIONS_PATH,
    CONVERSATION_CONCEPTS_PATH,
    MACRO_DOMAINS_PATH,
    OUTPUT_PATH,
    PROXIMITY_LAYOUT_PATH,
    SEGMENT_KEYWORDS_PATH,
    SEGMENTS_PATH,
    build_final_json_v3,
    validate_schema_v3,
)

SEGMENT_SAMPLE_COUNT = 12


def main() -> int:
    missing = [
        (path, script)
        for path, script in [
            (CONVERSATIONS_PATH, "scripts/run_ingest.py"),
            (SEGMENTS_PATH, "scripts/run_segmentation.py"),
            (SEGMENT_KEYWORDS_PATH, "scripts/run_grounded_extraction.py"),
            (CONVERSATION_CONCEPTS_PATH, "scripts/run_conversation_concepts.py"),
            (MACRO_DOMAINS_PATH, "scripts/run_macro_domains.py"),
            (PROXIMITY_LAYOUT_PATH, "scripts/run_proximity_layout.py"),
        ]
        if not path.is_file()
    ]
    if missing:
        for path, script in missing:
            print(f"ERROR: Missing input file: {path}", file=sys.stderr)
            print(f"Run {script} first.", file=sys.stderr)
        return 1

    payload = build_final_json_v3()

    try:
        validate_schema_v3(payload)
    except AssertionError as exc:
        print(f"SCHEMA VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("Schema validation passed.")
    print()
    print("Build output (v3) summary")
    print("=" * 40)
    print(f"Segments: {len(payload['segments'])}")
    print(f"Conversation concepts: {len(payload['conversation_concepts'])}")
    print(f"Macro domains: {len(payload['macro_domains'])}")
    if payload["density_field"] is not None:
        grid_n = len(payload["density_field"]["x"])
        print(f"Density field: {grid_n}x{grid_n} grid (run_density_field.py)")
    else:
        print("Density field: not built -- run scripts/run_density_field.py for the terrain layer")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("Macro domains (segment_count, concept_count):")
    for domain in sorted(payload["macro_domains"], key=lambda d: -d["segment_count"]):
        print(
            f"  {domain['label']:<30s} segments={domain['segment_count']:4d}  "
            f"concepts={domain['concept_count']:3d}  pos=({domain['x']:7.2f}, {domain['y']:7.2f}, {domain['z']:5.2f})"
        )

    random.seed(23)
    sample = random.sample(payload["segments"], min(SEGMENT_SAMPLE_COUNT, len(payload["segments"])))
    print()
    print(f"Sample of {len(sample)} segments (review the label + tier assignment):")
    concept_label_by_id = {c["id"]: c["label"] for c in payload["conversation_concepts"]}
    domain_label_by_id = {d["id"]: d["label"] for d in payload["macro_domains"]}
    for seg in sample:
        concept_label = concept_label_by_id.get(seg["conversation_concept_id"], "(none)")
        domain_label = domain_label_by_id.get(seg["macro_domain_id"], "(none)")
        tangent = " [tangent]" if seg["is_tangent"] else ""
        print(f"  [{seg['id']}]{tangent} \"{seg['label']}\"")
        print(f"      concept={concept_label!r}  domain={domain_label!r}  pos=({seg['x']:.1f}, {seg['y']:.1f}, {seg['z']:.1f})")

    print()
    print("BEFORE YOU MOVE ON: does each sampled segment's label read as a fair")
    print("summary of that piece of conversation, and does its concept/domain")
    print("assignment look sensible? data/processed/data.json is the frontend")
    print("contract -- copy it to frontend/public/data.json yourself when ready")
    print("(that copy step stays manual on purpose).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
