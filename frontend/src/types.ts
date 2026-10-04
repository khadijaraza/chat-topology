// Mirrors the schema written by src/build_output.py's build_final_json() ->
// data/processed/data.json. Keep in sync with that function's docstring.
//
// Schema v2 (TopicNode/Supertopic/TopicEdge/GraphData below) is superseded
// by schema v3 (SegmentNode/ConversationConcept/MacroDomain/GraphDataV3) --
// same "kept, not deleted" treatment as the Python word-level pipeline it
// mirrors (see CLAUDE.md). Scene.tsx now only reads v3; these v2 interfaces
// stay for reference/history, not live code.

export interface TopicNode {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  chat_count: number;
  platforms: string[];
  supertopic: string | null;
}

export interface Supertopic {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  chat_count: number;
  concept_count: number;
  platforms: string[];
}

export interface TopicEdge {
  source: string;
  target: string;
  weight: number;
}

export interface GraphData {
  version: number;
  generated_at: string;
  nodes: TopicNode[];
  edges: TopicEdge[];
  supertopics: Supertopic[];
  superedges: TopicEdge[];
}

// Schema v3: the proximity-based 3-tier hierarchy (segments -> conversation
// concepts -> macro domains). No edges/superedges -- UMAP proximity (x, y)
// replaces the PMI co-occurrence graph entirely for this pathway, confirmed
// via AskUserQuestion (see CLAUDE.md's "Pipeline reconstruction").
export interface SegmentNode {
  id: string;
  label: string;
  chat_id: string;
  platform: string;
  x: number;
  y: number;
  z: number;
  word_count: number;
  is_tangent: boolean;
  conversation_concept_id: string | null;
  macro_domain_id: string | null;
}

export interface ConversationConcept {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  segment_count: number;
  macro_domain_id: string | null;
}

export interface MacroDomain {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  concept_count: number;
  segment_count: number;
}

// Continuous alternative reading of the same corpus (src/density_field.py):
// a Dirichlet Process Gaussian Mixture Model fit over segment (x, y)
// positions, evaluated on a regular grid. `normalized_density` is a
// percentile-rank transform of the mixture's log-density (NOT raw
// probability) -- see evaluate_density_grid()'s docstring for why the raw
// math alone isn't legible for rendering. `x`/`y` are the grid's own axis
// coordinates (length = grid resolution each), `normalized_density` is
// grid_resolution x grid_resolution, row-major in y. Optional/nullable:
// run_density_field.py is its own manual review gate, same
// graceful-degradation pattern as v2's supertopics.
export interface DensityField {
  x: number[];
  y: number[];
  normalized_density: number[][];
  terrain_z_range: number;
}

export interface GraphDataV3 {
  version: number;
  generated_at: string;
  coordinate_range: number;
  density_field: DensityField | null;
  segments: SegmentNode[];
  conversation_concepts: ConversationConcept[];
  macro_domains: MacroDomain[];
}

// The shape PointCloud actually needs to render a point + its label -- every
// tier (segment/concept/domain) gets mapped into this by Scene.tsx. "size"
// is a generic scale driver since each tier's own "how big is this" field
// differs (word_count / segment_count / concept_count); chat_id/is_tangent
// are only meaningful at the segment tier (drive same-chat highlighting),
// so they're optional here rather than forcing every tier to fake them.
export interface RenderableNode {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  size: number;
  chat_id?: string;
  is_tangent?: boolean;
}
