// Mirrors the schema written by src/build_output.py's build_final_json() ->
// data/processed/data.json. Keep in sync with that function's docstring.

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

// The shape PointCloud/EdgeLines actually need to render a point + its
// label -- both TopicNode and Supertopic satisfy this structurally, so the
// same components render either the concept-level or supertopic-level view.
export interface RenderableNode {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  chat_count: number;
}
