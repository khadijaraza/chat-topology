import type { TopicEdge } from "../types";

// Rendering every edge in a dense keyword co-occurrence graph produces an
// unreadable haze and tanks frame rate well before node count becomes the
// bottleneck -- cap to the strongest connections by weight. Bump this if a
// dataset needs more edges visible and the frame rate can take it.
export const EDGE_RENDER_LIMIT = 500;

export function selectTopEdges(
  edges: TopicEdge[],
  limit: number = EDGE_RENDER_LIMIT,
): TopicEdge[] {
  return [...edges].sort((a, b) => b.weight - a.weight).slice(0, limit);
}
