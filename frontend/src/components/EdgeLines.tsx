import { Line } from "@react-three/drei";
import type { RenderableNode, TopicEdge } from "../types";

const DEFAULT_COLOR = "#6d8fe0";
const HIGHLIGHT_COLOR = "#ffb347";

interface EdgeLinesProps {
  // Pre-filtered by the caller (see selectTopEdges) -- this component just draws them.
  edges: TopicEdge[];
  nodesById: Map<string, RenderableNode>;
  hoveredId: string | null;
}

export default function EdgeLines({ edges, nodesById, hoveredId }: EdgeLinesProps) {
  if (edges.length === 0) return null;

  const weights = edges.map((edge) => edge.weight);
  const minWeight = Math.min(...weights);
  const maxWeight = Math.max(...weights);
  const spread = maxWeight - minWeight;

  return (
    <group>
      {edges.map((edge) => {
        const source = nodesById.get(edge.source);
        const target = nodesById.get(edge.target);
        if (!source || !target) return null;

        const norm = spread > 0 ? (edge.weight - minWeight) / spread : 0.5;
        const isHighlighted = hoveredId !== null && (edge.source === hoveredId || edge.target === hoveredId);

        return (
          <Line
            key={`${edge.source}->${edge.target}`}
            points={[
              [source.x, source.y, source.z],
              [target.x, target.y, target.z],
            ]}
            color={isHighlighted ? HIGHLIGHT_COLOR : DEFAULT_COLOR}
            lineWidth={isHighlighted ? 2.5 : 0.4 + norm * 1.4}
            transparent
            opacity={isHighlighted ? 0.9 : 0.08 + norm * 0.3}
          />
        );
      })}
    </group>
  );
}
