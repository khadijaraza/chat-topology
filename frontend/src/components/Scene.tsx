import { Canvas } from "@react-three/fiber";
import { Bounds, Html, OrbitControls } from "@react-three/drei";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { GraphData, RenderableNode, TopicEdge } from "../types";
import { selectTopEdges } from "../lib/selectTopEdges";
import PointCloud from "./PointCloud";
import EdgeLines from "./EdgeLines";
import Legend from "./Legend";

// data/processed/data.json copied to frontend/public/data.json by hand --
// see scripts/run_build_output.py. Kept manual on purpose, so it's always
// clear whether the frontend is showing stale or fresh output.
const DATA_URL = "/data.json";

type ViewMode = "overview" | "detail";

export default function Scene() {
  const [data, setData] = useState<GraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("overview");
  const [selectedSupertopicId, setSelectedSupertopicId] = useState<string | null>(null);

  useEffect(() => {
    fetch(DATA_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json() as Promise<GraphData>;
      })
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  // Older data.json files (built before the supertopic layer existed) have
  // an empty supertopics array -- fall straight back to the flat concept
  // view rather than showing an empty "overview".
  const hasSupertopics = (data?.supertopics.length ?? 0) > 0;

  const selectedSupertopic = useMemo(
    () => data?.supertopics.find((s) => s.id === selectedSupertopicId) ?? null,
    [data, selectedSupertopicId],
  );

  const detailNodes = useMemo(() => {
    if (!data || !selectedSupertopicId) return [];
    return data.nodes.filter((node) => node.supertopic === selectedSupertopicId);
  }, [data, selectedSupertopicId]);

  const detailNodeIds = useMemo(() => new Set(detailNodes.map((node) => node.id)), [detailNodes]);

  const detailEdges = useMemo(() => {
    if (detailNodeIds.size === 0) return [];
    return (data?.edges ?? []).filter(
      (edge) => detailNodeIds.has(edge.source) && detailNodeIds.has(edge.target),
    );
  }, [data, detailNodeIds]);

  const renderedNodes: RenderableNode[] = useMemo(() => {
    if (!data) return [];
    if (!hasSupertopics) return data.nodes;
    return viewMode === "overview" ? data.supertopics : detailNodes;
  }, [data, hasSupertopics, viewMode, detailNodes]);

  const renderedEdgesRaw: TopicEdge[] = useMemo(() => {
    if (!data) return [];
    if (!hasSupertopics) return data.edges;
    return viewMode === "overview" ? data.superedges : detailEdges;
  }, [data, hasSupertopics, viewMode, detailEdges]);

  const renderedEdges = useMemo(() => selectTopEdges(renderedEdgesRaw), [renderedEdgesRaw]);

  const nodesById = useMemo(() => {
    const map = new Map<string, RenderableNode>();
    for (const node of renderedNodes) {
      map.set(node.id, node);
    }
    return map;
  }, [renderedNodes]);

  const handleClickNode = useCallback(
    (node: RenderableNode) => {
      if (hasSupertopics && viewMode === "overview") {
        setSelectedSupertopicId(node.id);
        setViewMode("detail");
        setHoveredId(null);
        return;
      }
      // eslint-disable-next-line no-console
      console.log("clicked node:", node);
    },
    [hasSupertopics, viewMode],
  );

  const handleBack = useCallback(() => {
    setViewMode("overview");
    setSelectedSupertopicId(null);
    setHoveredId(null);
  }, []);

  if (error) {
    return (
      <div className="scene-status">
        Failed to load {DATA_URL}: {error}
        <br />
        Copy data/processed/data.json to frontend/public/data.json and reload.
      </div>
    );
  }

  if (!data) {
    return <div className="scene-status">Loading topic graph…</div>;
  }

  const hoveredNode = hoveredId ? nodesById.get(hoveredId) ?? null : null;
  const showBack = hasSupertopics && viewMode === "detail";
  const viewLabel = !hasSupertopics
    ? "All topics"
    : viewMode === "overview"
      ? "Overview"
      : (selectedSupertopic?.label ?? "Topic group");
  const nodeUnit = !hasSupertopics ? "topics" : viewMode === "overview" ? "super-topics" : "topics";

  return (
    <>
      <Canvas camera={{ position: [0, 0, 240], fov: 50, far: 2000 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={0.8} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        <Bounds fit clip observe margin={1.4} key={viewMode + (selectedSupertopicId ?? "")}>
          <PointCloud
            nodes={renderedNodes}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onClickNode={handleClickNode}
          />
          <EdgeLines edges={renderedEdges} nodesById={nodesById} hoveredId={hoveredId} />
        </Bounds>
        {hoveredNode && (
          <Html position={[hoveredNode.x, hoveredNode.y, hoveredNode.z]} style={{ pointerEvents: "none" }}>
            <div className="node-label">{hoveredNode.label}</div>
          </Html>
        )}
      </Canvas>
      <Legend
        viewLabel={viewLabel}
        totalNodeCount={renderedNodes.length}
        nodeUnit={nodeUnit}
        renderedEdgeCount={renderedEdges.length}
        totalEdgeCount={renderedEdgesRaw.length}
        showBack={showBack}
        onBack={handleBack}
      />
    </>
  );
}
