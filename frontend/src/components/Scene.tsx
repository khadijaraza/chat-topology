import { Canvas } from "@react-three/fiber";
import { Bounds, Html, OrbitControls } from "@react-three/drei";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { GraphDataV3, RenderableNode } from "../types";
import PointCloud from "./PointCloud";
import Legend from "./Legend";
import DensityTerrain from "./DensityTerrain";

// data/processed/data.json copied to frontend/public/data.json by hand --
// see scripts/run_build_output_v3.py. Kept manual on purpose, so it's
// always clear whether the frontend is showing stale or fresh output.
const DATA_URL = "/data.json";

// macro (all domains) -> concept (one domain's conversation concepts) ->
// segment (one concept's individual segments). No graph edges at this
// schema version -- UMAP proximity (x, y) is the only relatedness signal,
// confirmed via AskUserQuestion (see CLAUDE.md's "Pipeline reconstruction").
type ViewMode = "macro" | "concept" | "segment";

export default function Scene() {
  const [data, setData] = useState<GraphDataV3 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("macro");
  const [selectedDomainId, setSelectedDomainId] = useState<string | null>(null);
  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null);
  // Off by default -- the discrete point cloud (drill-down) is the primary
  // view; the continuous DPGMM terrain (src/density_field.py) is an
  // alternative reading of the same corpus, layered underneath it as an
  // opt-in overlay, not a replacement.
  const [showTerrain, setShowTerrain] = useState(false);

  useEffect(() => {
    fetch(DATA_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}`);
        }
        return response.json() as Promise<GraphDataV3>;
      })
      .then(setData)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const selectedDomain = useMemo(
    () => data?.macro_domains.find((d) => d.id === selectedDomainId) ?? null,
    [data, selectedDomainId],
  );
  const selectedConcept = useMemo(
    () => data?.conversation_concepts.find((c) => c.id === selectedConceptId) ?? null,
    [data, selectedConceptId],
  );

  const conceptsInDomain = useMemo(() => {
    if (!data || !selectedDomainId) return [];
    return data.conversation_concepts.filter((c) => c.macro_domain_id === selectedDomainId);
  }, [data, selectedDomainId]);

  const segmentsInConcept = useMemo(() => {
    if (!data || !selectedConceptId) return [];
    return data.segments.filter((s) => s.conversation_concept_id === selectedConceptId);
  }, [data, selectedConceptId]);

  const renderedNodes: RenderableNode[] = useMemo(() => {
    if (!data) return [];
    if (viewMode === "macro") {
      return data.macro_domains.map((d) => ({
        id: d.id,
        label: d.label,
        x: d.x,
        y: d.y,
        z: d.z,
        size: d.segment_count,
      }));
    }
    if (viewMode === "concept") {
      return conceptsInDomain.map((c) => ({
        id: c.id,
        label: c.label,
        x: c.x,
        y: c.y,
        z: c.z,
        size: c.segment_count,
      }));
    }
    // segment: an empty label (2 of 1,287 segments -- the ones Phase 2
    // found zero keyword candidates for, see CLAUDE.md) falls back to the
    // chat id so the point is still identifiable rather than showing
    // nothing on hover.
    return segmentsInConcept.map((s) => ({
      id: s.id,
      label: s.label || `(untitled segment · ${s.chat_id})`,
      x: s.x,
      y: s.y,
      z: s.z,
      size: s.word_count,
      chat_id: s.chat_id,
      is_tangent: s.is_tangent,
    }));
  }, [data, viewMode, conceptsInDomain, segmentsInConcept]);

  const nodesById = useMemo(() => {
    const map = new Map<string, RenderableNode>();
    for (const node of renderedNodes) {
      map.set(node.id, node);
    }
    return map;
  }, [renderedNodes]);

  // Segment tier only: every other currently-rendered segment sharing the
  // hovered/clicked one's chat_id. This is the "tangents conveyed by
  // proximity + highlighting, not edges" mechanism from the plan -- a
  // known simplification is that it only highlights siblings within the
  // currently-rendered concept, not ones that drifted into a different
  // concept/domain (that would need rendering across tiers at once, out
  // of scope for this pass).
  const highlightedIds = useMemo(() => {
    if (viewMode !== "segment" || !hoveredId) return undefined;
    const hovered = nodesById.get(hoveredId);
    if (!hovered?.chat_id) return undefined;
    const siblings = new Set<string>();
    for (const node of renderedNodes) {
      if (node.chat_id === hovered.chat_id && node.id !== hoveredId) {
        siblings.add(node.id);
      }
    }
    return siblings;
  }, [viewMode, hoveredId, nodesById, renderedNodes]);

  const handleClickNode = useCallback(
    (node: RenderableNode) => {
      if (viewMode === "macro") {
        setSelectedDomainId(node.id);
        setSelectedConceptId(null);
        setViewMode("concept");
        setHoveredId(null);
        return;
      }
      if (viewMode === "concept") {
        setSelectedConceptId(node.id);
        setViewMode("segment");
        setHoveredId(null);
        return;
      }
      // eslint-disable-next-line no-console
      console.log("clicked segment:", node);
    },
    [viewMode],
  );

  const handleBack = useCallback(() => {
    if (viewMode === "segment") {
      setViewMode("concept");
      setSelectedConceptId(null);
      setHoveredId(null);
      return;
    }
    if (viewMode === "concept") {
      setViewMode("macro");
      setSelectedDomainId(null);
      setHoveredId(null);
    }
  }, [viewMode]);

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
    return <div className="scene-status">Loading topic map…</div>;
  }

  const hoveredNode = hoveredId ? nodesById.get(hoveredId) ?? null : null;
  const viewLabel =
    viewMode === "macro"
      ? "All macro domains"
      : viewMode === "concept"
        ? (selectedDomain?.label ?? "Domain")
        : (selectedConcept?.label ?? "Concept");
  const nodeUnit = viewMode === "macro" ? "macro domains" : viewMode === "concept" ? "conversation concepts" : "segments";

  return (
    <>
      <Canvas camera={{ position: [0, 0, 240], fov: 50, far: 2000 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[100, 150, 100]} intensity={0.8} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
        {/* Terrain deliberately sits OUTSIDE Bounds: Bounds auto-fits the
            camera to its children's bounding box, and the terrain spans the
            full coordinate range regardless of which tier/domain is
            selected -- inside Bounds it would wreck the per-tier auto-fit
            framing that's the whole point of that wrapper. */}
        {showTerrain && data.density_field && (
          <DensityTerrain densityField={data.density_field} coordinateRange={data.coordinate_range} />
        )}
        <Bounds fit clip observe margin={1.4} key={viewMode + (selectedDomainId ?? "") + (selectedConceptId ?? "")}>
          <PointCloud
            nodes={renderedNodes}
            hoveredId={hoveredId}
            onHover={setHoveredId}
            onClickNode={handleClickNode}
            highlightedIds={highlightedIds}
          />
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
        showBack={viewMode !== "macro"}
        onBack={handleBack}
        showTerrain={showTerrain}
        onToggleTerrain={() => setShowTerrain((current) => !current)}
        terrainAvailable={data.density_field !== null}
      />
    </>
  );
}
