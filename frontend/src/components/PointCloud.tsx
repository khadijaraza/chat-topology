import { Instance, Instances } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import type { Dispatch, SetStateAction } from "react";
import type { RenderableNode } from "../types";

// Sphere over box: a low-poly icosphere (or here, a low-segment
// UV sphere) reads as an abstract "point" rather than a grid-aligned
// block, which fits a point-cloud aesthetic better. Cost-wise it doesn't
// matter -- Instances uploads one shared geometry/material and draws every
// node in a single GPU instanced call, so box vs. low-poly sphere triangle
// counts are both negligible until node counts reach the tens of thousands.
const SPHERE_SEGMENTS = 12;

const BASE_RADIUS = 0.7;
const RADIUS_PER_SQRT_SIZE = 0.03; // sqrt so one mega-topic doesn't dwarf everything

const DEFAULT_COLOR = "#6d8fe0";
const TANGENT_COLOR = "#8fa0bd"; // muted version of DEFAULT_COLOR -- per-instance
// opacity isn't reliably supported by drei's instanced Instance, so "this
// drifted from its chat's main thread" is conveyed via desaturated color
// instead, not transparency.
const HOVER_COLOR = "#ffb347";
// Same-chat_id siblings of the hovered/clicked segment -- a step between
// DEFAULT and HOVER so the pulse reads as "related to what you're pointing
// at" without being confused for the hovered node itself.
const SAME_CHAT_COLOR = "#ff8fa3";

interface PointCloudProps {
  nodes: RenderableNode[];
  hoveredId: string | null;
  onHover: Dispatch<SetStateAction<string | null>>;
  onClickNode: (node: RenderableNode) => void;
  // ids sharing the hovered/clicked segment's chat_id (segment tier only) --
  // null/empty everywhere else.
  highlightedIds?: Set<string>;
}

export default function PointCloud({ nodes, hoveredId, onHover, onClickNode, highlightedIds }: PointCloudProps) {
  return (
    <Instances limit={nodes.length} range={nodes.length}>
      <sphereGeometry args={[1, SPHERE_SEGMENTS, SPHERE_SEGMENTS]} />
      <meshStandardMaterial />
      {nodes.map((node) => {
        const radius = BASE_RADIUS + RADIUS_PER_SQRT_SIZE * Math.sqrt(node.size);
        const isHovered = node.id === hoveredId;
        const isSameChat = !isHovered && (highlightedIds?.has(node.id) ?? false);
        const color = isHovered
          ? HOVER_COLOR
          : isSameChat
            ? SAME_CHAT_COLOR
            : node.is_tangent
              ? TANGENT_COLOR
              : DEFAULT_COLOR;

        return (
          <Instance
            key={node.id}
            position={[node.x, node.y, node.z]}
            scale={isHovered || isSameChat ? radius * 1.5 : radius}
            color={color}
            onPointerOver={(event: ThreeEvent<PointerEvent>) => {
              event.stopPropagation();
              onHover(node.id);
            }}
            onPointerOut={(event: ThreeEvent<PointerEvent>) => {
              event.stopPropagation();
              onHover((current) => (current === node.id ? null : current));
            }}
            onClick={(event: ThreeEvent<MouseEvent>) => {
              event.stopPropagation();
              onClickNode(node);
            }}
          />
        );
      })}
    </Instances>
  );
}
