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
const RADIUS_PER_SQRT_CHAT = 0.03; // sqrt so one mega-topic doesn't dwarf everything

const DEFAULT_COLOR = "#6d8fe0";
const HOVER_COLOR = "#ffb347";

interface PointCloudProps {
  nodes: RenderableNode[];
  hoveredId: string | null;
  onHover: Dispatch<SetStateAction<string | null>>;
  onClickNode: (node: RenderableNode) => void;
}

export default function PointCloud({ nodes, hoveredId, onHover, onClickNode }: PointCloudProps) {
  return (
    <Instances limit={nodes.length} range={nodes.length}>
      <sphereGeometry args={[1, SPHERE_SEGMENTS, SPHERE_SEGMENTS]} />
      <meshStandardMaterial />
      {nodes.map((node) => {
        const radius = BASE_RADIUS + RADIUS_PER_SQRT_CHAT * Math.sqrt(node.chat_count);
        const isHovered = node.id === hoveredId;

        return (
          <Instance
            key={node.id}
            position={[node.x, node.y, node.z]}
            scale={isHovered ? radius * 1.5 : radius}
            color={isHovered ? HOVER_COLOR : DEFAULT_COLOR}
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
