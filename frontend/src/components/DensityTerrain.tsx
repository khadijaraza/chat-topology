import { useMemo } from "react";
import * as THREE from "three";
import type { DensityField } from "../types";

interface DensityTerrainProps {
  densityField: DensityField;
  coordinateRange: number;
}

// Cool/dark ("lowland plains") to warm/bright ("mountain peaks") ramp over
// normalized_density (already a [0, 1] percentile rank -- see DensityField
// in types.ts for why raw density alone isn't legible). Hand-rolled stops,
// not a colormap dependency, since this is the only place that needs one.
const COLOR_STOPS: [number, string][] = [
  [0.0, "#0b1020"],
  [0.35, "#1c3f5e"],
  [0.6, "#2f7a63"],
  [0.8, "#d9a441"],
  [1.0, "#f7eca0"],
];

function sampleColorRamp(t: number): THREE.Color {
  const clamped = Math.min(1, Math.max(0, t));
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const [t0, c0] = COLOR_STOPS[i];
    const [t1, c1] = COLOR_STOPS[i + 1];
    if (clamped <= t1) {
      const localT = t1 === t0 ? 0 : (clamped - t0) / (t1 - t0);
      return new THREE.Color(c0).lerp(new THREE.Color(c1), localT);
    }
  }
  return new THREE.Color(COLOR_STOPS[COLOR_STOPS.length - 1][1]);
}

export default function DensityTerrain({ densityField, coordinateRange }: DensityTerrainProps) {
  const geometry = useMemo(() => {
    const { x: xs, normalized_density: density, terrain_z_range: zRange } = densityField;
    const resolution = xs.length;

    const geom = new THREE.PlaneGeometry(
      2 * coordinateRange,
      2 * coordinateRange,
      resolution - 1,
      resolution - 1,
    );

    const positions = geom.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    // PlaneGeometry's vertex buffer is row-major starting at the TOP
    // (y = +height/2) going down to the bottom (y = -height/2), while our
    // density grid's y axis ascends (density[0] is y = -coordinateRange,
    // density[last] is y = +coordinateRange, via np.linspace on the Python
    // side) -- flip the row index to line the two up. Columns need no
    // flip: both the geometry's x and the grid's x ascend left to right.
    for (let row = 0; row < resolution; row++) {
      const gridRow = resolution - 1 - row;
      for (let col = 0; col < resolution; col++) {
        const vertexIndex = row * resolution + col;
        const value = density[gridRow][col];

        positions.setZ(vertexIndex, value * zRange);

        const color = sampleColorRamp(value);
        colors[vertexIndex * 3] = color.r;
        colors[vertexIndex * 3 + 1] = color.g;
        colors[vertexIndex * 3 + 2] = color.b;
      }
    }

    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.computeVertexNormals();
    return geom;
  }, [densityField, coordinateRange]);

  return (
    <mesh geometry={geometry} renderOrder={-1}>
      <meshStandardMaterial vertexColors roughness={0.95} metalness={0.02} side={THREE.DoubleSide} />
    </mesh>
  );
}
