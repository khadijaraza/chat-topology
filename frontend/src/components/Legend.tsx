interface LegendProps {
  viewLabel: string;
  totalNodeCount: number;
  nodeUnit: string;
  showBack: boolean;
  onBack: () => void;
  showTerrain: boolean;
  onToggleTerrain: () => void;
  terrainAvailable: boolean;
}

export default function Legend({
  viewLabel,
  totalNodeCount,
  nodeUnit,
  showBack,
  onBack,
  showTerrain,
  onToggleTerrain,
  terrainAvailable,
}: LegendProps) {
  return (
    <div className="legend">
      {showBack && (
        <button type="button" className="back-button" onClick={onBack}>
          ← back
        </button>
      )}
      <div className="legend-title">{viewLabel}</div>
      <div>
        {totalNodeCount.toLocaleString()} {nodeUnit}
      </div>
      {terrainAvailable && (
        <button type="button" className="terrain-toggle" onClick={onToggleTerrain}>
          {showTerrain ? "hide terrain" : "show terrain"}
        </button>
      )}
    </div>
  );
}
