interface LegendProps {
  viewLabel: string;
  totalNodeCount: number;
  nodeUnit: string;
  renderedEdgeCount: number;
  totalEdgeCount: number;
  showBack: boolean;
  onBack: () => void;
}

export default function Legend({
  viewLabel,
  totalNodeCount,
  nodeUnit,
  renderedEdgeCount,
  totalEdgeCount,
  showBack,
  onBack,
}: LegendProps) {
  return (
    <div className="legend">
      {showBack && (
        <button type="button" className="back-button" onClick={onBack}>
          ← back to overview
        </button>
      )}
      <div className="legend-title">{viewLabel}</div>
      <div>
        {totalNodeCount.toLocaleString()} {nodeUnit}
      </div>
      <div>
        {renderedEdgeCount.toLocaleString()} / {totalEdgeCount.toLocaleString()} edges shown
      </div>
    </div>
  );
}
