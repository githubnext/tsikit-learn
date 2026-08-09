/**
 * Time series datasets, financial datasets, and real-world dataset loaders.
 */

export interface TimeSeriesDataset {
  X: Float64Array[];
  y: Float64Array;
  timestamps: number[];
  featureNames: string[];
}

export function makeTimeSeries(nSamples = 200, nFeatures = 5, nLags = 3, noise = 0.1): TimeSeriesDataset {
  const featureNames = Array.from({ length: nFeatures }, (_, i) => `feature_${i}`);
  const raw: Float64Array[] = [];
  // Generate AR(nLags) process
  for (let f = 0; f < nFeatures; f++) {
    const series = new Float64Array(nSamples);
    const coefs = new Float64Array(nLags).map(() => (Math.random() - 0.5) * 0.8 / nLags);
    for (let t = 0; t < nSamples; t++) {
      let v = noise * (Math.random() - 0.5);
      for (let l = 0; l < nLags; l++) if (t - l - 1 >= 0) v += (coefs[l] ?? 0) * (series[t - l - 1] ?? 0);
      series[t] = v;
    }
    raw.push(series);
  }
  const X = Array.from({ length: nSamples }, (_, t) => new Float64Array(nFeatures).map((_, f) => raw[f]![t] ?? 0));
  const y = new Float64Array(nSamples - 1).map((_, t) => (X[t + 1]![0] ?? 0) + noise * (Math.random() - 0.5));
  const timestamps = Array.from({ length: nSamples }, (_, i) => i * 86400000); // Daily timestamps
  return { X: X.slice(0, -1), y, timestamps: timestamps.slice(0, -1), featureNames };
}

export function makeAnomalyDetectionDataset(nNormal = 200, nAnomalies = 20, nFeatures = 10): {
  X: Float64Array[];
  y: Int32Array;
} {
  const X: Float64Array[] = [];
  const labels: number[] = [];
  // Normal points: from Gaussian
  for (let i = 0; i < nNormal; i++) {
    X.push(new Float64Array(nFeatures).map(() => Math.random() * 2 - 1));
    labels.push(1);
  }
  // Anomalies: from wider distribution
  for (let i = 0; i < nAnomalies; i++) {
    X.push(new Float64Array(nFeatures).map(() => (Math.random() - 0.5) * 10));
    labels.push(-1);
  }
  return { X, y: new Int32Array(labels) };
}

export function makeMultilabelClassification(nSamples = 100, nFeatures = 20, nClasses = 5, nLabels = 2): {
  X: Float64Array[];
  Y: Int32Array[];
} {
  const X = Array.from({ length: nSamples }, () => new Float64Array(nFeatures).map(() => Math.random() - 0.5));
  const Y = Array.from({ length: nSamples }, () => {
    const labels = new Int32Array(nClasses);
    const selected = Array.from({ length: nClasses }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, nLabels);
    for (const l of selected) labels[l] = 1;
    return labels;
  });
  return { X, Y };
}

export function makeGraphDataset(nNodes = 50, edgeProbability = 0.1, nFeatures = 5): {
  adjacency: Float64Array[];
  nodeFeatures: Float64Array[];
  labels: Int32Array;
} {
  const adjacency = Array.from({ length: nNodes }, (_, i) =>
    new Float64Array(nNodes).map((_, j) => i !== j && Math.random() < edgeProbability ? 1 : 0)
  );
  const nodeFeatures = Array.from({ length: nNodes }, () => new Float64Array(nFeatures).map(() => Math.random()));
  const labels = new Int32Array(nNodes).map(() => Math.floor(Math.random() * 3));
  return { adjacency, nodeFeatures, labels };
}

export function loadSyntheticRegression(nSamples = 100, nFeatures = 10): {
  X: Float64Array[];
  y: Float64Array;
  coef: Float64Array;
} {
  const coef = new Float64Array(nFeatures).map(() => Math.random() * 2 - 1);
  const X = Array.from({ length: nSamples }, () => new Float64Array(nFeatures).map(() => Math.random()));
  const y = new Float64Array(nSamples).map((_, i) => X[i]!.reduce((s, v, j) => s + v * (coef[j] ?? 0), 0) + (Math.random() - 0.5) * 0.1);
  return { X, y, coef };
}
