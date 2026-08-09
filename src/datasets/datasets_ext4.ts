/**
 * Datasets extensions: makeTimeSeries, makeAnomalyDetection, makeGraphData, makeRankingData
 * Port of sklearn.datasets extensions
 */

function seededRng(seed: number): () => number {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
}

export function makeTimeSeries(opts: {
  nSamples?: number;
  nFeatures?: number;
  nTimesteps?: number;
  noise?: number;
  randomState?: number;
  trend?: boolean;
  seasonality?: boolean;
}): { X: Float64Array[][]; y: Float64Array } {
  const n = opts.nSamples ?? 100;
  const p = opts.nFeatures ?? 1;
  const T = opts.nTimesteps ?? 50;
  const noise = opts.noise ?? 0.1;
  const trend = opts.trend ?? true;
  const seasonality = opts.seasonality ?? true;
  const rng = seededRng(opts.randomState ?? 42);

  const X: Float64Array[][] = Array.from({ length: n }, () => {
    const series: Float64Array[] = Array.from({ length: T }, (_, t) => {
      const row = new Float64Array(p);
      for (let j = 0; j < p; j++) {
        let val = 0;
        if (trend) val += t / T * (rng() * 2 - 1);
        if (seasonality) val += Math.sin(2 * Math.PI * t / 12) * (rng() + 0.5);
        val += (rng() * 2 - 1) * noise;
        row[j] = val;
      }
      return row;
    });
    return series;
  });
  const y = Float64Array.from({ length: n }, (_, i) => X[i]!.reduce((s, ts) => s + (ts[0] ?? 0), 0) / T);
  return { X, y };
}

export function makeAnomalyDetection(opts: {
  nSamples?: number;
  nFeatures?: number;
  contamination?: number;
  randomState?: number;
}): { X: Float64Array[]; y: Int32Array; anomalyIndices: number[] } {
  const n = opts.nSamples ?? 200;
  const p = opts.nFeatures ?? 2;
  const contamination = opts.contamination ?? 0.1;
  const rng = seededRng(opts.randomState ?? 0);
  const nAnomalies = Math.floor(n * contamination);

  const X: Float64Array[] = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(p);
    const isAnomaly = i < nAnomalies;
    for (let j = 0; j < p; j++) {
      row[j] = isAnomaly ? (rng() * 10 - 5) + (rng() > 0.5 ? 5 : -5) : rng() * 4 - 2;
    }
    return row;
  });
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = X[i]!;
    X[i] = X[j]!;
    X[j] = tmp;
  }
  const anomalyIndices: number[] = [];
  const y = new Int32Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    const norm = X[i]!.reduce((s, v) => s + (v ?? 0) ** 2, 0);
    if (norm > p * 4) { y[i] = -1; anomalyIndices.push(i); }
  }
  return { X, y, anomalyIndices };
}

export function makeRankingData(opts: {
  nSamples?: number;
  nFeatures?: number;
  nGroups?: number;
  randomState?: number;
}): { X: Float64Array[]; y: Int32Array; groups: Int32Array; relevanceScores: Float64Array } {
  const n = opts.nSamples ?? 100;
  const p = opts.nFeatures ?? 10;
  const g = opts.nGroups ?? 10;
  const rng = seededRng(opts.randomState ?? 0);

  const X: Float64Array[] = Array.from({ length: n }, () => Float64Array.from({ length: p }, () => rng() * 2 - 1));
  const groups = Int32Array.from({ length: n }, (_, i) => Math.floor(i / Math.ceil(n / g)));
  const weights = Float64Array.from({ length: p }, () => rng() * 2 - 1);
  const relevanceScores = Float64Array.from(X.map(xi => {
    let s = 0;
    for (let j = 0; j < p; j++) s += (weights[j] ?? 0) * (xi[j] ?? 0);
    return s;
  }));
  const y = Int32Array.from(relevanceScores.map(s => Math.min(4, Math.max(0, Math.floor((s + 3) / 2)))));
  return { X, y, groups, relevanceScores };
}

export function makeMultiLabelData(opts: {
  nSamples?: number;
  nFeatures?: number;
  nClasses?: number;
  density?: number;
  randomState?: number;
}): { X: Float64Array[]; y: Int32Array[] } {
  const n = opts.nSamples ?? 100;
  const p = opts.nFeatures ?? 20;
  const c = opts.nClasses ?? 5;
  const density = opts.density ?? 0.2;
  const rng = seededRng(opts.randomState ?? 42);

  const X: Float64Array[] = Array.from({ length: n }, () => Float64Array.from({ length: p }, () => rng() * 2 - 1));
  const weights: Float64Array[] = Array.from({ length: c }, () => Float64Array.from({ length: p }, () => rng() * 2 - 1));
  const y: Int32Array[] = X.map(xi => {
    const labels = new Int32Array(c);
    for (let k = 0; k < c; k++) {
      const score = weights[k]!.reduce((s, w, j) => s + (w ?? 0) * (xi[j] ?? 0), 0);
      labels[k] = score > 0 && rng() < density + 0.5 ? 1 : 0;
    }
    return labels;
  });
  return { X, y };
}

export function makeGraphData(opts: {
  nNodes?: number;
  nFeatures?: number;
  edgeProbability?: number;
  randomState?: number;
}): { nodeFeatures: Float64Array[]; adjacency: Float64Array[]; labels: Int32Array } {
  const n = opts.nNodes ?? 50;
  const p = opts.nFeatures ?? 8;
  const edgeProb = opts.edgeProbability ?? 0.3;
  const rng = seededRng(opts.randomState ?? 0);

  const nodeFeatures: Float64Array[] = Array.from({ length: n }, () => Float64Array.from({ length: p }, () => rng() * 2 - 1));
  const adjacency: Float64Array[] = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n);
    for (let j = i + 1; j < n; j++) {
      if (rng() < edgeProb) { row[j] = 1; (adjacency[j] as Float64Array | undefined)?.set?.([1], i); }
    }
    return row;
  });
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) if ((adjacency[j]![i] ?? 0) > 0) adjacency[i]![j] = 1;
  const labels = Int32Array.from({ length: n }, (_, i) => {
    let degree = 0;
    for (let j = 0; j < n; j++) if ((adjacency[i]![j] ?? 0) > 0) degree++;
    return degree > n * edgeProb ? 1 : 0;
  });
  return { nodeFeatures, adjacency, labels };
}
